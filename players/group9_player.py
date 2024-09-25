import heapq
import math
import os
import pickle
import random
import time
import numpy as np
import logging
import sys
from collections import deque


import constants
from timing_maze_state import TimingMazeState

def get_neighbor(coordinates, direction):
    x,y = coordinates
    if direction % 2 == 0: # Left or Right
        return (x+direction-1, y)
    return (x, y+direction-2)

def get_neighbors(coordinates): 
    # left up right down
    x,y = coordinates
    return [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]

def manhattan_dist(coord1, coord2):
    x1,y1 = coord1
    x2,y2 = coord2
    return abs(x2-x1) + abs(y2-y1)

# Helper that maps directions to their opposites 
def opposite(direction) -> int:
    if direction == constants.UP:
        return constants.DOWN
    elif direction == constants.DOWN:
        return constants.UP
    elif direction == constants.LEFT:
        return constants.RIGHT
    return constants.LEFT

def GCD(a, b):
        if b == 0:
            return a
        return GCD(b, a % b)

def LCM(a, b):
        return a * b // GCD(a, b)

class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger,
                 precomp_dir: str, maximum_door_frequency: int, radius: int) -> None:
        """Initialise the player with the basic amoeba information

            Args:
                rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
                logger (logging.Logger): logger use this like logger.info("message")
                maximum_door_frequency (int): the maximum frequency of doors
                radius (int): the radius of the drone
                precomp_dir (str): Directory path to store/load pre-computation
        """

        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))

        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)

        self.rng = rng
        self.logger = logger
        self.maximum_door_frequency = maximum_door_frequency
        self.radius = radius

        self.step = 0 # Turn number
        self.cur_pos = [0, 0] # Current position of player
        self.epsilon = 0.0 # Epsilon value for randomization, 0 means no randomness
        self.door_states = {} # Door frequencies of each cell
        self.values = {} # Heuristics for directions
        self.best_path_found = {}
        self.boundary = [100, 100, 100, 100]
        self.corners = [[100, 100], [100, 100], [100, 100], [100, 100]] # (Left, Bottom), (Left, Top), (Right, Top), (Right, Bottom), 
        self.past_moves = []
        self.past_coords = [] 
        self.move_regrets = {}
        self.waited = 0
        self.escaping = False
        self.escape_route = deque([])

        self.reset_counter = 0

    class Corner:
        def __init__(self, end_x, end_y) -> None:
            self.end_x = end_x
            self.end_y = end_y

    def update_graph_information(self, current_percept):
        """
            Now that the percept has updated & another step has been taken, update our 
            knowledge of door frequencies and cells
        """
        for cell in current_percept.maze_state:
            relative_x = int(cell[0] + self.cur_pos[0])
            relative_y = int(cell[1] + self.cur_pos[1])
            cell_coordinates = (relative_x, relative_y)
            self.update_cell_state(cell_coordinates, cell[2], cell[3])
            self.update_cell_value(cell_coordinates, cell[3])

    def update_cell_state(self, coordinates, direction, state):
        """
            Update the known frequencies of the doors in the given cell
        """
        if coordinates not in self.door_states:
            self.door_states[coordinates] = [0, 0, 0, 0] # Left Top Right Bottom

        if state == constants.OPEN:
            if self.door_states[coordinates][direction] == 0:
                self.door_states[coordinates][direction] = self.step
            else:
                self.door_states[coordinates][direction] = GCD(self.door_states[coordinates][direction], self.step)

        elif state == constants.BOUNDARY and self.boundary[direction] == 100:
            # print("---Updating Boundary----")
            # print("Boundary Found at: ", coordinates)
            # print("Which Door is it? ", direction)
            self.door_states[coordinates][direction] = -1   
            x_or_y = 0 if direction % 2 == 0 else 1 # Which coordinate to update
            # print("X OR y (0 or 1)", x_or_y)
            self.boundary[direction] = coordinates[x_or_y]

            # Update self.corners
            self.corners[direction][x_or_y] = coordinates[x_or_y]
            self.corners[(direction + 1) % 4][x_or_y] = coordinates[x_or_y]
            # print(self.boundary)
            # print(self.corners)

    def update_cell_value(self, coordinates, door_type):
        """
            Update greedy-epsilon values
        """
        if coordinates not in self.values:
            if door_type == constants.BOUNDARY:
                self.values[coordinates] = -1
            else: 
                self.values[coordinates] = 1
        else:
            if door_type == constants.BOUNDARY:
                self.values[coordinates] = -1

            if self.values[coordinates] != -1:
                self.values[coordinates] += 1

    def close_to_corner(self) -> tuple:
        for i in range(len(self.corners)):
            coords_valid = True
            for coord in self.corners[i]:
                if coord == 100:
                    coords_valid = False
                    break
            if not coords_valid or tuple(self.corners[i]) in self.values:
                continue
            dist_from_corner = manhattan_dist(tuple(self.corners[i]), tuple(self.cur_pos))
            if dist_from_corner <= math.sqrt(2 * (self.radius ** 2)) + self.radius:
                return tuple(self.corners[i])

        return (100, 100)
    
    def cost_of_directions(self, coord, added_steps) -> list:
        current_turn = self.step + added_steps
        costs = []
        neighbors = get_neighbors(coord)

        # For each move, check when player can move
        for i in range(4):
            door_freq = self.door_states[coord][i]
            neighbor_door_freq = self.door_states[neighbors[i]][opposite(i)]
            # print("Direction: ", i, "Door Freq: ", door_freq)
            # print("Neighbor Direction: ", opposite(i), "Neighbor Door Freq: ", neighbor_door_freq)

            if door_freq <= 0 or neighbor_door_freq <= 0:
                costs.append(-1)
            else:
                common_freq = LCM(door_freq, neighbor_door_freq)
                can_move = common_freq
                # print("Common Freq: ", common_freq)
                if can_move >= common_freq:
                    # print("Current Turn is larger than common freq")
                    while (can_move < current_turn):
                        can_move += common_freq
                # print("Resulting common freq and current turn", can_move, current_turn)
                costs.append(can_move - current_turn)
                    
        return costs

    # Player is stuck. Formulate a plot or end up in jail or shot. Updates self.escape_route and self.escaping to True.
    def find_best_out(self):
        hard_limit = -1
        added_steps = 0
        move_taken = opposite(self.past_moves[-1])

        memo = {}
        memo_move = {}
        checked_coords = set()
        # print("Before evaluating Current Position")
        # Check current coord
        checked_coords.add(tuple(self.cur_pos)) 
        costs = self.cost_of_directions(tuple(self.cur_pos), added_steps)
        # print("Cost of moving in current position: ", costs)

        fastest_moves = sorted(range(len(costs)), key=lambda k : (costs[k], self.move_regrets[tuple(self.cur_pos)][k]))

        # print("After finding fastest moves")

        for move in fastest_moves:
            if move != move_taken and costs[move] >= 0:
                # print("---Setting Memo Stuff---")
                cost = costs[move]
                memo[tuple(self.cur_pos)] = cost
                # print("Cost of fastest move:", move, cost)
                
                actions = [constants.WAIT for i in range(cost)]
                actions.append(move)
                # print("What I need to do to get out: ", actions)

                memo_move[tuple(self.cur_pos)] = actions
                break

        # print("After updating fastest moves in memoization")

        added_steps += (1 + costs[move_taken])

        # There are no moves found
        if not memo or not memo_move:
            # print("No valid moves found, adding backtrack to memo")
            cost = costs[move_taken]
            memo[tuple(self.cur_pos)] = cost
            
            actions = [constants.WAIT for i in range(cost)]
            actions.append(move_taken)

            memo_move[tuple(self.cur_pos)] = actions
        else:
            # Set a hard limit
            for i in range(3, -1, -1):
                cost = costs[fastest_moves[i]]
                if cost != -1 and fastest_moves[i] != move_taken:
                    hard_limit = cost + 1
                    break
                
            # print("Hard Limit: ", hard_limit)

            # If backtracking takes more time then just waiting for a door to open
            if hard_limit != -1 and added_steps >= hard_limit:
                print("It is better to wait for another door to open")
                print("Highest Cost")
                # print(memo_move[tuple(self.cur_pos)])
                # print(self.escape_route)

                self.escaping = True
                self.escape_route = deque(memo_move[tuple(self.cur_pos)])
                print("Player's escape Route", self.escape_route)

                return self.escape_route.popleft()

            # print("After added_steps condition")

        # print("Before checking Past Coords")
        # Check all past coords to find if there is some out.
        for i in range(len(self.past_coords) - 1, -1, -1):
            coord = self.past_coords[i]
            prev_move = self.past_moves[i - 1] if i > 0 else -1

            if coord in checked_coords:
                continue

            checked_coords.add(coord)
            # print("Before finding costs of directions")
            costs = self.cost_of_directions(coord, added_steps)
            fastest_moves = sorted(range(len(costs)), key=lambda k : (costs[k], self.move_regrets[coord][k]))
            # print("Before looping through fastest moves")

            for move in fastest_moves:
                if move != opposite(prev_move) and costs[move] >= 0:
                    # print("Found something")
                    cost = costs[move]
                    prev_coord = None
                    # print("Before prev coord")
                    if i == (len(self.past_coords) - 1):
                        prev_coord = tuple(self.cur_pos)
                    else:
                        prev_coord = self.past_coords[i + 1]
                    # print("After prev coord")
                    # print("Current coord: ", coord)
                    # print("Prev coord: ", prev_coord)

                    memo[coord] = cost
                    cur_actions = []
                    if cost != 0:
                        cur_actions = [constants.WAIT for i in range(cost)]

                    cur_actions.append(move)
                    print("Before prev actions")
                    if prev_coord in memo_move:
                        prev_actions = memo_move[prev_coord]
                    else:
                        print("Could not find prev coordinate in MemoMove")
                    print(prev_actions)
                    print("Before extend")
                    memo_move[coord] = prev_actions.extend(cur_actions)
                    print("After extend")
                    break
            
            added_steps += 1 + costs[opposite(prev_move)]



            if coord not in memo or coord not in memo_move:
                print("No valid moves found at", coord)
                cost = costs[opposite(prev_move)]
                memo[coord] = cost
                
                actions = [constants.WAIT for i in range(cost)]
                actions.append(opposite(prev_move))

                memo_move[coord] = actions
            else:
                if hard_limit == -1 or added_steps >= hard_limit:
                    self.escape_route = deque(memo_move[prev_coord])
                    self.escaping = True

                    return self.escape_route.popleft()                    

            print("FINISHED CHECKING ", i)

        print("Reached the end")
        self.escape_route = deque(memo_move[(0, 0)])
        self.escaping = True

        return constants.WAIT
    
    # This function determines if the player is stuck by checking through past player history
    def is_stuck(self, best_move) -> bool:
        print("I am in is stuck")
        print("The chosen best move: ", best_move)
        past_moves = self.past_moves
        past_moves.append(best_move)
        total_moves = len(past_moves)
        # print(past_moves)
        if total_moves < 2:
            return False
        
        for i in range(1, (total_moves // 2)):
            stack_1_start = total_moves - i
            stack_2_start = total_moves - (2 * i)

            stack_1 = past_moves[stack_1_start:]
            stack_2 = past_moves[stack_2_start:stack_1_start]

            for j in range(i):
                stack_1_ele = stack_1.pop()
                if stack_1_ele != opposite(stack_2[-1]):
                    break
                else:
                    stack_2.pop()

            if len(stack_2) == 0:
                return True

        print("Looped through all possibilties")

        return False
    
    # MAIN FUNCTION GAME CALLS
    def move(self, current_percept) -> int:
        """Function which retrieves the current state of the amoeba map and returns an amoeba movement

            Args:
                current_percept(TimingMazeState): contains current state information
            Returns:
                int: This function returns the next move of the user:
                    WAIT = -1
                    LEFT = 0
                    UP = 1
                    RIGHT = 2
                    DOWN = 3
        """
        self.step += 1
        # Update our player memory
        self.update_graph_information(current_percept)

        # If the player is in escape mode
        if self.escaping:
            print("Player is attempting to escape!")
            print("Player plans to take the following path: ", self.escape_route)
            if not self.escape_route:
                self.escaping = False
            else:
                best_move = self.escape_route.popleft()
            
                if best_move != constants.WAIT:
                    self.update_position(best_move)

                return best_move
        
        # If we see the goal cell
        if current_percept.is_end_visible:
            best_move = self.move_toward_visible_end(current_percept)
            if best_move != constants.WAIT:
                self.update_position(best_move)
            return best_move
        # print("Before running close to corner")
        corner_coord = self.close_to_corner()
        # print("After running close to corner")
        # print(corner_coord)
        if (corner_coord != (100, 100)):
            print("I am close to a corner")

            # corner = self.Corner(corner_coord[0], corner_coord[1])

            # best_move = self.move_toward_visible_end()

            # self.update_position(best_move)
            # return best_move
        # print("After condition corner check")
        
        # The fundemental moves of player
        moves = [constants.LEFT, constants.UP, constants.RIGHT, constants.DOWN]

        # The available moves of player in the current turn
        available_moves = []
        for i in range(4):
            print(available_moves)
            if self.can_move_in_direction(i):
                print("can move in " + str(i))
                x_or_y = 0 if i % 2 == 0 else 1
                neg_or_pos = -1 if i <= 1 else 1

                changed_dim = self.cur_pos[x_or_y] + (self.radius * neg_or_pos)

                print("normal print")
                print(self.boundary[i], changed_dim)
                # Check if we have found a boundary and whether or not our vision is beyond it
                if self.boundary[i] != 100:
                    if i <= 1:
                        if changed_dim <= self.boundary[i]:
                            continue
                    else:
                        if changed_dim >= self.boundary[i]:
                            continue

                available_moves.append(i)

        if not available_moves:
            return constants.WAIT
        
        # print("Before Epsilon Greedy")
        #Epsilon-Greedy 
        exploit = random.choices([True, False], weights = [(1 - self.epsilon), self.epsilon], k = 1)
        best_move = constants.WAIT

        # Keeps track of the regret for each direction
        move_regret = []
        print("Exploiting" if exploit[0] else "Random move")
        
        if exploit[0]:
            for move in moves:
                # print("Checking Move: ", move)
                x_or_y = 0 if move % 2 == 0 else 1
                neg_or_pos = -1 if move <= 1 else 1
                changed_dim = self.cur_pos[x_or_y] + (self.radius * neg_or_pos) # The x or y value after adding radius (We want the edge cell)


                print("regret print")
                print(self.boundary[move], changed_dim)
                # If we have seen the boundary and the changed_dim is going to be over the boundary, then DO NOT move there.
                if self.boundary[i] != 100:
                    if i <= 1:
                        if changed_dim <= self.boundary[i]:
                            move_regret.append(math.inf)
                            continue
                    else:
                        if changed_dim >= self.boundary[i]:
                            move_regret.append(math.inf)
                            continue

                # if self.boundary[move] != 100 and abs(changed_dim) >= abs(self.boundary[move]):
                #     move_regret.append(math.inf)
                #     continue

                target_coord = (changed_dim if x_or_y == 0 else self.cur_pos[0], changed_dim if x_or_y == 1 else self.cur_pos[1]) # This simply sets the target coord (edge cell)
                # print("Boundary: ", self.boundary)
                # print("Boundary Check: ", self.boundary[move])
                # print("Shifted dim: ", changed_dim)
                # print("Before Regret")
                regret = 0
                
                # Loop through all surrounding cells of our target coord and add value
                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        cur_coord = (target_coord[0] + i, target_coord[1] + j)
                        if cur_coord in self.values:
                            val = self.values[cur_coord]
                            regret += val 
                # print("After heuritics calculations")
                move_regret.append(regret)
            
            # print("I am out of checking moves")
            print("Move Regret: ", move_regret)
            self.move_regrets[tuple(self.cur_pos)] = move_regret

            sorted_move_regret = sorted(range(len(move_regret)), key=lambda k : move_regret[k]) # Sorts the move regrets. Resulting values of result are the DIRECTIONS in ascending order.
            print("Sorted Move Regret: ", sorted_move_regret)

            print("Available Moves: ", available_moves)
            
            # print("After sorting")
            for move in sorted_move_regret:
                if move in available_moves:
                    best_move = move
                    break
            # print("After Regret")
            
            # If we are backtracking, then figure out why.

            start_time = time.time()

            if self.is_stuck(best_move):
                print("Took: ", time.time() - start_time, " seconds")
                print("I AM NOW STUCK")
                print("Move Regret: ", move_regret)
                best_move = self.find_best_out()
        else:
            best_move = random.choice(moves)
        
        # print("Chosen Move:", best_move)
        # print("Available moves:", moves)
        # print("Reward list:", move_rewards)
        if best_move != constants.WAIT:
            self.update_position(best_move)

        return best_move
    
    def update_position(self, direction):
        """
            Player has moved in the given direction. Append move to past_moves. Update known position to match. 
        """
        if not self.escaping:
            self.past_moves.append(direction)
            self.past_coords.append(tuple(self.cur_pos))

        if direction >= 0:
            self.cur_pos = get_neighbor(self.cur_pos, direction)

    def move_toward_visible_end(self, current_percept) -> int:
        """
            Give the next move that a player should take if they know where the endpoint is
        """
        curr_cell = tuple(self.cur_pos)
        # Look for the best path to current position if one hasn't been found yet

        if self.step == self.maximum_door_frequency and self.reset_counter <= 2:
            self.reset_counter += 1
            self.best_path_found = {}

        if len(self.best_path_found) == 0 or curr_cell not in self.best_path_found:
            result = self.find_path(current_percept.end_x, current_percept.end_y)
            if result != -2:
                return result
        
        if curr_cell not in self.best_path_found:
            return constants.WAIT
        
        direction_to_goal = self.best_path_found[curr_cell]
        if self.can_move_in_direction(direction_to_goal):
            return direction_to_goal 
        return constants.WAIT

    def can_move_in_direction(self, direction):
        """
            Whether the player can move in the specified direction.
            A door is open if our known frequency for the door is a multiple of the current step.
        """
        # Don't bother checking the neighbor's door if this door is a boundary or closed
        this_door_freq = self.door_states[tuple(self.cur_pos)][direction]

        if this_door_freq <= 0 or self.step % this_door_freq != 0:
            return False
        
        # Check neighbors door
        neighbor = get_neighbor(self.cur_pos, direction)
        if neighbor not in self.door_states:
            return False
        neighbor_door_freq = self.door_states[neighbor][opposite(direction)]

        return (neighbor_door_freq != 0) and (self.step % neighbor_door_freq == 0) and (self.step % LCM(this_door_freq, neighbor_door_freq) == 0)

    def find_path(self, goal_x, goal_y):
        """
            Given the current graph/board state and the end coordinates, use a dynamic
            Dijkstra's algorithm to find a path from every available cell to the end position,
            while minimizing the number of weighted steps (steps_in_dijkstras).
        """
        relative_x = int(goal_x + self.cur_pos[0])
        relative_y = int(goal_y + self.cur_pos[1])
        goal_coordinates = (relative_x, relative_y)

        if goal_coordinates not in self.door_states:
            return

        # Initialize steps_in_dijkstras for each cell
        steps_in_dijkstras = {cell: float('inf') for cell in self.door_states}
        steps_in_dijkstras[goal_coordinates] = 0
        self.best_path_found[goal_coordinates] = -1

        # Priority queue for Dijkstra's, based on steps_in_dijkstras
        queue = [(0, goal_coordinates)]  # (steps_in_dijkstras, cell)
        heapq.heapify(queue)
        visited = set()

        while len(queue) > 0:
            print("Queue Length: ", len(queue))
            print(queue)

            curr_steps, curr_cell = heapq.heappop(queue)  # node with min weighted steps
            if curr_cell == self.cur_pos:
                return

            if curr_cell in visited:
                continue
            visited.add(curr_cell)

            # Get neighbors
            neighbors = get_neighbors(curr_cell)
            for direction in range(4):
                neighbor = neighbors[direction]
                if (neighbor in visited) or (neighbor not in self.door_states):
                    continue

                # Get door states to calculate dynamic weight
                this_door = self.door_states[curr_cell][direction]
                neighbor_door = self.door_states[neighbor][opposite(direction)]
                if (this_door == 0) or (neighbor_door == 0):
                    continue

                max_wait = LCM(this_door, neighbor_door)
                
                # Calculate the dynamic weight based on steps_in_dijkstras
                dynamic_weight = (max_wait - ((curr_steps) % max_wait)) % max_wait
                
                # Calculate the number of weighted steps for the neighbor
                new_steps_in_dijkstras = curr_steps + dynamic_weight + 1

                # If we find a path with fewer weighted steps, update the path
                if new_steps_in_dijkstras < steps_in_dijkstras[neighbor]:
                    steps_in_dijkstras[neighbor] = new_steps_in_dijkstras
                    self.best_path_found[neighbor] = opposite(direction)

                    # Push the neighbor into the priority queue with the new steps_in_dijkstras
                    heapq.heappush(queue, (new_steps_in_dijkstras, neighbor))

        return
