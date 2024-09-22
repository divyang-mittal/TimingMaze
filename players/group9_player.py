import heapq
import math
import os
import pickle
import random
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

        self.step = 0
        self.cur_pos = [0, 0]
        self.epsilon = 0.0
        self.door_states = {}
        self.values = {}
        self.best_path_found = {}
        self.boundary = [100, 100, 100, 100]
        self.corners = [[100, 100], [100, 100], [100, 100], [100, 100]] # (Left, Bottom), (Left, Top), (Right, Top), (Right, Bottom), 
        self.past_moves = []
        self.past_coords = [] 
        self.sorted_moves = {}
        self.waited = 0
        self.escaping = False
        self.escape_route = deque([])

    class Corner:
        def __init__(self, x, y):
            self.end_x = x
            self.end_y = y

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
            if dist_from_corner <= math.sqrt(2(self.radius ** 2)) + self.radius:
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

            if door_freq <= 0 or neighbor_door_freq <= 0:
                costs.append(-1)
            else:
                common_freq = LCM(door_freq, neighbor_door_freq)
                if current_turn >= common_freq:
                    while (common_freq < current_turn):
                        common_freq *= 2
                costs.append(common_freq - current_turn)
                    
        return costs

    # Player is stuck. Formulate a plot or end up in jail or shot. Updates self.escape_route and self.escaping to True.
    def find_best_out(self):
        hard_limit = -1
        added_steps = 0

        memo = {}
        memo_move = {}
        checked_coords = set()

        # Check current coord
        checked_coords.add(tuple(self.cur_pos)) 
        costs = self.cost_of_directions(tuple(self.cur_pos), added_steps)
        # sorted_moves = self.sorted_moves[tuple(self.cur_pos)]
        fastest_moves = sorted(range(len(costs)), key=lambda k : costs[k])
        for move in fastest_moves:
            if move != opposite(self.past_moves[-1]):
                cost = costs[move]
                memo[tuple(self.cur_pos)] = cost

                actions = [constants.WAIT for i in range(cost)]
                actions.append(move)

                memo_move[tuple(self.cur_pos)] = actions

                break

        # Error Check
        if not memo or not memo_move:
            self.logger.error("Memo or Memo_move has not been filled")


        for i in range(3, -1, -1):
            cost = costs[fastest_moves[i]]
            if cost != -1:
                hard_limit = cost
                break

        if hard_limit == -1:
            self.logger.error("No Cost for Moves have been found")

        if hard_limit == memo[tuple(self.cur_pos)]:
            self.logger.log("There was only the option to go back, either need more info or dead end")
            return constants.WAIT
        
        added_steps += costs[opposite(self.past_moves[-1])]

        # If backtracking takes more time then just waiting
        if added_steps >= hard_limit:
            self.escape_route = deque(memo_move[tuple(self.cur_pos)])

            return self.escape_route.popleft()
        
        move_taken = opposite(self.past_moves[-1])
        moves_taken = 1
        # Check all past coords to find if there is some out.
        for i in range(len(self.past_coords) - 1, -1, -1):
            coord = self.past_coords[i]
            if coord in checked_coords:
                continue
            checked_coords.add(coord)

            costs = self.cost_of_directions(coord, added_steps)
            fastest_moves = sorted(range(len(costs)), key=lambda k : costs[k])
            for move in fastest_moves:
                if move != opposite(self.past_moves[(-1 - moves_taken)]):
                    cost = costs[move]
                    prev_coord = get_neighbor(coord, opposite(move_taken))

                    memo[coord] = cost

                    cur_actions = [constants.WAIT for i in range(cost)]
                    cur_actions.append(move)

                    prev_actions = memo_move[prev_coord]

                    memo_move[coord] = prev_actions.extend(cur_actions)

                    if (cost + memo[prev_coord]) >= hard_limit:
                        self.escape_route = deque(memo_move[prev_coord])

                        return self.escape_route.popleft()
                    
                    moves_taken += 1
                    move_taken = move
                    break

        self.escaping = True

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
        self.update_graph_information(current_percept)

        if self.escaping:
            best_move = self.escape_route.popleft()
            if self.escape_route:
                self.escaping = False
            self.update_position(best_move)
            return best_move

        if current_percept.is_end_visible:
            best_move = self.move_toward_visible_end(current_percept)
            self.update_position(best_move)
            return best_move

        corner_coord = self.close_to_corner()
        if (corner_coord != (100, 100)):
            self.logger.log("I am close to a corner")
            best_move = self.move_toward_visible_end(self.Corner(corner_coord[0], corner_coord[1]))
            self.update_position(best_move)
            return best_move

        moves = [constants.LEFT, constants.TOP, constants.RIGHT, constants.BOTTOM]

        available_moves = []
        for i in range(4):
            if self.can_move_in_direction(i):
                x_or_y = 0 if i % 2 == 0 else 1
                neg_or_pos = -1 if i <= 1 else 1

                changed_dim = self.cur_pos[x_or_y] + (self.radius * neg_or_pos)
                # Check if we have found a boundary and whether or not our vision is beyond it
                if self.boundary[i] != 100 and abs(changed_dim) >= abs(self.boundary[move]):
                    continue

                available_moves.append(i)

        if not available_moves:
            return constants.WAIT

        #Epsilon-Greedy 
        exploit = random.choices([True, False], weights = [(1 - self.epsilon), self.epsilon], k = 1)
        best_move = constants.WAIT

        move_regret = []
        print("Exploiting" if exploit[0] else "Random move")
        
        if exploit[0]:
            for move in moves:
                # print("Checking Move: ", move)
                x_or_y = 0 if move % 2 == 0 else 1
                neg_or_pos = -1 if move <= 1 else 1
                changed_dim = self.cur_pos[x_or_y] + (self.radius * neg_or_pos)
                if self.boundary[i] != 100 and abs(changed_dim) >= abs(self.boundary[move]):
                    move_regret.append(math.inf)
                    continue
                target_coord = (changed_dim if x_or_y == 0 else self.cur_pos[0], changed_dim if x_or_y == 1 else self.cur_pos[1]) 
                # print("Boundary: ", self.boundary)
                # print("Boundary Check: ", self.boundary[move])
                # print("Shifted dim: ", changed_dim)
                
                regret = 0

                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        cur_coord = (target_coord[0] + i, target_coord[1] + j)
                        if cur_coord in self.values:
                            val = self.values[cur_coord]
                            regret += val 
                move_regret.append(regret)
            
            sorted_move_regret = sorted(range(len(move_regret)), key=lambda k : move_regret[k])
            self.sorted_moves[self.cur_pos] = sorted_move_regret

            for move in sorted_move_regret:
                if move in available_moves:
                    best_move = move
                    break
            
            # TODO: Find a good heuristic for waiting
            # If we are backtracking, then figure out why.
            if (opposite(self.past_moves[-1]) == best_move):
                best_move = self.find_best_out()
        else:
            best_move = random.choice(moves)
        
        # print("Chosen Move:", best_move)
        # print("Available moves:", moves)
        # print("Reward list:", move_rewards)
        self.update_position(best_move)

        return best_move
    
    def update_position(self, direction):
        """
            Player has moved in the given direction. Append move to past_moves. Update known position to match. 
        """
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
        if len(self.best_path_found) == 0 or curr_cell not in self.best_path_found:
            self.find_path(current_percept.end_x, current_percept.end_y)
        
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
            Given the current graph/board state and the end coordinates, use Dijkstra's 
            algorithm to find a path from every available cell to the end position.
        """
        relative_x = int(goal_x + self.cur_pos[0])
        relative_y = int(goal_y + self.cur_pos[1])
        goal_coordinates = (relative_x, relative_y)

        if goal_coordinates not in self.door_states:
            return
        
        dist = {cell: (self.maximum_door_frequency * 100) for cell in self.door_states}
        dist[goal_coordinates] = 0
        self.best_path_found[goal_coordinates] = -1

        queue = [(0, goal_coordinates)]
        heapq.heapify(queue)
        visited = set()

        while len(queue) > 0:
            curr_dist, curr_cell = heapq.heappop(queue) # node with min dist
            if curr_cell == self.cur_pos:
                return
            
            if curr_cell in visited:
                continue
            visited.add(curr_cell)

            neighbors = get_neighbors(curr_cell)
            for direction in range(4):
                neighbor = neighbors[direction]
                if (neighbor in visited) or (neighbor not in self.door_states):
                    continue

                # how often are we able to go pn this direction?
                # lowest common multiple
                this_door = self.door_states[curr_cell][direction]
                neighbor_door = self.door_states[neighbor][opposite(direction)]
                if (this_door == 0) or (neighbor_door == 0):
                    continue

                max_wait = LCM(this_door, neighbor_door)
                dist_from_here = curr_dist + max_wait + 1 + manhattan_dist(self.cur_pos, neighbor)
                if dist_from_here < dist[neighbor]:
                    # best way to get to neighbor (so far) is from here. 
                    # meaning if player is at neighbor, it should go to curr_cell to reach goal
                    dist[neighbor] = dist_from_here
                    self.best_path_found[neighbor] = opposite(direction)

                    heapq.heappush(queue, (dist_from_here, neighbor))