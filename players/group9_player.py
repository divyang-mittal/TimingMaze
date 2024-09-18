import heapq
import math
import os
import pickle
import random
import numpy as np
import logging
import sys

import constants
from timing_maze_state import TimingMazeState

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
        self.epsilon = 0.05
        self.door_states = {}
        self.values = {}
        self.best_path_found = {}
        self.boundary = [100, 100, 100, 100]

    def update_graph_information(self, current_percept):
        for cell in current_percept.maze_state:
            relative_x = cell[0] + self.cur_pos[0]
            relative_y = cell[1] + self.cur_pos[1]
            cell_coordinates = (relative_x, relative_y)
            self.update_cell_state(cell_coordinates, cell[2], cell[3])
            self.update_cell_value(cell_coordinates, cell[3])

    def update_cell_state(self, coordinates, direction, state):
        if coordinates not in self.door_states:
            self.door_states[coordinates] = [0, 0, 0, 0] # Left Top Right Bottom

        if state == constants.OPEN:
            if self.door_states[coordinates][direction] == 0:
                self.door_states[coordinates][direction] = self.step
            else:
                self.door_states[coordinates][direction] = GCD(self.door_states[coordinates][direction], self.step)

        elif state == constants.BOUNDARY:
            self.door_states[coordinates][direction] = -1   
            x_or_y = 1 if direction % 2 == 0 else 0 # Which coordinate to update
            # print("Direction, Coordinates, x or y: ", direction, coordinates, x_or_y)
            self.boundary[direction] = coordinates[x_or_y]

    def update_cell_value(self, coordinates, door_type):
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
        # for state in current_percept.maze_state:
        #     if state[3] == 3:
        #         print(state)
        self.step += 1
        self.update_graph_information(current_percept)

        if current_percept.is_end_visible:
            return self.move_toward_visible_end(current_percept)
        
        moves = []
        for i in range(4):
            if self.can_move_in_direction(i):
                moves.append(i)

        if not moves:
            return constants.WAIT
        
        #Epsilon-Greedy 
        exploit = random.choices([True, False], weights = [(1 - self.epsilon), self.epsilon], k = 1)
        best_move = constants.WAIT

        move_rewards = []
        print("Am I Exploiting?", exploit[0])

        if exploit[0]:
            for move in moves:
                x_or_y = 0 if move % 2 == 0 else 1
                neg_or_pos = -1 if move <= 1 else 1
                changed_dim = self.cur_pos[x_or_y] + (self.radius * neg_or_pos)
                target_coord = (changed_dim if x_or_y == 0 else self.cur_pos[0], changed_dim if x_or_y == 1 else self.cur_pos[1]) 
                
                # Check if we have found a boundary and whether or not our vision is beyond it
                # print("Move: ", move)
                # print("boundary for move: ", self.boundary[move])
                # print("shifted: ", abs(changed_dim) )
                if self.boundary[move] != 100 and abs(changed_dim) >= abs(self.boundary[move]):
                    move_rewards.append(math.inf)
                    continue
                # print(self.boundary)
                regret = 0

                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        cur_coord = (target_coord[0] + i, target_coord[1] + j)
                        if cur_coord in self.values:
                            val = self.values[cur_coord]
                            regret += val if val > 0 else 50 #Arbitrary 50 for the reward of a nearby boundary
                move_rewards.append(regret)
            
            min_regret = math.inf
            for i in range(len(move_rewards)):
                if move_rewards[i] < min_regret:
                    min_regret = move_rewards[i]
                    best_move = moves[i]
        else:
            best_move = random.choice(moves)
        
        match best_move:
            case constants.LEFT:
                self.cur_pos[0] -= 1
            case constants.UP:
                self.cur_pos[1] -= 1
            case constants.RIGHT:
                self.cur_pos[0] += 1
            case constants.DOWN:
                self.cur_pos[1] += 1

        return best_move

    
    def move_toward_visible_end(self, current_percept) -> int:
        """
            Give the next move that a player should take if they know where the endpoint is
        """
        curr_cell = (self.cur_pos[0], self.cur_pos[1])
        # Look for the best path to current position if one hasn't been found yet
        if len(self.best_path_found) == 0 or curr_cell not in self.best_path_found:
            self.find_path(current_percept.end_x, current_percept.end_y)
            return constants.WAIT
        
        

        direction_to_goal = self.best_path_found[curr_cell]

        if self.can_move_in_direction(direction_to_goal):
            print(direction_to_goal)
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
        neighbors = get_neighbors(self.cur_pos) #Could be optimized, currently getting all neighbors in every call
        neighbor = neighbors[direction]
        neighbor_door_freq = self.door_states[neighbor][opposite(direction)]

        return (neighbor_door_freq != 0) and (self.step % neighbor_door_freq == 0) and (self.step % LCM(this_door_freq, neighbor_door_freq) == 0)

    def find_path(self, goal_x, goal_y):
        """
            Given the current graph/board state and the end coordinates, use Dijkstra's 
            algorithm to find a path from every available cell to the end position.
        """
        print(goal_x, goal_y)

        print(self.door_states[(goal_x, goal_y)])

        goal_coordinates = (goal_x, goal_y)
        if goal_coordinates not in self.door_states:
            return {}

        dist = {cell: float("inf") for cell in self.door_states}
        dist[(goal_coordinates)] = 0

        queue = [(0, goal_coordinates)]
        heapq.heapify(queue)
        visited = {}
        # direction_to_goal = {} # <k = cell coordinate, v = what direction player should go from that cell>

        print("find path 2")

        while len(queue) > 0:
            curr_dist, curr_cell = heapq.heappop(queue) # node with min dist
            print(visited)
            if curr_cell in visited:
                continue
            visited.add(curr_cell)
            neighbors = get_neighbors(curr_cell)

            for direction in range(3):
                neighbor = neighbors[direction]
                if (neighbor in visited) or (neighbor not in self.door_states):
                    continue

                # how often are we able to go in this direction?
                # lowest common multiple
                max_wait = LCM(self.door_states[curr_cell][direction], self.door_states[neighbor][opposite(direction)])

                dist_from_here = curr_dist + max_wait + 1
                if dist_from_here < dist[neighbor]:
                    # best way to get to neighbor (so far) is from here. 
                    # meaning if player is at neighbor, it should go to curr_cell to reach goal
                    dist[neighbor] = dist_from_here
                    self.best_path_found[neighbor] = opposite(direction)

                    heapq.heappush(queue, (dist_from_here, neighbor))


        #print(direction_to_goal)

        #return direction_to_goal

def get_neighbors(coordinates): 
    # left up right down
    x = coordinates[0]
    y = coordinates[1]
    return [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]

# Helper that maps directions to their opposites 
def opposite(direction) -> int:
    if direction == constants.UP:
        return constants.DOWN
    elif direction == constants.DOWN:
        return constants.UP
    elif direction == constants.LEFT:
        return constants.RIGHT
    return constants.LEFT