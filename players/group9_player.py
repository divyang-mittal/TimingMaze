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

def valid_moves(surrounding_doors) -> list[int]:
        moves = []
        boundaries = 0
        for direction in range(4):
            if surrounding_doors[direction][3] == constants.BOUNDARY:
                boundaries -= 1
                continue
            if surrounding_doors[direction][3] == constants.OPEN and surrounding_doors[((direction + boundaries + 1) * 4) + ((direction + 2) % 4)][3] == constants.OPEN:
                moves.append(direction)
        
        moves.append(constants.WAIT)
        return moves

def GCD(a, b):
        if b == 0: # TODO: Should this be 1?
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
        self.epsilon = 0.2
        self.door_states = {}
        self.values = {}
        self.best_path_found = {}

    def update_graph_information(self, current_percept):
        for cell in current_percept.maze_state:
            relative_x = int(cell[0] + self.cur_pos[0])
            relative_y = int(cell[1] + self.cur_pos[1])
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


    def update_cell_value(self, coordinates, door_type):
        if coordinates not in self.values:
            if door_type == constants.BOUNDARY:
                self.values[coordinates] = -1
            else: 
                self.values[coordinates] = 0
        else:
            if self.values[coordinates] != -1:
                self.values[coordinates] += 1

        self.values[(self.cur_pos[0], self.cur_pos[1])] += self.maximum_door_frequency

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


        if current_percept.is_end_visible:
            best_move = self.move_toward_visible_end(current_percept)
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
        
        moves = valid_moves(current_percept.maze_state[:20])
        if not moves:
            return constants.WAIT
        
        # Epsilon-Greedy 
        exploit = random.choices([True, False], weights = [(1 - self.epsilon), self.epsilon], k = 1)
        best_move = constants.WAIT

        if exploit[0]:
            move_values = [0, 0, 0, 0]
            for key, val in self.values.items():
                if val == -1 and math.sqrt((key[0] - self.cur_pos[0]) ** 2 + (key[1] - self.cur_pos[1]) ** 2) <= self.radius:
                    bound = [-1, -1, -1, -1]

                    if key[0] <= self.cur_pos[0]:
                        bound[constants.LEFT] = abs(key[0] - self.cur_pos[0])
                    else:
                        bound[constants.RIGHT] = abs(key[0] - self.cur_pos[0])
                    if key[1] <= self.cur_pos[1]:
                        bound[constants.UP] = abs(key[1] - self.cur_pos[1])
                    else:
                        bound[constants.DOWN] = abs(key[1] - self.cur_pos[1])
                    
                    max_dist = -math.inf
                    closest_dir = [-1]
                    for ind in range(4):
                        if bound[ind] > max_dist and bound[ind] >= 0:
                            max_dist = bound[ind]
                            closest_dir[0] = ind
                        elif bound[ind] == max_dist and bound[ind] >= 0:
                            closest_dir.append(ind)
                        
                    for dir in closest_dir:
                        move_values[dir] = math.inf
                
                if key[0] < self.cur_pos[0]:
                    move_values[constants.LEFT] += val
                if key[1] > self.cur_pos[1]:
                    move_values[constants.UP] += val
                if key[0] > self.cur_pos[0]:
                    move_values[constants.RIGHT] += val
                if key[1] < self.cur_pos[1]:
                    move_values[constants.DOWN] += val

            min_val = math.inf
            for ind in range(4):
                if move_values[ind] < min_val and ind in moves:
                    min_val = move_values[ind]
                    best_move = ind
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
        curr_cell = tuple(self.cur_pos)
        # Look for the best path to current position if one hasn't been found yet
        if len(self.best_path_found) == 0 or curr_cell not in self.best_path_found:
            self.find_path(current_percept.end_x, current_percept.end_y)
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
        if this_door_freq == 0 or self.step % this_door_freq != 0:
            return False
        
        # Check neighbors door
        neighbor = get_neighbor(self.cur_pos, direction)
        if neighbor not in self.door_states:
            return False
        neighbor_door_freq = self.door_states[neighbor][opposite(direction)]
        return (neighbor_door_freq != 0) and (self.step % neighbor_door_freq == 0)

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

                
def get_neighbor(coordinates, direction):
    return get_neighbors(coordinates)[direction]

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