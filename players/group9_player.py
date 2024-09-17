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
        if b == 0:
            return a
        return GCD(b, a % b)

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
        self.positions = {}
        self.values = {}

    def update_graph_information(self,current_percept):
        for cell in current_percept.maze_state:
            relative_x = cell[0] + self.cur_pos[0]
            relative_y = cell[1] + self.cur_pos[1]
            cell_coordinates = (relative_x, relative_y)

            update_cell_state(coordinates, cell)
            update_cell_value(coordinates)

    def update_cell_state(coordinates, direction, state):
        if coordinates not in self.positions:
            self.positions[coordinates] = [0, 0, 0, 0] # Left Top Right Bottom

        if state == constants.OPEN:
            if self.positions[coordinates][direction] == 0:
                self.positions[coordinates][direction] = self.step # TODO: set this to be greatest factor of step <= L
            else:
                self.positions[coordinates][direction] = GCD(self.positions[coordinates][direction], self.step)

        elif state == constants.BOUNDARY:
            # TODO: handle boundary


    def update_cell_value(coordinates):
        if coordinates not in self.values:
            if cell[3] == constants.BOUNDARY:
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
        if current_percept.is_end_visible:
            return self.move_toward_visible_end()
        
        self.update_door_state(current_percept)
        self.updateValues(current_percept.maze_state)
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

    
    def move_toward_visible_end(self, door_info) -> int:
        """
            Give the next move that a player should take if they know where the endpoint is
        """
        curr_cell = (self.pos[0], self.pos[1])
        if len(self.best_path_found) == 0 or curr_cell not in self.best_path_found:
            G = Graph(door_info)
            self.best_path_found = G.find_path(current_percept.end_x, current_percept.end_y)

        if curr_cell not in self.best_path_found:
            return constants.WAIT # TODO: This should be some error case. It means no possible path was found given our visible state.

        direction_to_goal = self.best_path_found[curr_cell]
        if self.can_move_in_direction(direction_to_goal):
            return direction_to_goal
        return constants.WAIT

        # ideally: 

    def can_move_in_direction(self, direction):
        # TODO: this depends on library
        return True


class Door():
    # PLACEHOLDER for library classes
    def __init__(self, x, y, direction, frequency):
        self.x = x 
        self.y = y 
        self.direction = direction
        self.frequency = frequency 
    
class Cell():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.door_freqs = {}
        self.neighbors = {
            constants.UP: (self.x, self.y - 1),
            constants.DOWN: (self.x, self.y + 1),
            constants.LEFT: (self.x-1, self.y),
            constants.RIGHT: (self.x+1, self.y)
        }

class Graph():
    # TODO: depending on how doors are stored in the library, this initial processing
    # may not be necessary at all. I did this to be able to more efficiently find cells at 
    # a given cardinal position.
    def __init__(self, door_info):
        # a cell at coordinate (x, y) can be found with self.V[(x,y)]
        self.V = {}
        for door in door_info:
            coordinates = (door.x, door.y)
            if coordinates not in self.V:
                self.V[coordinates] = Cell(door.x, door.y)
            
            self.V[coordinates].door_freqs[door.direction] = door.frequency

    def find_path(self, goal_x, goal_y):
        """
            Given the current graph/board state and the end coordinates, use Dijkstra's 
            algorithm to find a path from every available cell to the end position.
        """
        goal_coordinates = (goal_x, goal_y)
        if goal_coordinates not in self.V:
            return {}
        goal_cell = self.V[goal_coordinates]

        dist = {cell: float("inf") for cell in self.V}
        dist[goal_cell] = 0

        queue = [(0, goal_cell)]
        heapq.heapify(queue)
        
        visited = {}
        direction_to_goal = {}

        while len(queue) > 0:
            curr_dist, curr_cell = heapq.heappop(queue) # node with min dist

            if curr_cell in visited:
                continue
            visited.add(curr_cell)

            for direction in range(3):
                neighbor_coordinates = cell.neighbors[direction]
                if neighbor_coordinates not in self.V:
                    continue
                neighbor = self.V[neighbor_coordinates]

                # how often are we able to go in this direction?
                edge_freq = curr_cell.door_freqs[direction] * neighbor.door_freqs[opposite(direction)]

                dist_from_here = curr_dist + edge_freq
                if dist_from_here < dist[neighbor]:
                    # best way to get to neighbor (so far) is from here. 
                    # meaning if player is at neighbor, it should go to curr_cell to reach goal
                    dist[neighbor] = dist_from_here
                    direction_to_goal[neighbor_coordinates] = opposite(direction)

                    heapq.heappush(queue, (dist_from_here, neighbor))

        return direction_to_goal

# Helper that maps directions to their opposites 
def opposite(direction) -> int:
    if direction == constants.UP:
        return constants.DOWN
    elif direction == constants.DOWN:
        return constants.UP
    elif direction == constants.LEFT:
        return constants.RIGHT
    return constants.LEFT