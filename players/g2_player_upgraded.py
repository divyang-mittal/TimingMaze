import os
import pickle
import numpy as np
import logging
import math
import heapq
from collections import deque

import constants
from timing_maze_state import TimingMazeState

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
        
        self.curr_x = 0
        self.curr_y = 0
        
        self.is_end_visible = False
        self.path_to_end = None
        self.target_x = None
        self.target_y = None
        
        self.door_frequencies = {}
        self.default_frequency = sum(range(1, self.maximum_door_frequency + 1))/self.maximum_door_frequency
        
        self.turn_number = 0
    
    def update_door_frequencies(self, maze_state):
        for cell_door in maze_state:
            # Convert cell coordinates to global coordinates
            cell_x = cell_door[0] + self.curr_x
            cell_y = cell_door[1] + self.curr_y
            
            # If the cell is not in the door frequencies, add it
            if (cell_x, cell_y) not in self.door_frequencies:
                self.door_frequencies[(cell_x, cell_y)] = {}
                
            # If the door is not in the cell's door frequencies, add it
            if cell_door[2] not in self.door_frequencies[(cell_x, cell_y)]:
                self.door_frequencies[(cell_x, cell_y)][cell_door[2]] = {
                    'possibilities': set(range(1, self.maximum_door_frequency + 1)), 
                    'frequency': self.default_frequency, 
                    'certain': False
                }
            
            current_door = self.door_frequencies[(cell_x, cell_y)][cell_door[2]]
            
            # If the door is not certain, update the door frequency
            if not current_door['certain']:
                # If the door is a boundary, it is certainly closed
                if cell_door[3] == constants.BOUNDARY:
                    current_door['frequency'] = float('inf')
                    current_door['certain'] = True
                    continue
                
                # Remove possibilities that are not consistent with the current state
                to_remove = set()
                for possibility in current_door['possibilities']:
                    if cell_door[3] == constants.OPEN and self.turn_number % possibility != 0:
                        to_remove.add(possibility)
                    if cell_door[3] == constants.CLOSED and self.turn_number % possibility == 0:
                        to_remove.add(possibility)
                current_door['possibilities'] -= to_remove
                
                # If there are no possibilities left, the door is certainly closed
                if len(current_door['possibilities']) == 0:
                    current_door['frequency'] = float('inf')
                    current_door['certain'] = True
                # If the door is open and there is only one possibility left, the door is certainly that frequency
                elif cell_door[3] == constants.OPEN and len(current_door['possibilities']) == 1:
                    current_door['frequency'] = list(current_door['possibilities'])[0]
                    current_door['certain'] = True
                else:
                    current_door['frequency'] = sum(current_door['possibilities']) // len(current_door['possibilities'])

    def a_star_search(self, start, target):
        # Open set represented as a priority queue with (f_score, node)
        open_set = []
        heapq.heappush(open_set, (0, start, self.turn_number))

        # Maps nodes to their parent node
        came_from = {} # (x, y) -> (x, y)

        # Cost from start to a node
        g_score = {start: 0} # (x, y) -> int

        vis = set()
        while open_set:
            # Get the node in open_set with the lowest f_score
            current_f_score, current, current_turn = heapq.heappop(open_set)
            vis.add(current)
            
            # Check if we have reached the goal
            if current == target:
                return self.reconstruct_path(came_from, current)
            
            # Explore neighbors of current node
            moves = [
                (-1, 0, 0, 2), # Left (x - 1, y, left, right)
                (0, -1, 1, 3), # Up (x, y - 1, up, down)
                (1, 0, 2, 0), # Right (x + 1, y, right, left)
                (0, 1, 3, 1) # Down (x, y + 1, down, up)
            ]
            
            # MAKE SURE TO CHECK FOR BOUNDARIES!!!
            for move in moves:
                neighbor = (current[0] + move[0], current[1] + move[1])
                lcm = self.calculate_LCM(self.door_frequencies[current][move[2]]['frequency'], self.door_frequencies[neighbor][move[3]]['frequency'])
                tentative_g_score = g_score[current] + lcm - current_turn % lcm

                if (neighbor not in g_score or tentative_g_score < g_score[neighbor]) and neighbor not in vis:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_score, neighbor, current_turn + 1))

        # No path found
        return None
    
    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def calculate_LCM(self, freq_1, freq_2):
        if freq_1 == float('inf') or freq_2 == float('inf'):
            return float('inf')
        return abs(freq_1 * freq_2) // math.gcd(freq_1, freq_2)
    
    def heuristic(self, start, end):
        return abs(start[0] - end[0]) + abs(start[1] - end[1])
    
    def coordinates_to_moves(self, path):
        moves = deque()
        for i in range(1, len(path)):
            dx, dy = path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]
            if dx == -1:
                moves.append(constants.LEFT)
            elif dx == 1:
                moves.append(constants.RIGHT)
            elif dy == -1:
                moves.append(constants.UP)
            elif dy == 1:
                moves.append(constants.DOWN)
        return moves
    
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
        self.turn_number += 1
                
        self.curr_x, self.curr_y = -current_percept.start_x, -current_percept.start_y
        print(f'Start: [{self.curr_x}, {self.curr_y}]')
        if current_percept.is_end_visible:
            self.target_x = current_percept.end_x + self.curr_x
            self.target_y = current_percept.end_y + self.curr_y
            self.is_end_visible = True
            print(f'End: [{self.target_x}, {self.target_y}]')
            
        self.update_door_frequencies(current_percept.maze_state)
        
        if self.turn_number < 100:
            return constants.WAIT
        
        if not self.path_to_end:
            path = self.a_star_search((self.curr_x, self.curr_y), (self.target_x, self.target_y))
            print(path)
            self.path_to_end = self.coordinates_to_moves(path)
    
        attempted_direction = self.path_to_end[0]
            
        direction = [0, 0, 0, 0]
        for maze_state in current_percept.maze_state:
            if maze_state[0] == 0 and maze_state[1] == 0:
                direction[maze_state[2]] = maze_state[3]

        if attempted_direction == constants.LEFT and direction[constants.LEFT] == constants.OPEN:
            for maze_state in current_percept.maze_state:
                if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                        and maze_state[3] == constants.OPEN):
                    return self.path_to_end.popleft()
        if attempted_direction == constants.DOWN and direction[constants.DOWN] == constants.OPEN:
            for maze_state in current_percept.maze_state:
                if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                        and maze_state[3] == constants.OPEN):
                    return self.path_to_end.popleft()
        if attempted_direction == constants.RIGHT and direction[constants.RIGHT] == constants.OPEN:
            for maze_state in current_percept.maze_state:
                if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                        and maze_state[3] == constants.OPEN):
                    return self.path_to_end.popleft()
        if attempted_direction == constants.UP and direction[constants.UP] == constants.OPEN:
            for maze_state in current_percept.maze_state:
                if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                        and maze_state[3] == constants.OPEN):
                    return self.path_to_end.popleft()
        return constants.WAIT