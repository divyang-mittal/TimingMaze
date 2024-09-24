import os
import pickle
import numpy as np
import logging
import math
import heapq
import random
from collections import deque, defaultdict

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
        self.maximum_door_frequency = min(maximum_door_frequency, 5000)
        self.radius = radius
        
        self.curr_x = 0
        self.curr_y = 0
        
        self.is_end_visible = False
        self.path_to_end = None
        self.target_x = None
        self.target_y = None
        
        self.door_frequencies = {}
        self.default_frequency = sum(range(1, self.maximum_door_frequency + 1))/self.maximum_door_frequency
        
        self.times_discovered = defaultdict(int)
        self.current_destination = None
        
        self.turn_number = 0
        self.turn_path_changed = 0
        
    def determine_destination(self, invalid_tries):
        """
        Determines the next destination based on the least discovered cells and heuristic score, 
        while excluding the current cell and any invalid destination attempts.

        Inputs:
            invalid_tries (set): A set of cells that have already been tried and deemed invalid.

        Outputs:
            destination (tuple): The cell with the lowest score based on the number of times it has been discovered 
                                and its heuristic distance from the current position. Returns None if no valid destination is found.
        """
        min_score = float('inf')
        destination = None
        for cell in self.times_discovered:
            if cell == (self.curr_x, self.curr_y) or cell in invalid_tries:
                continue
            score = self.times_discovered[cell] + self.heuristic((self.curr_x, self.curr_y), cell)
            if score < min_score:
                min_score = score
                destination = cell
        return destination
    
    def update_door_frequencies(self, maze_state):
        """
        Updates the frequency estimation for each door in the visible portion of the maze.

        Inputs:
            maze_state (list): List of door states in the maze within the current view (each entry contains coordinates and door state).
            
        Outputs:
            None (modifies class attributes: self.door_frequencies, self.times_discovered).
        """
        for cell_door in maze_state:
            # Convert cell coordinates to global coordinates
            cell_x = cell_door[0] + self.curr_x
            cell_y = cell_door[1] + self.curr_y
            
            self.times_discovered[(cell_x, cell_y)] += 1
            
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
        """
        Performs A* search to find the optimal path from the start position to the target position.

        Inputs:
            start (tuple): The starting coordinates of the search (x, y).
            target (tuple): The target coordinates to reach (x, y).

        Outputs:
            list: A list representing the path from start to target, or None if no path is found.
        """
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
            
            for move in moves:
                neighbor = (current[0] + move[0], current[1] + move[1])
                
                # Check for boundaries
                if neighbor not in self.door_frequencies or current not in self.door_frequencies:
                    continue  # Skip if no information is available for the current or neighbor position

                if move[2] not in self.door_frequencies[current] or move[3] not in self.door_frequencies[neighbor]:
                    continue  # Skip if the required door information is not available

                lcm = self.calculate_LCM(
                    self.door_frequencies[current][move[2]]['frequency'],
                    self.door_frequencies[neighbor][move[3]]['frequency']
                )
                
                tentative_g_score = g_score[current] + lcm - current_turn % lcm

                if (neighbor not in g_score or tentative_g_score < g_score[neighbor]) and neighbor not in vis:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_score, neighbor, current_turn + 1))

        # No path found
        return None
    
    def reconstruct_path(self, came_from, current):
        """
        Reconstructs the path from the start to the target node.

        Inputs:
            came_from (dict): A mapping of each node to its parent node (i.e., where it came from).
            current (tuple): The current node (usually the target) to start backtracking from.

        Outputs:
            path (list): A list representing the path from start to target, in order.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def calculate_LCM(self, freq_1, freq_2):
        """
        Calculates the least common multiple (LCM) of two frequencies, handling infinity as a special case.

        Inputs:
            freq_1 (int or float): The frequency of the first door.
            freq_2 (int or float): The frequency of the second door.

        Outputs:
            int: The LCM of the two frequencies, or infinity if either frequency is infinite.
        """
        if freq_1 == float('inf') or freq_2 == float('inf'):
            return float('inf')
        return abs(freq_1 * freq_2) // math.gcd(freq_1, freq_2)
    
    def heuristic(self, start, end):
        """
        Calculates the Manhattan distance (heuristic) between two points.

        Inputs:
            start (tuple): The starting coordinates (x, y).
            end (tuple): The target coordinates (x, y).

        Outputs:
            int: The Manhattan distance between the start and end points.
        """
        return abs(start[0] - end[0]) + abs(start[1] - end[1])
    
    def coordinates_to_moves(self, path):
        """
        Converts a sequence of coordinates into a sequence of movement directions.

        Inputs:
            path (list): A list of tuples representing the path, where each tuple is a (x, y) coordinate.

        Outputs:
            moves (deque): A deque containing the movement directions (constants.LEFT, constants.RIGHT, constants.UP, constants.DOWN)
                corresponding to the changes in coordinates along the path.
        """
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
                current_percept (TimingMazeState): contains current state information
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
        
        if current_percept.is_end_visible and not self.is_end_visible:
            self.target_x = current_percept.end_x + self.curr_x
            self.target_y = current_percept.end_y + self.curr_y
            self.is_end_visible = True
            
        self.update_door_frequencies(current_percept.maze_state)
        
        if self.is_end_visible and self.current_destination != (self.target_x, self.target_y):
            path = self.a_star_search((self.curr_x, self.curr_y), (self.target_x, self.target_y))
            if path:
                self.path_to_end = self.coordinates_to_moves(path)
                self.turn_path_changed = self.turn_number
                self.current_destination = (self.target_x, self.target_y)
                
        num_tries = 0
        invalid_tries = set()
        while not self.path_to_end and num_tries < 5:
            self.current_destination = self.determine_destination(invalid_tries)
            path = self.a_star_search((self.curr_x, self.curr_y), self.current_destination)
            self.path_to_end = self.coordinates_to_moves(path)
            self.turn_path_changed = self.turn_number
            invalid_tries.add(self.current_destination)
            num_tries += 1
            
        if self.path_to_end and (self.turn_number - self.turn_path_changed) % 3 == 0:
            path = self.a_star_search((self.curr_x, self.curr_y), self.current_destination)
            self.path_to_end = self.coordinates_to_moves(path)
            
        if not self.path_to_end:
            return random.choice([constants.LEFT, constants.UP, constants.RIGHT, constants.DOWN])
        
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