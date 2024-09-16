import numpy as np
import logging
from timing_maze_state import TimingMazeState
import constants
import math
from experience import Experience
from collections import deque
import heapq

class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger,
                 precomp_dir: str, maximum_door_frequency: int, radius: int) -> None:
        self.rng = rng
        self.logger = logger
        self.maximum_door_frequency = maximum_door_frequency
        self.radius = radius
        self.turn = 0
        self.experience = Experience(self.maximum_door_frequency, self.radius)
        self.current_path = []
        self.frequency = {}  
        self.frequency_collection_turns = 4  #Have to find optimal waiting time need to threshhold for this

    def move(self, current_percept) -> int:
        self.turn += 1
        self.update_door_frequencies(current_percept)

        if self.turn <= self.frequency_collection_turns:
            self.logger.info(f"Collecting frequency data. Turn {self.turn}/{self.frequency_collection_turns}")
            return constants.WAIT

        if current_percept.is_end_visible:
            goal = (current_percept.end_x, current_percept.end_y)
            start = (0, 0)

            if not self.current_path:
                self.current_path = self.frequency_aware_astar(start, goal, current_percept)
                self.logger.info(f"Calculated path: {self.current_path}")

            if self.current_path:
                next_move = self.current_path[0]
                self.logger.info(f"Next move: {next_move}")
                if self.experience.is_valid_move(current_percept, next_move):
                    self.current_path.pop(0)
                    return next_move
                else:
                    self.current_path = []
        else:
            move = self.experience.move(current_percept)
            if self.experience.is_valid_move(current_percept, move):
                return move
            self.experience.wait()
            return constants.WAIT

        # If no valid move is available or the path is empty, wait
        # Need to include threshold if waiting for too long take greedy approach or random step
        self.logger.info("No valid move available, waiting")
        return constants.WAIT

    def update_door_frequencies(self, current_percept):
        for x, y, direction, state in current_percept.maze_state:
            if state == constants.OPEN:
                glob_x = x-current_percept.start_x
                glob_y = y-current_percept.start_y
                key = (glob_y, glob_x, direction)
                self.logger.info(f"{x},{y} direction: {direction} is open at turn {self.turn}")
                if key not in self.frequency:
                    self.frequency[key] = self.turn
                else:
                    self.frequency[key]= math.gcd(self.turn, self.frequency[key])

    def get_door_frequency(self, x, y, direction):
        key = (x, y, direction)
        if key in self.frequency:
            return self.frequency[key]
        return self.maximum_door_frequency

    def frequency_aware_astar(self, start, goal, current_percept):
        open_list = []
        closed_set = set()
        
        start_node = FrequencyAwareNode(start[0], start[1], 0, self.manhattan_distance(start, goal))
        heapq.heappush(open_list, (start_node.f_cost, start_node))
        
        while open_list:
            current_node = heapq.heappop(open_list)[1]
            
            if (current_node.x, current_node.y) == goal:
                path = self.reconstruct_path(current_node)
                return path
            
            closed_set.add((current_node.x, current_node.y))
            
            for direction in [constants.LEFT, constants.UP, constants.RIGHT, constants.DOWN]:
                neighbor_x = current_node.x + self.DIRECTION_TO_DX[direction]
                neighbor_y = current_node.y + self.DIRECTION_TO_DY[direction]
                
                if (neighbor_x, neighbor_y) in closed_set:
                    continue
                
                if not self.is_valid_move(current_percept, current_node.x, current_node.y, direction):
                    continue
                
                frequency = self.get_door_frequency(current_node.x, current_node.y, direction)
                
                g_cost = current_node.g_cost + frequency  # Use frequency as cost -> Lower frequency means better
                h_cost = self.manhattan_distance((neighbor_x, neighbor_y), goal)
                
                neighbor = FrequencyAwareNode(neighbor_x, neighbor_y, g_cost, h_cost, current_node)
                neighbor.door_frequencies[direction] = frequency
                
                if not any(node for _, node in open_list if node.x == neighbor.x and node.y == neighbor.y):
                    heapq.heappush(open_list, (neighbor.f_cost, neighbor))
                else:
                    existing_node = next(node for _, node in open_list if node.x == neighbor.x and node.y == neighbor.y)
                    if neighbor.g_cost < existing_node.g_cost:
                        open_list.remove((existing_node.f_cost, existing_node))
                        heapq.heappush(open_list, (neighbor.f_cost, neighbor))
        return None  # No path found

    def manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reconstruct_path(self, node):
        path = []
        while node.parent:
            dx = node.x - node.parent.x
            dy = node.y - node.parent.y
            path.append(self.get_direction(dx, dy))
            node = node.parent
        return path[::-1]

    def get_direction(self, dx, dy):
        if dx == -1:
            return constants.LEFT
        elif dx == 1:
            return constants.RIGHT
        elif dy == -1:
            return constants.UP
        elif dy == 1:
            return constants.DOWN

    DIRECTION_TO_DX = {constants.LEFT: -1, constants.RIGHT: 1, constants.UP: 0, constants.DOWN: 0}
    DIRECTION_TO_DY = {constants.LEFT: 0, constants.RIGHT: 0, constants.UP: -1, constants.DOWN: 1}

    def is_valid_move(self, current_percept, x, y, direction):
        for state_x, state_y, state_dir, state in current_percept.maze_state:
            if state_x == x and state_y == y and state_dir == direction:
                if state == constants.OPEN:
                    frequency = self.get_door_frequency(x, y, direction)
                    is_valid = 0 < frequency <= self.maximum_door_frequency
                    return is_valid
        return False


class FrequencyAwareNode:
    def __init__(self, x, y, g_cost, h_cost, parent=None):
        self.x = x
        self.y = y
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent
        self.door_frequencies = {constants.LEFT: 0, constants.UP: 0, constants.RIGHT: 0, constants.DOWN: 0}

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))
    

