import os
import pickle
import numpy as np
import logging

import constants
import heapq
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
        self.pos = [0, 0] # X, Y
        self.best_path_found = {}

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
        direction = [0, 0, 0, 0]
        for maze_state in current_percept.maze_state:
            if maze_state[0] == 0 and maze_state[1] == 0:
                direction[maze_state[2]] = maze_state[3]

        if current_percept.is_end_visible:
            if abs(current_percept.end_x) >= abs(current_percept.end_y):
                if current_percept.end_x > 0 and direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                                and maze_state[3] == constants.OPEN):
                            return constants.RIGHT
                if current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                                and maze_state[3] == constants.OPEN):
                            return constants.LEFT
                if current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                                and maze_state[3] == constants.OPEN):
                            return constants.UP
                if current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                                and maze_state[3] == constants.OPEN):
                            return constants.DOWN
                return constants.WAIT
            else:
                if current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                                and maze_state[3] == constants.OPEN):
                            return constants.UP
                if current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                                and maze_state[3] == constants.OPEN):
                            return constants.DOWN
                if current_percept.end_x > 0 and direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                                and maze_state[3] == constants.OPEN):
                            return constants.RIGHT
                if current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                                and maze_state[3] == constants.OPEN):
                            return constants.LEFT
                return constants.WAIT
        else:
            if direction[constants.LEFT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                            and maze_state[3] == constants.OPEN):
                        return constants.LEFT
            if direction[constants.DOWN] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                            and maze_state[3] == constants.OPEN):
                        return constants.DOWN
            if direction[constants.RIGHT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                            and maze_state[3] == constants.OPEN):
                        return constants.RIGHT
            if direction[constants.UP] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                            and maze_state[3] == constants.OPEN):
                        return constants.UP
            return constants.WAIT
    
    def move_toward_visible_end(self, door_info):
         """
            Args:
                door_info
            Return: True if the optimal path was found, false otherwise
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

    def can_move_in_direction(self, direction):
        # TODO: this depends on library
        return True


class Door():
    def __init__(x, y, direction, frequency):
        self.x = x 
        self.y = y 
        self.direction = direction
        self.frequency = frequency 
    

class Graph():
    def __init__(self, door_info):
        # a cell at coordinate (x, y) can be found with self.V[(x,y)]
        self.V = {}
        for door in door_info:
            coordinates = (door.x, door.y)
            if coordinates not in self.V:
                self.V[coordinates] = new Cell(door.x, door.y)
            
            self.V[coordinates].door_freqs[door.direction] = door.frequency

    def find_path(self, goal_x, goal_y):
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

def opposite(direction) -> int:
    if direction == constants.UP:
        return constants.DOWN
    else if direction == constants.DOWN:
        return constants.UP

    else if direction == constants.LEFT:
        return constants.RIGHT
    return constants.LEFT