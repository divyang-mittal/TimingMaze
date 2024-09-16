import os
import pickle
import numpy as np
import logging

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

        self.cur_pos = [0, 0] # X, Y
        self.positions = {}
        self.step = 0

    def GCD(self, a, b):
        if b == 0:
            return a
        return self.GCD(b, a % b)

    def update_position(self, current_percept) -> None:
        """Function which updates the position of the amoeba

            Args:
                current_percept(TimingMazeState): contains current state information
        """
        if self.step != 0:
            for maze_state in current_percept.maze_state:
                relative_x = int(maze_state[0]) + self.cur_pos[0]
                relative_y = int(maze_state[1]) + self.cur_pos[1]
                if (relative_x, relative_y) not in self.positions:
                    self.positions[(relative_x, relative_y)] = [0, 0, 0, 0]
                
                if self.positions[(relative_x, relative_y)][maze_state[2]] == 0 and maze_state[3] != 0:
                    self.positions[(relative_x, relative_y)][maze_state[2]] = self.step
                elif maze_state[3] != 0:
                    self.positions[(relative_x, relative_y)][maze_state[2]] = self.GCD(self.positions[(relative_x, relative_y)][maze_state[2]], self.step)

        print(self.positions)

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

        print(self.cur_pos, self.step)

        self.update_position(current_percept)

        self.step += 1


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
                            self.cur_pos[0] += 1
                            return constants.RIGHT
                if current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                                and maze_state[3] == constants.OPEN):
                            self.cur_pos[0] -= 1
                            return constants.LEFT
                if current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                                and maze_state[3] == constants.OPEN):
                            self.cur_pos[1] -= 1
                            return constants.UP
                if current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                                and maze_state[3] == constants.OPEN):
                            self.cur_pos[1] += 1
                            return constants.DOWN
                return constants.WAIT
            else:
                if current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                                and maze_state[3] == constants.OPEN):
                            self.cur_pos[1] -= 1
                            return constants.UP
                if current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                                and maze_state[3] == constants.OPEN):
                            self.cur_pos[1] += 1
                            return constants.DOWN
                if current_percept.end_x > 0 and direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                                and maze_state[3] == constants.OPEN):
                            self.cur_pos[0] += 1
                            return constants.RIGHT
                if current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                                and maze_state[3] == constants.OPEN):
                            self.cur_pos[0] -= 1
                            return constants.LEFT
                return constants.WAIT
        else:
            if direction[constants.LEFT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                            and maze_state[3] == constants.OPEN):
                        self.cur_pos[0] -= 1
                        return constants.LEFT
            if direction[constants.DOWN] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                            and maze_state[3] == constants.OPEN):
                        self.cur_pos[1] += 1
                        return constants.DOWN
            if direction[constants.RIGHT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                            and maze_state[3] == constants.OPEN):
                        self.cur_pos[0] += 1
                        return constants.RIGHT
            if direction[constants.UP] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                            and maze_state[3] == constants.OPEN):
                        self.cur_pos[1] -= 1
                        return constants.UP
            return constants.WAIT
