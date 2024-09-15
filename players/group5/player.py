import os
import pickle
import numpy as np
import logging

import constants
from players.group5.player_map import PlayerMapInterface, SimplePlayerMap
from timing_maze_state import TimingMazeState


class G5_Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger, precomp_dir: str, maximum_door_frequency: int, radius: int) -> None:
        """Initialise the player with the basic amoeba information

            Args:
                rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
                logger (logging.Logger): logger use this like logger.info("message")
                maximum_door_frequency (int): the maximum frequency of doors
                radius (int): the radius of the drone
                precomp_dir (str): Directory path to store/load pre-computation
        """
        self.rng = rng
        self.logger = logger
        self.maximum_door_frequency = maximum_door_frequency
        self.radius = radius
        self.player_map: PlayerMapInterface = SimplePlayerMap(maximum_door_frequency, logger)
        self.turns = 0

    def move(self, current_percept: TimingMazeState) -> int:
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
        self.turns += 1
        self.player_map.update_map(self.turns, current_percept.maze_state)
        return constants.UP

        # direction = [0, 0, 0, 0]
        # for maze_state in current_percept.maze_state:
        #     if maze_state[0] == 0 and maze_state[1] == 0:
        #         direction[maze_state[2]] = maze_state[3]

        # if current_percept.is_end_visible:
        #     if abs(current_percept.end_x) >= abs(current_percept.end_y):
        #         if current_percept.end_x > 0 and direction[constants.RIGHT] == constants.OPEN:
        #             for maze_state in current_percept.maze_state:
        #                 if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
        #                         and maze_state[3] == constants.OPEN):
        #                     return constants.RIGHT
        #         if current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
        #             for maze_state in current_percept.maze_state:
        #                 if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
        #                         and maze_state[3] == constants.OPEN):
        #                     return constants.LEFT
        #         if current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
        #             for maze_state in current_percept.maze_state:
        #                 if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
        #                         and maze_state[3] == constants.OPEN):
        #                     return constants.UP
        #         if current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
        #             for maze_state in current_percept.maze_state:
        #                 if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
        #                         and maze_state[3] == constants.OPEN):
        #                     return constants.DOWN
        #         return constants.WAIT
        #     else:
        #         if current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
        #             for maze_state in current_percept.maze_state:
        #                 if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
        #                         and maze_state[3] == constants.OPEN):
        #                     return constants.UP
        #         if current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
        #             for maze_state in current_percept.maze_state:
        #                 if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
        #                         and maze_state[3] == constants.OPEN):
        #                     return constants.DOWN
        #         if current_percept.end_x > 0 and direction[constants.RIGHT] == constants.OPEN:
        #             for maze_state in current_percept.maze_state:
        #                 if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
        #                         and maze_state[3] == constants.OPEN):
        #                     return constants.RIGHT
        #         if current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
        #             for maze_state in current_percept.maze_state:
        #                 if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
        #                         and maze_state[3] == constants.OPEN):
        #                     return constants.LEFT
        #         return constants.WAIT
        # else:
        #     if direction[constants.LEFT] == constants.OPEN:
        #         for maze_state in current_percept.maze_state:
        #             if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
        #                     and maze_state[3] == constants.OPEN):
        #                 return constants.LEFT
        #     if direction[constants.DOWN] == constants.OPEN:
        #         for maze_state in current_percept.maze_state:
        #             if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
        #                     and maze_state[3] == constants.OPEN):
        #                 return constants.DOWN
        #     if direction[constants.RIGHT] == constants.OPEN:
        #         for maze_state in current_percept.maze_state:
        #             if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
        #                     and maze_state[3] == constants.OPEN):
        #                 return constants.RIGHT
        #     if direction[constants.UP] == constants.OPEN:
        #         for maze_state in current_percept.maze_state:
        #             if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
        #                     and maze_state[3] == constants.OPEN):
        #                 return constants.UP
        #     return constants.WAIT
