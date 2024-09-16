import os
import pickle
import numpy as np
import logging

import constants
from players.group5.player_map import PlayerMapInterface, SimplePlayerMap
from timing_maze_state import TimingMazeState
from players.group5.converge import converge_basic, converge


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
        self.mode = 0

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
        self.player_map.update_map(self.turns, current_percept)

        if self.mode == 0:
            move = self.explore()
            if move[1] == 1:
                self.mode = 1
            return move[0]
        

        
        # print("Treasure: ", self.player_map.get_end_pos_if_known(self.player_map))

        # print("Treasure: ", self.player_map.get_end_pos_if_known(self.player_map)[1])

        exists, end_pos = self.player_map.get_end_pos_if_known()
        if exists:
            return converge(self.player_map.get_cur_pos(), end_pos)
        else:
            # randomly return a number between 0 and 3
            num = self.rng.integers(0, 4)
            print("Random Move: ", num)
            return num

                # return self.explore()
        # exists, end_pos = self.player_map.get_end_pos_if_known()
        # if exists:
        #     return converge(self.player_map.get_cur_pos(), end_pos)


        # return converge(self.player_map.get_cur_pos(), self.player_map.get_end_pos())
        # return converge(self.player_map.get_cur_pos(), self.player_map.get_end_pos_if_known()[1])

