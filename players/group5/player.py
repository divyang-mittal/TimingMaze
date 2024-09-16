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

    def simple_search(self):        
        nw, sw, ne, se = 0, 0, 0, 0

        for i in range(self.radius):
            for j in range(self.radius):
                if self.player_map.get_seen_counts([[-i, -j]])[0]>0:
                    nw += 1
                if self.player_map.get_seen_counts([[i, -j]])[0]>0:
                    sw += 1
                if self.player_map.get_seen_counts([[-i, j]])[0]>0:
                    ne += 1
                if self.player_map.get_seen_counts([[i, j]])[0]>0:
                    se += 1
        best_diagonal = max(nw, sw, ne, se)
        if best_diagonal == nw:
            if ne > sw:
                return constants.UP
            return constants.LEFT
        elif best_diagonal == sw:
            if se > nw:
                return constants.DOWN
            return constants.LEFT
        elif best_diagonal == ne:
            if nw > se:
                return constants.UP
            return constants.RIGHT
        else:
            if sw > ne:
                return constants.DOWN
            return constants.RIGHT

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

        exists, end_pos = self.player_map.get_end_pos_if_known()
        if not exists:
            return self.simple_search()
        return converge(self.player_map.get_cur_pos(), end_pos)


