from typing import List
import numpy as np
import logging

import constants
from players.group5.player_map import PlayerMapInterface, StartPosCentricPlayerMap
from players.group5.util import setup_file_logger
from timing_maze_state import TimingMazeState
from players.group5.converge import ConvergeStrategy
from players.group5.simple_search import simple_search


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
        self._setup_logger(logger)
        self.rng = rng
        self.maximum_door_frequency = maximum_door_frequency
        self.radius = radius
        self.player_map: PlayerMapInterface = StartPosCentricPlayerMap(maximum_door_frequency, logger)
        self.turns = 0

        self.search_strategy = None
        
    def _setup_logger(self, logger):
        logger = setup_file_logger(logger, self.__class__.__name__, "./log")
        self.logger = logger

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
        try:
            self.turns += 1
            self.player_map.update_map(self.turns, current_percept)
            
            valid_moves = self.player_map.get_valid_moves(self.turns)
            self.logger.debug(f"Valid moves: {valid_moves}")

            exists, end_pos = self.player_map.get_end_pos_if_known()
            if not exists:
                if self.search_strategy is None:
                    self.search_strategy = SearchStrategy(self.player_map, self.turns)
                return self.search_strategy.move(current_percept)
            return ConvergeStrategy(self.player_map.get_cur_pos(), [end_pos], self.turns, self.player_map, self.maximum_door_frequency).move()
        except Exception as e:
            self.logger.debug(e, e.with_traceback)
            return constants.WAIT

    def simple_search(self):
        return simple_search(self.player_map, self.radius)


# ENUM representing player state
class PlayerState:
    SEARCH = 0
    CONVERGE = 1


# SEARCH SPECIFIC FIELDS
class SearchStrategy:
    def __init__(self, player_map: PlayerMapInterface, turns: int) -> None:
        self.player_map = player_map
        self.start_turn = turns
        self.corridors = []
        # self.current_corridor = None TODO: might not need if corridor[0] is the current corridor

    def move(self, current_percept: TimingMazeState) -> int:
        return 1

class G5_Player_Refactored:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger, precomp_dir: str, maximum_door_frequency: int, radius: int, boundaries: List[int]) -> None:
        self.player_state = PlayerState.SEARCH
        self.corridors = []
        self.global_boundaries = boundaries

        pass

    def move(self, current_percept: TimingMazeState) -> int:
        pass
