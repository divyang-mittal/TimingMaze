import os
import random
from typing import List
import numpy as np
import logging

import constants
from players.group5.door import DoorIdentifier
from players.group5.player_map import PlayerMapInterface, StartPosCentricPlayerMap
from timing_maze_state import TimingMazeState
from players.group5.converge import converge
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
        self.mode = 0

        self.last_move = constants.WAIT
        
        # TODO REMOVE THIS FEATURE ONCE VALID MOVES BUG IS FIXED
        self.last_pos = self.player_map.get_cur_pos()
        self.stuck_counter = 0

    def _setup_logger(self, logger):
        self.logger = logger
        self.logger.setLevel(logging.DEBUG)
        self.log_dir = "./log"
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.log_dir, 'Player 5.log'), mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(fh)

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
            cur_pos = self.player_map.get_cur_pos()
            
            valid_moves = self.player_map.get_valid_moves(self.turns)
            self.logger.debug(f"Valid moves: {valid_moves}")

            # example_freq_set = self.player_map.get_wall_freq_candidates(door_id=DoorIdentifier(absolute_coord=cur_pos, door_type=0))
            # self.logger.debug(f"Example freq set for coordinate {cur_pos}: {example_freq_set}")

            exists, end_pos = self.player_map.get_end_pos_if_known()
            if not exists:
                move = self.simple_search()
                return move if move in valid_moves else constants.WAIT  # TODO: this is if-statement is to demonstrate valid_moves is correct (@eylam, replace with actual logic)
            move = converge(self.player_map.get_cur_pos(), [end_pos], self.turns, self.player_map, self.maximum_door_frequency)
            return move
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
        pass


class G5_Player_Refactored:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger, precomp_dir: str, maximum_door_frequency: int, radius: int, boundaries: List[int]) -> None:
        self.player_state = PlayerState.SEARCH
        self.corridors = []
        self.global_boundaries = boundaries

        pass

    def move(self, current_percept: TimingMazeState) -> int:
        pass
