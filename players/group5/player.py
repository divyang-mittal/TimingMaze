import os
import numpy as np
import logging

import constants
from players.group5.door import DoorIdentifier
from players.group5.player_map import PlayerMapInterface, SimplePlayerCentricMap, StartPosCentricPlayerMap
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


    def simple_search(self):
        valid_moves = self.player_map.get_valid_moves(self.turns)
        
        # can't move anywhere
        if not valid_moves:
            return constants.WAIT

        cur_pos = self.player_map.get_cur_pos()
        cur_pos_i, cur_pos_j = cur_pos[0], cur_pos[1]
        
        # number of times cells were unseen in 2r by 2r squares in each direction OUTSIDE the radius
        # indices match direction (LEFT, UP, RIGHT, DOWN)
        # curiosity = [
        #     (2 * self.radius) ** 2 - sum(self.player_map.get_seen_counts([[cur_pos_i - self.radius * 3 + i, cur_pos_j - self.radius + j] for j in range(self.radius * 2) for i in range(self.radius * 2)])),
        #     (2 * self.radius) ** 2 - sum(self.player_map.get_seen_counts([[cur_pos_i - self.radius + i, cur_pos_j - self.radius * 3 + j] for j in range(self.radius * 2) for i in range(self.radius * 2)])),
        #     (2 * self.radius) ** 2 - sum(self.player_map.get_seen_counts([[cur_pos_i + self.radius + i + 1, cur_pos_j - self.radius + j] for j in range(self.radius * 2) for i in range(self.radius * 2)])),
        #     (2 * self.radius) ** 2 - sum(self.player_map.get_seen_counts([[cur_pos_i - self.radius + i, cur_pos_j + self.radius + j + 1] for j in range(self.radius * 2) for i in range(self.radius * 2)]))
        # ]

        # print('Curiosity:', curiosity)

        
        # create a list of indices of the maximum curiosity values
        # max_curiosity = max(curiosity)
        # max_indices = [i for i, j in enumerate(curiosity) if j == max_curiosity]

        near_discovered_counts = [
            self.player_map.get_unseen_counts([[cur_pos_i - self.radius * 3 + i, cur_pos_j - self.radius + j] for j in range(self.radius * 2) for i in range(self.radius * 2)]),
            self.player_map.get_unseen_counts([[cur_pos_i - self.radius + i, cur_pos_j - self.radius * 3 + j] for j in range(self.radius * 2) for i in range(self.radius * 2)]),
            self.player_map.get_unseen_counts([[cur_pos_i + self.radius + i + 1, cur_pos_j - self.radius + j] for j in range(self.radius * 2) for i in range(self.radius * 2)]),
            self.player_map.get_unseen_counts([[cur_pos_i - self.radius + i, cur_pos_j + self.radius + j + 1] for j in range(self.radius * 2) for i in range(self.radius * 2)])
        ]
        
        far_discovered_counts = [
            self.player_map.get_unseen_counts([[cur_pos_i - 200 + i, cur_pos_j - 100 + j] for j in range(200) for i in range(200)]),
            self.player_map.get_unseen_counts([[cur_pos_i - 100 + i, cur_pos_j - 200 + j] for j in range(200) for i in range(200)]),
            self.player_map.get_unseen_counts([[cur_pos_i + i + 1, cur_pos_j - 100 + j] for j in range(200) for i in range(200)]),
            self.player_map.get_unseen_counts([[cur_pos_i - 100 + i, cur_pos_j + j + 1] for j in range(200) for i in range(200)])
        ]

        weighted_counts = [near_discovered_counts[i] / self.radius**2 + far_discovered_counts[i] / 100**2 for i in range(4)]

        print('Near undiscovered counts:', near_discovered_counts)
        print('Far undiscovered counts:', far_discovered_counts)
        print('Weighted counts:', weighted_counts)
        
        best_direction = max(weighted_counts)
        best_indices = [i for i, j in enumerate(weighted_counts) if j == best_direction]

        # intersection between max_indices and valid_moves
        best_moves = list(set(best_indices) & set(valid_moves))

        # making the opposite move as the last one
        opposite_move = (self.last_move + 2) % 4
        if self.last_move != constants.WAIT and opposite_move in best_moves:
            best_moves.remove(opposite_move)

        # TODO REMOVE THIS FEATURE ONCE VALID MOVES BUG IS FIXED
        if self.last_pos == cur_pos:
            self.stuck_counter += 1

        if self.last_move in best_moves and self.stuck_counter >= self.maximum_door_frequency * (self.maximum_door_frequency - 1):
            best_moves.remove(self.last_move)
            self.stuck_counter = 0

        move = self.rng.choice(best_moves) if best_moves else self.rng.choice(valid_moves)
        
        return (int)(move)

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

            return move if move in valid_moves else constants.WAIT
        except Exception as e:
            self.logger.debug(e, e.with_traceback)
            return constants.WAIT

    def simple_search(self):
        return simple_search(self.player_map, self.radius)
