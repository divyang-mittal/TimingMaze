from abc import ABC, abstractmethod
from collections import defaultdict
import logging
import math
import os
from typing import List, Optional, Set, Tuple

import constants
from players.group5.door import DoorIdentifier, get_updated_frequency_candidates
from players.group5.util import setup_file_logger
from timing_maze_state import TimingMazeState


class PlayerMapInterface(ABC):
    @abstractmethod
    def get_start_pos(self) -> List[int]:  # NOTE: output type may become dataclass in the future (with type: relative/absolute)
        """Function which returns the start position of the game

            Returns:
                List[int]: List containing the x and y coordinates of the start position (relative to current position)
        """
        pass

    @abstractmethod
    def get_end_pos_if_known(self) -> Tuple[bool, Optional[List[int]]]:
        """Function which returns the end position of the game if it is known

            Returns:
                bool: Boolean indicating if the end position is known
                Optional[List[int]]: Optional list containing the x and y coordinates of the end position (relative to current position)
        """
        pass

    @abstractmethod
    def get_cur_pos(self) -> List[int]:
        """Function which returns the current position of the player

            Returns:
                List[int]: List containing the x and y coordinates of the current position (relative to current position)
        """
        pass

    @abstractmethod
    def get_freq_candidates(self, door_id: DoorIdentifier) -> Set[int]:
        """Function which returns the frequency candidates for the given door

            Args:
                door_id (DoorIdentifier): DoorIdentifier object containing the relative coordinates and door type
            Returns:
                List[int]: List containing the door frequency candidates for the given cell and direction
        """
        pass

    @abstractmethod
    def get_wall_freq_candidates(self, door_id: DoorIdentifier) -> List[Set[int]]:
        """Function which returns the frequency candidates for the given door and its touching door (i.e., collectively called a wall)

            Args:
                door_id (DoorIdentifier): DoorIdentifier object containing the relative coordinates and door type
            Returns:
                List[int]: List containing the door frequency candidates for the given door and its touching door (2nd element is the touching door)
        """
        pass

    @abstractmethod
    def update_map(self, turn_num: int, percept: TimingMazeState):  # TODO: check type of maze_state
        """Function which updates the map with the given maze state

            Args:
                turn_num (int): Integer representing the current turn number
                percept (TimingMazeState): TimingMazeState object describing the current state of the visible maze
        """
        pass

    @abstractmethod
    def get_valid_moves(self) -> List[int]:
        """Function which returns the valid moves for the player

            Returns:
                List[int]: List containing the valid moves for the player
        """
        pass

    @abstractmethod
    def get_boundaries(self) -> List[int]:
        """Function which returns the boundaries of the map

            Returns:
                List[int]: List containing the boundaries of the map (left, up, right, down)
        """
        pass

    @abstractmethod
    def set_boundaries(self, boundaries: List[int]):
        """Function which sets the boundaries of the map

            Args:
                boundaries (List[int]): List containing the boundaries of the map (left, up, right, down)
        """
        pass

# TODO: check this
def default_freq_candidates(max_door_frequency: int):
    # generate a set of door frequencies from 0 to max_door_frequency
    return lambda: set(range(max_door_frequency+1))


"""
StartPosCentricPlayerMap is similar to SimplePlayerCentricMap but is not centric on the user's current position but rather the first start position which we assign to a constant coordinate (i.e., it receives/outputs start position-centric coordinate data). 
In the SimplePlayerCentricMap, the player was always at (0,0). This is not the case in the StartPosCentricPlayerMap.
"""
class StartPosCentricPlayerMap(PlayerMapInterface):
    def __init__(self, max_door_frequency: int, logger: logging.Logger, map_dim: int = constants.map_dim):
        self._setup_logger(logger)
        
        # General map properties (constant after initialization)
        self._ACTUAL_MAP_DIM = map_dim
        self._GLOBAL_MAP_LEN = 2 * (map_dim-1) + 1
        self._START_POS = [map_dim-1, map_dim-1]

        self._end_pos = None
        self._boundaries = [-1, -1, self._GLOBAL_MAP_LEN, self._GLOBAL_MAP_LEN]  # will be set to last cell whose outer wall is a boundary
        self._prev_start_ref = [0,0]  # TODO: rename. currently functional.

        self._door_freqs = defaultdict(default_freq_candidates(max_door_frequency))
        self._door_status = defaultdict(int)
        
        self.turn_num = 0
        self.cur_pos = self._START_POS  # (x, y) start pos centric

    def _setup_logger(self, logger):
        logger = setup_file_logger(logger, self.__class__.__name__, "./log")
        self.logger = logger

    def _get_map_coordinates(self, player_centric_coordinates: List[int]) -> List[int]:
        return [
            self.cur_pos[0] + player_centric_coordinates[0],
            self.cur_pos[1] + player_centric_coordinates[1],
        ]

    def get_start_pos(self) -> List[int]:
        return self._START_POS

    def get_end_pos_if_known(self) -> Tuple[bool, Optional[List[int]]]:
        if self._end_pos is None:
            return False, None
        return True, self._end_pos
    
    def set_end_pos(self, relative_coord: List[int]):
        self._end_pos = self._get_map_coordinates(relative_coord)

    def get_cur_pos(self) -> List[int]:
        return self.cur_pos
    
    def get_boundaries(self) -> List[int]:
        return self._boundaries
    
    def set_boundaries(self, boundaries: List[int]):
        self._boundaries = boundaries
    
    def _door_dictkey(self, map_coords, door_type) -> List[int]:
        return f"({map_coords[0]},{map_coords[1]})_{door_type}"

    def _get_freq_candidates_usecase(self, coord, door_type) -> Set[int]:
        if self._is_out_of_bound(coord):
            return {0}

        key = self._door_dictkey(map_coords=coord, door_type=door_type)
        return self._door_freqs[key]

    def get_freq_candidates(self, door_id: DoorIdentifier) -> Set[int]:
        return self._get_freq_candidates_usecase(door_id.absolute_coord, door_id.door_type)
    
    def _set_freq_candidates_usecase(self, coord, door_type, freq_candidates: Set[int]):
        key = self._door_dictkey(map_coords=coord, door_type=door_type)
        self._door_freqs[key] = freq_candidates
    
    def _is_boundary_found(self, dir: int) -> bool:
        BOUNDARY_NOT_FOUND_VALUES = {-1, self._GLOBAL_MAP_LEN}
        return self._boundaries[dir] not in BOUNDARY_NOT_FOUND_VALUES

    def _update_boundaries(self, door_type: int, coord: List[int]):
        map_e2e_dist = self._ACTUAL_MAP_DIM - 1
        
        if door_type == constants.LEFT:
            self._boundaries[constants.LEFT] = coord[0]
            self._boundaries[constants.RIGHT] = coord[0] + map_e2e_dist
        elif door_type == constants.RIGHT:
            self._boundaries[constants.RIGHT] = coord[0]
            self._boundaries[constants.LEFT] = coord[0] - map_e2e_dist
        elif door_type == constants.UP:
            self._boundaries[constants.UP] = coord[1]
            self._boundaries[constants.DOWN] = coord[1] + map_e2e_dist
        elif door_type == constants.DOWN:
            self._boundaries[constants.DOWN] = coord[1]
            self._boundaries[constants.UP] = coord[1] - map_e2e_dist

    def _update_cur_pos(self, new_start_ref: List[int]):
        # start_ref is the user-centric coordinate of the start position provided by the current_percept at each turn TODO: rename
        self.cur_pos[0] -= (new_start_ref[0] - self._prev_start_ref[0])
        self.cur_pos[1] -= (new_start_ref[1] - self._prev_start_ref[1])

        self.logger.debug(f"Updating cur pos: {self.cur_pos}")
        self._prev_start_ref = new_start_ref

    def update_door_status(self, coord: List[int], door_type: int, door_state: int):
        key = self._door_dictkey(map_coords=coord, door_type=door_type)

        self.logger.debug(f"Updating door status for {key} to {door_state}") if coord[0] == 90 and coord[1] == 99 else None
        self._door_status[key] = door_state

    def update_map(self, turn_num: int, percept: TimingMazeState):
        self.turn_num = turn_num
        self._update_cur_pos([percept.start_x, percept.start_y])

        self.logger.debug(f"!!!!Updating map for turn {turn_num}")
        for door in percept.maze_state:
            player_relative_coordinates, door_type, door_state = door[:2], door[2], door[3]
            self.logger.debug(f">percept_coords: {player_relative_coordinates}, {door_type}, {door_state} ; cur_pos {self.cur_pos}") if door_state == constants.CLOSED else None

            self.logger.debug(f"Door seen in percept: {player_relative_coordinates}, {door_type}, {door_state}") if player_relative_coordinates[0] == 90 and player_relative_coordinates[1] == 99 else None

            coord = self._get_map_coordinates(player_relative_coordinates)

            # update boundaries if newly found
            self.logger.debug(f"boundary found!!!") if door_state == constants.BOUNDARY and door_type == constants.UP else None
            if door_state == constants.BOUNDARY and not self._is_boundary_found(door_type):
                self._update_boundaries(door_type, coord)
                self.logger.debug(f"Boundaries updated: {self._boundaries}")

            # update frequencies (TODO: refactor for readability)
            cur_freq_candidates = self._get_freq_candidates_usecase(coord, door_type)  # TODO: consider refactoring how doorID is used
            new_freq_candidates = get_updated_frequency_candidates(cur_freq_candidates, turn_num=turn_num, door_state=door_state)
            self._set_freq_candidates_usecase(coord, door_type, new_freq_candidates)

            self.update_door_status(coord, door_type, door_state)

        if percept.is_end_visible and self._end_pos is None:
            self.set_end_pos([percept.end_x, percept.end_y])

    OUT_OF_BOUND_SEEN_COUNT = 1000

    def _is_out_of_bound(self, coord: List[int]) -> bool:
        return any([
            coord[0] < self._boundaries[constants.LEFT], 
            coord[0] > self._boundaries[constants.RIGHT],
            coord[1] < self._boundaries[constants.UP],
            coord[1] > self._boundaries[constants.DOWN],
        ])
    
    def get_valid_moves(self, turn_num: int) -> List[int]:
        if turn_num != self.turn_num:
            raise ValueError("Turn number does not match map's current turn number")

        self.logger.debug(f"Getting valid moves for turn {turn_num}")
        cur_pos = self.cur_pos
        valid_moves_dependent_doors = {
            constants.LEFT: [
                self._door_dictkey(cur_pos, constants.LEFT), 
                self._door_dictkey([cur_pos[0]-1, cur_pos[1]], constants.RIGHT),
            ],
            constants.UP: [
                self._door_dictkey(cur_pos, constants.UP),
                self._door_dictkey([cur_pos[0], cur_pos[1]-1], constants.DOWN),
            ],
            constants.RIGHT: [
                self._door_dictkey(cur_pos, constants.RIGHT),
                self._door_dictkey([cur_pos[0]+1, cur_pos[1]], constants.LEFT),
            ],
            constants.DOWN: [
                self._door_dictkey(cur_pos, constants.DOWN),
                self._door_dictkey([cur_pos[0], cur_pos[1]+1], constants.UP),
            ],
        }

        valid_moves = []
        for move, door_keys in valid_moves_dependent_doors.items():
            self.logger.debug(f"Door {door_keys[0]} status: {self._door_status[door_keys[0]]}, {door_keys[1]} {self._door_status[door_keys[1]]}")

            if all([self._door_status[key] == constants.OPEN for key in door_keys]):
                valid_moves.append(move)

        return valid_moves

    def get_wall_freq_candidates(self, door_id: DoorIdentifier) -> List[Set[int]]:
        door_freq_candidates = self._get_freq_candidates_usecase(door_id.absolute_coord, door_id.door_type)

        # print("Inner Door freq candidates: ", door_freq_candidates)
        # print("Door ID: ", door_id)
        
        door_type_to_touching_door_offsets = {
            constants.LEFT: ([-1, 0], constants.RIGHT),
            constants.UP: ([0, -1], constants.DOWN),
            constants.RIGHT: ([1, 0], constants.LEFT),
            constants.DOWN: ([0, 1], constants.UP),
        }
        touching_door_offset, touching_door_type = door_type_to_touching_door_offsets[door_id.door_type]
        touching_door_coord = [door_id.absolute_coord[0] + touching_door_offset[0], door_id.absolute_coord[1] + touching_door_offset[1]]

        # print("Touching Door coord: ", touching_door_coord)
        
        touching_door_freq_candidates = self._get_freq_candidates_usecase(touching_door_coord, touching_door_type)

        # print("Outer Door freq candidates: ", touching_door_freq_candidates)

        wall_freq_candidates = [door_freq_candidates, touching_door_freq_candidates]

        # find LCM of two lists
        def lcm(a, b):
            if a==0 or b==0:
                return 0
            return abs(a * b) // math.gcd(a, b)

        return [lcm(f1, f2) for f1 in door_freq_candidates for f2 in touching_door_freq_candidates]

        # return [door_freq_candidates, touching_door_freq_candidates]

    def get_freq_candidates(self, door_id: DoorIdentifier) -> Set[int]:
        return self._get_freq_candidates_usecase(door_id.absolute_coord, door_id.door_type)
