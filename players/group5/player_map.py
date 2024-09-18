from abc import ABC, abstractmethod
from collections import defaultdict
import logging
import math
import os
from typing import List, Optional, Set, Tuple

import constants
from players.group5.door import DoorIdentifier, get_updated_frequency_candidates
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
    def get_seen_counts(self, cell_coords: List[List[int]]) -> List[int]:
        """Function which returns the number of times each cell has been seen

            Args:
                cell_coords (List[List[int]]): List of cell coordinates
            Returns:
                List[int]: List containing the number of times each cell has been seen
        """
        pass

    @abstractmethod
    def get_valid_moves(self) -> List[int]:
        """Function which returns the valid moves for the player

            Returns:
                List[int]: List containing the valid moves for the player
        """
        pass


def default_freq_candidates(max_door_frequency: int):
    # generate a set of door frequencies from 0 to max_door_frequency
    return lambda: set(range(max_door_frequency+1))

# SimplePlayerCentricMap is a simple implementation of PlayerMapInterface that is user-centric (receives/outputs user-centric coordinate data)
class SimplePlayerCentricMap(PlayerMapInterface):
    def __init__(self, max_door_frequency: int, logger: logging.Logger, map_dim: int = constants.map_dim):
        # TODO: clean up later
        self.logger = logger

        self.logger.setLevel(logging.DEBUG)
        self.log_dir = "./log"
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.log_dir, 'Group 5.log'), mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(fh)

        self._real_map_dim = map_dim
        self._map_len = 2 * (map_dim-1) + 1
        self._start_pos = [map_dim-1, map_dim-1]
        self._end_pos = None
        self._boundaries = [-1, -1, self._map_len, self._map_len]  # LEFT, UP, RIGHT, DOWN (see constants.py)

        self.cur_pos = self._start_pos
        self._door_freqs = defaultdict(default_freq_candidates(max_door_frequency))
        self._cell_seen_count = defaultdict(int)


    def _get_player_relative_coordinates(self, map_coordinates: List[int]) -> List[int]:
        return [map_coordinates[0] - self.cur_pos[0], map_coordinates[1] - self.cur_pos[1]]
    
    def _get_map_coordinates(self, player_relative_coordinates: List[int]) -> List[int]:
        return [player_relative_coordinates[0] + self.cur_pos[0], player_relative_coordinates[1] + self.cur_pos[1]]

    def get_start_pos(self) -> List[int]:
        return self._get_player_relative_coordinates(self._start_pos)

    def get_end_pos_if_known(self) -> Tuple[bool, Optional[List[int]]]:
        if self._end_pos is None:
            return False, None
        return True, self._get_player_relative_coordinates(self._end_pos)
    
    def set_end_pos(self, relative_coord: List[int]):
        self._end_pos = self._get_map_coordinates(relative_coord)

    def get_cur_pos(self) -> List[int]:
        # NOTE: this will always return 0,0 for SimplePlayerMap where all output coordinates are relative to cur_pos
        return self._get_player_relative_coordinates(self.cur_pos)
    
    def _door_dictkey(self, map_coords, door_type) -> List[int]:
        return f"({map_coords[0]},{map_coords[1]})_{door_type}"

    def _get_freq_candidates_usecase(self, relative_coord, door_type) -> Set[int]:
        key = self._door_dictkey(
            map_coords=self._get_map_coordinates(relative_coord), 
            door_type=door_type,
        )
        return self._door_freqs[key]

    def get_freq_candidates(self, door_id: DoorIdentifier) -> Set[int]:
        return self._get_freq_candidates_usecase(door_id.relative_coord, door_id.door_type)
    
    def _set_freq_candidates_usecase(self, relative_coord, door_type, freq_candidates: Set[int]):
        key = self._door_dictkey(
            map_coords=self._get_map_coordinates(relative_coord), 
            door_type=door_type,
        )
        self._door_freqs[key] = freq_candidates
    
    def apply_move(self, move: int):
        if move == constants.LEFT:
            self.cur_pos[0] -= 1
        elif move == constants.UP:
            self.cur_pos[1] -= 1
        elif move == constants.RIGHT:
            self.cur_pos[0] += 1
        elif move == constants.DOWN:
            self.cur_pos[1] += 1

    def _is_boundary_found(self, door_type: int) -> bool:
        BOUNDARY_NOT_FOUND_VALUES = {-1, self._map_len}
        return self._boundaries[door_type] not in BOUNDARY_NOT_FOUND_VALUES

    def _update_boundaries(self, door_type: int, relative_coordinates: List[int]):
        map_e2e_dist = self._real_map_dim - 1
        door_coordinates = self._get_map_coordinates(relative_coordinates)
        
        if door_type == constants.LEFT:
            self._boundaries[constants.LEFT] = door_coordinates[0]
            self._boundaries[constants.RIGHT] = door_coordinates[0] + map_e2e_dist
        elif door_type == constants.RIGHT:
            self._boundaries[constants.RIGHT] = door_coordinates[0]
            self._boundaries[constants.LEFT] = door_coordinates[0] - map_e2e_dist
        elif door_type == constants.UP:
            self._boundaries[constants.UP] = door_coordinates[1]
            self._boundaries[constants.DOWN] = door_coordinates[1] + map_e2e_dist
        elif door_type == constants.DOWN:
            self._boundaries[constants.DOWN] = door_coordinates[1]
            self._boundaries[constants.UP] = door_coordinates[1] - map_e2e_dist

    def update_map(self, turn_num: int, percept: TimingMazeState):
        maze_state = percept.maze_state

        cells_seen = set()
        # before = self._door_freqs.copy()
        for door in maze_state:
            player_relative_coordinates, door_type, door_state = door[:2], door[2], door[3]

            if door_state == constants.BOUNDARY and not self._is_boundary_found(door_type):
                self._update_boundaries(door_type, player_relative_coordinates)

            cur_freq_candidates = self._get_freq_candidates_usecase(player_relative_coordinates, door_type)  # TODO: consider the use of DoorID as DTO or PK...
            new_freq_candidates = get_updated_frequency_candidates(cur_freq_candidates, turn_num=turn_num, door_state=door_state)

            self._set_freq_candidates_usecase(player_relative_coordinates, door_type, new_freq_candidates)

            cells_seen.add(tuple(self._get_map_coordinates(player_relative_coordinates)))

        # validation log 
        # after = self._door_freqs
        # diff = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
        # self.logger.debug(f"Diff after turn {turn_num}: {diff}")

        # update seen count
        for cell in cells_seen:
            self._cell_seen_count[cell] += 1

        if percept.is_end_visible and self._end_pos is None:
            self.set_end_pos([percept.end_x, percept.end_y])

    def get_seen_counts(self, relative_coords: List[List[int]]) -> List[int]:
        return [self._cell_seen_count.get(tuple(self._get_map_coordinates(c)), 0) if isinstance(c, list) else 0 for c in relative_coords]
    
    def get_valid_moves(self) -> List[int]:
        # Not implemented
        pass


"""
StartPosCentricPlayerMap is similar to SimplePlayerCentricMap but is not centric on the user's current position but rather the first start position which we assign to a constant coordinate (i.e., it receives/outputs start position-centric coordinate data). 
In the SimplePlayerCentricMap, the player was always at (0,0). This is not the case in the StartPosCentricPlayerMap.
"""
class StartPosCentricPlayerMap(PlayerMapInterface):
    def __init__(self, max_door_frequency: int, logger: logging.Logger, map_dim: int = constants.map_dim):
        self._setup_logger(logger)
        
        self._ACTUAL_MAP_DIM = map_dim
        self._GLOBAL_MAP_LEN = 2 * (map_dim-1) + 1
        self._START_POS = [map_dim-1, map_dim-1]

        self.turn_num = 0
        self._end_pos = None
        self._boundaries = [-1, -1, self._GLOBAL_MAP_LEN, self._GLOBAL_MAP_LEN]  # LEFT, UP, RIGHT, DOWN

        self.cur_pos = self._START_POS
        self._door_freqs = defaultdict(default_freq_candidates(max_door_frequency))
        self._cell_seen_count = defaultdict(int)
        self._door_status = defaultdict(int)

        self._prev_start_ref = [0,0]

    def _setup_logger(self, logger):
        self.logger = logger
        self.logger.setLevel(logging.DEBUG)
        self.log_dir = "./log"
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.log_dir, 'Group 5.log'), mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(fh)

    # converts player-centric coordinates (in drone view) to map-centric coordinates
    def _get_map_coordinates(self, player_centric_coordinates: List[int]) -> List[int]:
        return [
            player_centric_coordinates[0] + self.cur_pos[0],
            player_centric_coordinates[1] + self.cur_pos[1],
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
    
    def _door_dictkey(self, map_coords, door_type) -> List[int]:
        return f"({map_coords[0]},{map_coords[1]})_{door_type}"

    def _get_freq_candidates_usecase(self, coord, door_type) -> Set[int]:
        key = self._door_dictkey(
            map_coords=coord, 
            door_type=door_type,
        )
        self.logger.debug(f"Getting freq candidates for {key}, {self._door_freqs[key]}")
        return self._door_freqs[key]

    def get_freq_candidates(self, door_id: DoorIdentifier) -> Set[int]:
        return self._get_freq_candidates_usecase(door_id.absolute_coord, door_id.door_type)
    
    def _set_freq_candidates_usecase(self, coord, door_type, freq_candidates: Set[int]):
        key = self._door_dictkey(
            map_coords=coord, 
            door_type=door_type,
        )
        self.logger.debug(f"Setting freq candidates for {key}: {freq_candidates}")
        self._door_freqs[key] = freq_candidates
    
    def _is_boundary_found(self, door_type: int) -> bool:
        BOUNDARY_NOT_FOUND_VALUES = {-1, self._GLOBAL_MAP_LEN}
        return self._boundaries[door_type] not in BOUNDARY_NOT_FOUND_VALUES

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
        # start_ref is the user-centric coordinate of the start position provided by the current_percept at each turn
        self.cur_pos[0] -= (new_start_ref[0] - self._prev_start_ref[0])
        self.cur_pos[1] -= (new_start_ref[1] - self._prev_start_ref[1])
        self._prev_start_ref = new_start_ref

    def update_door_status(self, coord: List[int], door_type: int, door_state: int):
        key = self._door_dictkey(
            map_coords=coord, 
            door_type=door_type,
        )
        self._door_status[key] = door_state

    def update_map(self, turn_num: int, percept: TimingMazeState):
        self.turn_num = turn_num
        self._update_cur_pos([percept.start_x, percept.start_y])

        cells_seen = set()
        # before = self._door_freqs.copy()
        for door in percept.maze_state:
            player_relative_coordinates, door_type, door_state = door[:2], door[2], door[3]
            coord = self._get_map_coordinates(player_relative_coordinates)

            # update boundaries if newly found
            if door_state == constants.BOUNDARY and not self._is_boundary_found(door_type):
                self._update_boundaries(door_type, coord)

            # update frequencies (TODO: refactor for readability)
            cur_freq_candidates = self._get_freq_candidates_usecase(coord, door_type)  # TODO: consider refactoring how doorID is used
            self.logger.debug(f"Current freq candidates for {coord}: {cur_freq_candidates}")
            new_freq_candidates = get_updated_frequency_candidates(cur_freq_candidates, turn_num=turn_num, door_state=door_state)
            self._set_freq_candidates_usecase(coord, door_type, new_freq_candidates)

            # record seen status
            cells_seen.add(tuple(coord))

            self.update_door_status(coord, door_type, door_state)

        # validation log 
        # after = self._door_freqs
        # diff = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
        # self.logger.debug(f"Diff after turn {turn_num}: {diff}")

        # update seen count
        for cell in cells_seen:
            self._cell_seen_count[cell] += 1

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
    
    def get_seen_counts(self, coords: List[List[int]]) -> List[int]:
        """
        Function which returns the number of times each cell has been seen
            
            Args:
                coords (List[List[int]]): List of cell coordinates (not user-centric, but map-centric/absolute)
            Returns:
                List[int]: List containing the number of times each cell has been seen
        """
        # if coord is outside of boundary, return large int (to deincentivize exploration)
        seen_counts = []
        for c in coords:
            seen_count = 0
            if not isinstance(c, list):
                seen_count = 0
            elif self._is_out_of_bound(c):
                seen_count = self.OUT_OF_BOUND_SEEN_COUNT
            else:
                seen_count = self._cell_seen_count.get(tuple(c), 0)
            seen_counts.append(seen_count)

        self.logger.debug(f"Seen counts for {coords}: {seen_counts}")
        return seen_counts

    def get_valid_moves(self, turn_num: int) -> List[int]:
        if turn_num != self.turn_num:
            raise ValueError("Turn number does not match map's current turn number")

        cur_pos = self.cur_pos
        valid_moves_dependent_doors = {
            constants.LEFT: [
                self._door_dictkey(cur_pos, constants.LEFT), 
                self._door_dictkey([cur_pos[0], cur_pos[1]-1], constants.RIGHT),
            ],
            constants.UP: [
                self._door_dictkey(cur_pos, constants.UP),
                self._door_dictkey([cur_pos[0]-1, cur_pos[1]], constants.DOWN),
            ],
            constants.RIGHT: [
                self._door_dictkey(cur_pos, constants.RIGHT),
                self._door_dictkey([cur_pos[0], cur_pos[1]+1], constants.LEFT),
            ],
            constants.DOWN: [
                self._door_dictkey(cur_pos, constants.DOWN),
                self._door_dictkey([cur_pos[0]+1, cur_pos[1]], constants.UP),
            ],
        }

        valid_moves = []
        for move, door_keys in valid_moves_dependent_doors.items():
            if all([self._door_status[key] == constants.OPEN for key in door_keys]):
                valid_moves.append(move)
        return valid_moves


    def get_wall_freq_candidates(self, door_id: DoorIdentifier) -> List[Set[int]]:
        # Not implemented
        pass

"""
StartPosCentricPlayerMap is similar to SimplePlayerCentricMap but is not centric on the user's current position but rather the first start position which we assign to a constant coordinate (i.e., it receives/outputs start position-centric coordinate data). 
In the SimplePlayerCentricMap, the player was always at (0,0). This is not the case in the StartPosCentricPlayerMap.
"""
class StartPosCentricPlayerMap(PlayerMapInterface):
    def __init__(self, max_door_frequency: int, logger: logging.Logger, map_dim: int = constants.map_dim):
        self._setup_logger(logger)
        
        self._ACTUAL_MAP_DIM = map_dim
        self._GLOBAL_MAP_LEN = 2 * (map_dim-1) + 1
        self._START_POS = [map_dim-1, map_dim-1]

        self.turn_num = 0
        self._end_pos = None
        self._boundaries = [-1, -1, self._GLOBAL_MAP_LEN, self._GLOBAL_MAP_LEN]  # LEFT, UP, RIGHT, DOWN

        self.cur_pos = self._START_POS
        self._door_freqs = defaultdict(default_freq_candidates(max_door_frequency))
        self._cell_seen_count = defaultdict(int)
        self._door_status = defaultdict(int)

        self._prev_start_ref = [0,0]

    def _setup_logger(self, logger):
        self.logger = logger
        self.logger.setLevel(logging.DEBUG)
        self.log_dir = "./log"
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.log_dir, 'Group 5.log'), mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(fh)

    # converts player-centric coordinates (in drone view) to map-centric coordinates
    def _get_map_coordinates(self, player_centric_coordinates: List[int]) -> List[int]:
        return [
            player_centric_coordinates[0] + self.cur_pos[0],
            player_centric_coordinates[1] + self.cur_pos[1],
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
    
    def _door_dictkey(self, map_coords, door_type) -> List[int]:
        return f"({map_coords[0]},{map_coords[1]})_{door_type}"

    def _get_freq_candidates_usecase(self, coord, door_type) -> Set[int]:
        key = self._door_dictkey(
            map_coords=coord, 
            door_type=door_type,
        )
        return self._door_freqs[key]

    def get_freq_candidates(self, door_id: DoorIdentifier) -> Set[int]:
        return self._get_freq_candidates_usecase(door_id.absolute_coord, door_id.door_type)
    
    def _set_freq_candidates_usecase(self, coord, door_type, freq_candidates: Set[int]):
        key = self._door_dictkey(
            map_coords=coord, 
            door_type=door_type,
        )
        self._door_freqs[key] = freq_candidates
    
    def _is_boundary_found(self, door_type: int) -> bool:
        BOUNDARY_NOT_FOUND_VALUES = {-1, self._GLOBAL_MAP_LEN}
        return self._boundaries[door_type] not in BOUNDARY_NOT_FOUND_VALUES

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
        # start_ref is the user-centric coordinate of the start position provided by the current_percept at each turn
        self.cur_pos[0] -= (new_start_ref[0] - self._prev_start_ref[0])
        self.cur_pos[1] -= (new_start_ref[1] - self._prev_start_ref[1])
        self._prev_start_ref = new_start_ref

    def update_door_status(self, coord: List[int], door_type: int, door_state: int):
        key = self._door_dictkey(
            map_coords=coord, 
            door_type=door_type,
        )
        self._door_status[key] = door_state

    def update_map(self, turn_num: int, percept: TimingMazeState):
        self.turn_num = turn_num
        self._update_cur_pos([percept.start_x, percept.start_y])

        cells_seen = set()
        for door in percept.maze_state:
            player_relative_coordinates, door_type, door_state = door[:2], door[2], door[3]
            coord = self._get_map_coordinates(player_relative_coordinates)

            # update boundaries if newly found
            if door_state == constants.BOUNDARY and not self._is_boundary_found(door_type):
                self._update_boundaries(door_type, coord)

            # update frequencies (TODO: refactor for readability)
            cur_freq_candidates = self._get_freq_candidates_usecase(coord, door_type)  # TODO: consider refactoring how doorID is used
            new_freq_candidates = get_updated_frequency_candidates(cur_freq_candidates, turn_num=turn_num, door_state=door_state)
            self._set_freq_candidates_usecase(coord, door_type, new_freq_candidates)

            # record seen status
            cells_seen.add(tuple(coord))

            self.update_door_status(coord, door_type, door_state)

        # update seen count
        for cell in cells_seen:
            self._cell_seen_count[cell] += 1

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
    
    def get_seen_counts(self, coords: List[List[int]]) -> List[int]:
        """
        Function which returns the number of times each cell has been seen
            
            Args:
                coords (List[List[int]]): List of cell coordinates (not user-centric, but map-centric/absolute)
            Returns:
                List[int]: List containing the number of times each cell has been seen
        """
        # if coord is outside of boundary, return large int (to deincentivize exploration)
        seen_counts = []
        for c in coords:
            seen_count = 0
            if not isinstance(c, list):
                seen_count = 0
            elif self._is_out_of_bound(c):
                seen_count = self.OUT_OF_BOUND_SEEN_COUNT
            else:
                seen_count = self._cell_seen_count.get(tuple(c), 0)
            seen_counts.append(seen_count)

        return seen_counts

    def get_valid_moves(self, turn_num: int) -> List[int]:
        if turn_num != self.turn_num:
            raise ValueError("Turn number does not match map's current turn number")

        cur_pos = self.cur_pos
        valid_moves_dependent_doors = {
            constants.LEFT: [
                self._door_dictkey(cur_pos, constants.LEFT), 
                self._door_dictkey([cur_pos[0], cur_pos[1]-1], constants.RIGHT),
            ],
            constants.UP: [
                self._door_dictkey(cur_pos, constants.UP),
                self._door_dictkey([cur_pos[0]-1, cur_pos[1]], constants.DOWN),
            ],
            constants.RIGHT: [
                self._door_dictkey(cur_pos, constants.RIGHT),
                self._door_dictkey([cur_pos[0], cur_pos[1]+1], constants.LEFT),
            ],
            constants.DOWN: [
                self._door_dictkey(cur_pos, constants.DOWN),
                self._door_dictkey([cur_pos[0]+1, cur_pos[1]], constants.UP),
            ],
        }

        valid_moves = []
        for move, door_keys in valid_moves_dependent_doors.items():
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
