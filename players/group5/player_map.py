from abc import ABC, abstractmethod
from collections import defaultdict
import logging
import os
from typing import List, Optional, Set, Tuple

import constants
from players.group5.door import DoorIdentifier, update_frequency_candidates
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
    def apply_move(self, move: int):
        """Function which applies the given move to the current position

            Args:
                move (int): Integer representing the move to be applied
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


def default_freq_candidates(max_door_frequency: int):
    # generate a set of door frequencies from 0 to max_door_frequency
    return lambda: set(range(max_door_frequency+1))


class SimplePlayerMap(PlayerMapInterface):
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
            new_freq_candidates = update_frequency_candidates(cur_freq_candidates, turn_num=turn_num, door_state=door_state)

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

    def get_seen_counts(self, relative_coord: List[List[int]]) -> List[int]:
        return [self._cell_seen_count.get(tuple(self._get_map_coordinates(cell)), 0) for cell in relative_coord]
        

