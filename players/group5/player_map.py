from abc import ABC, abstractmethod
from collections import defaultdict
import logging
import os
from typing import List, Set

import constants
from players.group5.door import DoorIdentifier, update_frequency_candidates


class PlayerMapInterface(ABC):
    @abstractmethod
    def get_start_pos(self) -> List[int]:
        """Function which returns the start position of the game

            Returns:
                List[int]: List containing the x and y coordinates of the start position (relative to current position)
        """
        pass

    @abstractmethod
    def get_end_pos(self) -> List[int]:
        """Function which returns the end position of the game

            Returns:
                List[int]: List containing the x and y coordinates of the end position (relative to current position)
        """
        pass

    @abstractmethod
    def get_freq_candidates(self, door_id: DoorIdentifier) -> List[int]:
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
    
    def update_map(self, turn_num: int, maze_state: List[List[int]]):  # TODO: check type of maze_state
        """Function which updates the map with the given maze state

            Args:
                turn_num (int): Integer representing the current turn number
                maze_state (List[List[int]]): 2D list of integers representing the maze state (within the radius) at the given turn
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

        self._map_len = 2 * (map_dim-1) + 1
        self._start_pos = [map_dim-1, map_dim-1]
        self._end_pos = [None, None]
        self._door_freqs = defaultdict(default_freq_candidates(max_door_frequency))
        self._boundaries = [-1, self._map_len+1, -1, self._map_len+1]
        # visited cells

        self.cur_pos = self._start_pos

    def _get_player_relative_coordinates(self, map_coordinates: List[int]) -> List[int]:
        return [map_coordinates[0] - self.cur_pos[0], map_coordinates[1] - self.cur_pos[1]]
    
    def _get_map_coordinates(self, player_relative_coordinates: List[int]) -> List[int]:
        return [player_relative_coordinates[0] + self.cur_pos[0], player_relative_coordinates[1] + self.cur_pos[1]]

    def get_start_pos(self) -> List[int]:
        return self._get_player_relative_coordinates(self._start_pos)

    def get_end_pos(self) -> List[int]:
        if self._end_pos[0] is None or self._end_pos[1] is None:
            return [None, None]
        return self._get_player_relative_coordinates(self._end_pos)

    def _door_key(self, door_id: DoorIdentifier) -> List[int]:
        map_coords = self._get_map_coordinates(door_id.relative_coord)
        return f"({map_coords[0]},{map_coords[1]})_{door_id.door_type}"
    
    def get_freq_candidates(self, door_id: DoorIdentifier) -> List[int]:
        return self._door_freqs[self._door_key(door_id)]
    
    def _set_freq_candidates(self, door_id: DoorIdentifier, freq_candidates: Set[int]):
        self._door_freqs[self._door_key(door_id)] = freq_candidates
    
    def apply_move(self, move: int):
        if move == constants.LEFT:
            self.cur_pos[0] -= 1
        elif move == constants.UP:
            self.cur_pos[1] -= 1
        elif move == constants.RIGHT:
            self.cur_pos[0] += 1
        elif move == constants.DOWN:
            self.cur_pos[1] += 1

    def update_map(self, turn_num: int, maze_state: List[List[int]]):
        self.logger.debug(f"Before turn {turn_num}: {self._door_freqs}")
        for door in maze_state:
            door_coordinates = door[:2]
            door_type = door[2]
            door_state = door[3]

            # TODO: clean this up
            if door_state == constants.BOUNDARY and (self._boundaries[door_type] == -1 or self._boundaries[door_type] == self._map_len+1):
                if door_type == constants.LEFT:
                    # then its the right barrier? TODO: check
                    self._boundaries[constants.RIGHT] = door_coordinates[0]
                    self._boundaries[constants.LEFT] = door_coordinates[0] - 101
                elif door_type == constants.UP:
                    self._boundaries[constants.DOWN] = door_coordinates[1]
                    self._boundaries[constants.UP] = door_coordinates[1] - 101
                elif door_type == constants.RIGHT:
                    self._boundaries[constants.LEFT] = door_coordinates[0]
                    self._boundaries[constants.RIGHT] = door_coordinates[0] + 101
                elif door_type == constants.DOWN:
                    self._boundaries[constants.UP] = door_coordinates[1]
                    self._boundaries[constants.DOWN] = door_coordinates[1] + 101    
                
                self.logger.debug(f"Boundaries: {self._boundaries}")

            door_id = DoorIdentifier(relative_coord=door_coordinates, door_type=door_type)
            frequency_candidates = self.get_freq_candidates(door_id)
            
            # TODO: remove
            # self.logger.debug(f"Before turn {turn_num}: {frequency_candidates}")

            new_freq_candidates = update_frequency_candidates(frequency_candidates, turn_num=turn_num, door_state=door[3], logger=self.logger)

            self._set_freq_candidates(door_id, new_freq_candidates)
            # self.logger.debug(f"Afterr: {self.get_freq_candidates(door_id)}")
        self.logger.debug(f"After turn {turn_num}: {self._door_freqs}\n==============\n")

        
        

