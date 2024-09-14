import numpy as np
import logging
from typing import List
from players.g6_player.data import Move
from timing_maze_state import TimingMazeState


class G6_Player:
    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        precomp_dir: str,
        maximum_door_frequency: int,
        radius: int,
    ) -> None:
        """Initialise the player with the basic amoeba information
        Args:
            maximum_door_frequency (int): the maximum frequency of doors
            radius (int): the radius of the drone
        """

        self.rng = rng
        self.logger = logger
        self.maximum_door_frequency = maximum_door_frequency
        self.radius = radius
        self.known_target = False
        # Data structure to hold state information about the doors inside the radius
        self.curr_maze = {}
        self.turn = 0
        self.move_history = np.array([0,0])

    def move(self, current_percept: TimingMazeState) -> int:
        self.turn += 1
        # TimingMazeState with updated coordinates to be relative to the start based on our player's history
        converted_maze_status = self.__convert_state_coordinates(current_percept.maze_state)
        # Todo: set frequencies on the map
            # Case: First turn anything open == always open
        # Todo: use frequencies to update worst case distance between two cells
            # Note that THIS is the object we likely want to use to make decisions about explore and exploit
        if not self.known_target:
            return self.__explore()
        return self.__exploit()

    def __convert_state_coordinates(self, curr_state: TimingMazeState):
        # First turn edge case: the current coordinates centered on player position [0,0] do not need to be converted
        if self.turn == 1:
            return TimingMazeState
        # Sum all previous moves for single vector summarizing x,y moves since start of game
        sum_history = np.sum(self.move_history, axis=0)
        # Convert coordinates in curr_state to be relative to player starting position
        for door in range(0, len(curr_state)):
            cell = np.array(curr_state[door])
            cell[0] = cell[0] + sum_history[0]
            cell[1] = cell[1] + sum_history[1]
            curr_state[door] = tuple(cell)

        return curr_state

    def __update_history(self, new_move) -> np.ndarray:
        if new_move == Move.LEFT:
            return np.vstack((self.move_history, np.array([-1,0])))
        elif new_move == Move.RIGHT:
            return np.vstack((self.move_history, np.array([1,0])))
        elif new_move == Move.UP:
            return np.vstack((self.move_history, np.array([0, 1])))
        elif new_move == Move.DOWN:
            return np.vstack((self.move_history, np.array([0, -1])))
        else:
            return  self.move_history + np.array([0,0])

    def __update_maze(self) -> dict[str, int]:
        # Update current maze with new info from the drone
        return {}

    def __explore(self) -> Move:
        new_move = Move.RIGHT
        # Todo: this might be the wrong place to update history because if we recommend a move that is Invalid the history is appended even though the move does not ultimately happen..
            # Todo: @robin - fix to use the current_percept.start_x and start_y because these are adjusted AFTER a move is executed
        self.move_history = self.__update_history(new_move)
        return new_move.value

    def __exploit(self) -> Move:
        new_move = Move.RIGHT.value
        # Todo: same as above comment about updating the move history
        self.move_history = self.__update_history(new_move)
        return new_move.value
