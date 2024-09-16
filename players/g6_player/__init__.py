import numpy as np
import logging
import random

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
        self.seen = np.zeros((100, 100))
        self.curr_pos = (0, 0)
        self.turn = 0

    def move(self, current_percept: TimingMazeState) -> int:
        self.turn += 1

        self.__update_maze(current_percept)

        # seen_count = np.sum(self.seen)

        movement = self.__move(current_percept)
        self.__update_curr_pos(movement)

        return movement.value

    def __move(self, current_percept: TimingMazeState) -> Move:
        if not current_percept.is_end_visible:
            # SEARCH FOR TARGET
            return self.__explore()

        # GO TO TARGET
        return self.__exploit(current_percept)

    def __update_maze(self, curr_state: TimingMazeState):
        # Update current maze with new info from the drone
        for cell in curr_state.maze_state:
            cell_row = (cell[0] + self.curr_pos[0]) % 100
            cell_col = (cell[1] + self.curr_pos[1]) % 100

            self.seen[cell_row][cell_col] = True

    def __convert_state(self, curr_state: TimingMazeState):
        # Update self.curr_maze with new state information
        pass

    def __update_curr_pos(self, movement: Move):
        match movement:
            case Move.LEFT:
                self.curr_pos = (
                    (self.curr_pos[0] - 1) % 100,
                    self.curr_pos[1] % 100,
                )
            case Move.RIGHT:
                self.curr_pos = (
                    (self.curr_pos[0] + 1) % 100,
                    self.curr_pos[1] % 100,
                )
            case Move.UP:
                self.curr_pos = (
                    (self.curr_pos[0]) % 100,
                    (self.curr_pos[1] - 1) % 100,
                )
            case Move.DOWN:
                self.curr_pos = (
                    (self.curr_pos[0]) % 100,
                    (self.curr_pos[1] + 1) % 100,
                )

    def __explore(self) -> Move:
        return Move.LEFT

    def __exploit(self, current_state: TimingMazeState) -> Move:
        if random.random() < 0.1:
            return random.choice(list(Move))

        if 0 > current_state.end_x:
            return Move.LEFT

        if 0 < current_state.end_x:
            return Move.RIGHT

        if 0 > current_state.end_y:
            return Move.UP

        if 0 < current_state.end_y:
            return Move.DOWN

        return Move.WAIT
