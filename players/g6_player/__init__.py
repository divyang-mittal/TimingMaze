import numpy as np
from constants import LEFT
import logging
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

    def move(self, current_percept: TimingMazeState) -> int:
        # Bug: Move Enum is not an int and will not be accepted as a move
        self.turn += 1

        print(f"x: {current_percept.start_x}, y: {current_percept.start_y}")
        if not self.known_target:
            # SEARCH FOR TARGET
            return self.__explore()

        # GO TO TARGET
        return self.__exploit()

    def __update_maze(self) -> dict[str, int]:
        # Update current maze with new info from the drone

        return {}

    def __convert_state(self, curr_state: TimingMazeState):
        # Update self.curr_maze with new state information
        pass

    def __explore(self) -> Move:
        return Move.LEFT

    def __exploit(self) -> Move:
        return Move.RIGHT
