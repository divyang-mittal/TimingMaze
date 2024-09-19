import numpy as np
import logging

from constants import *
from players.g6_player.data import Move
from timing_maze_state import TimingMazeState
from players.g6_player.classes.maze import Maze
from players.g6_player.explore import *
from players.g6_player.exploit import *


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
                rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
                logger (logging.Logger): logger use this like logger.info("message")
                maximum_door_frequency (int): the maximum frequency of doors
                radius (int): the radius of the drone
                precomp_dir (str): Directory path to store/load pre-computation
        """
        self.rng = rng
        self.logger = logger
        self.maximum_door_frequency = maximum_door_frequency
        self.radius = radius
        self.turn = 0

        # Initialize Maze object to hold information about cells and doors perceived by the drone
        self.maze = Maze(self.turn, self.maximum_door_frequency, self.radius)

        # an interim target which the agent tries to navigate towards
        self.search_target = None

    def move(self, current_percept: TimingMazeState) -> int:
        """
        Increments the turn count and updates the maze with the current percept. Calls
        __move() to determine the next move.
        """
        self.turn += 1
        self.maze.update_maze(current_percept, self.turn)
        player_move = self.__move(current_percept)
        
        return player_move.value

    def __move(self, current_percept: TimingMazeState) -> Move:
        if not current_percept.is_end_visible:
            # Explore map to get target within drone's view
            return explore()

        # Go to target
        return exploit(current_percept)
    
