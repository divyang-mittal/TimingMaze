from array import ArrayType
from random import random

import numpy as np
import logging
from typing import List
from players.g6_player.data import Move
from timing_maze_state import TimingMazeState
from constants import LEFT, UP, RIGHT, DOWN


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
        self.maze_boundary_x = 201
        self.maze_boundary_y = 201
        # player_maze[x][y][cycle][door_direction] = boolean door open status
        self.player_maze_freq = np.zeros((self.maze_boundary_x, self.maze_boundary_y, self.maximum_door_frequency, 4))
        # player_maze_path_intermediate[x][y][4] = frequency each door is open
        self.player_maze_path_intermediate = np.zeros((self.maze_boundary_x, self.maze_boundary_y, 4))
        # player_maze_path[x][y][4] = frequency each path is open
        self.player_maze_path = np.zeros((self.maze_boundary_x, self.maze_boundary_y, 4))
        self.turn = 0
        self.cycle = 0
        self.move_history = [np.array([0,0])]

    def move(self, current_percept: TimingMazeState) -> int:
        # Bug: Move Enum is not an int and will not be accepted as a move
        self.turn += 1
        self.cycle = self.turn % self.maximum_door_frequency
        if self.cycle == 0:
            self.cycle = self.maximum_door_frequency
        # Adjust drone coordinates to be positive when going right and center on the 200,200 grid.
        adjusted_x = (current_percept.start_x * -1) + 100
        adjusted_y = current_percept.start_y + 100
        adjusted_coords = np.array([adjusted_x, adjusted_y])
        self.__update_history(adjusted_coords)
        self.__convert_to_freq_matrix(current_percept.maze_state, adjusted_coords)
        self.__convert_to_path_matrix()

        # Todo - the code below will want to use the self.player_maze_path to understand distances
        # Todo - currently most of the distances in our frquency matrices are set to 0 automatically. More thought needs to consider what values to initialize with and the downstream implications (to tell doors that NEVER open which should be 0 apart from doors that we just haven't observed open YET)
        if not current_percept.is_end_visible:
            return self.__explore()
        return self.__exploit()

    def __update_history(self, adjusted_coords: ArrayType[int, int]):
        """
        This function adjusts the move_history ordered list of coordinates that the player has already visited

        :param adjusted_coords: coordinates of the player relative to the player's starting position at 0,0 and shifted to the center of our player maze matrix.
        :return:
        """
        if np.array_equal(adjusted_coords, self.move_history[-1]):
            # No move on previous turn so no move added to move history
            return
        self.move_history.append(adjusted_coords)
        return

    def __convert_to_freq_matrix(self, drone_state: TimingMazeState, adjusted_coords: ArrayType[int, int]):
        """
        Take the current maze state observed by the drone and adjust the frequency matrix that contains a memory of every turn a door is observed to be open on.

        :param drone_state: object returned from the drone with observed maze state describing coordinates relative to current player location and door status's observed in radius.
        :param adjusted_coords: coordinates of the player relative to the player's starting position at 0,0 and shifted to the center of our player maze matrix.
        :return:
        """
        # Todo - this doesn't need to happen everytime and computationally will be duplicative after a door is either observed on the first turn (as open is always open, as closed it is always closed) or for max_frequency number cycles.
        for door in drone_state:
            drone_x, drone_y, door_direction, status = door[0], door[1], door[2], door[3]
            # Maps drone door coord (centered on player always) --> relative coordinates to the start --> global coordinates
            x = drone_x + adjusted_coords[0]
            y = drone_y + adjusted_coords[1]

            if status == 1:
                self.player_maze_freq[x][y][self.cycle-1][door_direction] = False
            elif status == 2:
                self.player_maze_freq[x][y][self.cycle-1][door_direction] = True
            else:
                # Todo - this is the case where status = 3 = Boundary. Here we likely reframe the full coordinate system for search.
                self.player_maze_freq[x][y][self.cycle-1][door_direction] = False
        return

    def __convert_to_path_matrix(self):
        """
        This function uses the frequency matrix to:
            1. Summarize the frequency based on observations that each door is open at
            2. Create a matrix which describes the frequency with which each path is open between two doors
        :return:
        """
        # Todo - this should be bounded only on coordinates we have seen
        # For every cell in the maze get the least common denominator freq for the doors opening
        for x in range(0,self.maze_boundary_x):
            for y in range(0, self.maze_boundary_y):
                for door in range(0, 4):
                    indices = np.where(self.player_maze_freq[x][y][:, door] == 1)[0]
                    if len(indices) > 0:
                        # Get the minimum index
                        min_index = np.min(indices)+1
                    else:
                        min_index = 0
                    self.player_maze_path_intermediate[x][y][door] = min_index

        # Todo - probably a less gross way to multiply this intermediate matrix to get the path frequencies
        # Multiply the door and neighboring door frequencies to get path frequency
        for x in range(0, self.maze_boundary_x):
            for y in range(0, self.maze_boundary_y):
                print(x, y)
                for door in range(0, 4):
                    # Todo - this is a weak appraoch to handle for hitting boundaries and reallly should be improved.
                    if x == 200 and y == 200:
                        self.player_maze_path[x][y][LEFT] = self.player_maze_path_intermediate[x][y][LEFT] * \
                                                            self.player_maze_path_intermediate[x - 1][y][RIGHT]
                        self.player_maze_path[x][y][DOWN] = self.player_maze_path_intermediate[x][y][DOWN] * \
                                                            self.player_maze_path_intermediate[x][y - 1][UP]

                    elif x == 200:
                        self.player_maze_path[x][y][LEFT] = self.player_maze_path_intermediate[x][y][LEFT] * \
                                                            self.player_maze_path_intermediate[x - 1][y][RIGHT]
                        self.player_maze_path[x][y][UP] = self.player_maze_path_intermediate[x][y][UP] * \
                                                          self.player_maze_path_intermediate[x][y + 1][DOWN]
                        self.player_maze_path[x][y][DOWN] = self.player_maze_path_intermediate[x][y][DOWN] * \
                                                            self.player_maze_path_intermediate[x][y - 1][UP]
                    elif y == 200:
                        self.player_maze_path[x][y][RIGHT] = self.player_maze_path_intermediate[x][y][RIGHT] * \
                                                             self.player_maze_path_intermediate[x + 1][y][LEFT]
                        self.player_maze_path[x][y][LEFT] = self.player_maze_path_intermediate[x][y][LEFT] * \
                                                            self.player_maze_path_intermediate[x - 1][y][RIGHT]
                        self.player_maze_path[x][y][DOWN] = self.player_maze_path_intermediate[x][y][DOWN] * \
                                                            self.player_maze_path_intermediate[x][y - 1][UP]

                    else:
                        self.player_maze_path[x][y][RIGHT] = self.player_maze_path_intermediate[x][y][RIGHT] * \
                                                             self.player_maze_path_intermediate[x + 1][y][LEFT]
                        self.player_maze_path[x][y][LEFT] = self.player_maze_path_intermediate[x][y][LEFT] * \
                                                            self.player_maze_path_intermediate[x - 1][y][RIGHT]
                        self.player_maze_path[x][y][DOWN] = self.player_maze_path_intermediate[x][y][DOWN] * \
                                                            self.player_maze_path_intermediate[x][y - 1][UP]
                        self.player_maze_path[x][y][UP] = self.player_maze_path_intermediate[x][y][UP] * \
                                                          self.player_maze_path_intermediate[x][y + 1][DOWN]



        return

    def __explore(self) -> Move:
        rand = random()
        if rand <= 0.25:
            new_move = Move.LEFT
        elif rand <= 0.5:
            new_move = Move.RIGHT
        elif rand <= 0.75:
            new_move = Move.DOWN
        else:
            new_move = Move.UP
        return new_move.value

    def __exploit(self, current_state: TimingMazeState) -> Move:
        if random() < 0.1:
            new_move = random.choice(list(Move))
        elif 0 > current_state.end_x:
            new_move = Move.LEFT.value
        elif 0 < current_state.end_x:
            new_move = Move.RIGHT.value
        elif 0 > current_state.end_y:
            new_move = Move.UP.value
        elif 0 < current_state.end_y:
            new_move = Move.DOWN.value
        else:
            new_move = Move.WAIT.value
        # Todo: same as above comment about updating the move history
        self.move_history = self.__update_history(new_move)
        return new_move
