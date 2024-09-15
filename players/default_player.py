import os
import pickle
import numpy as np
import logging
import constants
from timing_maze_state import TimingMazeState


class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger,
                 precomp_dir: str, maximum_door_frequency: int, radius: int) -> None:
        """Initialise the player with the basic amoeba information

            Args:
                rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
                logger (logging.Logger): logger use this like logger.info("message")
                maximum_door_frequency (int): the maximum frequency of doors
                radius (int): the radius of the drone
                precomp_dir (str): Directory path to store/load pre-computation
        """

        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))

        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)

        self.rng = rng
        self.logger = logger
        self.maximum_door_frequency = maximum_door_frequency
        self.radius = radius

        # relative frequency map is stored in a 3D matrix (x = relative x pos, y = relative y pos, 
        # z = one of the four doors (0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT))
        # for example, self.relative_frequencies[x, y, 1] represents the frequency of the RIGHT door

        self.relative_frequencies = np.full((201, 201, 4), -1, dtype=int)
        # because the relative frequency map is 201x201, setting a [100,100] relative position puts us at the center of this map,
        # so the player can then move up to 100 cells in any direction without going out of bounds
        # also, since the player doesn't know their absolute position in the map, starting here allows for easier tracking
        self.current_relative_pos = [100, 100]


    def update_relative_frequencies(self, current_percept):
        for cell in current_percept.maze_state:
            print(cell)
            x, y, direction, state = cell
            abs_x = self.current_relative_pos[0] + x
            abs_y = self.current_relative_pos[1] + y
            
            if state == constants.OPEN:
                # door is always open
                self.relative_frequencies[abs_x, abs_y, direction] = 1
            elif state == constants.CLOSED:
                # if it's the first time seeing the door (since -1 indicates unexplored), set to max_freq so we assume the worst case
                if self.relative_frequencies[abs_x, abs_y, direction] == -1:
                    self.relative_frequencies[abs_x, abs_y, direction] = self.maximum_door_frequency
                # if we've seen it before and it's closed, decrement the frequency (it opens less frequently than we thought)
                elif self.relative_frequencies[abs_x, abs_y, direction] > 1:
                    self.relative_frequencies[abs_x, abs_y, direction] -= 1
            elif state == constants.BOUNDARY:
                # basically the door always closed
                self.relative_frequencies[abs_x, abs_y, direction] = 0

    def update_position(self, move):
        if move == constants.RIGHT:
            self.current_relative_pos[0] += 1
        elif move == constants.LEFT:
            self.current_relative_pos[0] -= 1
        elif move == constants.DOWN:
            self.current_relative_pos[1] += 1
        elif move == constants.UP:
            self.current_relative_pos[1] -= 1

    def move(self, current_percept) -> int:
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
        
        # update our knowledge of the maze at the beginning of the move
        self.update_relative_frequencies(current_percept)

        direction = [0, 0, 0, 0]
        for maze_state in current_percept.maze_state:
            if maze_state[0] == 0 and maze_state[1] == 0:
                direction[maze_state[2]] = maze_state[3]

        chosen_move = constants.WAIT  # default to WAIT

        if current_percept.is_end_visible:
            if abs(current_percept.end_x) >= abs(current_percept.end_y):
                if current_percept.end_x > 0 and direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                                and maze_state[3] == constants.OPEN):
                            chosen_move = constants.RIGHT
                            break
                elif current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                                and maze_state[3] == constants.OPEN):
                            chosen_move = constants.LEFT
                            break
                elif current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                                and maze_state[3] == constants.OPEN):
                            chosen_move = constants.UP
                            break
                elif current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                                and maze_state[3] == constants.OPEN):
                            chosen_move = constants.DOWN
                            break
            else:
                if current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                                and maze_state[3] == constants.OPEN):
                            chosen_move = constants.UP
                            break
                elif current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                                and maze_state[3] == constants.OPEN):
                            chosen_move = constants.DOWN
                            break
                elif current_percept.end_x > 0 and direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                                and maze_state[3] == constants.OPEN):
                            chosen_move = constants.RIGHT
                            break
                elif current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                                and maze_state[3] == constants.OPEN):
                            chosen_move = constants.LEFT
                            break
        else:
            if direction[constants.LEFT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                            and maze_state[3] == constants.OPEN):
                        chosen_move = constants.LEFT
                        break
            elif direction[constants.DOWN] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                            and maze_state[3] == constants.OPEN):
                        chosen_move = constants.DOWN
                        break
            elif direction[constants.RIGHT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                            and maze_state[3] == constants.OPEN):
                        chosen_move = constants.RIGHT
                        break
            elif direction[constants.UP] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                            and maze_state[3] == constants.OPEN):
                        chosen_move = constants.UP
                        break

        # update position based on the chosen move
        self.update_position(chosen_move)

        return chosen_move