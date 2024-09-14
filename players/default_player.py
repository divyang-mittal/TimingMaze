import os
import pickle
import numpy as np
import logging
from utils import get_divisors

import constants
from timing_maze_state import TimingMazeState

class MemoryDoor:
    def __init__(self):
        self.is_certain_freq = False
        self.observations = {} # {turn : 1 - Closed / 2 - Open / 3 - Boundary}
        self.freq_distribution = []

    def update_observations(self, door_state, turn):
        # Updates observed freqs, runs get_freq
        self.observations[turn] = door_state
        self.freq_distribution = self.get_freq()
        if len(self.freq_distribution) == 1:
            self.is_certain_freq = True
    
    def get_freq(self):
        # Tries to find the frequency given observed, returns a probability distribution
        possible_open_frequencies = set()
        closed_frequencies = set()

        # Iterate over the frequency conditions
        for freq, status in self.observations.items():
            if status == 2:
                # Add all divisors of the frequency if it's open
                if not possible_open_frequencies:
                    possible_open_frequencies = get_divisors(freq)
                else:
                    possible_open_frequencies &= get_divisors(freq)
            else:
                # Add all divisors of the closed frequency
                closed_frequencies |= get_divisors(freq)

        # Remove any closed frequencies from the possible open set
        possible_open_frequencies -= closed_frequencies

        # Assign equal probabilities to each remaining frequency
        total_frequencies = len(possible_open_frequencies)
        probability_distribution = {
            freq: 1/total_frequencies for 
            freq in possible_open_frequencies} if total_frequencies > 0 else {}

        return probability_distribution
    

class MemorySquare:
    def __init__(self):
        left = MemoryDoor()
        up = MemoryDoor()
        right = MemoryDoor()
        down = MemoryDoor()
        self.doors = {constants.LEFT:left, constants.UP:up, constants.RIGHT:right, constants.DOWN:down}

class PlayerMemory:
    def __init__(self, map_size: int = 100):
        self.memory = [[MemorySquare() for _ in range(map_size * 2)] for _ in range(map_size * 2)]
        self.pos = (map_size, map_size)
    
    def update_memory(self, state, turn):
        # state = [door] = (row_offset, col_offset, door_type, door_status)
        for s in state:
            square = self.memory[self.pos[0] + s[0]][self.pos[1] + s[1]]
            door = square.doors[s[2]]
            door_state = s[3]
            door.update_observations(door_state, turn)

    def update_pos(self, move):
        if move == constants.LEFT:
            self.pos[1] -= 1
        if move == constants.UP:
            self.pos[0] -= 1
        if move == constants.RIGHT:
            self.pos[1] += 1
        if move == constants.DOWN:
            self.pos[0] += 1



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
        direction = [0, 0, 0, 0]
        for maze_state in current_percept.maze_state:
            if maze_state[0] == 0 and maze_state[1] == 0:
                direction[maze_state[2]] = maze_state[3]

        if current_percept.is_end_visible:
            if abs(current_percept.end_x) >= abs(current_percept.end_y):
                if current_percept.end_x > 0 and direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                                and maze_state[3] == constants.OPEN):
                            return constants.RIGHT
                if current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                                and maze_state[3] == constants.OPEN):
                            return constants.LEFT
                if current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                                and maze_state[3] == constants.OPEN):
                            return constants.UP
                if current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                                and maze_state[3] == constants.OPEN):
                            return constants.DOWN
                return constants.WAIT
            else:
                if current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                                and maze_state[3] == constants.OPEN):
                            return constants.UP
                if current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                                and maze_state[3] == constants.OPEN):
                            return constants.DOWN
                if current_percept.end_x > 0 and direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                                and maze_state[3] == constants.OPEN):
                            return constants.RIGHT
                if current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                                and maze_state[3] == constants.OPEN):
                            return constants.LEFT
                return constants.WAIT
        else:
            if direction[constants.LEFT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                            and maze_state[3] == constants.OPEN):
                        return constants.LEFT
            if direction[constants.DOWN] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                            and maze_state[3] == constants.OPEN):
                        return constants.DOWN
            if direction[constants.RIGHT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                            and maze_state[3] == constants.OPEN):
                        return constants.RIGHT
            if direction[constants.UP] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                            and maze_state[3] == constants.OPEN):
                        return constants.UP
            return constants.WAIT
