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

        self.rng = rng
        self.logger = logger
        self.maximum_door_frequency = maximum_door_frequency
        self.radius = radius
        self.turn = 0
        # x, y in seens and knowns is centered around start x, y
        self.seens = dict() # dictionary w/ kv - (x, y, d): [list of turns at which x, y, d was open]
        self.knowns = dict() # dictionary w/ kv - (x, y): {0: freq(L), 1: freq(U), 2: freq(R), 3: freq(D)}, freq = -1 if unknown
        self.cur_x = 0 # initializing to start x
        self.cur_y = 0 # initializing to start y

    @staticmethod
    def lcm(x, y):
        if x > y:
            greater = x
        else:
            greater = y
        while(True):
            if((greater % x == 0) and (greater % y == 0)):
                lcm = greater
                break
            greater += 1
        return lcm
    
    def setInfo(self, maze_state: list, turn: int) -> dict:
        """Function receives the current state of the amoeba map and returns a dictionary of door frequencies centered around the start position.

        notes: 
        current_percept.maze_state[0,1]: coordinates around current position
        current_percept.maze_state[2]: direction of door (L: 0, U: 1, R: 2, D: 3)
        current_percept.maze_state[3]: status of door (Closed: 1, Open: 2, Boundary: 3)

        doors that touch each other (n, m, d): 
        (n, m, 0) - (n - 1, m, 2)
        (n, m, 1) - (n, m - 1, 3)
        (n, m, 2) - (n + 1, m, 0)
        (n, m, 3) - (n, m + 1, 1)
     
        returns: dictionary that changes the keys of knowns (within current radius) to center around cur_x, cur_y and randomizes unknown frequencies
        """

        drone = {}
        for door in maze_state:
            if (door[0], door[1]) not in drone:
                drone[(door[0], door[1])] = {constants.LEFT: -1, constants.UP: -1, constants.RIGHT: -1, constants.DOWN: -1}
        for (x, y) in drone: 
            if (x - 1, y) in drone:
                if drone[x, y][constants.LEFT] == -1 and drone [x - 1, y][constants.RIGHT] == -1:
                    f = self.lcm(self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True), self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True))
                    drone[x, y][constants.LEFT] = f
                    drone [x - 1, y][constants.RIGHT] = f
            else:
                if drone[x, y][constants.LEFT] == -1: 
                    drone[x, y][constants.LEFT] = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
            if (x, y - 1) in drone:
                if drone[x, y][constants.UP] == -1 and drone [x, y - 1][constants.DOWN] == -1:
                    f = self.lcm(self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True), self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True))
                    drone[x, y][constants.UP] = f
                    drone [x, y - 1][constants.DOWN] = f
            else:
                if drone[x, y][constants.UP] == -1: 
                    drone[x, y][constants.UP] = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
            if (x + 1, y) in drone:
                if drone[x, y][constants.RIGHT] == -1 and drone [x + 1, y][constants.LEFT] == -1:
                    f = self.lcm(self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True), self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True))
                    drone[x, y][constants.RIGHT] = f
                    drone [x + 1, y][constants.LEFT] = f
            else:
                if drone[x, y][constants.RIGHT] == -1: 
                    drone[x, y][constants.RIGHT] = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)

            if (x, y + 1) in drone: 
                if drone[x, y][constants.DOWN] == -1 and drone [x, y + 1][constants.UP] == -1:
                    f = self.lcm(self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True), self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True))
                    drone[x, y][constants.DOWN] = f
                    drone [x, y + 1][constants.UP] = f
            else:
                if drone[x, y][constants.DOWN] == -1: 
                    drone[x, y][constants.DOWN] = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    
        # print ("seens:", self.seens)
        # print ("knowns:", self.knowns)
        # print ("drone:", drone)
        return drone


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
        turn += 1
        self.setInfo(current_percept.maze_state, self.turn)
        return 0