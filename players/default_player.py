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
        # x, y in seens and knowns is centered around start x, y
        self.seens = dict() # dictionary w/ kv - (x, y, d): (False (uncertain)/True (certain), assumed freq, [list of turns at which x, y, d was open], [list of turns at which x, y, d could be seen])
        self.knowns = dict() # dictionary w/ kv - (x, y): {0: freq(L), 1: freq(U), 2: freq(R), 3: freq(D)}, freq = -1 if unknown
        self.cur_x = 0 # initializing to start x
        self.cur_y = 0 # initializing to start y
        self.turn = 0

    @staticmethod
    def findSmallestGap(seen):
        if len(seen) < 2:
            return -1
        gap = seen[1] - seen[0]
        for i in range(len(seen) - 2):
            if seen[i + 1] - seen[i] < gap:
                gap = seen[i + 1] - seen[i]
        return gap
    
    def setInfo(self, maze_state, turn) -> dict:
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

        drone = {} # drone view around the current x, y, at radius r

        for ms in maze_state:
            if turn == 1:
                if ms[3] == constants.CLOSED:
                    continue
                elif ms[3] == constants.OPEN:
                    if (ms[0], ms[1]) not in self.knowns:
                        self.knowns[(ms[0], ms[1])] = {}
                    self.knowns[(ms[0], ms[1])][ms[2]] = 1
                    if (ms[0], ms[1], ms[2]) not in self.seens:
                        self.seens[(ms[0], ms[1], ms[2])] = (True, 1, [0, 1], [1])
                elif ms[3] == constants.BOUNDARY:
                    if (ms[0], ms[1]) not in self.knowns:
                        self.knowns[(ms[0], ms[1])] = {}
                    self.knowns[(ms[0], ms[1])][ms[2]] = 0 # 0 as frequency will mean boundary
            else: # turns after turn 1
                if (ms[0] + self.cur_x, ms[1] + self.cur_y, ms[2]) not in self.seens:
                    self.seens[(ms[0] + self.cur_x, ms[1] + self.cur_y, ms[2])] = [False, -1, [0], []]
                if ms[3] == constants.CLOSED:
                    self.seens[(ms[0] + self.cur_x, ms[1] + self.cur_y, ms[2])][3].append(turn)
                    continue
                elif ms[3] == constants.OPEN:
                    # already certain about frequency
                    if ((ms[0] + self.cur_x, ms[1] + self.cur_y, ms[2]) in self.seens) and (self.seens[(ms[0] + self.cur_x, ms[1] + self.cur_y, ms[2])][0] == True):
                        continue
                    # uncertain about frequency
                    else:
                        self.seens[(ms[0] + self.cur_x, ms[1] + self.cur_y, ms[2])][2].append(turn)
                        self.seens[(ms[0] + self.cur_x, ms[1] + self.cur_y, ms[2])][3].append(turn)
                elif ms[3] == constants.BOUNDARY:
                    if (ms[0] + self.cur_x, ms[1] + self.cur_y) not in self.knowns:
                        self.knowns[(ms[0] + self.cur_x, ms[1] + self.cur_y)] = {}
                    self.knowns[(ms[0] + self.cur_x, ms[1] + self.cur_y)][ms[2]] = 0 # 0 as frequency will mean boundary
                    self.seens[(ms[0] + self.cur_x, ms[1] + self.cur_y, ms[2])][0] = True
                    self.seens[(ms[0] + self.cur_x, ms[1] + self.cur_y, ms[2])][1] = 0

        # setting frequencies for doors that have been seen
        # TODO: adapt the data structure so that it tells us which turns a door was in sight for (to help with certainty); currently will never be certain, will just assume the smallest difference
        for (x, y, d), (certainty, freq, open, seen) in self.seens.items():
            # print ("x, y, d:", x, y, d)
            # print ("certainty, freq, open, seen:", certainty, freq, open, seen)
            if not certainty:
                smallestGap = self.findSmallestGap(open)
                # print ("smallestGap:", smallestGap)
                if freq == -1 or smallestGap < freq:
                    self.seens[(x, y, d)][1] = smallestGap
                    if (x, y) not in self.knowns:
                        self.knowns[(x, y)] = {}
                    self.knowns[(x, y)][d] = smallestGap
                # print ("self.seens[(x, y, d)]", self.seens[(x, y, d)])
            
        # create the final dictionary with all doors within the radius with LCMs. 
        # TODO: make this more efficient
        #for 


        print ("seens:", self.seens)
        print ("knowns:", self.knowns)
        return {}

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
        self.turn += 1

        self.setInfo(current_percept.maze_state, self.turn)
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
                            self.cur_x += 1
                            return constants.RIGHT
                if current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                                and maze_state[3] == constants.OPEN):
                            self.cur_x -= 1
                            return constants.LEFT
                if current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                                and maze_state[3] == constants.OPEN):
                            self.cur_y -= 1
                            return constants.UP
                if current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                                and maze_state[3] == constants.OPEN):
                            self.cur_y += 1
                            return constants.DOWN
                return constants.WAIT
            else:
                if current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                                and maze_state[3] == constants.OPEN):
                            self.cur_y -= 1
                            return constants.UP
                if current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                                and maze_state[3] == constants.OPEN):
                            self.cur_y += 1
                            return constants.DOWN
                if current_percept.end_x > 0 and direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                                and maze_state[3] == constants.OPEN):
                            self.cur_x += 1
                            return constants.RIGHT
                if current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                                and maze_state[3] == constants.OPEN):
                            self.cur_x -= 1
                            return constants.LEFT
                return constants.WAIT
        else:
            if direction[constants.LEFT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                            and maze_state[3] == constants.OPEN):
                        self.cur_x -= 1
                        return constants.LEFT
            if direction[constants.DOWN] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                            and maze_state[3] == constants.OPEN):
                        self.cur_y += 1
                        return constants.DOWN
            if direction[constants.RIGHT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                            and maze_state[3] == constants.OPEN):
                        self.cur_x += 1
                        return constants.RIGHT
            if direction[constants.UP] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                            and maze_state[3] == constants.OPEN):
                        self.cur_y -= 1
                        return constants.UP
            return constants.WAIT