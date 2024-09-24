import heapq
import numpy as np
import logging

import constants
from timing_maze_state import TimingMazeState

class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger, precomp_dir: str, maximum_door_frequency: int, radius: int) -> None:
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

        # initializing cur_x, cur_y to start x, y
        self.cur_x = 0 
        self.cur_y = 0
        self.start = (self.cur_x, self.cur_y)
        # tracking turn #
        self.turn = 0 
        # x, y in seens and knowns is centered around start x, y
        self.seens = dict() # dictionary w/ kv - (x, y, d): (False (uncertain)/True (certain), assumed freq, [list of turns at which x, y, d was open], [list of turns at which x, y, d could be seen])
        self.knowns = dict() # dictionary w/ kv - (x, y): {0: freq(L), 1: freq(U), 2: freq(R), 3: freq(D)}, freq = -1 if unknown
        self.boundaryCoordinates = list

        self.to_end_directions = []
        self.move_i = 0

        self.move_directions = []
        self.final_path = []
        self.final_move_directions = []

    """
    helper functions for information collection

    findFreq(seenOpen: list)
    - finds the smallest gap between turns when a door is known to have been seen open.

    determineCertainty (freq: int, openPair: tuple, seen: list)
    - determines the certainty of a frequency by looking at the seen doors between the two open doors
        - if there are no unseen doors or if none of the unseen doors could introduce a lower frequency, then we can be certain

    gcd (x: int, y: int) 
    - finds the gcd of two numbers using euclidean algorithm 

    lcm(x: int, y: int)
    - finds the lcm of two numbers by dividing (x * y) by gcd (x, y)
    """
        
    @staticmethod
    def findFreq(seenOpen: list) -> int:
        if len(seenOpen) < 2:
            return -1
        gap = seenOpen[1] - seenOpen[0]
        freqDict = {gap: [(0, 1)]}
        for i in range(len(seenOpen) - 1):
            if seenOpen[i + 1] - seenOpen[i] < gap:
                gap = seenOpen[i + 1] - seenOpen[i]
                if gap in freqDict:
                    freqDict[gap].append((i, i + 1))
                else:
                    freqDict[gap] = [(i, i + 1)]
                
        return gap, freqDict
    
    @staticmethod
    def determineCertainty(freq: int, openPair: tuple, seen: list) -> bool:
        
        i = seen.index(openPair[0])
        j = seen.index(openPair[1])
        unseen = []
        k = 1
        l = 1
        # if we have seen every door between the pair, we can be certain
        if (j - i) == freq:
            return True
        
        # created unseen, a list of turns this door was not seen. 
        while i + k < j and openPair[0] + l < openPair[1]:
            if seen[i + k] == openPair[0] + l:
                k += 1
                l += 1
                continue
            else:
                unseen.append(openPair[0] + k)
                l += 1

        # if the gcd of pair[0], pair[1] and unseen is a factor of freq, we can be uncertain. 
        for turn in unseen: 
            gcd = Player.gcd(openPair[0], turn)
            if freq % gcd == 0:
                return False
        # if the above doesn't return False, we can be certain
        return True
    
    @staticmethod
    def gcd(x: int, y: int) -> int:
        if x < y:
            temp = x 
            x = y
            y = temp
        if y == 0:
            return x
        return Player.gcd(y, x % y)
    
    @staticmethod
    def lcm(x: int, y: int):
        return ((x * y) / Player.gcd(x, y))
    
    """
    information collection notes & functions
    
    maze_state: list (
                    tuple (
                        0: x-coordinate,
                        1: y-coordinate, 
                        2: door at x, y (constants.LEFT, constants.UP, constants.RIGHT, constants.DOWN), 
                        3: door status (constants.CLOSED, constants.OPEN, constants.BOUNDARY)
                        )
                    )
    self.seens: dictionary 
        -  (x, y, d): tuple (
                        0: certainty (False/True), 
                        1: assumed_freq,
                        2: turns_open (list of turns when (x, y, d) was open),
                        3: turns_seen (list of turns when (x, y, d) was seen)
                        )
        - centered around start_x, start_y
    
    self.knowns: dictionary
        - (x, y): dict {
                    constants.LEFT: freq(LEFT), 
                    constants.UP: freq(UP),
                    constants.RIGHT: freq(RIGHT),
                    constants.DOWN: freq(DOWN)
                    }
        - frequencies will be 0 if boundary
        - frequencies will be -1 if unknown
        - centered around start_x, start_y

    setSeensKnowns(self, maze_state: TimingMazeState.maze_state)
        - utilizes the information in maze_state and the turn to populate self.seens and self.knowns
        - on turn 1, every door within the radius that is opened will be assigned a frequency of 1

    setFreqs(self)
        - utilizes the smallest gap between turns when a door was open to determine an assumed frequency
        - determines certainty regarding the freqs

    """

    def setSeensKnowns(self, maze_state) -> None:
        for ms in maze_state:
            x = ms[0]
            y = ms[1]
            door = ms[2]
            status = ms[3]
            
            if self.turn == 1:
                if status == constants.CLOSED: 
                    continue
                elif status == constants.OPEN:
                    if (x, y) not in self.knowns:
                        self.knowns[(x, y)] = {}
                        self.knowns[(x, y)][door] = 1
                    if (x, y, door) not in self.seens:
                        self.seens[(x, y, door)] = (True, 1, [0, 1], [1])
                elif status == constants.BOUNDARY:
                    if (x, y) not in self.knowns:
                        self.knowns[(x, y)] = {}
                    self.knowns[(x, y)][door] = 0
                    self.boundaryCoordinates.append((x, y))
            else: # turns after turn 1
                x = ms[0] + self.cur_x
                y = ms[1] + self.cur_y
                if (x, y, door) not in self.seens:
                    self.seens[(x, y, door)] = [False, -1, [0], []]
                # append the turn # to the list of turns when the door has been seen
                if door == constants.CLOSED:
                    self.seens[(x, y, door)][3].append (self.turn)
                elif door == constants.OPEN:
                    # if uncertain about frequency
                    if ((x, y, door) in self.seens) and (not self.seens[(x, y, door)][0]):
                        self.seens[(x, y, door)][2].append (self.turn)
                        self.seens[(x, y, door)][3].append (self.turn)
                elif door == constants.BOUNDARY:
                    if (x, y) not in self.knowns: 
                        self.knowns[(x, y)] = {}
                    self.knowns[(x, y)][door] = 0
                    self.seens[(x, y, door)] = [True, 0, [], []]
                    self.boundaryCoordinates.append((x, y))
        return 
    
    def setFreqs(self) -> None:
        for (x, y, d), (certainty, assumed_freq, turns_open, turns_seen) in self.seens.items():
            if not certainty:
                smallestGap = self.findFreq(turns_open)
                freq = smallestGap[0]
                # first pair of turns that (x, y, d) had the frequency as their gap
                openPair = smallestGap[1][freq][0]
                # adjusting certainty
                self.seens[(x, y, d)][0] = self.determineCertainty(freq, openPair, turns_seen)

                if freq == -1: 
                    continue
                elif assumed_freq == -1 or freq < assumed_freq:
                    self.seens[(x, y, d)][1] = freq
                    if (x, y) not in self.knowns:
                        self.knowns[(x, y)] = {}
                    self.knowns[(x, y)][d] = freq
                
    # def getDrone(self, maze_state) -> None:
    #     drone = {} # drone view around the cur_x, cur_y, at radius r
    #     doors = {constants.LEFT: -1, constants.UP: -1, constants.RIGHT: -1, constants.DOWN: -1}

    #     # part 1: add dictionary key value pairs for each door in maze_state (all doors within radius r, centered at cur_x, cur_y) 
    #     # part 2: fill in doors dictionary (open/closed status of surrounding doors)
    #     for (x, y, d, s) in maze_state:
    #         # part 1
    #         if (x, door[1]) not in drone:
    #             drone[(door[0], door[1])] = {constants.LEFT: -1, constants.UP: -1, constants.RIGHT: -1, constants.DOWN: -1}
    #         # part 2
    #         # fill in the values of the doors of cur_x, cur_y before adjusting for the doors that touch them 
    #         if door[0] == self.cur_x and door[1] == self.cur_y: 
    #             print ("(cur_x, cur_y):", door)
    #             if doors[door[2]] == -1:
    #                 doors[door[2]] = door[3]
    #             elif doors[door[2]] == 2 and door[3] != 2:
    #                 doors[door[2]] = door[3]