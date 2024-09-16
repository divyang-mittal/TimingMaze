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
    
    # setting frequencies for doors that have been seen
    # TODO: adapt the data structure so that it tells us which turns a door was in sight for (to help with certainty); currently will never be certain, will just assume the smallest difference
    def setFreqs(self):
        for (x, y, d), (certainty, freq, open, seen) in self.seens.items():
            if not certainty:
                smallestGap = self.findSmallestGap(open)
                if smallestGap == -1: 
                    continue
                elif freq == -1 or smallestGap < freq:
                    self.seens[(x, y, d)][1] = smallestGap
                    if (x, y) not in self.knowns:
                        self.knowns[(x, y)] = {}
                    self.knowns[(x, y)][d] = smallestGap

    # create the final dictionary with all doors within the radius with LCMs. 
    # TODO: make this more efficient
    def getDrone(self, maze_state):

        drone = {} # drone view around the current x, y, at radius r
        for door in maze_state:
            if (door[0], door[1]) not in drone:
                drone[(door[0], door[1])] = {constants.LEFT: -1, constants.UP: -1, constants.RIGHT: -1, constants.DOWN: -1}
        for (x, y) in drone: 
            f1 = 0
            f2 = 0
            if (x - 1, y) in drone:
                if drone[x, y][constants.LEFT] == -1 and drone [x - 1, y][constants.RIGHT] == -1:
                    try: 
                        f1 = self.knowns[(x, y)][constants.LEFT]
                    except:
                        f1 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    try: 
                        f2 = self.knowns[(x - 1, y)][constants.RIGHT]
                    except:
                        f2 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    f = self.lcm(f1, f2)
                    drone[x, y][constants.LEFT] = f
                    drone [x - 1, y][constants.RIGHT] = f
            else:
                if drone[x, y][constants.LEFT] == -1: 
                    try: 
                        drone[x, y][constants.LEFT] = self.knowns[(x, y)][constants.LEFT]
                    except:
                        drone[x, y][constants.LEFT] = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)

            if (x, y - 1) in drone:
                if drone[x, y][constants.UP] == -1 and drone [x, y - 1][constants.DOWN] == -1:
                    try: 
                        f1 = self.knowns[(x, y)][constants.UP]
                    except:
                        f1 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    try: 
                        f2 = self.knowns[(x, y - 1)][constants.DOWN]
                    except:
                        f2 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    f = self.lcm(f1, f2)
                    drone[x, y][constants.UP] = f
                    drone [x, y - 1][constants.DOWN] = f
            else:
                if drone[x, y][constants.UP] == -1: 
                    try: 
                        drone[x, y][constants.UP] = self.knowns[(x, y)][constants.UP]
                    except:
                        drone[x, y][constants.UP] = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)

            if (x + 1, y) in drone:
                if drone[x, y][constants.RIGHT] == -1 and drone [x + 1, y][constants.LEFT] == -1:
                    try: 
                        f1 = self.knowns[(x, y)][constants.RIGHT]
                    except:
                        f1 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    try: 
                        f2 = self.knowns[(x + 1, y)][constants.LEFT]
                    except:
                        f2 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    f = self.lcm(f1, f2)
                    drone[x, y][constants.RIGHT] = f
                    drone [x + 1, y][constants.LEFT] = f
            else:
                if drone[x, y][constants.RIGHT] == -1: 
                    try: 
                        drone[x, y][constants.RIGHT] = self.knowns[(x, y)][constants.RIGHT]
                    except:
                        drone[x, y][constants.RIGHT] = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)

            if (x, y + 1) in drone: 
                if drone[x, y][constants.DOWN] == -1 and drone [x, y + 1][constants.UP] == -1:
                    try: 
                        f1 = self.knowns[(x, y)][constants.DOWN]
                    except:
                        f1 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    try: 
                        f2 = self.knowns[(x, y + 1)][constants.UP]
                    except:
                        f2 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    f = self.lcm(f1, f2)
                    drone[x, y][constants.DOWN] = f
                    drone [x, y + 1][constants.UP] = f
            else:
                if drone[x, y][constants.DOWN] == -1: 
                    try: 
                        drone[x, y][constants.DOWN] = self.knowns[(x, y)][constants.DOWN]
                    except:
                        drone[x, y][constants.DOWN] = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
        return drone

    
            
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

        # gathers info from the maze_state and populates self.seens and self.knowns
        for ms in maze_state:
            if self.turn == 1:
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
                    
        self.setFreqs()
        drone = self.getDrone(maze_state)

        """print statements for debugging"""
        # print ("seens:", self.seens)

        # for k in drone:
        #     print (k)
        #     try:
        #         print ("knowns:", self.knowns[k])
        #     except: 
        #         print ("no knowns")
        #     print ("drone:", drone[k])

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
        drone = self.setInfo(current_percept.maze_state, self.turn)
        return 0

    def a_star_search(self, start, goal, LCM_map):
        # LCM_map: (x, y) -> {LEFT: #, ...)}

        # Open set represented as a priority queue with (f_score, node)
        open_set = []
        heapq.heappush(open_set, (0, start, self.turn))

        # Maps nodes to their parent node
        came_from = {} # (x, y) -> (x, y)

        # Cost from start to a node
        g_score = {start: 0} # (x, y) -> int

        while open_set:
            # Get the node in open_set with the lowest f_score
            current_f_score, current, current_turn = heapq.heappop(open_set)

            # Check if we have reached the goal
            if current == goal:
                return self.reconstruct_path(came_from, current)

            # Explore neighbors
            moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            for i, move in enumerate(moves):
                neighbor = (current[0] + move[0], current[1] + move[1])
                tentative_g_score = g_score[current] + LCM_map[current][i] - current_turn % LCM_map[current][i]

                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor, current_turn + 1))

        # No path found
        return None
    
    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def heuristic(self, current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])
