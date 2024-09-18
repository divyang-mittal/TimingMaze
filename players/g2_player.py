import heapq
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
        self.path = []
        self.move_directions = []
        self.final_path = []
        self.final_move_directions = []
        self.start = (self.cur_x, self.cur_y)

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
    # create the final dictionary with all doors within the radius with LCMs. 
    # TODO: make this more efficient, make it account for boundaries (when self.knowns[(x, y)][d]
    def getDrone(self, maze_state):

        drone = {} # drone view around the current x, y, at radius r
        for door in maze_state:
            if (door[0], door[1]) not in drone:
                drone[(door[0], door[1])] = {constants.LEFT: -1, constants.UP: -1, constants.RIGHT: -1, constants.DOWN: -1}
        for (x, y) in drone: # these x, y are centered around 
            # print ("x, y:", x, y)
            # if (x, y) in self.knowns:
            #     print ("self.knowns(x, y):", self.knowns[(x, y)])
            f1 = 0
            f2 = 0

            # filling in the left edge of (x, y)
            if (x - 1, y) in drone:
                if drone[x, y][constants.LEFT] == -1 and drone [x - 1, y][constants.RIGHT] == -1:
                    if ((x + self.cur_x, y + self.cur_y) in self.knowns) and (constants.LEFT in self.knowns[(x + self.cur_x, y + self.cur_y)]):
                        f1 = self.knowns[(x + self.cur_x, y + self.cur_y)][constants.LEFT]
                    else:
                        f1 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    if ((x - 1 + self.cur_x, y + self.cur_y) in self.knowns) and (constants.RIGHT in self.knowns[(x - 1 + self.cur_x, y + self.cur_y)]):
                        f2 = self.knowns[(x - 1 + self.cur_x, y + self.cur_y)][constants.RIGHT]
                    else:
                        f2 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    f = self.lcm(f1, f2)
                    drone[x, y][constants.LEFT] = f
                    drone [x - 1, y][constants.RIGHT] = f
            else:
                if drone[x, y][constants.LEFT] == -1: 
                    if ((x + self.cur_x, y + self.cur_y) in self.knowns) and (constants.LEFT in self.knowns[(x + self.cur_x, y + self.cur_y)]): 
                        drone[x, y][constants.LEFT] = self.knowns[(x + self.cur_x, y + self.cur_y)][constants.LEFT]
                    else:
                        drone[x, y][constants.LEFT] = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)

            # filling in the up edge of (x, y)
            if (x, y - 1) in drone:
                if drone[x, y][constants.UP] == -1 and drone [x, y - 1][constants.DOWN] == -1:
                    if ((x + self.cur_x, y + self.cur_y) in self.knowns) and (constants.UP in self.knowns[(x + self.cur_x, y + self.cur_y)]): 
                        f1 = self.knowns[(x + self.cur_x, y + self.cur_y)][constants.UP]
                    else:
                        f1 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    if ((x + self.cur_x, y - 1 + self.cur_y) in self.knowns) and (constants.DOWN in self.knowns[(x + self.cur_x, y - 1 + self.cur_y)]):  
                        f2 = self.knowns[(x + self.cur_x, y - 1 + self.cur_y)][constants.DOWN]
                    else:
                        f2 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    f = self.lcm(f1, f2)
                    drone[x, y][constants.UP] = f
                    drone [x, y - 1][constants.DOWN] = f
            else:
                if drone[x, y][constants.UP] == -1: 
                    if ((x + self.cur_x, y + self.cur_y) in self.knowns) and (constants.UP in self.knowns[(x + self.cur_x, y + self.cur_y)]): 
                        drone[x, y][constants.UP] = self.knowns[(x + self.cur_x, y + self.cur_y)][constants.UP]
                    else:
                        drone[x, y][constants.UP] = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)

            if (x + 1, y) in drone:
                if drone[x, y][constants.RIGHT] == -1 and drone [x + 1, y][constants.LEFT] == -1:
                    if ((x + self.cur_x, y + self.cur_y) in self.knowns) and (constants.RIGHT in self.knowns[(x + self.cur_x, y + self.cur_y)]):
                        f1 = self.knowns[(x + self.cur_x, y + self.cur_y)][constants.RIGHT]
                    else:
                        f1 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    if ((x + 1 + self.cur_x, y + self.cur_y) in self.knowns) and (constants.LEFT in self.knowns[(x + 1 + self.cur_x, y + self.cur_y)]): 
                        f2 = self.knowns[(x + 1 + self.cur_x, y + self.cur_y)][constants.LEFT]
                    else:
                        f2 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    f = self.lcm(f1, f2)
                    drone[x, y][constants.RIGHT] = f
                    drone [x + 1, y][constants.LEFT] = f
            else:
                if drone[x, y][constants.RIGHT] == -1: 
                    if ((x + self.cur_x, y + self.cur_y) in self.knowns) and (constants.RIGHT in self.knowns[(x + self.cur_x, y + self.cur_y)]): 
                        drone[x, y][constants.RIGHT] = self.knowns[(x + self.cur_x, y + self.cur_y)][constants.RIGHT]
                    else:
                        drone[x, y][constants.RIGHT] = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)

            if (x, y + 1) in drone: 
                if drone[x, y][constants.DOWN] == -1 and drone [x, y + 1][constants.UP] == -1:
                    if ((x + self.cur_x, y + self.cur_y) in self.knowns) and (constants.DOWN in self.knowns[(x + self.cur_x, y + self.cur_y)]): 
                        f1 = self.knowns[(x + self.cur_x, y + self.cur_y)][constants.DOWN]
                    else:
                        f1 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    if ((x + self.cur_x, y + 1 + self.cur_y) in self.knowns) and (constants.UP in self.knowns[(x + self.cur_x, y + 1 + self.cur_y)]): 
                        f2 = self.knowns[(x + self.cur_x, y + 1 + self.cur_y)][constants.UP]
                    else:
                        f2 = self.rng.integers(low= 1, high=self.maximum_door_frequency, endpoint=True)
                    f = self.lcm(f1, f2)
                    drone[x, y][constants.DOWN] = f
                    drone [x, y + 1][constants.UP] = f
            else:
                if drone[x, y][constants.DOWN] == -1: 
                    if ((x + self.cur_x, y + self.cur_y) in self.knowns) and (constants.DOWN in self.knowns[(x + self.cur_x, y + self.cur_y)]): 
                        drone[x, y][constants.DOWN] = self.knowns[(x + self.cur_x, y + self.cur_y)][constants.DOWN]
                    else:
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

        # print("I am inside drone")
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
       # print("I am right before creating drone")
        drone = self.getDrone(maze_state)
       # print(drone)
       # print("I am right after creating drone")

        """print statements for debugging"""
        #print ("seens:", self.seens)

        # for k in drone:
        #     print (k)
        #     try:
        #         print ("knowns:", self.knowns[k])
        #     except: 
        #         print ("no knowns")
        #     print ("drone:", drone[k])

        return drone
    
    def move(self, current_percept) -> int:
        # print("Im inside move")
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

        """ Until you find target, find a random dest to move to, and return the move type. 
        Once you find destination, call A* again. 



        """
        self.turn = self.turn + 1
        # print("Im before drone")
        drone = self.setInfo(current_percept.maze_state, self.turn)
        # print("Im after drone")

        if current_percept.is_end_visible:
            # print("I am inside is end visible")
            if not self.final_move_directions:
                # print("creating path")
                # print((self.cur_x, self.cur_y))
                # print((current_percept.end_x, current_percept.end_y))
                final_path = self.a_star_search((self.cur_x, self.cur_y), (current_percept.end_x, current_percept.end_y), drone)
                print(final_path)

                for i in range(len(final_path) - 2):
                    # print("creating move dirctions")
                    self.final_move_directions.append(self.get_move_direction(final_path[i], final_path[i+1]))
                print(self.final_move_directions)

            if self.final_move_directions[0] == constants.LEFT:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == self.cur_x and maze_state[1] == self.cur_y and maze_state[2] == constants.LEFT
                            and maze_state[3] == constants.OPEN):
                        self.cur_x -= 1
                        self.final_move_directions.pop(0)
                        return constants.LEFT
                return constants.WAIT

            elif self.final_move_directions[0] == constants.RIGHT:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == self.cur_x and maze_state[1] == self.cur_y and maze_state[2] == constants.RIGHT
                            and maze_state[3] == constants.OPEN):
                        self.cur_x += 1
                        self.final_move_directions.pop(0)
                        return constants.RIGHT
                return constants.WAIT

            elif self.final_move_directions[0] == constants.UP:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == self.cur_x and maze_state[1] == self.cur_y and maze_state[2] == constants.UP
                            and maze_state[3] == constants.OPEN):
                        self.cur_y +=1 
                        self.final_move_directions.pop(0)
                        return constants.UP
                return constants.WAIT

            elif self.final_move_directions[0] == constants.DOWN:
                # print("I am inside")

                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == self.cur_x and maze_state[1] == self.cur_y and maze_state[2] == constants.DOWN
                            and maze_state[3] == constants.OPEN):
                        
                        self.final_move_directions.pop(0)
                        self.cur_y -= 1
                        return constants.DOWN
                return constants.WAIT

            return constants.WAIT
                
        else:
            # print("Im before move directions")
            if not self.move_directions:
                # print("generate goal")
                (x, y) = self.generate_goal()
                path = self.a_star_search((self.cur_x, self.cur_y), (x,y), drone)
                # print("Im after move directions" + path)
                # print(x,y)

                for i in range(len(path) - 2):
                    self.move_directions.append(self.get_move_direction(path[i], path[i+1]))

            if self.move_directions[0] == constants.LEFT:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == self.cur_x and maze_state[1] == self.cur_y and maze_state[2] == constants.LEFT
                            and maze_state[3] == constants.OPEN):
                        self.cur_x -= 1
                        self.final_move_directions.pop(0)
                        return constants.LEFT
                return constants.WAIT
            elif self.move_directions[0] == constants.RIGHT:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == self.cur_x and maze_state[1] == self.cur_y and maze_state[2] == constants.RIGHT
                            and maze_state[3] == constants.OPEN):
                        self.cur_x += 1
                        self.final_move_directions.pop(0)
                        return constants.RIGHT
                return constants.WAIT

            elif self.move_directions[0] == constants.UP:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == self.cur_x and maze_state[1] == self.cur_y and maze_state[2] == constants.UP
                            and maze_state[3] == constants.OPEN):
                        self.cur_y +=1 
                        self.final_move_directions.pop(0)
                        return constants.UP
                return constants.WAIT

            elif self.move_directions[0] == constants.DOWN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == self.cur_x and maze_state[1] == self.cur_y and maze_state[2] == constants.DOWN
                            and maze_state[3] == constants.OPEN):
                        self.cur_y -= 1
                        self.final_move_directions.pop(0)
                        return constants.DOWN
                return constants.WAIT

        return 0

    def generate_goal(self):
        x = y = -float('inf')
        while not self.is_valid(x, y): 
                # Generate random integers between -100 and 100
                #print(x, y)
                x = self.rng.integers(-99, 99) 
                y = self.rng.integers(-99, 99) 
                goal = (x,y)
        #print(goal)
        return goal



    def is_valid(self, row, col):
        # If cell lies out of bounds
        if row < 0 or col < 0 or row >= constants.map_dim or col >= constants.map_dim:
            return False

        # Otherwise
        return True

    def get_move_direction(self, current_position, next_position):
        """Determine the move direction from current position to next position
            Returns:
                int: Move direction
                    LEFT = 0
                    UP = 1
                    RIGHT = 2
                    DOWN = 3
        """
        dx = next_position[0] - current_position[0]
        dy = next_position[1] - current_position[1]
        # print("Inside move dir")
        # print(dx , dy)
        
        if dx == -1 and dy == 0:
            return constants.LEFT  # LEFT
        elif dx == 0 and dy == -1:
            return constants.UP  # UP
        elif dx == 1 and dy == 0:
            return constants.RIGHT  # RIGHTx
        elif dx == 0 and dy == 1:
            return constants.DOWN  # DOWN
        else:
            return constants.WAIT  # WAIT or invalid move

    def a_star_search(self, start, goal, LCM_map):
        # print("Im inside A*")
        # LCM_map: (x, y) -> {LEFT: #, ...}

        # Open set represented as a priority queue with (f_score, node)
        open_set = []
        heapq.heappush(open_set, (0, start, self.turn))

        # Maps nodes to their parent node
        came_from = {} # (x, y) -> (x, y)

        # Cost from start to a node
        g_score = {start: 0} # (x, y) -> int

        vis = set({})
        while open_set:
            # print("Im inside open set")
            # Get the node in open_set with the lowest f_score
            current_f_score, current, current_turn = heapq.heappop(open_set)
            vis.add(current)

            # Check if we have reached the goal
            if current == goal:
                # print("i am inside path found")
                print(self.reconstruct_path(came_from, current))
                return self.reconstruct_path(came_from, current)

            # Explore neighbors
            moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            for i, move in enumerate(moves):
                # print("Im inside enumerate move")
                neighbor = (current[0] + move[0], current[1] + move[1])
                tentative_g_score = g_score[current] + LCM_map[current][i] - current_turn % LCM_map[current][i]

                # If this path to neighbor is better than any previous one
                if (neighbor not in g_score or tentative_g_score < g_score[neighbor]) and neighbor not in vis:
                    # print("Im inside random neighbour")
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    # f_score = tentative_g_score + self.heuristic_(neighbor, goal)
                    f_score = tentative_g_score + self.heuristic_manhatten(neighbor, goal, current_turn, LCM_map)
                    # print(f_score)
                    heapq.heappush(open_set, (f_score, neighbor, current_turn + 1))

        # No path found
        print("NO PATH FOUND")
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
    
    def heuristic_manhatten(self, current, goal, current_turn, LCM_map):
        moves = [[-1, 0], [0, -1], [1, 0], [0, 1]] # [L, U, R, D]

        dir = self.path_to_directions(self.manhattan_path(current, goal))

        # print(dir)

        cost = current_turn
        for i in dir:
            # print("i" + str(i))
            LCM = LCM_map[tuple(current)][i]
            # print("LCM" + str(LCM))
            if (cost + 1) % LCM == 0:
                # print("if" + str(cost))
                cost = cost + 1
                # print("if" + str(cost))
            else:
                # print("else" + str(cost))
                cost = cost + LCM - (cost % LCM)
                # print("else" + str(cost))
            
            current = list(current)
            # print(type(current))
            # print(current[0] + moves[i][0])
            current[0] = current[0] + moves[i][0]
            # print(current)
            current[1] = current[1] + moves[i][1]
            # print(current)
            # print("==============")
            # print(i)
            # print(cost)
            # print(current)
        cost = cost + 1
        # print("returning cost: " + str(cost))
        return cost
    
    def manhattan_path(self, start, goal):
        """
        Generates the Manhattan path between the given start and goal coordinates on a grid.

        Args:
            start: A tuple representing the x and y coordinates of the starting point.
            goal: A tuple representing the x and y coordinates of the goal point.

        Returns:
            A list of tuples representing the coordinates of the points on the Manhattan path.
        """

        path = []
        x_start, y_start = start[0], start[1]
        x_goal, y_goal = goal[0], goal[1]

        cnt = 0

        while x_start != x_goal and y_start != y_goal and cnt < 4:
            if x_start < x_goal:
                x_start += 1
            else:
                x_start -= 1
            path.append((x_start, y_start))
            cnt = cnt + 1

            if y_start < y_goal:
                y_start += 1
            else:
                y_start -= 1
            path.append((x_start, y_start))
            cnt = cnt + 1
        
        while cnt < 4 and x_start != x_goal:
            if x_start < x_goal:
                x_start += 1
            else:
                x_start -= 1
            path.append((x_start, y_start))
            cnt = cnt + 1

        while cnt < 4 and y_start != y_goal:
            if y_start < y_goal:
                y_start += 1
            else:
                y_start -= 1
            path.append((x_start, y_start))
            cnt = cnt + 1

        # # Move along the x-axis first
        # while x_start != x_goal:
        #     if x_start < x_goal:
        #         x_start += 1
        #     else:
        #         x_start -= 1
        #     path.append((x_start, y_start))

        # # Move along the y-axis
        # while y_start != y_goal:
        #     if y_start < y_goal:
        #         y_start += 1
        #     else:
            #     y_start -= 1
            # path.append((x_start, y_start))

        # print(path)

        return path
    
    def path_to_directions(self, path):
        """
        Converts a path represented as a list of coordinates to a list of directions (LEFT, RIGHT, UP, DOWN).

        Args:
            path: A list of tuples representing the coordinates of the points on the path.

        Returns:
            A list of strings representing the directions between each pair of points on the path.
        """

        directions = []
        for i in range(1, len(path)):
            x1, y1 = path[i - 1][0], path[i - 1][1] 
            x2, y2 = path[i][0], path[i][1]
            if x1 < x2:
                directions.append(constants.RIGHT)
            elif x1 > x2:
                directions.append(constants.LEFT)
            elif y1 < y2:
                directions.append(constants.DOWN)
            else:
                directions.append(constants.UP)

        # print(directions)

        return directions
