import heapq
import numpy as np
import logging

from constants import OPEN, CLOSED, BOUNDARY, LEFT, RIGHT, UP, DOWN, WAIT, CLOSED_PROB, map_dim
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

        ''' drone global variables '''
        # x, y in seens and knowns is centered around start x, y
        self.seens = dict() # dictionary w/ kv - (x, y, d): (False (uncertain)/True (certain), assumed freq, [list of turns at which x, y, d was open], [list of turns at which x, y, d could be seen])
        self.knowns = dict() # dictionary w/ kv - (x, y): {0: freq(L), 1: freq(U), 2: freq(R), 3: freq(D)}, freq = -1 if unknown
        ''' boundary global variables '''
        self.boundaryCoordinates = list
        # boundary (x, y) in relation to the start coordinate
        self.LRB = -100 # left-right (x of left edge)
        self.UDB = -100 # up-down (y of up edge)

        self.start_global_x = 0
        self.start_global_y = 0
    
        ''' move global variables '''
        self.path = []
        self.move_directions = []
        self.final_path = []
        self.next_move = 0
        self.start = (self.cur_x, self.cur_y)
        self.curr_stationary_moves = 0

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

    setBoundaryCoords(self, x: int, y: int, d: int)
    - gets called every time a boundary is encountered and updates global variables
    """
        
    @staticmethod
    def findFreq(seenOpen: list) -> int:
        if len(seenOpen) < 2:
            return -1
        gcd = Player.gcd(seenOpen[1], seenOpen[0])
        freqDict = {gcd: [(0, 1)]}
        for i in range(len(seenOpen) - 1):
            tempGCD = Player.gcd(seenOpen[i + 1], seenOpen[i])
            if tempGCD > 1 and tempGCD < gcd:
                gcd = tempGCD
                if gcd in freqDict:
                    freqDict[gcd].append((i, i + 1))
                else:
                    freqDict[gcd] = [(i, i + 1)]
                
        return gcd, freqDict
    
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
    
    def setBoundaryCoords(self, x: int, y: int, d: int) -> None:
        if self.LRB != -100 and self.UDB != -100:
            self.start_global_x = 0 - self.LRB
            self.start_global_y = 0 - self.UDB
            return
        self.boundaryCoordinates.append((x, y))
        if d == LEFT:
            self.LRB = x
        elif d == RIGHT:
            self.LRB = x - map_dim + 1
        elif d == UP:
            self.UDB = y
        elif d == DOWN:
            self.UDB = y - map_dim + 1
        return
    
    """
    information collection notes & functions
    
    maze_state: list (
                    tuple (
                        0: x-coordinate,
                        1: y-coordinate, 
                        2: door at x, y (LEFT, UP, RIGHT, DOWN), 
                        3: door status (CLOSED, OPEN, BOUNDARY)
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
                    LEFT: freq(LEFT), 
                    UP: freq(UP),
                    RIGHT: freq(RIGHT),
                    DOWN: freq(DOWN)
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

    getDrone(self, maze_state)
        - creates drone which is a dictionary of all the x, y in a radius r to cur_x, cur_y and each doors assumed frequency
            - utilizes LCM and when needed, random frequencies 
        - creates doors which is a dictionary of the status (open/closed/boundary) of the 4 edges surrounding the cur_x, cur_y. checks that both doors are open on each edge.

    setInfo(self, maze_state)
        - calls setSeensKnowns, setFreqs and getDrone to return info (tuple(drone, doors))
    """

    def setSeensKnowns(self, maze_state) -> None:
        for ms in maze_state:
            x = ms[0]
            y = ms[1]
            door = ms[2]
            status = ms[3]
            
            if self.turn == 1:
                if status == CLOSED: 
                    continue
                elif status == OPEN:
                    if (x, y) not in self.knowns:
                        self.knowns[(x, y)] = {}
                        self.knowns[(x, y)][door] = 1
                    if (x, y, door) not in self.seens:
                        self.seens[(x, y, door)] = (True, 1, [0, 1], [1])
                elif status == BOUNDARY:
                    if (x, y) not in self.knowns:
                        self.knowns[(x, y)] = {}
                    self.knowns[(x, y)][door] = 0
                    self.setBoundaryCoords(x, y, door)
            else: # turns after turn 1
                x = ms[0] + self.cur_x
                y = ms[1] + self.cur_y
                if (x, y, door) not in self.seens:
                    self.seens[(x, y, door)] = [False, -1, [0], []]
                # append the turn # to the list of turns when the door has been seen
                if door == CLOSED:
                    self.seens[(x, y, door)][3].append (self.turn)
                elif door == OPEN:
                    # if uncertain about frequency
                    if ((x, y, door) in self.seens) and (not self.seens[(x, y, door)][0]):
                        self.seens[(x, y, door)][2].append (self.turn)
                        self.seens[(x, y, door)][3].append (self.turn)
                elif door == BOUNDARY:
                    if (x, y) not in self.knowns: 
                        self.knowns[(x, y)] = {}
                    self.knowns[(x, y)][door] = 0
                    self.seens[(x, y, door)] = [True, 0, [], []]
                    self.setBoundaryCoords(x, y, door)
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
          
    def getDrone(self, maze_state) -> tuple:
        drone = {} # drone view around the cur_x, cur_y, at radius r
        doors = {LEFT: -1, UP: -1, RIGHT: -1, DOWN: -1}

        # setting a weighted probability scheme for randomizing unknown frequencies 
        probs = [CLOSED_PROB] + ((self.maximum_door_frequency) * [0])
        for i in range (1, len(probs)):
            inc = (self.maximum_door_frequency / 2 - i + 0.5) * (self.maximum_door_frequency / 2000)
            probs[i] = (1 - CLOSED_PROB)/self.maximum_door_frequency - inc

        # part 1: add dictionary key value pairs for each door in maze_state (all doors within radius r, centered at cur_x, cur_y) 
        # part 2: fill in doors dictionary (open/closed status of surrounding doors)
        for (x, y, d, s) in maze_state:
        # part 1
            if (x, y) not in drone:
                drone[(x, y)] = {LEFT: -1, UP: -1, RIGHT: -1, DOWN: -1}
        # part 2
            # fill in the values of the doors of cur_x, cur_y before adjusting for the doors that touch them 
            if x == self.cur_x and y == self.cur_y: 
                # print ("(cur_x, cur_y):", (x, y, d, s))
                if doors[d] == -1:
                    doors[d] = s
                elif doors[d] == OPEN and s != OPEN:
                    doors[d] = s
            # doors[LEFT], touches (x - 1, y), RIGHT
            elif x == self.cur_x - 1 and y == self.cur_y and d == RIGHT:
                # print ("right door on the left:", (x, y, d, s))
                if doors[LEFT] == -1:
                    doors[LEFT] = s
                elif doors[LEFT] == OPEN and s != OPEN:
                    doors[LEFT] = s
            # doors[UP], touches (x, y - 1), DOWN
            elif x == self.cur_x and y == self.cur_y - 1 and d == DOWN:
                # print ("bottom door on the top:", (x, y, d, s))
                if doors[UP] == -1:
                    doors[UP] = s
                elif doors[UP] == OPEN and s != OPEN:
                    doors[UP] = s
            # doors[RIGHT], touches (x + 1, y), LEFT
            elif x == self.cur_x + 1 and y == self.cur_y and d == LEFT:
                # print ("left door on the right:", (x, y, d, s))
                if doors[RIGHT] == -1:
                    doors[RIGHT] = s
                elif doors[RIGHT] == OPEN and s != OPEN:
                    doors[RIGHT] = s
            # doors[DOWN], touches (x, y + 1), UP
            elif x == self.cur_x and y == self.cur_y + 1 and d == UP:
                # print ("top door on the bottom:", (x, y, d, s))
                if doors[DOWN] == -1:
                    doors[DOWN] = s
                elif doors[DOWN] == OPEN and s != OPEN:
                    doors[DOWN] = s

        # create a frequency dictionary that is centered around cur_x, cur_y
        for (x, y) in drone:
            f1 = 0
            f2 = 0
            adjX = x + self.cur_x
            adjY = y + self.cur_y

            # LEFT edge of (x, y)
            if (x - 1, y) in drone:
                # if neither freq has been set
                if drone[(x, y)][LEFT] == -1 and drone[(x - 1, y)][RIGHT] == -1:
                    if ((adjX, adjY) in self.knowns) and (LEFT in self.knowns[(adjX, adjY)]):
                        f1 = self.knowns[(adjX, adjY)][LEFT]
                    else:
                        f1 = np.random.choice(a=self.maximum_door_frequency + 1, p=probs)
                    if ((adjX - 1, adjY) in self.knowns) and (RIGHT in self.knowns[(adjX - 1, adjY)]):
                        f2 = self.knowns[(adjX - 1, adjY)][RIGHT]
                    else:
                        f2 = np.random.choice(a=self.maximum_door_frequency + 1, p=probs)
                    
                    f = 0 
                    if f1 != 0 and f2 != 0:
                        f = self.lcm(f1, f2)
                    drone[(x, y)][LEFT] = f
                    drone[(x - 1, y)][RIGHT] = f
            else:
                if drone[(x,y)][LEFT] == -1:
                    if ((adjX, adjY) in self.knowns) and (LEFT in self.knowns[(adjX, adjY)]):
                        drone[(x, y)][LEFT] = self.knowns[(adjX, adjY)][LEFT]
                    else:
                        drone[(x, y)][LEFT] = np.random.choice(a=self.maximum_door_frequency + 1, p=probs)

            # UP edge of (x, y)
            if (x, y - 1) in drone:
                if drone[(x, y)][UP] == -1 and drone[(x, y - 1)][DOWN] == -1:
                    if ((adjX, adjY) in self.knowns) and (UP in self.knowns[(adjX, adjY)]):
                        f1 = self.knowns[(adjX, adjY)][UP]
                    else:
                        f1 = np.random.choice(a=self.maximum_door_frequency + 1, p=probs)
                    if ((adjX, adjY - 1) in self.knowns) and (DOWN in self.knowns[(adjX, adjY - 1)]):
                        f2 = self.knowns[(adjX, adjY - 1)][DOWN]
                    else:
                        f2 = np.random.choice(a=self.maximum_door_frequency + 1, p=probs)
                    
                    f = 0 
                    if f1 != 0 and f2 != 0:
                        f = self.lcm(f1, f2)
                    drone[(x, y)][UP] = f
                    drone[(x, y - 1)][DOWN] = f
            else:
                if drone[(x,y)][UP] == -1:
                    if ((adjX, adjY) in self.knowns) and (UP in self.knowns[(adjX, adjY)]):
                        drone[(x, y)][UP] = self.knowns[(adjX, adjY)][UP]
                    else:
                        drone[(x, y)][UP] = np.random.choice(a=self.maximum_door_frequency + 1, p=probs)

            # RIGHT edge of (x, y)
            if (x + 1, y) in drone:
                # if neither freq has been set
                if drone[(x, y)][RIGHT] == -1 and drone[(x + 1, y)][LEFT] == -1:
                    if ((adjX, adjY) in self.knowns) and (RIGHT in self.knowns[(adjX, adjY)]):
                        f1 = self.knowns[(adjX, adjY)][RIGHT]
                    else:
                        f1 = np.random.choice(a=self.maximum_door_frequency + 1, p=probs)
                    if ((adjX + 1, adjY) in self.knowns) and (LEFT in self.knowns[(adjX + 1, adjY)]):
                        f2 = self.knowns[(adjX + 1, adjY)][LEFT]
                    else:
                        f2 = np.random.choice(a=self.maximum_door_frequency + 1, p=probs)
                    
                    f = 0 
                    if f1 != 0 and f2 != 0:
                        f = self.lcm(f1, f2)
                    drone[(x, y)][RIGHT] = f
                    drone[(x + 1, y)][LEFT] = f
            else:
                if drone[(x,y)][RIGHT] == -1:
                    if ((adjX, adjY) in self.knowns) and (RIGHT in self.knowns[(adjX, adjY)]):
                        drone[(x, y)][RIGHT] = self.knowns[(adjX, adjY)][RIGHT]
                    else:
                        drone[(x, y)][RIGHT] = np.random.choice(a=self.maximum_door_frequency + 1, p=probs)

            # DOWN edge of (x, y)
            if (x, y + 1) in drone:
                if drone[(x, y)][DOWN] == -1 and drone[(x, y + 1)][UP] == -1:
                    if ((adjX, adjY) in self.knowns) and (DOWN in self.knowns[(adjX, adjY)]):
                        f1 = self.knowns[(adjX, adjY)][DOWN]
                    else:
                        f1 = np.random.choice(a=self.maximum_door_frequency + 1, p=probs)
                    if ((adjX, adjY + 1) in self.knowns) and (UP in self.knowns[(adjX, adjY + 1)]):
                        f2 = self.knowns[(adjX, adjY + 1)][UP]
                    else:
                        f2 = np.random.choice(a=self.maximum_door_frequency + 1, p=probs)
                    
                    f = 0 
                    if f1 != 0 and f2 != 0:
                        f = self.lcm(f1, f2)
                    drone[(x, y)][DOWN] = f
                    drone[(x, y + 1)][UP] = f
            else:
                if drone[(x,y)][DOWN] == -1:
                    if ((adjX, adjY) in self.knowns) and (DOWN in self.knowns[(adjX, adjY)]):
                        drone[(x, y)][DOWN] = self.knowns[(adjX, adjY)][DOWN]
                    else:
                        drone[(x, y)][DOWN] = np.random.choice(a=self.maximum_door_frequency + 1, p=probs)
        return (drone, doors)

    def setInfo (self, maze_state) -> dict:
        self.setSeensKnowns(maze_state=maze_state)
        self.setFreqs()
        info  = self.getDrone(maze_state=maze_state)
        return info

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
        - Until you find target, find a random dest to move to, and return the move type. 
        - Once you find destination, call A* again. 
        """
        self.turn = self.turn + 1

        # Waiting for 100 turns to gain LCM information 
        # if self.turn <= 100:
        #     return WAIT

        # get LCM map & open doors information
        # print ("error in drone")
        drone, doors = self.setInfo(current_percept.maze_state)
        # print ("drone:", drone)
        # print ("doors:", doors)
        # print ("not error in drone")
        # Checking if end is visible, however, even when end is not visible our goal will probably be the edge of the 
        # radius so this should always be true in our case during exploration and final search
        if current_percept.is_end_visible:
            
            # Call A* with current coordinates as start and end coordinates that should 
            # be set when end is visible by timing_maze-_state line 15
            # Also we're calling A* everytime
            # print ("error in a*")
            final_path = self.a_star_search((self.cur_x, self.cur_y), (current_percept.end_x, current_percept.end_y), drone)
            # print ("not error in a*")
            # Convert the change in coordinates to a direction 
            # print ("error in get_move_direction")
            self.next_move = self.get_move_direction(final_path[0], final_path[1])
            # print ("not error in get_move_direction")
        
            print("next move:", self.next_move)

            if self.next_move == LEFT:
                if (doors[LEFT] == OPEN):
                    self.cur_x -= 1
                    self.final_move_directions.pop(0)
                    self.curr_stationary_moves = 0
                    return LEFT

            elif self.next_move == RIGHT:
                if (doors[RIGHT] == OPEN):
                    self.cur_x += 1
                    self.final_move_directions.pop(0)
                    self.curr_stationary_moves = 0
                    return RIGHT

            elif self.next_move == UP:
                if (doors[UP] == OPEN):
                    self.cur_y -=1 
                    self.final_move_directions.pop(0)
                    self.curr_stationary_moves = 0
                    return UP


            elif self.next_move == DOWN:
                if (doors[DOWN] == OPEN):
                    self.cur_y += 1
                    self.final_move_directions.pop(0)
                    return DOWN
            self.curr_stationary_moves +=1
            return WAIT 
        return 0
    
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
            return LEFT  # LEFT
        elif dx == 0 and dy == -1:
            return UP  # UP
        elif dx == 1 and dy == 0:
            return RIGHT  # RIGHTx
        elif dx == 0 and dy == 1:
            return DOWN  # DOWN
        else:
            return WAIT  # WAIT or invalid move

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
                    # print ("error in heuristic_manhatten")
                    f_score = tentative_g_score + self.heuristic_manhatten(neighbor, goal, current_turn, LCM_map)
                    # print ("not error in heuristic_manhatten")
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
                directions.append(RIGHT)
            elif x1 > x2:
                directions.append(LEFT)
            elif y1 < y2:
                directions.append(DOWN)
            else:
                directions.append(UP)

        # print(directions)

        return directions

    #  def take_next_open_move(self, current_percept):
    #     if (self.curr_stationary_moves >= 5):
    #             for maze_state in current_percept.maze_state:
    #                 if (maze_state[0] == self.cur_x and maze_state[1] == self.cur_y and maze_state[2] == DOWN
    #                         and maze_state[3] == OPEN):
                        
    #                     self.final_move_directions.pop(0)
    #                     if ( maze_state[2] == DOWN): self.cur_y -= 1
    #                     if ( maze_state[2] == UP): self.cur_y += 1
    #                     if ( maze_state[2] == LEFT): self.cur_x -= 1
    #                     if ( maze_state[2] == RIGHT): self.cur_x += 1
    #                     return maze_state[2]
    #                 else: self.curr_stationary_moves +=1
    #             return WAIT