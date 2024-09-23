import os
import pickle
import numpy as np
import logging

import constants
from timing_maze_state import TimingMazeState

##### Frank (9/16):
# For exploration algorithm
from players.group1_misc.experience import Experience
#################

##### Tom (9/15):
# For heap in a*
import heapq
#################
import math
import traceback

class Player:
    turn =0
    def __init__(self, rng: np.random.Generator, logger: logging.Logger,
                 precomp_dir: str, maximum_door_frequency: int, radius: int, wait_penalty: int) -> None:
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
        
        ################# Lingyi & Tom (9/23):
        self.wait_penalty = wait_penalty
        ######################################
        ########## Tom (9/15):
        self.frontier = []
        self.path = []
        ######################

        ########## Frank (9/16), edited by Tom (9/23):
        self.experience = Experience(self.maximum_door_frequency, self.radius, self.wait_penalty)
        ######################

        self.frequency={}
        self.cur_percept={}
        self.explored=set()

    ####### Adithi:
    def update_door_frequencies(self, current_percept):
        for x, y, direction, state in current_percept.maze_state:
            if state == constants.OPEN:
                glob_x = x-current_percept.start_x
                glob_y = y-current_percept.start_y
                key = (glob_x, glob_y, direction)
                self.cur_percept[key]=1
                #self.logger.info(f"{x},{y} direction: {direction} is open at turn {self.turn}")
                if key not in self.frequency:
                    self.frequency[key] = Player.turn
                else:
                    self.frequency[key]= math.gcd(Player.turn, self.frequency[key])

    
    def heuristic(self, cur, target, parent):
        distance = abs(cur[0] - target[0]) + abs(cur[1] - target[1])
        wait_time = self.find_wait(parent, cur)
        revisit_penalty = 5 if cur in self.explored else 0  # Add a penalty for revisiting cells
        self.logger.info(f"From cur {cur} to target {target} which came from {parent} dist: {distance} wait: {wait_time} penalty: {revisit_penalty}")
        return distance + wait_time + revisit_penalty

    
    def find_wait(self,cur,next):
         cur_to_next = (cur[0],cur[1],self.get_dir(cur,next))
         next_to_cur = (next[0],next[1],self.get_dir(next,cur))
         frequency= self.maximum_door_frequency+1
         if cur_to_next in self.frequency and next_to_cur in self.frequency:
             frequency = math.gcd(self.frequency[cur_to_next], self.frequency[next_to_cur])
         return (frequency - (Player.turn%frequency))%Player.turn
         #return float('inf')
                #  Wait time is (x−(ymodx))modx
                # Where: x is the number of turns after which the door opens, y is the current turn.
    
    def get_neighbours(self,node):
        neighbours = []
        if (node[0],node[1], constants.LEFT) in self.cur_percept and (node[0]-1, node[1], constants.RIGHT) in self.cur_percept:
            neighbours.append((node[0]-1, node[1], constants.LEFT))
        if (node[0],node[1], constants.RIGHT) in self.cur_percept and (node[0]+1, node[1], constants.LEFT) in self.cur_percept:
            neighbours.append((node[0]+1, node[1], constants.RIGHT))
        if (node[0],node[1], constants.UP) in self.cur_percept and (node[0], node[1]-1, constants.DOWN) in self.cur_percept:
            neighbours.append((node[0], node[1]-1, constants.UP))
        if (node[0],node[1], constants.DOWN) in self.cur_percept and (node[0], node[1]+1, constants.UP) in self.cur_percept:
            neighbours.append((node[0], node[1]+1, constants.DOWN))
        self.logger.info(f"Neighbours for node {node} is {neighbours}")
        return neighbours
    
    def get_rel_start(self,cur,start):
         return (cur[0]-start[0], cur[1]-start[1])
    
    def get_dir(self,cur,next_move):
        dx = next_move[0] - cur[0]
        dy = next_move[1] - cur[1]
        if dx == -1:
            return constants.UP
        elif dx == 1:
            return constants.DOWN
        elif dy == -1:
            return constants.LEFT
        elif dy == 1:
            return constants.RIGHT
        return constants.WAIT
    
    def isvalid(self,cur,move):
        if (cur[0],cur[1],move) in self.cur_percept:
            return True
        return False
    #########

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
        try:
            Player.turn+=1
            self.cur_percept={}
            self.update_door_frequencies(current_percept)
            ################################ Tom (9/15)
            if current_percept.is_end_visible:
                cur =self.get_rel_start((0,0),(current_percept.start_x, current_percept.start_y))
                target =self.get_rel_start((current_percept.end_x, current_percept.end_y),(current_percept.start_x, current_percept.start_y))
                self.logger.info(f"Cur {cur}, Target: {target}")
                # If there's no path, run A* to find one
                if not self.path:
                    #print("not self.path")
                    self.path = self.a_star(cur, target)
                    if self.path:
                        next_move = self.path.pop(0) 
                        return next_move
                # If A* found a path, execute the next move
                else:
                    # Get the next move from the path
                    #print("yes self.path")
                    next_move = self.path.pop(0) 
                    if(self.isvalid(cur,next_move)):
                        return next_move
                    else:
                        self.path= self.a_star(cur, target)
                        if self.path:
                            next_move = self.path.pop(0)
                        else:
                            return constants.WAIT
                    #print("next move is")
                    return next_move
            else: # If End is not visible

                ########## Frank (9/16):
                return self.experience.move(current_percept)
                ###########################
        except Exception as e:
            print(e)
            traceback.print_exc()
    

    def a_star (self, start, goal):
        # Reset frontier and explored set
        self.frontier = []
        self.explored = set()
        self.incons = [] # Reset the inconsistency list.
        self.g_values = {start: float('inf')}  # Initialize g-value of the start node to infinity (not yet discovered).
        self.rhs_values = {start: 0}  # Set the rhs-value of the start node to 0 (starting point).
        self.goal = goal  # Store the goal node for use in heuristic calculations.
        heapq.heappush(self.frontier, (self.calculate_key(start), start))

        # # Start position and goal position
        # start = (0, 0)  # (x, y) relative position
        # print("start is: ")
        # print(start)
        # #goal = (current_percept.end_x, current_percept.end_y)
        # print("goal is:")
        # print(goal)

        # Push the start state to the frontier with a cost of 0
        heapq.heappush(self.frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while self.frontier:
            _, current = heapq.heappop(self.frontier)
            #print ("current is:")
            #print (current)
            # Loop until the current state is the goal;
            # Once the current state is the goal, we know that the path is found;
            # then call the reconstruct_path function to construct a path.
            if current == goal:
                #print(came_from)
                return self.reconstruct_path(came_from, start, goal)

            self.explored.add(current)

            neighbours = self.get_neighbours(current)
            #print(neighbours)
            for i in neighbours:
                new_cost = cost_so_far[current] + 1
                x, y, dir = i
                revisit_penalty = 5 if (x, y) in self.explored else 0  # Apply penalty for revisits
                if (x, y) not in cost_so_far or new_cost < cost_so_far[(x, y)]:
                    cost_so_far[(x, y)] = new_cost + revisit_penalty
                    priority = new_cost + revisit_penalty + self.heuristic((x, y), goal, current)
                    heapq.heappush(self.frontier, (priority, (x, y)))
                    came_from[(x, y)] = (current, dir)

        
        return None  # No path found

    def reconstruct_path(self, came_from, start, goal):
        """Reconstruct the path from start to goal."""
        path = []
        current = goal
        while current != start:
            current, direction = came_from[current]
            path.append(direction)
        path.reverse()
        print("path is:")
        print(path)
        return path