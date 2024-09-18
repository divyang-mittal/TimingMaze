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
import copy
from collections import defaultdict
import traceback

class Player:
    turn =0 

    def __init__(self, rng: np.random.Generator, logger: logging.Logger,
                 precomp_dir: str, maximum_door_frequency: int, radius: int) -> None:


        self.rng = rng
        self.logger = logger
        self.maximum_door_frequency = maximum_door_frequency
        self.radius = radius

        self.pq =[]
        self.g = {}
        self.f = {}
        self.parent ={}

        self.frequency={}  
        self.cur_percept={} 
        self.path=[]

    ####### Adithi (9/16):
    def update_door_frequencies(self, current_percept):
        for x, y, direction, state in current_percept.maze_state:
            if state == constants.OPEN:
                glob_x = x-current_percept.start_x
                glob_y = y-current_percept.start_y
                key = (glob_x, glob_y, direction)
                self.cur_percept[key]=1
                #self.logger.info(f"{x},{y} direction: {direction} is open at turn {self.turn}")
                if key not in self.frequency:
                    self.frequency[key] = self.turn
                else:
                    self.frequency[key]= math.gcd(self.turn, self.frequency[key])

    
    def heuristic(self,cur, target):
         distance = abs(cur[0]- target[0])+ abs(cur[1]-target[1])
         wait_time = self.find_wait(cur)
         # Todo: Try different weights for wait_Time
         return distance+ wait_time*10
    
    def find_wait(self,cur):
         if cur in self.frequency:
              f=self.frequency[cur]
              t=self.g.get(cur,Player.turn)
              return (f-(t%f))%f
         return self.maximum_door_frequency+1
    
    def get_neighbours(self,node):
        neighbours = []
        if (node[0],node[1], constants.LEFT) in self.cur_percept and (node[0]-1, node[1], constants.RIGHT) in self.cur_percept:
            neighbours.append((node[0]-1, node[1]))
        if (node[0],node[1], constants.RIGHT) in self.cur_percept and (node[0]+1, node[1], constants.LEFT) in self.cur_percept:
            neighbours.append((node[0]+1, node[1]))
        if (node[0],node[1], constants.UP) in self.cur_percept and (node[0], node[1]-1, constants.DOWN) in self.cur_percept:
            neighbours.append((node[0], node[1]-1))
        if (node[0],node[1], constants.DOWN) in self.cur_percept and (node[0], node[1]+1, constants.UP) in self.cur_percept:
            neighbours.append((node[0], node[1]+1))
        self.logger.info(f"Neighbours for node {node} is {neighbours}")
        return neighbours
    
    def reconstruct_path(self,cur):
         path =[cur]
         while cur in self.parent:
              cur = self.parent[cur]
              path.append(cur)

         return path

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
            self.cur_percept={}
            Player.turn+=1
            self.update_door_frequencies(current_percept)

            if current_percept.is_end_visible:
                cur =self.get_rel_start((0,0),(current_percept.start_x, current_percept.start_y))
                target =self.get_rel_start((current_percept.end_x, current_percept.end_y),(current_percept.start_x, current_percept.start_y))
                self.logger.info(f"Cur {cur}, Target: {target}")
                if self.path:
                    next_move=self.path.pop(1)
                    self.logger.info(f"Next move: {next_move}, Cur:{cur}, Direction: {self.get_dir(cur,next_move)}")
                    return self.get_dir(cur,next_move)

                self.g[cur]=0
                self.f[cur]=self.heuristic(cur,target)
                heapq.heappush(self.pq, (self.f[cur],cur))

                while self.pq:
                    _,cur = heapq.heappop(self.pq)

                    if cur==target:
                        path=self.reconstruct_path(cur)
                        self.path=path
                        path.pop(0)
                        next_move=path.pop(1)
                        self.logger.info(f"Path found {path}")
                        self.logger.info(f"Next move: {next_move}, Cur:{cur}, Direction: {self.get_dir(cur,next_move)}")
                        return self.get_dir(cur,next_move)
                    
                    neighbours = self.get_neighbours(cur)
                    for neighbour in neighbours:
                        g = self.g[cur]+self.find_wait(neighbour)
                        if neighbour not in self.g or g<self.g[neighbour]:
                                self.parent[neighbour]=cur
                                self.g[neighbour] =g
                                f=g+self.heuristic(neighbour,target)
                                self.f[neighbour]=f
                                heapq.heappush(self.pq,(f,neighbour))

                return constants.WAIT #No path
        except Exception as e:
            print(e)
            traceback.print_exc()
        

             
             





        # ################################ Tom (9/15): Comment this chunk to go back to default player
        # if current_percept.is_end_visible:
        #     # If there's no path, run A* to find one
        #     if not self.path:
        #         #print("not self.path")
        #         self.path = self.a_star(current_percept)
        #         #self.path = self.a_star_freq(current_percept)
            
        #     # If A* found a path, execute the next move
        #     else:
        #         # Get the next move from the path
        #         #print("yes self.path")
        #         next_move = self.path.pop(0) 
        #         print("next move is")
        #         #print(next_move)
        #         return next_move
        # else: # If End is not visible

        #     ########## Frank (9/16):
        #     move = self.experience.move(current_percept)

        #     if self.experience.is_valid_move(current_percept, move):
        #         return move

        #     self.experience.wait()
        #     return constants.WAIT
        #     ###########################
            
        # ############################################ Comment this chunk to go back to default player

        # # default_player:
        # # The condition of the four doors at the current cell
        # direction = [0, 0, 0, 0] # [left, up, right, down]; 1 is closed, 2 is open, 3 is boundary
        # for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
        #     if maze_state[0] == 0 and maze_state[1] == 0: # (0,0) is the current loc; -> Looking to see the conditions of the four doors at the current location.
        #         direction[maze_state[2]] = maze_state[3] # import that information into direction

        # if current_percept.is_end_visible:
        #     if abs(current_percept.end_x) >= abs(current_percept.end_y): # huhh???
        #         if (current_percept.end_x > 0 # if goal is on the right side
        #             and direction[constants.RIGHT] == constants.OPEN # if the door on the right is open
        #             ):
        #             for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
        #                 if (maze_state[0] == 1 and maze_state[1] == 0 # (1,0) is the cell on the right; -> Looking to see the conditions of the four doors at the cell on the right.
        #                     and maze_state[2] == constants.LEFT and maze_state[3] == constants.OPEN # if the left door of the cell on the right (the adjacent door to the current cell) is open
        #                     ):
        #                     return constants.RIGHT # goes right -> returning 2
                        
        #         if (current_percept.end_x < 0 # if goal is on the left side
        #             and direction[constants.LEFT] == constants.OPEN # if the door on the left is open
        #             ):
        #             for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
        #                 if (maze_state[0] == -1 and maze_state[1] == 0 # (-1,0) is the cell on the left; -> Looking to see the conditions of the four doors at the cell on the left.
        #                     and maze_state[2] == constants.RIGHT and maze_state[3] == constants.OPEN # if the right door of the cell on the left (the adjacent door to the current cell) is open
        #                     ):
        #                     return constants.LEFT # goes left -> returning 0
                        
        #         if (current_percept.end_y < 0 # if goal is above
        #             and direction[constants.UP] == constants.OPEN # if the door above is open
        #             ):
        #             for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
        #                 if (maze_state[0] == 0 and maze_state[1] == -1 # (0,-1) is the cell above; -> Looking to see the conditions of the four doors at the cell above.
        #                     and maze_state[2] == constants.DOWN and maze_state[3] == constants.OPEN # if the down door of the cell above (the adjacent door to the current cell) is open
        #                     ):
        #                     return constants.UP # goes up -> returning 1
                        
        #         if (current_percept.end_y > 0 # if goal is below
        #             and direction[constants.DOWN] == constants.OPEN # if the door below is open
        #             ):
        #             for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
        #                 if (maze_state[0] == 0 and maze_state[1] == 1 # (0,1) is the cell below; -> Looking to see the conditions of the four doors at the cell below.
        #                     and maze_state[2] == constants.UP and maze_state[3] == constants.OPEN # if the above door of the cell below (the adjacent door to the current cell) is open
        #                     ): 
        #                     return constants.DOWN # goes down -> returning 3
                        
        #         return constants.WAIT # return -1
            
        #     else:
        #         if (current_percept.end_y < 0 # if goal is above
        #             and direction[constants.UP] == constants.OPEN # if the door above is open
        #             ):
        #             for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
        #                 if (maze_state[0] == 0 and maze_state[1] == -1 # (0,-1) is the cell above; -> Looking to see the conditions of the four doors at the cell above.
        #                     and maze_state[2] == constants.DOWN and maze_state[3] == constants.OPEN # if the down door of the cell above (the adjacent door to the current cell) is open
        #                     ):
        #                     return constants.UP # return 1
                        
        #         if (current_percept.end_y > 0 # if goal is below
        #             and direction[constants.DOWN] == constants.OPEN # if the door below is open
        #             ):
        #             for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
        #                 if (maze_state[0] == 0 and maze_state[1] == 1 # (0,1) is the cell below; -> Looking to see the conditions of the four doors at the cell below.
        #                     and maze_state[2] == constants.UP and maze_state[3] == constants.OPEN # if the above door of the cell below (the adjacent door to the current cell) is open
        #                     ):
        #                     return constants.DOWN # return 3
                        
        #         if (current_percept.end_x > 0 # if goal is on the right side
        #             and direction[constants.RIGHT] == constants.OPEN # if the door on the right is open
        #             ):
        #             for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
        #                 if (maze_state[0] == 1 and maze_state[1] == 0 # (1,0) is the cell on the right; -> Looking to see the conditions of the four doors at the cell on the right.
        #                     and maze_state[2] == constants.LEFT and maze_state[3] == constants.OPEN # if the left door of the cell on the right (the adjacent door to the current cell) is open
        #                     ):
        #                     return constants.RIGHT # return 2
                        
        #         if (current_percept.end_x < 0 # if goal is on the left side
        #             and direction[constants.LEFT] == constants.OPEN # if the door on the left is open
        #             ):
        #             for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
        #                 if (maze_state[0] == -1 and maze_state[1] == 0 # (-1,0) is the cell on the left; -> Looking to see the conditions of the four doors at the cell on the left.
        #                     and maze_state[2] == constants.RIGHT and maze_state[3] == constants.OPEN # if the right door of the cell on the left (the adjacent door to the current cell) is open
        #                     ):
        #                     return constants.LEFT #return 0
                        
        #         return constants.WAIT # return -1
            
        # else: # If End is not visible

        #     ########## Frank (9/16):
        #     move = self.experience.move(current_percept)

        #     if self.experience.is_valid_move(current_percept, move):
        #         return move

        #     self.experience.wait()
        #     return constants.WAIT
        #     ###########################
    

    # ########################################## Tom (9/15):
    # def a_star (self, current_percept):
    #     # Reset frontier and explored set
    #     self.frontier = []
    #     self.explored = set()

    #     # Start position and goal position
    #     start = (0, 0)  # (x, y) relative position
    #     #print("start is: ")
    #     #print(start)
    #     goal = (current_percept.end_x, current_percept.end_y)
    #     #print("goal is:")
    #     #print(goal)

    #     # Push the start state to the frontier with a cost of 0
    #     heapq.heappush(self.frontier, (0, start))
    #     came_from = {start: None}
    #     cost_so_far = {start: 0}

    #     while self.frontier:
    #         _, current = heapq.heappop(self.frontier)
    #         #print ("current is:")
    #         #print (current)
    #         # Loop until the current state is the goal;
    #         # Once the current state is the goal, we know that the path is found;
    #         # then call the reconstruct_path function to construct a path.
    #         if current == goal:
                
    #             return self.reconstruct_path(came_from, start, goal)

    #         self.explored.add(current)
            
    #         # Get the possible moves from the current position
    #         neighbors = self.get_neighbors(current, current_percept)
    #         #print("neighbors are:")
    #         #print(neighbors)
    #         for direction, neighbor in neighbors:
    #             new_cost = cost_so_far[current] + 1  # Assume each move costs 1
    #             if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
    #                 cost_so_far[neighbor] = new_cost
    #                 priority = new_cost + self.heuristic(neighbor, goal)
    #                 heapq.heappush(self.frontier, (priority, neighbor))
    #                 came_from[neighbor] = (current, direction)
        
    #     return None  # No path found
    

    # # Manhattan distance heuristic for A*
    # def heuristic(self, current, goal):
    #     return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    # # Get neighbors and their movement directions based on door states.
    # def get_neighbors(self, current, current_percept):
    #     x, y = current
    #     #directions = [constants.LEFT, constants.UP, constants.RIGHT, constants.DOWN]
    #     #moves = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # left, up, right, down
    #     neighbors = []
        
    #     # The condition of the four doors at the current cell
    #     counter1 = 0
    #     direction = [0, 0, 0, 0] # [left, up, right, down]; 1 is closed, 2 is open, 3 is boundary
    #     for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
    #         if maze_state[0] == x and maze_state[1] == y: # (x,y) is the current loc; -> Looking to see the conditions of the four doors at the current location.
    #             direction[maze_state[2]] = maze_state[3] # import that information into direction
    #             counter1 += 1
    #         # Meaning we have gathered everything needed for the variable direction; no need to keep going
    #         if counter1 == 4: 
    #             break

    #     # Check if "moving to right" is a legit move: checking 1). the right door is open; 2) the left door on the cell to the right is open 
    #     if (direction[constants.RIGHT] == constants.OPEN): # if the door on the right is open
    #         for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
    #                     if (maze_state[0] == x+1 and maze_state[1] == y+0 # (x+1,y+0) is the cell on the right; -> Looking to see the conditions of the four doors at the cell on the right.
    #                         and maze_state[2] == constants.LEFT and maze_state[3] == constants.OPEN # if the left door of the cell on the right (the adjacent door to the current cell) is open
    #                         ):
    #                         neighbors.append((constants.RIGHT, (x + 1, y)))
    #                         break

    #     # Check if "moving to left" is a legit move: checking 1). the left door is open; 2) the right door on the cell to the left is open 
    #     if (direction[constants.LEFT] == constants.OPEN): # if the door on the left is open
    #         for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
    #                     if (maze_state[0] == x-1 and maze_state[1] == y+0 # (x-1,y+0) is the cell on the left; -> Looking to see the conditions of the four doors at the cell on the left.
    #                         and maze_state[2] == constants.RIGHT and maze_state[3] == constants.OPEN # if the right door of the cell on the left (the adjacent door to the current cell) is open
    #                         ):
    #                         neighbors.append((constants.LEFT, (x - 1, y)))
    #                         break
                
    #     # Check if "moving up" is a legit move: checking 1). the up door is open; 2) the down door on the cell above is open 
    #     if (direction[constants.UP] == constants.OPEN): # if the door above is open
    #         for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
    #                     if (maze_state[0] == x+0 and maze_state[1] == y-1 # (x+0,y-1) is the cell above; -> Looking to see the conditions of the four doors at the cell above.
    #                         and maze_state[2] == constants.DOWN and maze_state[3] == constants.OPEN # if the down door of the cell above (the adjacent door to the current cell) is open
    #                         ):
    #                         neighbors.append((constants.UP, (x, y - 1)))
    #                         break

    #     # Check if "moving down" is a legit move: checking 1). the down door is open; 2) the up door on the cell to the below is open 
    #     if (direction[constants.DOWN] == constants.OPEN): # if the door below is open
    #         for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
    #                     if (maze_state[0] == x+0 and maze_state[1] == y+1 # (x+0,y+1) is the cell below; -> Looking to see the conditions of the four doors at the cell below.
    #                         and maze_state[2] == constants.UP and maze_state[3] == constants.OPEN # if the up door of the cell below (the adjacent door to the current cell) is open
    #                         ):
    #                         neighbors.append((constants.DOWN, (x, y + 1)))
    #                         break

    #     # Finally return the neighbors that we can possibly move to.
    #     return neighbors


    # def reconstruct_path(self, came_from, start, goal):
    #     """Reconstruct the path from start to goal."""
    #     path = []
    #     current = goal
    #     while current != start:
    #         current, direction = came_from[current]
    #         path.append(direction)
    #     path.reverse()
    #     #print("path is:")
    #     #print(path)
    #     return path
    # ##########################################