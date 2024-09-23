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

        ########## Tom (9/15):
        self.frontier = []
        self.explored = set()
        self.path = []
        ######################

        ########## Frank (9/16):
        self.experience = Experience(self.maximum_door_frequency, self.radius)
        ######################

        self.frequency={}
        self.cur_percept={}

        ###################################################### Tom (9/21) (ADA*):
        self.epsilon = 2.5  # Initial ε value for suboptimality (can be adjusted)
        self.incons = []  # Inconsistency list for nodes
        self.g_values = {}
        self.rhs_values = {}       
        #########################################################################
    
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

    
    def heuristic(self,cur, target,parent):
         distance = abs(cur[0]- target[0])+ abs(cur[1]-target[1])
         wait_time = self.find_wait(parent,cur)
         # Todo: Try different weights for wait_Time
         return (distance*1)+ (wait_time*1)
    
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
            ################################ Tom (9/15): Comment this chunk to go back to default player
            if current_percept.is_end_visible:
                cur =self.get_rel_start((0,0),(current_percept.start_x, current_percept.start_y))
                target =self.get_rel_start((current_percept.end_x, current_percept.end_y),(current_percept.start_x, current_percept.start_y))
                self.logger.info(f"Cur {cur}, Target: {target}")
                # If there's no path, run A* to find one
                if not self.path:
                    #print("not self.path")
                    self.path = self.a_star(cur, target)
                
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
        ############################################ Comment this chunk to go back to default player

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
            ###########################
    

    ########################################## Tom (9/15):
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
                new_cost = cost_so_far[current]+1
                x,y,dir = i
                #print(i)
                if (x,y) not in cost_so_far:
                    cost_so_far[(x,y)]=new_cost
                    priority=new_cost+self.heuristic((x,y),goal, current)
                    heapq.heappush(self.frontier, (priority, (x,y)))
                    came_from[(x,y)]=(current, dir)
            
            # Get the possible moves from the current position
            # neighbors = self.get_neighbors(current, current_percept)
            # print("neighbors are:")
            # print(neighbors)
            # for direction, neighbor in neighbors:
            #     new_cost = cost_so_far[current] + 1  # Assume each move costs 1
            #     if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
            #         cost_so_far[neighbor] = new_cost
            #         priority = new_cost + self.heuristic(neighbor, goal, current)
            #         heapq.heappush(self.frontier, (priority, neighbor))
            #         came_from[neighbor] = (current, direction)
        
        return None  # No path found
    
    def calculate_key(self, state):
        # Calculate the priority key of a given state based on g and rhs values
        
        g_rhs_min = min(self.g_values.get(state, float('inf')), self.rhs_values.get(state, float('inf')))
        # Return the priority tuple (key) consisting of an estimate and the minimum of g-value and rhs-value
        return (g_rhs_min + self.epsilon * self.heuristic(state, self.goal, None), g_rhs_min)
    
    def update_vertex(self, state):
        # Update the vertex information for a given state
        
        if state != self.goal:
            # Update the rhs-value of the state based on its neighbors' costs
            self.rhs_values[state] = min(
                self.g_values.get(neighbor, float('inf')) + self.cost(state, neighbor)
                for neighbor in self.get_neighbours(state)
            )
        
        # Remove the state from the priority queue if it exists
        self.frontier = [(k, s) for (k, s) in self.frontier if s != state]
        heapq.heapify(self.frontier)  # Rebuild the heap to maintain heap properties
        
        if self.g_values.get(state, float('inf')) != self.rhs_values.get(state, float('inf')):
            if state not in self.explored:
                # If the state is not yet explored, add it to the priority queue
                heapq.heappush(self.frontier, (self.calculate_key(state), state))
            else:
                # Otherwise, add it to the inconsistency list
                self.incons.append(state)

    def cost(self, state1, state2):
        # Return the cost of moving from state1 to state2, assuming uniform cost
        return 1  # Modify this if costs vary between nodes

    def compute_path(self):
        # Main loop that performs the Anytime Dynamic A* search
        
        # While there are nodes in the priority queue, and the priority queue's top key is less than the goal's key
        while self.frontier and (self.frontier[0][0] < self.calculate_key(self.goal) or self.rhs_values.get(self.goal, float('inf')) != self.g_values.get(self.goal, float('inf'))):
            _, current = heapq.heappop(self.frontier)  # Pop the node with the highest priority from the queue
            
            if self.g_values.get(current, float('inf')) > self.rhs_values.get(current, float('inf')):
                # If g-value is greater than rhs-value, update g-value
                self.g_values[current] = self.rhs_values[current]
                # Update all neighbors of the current state
                for neighbor in self.get_neighbours(current):
                    self.update_vertex(neighbor)
            else:
                # Otherwise, set g-value to infinity (remove the state)
                self.g_values[current] = float('inf')
                # Update the state itself and its neighbors
                for neighbor in self.get_neighbours(current) + [current]:
                    self.update_vertex(neighbor)
    
    def update_graph(self, state_changes):
        # Update the graph based on changes (e.g., obstacles appearing or disappearing)
        for state in state_changes:
            self.update_vertex(state)  # Update each changed state

    def improve_solution(self):
        # Gradually improve the solution by reducing epsilon
        
        self.epsilon -= 0.5  # Reduce epsilon to gradually make the solution more optimal
        if self.epsilon < 1:
            self.epsilon = 1  # Ensure epsilon never goes below 1 (fully optimal)
        
        # Add all nodes from INCONS to the priority queue
        for state in self.incons:
            heapq.heappush(self.frontier, (self.calculate_key(state), state))
        
        self.incons.clear()  # Clear the inconsistency list since nodes are now back in the priority queue
        heapq.heapify(self.frontier)  # Rebuild the heap structure
        self.compute_path()  # Recompute the path with the updated ε value

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

    # Manhattan distance heuristic for A*
    # def heuristic(self, current, goal):
    #     return abs(current[0] - goal[0]) + abs(current[1] - goal[1])


    # Get neighbors and their movement directions based on door states.
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
    ##########################################