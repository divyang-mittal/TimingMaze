import numpy as np
import logging
import math
import traceback
import heapq
import constants
from players.group1_misc.experience import Experience

class Player:
    turn = 0

    def __init__(self, rng, logger, precomp_dir, maximum_door_frequency, radius):
        self.rng = rng
        self.logger = logger
        self.maximum_door_frequency = maximum_door_frequency
        self.radius = radius
        self.frequency = {}
        self.cur_percept = set()
        self.path = []  # Current path from start to end
        self.cost = {}  # Cost to reach each node
        self.end = -1
        self.experience = Experience(self.maximum_door_frequency, self.radius,2)
        self.newcells = set()  # Cells whose frequencies changed or are seen for the first time
        self.open_list = []  # Priority queue for the D* Lite algorithm
        self.parent = {}  # Keeps track of the parent node for path reconstruction
        self.depth = {}  # Stores the depth (turn count) for each node

    def get_rel_start(self, x, y, startx, starty):
        newx = x - startx
        newy = y - starty
        return newx, newy

    def update_door_frequencies(self, current_percept):
        for x, y, direction, state in current_percept.maze_state:
            if state == constants.OPEN:
                glob_x, glob_y = self.get_rel_start(x, y, current_percept.start_x, current_percept.start_y)
                key = (glob_x, glob_y, direction)
                self.cur_percept.add(key)
                if key not in self.frequency:
                    self.frequency[key] = Player.turn
                    self.newcells.add((glob_x, glob_y))
                else:
                    newfrequency = math.gcd(Player.turn, self.frequency[key])
                    if newfrequency < self.frequency[key]:
                        self.newcells.add((glob_x, glob_y))
                    self.frequency[key] = newfrequency

    def lcm(self, a, b):
        return abs(a * b) // math.gcd(a, b) if a and b else 0

    def find_wait(self, cur, next, turn):
        cur_to_next = (cur[0], cur[1], self.get_dir(cur, next))
        next_to_cur = (next[0], next[1], self.get_dir(next, cur))
        
        frequency = self.maximum_door_frequency + 1
        wait_time = frequency
        
        # Cache frequency values
        cur_freq = self.frequency.get(cur_to_next, None)
        next_freq = self.frequency.get(next_to_cur, None)

        if cur_freq is not None and next_freq is not None:
            frequency = self.lcm(cur_freq, next_freq)
            if frequency!=0:
                wait_time = (frequency - (turn % frequency)) % frequency

        return wait_time

    def get_neighbours(self, node):
        x,y = node
        neighbours = [
            (x - 1, y, constants.LEFT),
            (x + 1, y, constants.RIGHT),
            (x, y - 1, constants.UP),
            (x, y + 1, constants.DOWN)
        ]
        return neighbours

    def move(self, current_percept) -> int:
        try:
            Player.turn += 1
            self.cur_percept.clear()
            self.update_door_frequencies(current_percept)

            if current_percept.is_end_visible or self.end!=-1:
                cur = self.get_rel_start(0, 0, current_percept.start_x, current_percept.start_y)
                target = self.get_rel_start(current_percept.end_x, current_percept.end_y, current_percept.start_x, current_percept.start_y)
                self.end = target  # Set the end position

                # Use D* Lite algorithm to either find or update the path incrementally
                self.path = self.d_star_lite(cur, target)

                if self.path:
                    next_move = self.path[0]
                    if self.is_valid_move(cur, next_move):
                        self.logger.info(f"Move valid {next_move} from {cur}")
                        return self.get_dir(cur, next_move)
                    else:
                        wait_time = self.find_wait(cur, next_move, Player.turn)
                        self.logger.info(f"Wait time since move {next_move} is {wait_time}")
                        if wait_time > self.maximum_door_frequency:
                            self.initialize_path(cur,self.end)
                            self.path = self.d_star_lite(cur, self.end)
                            self.logger.info(f"Path: {self.path}")
                            if self.path and self.is_valid_move(cur, self.path[0]):
                                return self.get_dir(cur, self.path[0])
                            else:
                                self.logger.info(f"No path so exploring as wait {wait_time} is long")
                                for i in self.get_neighbours(cur):
                                    if self.is_valid_move(cur, i):
                                        return self.get_dir(cur,i)
                                self.logger.info("No immediate valid moves so exploring")
                                return self.experience.move(current_percept)
                        else:
                            return constants.WAIT
                        
            self.logger.info(f"Exploring since no path or end not seen")
            next_move = self.experience.move(current_percept)
            if self.experience.is_valid_move(current_percept, next_move):
                    return self.experience.move(current_percept)
            return constants.WAIT

        except Exception as e:
            print(e)
            traceback.print_exc()

    def initialize_path(self,start,goal):
            self.cost[start] = 0
            self.open_list.clear()
            heapq.heappush(self.open_list, (self.cost[start], start))
            self.path.clear()  # Store the resulting path
            self.parent = {start: None}
            self.depth = {start: Player.turn}

    def d_star_lite(self, start, goal):
    # Initialize costs and priority queue if first time or when major recalculation is needed
        if not self.cost:
            self.initialize_path(start, goal)

        #self.logger.info(f"Start {start} Goal {goal}")

        # Check if a path has already been found
        if not self.path or goal not in self.parent:
            self.logger.info("No pre-existing path found. Calculating a new path.")
            # Perform full path calculation if no path exists
            self.find_full_path(start, goal)
        else:
            # If there are new cells, update the path incrementally
            if self.newcells:
                self.incremental_update_with_new_cells(start, goal)

        # If path to goal has been found or updated, retrace the path
        if goal in self.parent:
            self.logger.info(f"Trying to retrace path")
            self.path = self.retrace_path(start, goal, self.parent)
        else:
            self.logger.info(f"No path for {start} to {goal} at {Player.turn}")
            self.path = []
        return self.path

    def find_full_path(self, start, goal):
        heapq.heappush(self.open_list, (0, start))
        timeout =10000
        while self.open_list and timeout>0:
            timeout-=1
            current_cost, current = heapq.heappop(self.open_list)
            #print(current,goal)
            if current == goal:
                self.logger.info(f"Found path in full path")
                break 

            # Get neighbours and update costs for all potential paths
            for neighbour in self.get_neighbours(current):
                x, y, direction = neighbour
                wait_time = self.find_wait(current, (x, y), self.depth[current] + 1)
                #self.logger.info(f"Wait for {current} to {(x, y)} is {wait_time} at turn {self.depth[current] + 1}")
                
                new_cost = current_cost + wait_time + self.cost_function((x, y), goal)+1
                if (x, y) not in self.cost or new_cost < self.cost[(x, y)]:
                    self.parent[(x, y)] = current
                    self.depth[(x, y)] = self.depth[current] + 1
                    self.cost[(x, y)] = new_cost
                    heapq.heappush(self.open_list, (new_cost, (x, y)))


    def incremental_update_with_new_cells(self, start, goal):
        changed =False
        for new_cell in self.newcells:
            node = self.is_near_path(new_cell)
            if node !=-1:
                wait_time = self.find_wait(node, new_cell, self.depth[node] + 1)
                new_cost = 0
                if node != new_cell:
                    if self.is_valid_move(node, new_cell):
                        new_cost = self.cost[node]+ wait_time + self.cost_function(new_cell, goal)+1
                        self.depth[new_cell]= self.depth[node]+1
                        self.parent[new_cell]=node
                        self.cost[new_cell] = new_cost
                        heapq.heappush(self.open_list, (self.cost[new_cell], new_cell))
                        changed=True
                else:
                    parent = self.parent[node]
                    wait_time = self.find_wait(parent, node, self.depth[node])
                    new_cost = self.cost[parent]+wait_time+ self.cost_function(node,goal)+1
                    if new_cost < self.cost[node] and self.is_valid_move(parent, node):
                        self.cost[node] = new_cost
                        heapq.heappush(self.open_list, (self.cost[node], node))
                        changed=True
        if changed:
            self.logger.info(f"Changed so updating path")
            self.update_path_segment(goal)

        # Clear newcells after using them
        self.newcells.clear()

    def update_path_segment(self, goal):  
        timeout = 10000      
        while self.open_list and timeout>0:
            timeout-=1
            current_cost, current = heapq.heappop(self.open_list)
            if current == goal:
                self.logger.info("Goal reached in update")
                break  # Stop as soon as the goal is reached

            # Get neighbours and update costs based on the new cell
            for neighbour in self.get_neighbours(current):
                x, y, direction = neighbour
                wait_time = self.find_wait(current, (x, y), self.depth[current] + 1)
                #self.logger.info(f"Wait for {current} to {(x, y)} is {wait_time} at turn {self.depth[current] + 1}")

                new_cost = current_cost + wait_time + self.cost_function((x, y), goal)+1
                if (x, y) not in self.cost or new_cost < self.cost[(x, y)]:
                    self.parent[(x, y)] = current
                    self.depth[(x, y)] = self.depth[current] + 1
                    self.cost[(x, y)] = new_cost
                    heapq.heappush(self.open_list, (new_cost, (x, y)))

    def is_near_path(self, new_cell):
        # Check if the new cell is adjacent to any node in the current path or on the path
        for node in self.path:
            if abs(node[0] - new_cell[0]) + abs(node[1] - new_cell[1]) <=1:  # Manhattan distance of 1 or 0
                return node
        return -1

    def retrace_path(self, start, goal, parent):
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = parent[current]
        path.reverse()
        self.logger.info(f"Path at turn {Player.turn} is {path}")
        return path

    def cost_function(self, neighbour, goal):
        return abs(neighbour[0] - goal[0]) + abs(neighbour[1] - goal[1])

    def get_dir(self, cur, next):
        if cur[0] - next[0] >= 1: return constants.LEFT
        if cur[0] - next[0] <= -1: return constants.RIGHT
        if cur[1] - next[1] >= 1: return constants.UP
        if cur[1] - next[1] <= -1: return constants.DOWN
        return constants.WAIT

    def is_valid_move(self, cur, next):
        cur_to_next = (cur[0], cur[1], self.get_dir(cur, next))
        next_to_cur = (next[0], next[1], self.get_dir(next, cur))
        return (cur_to_next in self.cur_percept) and (next_to_cur in self.cur_percept)
