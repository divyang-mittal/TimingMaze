import os
import pickle
import numpy as np
import logging
import random
# from utils import get_divisors
# from dataclasses import dataclass
import networkx as nx # pip install networkx
import matplotlib.pyplot as plt # pip install matplotlib
import math
from players.g7.player_helper_code import build_graph_from_memory, MazeGraph, PlayerMemory, findShortestPathsToEachNode, reconstruct_path, is_move_valid, MemorySquare, Boundary


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
        self.memory: PlayerMemory = PlayerMemory()
        self.turn = 0
        self.starting_position_set = False #check
        self.target_node_absolute_coords = None
        self.current_intermediate_target = None
        self.current_intermediate_target_age = 0

    
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
        # if not self.starting_position_set: #setting starting position if not set
        #     self.memory.pos = (current_percept.player_x, current_percept.player_y)
        #     self.starting_position_set = True
        self.turn += 1
        move = constants.WAIT

        # Decide on the next move based on the current percept.
        self.memory.update_memory(current_percept.maze_state, self.turn)
        
        # Build the graph from the updated memory
        currentGraph = build_graph_from_memory(self.memory, True)
        minDistanceArray, parent = findShortestPathsToEachNode(currentGraph, self.memory.pos, self.turn)

        # Case 1: We know the end position and we can reach it Follow path.
        # Case 2: We know the end position but we can't reach it. 
        # Case 3: We don't know the end position.

        # This should only run once
        if current_percept.is_end_visible and not self.target_node_absolute_coords:
            self.target_node_absolute_coords = (self.memory.pos[0] + current_percept.end_y, self.memory.pos[1] + current_percept.end_x)
            print("Found target location: ", self.target_node_absolute_coords)

        if self.target_node_absolute_coords:
            print("We know target location and we have a path to get to it")
            
            path = reconstruct_path(parent, self.target_node_absolute_coords)
            if path and len(path) > 1:
            # Case 1: We know the end position and we can reach it. Follow path..
                next_move = self.get_move_direction(path)
                print(path)
                print("Want to make next move: ", next_move)
                if is_move_valid(next_move, current_percept.maze_state):
                    self.memory.update_pos(next_move)
                    print("New Pos: ", self.memory.pos)
                    return next_move
                else: 
                    print("Desired Next Move is Invalid. Waiting.")
                    return constants.WAIT
            #else: 
                # Case 2: We know the end position but we can't reach it. 
                # Default to explore mode

        # If we are here we are in Case 2 or 3

        if self.target_node_absolute_coords:
            print("We know target location but we don't have a path to get to it")
            # We know the target location but we can't get to it. 
            # We need to explore the map to find a path to the target location
            # We need to find a path


        # If the end is not visible, choosing an intermediate node

        # If we have intermediate target node. Follow it for at least 10 steps. Then switch to a new one.

        if self.current_intermediate_target == self.memory.pos: # we have reached this intermediate target
            self.current_intermediate_target = None
        
        how_long_to_follow_intermediate_target = 150
        how_long_to_follow_intermediate_target = max(self.memory.memory[self.memory.pos[0]][self.memory.pos[1]].visited * 2, 5)
        if self.current_intermediate_target and self.current_intermediate_target_age < how_long_to_follow_intermediate_target:
            self.current_intermediate_target_age += 1 # increment the age
        else: # update the intermediate target
            self.current_intermediate_target_age = 0
            self.current_intermediate_target = self.choose_intermediate_target_node(minDistanceArray)

        path = reconstruct_path(parent, self.current_intermediate_target)
        if path and len(path) > 1:
            # Case 1: We know the end position and we can reach it. Follow path..
                next_move = self.get_move_direction(path)
                print(path)
                print("Want to make next move: ", next_move)
                if is_move_valid(next_move, current_percept.maze_state):
                    self.memory.update_pos(next_move)
                    print("New Pos: ", self.memory.pos)
                    return next_move
                else: 
                    print("Desired Next Move is Invalid. Waiting.")
                    return constants.WAIT
        return constants.WAIT
    

    def choose_intermediate_target_node(self, min_dist_array):
        options =  {}
        for y in range(len(min_dist_array)):
            for x, dist in enumerate(min_dist_array[y]):
                if dist < float("inf"):
                    # Find new Squares
                    options[(y, x)] = {"dist": dist, "euclidean_dist": math.sqrt((y - self.memory.pos[0]) ** 2 + (x - self.memory.pos[1]) ** 2), "final_score": 0}
        
        most_new_visible_squares = 0
        best_new_pos = self.memory.pos
        boundary: Boundary = self.memory.get_boundary_coords()
        for new_pos in options:
            num_unseen_squares = len(self.get_unseen_squares(new_pos, boundary))
            options[new_pos]["num_unseen"] = num_unseen_squares
        
        best_new_pos = self.generate_best_option(options, min_dist_array)
            # if num_unseen_squares > most_new_visible_squares:
            #     most_new_visible_squares = num_unseen_squares
            #     best_new_pos = new_pos

        return best_new_pos
    
    def generate_best_option(self, options, min_dist_array):
        # This function is the "brain" of exploring
        # The weights are arbitrary at the moment
        best_score = 0
        best_pos = self.memory.pos
        unseen_weight = 2
        distance_weight = 1
        time_weight = 1
        min_min_dist = np.min(min_dist_array)
        max_min_dist = np.max([i for i in np.reshape(min_dist_array, -1) if i < float("inf")])
        for pos in options:
            final_score = (
                # Normalizing each factor
                unseen_weight * (options[pos]["num_unseen"] - 0) / (3 * self.radius**2) + 
                # The more times we visited the current square - the less we encourage going far
                distance_weight * (options[pos]["euclidean_dist"] - 0) / (self.radius) / (1 + self.memory.memory[self.memory.pos[0]][self.memory.pos[1]].visited) -
                (time_weight * (options[pos]["dist"] - min_min_dist) / (max_min_dist))
            )
            options[pos]["final_score"] = final_score
            if final_score > best_score:
                best_score = final_score
                best_pos = pos

        print("best pos: ", best_pos)
        return best_pos

    
    # def find_min_time_max_dist(self, options):
    #     best = self.memory.pos
    #     best_dist = 0
    #     for coord, dist in options.items():
    #         newdist = np.linalg.norm(np.array(coord) - np.array(self.memory.pos)) / (dist + 1)
    #         if newdist > best_dist and not self.memory.memory[coord[0]][coord[1]].visited:
    #             best = coord
    #             best_dist = newdist
    #     return (best[0] - self.memory.pos[0], best[1] - self.memory.pos[1])

    def get_unseen_squares(self, pos, boundary: Boundary):
        squares = self.get_visible_squares_at_pos(pos, boundary)
        not_seen_squares = []

        if squares:
            for y in range(len(squares)):
                for x in range(len(squares[y])):
                    square = squares[y][x]
                    if not square.seen:  # Add only unseen squares within boundaries
                        not_seen_squares.append((y, x))
        return not_seen_squares

    def get_visible_squares_at_pos(self, pos, boundary: Boundary):
        """
        Get the visible squares at the current position
        NOTE: THIS MAKES SURE THEY ARE WITHIN THE BOUNDARY
        """
        y, x = pos
        visible_squares = []
        
        y_start = max(y - self.radius, boundary.up if boundary.is_vertical_boundary_known() else y - self.radius)
        y_end = min(y + self.radius, boundary.down if boundary.is_vertical_boundary_known() else y + self.radius)
        x_start = max(x - self.radius, boundary.left if boundary.is_horizontal_boundary_known() else x - self.radius)
        x_end = min(x + self.radius, boundary.right if boundary.is_horizontal_boundary_known() else x + self.radius)

        # A square is only visible if it is inside the boundary
        # for row in self.memory.memory[y_start:y_end]:
        #     visible_squares.append(row[x_start:x_end])

        # A square is only visible if it is inside the boundary and within the Euclidean distance of 'r'
        for row in range(y_start, y_end + 1):
            visible_row = []
            for col in range(x_start, x_end + 1):
                # Calculate the Euclidean distance from the current position to the square (row, col)
                distance = math.sqrt((row - y) ** 2 + (col - x) ** 2)

                # If the square is within the visibility radius and boundary, add it to visible squares
                if distance <= self.radius:
                    visible_row.append(self.memory.memory[row][col])

            visible_squares.append(visible_row) 

        return visible_squares

    def get_move_direction(self, path): #current to next position
        """        
        Args:
            path (tuple): A tuple of tuples containing the path from current position to the target position.
        
        Returns:
            int: Direction of movement (LEFT, UP, RIGHT, DOWN).
        """

        # this is the DY, DX from the Min distance array
        dy = path[1][0] - path[0][0]
        dx = path[1][1] - path[0][1]
        # Convert this to the direction
        if dx == -1 and dy == 0:
            return constants.LEFT
        elif dx == 1 and dy == 0:
            return constants.RIGHT
        elif dx == 0 and dy == -1:
            return constants.UP
        elif dx == 0 and dy == 1:
            return constants.DOWN
        else:
            return constants.WAIT


    def get_unexplored_nodes(self, current_percept): #Placeholder 
        unexplored_nodes = []
        return unexplored_nodes


def print_min_dist_array(minDistanceArray, start_row, end_row, start_col, end_col, width=4):
    for y in range(len(minDistanceArray)):
        if y >= start_row and y <= end_row:
            row = minDistanceArray[y]
            for x in range(len(row)):
                if x >= start_col and x <= end_col:
                    # Print each element with a fixed width
                    print(f"{row[x]:>{width}}", end=" ")
            print()

        
# THIS CODE CAN HELP WITH DOING STUFF WITH THE BOUNDARY:

    # if any(bound != -1 for bound in boundary):
    #     left_bound, right_bound, up_bound, down_bound = boundary
    #     # We can see a boundary. Use this and our current position to determine where to go.
    #     # Note: If we know the left boundary, we also know the right one (and vice versa). Same for up and down
    #     # We should move away from boundary so we can maximize our view.
    #     if up_bound > -1 and down_bound > -1:
    #         # we know left and right bounds. How far do we want to be from the bounds? radius - 1 maybe?
    #         current_y = self.memory.pos[0]
    #         distance_to_up_bound = current_y - up_bound
    #         distance_to_down_bound = down_bound - current_y

    #         if distance_to_up_bound < self.radius - 1: # The plus one is to make sure we see corners
    #             # We are too close to the upper bound. Move down
    #             desired_y = up_bound + self.radius - 1
    #         elif distance_to_down_bound < self.radius - 1:
    #             # We are too close to the lower bound. Move up
    #             desired_y = down_bound - self.radius + 1

    #     if left_bound > -1 and right_bound > -1:
    #         # we know left and right bounds. How far do we want to be from the bounds? radius - 1 maybe?
    #         current_x = self.memory.pos[1]
    #         distance_to_left_bound = current_x - left_bound
    #         distance_to_right_bound = right_bound - current_x

    #         if distance_to_left_bound < self.radius - 1: # The plus one is to make sure we see corners
    #             # We are too close to the left bound. Move right
    #             desired_x = left_bound + self.radius - 1
    #         elif distance_to_right_bound < self.radius - 1:
    #             # We are too close to the right bound. Move left
    #             desired_x = right_bound - self.radius + 1