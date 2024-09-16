import os
import pickle
import numpy as np
import logging

import constants
from timing_maze_state import TimingMazeState

# from gridworld import GridWorld
# from qtable import QTable
# from q_policy import QPolicy
# from multi_armed_bandit.ucb import UpperConfidenceBounds
from sympy import divisors
from collections import defaultdict

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
        self.turn = 0
        self.frequencies_per_cell = defaultdict(
            lambda: set(range(maximum_door_frequency + 1))
        )
        self.have_seen_target = False
        self.target_x = None
        self.target_y = None
        self.random_horizontal_exploration_direction = self.rng.choice([constants.LEFT, constants.RIGHT])
        self.horizontal_search_is_complete = False
        self.random_vertical_exploration_direction = self.rng.choice([constants.UP, constants.DOWN])
        self.vertical_search_is_complete = False
        self.left_wall_pos = None
        self.right_wall_pos = None
        self.up_wall_pos = None
        self.down_wall_pos = None
        self.corner_found = False
        self.threshold = 3

    def move_random_vertically_or_wait(self, current_percept, direction) -> int:
        random_vertical_exploration_direction = self.rng.choice([constants.UP, constants.DOWN])
        if random_vertical_exploration_direction == constants.DOWN:
            if direction[constants.DOWN] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                            and maze_state[3] == constants.OPEN):
                        return constants.DOWN
            if direction[constants.UP] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                            and maze_state[3] == constants.OPEN):
                        return constants.UP
        else:
            if direction[constants.UP] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                            and maze_state[3] == constants.OPEN):
                        return constants.UP
            if direction[constants.DOWN] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                            and maze_state[3] == constants.OPEN):
                        return constants.DOWN
        return constants.WAIT
    
    def move_random_horizontally_or_wait(self, current_percept, direction) -> int:
        random_horizontal_exploration_direction = self.rng.choice([constants.LEFT, constants.RIGHT])
        if random_horizontal_exploration_direction == constants.LEFT:
            if direction[constants.LEFT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                            and maze_state[3] == constants.OPEN):
                        return constants.LEFT
            if direction[constants.RIGHT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                            and maze_state[3] == constants.OPEN):
                        return constants.RIGHT
        else:
            if direction[constants.RIGHT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                            and maze_state[3] == constants.OPEN):
                        return constants.RIGHT
            if direction[constants.LEFT] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                            and maze_state[3] == constants.OPEN):
                        return constants.LEFT
        return constants.WAIT

    def move_up_if_open(self, current_percept, direction) -> int:
        if direction[constants.UP] == constants.OPEN:
                for maze_state in current_percept.maze_state:
                    if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                            and maze_state[3] == constants.OPEN):
                        return constants.UP
        return self.move_random_horizontally_or_wait(current_percept, direction)
                    
    def move_down_if_open(self, current_percept, direction) -> int:
        if direction[constants.DOWN] == constants.OPEN:
            for maze_state in current_percept.maze_state:
                if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                        and maze_state[3] == constants.OPEN):
                    return constants.DOWN
        return self.move_random_horizontally_or_wait(current_percept, direction)
                
                            
    def move_left_if_open(self, current_percept, direction) -> int:
        if direction[constants.LEFT] == constants.OPEN:
            for maze_state in current_percept.maze_state:
                if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                        and maze_state[3] == constants.OPEN):
                    return constants.LEFT
        return self.move_random_vertically_or_wait(current_percept, direction)


    def move_right_if_open(self, current_percept, direction) -> int:
        print(direction)
        if direction[constants.RIGHT] == constants.OPEN:
            print("RIGHT")
            for maze_state in current_percept.maze_state:
                if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                        and maze_state[3] == constants.OPEN):
                    return constants.RIGHT
        return self.move_random_vertically_or_wait(current_percept, direction)

    
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

        print("Right Wall Pos:" + str(self.right_wall_pos))
        print("Left Wall Pos:" + str(self.left_wall_pos))
        print("Up Wall Pos:" + str(self.up_wall_pos))
        print("Down Wall Pos:" + str(self.down_wall_pos))

        curr_x, curr_y = -current_percept.start_x, -current_percept.start_y # Our current position relative to the start cell
        self.turn += 1
        factors = set(divisors(self.turn))
        for dX, dY, door, state in current_percept.maze_state:
            #print(curr_x + dX, curr_y + dY, door, state)
            if state == constants.CLOSED:
                self.frequencies_per_cell[(curr_x + dX, curr_y + dY, door)] -= factors
            elif state == constants.OPEN:
                self.frequencies_per_cell[(curr_x + dX, curr_y + dY, door)] &= factors
            elif state == constants.BOUNDARY:
                print(dX, dY, door, state)
                # set to a set with single value of 0
                self.frequencies_per_cell[(curr_x + dX, curr_y + dY, door)] = {0}
                # if self.down_wall_pos is None and door == constants.DOWN:
                #     self.down_wall_pos = curr_y + dY
                #     self.up_wall_pos = self.down_wall_pos - 100
                #     self.vertical_search_is_complete = True
                # if self.up_wall_pos is None and door == constants.UP:
                #     self.up_wall_pos = curr_y + dY
                #     self.down_wall_pos = self.up_wall_pos + 100
                #     self.vertical_search_is_complete = True
                # if self.right_wall_pos is None and door == constants.RIGHT:
                #     print("Curr Pos: " + str(curr_x))
                #     print("dX: " + str(dX))
                #     self.right_wall_pos = curr_x + dX
                #     self.left_wall_pos = self.right_wall_pos - 100
                #     self.horizontal_search_is_complete = True
                # if self.left_wall_pos is None and door == constants.LEFT:
                #     self.left_wall_pos = curr_x + dX
                #     self.right_wall_pos = self.left_wall_pos + 100
                #     print("Curr Pos: " + str(curr_x))
                #     print("dX: " + str(dX))
                #     self.horizontal_search_is_complete = True
                if self.down_wall_pos is None and door == 2:
                    self.down_wall_pos = curr_y + dY
                    self.up_wall_pos = self.down_wall_pos - 100
                    self.vertical_search_is_complete = True
                if self.up_wall_pos is None and door == 0:
                    self.up_wall_pos = curr_y + dY
                    self.down_wall_pos = self.up_wall_pos + 100
                    self.vertical_search_is_complete = True
                if self.right_wall_pos is None and door == 3:
                    print("Curr Pos: " + str(curr_x))
                    print("dX: " + str(dX))
                    self.right_wall_pos = curr_x + dX
                    self.left_wall_pos = self.right_wall_pos - 100
                    self.horizontal_search_is_complete = True
                if self.left_wall_pos is None and door == 1:
                    self.left_wall_pos = curr_x + dX
                    self.right_wall_pos = self.left_wall_pos + 100
                    print("Curr Pos: " + str(curr_x))
                    print("dX: " + str(dX))
                    self.horizontal_search_is_complete = True
                # if abs(dX) <= self.radius:
                #     print("Horizontal Search Complete")
                #     self.horizontal_search_is_complete = True
                # if abs(dY) <= self.radius:
                #     print("Vertical Search Complete")
                #     self.vertical_search_is_complete = True

        if current_percept.is_end_visible and not self.have_seen_target:
            self.have_seen_target = True
            self.horizontal_search_is_complete = True
            self.vertical_search_is_complete = True
            self.target_x = current_percept.end_x - current_percept.start_x # Target position relative to start cell
            self.target_y = current_percept.end_y - current_percept.start_y

        direction = [0, 0, 0, 0]
        for maze_state in current_percept.maze_state:
            if maze_state[0] == 0 and maze_state[1] == 0: # If the cell is the current cell
                direction[maze_state[2]] = maze_state[3] # Set the direction to the state of the door

        if self.have_seen_target:
            if abs(current_percept.end_x) >= abs(current_percept.end_y):
                if current_percept.end_x > 0 and direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                                and maze_state[3] == constants.OPEN):
                            return constants.RIGHT
                if current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                                and maze_state[3] == constants.OPEN):
                            return constants.LEFT
                if current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                                and maze_state[3] == constants.OPEN):
                            return constants.UP
                if current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                                and maze_state[3] == constants.OPEN):
                            return constants.DOWN
                return constants.WAIT
            else:
                if current_percept.end_y < 0 and direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN
                                and maze_state[3] == constants.OPEN):
                            return constants.UP
                if current_percept.end_y > 0 and direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP
                                and maze_state[3] == constants.OPEN):
                            return constants.DOWN
                if current_percept.end_x > 0 and direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT
                                and maze_state[3] == constants.OPEN):
                            return constants.RIGHT
                if current_percept.end_x < 0 and direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if (maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT
                                and maze_state[3] == constants.OPEN):
                            return constants.LEFT
                return constants.WAIT
        # If we have not seen the target yet, we will explore the maze, starting from horizontal search
        else:
            print("ENTERED HERE")
            if not self.horizontal_search_is_complete:
                print("ENTERED HERE")
                if self.random_horizontal_exploration_direction == constants.LEFT:
                    print("Moving Left")
                    a = self.move_left_if_open(current_percept, direction)
                    print("Printing a")
                    print(a)
                    return a
                else:
                    print("Moving Right")
                    a = self.move_right_if_open(current_percept, direction)
                    print("Printing a")
                    print(a)
                    return a
            # If horizontal search is complete, we will start vertical search
            elif not self.vertical_search_is_complete:
                if self.random_vertical_exploration_direction == constants.UP:
                    return self.move_up_if_open(current_percept, direction)
                else:
                    return self.move_down_if_open(current_percept, direction)
            else: # If both horizontal and vertical search are complete, we will move towards the corner
            # THE REST OF THE CODE IS NOT WORKING AS INTENDED
                distances_to_corners = {
                    "top_left": np.sqrt((self.left_wall_pos - current_percept.current_pos_x)**2 + (self.up_wall_pos - current_percept.current_pos_y)**2),
                    "top_right": np.sqrt((self.right_wall_pos - current_percept.current_pos_x)**2 + (self.up_wall_pos - current_percept.current_pos_y)**2),
                    "bottom_left": np.sqrt((self.left_wall_pos - current_percept.current_pos_x)**2 + (self.down_wall_pos - current_percept.current_pos_y)**2),
                    "bottom_right": np.sqrt((self.right_wall_pos - current_percept.current_pos_x)**2 + (self.down_wall_pos - current_percept.current_pos_y)**2)
                }
                
                # Find the nearest corner
                nearest_corner = min(distances_to_corners, key=distances_to_corners.get)
                print(nearest_corner)

                if distances_to_corners[nearest_corner] > self.radius:
                    print("Moving Diagonally")
                    return self.move_diagonally(current_percept, direction, nearest_corner)
                else:
                    self.corner_found = True
                    print("Corner Found")
                    print(nearest_corner)
                    return self.move_from_corner(current_percept, direction, nearest_corner)
    
    def move_diagonally(self, current_percept, direction, corner) -> int:
        """Move towards the specified corner by checking diagonal movement options.

        Args:
            current_percept (TimingMazeState): Contains current state information.
            direction (list): The available directions from the current position.
            corner (str): The nearest corner to move towards.
        
        Returns:
            int: The move decision based on the available directions.
        """
        if corner == "top_left":
            if self.rng.choice([constants.UP, constants.LEFT]) == constants.LEFT:  # Randomly choose between the two
                if direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT and maze_state[3] == constants.OPEN:
                            return constants.LEFT
                if direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN and maze_state[3] == constants.OPEN:
                            return constants.UP
            else:
                if direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN and maze_state[3] == constants.OPEN:
                            return constants.UP
                if direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT and maze_state[3] == constants.OPEN:
                            return constants.LEFT
            print("WAIT BECAUSE IT IS NOT POSSIBLE TO MOVE ANY DIAGONALLY TOWARDS CORNER")
            return constants.WAIT
            
        elif corner == "top_right":
            if self.rng.choice([constants.UP, constants.RIGHT]) == constants.RIGHT:
                if direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT and maze_state[3] == constants.OPEN:
                            return constants.RIGHT
                if direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN and maze_state[3] == constants.OPEN:
                            return constants.UP
            else:
                if direction[constants.UP] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN and maze_state[3] == constants.OPEN:
                            return constants.UP
                if direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT and maze_state[3] == constants.OPEN:
                            return constants.RIGHT
            print("WAIT BECAUSE IT IS NOT POSSIBLE TO MOVE ANY DIAGONALLY TOWARDS CORNER")
            return constants.WAIT
        
        elif corner == "bottom_left":
            if self.rng.choice([constants.DOWN, constants.LEFT]) == constants.LEFT:
                if direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT and maze_state[3] == constants.OPEN:
                            return constants.LEFT
                if direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP and maze_state[3] == constants.OPEN:
                            return constants.DOWN
            else:
                if direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP and maze_state[3] == constants.OPEN:
                            return constants.DOWN
                if direction[constants.LEFT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT and maze_state[3] == constants.OPEN:
                            return constants.LEFT
            print("WAIT BECAUSE IT IS NOT POSSIBLE TO MOVE ANY DIAGONALLY TOWARDS CORNER")
            return constants.WAIT
    
        elif corner == "bottom_right":
            if self.rng.choice([constants.DOWN, constants.RIGHT]) == constants.RIGHT:
                if direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT and maze_state[3] == constants.OPEN:
                            return constants.RIGHT
                if direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP and maze_state[3] == constants.OPEN:
                            return constants.DOWN
            else:
                if direction[constants.DOWN] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP and maze_state[3] == constants.OPEN:
                            return constants.DOWN
                if direction[constants.RIGHT] == constants.OPEN:
                    for maze_state in current_percept.maze_state:
                        if maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT and maze_state[3] == constants.OPEN:
                            return constants.RIGHT
            print("WAIT BECAUSE IT IS NOT POSSIBLE TO MOVE ANY DIAGONALLY TOWARDS CORNER")
            return constants.WAIT


    def move_from_corner(self, current_percept, direction, corner) -> int:
        print("TO DO")
        return constants.WAIT