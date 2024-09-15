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
        # The condition of the four doors at the current cell
        direction = [0, 0, 0, 0] # [left, up, right, down]; 1 is closed, 2 is open, 3 is boundary
        for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
            if maze_state[0] == 0 and maze_state[1] == 0: # (0,0) is the current loc; -> Looking to see the conditions of the four doors at the current location.
                direction[maze_state[2]] = maze_state[3] # import that information into direction

        if current_percept.is_end_visible:
            if abs(current_percept.end_x) >= abs(current_percept.end_y): # huhh???
                if (current_percept.end_x > 0 # if goal is on the right side
                    and direction[constants.RIGHT] == constants.OPEN # if the door on the right is open
                    ):
                    for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
                        if (maze_state[0] == 1 and maze_state[1] == 0 # (1,0) is the cell on the right; -> Looking to see the conditions of the four doors at the cell on the right.
                            and maze_state[2] == constants.LEFT and maze_state[3] == constants.OPEN # if the left door of the cell on the right (the adjacent door to the current cell) is open
                            ):
                            return constants.RIGHT # goes right -> returning 2
                        
                if (current_percept.end_x < 0 # if goal is on the left side
                    and direction[constants.LEFT] == constants.OPEN # if the door on the left is open
                    ):
                    for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
                        if (maze_state[0] == -1 and maze_state[1] == 0 # (-1,0) is the cell on the left; -> Looking to see the conditions of the four doors at the cell on the left.
                            and maze_state[2] == constants.RIGHT and maze_state[3] == constants.OPEN # if the right door of the cell on the left (the adjacent door to the current cell) is open
                            ):
                            return constants.LEFT # goes left -> returning 0
                        
                if (current_percept.end_y < 0 # if goal is above
                    and direction[constants.UP] == constants.OPEN # if the door above is open
                    ):
                    for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
                        if (maze_state[0] == 0 and maze_state[1] == -1 # (0,-1) is the cell above; -> Looking to see the conditions of the four doors at the cell above.
                            and maze_state[2] == constants.DOWN and maze_state[3] == constants.OPEN # if the down door of the cell above (the adjacent door to the current cell) is open
                            ):
                            return constants.UP # goes up -> returning 1
                        
                if (current_percept.end_y > 0 # if goal is below
                    and direction[constants.DOWN] == constants.OPEN # if the door below is open
                    ):
                    for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
                        if (maze_state[0] == 0 and maze_state[1] == 1 # (0,1) is the cell below; -> Looking to see the conditions of the four doors at the cell below.
                            and maze_state[2] == constants.UP and maze_state[3] == constants.OPEN # if the above door of the cell below (the adjacent door to the current cell) is open
                            ): 
                            return constants.DOWN # goes down -> returning 3
                        
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
            
        else: # If End is not visible
            if direction[constants.LEFT] == constants.OPEN: # If left door is open
                for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
                    if (maze_state[0] == -1 and maze_state[1] == 0 # (-1,0) is the cell on the left; -> Looking to see the conditions of the four doors at the cell on the left.
                        and maze_state[2] == constants.RIGHT and maze_state[3] == constants.OPEN # if the right door of the cell on the left (the adjacent door to the current cell) is open
                        ):
                        return constants.LEFT #return 0
                    
            if direction[constants.DOWN] == constants.OPEN: # If down door is open
                for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
                    if (maze_state[0] == 0 and maze_state[1] == 1 # (0,1) is the cell below; -> Looking to see the conditions of the four doors at the cell below.
                        and maze_state[2] == constants.UP and maze_state[3] == constants.OPEN # if the above door of the cell below (the adjacent door to the current cell) is open
                        ):
                        return constants.DOWN # return 3
                    
            if direction[constants.RIGHT] == constants.OPEN: # If right door is open
                for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
                    if (maze_state[0] == 1 and maze_state[1] == 0 # (1,0) is the cell on the right; -> Looking to see the conditions of the four doors at the cell on the right.
                        and maze_state[2] == constants.LEFT and maze_state[3] == constants.OPEN # if the left door of the cell on the right (the adjacent door to the current cell) is open
                        ):
                        return constants.RIGHT # return 2
                    
            if direction[constants.UP] == constants.OPEN: # If up door is open
                for maze_state in current_percept.maze_state: # looping through all the cells visible by the drone
                    if (maze_state[0] == 0 and maze_state[1] == -1 # (0,-1) is the cell above; -> Looking to see the conditions of the four doors at the cell above.
                        and maze_state[2] == constants.DOWN and maze_state[3] == constants.OPEN # if the down door of the cell above (the adjacent door to the current cell) is open
                        ):
                        return constants.UP # return 1
                    
            return constants.WAIT
