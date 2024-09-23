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

        self.stuck_for_rounds_count = 0
        self.stuck_threshold = 10
        self.previous_position = None

        self.recently_seen_positions_list = []
        self.recently_seen_positions_list_size = 25
        self.stuck_seeing_same_positions_threshold = 4

        self.random_movements_started = False
        self.number_of_random_moves = 0
        self.number_of_random_moves_threshold = 20

        self.location_of_first_corner_to_visit = None

        self.corners_to_visit = None
        self.which_corner_to_visit = 0
        self.seen_in_random_walk = set()
        


    def set_target(self, current_percept, direction) -> tuple:

        curr_x, curr_y = -current_percept.start_x, -current_percept.start_y # Our current position relative to the start cell
        
        if self.previous_position is not None and self.previous_position == (curr_x, curr_y):
            self.stuck_for_rounds_count += 1
        else:
            self.stuck_for_rounds_count = 0
            self.previous_position = (curr_x, curr_y)
        
        self.recently_seen_positions_list.insert(0, (curr_x, curr_y))  # unshift equivalent in Python is insert(0, element)
        if len(self.recently_seen_positions_list) > self.recently_seen_positions_list_size:
            print("Popping from recently seen positions list")
            self.recently_seen_positions_list.pop()    
            
        self.turn += 1
        factors = set(divisors(self.turn))
        for dX, dY, door, state in current_percept.maze_state:
            #print(curr_x + dX, curr_y + dY, door, state)
            if state == constants.CLOSED:
                self.frequencies_per_cell[(curr_x + dX, curr_y + dY, door)] -= factors
            elif state == constants.OPEN:
                self.frequencies_per_cell[(curr_x + dX, curr_y + dY, door)] &= factors
            elif state == constants.BOUNDARY:
                # print(dX, dY, door, state)
                # set to a set with single value of 0
                self.frequencies_per_cell[(curr_x + dX, curr_y + dY, door)] = {0}

                if self.down_wall_pos is None and door == constants.DOWN:
                    print("Down Wall Seen at position: " + str(curr_y + dY))
                    print("Vertical Search Complete")
                    self.down_wall_pos = curr_y + dY
                    self.up_wall_pos = self.down_wall_pos - 100
                    self.vertical_search_is_complete = True
                if self.up_wall_pos is None and door == constants.UP:
                    print("Up Wall Seen at position: " + str(curr_y + dY))
                    print("Vertical Search Complete")
                    self.up_wall_pos = curr_y + dY
                    self.down_wall_pos = self.up_wall_pos + 100
                    self.vertical_search_is_complete = True
                if self.right_wall_pos is None and door == constants.RIGHT:
                    print("Right Wall Seen at position: " + str(curr_x + dX))
                    print("Horizontal Search Complete")
                    print("Curr Pos: " + str(curr_x))
                    self.right_wall_pos = curr_x + dX
                    self.left_wall_pos = self.right_wall_pos - 100
                    self.horizontal_search_is_complete = True
                if self.left_wall_pos is None and door == constants.LEFT:
                    print("Left Wall Seen at position: " + str(curr_x + dX))
                    print("Horizontal Search Complete")
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
            print("Target Seen, so horizontal and vertical search are complete")
            self.horizontal_search_is_complete = True
            self.vertical_search_is_complete = True
            self.target_x = current_percept.end_x - current_percept.start_x # Target position relative to start cell
            self.target_y = current_percept.end_y - current_percept.start_y

        if self.have_seen_target:
            return (self.target_x, self.target_y)
        
        """        if not self.horizontal_search_is_complete or not self.vertical_search_is_complete:
            if self.stuck_for_rounds_count >= self.stuck_threshold:
                if not self.horizontal_search_is_complete:
                    print("Stuck for too long, switching random horizontal exploration direction")
                    self.random_horizontal_exploration_direction = self.switch_random_exploration_direction(self.random_horizontal_exploration_direction)
                    self.stuck_for_rounds_count = 0
                elif not self.vertical_search_is_complete:
                    print("Stuck for too long, switching random vertical exploration direction")
                    self.random_vertical_exploration_direction = self.switch_random_exploration_direction(self.random_vertical_exploration_direction)
                    self.stuck_for_rounds_count = 0
            elif len(set(self.recently_seen_positions_list)) <= self.stuck_seeing_same_positions_threshold and len(self.recently_seen_positions_list) == self.recently_seen_positions_list_size:
                print("Stuck for too long in same cells, switching random exploration directions")
                self.random_horizontal_exploration_direction = self.switch_random_exploration_direction(self.random_horizontal_exploration_direction)
                self.random_vertical_exploration_direction = self.switch_random_exploration_direction(self.random_vertical_exploration_direction)
        """
        if not self.horizontal_search_is_complete:
            if self.random_horizontal_exploration_direction == constants.LEFT:
                return (-self.radius, 0)
            elif self.random_horizontal_exploration_direction == constants.RIGHT:
                return (self.radius, 0)
        elif not self.vertical_search_is_complete:
            if self.random_vertical_exploration_direction == constants.UP:
                return (0, -self.radius)
            elif self.random_vertical_exploration_direction == constants.DOWN:
                return (0, self.radius)
        else:
            distances_to_corners = {
            "top_left": np.sqrt((self.left_wall_pos - curr_x)**2 + (self.up_wall_pos - curr_y)**2),
            "top_right": np.sqrt((self.right_wall_pos - curr_x)**2 + (self.up_wall_pos - curr_y)**2),
            "bottom_left": np.sqrt((self.left_wall_pos - curr_x)**2 + (self.down_wall_pos - curr_y)**2),
            "bottom_right": np.sqrt((self.right_wall_pos - curr_x)**2 + (self.down_wall_pos - curr_y)**2)
            }
            corner_locations_in_decreasing_order_of_distance = sorted(
            {
                (self.left_wall_pos, self.up_wall_pos): np.sqrt((self.left_wall_pos - curr_x)**2 + (self.up_wall_pos - curr_y)**2),   # Top-left
                (self.right_wall_pos, self.up_wall_pos): np.sqrt((self.right_wall_pos - curr_x)**2 + (self.up_wall_pos - curr_y)**2),  # Top-right
                (self.left_wall_pos, self.down_wall_pos): np.sqrt((self.left_wall_pos - curr_x)**2 + (self.down_wall_pos - curr_y)**2), # Bottom-left
                (self.right_wall_pos, self.down_wall_pos): np.sqrt((self.right_wall_pos - curr_x)**2 + (self.down_wall_pos - curr_y)**2) # Bottom-right
            }.items(),
            key=lambda x: x[1],  # Sort by the distance values
            reverse=True  # Sort in decreasing order of distance
            )
            corner_locations_in_increasing_order_of_distance = sorted(
            {
                (self.left_wall_pos, self.up_wall_pos): np.sqrt((self.left_wall_pos - curr_x)**2 + (self.up_wall_pos - curr_y)**2),   # Top-left
                (self.right_wall_pos, self.up_wall_pos): np.sqrt((self.right_wall_pos - curr_x)**2 + (self.up_wall_pos - curr_y)**2),  # Top-right
                (self.left_wall_pos, self.down_wall_pos): np.sqrt((self.left_wall_pos - curr_x)**2 + (self.down_wall_pos - curr_y)**2), # Bottom-left
                (self.right_wall_pos, self.down_wall_pos): np.sqrt((self.right_wall_pos - curr_x)**2 + (self.down_wall_pos - curr_y)**2) # Bottom-right
            }.items(),
            key=lambda x: x[1],  # Sort by the distance values
            reverse=False  # Sort in increasing order of distance
            )

            # Find the nearest corner
            nearest_corner = min(distances_to_corners, key=distances_to_corners.get)
            print(nearest_corner)
            
            if self.location_of_first_corner_to_visit is None:
                self.location_of_first_corner_to_visit = corner_locations_in_increasing_order_of_distance[0][0]

            print("Radius: " + str(self.radius))
            print("Distance to Nearest Corner: " + str(distances_to_corners[nearest_corner]))

            if self.corner_found == False and distances_to_corners[nearest_corner] > self.radius:
                print("Moving towards nearest corner")
                return self.location_of_first_corner_to_visit
            elif self.corner_found == False and distances_to_corners[nearest_corner] <= self.radius:
                print("Corner Found")
                print("Corner: " + nearest_corner)
                self.corner_found = True
                if self.corners_to_visit is None:
                    # Dictionary containing corner locations based on wall positions
                    # Extract only the corner locations in the sorted order
                    self.corners_to_visit = [corner[0] for corner in corner_locations_in_decreasing_order_of_distance]
            else: # Corner already found
                # if distance between corners_to_visit[self.which_corner_to_visit] and current position is less than radius, move to next corner
                if np.sqrt((self.corners_to_visit[self.which_corner_to_visit][0] - curr_x)**2 + (self.corners_to_visit[self.which_corner_to_visit][1] - curr_y)**2) <= self.radius:
                    self.which_corner_to_visit += 1
                    if self.which_corner_to_visit == 4:
                        self.which_corner_to_visit = 0
                    print("Moving to next corner")
                    return self.corners_to_visit[self.which_corner_to_visit]
                return self.corners_to_visit[self.which_corner_to_visit]
            
    def move_random_open_direction(self, direction, current_percept) -> int:
        """Move in a random direction that is open with a fresh random generator each time.

        Args:
            current_percept (TimingMazeState): Contains the current state information.
            direction (list): A list of the current available directions, where constants.OPEN denotes an open direction.
        
        Returns:
            int: The chosen direction to move or constants.WAIT if no open direction is available.
        """
        # Create a fresh random number generator instance for this function call
        fresh_rng = np.random.default_rng()
        curr_x, curr_y = -current_percept.start_x, -current_percept.start_y
        self.seen_in_random_walk.add((curr_x, curr_y))

        open_directions = [constants.WAIT]
        open_unseen_directions = []

        if direction[constants.LEFT] == constants.OPEN:
            open_directions.append(constants.LEFT)
            if (curr_x - 1, curr_y) not in self.seen_in_random_walk:
                open_unseen_directions.append(constants.LEFT)
        if direction[constants.RIGHT] == constants.OPEN:
            open_directions.append(constants.RIGHT)
            if (curr_x + 1, curr_y) not in self.seen_in_random_walk:
                open_unseen_directions.append(constants.RIGHT)
        if direction[constants.UP] == constants.OPEN:
            open_directions.append(constants.UP)
            if (curr_x, curr_y - 1) not in self.seen_in_random_walk:
                open_unseen_directions.append(constants.UP)
        if direction[constants.DOWN] == constants.OPEN:
            open_directions.append(constants.DOWN)
            if (curr_x, curr_y + 1) not in self.seen_in_random_walk:
                open_unseen_directions.append(constants.DOWN)

        if open_unseen_directions:
            random_direction = fresh_rng.choice(open_unseen_directions)  # Randomly select an open unseen direction with fresh randomness
            print("Random Unseen Direction: " + str(random_direction))
            return int(random_direction)
        elif open_directions:
            print(open_directions)
            random_direction = fresh_rng.choice(open_directions)  # Randomly select an open direction with fresh randomness
            print("Random Direction: " + str(random_direction))
            return int(random_direction)


    def switch_random_exploration_direction(self, random_direction) -> int:
        if random_direction == constants.LEFT:
            return constants.RIGHT
        elif random_direction == constants.RIGHT:
            return constants.LEFT
        elif random_direction == constants.UP:
            return constants.DOWN
        elif random_direction == constants.DOWN:
            return constants.UP
    
    def move_random_vertically_or_wait(self, current_percept, direction) -> int:
        random_vertical_exploration_direction = self.rng.choice([constants.UP, constants.DOWN])
        if random_vertical_exploration_direction == constants.DOWN:
            if direction[constants.DOWN] == constants.OPEN:
                return constants.DOWN
            if direction[constants.UP] == constants.OPEN:
                return constants.UP
        else:
            if direction[constants.UP] == constants.OPEN:
                return constants.UP
            if direction[constants.DOWN] == constants.OPEN:
                return constants.DOWN
        print("WAIT BECAUSE IT IS NOT POSSIBLE TO MOVE ANY RANDOMLY VERTICALLY")
        return constants.WAIT
    
    def move_random_horizontally_or_wait(self, current_percept, direction) -> int:
        random_horizontal_exploration_direction = self.rng.choice([constants.LEFT, constants.RIGHT])
        if random_horizontal_exploration_direction == constants.LEFT:
            if direction[constants.LEFT] == constants.OPEN:
                return constants.LEFT
            if direction[constants.RIGHT] == constants.OPEN:
                return constants.RIGHT
        else:
            if direction[constants.RIGHT] == constants.OPEN:
                return constants.RIGHT
            if direction[constants.LEFT] == constants.OPEN:
                return constants.LEFT
        print("WAIT BECAUSE IT IS NOT POSSIBLE TO MOVE ANY RANDOMLY HORIZONTALLY")
        return constants.WAIT

    def move_up_if_open(self, current_percept, direction) -> int:
        print("TRYING UP")
        if direction[constants.UP] == constants.OPEN:
            return constants.UP
        print("TRYING RANDOM HORIZONTAL")
        return self.move_random_horizontally_or_wait(current_percept, direction)
                    
    def move_down_if_open(self, current_percept, direction) -> int:
        print("TRYING DOWN")
        if direction[constants.DOWN] == constants.OPEN:
            return constants.DOWN
        print("TRYING RANDOM HORIZONTAL")
        return self.move_random_horizontally_or_wait(current_percept, direction)
                                          
    def move_left_if_open(self, current_percept, direction) -> int:
        print("TRYING LEFT")
        if direction[constants.LEFT] == constants.OPEN:
            return constants.LEFT
        print("TRYING RANDOM VERTICAL")
        return self.move_random_vertically_or_wait(current_percept, direction)

    def move_right_if_open(self, current_percept, direction) -> int:
        print("TRYING RIGHT")
        if direction[constants.RIGHT] == constants.OPEN:
            return constants.RIGHT
        print("TRYING RANDOM VERTICAL")
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

        direction = [constants.CLOSED, constants.CLOSED, constants.CLOSED, constants.CLOSED]
        current_cell_doors = [0, 0, 0, 0]
        for maze_state in current_percept.maze_state:
            if maze_state[0] == 0 and maze_state[1] == 0: # If the cell is the current cell
                current_cell_doors[maze_state[2]] = maze_state[3] # Set the direction to the state of the door
                if maze_state[3] == constants.BOUNDARY:
                    direction[maze_state[2]] = constants.BOUNDARY

        if current_cell_doors[constants.LEFT] == constants.OPEN:
            for maze_state in current_percept.maze_state:
                if maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT and maze_state[3] == constants.OPEN:
                    direction[constants.LEFT] = constants.OPEN
        if current_cell_doors[constants.RIGHT] == constants.OPEN:
            for maze_state in current_percept.maze_state:
                if maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT and maze_state[3] == constants.OPEN:
                    direction[constants.RIGHT] = constants.OPEN
        if current_cell_doors[constants.UP] == constants.OPEN:
            for maze_state in current_percept.maze_state:
                if maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN and maze_state[3] == constants.OPEN:
                    direction[constants.UP] = constants.OPEN
        if current_cell_doors[constants.DOWN] == constants.OPEN:    
            for maze_state in current_percept.maze_state:
                if maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP and maze_state[3] == constants.OPEN:
                    direction[constants.DOWN] = constants.OPEN
        
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
            if self.random_movements_started == False and len(set(self.recently_seen_positions_list)) <= self.stuck_seeing_same_positions_threshold and len(self.recently_seen_positions_list) == self.recently_seen_positions_list_size:
                print("Stuck for too long in same cells, moving randomly for " + str(self.number_of_random_moves_threshold) + " moves")
                self.number_of_random_moves = self.number_of_random_moves_threshold
                self.random_movements_started = True
            
            if self.random_movements_started:
                if self.number_of_random_moves == 0:
                    self.random_movements_started = False
                    self.number_of_random_moves = self.number_of_random_moves_threshold
                    print("RANDOM MOVEMENTS COMPLETE")
                    self.seen_in_random_walk = set()
                else:
                    self.number_of_random_moves -= 1
                    return self.move_random_open_direction(direction, current_percept)
                
            set_target = self.set_target(current_percept, direction)
            if set_target is None:
                print("WAIT BECAUSE NO TARGET IS RETURNED DUE TO SOME ERROR")
                # This is entered sometimes, I noticed that right after the first corner is found, for the first move, this is entered. Fix this.
                return constants.WAIT
            print("Target: " + str(set_target))
            if(set_target[0] > 0 and set_target[1] == 0):
                print("Trying to move Right")
                return self.move_right_if_open(current_percept, direction)
            elif(set_target[0] < 0 and set_target[1] == 0):
                print("Trying to move Left")
                return self.move_left_if_open(current_percept, direction)
            elif(set_target[0] == 0 and set_target[1] > 0):
                print("Trying to move Down")
                return self.move_down_if_open(current_percept, direction)
            elif(set_target[0] == 0 and set_target[1] < 0):
                print("Trying to move Up")
                return self.move_up_if_open(current_percept, direction)
            else:                    
                print("Moving Diagonally")
                return self.move_diagonally(current_percept, direction, set_target[0], set_target[1])
    
    def move_diagonally(self, current_percept, direction, target_x, target_y) -> int:
        """Move towards the specified corner by checking diagonal movement options.

        Args:
            current_percept (TimingMazeState): Contains current state information.
            direction (list): The available directions from the current position.
            corner (str): The nearest corner to move towards.
        
        Returns:
            int: The move decision based on the available directions.
        """
        print("Target X: " + str(target_x))
        print("Target Y: " + str(target_y))
        print("Entered here")
        if target_x < 0 and target_y < 0:
            if self.rng.choice([constants.UP, constants.LEFT]) == constants.LEFT:  # Randomly choose between the two
                if direction[constants.LEFT] == constants.OPEN:
                    return constants.LEFT
                if direction[constants.UP] == constants.OPEN:
                    return constants.UP
            else:
                if direction[constants.UP] == constants.OPEN:
                    return constants.UP
                if direction[constants.LEFT] == constants.OPEN:
                    return constants.LEFT
            print("WAIT BECAUSE IT IS NOT POSSIBLE TO MOVE ANY DIAGONALLY TOWARDS CORNER")
            return constants.WAIT
            
        elif target_x > 0 and target_y < 0:
            if self.rng.choice([constants.UP, constants.RIGHT]) == constants.RIGHT:
                if direction[constants.RIGHT] == constants.OPEN:
                    return constants.RIGHT
                if direction[constants.UP] == constants.OPEN:
                    return constants.UP
            else:
                if direction[constants.UP] == constants.OPEN:
                    return constants.UP
                if direction[constants.RIGHT] == constants.OPEN:
                    return constants.RIGHT
            print("WAIT BECAUSE IT IS NOT POSSIBLE TO MOVE ANY DIAGONALLY TOWARDS CORNER")
            return constants.WAIT
        
        elif target_x < 0 and target_y > 0:
            if self.rng.choice([constants.DOWN, constants.LEFT]) == constants.LEFT:
                if direction[constants.LEFT] == constants.OPEN:
                    return constants.LEFT
                if direction[constants.DOWN] == constants.OPEN:
                    return constants.DOWN
            else:
                if direction[constants.DOWN] == constants.OPEN:
                    return constants.DOWN
                if direction[constants.LEFT] == constants.OPEN:
                    return constants.LEFT
            print("WAIT BECAUSE IT IS NOT POSSIBLE TO MOVE ANY DIAGONALLY TOWARDS CORNER")
            return constants.WAIT
    
        elif target_x > 0 and target_y > 0:
            if self.rng.choice([constants.DOWN, constants.RIGHT]) == constants.RIGHT:
                if direction[constants.RIGHT] == constants.OPEN:
                    return constants.RIGHT
                if direction[constants.DOWN] == constants.OPEN:
                    return constants.DOWN
            else:
                if direction[constants.DOWN] == constants.OPEN:
                    return constants.DOWN
                if direction[constants.RIGHT] == constants.OPEN:
                    return constants.RIGHT
            print("WAIT BECAUSE IT IS NOT POSSIBLE TO MOVE ANY DIAGONALLY TOWARDS CORNER")
            return constants.WAIT
