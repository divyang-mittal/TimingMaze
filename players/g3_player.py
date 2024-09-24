import math
from audioop import reverse
from lib2to3.fixer_util import parenthesize

import numpy as np
import logging

# from scipy.optimize import direct

import constants
from timing_maze_state import TimingMazeState

import heapq

class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger,
                 precomp_dir: str, maximum_door_frequency: int, radius: int) -> None:
        """Initialise the player with the basic information

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
        self.inner_boundary = [constants.map_dim, constants.map_dim, constants.map_dim, constants.map_dim]
        self.inside_out_state = 0
        self.inside_out_rem = None
        self.inside_out_start_radius = self.radius
        self.rush_in_timer = maximum_door_frequency
        self.rush_in_reverse_timer = maximum_door_frequency
        self.relative_frequencies = np.full((201, 201, 4), -1, dtype=int)
        self.door_timers = np.full((201, 201, 4), 0, dtype=int)
        self.always_closed = np.full((201, 201, 4), False, dtype=bool)
        self.turn_counter = 0
        self.opposite = [constants.RIGHT, constants.DOWN, constants.LEFT, constants.UP]
        self.dRow = [-1, 0, 1, 0]
        self.dCol = [0, -1, 0, 1]
        self.is_end_visible = False
        self.end_x = 0
        self.end_y = 0
        self.global_unvisited_map = np.full((201, 201), -1, dtype=int)

        self.is_djikstra_available = False
        self.end_visible_timer = 0

        self.a_star_path = np.full((201, 201), -1, dtype=int)
        self.is_a_star_available = False


        #assuming it starts from a corner
        self.outside_in_start_radius = 100-0.8*self.radius
        self.outside_in_timer = 0
        self.outside_in_reverse_timer = 0
        self.rush_in_timer = maximum_door_frequency
        self.rush_in_reverse_timer = maximum_door_frequency
        self.x_axis_dist = 0
        self.y_axis_dist = 0
        self.global_counter=0
        self.update_case=0
        self.outside_in_state = 0

        self.synthetic_goal = None

        self.previous_position = None
        self.stuck_turn_counter = 0
        
        self.outside_in_mode =False
        
    def get_corner(self, current_percept) -> int:
        print("Get Corner started")
        for maze_state in current_percept.maze_state:
            x_1=maze_state[0]
            y_1=maze_state[1]
            if maze_state[3]==constants.BOUNDARY:
                direction_1 = maze_state[2]
                for maze_state in current_percept.maze_state:
                    if x_1==maze_state[0] and y_1==maze_state[1] and maze_state[3]==constants.BOUNDARY:
                        # x_axis_dist= abs(x_1-x)
                        # print(x)
                        if direction_1==constants.RIGHT and maze_state[2]==constants.DOWN:
                            return 3
                        elif direction_1==constants.RIGHT and maze_state[2]==constants.UP:
                            return 2
                        elif direction_1==constants.LEFT and maze_state[2]==constants.DOWN:
                            return 4
                        elif direction_1==constants.LEFT and maze_state[2]==constants.UP:
                            return 1
            
        return 0
    
    def get_corner2(self, x,y, current_percept) -> int:
        # print("Get Corner2 started")
        for maze_state in current_percept.maze_state:
            x_1=maze_state[0]
            y_1=maze_state[1]
            if maze_state[3]==constants.BOUNDARY:
                direction_1 = maze_state[2]
                for maze_state in current_percept.maze_state:
                    if x_1==maze_state[0] and y_1==maze_state[1] and maze_state[3]==constants.BOUNDARY:
                        self.x_axis_dist= abs(x_1-x)
                        self.y_axis_dist= abs(y_1-y)
                        if direction_1==constants.RIGHT and maze_state[2]==constants.DOWN:
                            return 3
                        elif direction_1==constants.RIGHT and maze_state[2]==constants.UP:
                            return 2
                        elif direction_1==constants.LEFT and maze_state[2]==constants.DOWN:
                            return 4
                        elif direction_1==constants.LEFT and maze_state[2]==constants.UP:
                            return 1
            
        return 0

    def has_been_stuck_too_long(self, current_percept) -> bool:
        # print("start stuck too long function")
        # Get current position
        current_position = (constants.map_dim - current_percept.start_x, constants.map_dim - current_percept.start_y)

        # If the player is in the same position as before, increase the stuck counter
        if self.previous_position == current_position:
            self.stuck_turn_counter += 1
            # print("stuck counter incremented")
        else:
            # If the player moved, reset the stuck counter
            self.stuck_turn_counter = 0
            self.previous_position = current_position
            # print("stuck counter reset")

        # print("end stuck too long function")
        return self.stuck_turn_counter > (self.maximum_door_frequency-1)*self.maximum_door_frequency

    def pick_new_synthetic_goal(self, current_percept):
        """Pick a new synthetic goal from unvisited cells within the given radius."""

        unvisited_cells = []
        # Find unvisited cells within the radius
        for i in range(201):
            for j in range(201):
                if self.global_unvisited_map[i, j] == 0:
                    unvisited_cells.append((i, j))

        if len(unvisited_cells) > 0:
            random_idx = self.rng.choice(len(unvisited_cells))
            synthetic_goal_global = unvisited_cells[random_idx]
            self.synthetic_goal = synthetic_goal_global
            self.stuck_turn_counter = 0
            # print(f"Selected synthetic goal: {self.synthetic_goal}")

            # Try Dijkstra towards the synthetic goal
            move = self.find_djikstra_step(current_percept, synthetic_goal_global[0], synthetic_goal_global[1])

            return move
        return constants.WAIT

    def reached_synthetic_goal(self, current_percept):
        """Check if the player has reached the synthetic goal and pick a new one if necessary."""

        # Get current position of the player
        current_x = constants.map_dim - current_percept.start_x
        current_y = constants.map_dim - current_percept.start_y

        # Check if the player reached the synthetic goal
        if (current_x, current_y) == tuple(self.synthetic_goal):
            # print(f"Reached synthetic goal at: {self.synthetic_goal}")

            # If the end is visible, stop picking synthetic goals and pursue the end goal
            if current_percept.is_end_visible:
                # print("End goal is visible! Stopping synthetic goal selection.")
                return True

            # If the end is not visible, pick a new synthetic goal
            # print("End goal not visible. Picking a new synthetic goal.")
            self.pick_new_synthetic_goal(current_percept)
            return True

        return False

    def move(self, current_percept) -> int:
        """Function which retrieves the current state of the map and returns a movement

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
        current_x = constants.map_dim - current_percept.start_x
        current_y = constants.map_dim - current_percept.start_y
        self.global_unvisited_map[current_x, current_y] = 1

        self.update_door_timers(current_percept)
        self.update_relative_frequencies(current_percept)

        if current_percept.is_end_visible:
            self.is_end_visible = True
            self.end_x = current_percept.end_x - current_percept.start_x
            self.end_y = current_percept.end_y - current_percept.start_y

        if self.is_end_visible:
            if self.end_visible_timer > 1.5 * self.maximum_door_frequency:
                dijkstra_step = self.find_djikstra_step(current_percept, constants.map_dim + self.end_x, constants.map_dim + self.end_y)
                if dijkstra_step != -1:
                    return dijkstra_step

                # A* is triggered if Dijkstra fails to find the path
                found_a_star = self.calculate_a_star(current_percept)
                if found_a_star:
                    move = self.get_a_star(current_percept)
                    return move

            self.end_visible_timer += 1
            direction = [0, 0, 0, 0]
            reverse_direction = [0, 0, 0, 0]
            for maze_state in current_percept.maze_state:
                if maze_state[0] == 0 and maze_state[1] == 0:
                    direction[maze_state[2]] = maze_state[3]
                if maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP:
                    reverse_direction[constants.DOWN] = maze_state[3]
                if maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN:
                    reverse_direction[constants.UP] = maze_state[3]
                if maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT:
                    reverse_direction[constants.RIGHT] = maze_state[3]
                if maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT:
                    reverse_direction[constants.LEFT] = maze_state[3]
            ratio = abs(self.end_x + current_percept.start_x + 0.05)/(abs(self.end_x + current_percept.start_x) + abs(self.end_y + current_percept.start_y) + 0.1)

            if self.rng.random() < ratio:
                val = self.rush_in_horizontal(current_percept, direction, reverse_direction)
                if val != -1:
                    return val
                val = self.rush_in_vertical(current_percept, direction, reverse_direction)
                if val != -1:
                    return val
                return constants.WAIT
            else:
                val = self.rush_in_vertical(current_percept, direction, reverse_direction)
                if val != -1:
                    return val
                val = self.rush_in_horizontal(current_percept, direction, reverse_direction)
                if val != -1:
                    return val
                return constants.WAIT

        # Exploratory phase
        if self.has_been_stuck_too_long(current_percept):
            print("Stuck too long")
            self.pick_new_synthetic_goal(current_percept)

        if self.synthetic_goal is not None:
            print("Synthetic goal is not None", self.synthetic_goal)
            self.reached_synthetic_goal(current_percept)

            move = self.find_djikstra_step(current_percept, self.synthetic_goal[0], self.synthetic_goal[1])
            if move != -1:
                return move

        corner= self.get_corner(current_percept)
        if corner!=0:
            self.outside_in_mode=True
        if self.outside_in_mode:
            print("Outside in mode")
            return self.move_outside_in_3(current_percept)
        else:
            print("Inside out mode")
            return self.move_inside_out(current_percept)

    def update_door_timers(self, current_percept):
        door_seen = np.full((201, 201, 4), False, dtype=bool)
        for cell in current_percept.maze_state:
            x, y, direction, state = cell

            # positions relative to start position
            abs_x = x - current_percept.start_x
            abs_y = y - current_percept.start_y
            door_seen[abs_x, abs_y, direction] = True

            if self.global_unvisited_map[abs_x, abs_y] == -1:
                self.global_unvisited_map[abs_x, abs_y] = 0

            # if we see the door on this turn, then increment the timer if the door is closed
            if state == constants.CLOSED:
                self.door_timers[abs_x, abs_y, direction] += 1

            # if the door is closed for more than the maximum frequency, then mark it as always closed
            if self.door_timers[abs_x, abs_y, direction] >= self.maximum_door_frequency:
                self.always_closed[abs_x, abs_y, direction] = True

            #if the door is open, then reset the timer
            if state == constants.OPEN:
                self.door_timers[abs_x, abs_y, direction] = 0

            if state == constants.BOUNDARY:
                self.always_closed[abs_x, abs_y, direction] = True

        # for all doors that we were not present in current_percept.maze_state, set the timer as -1
        for i in range(201):
            for j in range(201):
                for k in range(4):
                    if not door_seen[i, j, k]:
                        self.door_timers[i, j, k] = 0


    # def traverse_djikstra(self, current_percept) -> int:
    #     abs_x = constants.map_dim - current_percept.start_x
    #     abs_y = constants.map_dim - current_percept.start_y
    #     return self.djikstra_path[abs_x, abs_y].item()

    def find_djikstra_step(self, current_percept, end_x, end_y) -> int:
        # start_x and start_y are the relative positions of the start position
        # self.is_djikstra_available = False
        direction = -1
        start_x = constants.map_dim - current_percept.start_x
        start_y = constants.map_dim - current_percept.start_y

        size = constants.map_dim*2 + 1

        # initialize the distance array
        distance = np.full((size, size), np.inf)
        distance[start_x, start_y] = 0

        # initialize the visited array
        visited = np.full((size, size), False)

        # initialize the queue
        q = []
        heapq.heappush(q, (self.turn_counter, start_x, start_y))

        # initialize the direction array
        parent_direction = np.full((size, size), -1)

        while len(q) > 0:
            current = heapq.heappop(q)
            turn, x, y = current

            if visited[x, y]:
                continue

            # mark the current node as visited
            visited[x, y] = True

            # check if we are at the end position
            if x == end_x and y == end_y:
                break

            # check for all the four directions
            for i in range(4):
                new_x = x + self.dRow[i]
                new_y = y + self.dCol[i]

                # print("Checking for new_x, new_y: ", new_x, new_y)
                # check if the new position is valid
                if new_x >= 0 and new_x < size and new_y >= 0 and new_y < size and not visited[new_x, new_y]:
                    # check if the door is open
                    # print("Checking relative frequencies for x, y: ", new_x, new_y, self.relative_frequencies[x, y, i], self.relative_frequencies[new_x, new_y, self.opposite[i]])
                    if self.relative_frequencies[x, y, i] <= 0 or self.relative_frequencies[new_x, new_y, self.opposite[i]] <= 0:
                        continue

                    # print("Calculating for new_x, new_y: ", new_x, new_y)
                    lcm = math.lcm(self.relative_frequencies[x, y, i], self.relative_frequencies[new_x, new_y,
                        self.opposite[i]])

                    next_turn = turn+1
                    if lcm % next_turn != 0:
                        next_turn = (next_turn // lcm + 1) * lcm

                    # check if the new distance is less than the previous distance
                    if distance[new_x, new_y] > next_turn:
                        distance[new_x, new_y] = next_turn
                        parent_direction[new_x, new_y] = self.opposite[i]
                        heapq.heappush(q, (next_turn, new_x, new_y))

        # if end position is not reachable, then return -1
        if not visited[end_x, end_y]:
            return -1

        # # if we found the position set it as path available and traverse the parent direction
        # self.is_djikstra_available = True

        # traverse the parent direction to get the path from end to start and update djikstra path
        x = end_x
        y = end_y
        while x != start_x or y != start_y:
            parent_x = x
            parent_y = y
            if parent_direction[x, y] == constants.RIGHT:
                parent_x += 1
            elif parent_direction[x, y] == constants.LEFT:
                parent_x -= 1
            elif parent_direction[x, y] == constants.DOWN:
                parent_y += 1
            elif parent_direction[x, y] == constants.UP:
                parent_y -= 1

            direction = self.opposite[parent_direction[x, y]]
            x = parent_x
            y = parent_y

        return direction
    
    def get_a_star(self, current_percept):
        abs_x = 100 - current_percept.start_x
        abs_y = 100 - current_percept.start_y
        return self.a_star_path[abs_x, abs_y].item()
    
    def calculate_a_star(self, current_percept) -> bool:
        # calculates A* path to the end and stores it in 'self.a_star_path'
        
        # start_x and start_y are the relative positions of the start position
        start_x = 100 - current_percept.start_x
        start_y = 100 - current_percept.start_y

        end_x = 100 + self.end_x
        end_y = 100 + self.end_y

        # initialize priority queue and arrays
        pq = []
        distance = np.full((201, 201), np.inf)
        parent_direction = np.full((201, 201), -1)
        visited = np.full((201, 201), False)
        
        # heuristic function (Manhattan distance)
        def heuristic(x, y):
            return abs(x - end_x) + abs(y - end_y)

        # set initial conditions
        distance[start_x, start_y] = 0
        heapq.heappush(pq, (heuristic(start_x, start_y), start_x, start_y, 0))  # (f(x), x, y, g(x))

        # main A* loop
        while pq:
            f_x, x, y, g_x = heapq.heappop(pq)

            if visited[x, y]:
                continue
            visited[x, y] = True
            
            # if end is reached
            if x == end_x and y == end_y:
                break
            
            # explore all 4 neighbors
            for i in range(4):
                new_x = x + self.dRow[i]
                new_y = y + self.dCol[i]

                if 0 <= new_x < 201 and 0 <= new_y < 201 and not visited[new_x, new_y]:
                    # check if doors are passable
                    if self.relative_frequencies[x, y, i] > 0 and self.relative_frequencies[new_x, new_y, self.opposite[i]] > 0:
                        lcm = math.lcm(self.relative_frequencies[x, y, i], self.relative_frequencies[new_x, new_y, self.opposite[i]])
                        next_turn = g_x
                        if next_turn % lcm != 0:
                            new_distance = (next_turn // lcm + 1) * lcm
                        new_distance = next_turn
                        
                        # check if new path is shorter
                        if distance[new_x, new_y] > new_distance:
                            distance[new_x, new_y] = new_distance
                            parent_direction[new_x, new_y] = self.opposite[i]
                            f_x = new_distance + heuristic(new_x, new_y)
                            heapq.heappush(pq, (f_x, new_x, new_y, new_distance))

        # if end is not reachable, return false
        if distance[end_x, end_y] == np.inf:
            return False
        
        # path construction
        x, y = end_x, end_y
        while x != start_x or y != start_y:
            parent_x, parent_y = x, y
            if parent_direction[x, y] == constants.RIGHT:
                parent_x -= 1
            elif parent_direction[x, y] == constants.LEFT:
                parent_x += 1
            elif parent_direction[x, y] == constants.DOWN:
                parent_y -= 1
            elif parent_direction[x, y] == constants.UP:
                parent_y += 1
            self.a_star_path[parent_x, parent_y] = self.opposite[parent_direction[x, y]]
            x, y = parent_x, parent_y

        self.is_a_star_available = True
        return True


    def rush_in_vertical(self, current_percept, direction, rev_direction) -> int:
        if self.end_y + current_percept.start_y < 0:
            if self.rush_in_timer < 0 or self.rush_in_reverse_timer < 0:
                if direction[constants.RIGHT] == constants.OPEN:
                    self.rush_in_timer = self.maximum_door_frequency
                    self.rush_in_reverse_timer = self.maximum_door_frequency
                    return constants.RIGHT
                if direction[constants.LEFT] == constants.OPEN:
                    self.rush_in_timer = self.maximum_door_frequency
                    self.rush_in_reverse_timer = self.maximum_door_frequency
                    return constants.LEFT
            if direction[constants.UP] == constants.OPEN and rev_direction[constants.UP] == constants.OPEN:
                self.rush_in_timer = self.maximum_door_frequency
                self.rush_in_reverse_timer = self.maximum_door_frequency
                return constants.UP
            if direction[constants.UP] != constants.OPEN:
                self.rush_in_timer -= 1
            if rev_direction[constants.UP] != constants.OPEN:
                self.rush_in_reverse_timer -= 1

        if self.end_y + current_percept.start_y > 0:
            if self.rush_in_timer < 0 or self.rush_in_reverse_timer < 0:
                if direction[constants.RIGHT] == constants.OPEN and rev_direction[constants.RIGHT] == constants.OPEN:
                    self.rush_in_timer = self.maximum_door_frequency
                    self.rush_in_reverse_timer = self.maximum_door_frequency
                    return constants.RIGHT
                if direction[constants.LEFT] == constants.OPEN and rev_direction[constants.LEFT] == constants.OPEN:
                    self.rush_in_timer = self.maximum_door_frequency
                    self.rush_in_reverse_timer = self.maximum_door_frequency
                    return constants.LEFT
            if direction[constants.DOWN] == constants.OPEN and rev_direction[constants.DOWN] == constants.OPEN:
                self.rush_in_timer = self.maximum_door_frequency
                self.rush_in_reverse_timer = self.maximum_door_frequency
                return constants.DOWN
            if direction[constants.DOWN] != constants.OPEN:
                self.rush_in_timer -= 1
            if rev_direction[constants.DOWN] != constants.OPEN:
                self.rush_in_reverse_timer -= 1
        return -1

    def rush_in_horizontal(self, current_percept, direction, rev_direction) -> int:
        if self.end_x + current_percept.start_x > 0:
            if self.rush_in_timer < 0 or self.rush_in_reverse_timer < 0:
                if direction[constants.UP] == constants.OPEN:
                    self.rush_in_timer = self.maximum_door_frequency
                    self.rush_in_reverse_timer = self.maximum_door_frequency
                    return constants.UP
                if direction[constants.DOWN] == constants.OPEN:
                    self.rush_in_timer = self.maximum_door_frequency
                    self.rush_in_reverse_timer = self.maximum_door_frequency
                    return constants.DOWN
            if direction[constants.RIGHT] == constants.OPEN and rev_direction[constants.RIGHT] == constants.OPEN:
                self.rush_in_timer = self.maximum_door_frequency
                self.rush_in_reverse_timer = self.maximum_door_frequency
                return constants.RIGHT
            if direction[constants.RIGHT] != constants.OPEN:
                self.rush_in_timer -= 1
            if rev_direction[constants.RIGHT] != constants.OPEN:
                self.rush_in_reverse_timer -= 1

        if self.end_x + current_percept.start_x < 0:
            if self.rush_in_timer < 0 or self.rush_in_reverse_timer < 0:
                if direction[constants.UP] == constants.OPEN:
                    self.rush_in_timer = self.maximum_door_frequency
                    self.rush_in_reverse_timer = self.maximum_door_frequency
                    return constants.UP
                if direction[constants.DOWN] == constants.OPEN:
                    self.rush_in_timer = self.maximum_door_frequency
                    self.rush_in_reverse_timer = self.maximum_door_frequency
                    return constants.DOWN
            if direction[constants.LEFT] == constants.OPEN and rev_direction[constants.LEFT] == constants.OPEN:
                self.rush_in_timer = self.maximum_door_frequency
                self.rush_in_reverse_timer = self.maximum_door_frequency
                return constants.LEFT
            if direction[constants.LEFT] != constants.OPEN:
                self.rush_in_timer -= 1
            if rev_direction[constants.LEFT] != constants.OPEN:
                self.rush_in_reverse_timer -= 1
        return -1

    def move_inside_out(self, current_percept) -> int:
        # Move towards the boundary but not more than the radius away from the inner boundary
        # if the boundary is less than the radius away, change direction
        direction = [0, 0, 0, 0]
        reverse_direction = [0, 0, 0, 0]
        for maze_state in current_percept.maze_state:
            if maze_state[0] == 0 and maze_state[1] == 0:
                direction[maze_state[2]] = maze_state[3]
            if maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP:
                reverse_direction[constants.DOWN] = maze_state[3]
            if maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN:
                reverse_direction[constants.UP] = maze_state[3]
            if maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT:
                reverse_direction[constants.RIGHT] = maze_state[3]
            if maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT:
                reverse_direction[constants.LEFT] = maze_state[3]
        abs_x = - current_percept.start_x
        abs_y = - current_percept.start_y

        if self.inside_out_state == 0:
            self.inside_out_rem = [self.inside_out_start_radius+self.radius, self.inside_out_start_radius+self.radius, self.inside_out_start_radius, self.inside_out_start_radius]
            self.inside_out_start_radius = self.inside_out_start_radius + 2*self.radius
            self.inside_out_state = 1

        if self.inside_out_state == 1:
            if self.inside_out_rem[2] <= 0:
                self.inside_out_state = 2
            else:
                if self.inside_out_rem[2] > 0:
                    if direction[constants.RIGHT] == constants.OPEN and reverse_direction[constants.RIGHT] == constants.OPEN:
                        self.inside_out_rem[2] -= 1
                        return constants.RIGHT

                if ((self.always_closed[abs_x][abs_y][constants.RIGHT] or self.always_closed[abs_x+1][abs_y][constants.LEFT])
                        and direction[constants.DOWN] == constants.OPEN and reverse_direction[constants.DOWN] == constants.OPEN):
                    self.inside_out_rem[3] -= 1
                    return constants.DOWN

                if ((self.always_closed[abs_x][abs_y][constants.RIGHT] or self.always_closed[abs_x+1][abs_y][constants.LEFT])
                        and (self.always_closed[abs_x][abs_y][constants.DOWN] or self.always_closed[abs_x][abs_y+1][constants.UP])
                        and direction[constants.UP] == constants.OPEN and reverse_direction[constants.UP] == constants.OPEN):
                    self.inside_out_rem[1] -= 1
                    return constants.UP

        if self.inside_out_state == 2:
            if self.inside_out_rem[3] <= 0:
                self.inside_out_state = 3
            else:
                if self.inside_out_rem[3] > 0:
                    if direction[constants.DOWN] == constants.OPEN and reverse_direction[constants.DOWN] == constants.OPEN:
                        self.inside_out_rem[3] -= 1
                        return constants.DOWN

                if ((self.always_closed[abs_x][abs_y][constants.DOWN] or self.always_closed[abs_x][abs_y+1][constants.UP])
                        and direction[constants.LEFT] == constants.OPEN and reverse_direction[constants.LEFT] == constants.OPEN):
                    self.inside_out_rem[0] -= 1
                    return constants.LEFT

                if ((self.always_closed[abs_x][abs_y][constants.DOWN] or self.always_closed[abs_x][abs_y+1][constants.UP])
                        and (self.always_closed[abs_x][abs_y][constants.LEFT] or self.always_closed[abs_x-1][abs_y][constants.RIGHT])
                        and direction[constants.RIGHT] == constants.OPEN and reverse_direction[constants.RIGHT] == constants.OPEN):
                    self.inside_out_rem[2] -= 1
                    return constants.RIGHT

        if self.inside_out_state == 3:
            if self.inside_out_rem[0] <= 0:
                self.inside_out_state = 4
            else:
                if self.inside_out_rem[0] > 0:
                    if direction[constants.LEFT] == constants.OPEN and reverse_direction[constants.LEFT] == constants.OPEN:
                        self.inside_out_rem[0] -= 1
                        return constants.LEFT

                if ((self.always_closed[abs_x][abs_y][constants.LEFT] or self.always_closed[abs_x-1][abs_y][constants.RIGHT])
                        and direction[constants.UP] == constants.OPEN and reverse_direction[constants.UP] == constants.OPEN):
                    self.inside_out_rem[1] -= 1
                    return constants.UP

                if ((self.always_closed[abs_x][abs_y][constants.LEFT] or self.always_closed[abs_x-1][abs_y][constants.RIGHT])
                        and (self.always_closed[abs_x][abs_y][constants.UP] or self.always_closed[abs_x][abs_y-1][constants.DOWN])
                        and direction[constants.DOWN] == constants.OPEN and reverse_direction[constants.DOWN] == constants.OPEN):
                    self.inside_out_rem[3] -= 1
                    return constants.DOWN

        if self.inside_out_state == 4:
            if self.inside_out_rem[1] <= 0:
                self.inside_out_state = 0
            else:
                if self.inside_out_rem[1] > 0:
                    if direction[constants.UP] == constants.OPEN and reverse_direction[constants.UP] == constants.OPEN:
                        self.inside_out_rem[1] -= 1
                        return constants.UP

                if ((self.always_closed[abs_x][abs_y][constants.UP] or self.always_closed[abs_x][abs_y-1][constants.DOWN])
                        and direction[constants.RIGHT] == constants.OPEN and reverse_direction[constants.RIGHT] == constants.OPEN):
                    self.inside_out_rem[2] -= 1
                    return constants.RIGHT

                if ((self.always_closed[abs_x][abs_y][constants.UP] or self.always_closed[abs_x][abs_y-1][constants.DOWN])
                        and (self.always_closed[abs_x][abs_y][constants.RIGHT] or self.always_closed[abs_x+1][abs_y][constants.LEFT])
                        and direction[constants.LEFT] == constants.OPEN and reverse_direction[constants.LEFT] == constants.OPEN):
                    self.inside_out_rem[0] -= 1
                    return constants.LEFT

        return constants.WAIT

    def reset_for_outside_in(self):
        if self.global_counter%4==0 and self.global_counter!=0:
            # print('*****************################')
            outside_in_rem_dict={1:[self.outside_in_start_radius-self.radius, self.outside_in_start_radius-2*self.radius, self.outside_in_start_radius, self.outside_in_start_radius-self.radius],
                                2:[self.outside_in_start_radius, self.outside_in_start_radius, self.outside_in_start_radius-2*self.radius, self.outside_in_start_radius],
                                3:[self.outside_in_start_radius, self.outside_in_start_radius-self.radius, self.outside_in_start_radius-self.radius, self.outside_in_start_radius-2*self.radius],
                                4:[self.outside_in_start_radius-2*self.radius, self.outside_in_start_radius, self.outside_in_start_radius, self.outside_in_start_radius]}
            self.outside_in_rem = outside_in_rem_dict[self.update_case]
            self.outside_in_start_radius = self.outside_in_start_radius- 2*self.radius

    def move_outside_in_3(self,current_percept)->int:
        direction = [0, 0, 0, 0]
        reverse_direction = [0, 0, 0, 0]
        for maze_state in current_percept.maze_state:
            if maze_state[0] == 0 and maze_state[1] == 0:
                direction[maze_state[2]] = maze_state[3]
            if maze_state[0] == 0 and maze_state[1] == 1 and maze_state[2] == constants.UP:
                reverse_direction[constants.DOWN] = maze_state[3]
            if maze_state[0] == 0 and maze_state[1] == -1 and maze_state[2] == constants.DOWN:
                reverse_direction[constants.UP] = maze_state[3]
            if maze_state[0] == 1 and maze_state[1] == 0 and maze_state[2] == constants.LEFT:
                reverse_direction[constants.RIGHT] = maze_state[3]
            if maze_state[0] == -1 and maze_state[1] == 0 and maze_state[2] == constants.RIGHT:
                reverse_direction[constants.LEFT] = maze_state[3]
        #check for one side which will be short now
        abs_x = - current_percept.start_x
        abs_y = - current_percept.start_y
        print(abs_x,abs_y)
     
        if self.outside_in_state == 0:
            print("Outside IN started")
            self.corner_val= self.get_corner2(abs_x,abs_y, current_percept)      
            if self.corner_val== 1:
                    self.outside_in_state = 1
                    self.update_case =1
            elif self.corner_val ==2:
                    self.update_case = 2
                    self.outside_in_state = 2
            elif self.corner_val ==3: 
                    self.outside_in_state = 3
                    self.update_case = 3
            elif self.corner_val ==4:
                    self.outside_in_state = 4 
                    self.update_case = 4
            if self.corner_val ==1 and self.corner_val==3:
                self.outside_in_start_radius=100 - 2*self.x_axis_dist
            else:
                self.outside_in_start_radius=100 - 2*self.y_axis_dist
            print(self.outside_in_start_radius)
            outside_in_rem_dict={1:[self.outside_in_start_radius, self.outside_in_start_radius-self.radius, self.outside_in_start_radius, self.outside_in_start_radius],
                                2:[self.outside_in_start_radius, self.outside_in_start_radius, self.outside_in_start_radius-self.radius, self.outside_in_start_radius],
                                3:[self.outside_in_start_radius, self.outside_in_start_radius, self.outside_in_start_radius, self.outside_in_start_radius-self.radius],
                                4:[self.outside_in_start_radius-self.radius, self.outside_in_start_radius, self.outside_in_start_radius, self.outside_in_start_radius]}
            
            self.outside_in_rem = outside_in_rem_dict[self.update_case]
            self.outside_in_start_radius = self.outside_in_start_radius- self.radius

        if self.outside_in_state == 1:
            
            if self.outside_in_rem[2] <= 0:
                self.outside_in_state = 2
                self.global_counter+=1
               
                self.reset_for_outside_in()
                # print("Global Counter",self.global_counter)
            else:
                if self.outside_in_rem[2] > 0:
                    if direction[constants.RIGHT] == constants.OPEN and reverse_direction[constants.RIGHT] == constants.OPEN:
                        self.outside_in_rem[2] -= 1
                        return constants.RIGHT
                    else:
                        self.outside_in_timer -= 1

                if ((self.always_closed[abs_x][abs_y][constants.RIGHT] or self.always_closed[abs_x+1][abs_y][constants.LEFT])
                        and direction[constants.DOWN] == constants.OPEN and reverse_direction[constants.DOWN] == constants.OPEN):
                    self.outside_in_rem[3] -= 1
                    return constants.DOWN

                if ((self.always_closed[abs_x][abs_y][constants.RIGHT] or self.always_closed[abs_x + 1][abs_y][
                    constants.LEFT])
                        and (self.always_closed[abs_x][abs_y][constants.DOWN] or self.always_closed[abs_x][abs_y + 1][
                            constants.UP])
                        and direction[constants.UP] == constants.OPEN and reverse_direction[
                            constants.UP] == constants.OPEN):
                    self.outside_in_rem[1] -= 1
                    return constants.UP

        if self.outside_in_state == 2:
           
            if self.outside_in_rem[3] <= 0:
                self.outside_in_state = 3
                self.global_counter+=1
                self.reset_for_outside_in()
                # print("Global Counter",self.global_counter)
            else:
                if self.outside_in_rem[3] > 0:
                    if direction[constants.DOWN] == constants.OPEN and reverse_direction[constants.DOWN] == constants.OPEN:
                        self.outside_in_rem[3] -= 1
                        return constants.DOWN
                    else:
                        self.outside_in_timer -= 1

                if ((self.always_closed[abs_x][abs_y][constants.DOWN] or self.always_closed[abs_x][abs_y + 1][
                    constants.UP])
                        and direction[constants.LEFT] == constants.OPEN and reverse_direction[
                            constants.LEFT] == constants.OPEN):
                    self.outside_in_rem[0] -= 1
                    return constants.LEFT

                if ((self.always_closed[abs_x][abs_y][constants.DOWN] or self.always_closed[abs_x][abs_y + 1][
                    constants.UP])
                        and (self.always_closed[abs_x][abs_y][constants.LEFT] or self.always_closed[abs_x - 1][abs_y][
                            constants.RIGHT])
                        and direction[constants.RIGHT] == constants.OPEN and reverse_direction[
                            constants.RIGHT] == constants.OPEN):
                    self.outside_in_rem[2] -= 1
                    return constants.RIGHT

        if self.outside_in_state == 3:
           
            if self.outside_in_rem[0] <= 0:
                self.outside_in_state = 4
                self.global_counter+=1
                self.reset_for_outside_in()
                # print("Global Counter",self.global_counter)
            else:
                if self.outside_in_rem[0] > 0:
                    if direction[constants.LEFT] == constants.OPEN and reverse_direction[constants.LEFT] == constants.OPEN:
                        self.outside_in_rem[0] -= 1
                        return constants.LEFT
                    else:
                        self.outside_in_timer -= 1

                if ((self.always_closed[abs_x][abs_y][constants.LEFT] or self.always_closed[abs_x - 1][abs_y][
                    constants.RIGHT])
                        and direction[constants.UP] == constants.OPEN and reverse_direction[
                            constants.UP] == constants.OPEN):
                    self.outside_in_rem[1] -= 1
                    return constants.UP

                if ((self.always_closed[abs_x][abs_y][constants.LEFT] or self.always_closed[abs_x - 1][abs_y][
                    constants.RIGHT])
                        and (self.always_closed[abs_x][abs_y][constants.UP] or self.always_closed[abs_x][abs_y - 1][
                            constants.DOWN])
                        and direction[constants.DOWN] == constants.OPEN and reverse_direction[
                            constants.DOWN] == constants.OPEN):
                    self.outside_in_rem[3] -= 1
                    return constants.DOWN

        if self.outside_in_state == 4:
           
            if self.outside_in_rem[1] <= 0:
                self.outside_in_state = 1
                self.global_counter+=1
                self.reset_for_outside_in()
                # print("Global Counter",self.global_counter)
            else:
                if self.outside_in_rem[1] > 0:
                    if direction[constants.UP] == constants.OPEN and reverse_direction[constants.UP] == constants.OPEN:
                        self.outside_in_rem[1] -= 1
                        return constants.UP
                    else:
                        self.outside_in_timer -= 1

                if ((self.always_closed[abs_x][abs_y][constants.UP] or self.always_closed[abs_x][abs_y - 1][
                    constants.DOWN])
                        and direction[constants.RIGHT] == constants.OPEN and reverse_direction[
                            constants.RIGHT] == constants.OPEN):
                    return constants.RIGHT

                if ((self.always_closed[abs_x][abs_y][constants.UP] or self.always_closed[abs_x][abs_y - 1][
                    constants.DOWN])
                        and (self.always_closed[abs_x][abs_y][constants.RIGHT] or self.always_closed[abs_x + 1][abs_y][
                            constants.LEFT])
                        and direction[constants.LEFT] == constants.OPEN and reverse_direction[
                            constants.LEFT] == constants.OPEN):
                    self.outside_in_rem[0] -= 1
                    return constants.LEFT
     
        return constants.WAIT

    def update_relative_frequencies(self, current_percept):
        self.turn_counter += 1 # assuming that update_relative_frequencies is called every time we make a move or wait
        for cell in current_percept.maze_state:
            x, y, direction, state = cell
            abs_x = constants.map_dim + x - current_percept.start_x
            abs_y = constants.map_dim + y - current_percept.start_y

            # if the door is open
            if state == constants.OPEN:
                # and it's the first time we are seeing this door (because all rel freq cells are initialized to -1)
                if self.relative_frequencies[abs_x, abs_y, direction] == -1:
                    self.relative_frequencies[abs_x, abs_y, direction] = self.turn_counter
                else:
                    # otherwise, if we've seen this cell open before too, take GCD of current turn number and previous frequency estimate
                    # this way, over time, the values will all converge to their precise frequencies
                    self.relative_frequencies[abs_x, abs_y, direction] = math.gcd(self.relative_frequencies[abs_x, abs_y, direction], self.turn_counter)
            # if the door is instead closed
            elif state == constants.CLOSED:
                # don't do anything
                pass
            elif state == constants.BOUNDARY:
                self.relative_frequencies[abs_x, abs_y, direction] = 0