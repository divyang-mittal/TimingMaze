import json
import os
import time
import signal
import numpy as np
import math
from timing_maze_state import TimingMazeState
from constants import *
import constants
from utils import *
from players.default_player import Player as DefaultPlayer
from players.g1_player import Player as G1_Player
from players.g2_player import Player as G2_Player
from players.g3_player import Player as G3_Player
from players.g4_player import Player as G4_Player
from players.group5.player import G5_Player as G5_Player
# from players.G6_Player import G6_Player
# from players.g7.g7_player import Player as G7_Player
# from players.group9_player import Player as G9_Player
from collections import deque as queue
import tkinter as tk

class TimingMazeGame:
    # Direction vectors
    dRow = [-1, 0, 1, 0]
    dCol = [0, -1, 0, 1]

    def __init__(self, args, root):
        self.cur_pos = None
        self.end_pos = None
        self.start_pos = None
        self.start_time = time.time()
        self.use_gui = not args.no_gui
        self.do_logging = not args.disable_logging
        self.is_paused = False
        self.root = root
        self.game_state = "pause"
        self.game_speed = "normal"
        self.scale = int(args.scale)

        if self.use_gui:
            self.grid_width = constants.map_dim * constants.CELL_SIZE
            self.grid_height = constants.map_dim * constants.CELL_SIZE
            self.canvas_width = 155 * self.scale
            self.canvas_height = 100 * self.scale
            self.x_offset = (self.canvas_width - self.grid_width) // 2
            self.y_offset = (self.canvas_height - self.grid_height) // 4
            self.use_timeout = False
        else:
            self.use_timeout = not args.disable_timeout

        self.logger = logging.getLogger(__name__)
        # create file handler which logs even debug messages
        if self.do_logging:
            self.logger.setLevel(logging.DEBUG)
            self.log_dir = args.log_path
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(self.log_dir, 'debug.log'), mode="w")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter('%(message)s'))
            fh.addFilter(MainLoggingFilter(__name__))
            self.logger.addHandler(fh)
            result_path = os.path.join(self.log_dir, "results.log")
            rfh = logging.FileHandler(result_path, mode="w")
            rfh.setLevel(logging.INFO)
            rfh.setFormatter(logging.Formatter('%(message)s'))
            rfh.addFilter(MainLoggingFilter(__name__))
            self.logger.addHandler(rfh)
        else:
            if args.log_path:
                self.logger.setLevel(logging.INFO)
                result_path = args.log_path
                self.log_dir = os.path.dirname(result_path)
                if self.log_dir:
                    os.makedirs(self.log_dir, exist_ok=True)
                rfh = logging.FileHandler(result_path, mode="w")
                rfh.setLevel(logging.INFO)
                rfh.setFormatter(logging.Formatter('%(message)s'))
                rfh.addFilter(MainLoggingFilter(__name__))
                self.logger.addHandler(rfh)
            else:
                self.logger.setLevel(logging.ERROR)
                self.logger.disabled = True

        self.logger.info("Initialise random number generator with seed {}".format(args.seed))

        self.rng = np.random.default_rng(args.seed)

        self.player = None
        self.player_name = None
        self.player_time = constants.timeout
        self.player_timeout = False

        self.max_door_frequency = args.max_door_frequency
        self.radius = args.radius
        self.goal_reached = False
        self.turns = 0
        self.max_turns = 1e10
        self.valid_moves = 0
        self.map_state = np.zeros((constants.map_dim, constants.map_dim, 4), dtype=int)
        self.map_frequencies = np.zeros((constants.map_dim, constants.map_dim, 4), dtype=int)

        self.add_player(args.player)
        self.initialize(args.maze)

    def add_player(self, player_in):
        if player_in in constants.possible_players:
            if player_in.lower() == 'd':
                player_class = DefaultPlayer
                player_name = "Default Player"
            else:
                player_class = eval("G{}_Player".format(player_in))
                player_name = "Group {}".format(player_in)

            self.logger.info(
                "Adding player {} from class {}".format(player_name, player_class.__module__))
            precomp_dir = os.path.join("precomp", player_name)
            os.makedirs(precomp_dir, exist_ok=True)

            start_time = 0
            is_timeout = False
            if self.use_timeout:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(constants.timeout)
            try:
                start_time = time.time()
                player = player_class(rng=self.rng, logger=self.get_player_logger(player_name),
                                      precomp_dir=precomp_dir, maximum_door_frequency=self.max_door_frequency,
                                      radius=self.radius)
                if self.use_timeout:
                    signal.alarm(0)  # Clear alarm
            except TimeoutException:
                is_timeout = True
                player = None
                self.logger.error(
                    "Initialization Timeout {} since {:.3f}s reached.".format(player_name, constants.timeout))

            init_time = time.time() - start_time

            if not is_timeout:
                self.logger.info("Initializing player {} took {:.3f}s".format(player_name, init_time))
            self.player = player
            self.player_name = player_name

        else:
            self.logger.error("Failed to insert player {} since invalid player name provided.".format(player_in))

    def get_player_logger(self, player_name):
        player_logger = logging.getLogger("{}.{}".format(__name__, player_name))

        if self.do_logging:
            player_logger.setLevel(logging.INFO)
            # add handler to self.logger with filtering
            player_fh = logging.FileHandler(os.path.join(self.log_dir, '{}.log'.format(player_name)), mode="w")
            player_fh.setLevel(logging.DEBUG)
            player_fh.setFormatter(logging.Formatter('%(message)s'))
            player_fh.addFilter(PlayerLoggingFilter(player_name))
            self.logger.addHandler(player_fh)
        else:
            player_logger.setLevel(logging.ERROR)
            player_logger.disabled = True

        return player_logger

    def initialize(self, maze):
        # If maze is provided, load it in map_frequencies.
        if maze:
            self.logger.info("Loading maze from {}".format(maze))
            with open(maze, "r") as f:
                maze_obj = json.load(f)
            self.cur_pos = np.array(maze_obj["start_pos"])
            self.start_pos = self.cur_pos.copy()
            self.end_pos = np.array(maze_obj["end_pos"])
            self.map_frequencies = np.array(maze_obj["frequencies"])

            # Validate the map
            if not self.validate_maze():
                self.logger.error("Maze is invalid")
                raise Exception("Invalid Map")
        else:
            # If no map is provided, generate a random maze using the seed provided
            # Generate a frequency for each cell between 0 and max_door_frequency using the rng
            # self.logger.info("Generating random maze using seed {}".format(self.rng.bit_generator.seed))
            while 1:
                self.cur_pos = np.array([self.rng.integers(0, constants.map_dim),
                                         self.rng.integers(0, constants.map_dim)])
                self.start_pos = self.cur_pos.copy()
                while 1:
                    self.end_pos = np.array([self.rng.integers(0, constants.map_dim),
                                             self.rng.integers(0, constants.map_dim)])
                    if self.end_pos[0] != self.cur_pos[0] and self.end_pos[1] != self.cur_pos[1]:
                        break

                # Generate a random map
                for i in range(constants.map_dim):
                    for j in range(constants.map_dim):
                        for k in range(4):
                            if self.rng.random() < constants.CLOSED_PROB:
                                self.map_frequencies[i][j][k] = 0
                            else:
                                self.map_frequencies[i][j][k] = self.rng.integers(1, self.max_door_frequency)

                # Assign n=0 to all boundary doors
                for i in range (constants.map_dim):
                    self.map_frequencies[0][i][constants.LEFT] = 0
                    self.map_frequencies[constants.map_dim-1][i][constants.RIGHT] = 0
                    self.map_frequencies[i][0][constants.UP] = 0
                    self.map_frequencies[i][constants.map_dim-1][constants.DOWN] = 0

                if self.validate_maze():
                    break

                print("Retrying to generate a valid maze...")

        print("Maze created successfully...")

        # Uncomment to save the maze in a json file
        # data = {
        #     "frequencies": self.map_frequencies.tolist(),
        #     "start_pos": self.cur_pos.tolist(),
        #     "end_pos": self.end_pos.tolist()
        # }
        # filename = 'data.json'
        # file_path = os.path.join(os.getcwd(), filename)
        # with open(filename, 'w') as json_file:
        #     json.dump(data, json_file, indent=4)
        
        # print(f"JSON file '{filename}' created successfully at {file_path}")

        self.map_state = self.map_frequencies.copy()

        if self.use_gui:
            self.canvas = tk.Canvas(self.root, height=self.canvas_height, width=self.canvas_width, bg="#FCF1E3")
            self.canvas.pack()
            self.draw_grid()
            self.root.mainloop()
        else:
            self.play_game()

    def validate_maze(self):
        # Check the size of the map
        if self.map_frequencies.shape != (constants.map_dim, constants.map_dim, 4):
            print("Error with map size")
            return False

        # Check that all doors have a frequency between 0 and max_door_frequency
        for i in range(constants.map_dim):
            for j in range(constants.map_dim):
                for k in range(4):
                    if self.map_frequencies[i][j][k] < 0 or self.map_frequencies[i][j][k] > self.max_door_frequency:
                        print("Error with frequency")
                        return False

        # Check that all boundary doors have n=0 in map_frequencies.
        for i in range(constants.map_dim):
            if self.map_frequencies[0][i][constants.LEFT] != 0:
                print("Error with UP")
                return False
            if self.map_frequencies[constants.map_dim-1][i][constants.RIGHT] != 0:
                print("Error with DOWN")
                return False
            if self.map_frequencies[i][0][constants.UP] != 0:
                print("Error with LEFT")
                return False
            if self.map_frequencies[i][constants.map_dim-1][constants.DOWN] != 0:
                print("Error with RIGHT")
                return False

        # Check that map has a valid start and end position.
        if self.cur_pos[0] < 0 or self.cur_pos[0] >= constants.map_dim or self.cur_pos[1] < 0 or self.cur_pos[1] >= constants.map_dim:
            print("Error with start")
            return False

        if self.end_pos[0] < 0 or self.end_pos[0] >= constants.map_dim or self.end_pos[1] < 0 or self.end_pos[1] >= constants.map_dim:
            print("Error with end")
            return False

        if self.cur_pos[0] == self.end_pos[0] and self.cur_pos[1] == self.end_pos[1]:
            print("Error with start and end")
            return False

        # Check if all cells are reachable from one-another
        # Create an undirected graph and check if the map is valid by looking for islands in the graph.

        # Create a graph of the map with the doors as edges and a valid path if both doors are open at anytime.
        graph = np.zeros((constants.map_dim, constants.map_dim, 4), dtype=int)

        for i in range(constants.map_dim):
            for j in range(constants.map_dim):
                for k in range(4):
                    if self.map_frequencies[i][j][k] != 0:
                        if k == constants.LEFT:
                            if i > 0 and self.map_frequencies[i-1][j][constants.RIGHT] != 0:
                                graph[i][j][k] = 1
                        elif k == constants.RIGHT:
                            if i < constants.map_dim-1 and self.map_frequencies[i+1][j][constants.LEFT] != 0:
                                graph[i][j][k] = 1
                        elif k == constants.DOWN:
                            if j < constants.map_dim-1 and self.map_frequencies[i][j+1][constants.UP] != 0:
                                graph[i][j][k] = 1
                        elif k == constants.UP:
                            if j > 0 and self.map_frequencies[i][j-1][constants.DOWN] != 0:
                                graph[i][j][k] = 1


        # Create a visited array to keep track of visited cells
        visited = np.zeros((constants.map_dim, constants.map_dim), dtype=int)

        # Create a queue to perform BFS traversal
        q = queue()
        q.append((0, 0))
        visited[0][0] = 1
        visited_count = 0

        # Perform BFS traversal
        print("Validating reachability of all cells...")

        while len(q) > 0:
            cell = q.popleft()
            row = cell[0]
            col = cell[1]
            visited_count += 1

            # Check for all the four doors,
            for door_type in range(4):
                if graph[row][col][door_type] == 1:
                    # Get the adjacent cell
                    adj_x = row + self.dRow[door_type]
                    adj_y = col + self.dCol[door_type]
                    if 0 <= adj_x < constants.map_dim and  0 <= adj_y < constants.map_dim and visited[adj_x][adj_y] == 0:
                        q.append((adj_x, adj_y))
                        visited[adj_x][adj_y] = 1

        return visited_count == constants.map_dim * constants.map_dim

    def resume(self):
        if self.game_state == "pause":
            self.game_state = "resume"
            self.game_speed = "normal"
            self.root.after(50, self.play_game)

    def pause(self):
        if self.game_state != "over":
            self.game_state = "pause"

    def step(self):
        if self.game_state != "over":
            self.game_state = "pause"
            self.root.after(100, self.play_game)

    def toggle_speed(self):
        if self.game_state == "resume":
            if self.game_speed == "normal":
                self.game_speed = "fast"
            else:
                self.game_speed = "normal"

    def play_game(self):
        self.turns += 1

        # Get the drone visual for a radius of r

        drone_visual_time = time.time()
        maze_state, is_end_visible = self.get_drone_visual()
        drone_visual_time = time.time() - drone_visual_time
        self.logger.debug("Drone visual took {:.3f}s".format(drone_visual_time))

        # Create the state object for the player
        before_state = TimingMazeState(maze_state, is_end_visible,
                                       self.end_pos[0]-self.cur_pos[0], self.end_pos[1]-self.cur_pos[1],
                                       self.start_pos[0]-self.cur_pos[0], self.start_pos[1]-self.cur_pos[1])
        returned_action = None
        if not self.player_timeout:
            player_start = time.time()
            try:
                # Call the player's move function for turn on this move
                returned_action = self.player.move(
                    current_percept=before_state
                )
            except Exception:
                print("Exception in player code")
                returned_action = None

            player_time_taken = time.time() - player_start
            self.logger.debug("Player {} took {:.3f}s".format(self.player_name, player_time_taken))

            self.player_time -= player_time_taken
            if self.player_time <= 0:
                self.player_timeout = True
                returned_action = None

        if self.check_action(returned_action):
            move = returned_action
            if self.check_and_apply_move(move):
                print("Move Accepted! New position", self.cur_pos)
                self.logger.debug("Received move from {}".format(self.player_name))
                self.valid_moves += 1
            else:
                print("Invalid move as trying to cross some uncrossable boundaries hence cancelled: ", move,
                      self.cur_pos[0], self.cur_pos[1], self.end_pos[0], self.end_pos[1])
                self.logger.info("Invalid move from {} as it does not follow the rules".format(self.player_name))
        else:
            print("Invalid move")
            self.logger.info("Invalid move from {} as it doesn't follow the return format".format(self.player_name))

        if self.use_gui:
            self.draw_grid()

        self.update_door_state()


        print("Turn {} complete".format(self.turns))

        if self.cur_pos[0] == self.end_pos[0] and self.cur_pos[1] == self.end_pos[1]:
            self.game_state = "over"
            print("Goal reached!\n\n Turns taken: {}\n".format(self.turns))
            self.end_time = time.time()
            print("\nTime taken: {}\nValid moves: {}\n".format(self.end_time - self.start_time, self.valid_moves))
            return

        if self.turns < self.max_turns:
            if self.use_gui:
                if self.game_state == "resume":
                    if self.game_speed == "normal":
                        self.root.after(200, self.play_game)
                    else:
                        self.root.after(5, self.play_game)
            else:
                self.play_game()
        else:
            print("Goal not reached...\n\n")
            self.game_state = "over"
            self.end_time = time.time()
            print("\nTime taken: {}\nValid moves: {}\n".format(self.end_time - self.start_time, self.valid_moves))
            return

    @staticmethod
    def is_valid(row, col, vis):
        # If cell lies out of bounds
        if row < 0 or col < 0 or row >= constants.map_dim or col >= constants.map_dim:
            return False

        # If cell is already visited
        if vis[row][col]:
            return False

        # Otherwise
        return True

    @staticmethod
    def get_euclidean_distance_between_two_points(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def update_door_state(self):
        # Iterate all the doors and update their states
        for i in range(constants.map_dim):
            for j in range(constants.map_dim):
                for k in range(4):
                    if self.map_state[i][j][k] > 0:
                        if self.map_state[i][j][k] == 1:
                            self.map_state[i][j][k] = self.map_frequencies[i][j][k]
                        else:
                            self.map_state[i][j][k] -= 1

    def validate_distance_between_drone_and_door(self, row, col, door_type):
        # calculate the distance between the drone and three points of the door,
        # centre and the two ends of the door. If any of these points are visible from the drone,
        # then the door is visible from the drone
        drone_x = self.cur_pos[0]+0.5
        drone_y = self.cur_pos[1]+0.5
        distance = 0

        if door_type == constants.LEFT:
            # calculate the distance between the drone and the centre of the door
            door_x = row
            door_y = col + 0.5
            distance = self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y)

            # calculate the distance between the drone and the bottom end of the door
            door_x = row
            door_y = col + 1
            distance = min(distance,
                self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))

            # calculate the distance between the drone and the top end of the door
            door_x = row
            door_y = col
            distance = min(distance,
                self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))

        elif door_type == constants.RIGHT:
            # calculate the distance between the drone and the centre of the door
            door_x = row + 1
            door_y = col + 0.5
            distance = self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y)

            # calculate the distance between the drone and the bottom end of the door
            door_x = row + 1
            door_y = col + 1
            distance = min(distance,
                           self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))

            # calculate the distance between the drone and the top end of the door
            door_x = row + 1
            door_y = col
            distance = min(distance,
                           self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))

        elif door_type == constants.UP:
            # calculate the distance between the drone and the centre of the door
            door_x = row + 0.5
            door_y = col
            distance = self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y)

            # calculate the distance between the drone and the left end of the door
            door_x = row
            door_y = col
            distance = min(distance,
                           self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))

            # calculate the distance between the drone and the right end of the door
            door_x = row + 1
            door_y = col
            distance = min(distance,
                           self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))
        elif door_type == constants.DOWN:
            # calculate the distance between the drone and the centre of the door
            door_x = row + 0.5
            door_y = col + 1
            distance = self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y)

            # calculate the distance between the drone and the left end of the door
            door_x = row
            door_y = col + 1
            distance = min(distance,
                           self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))

            # calculate the distance between the drone and the right end of the door
            door_x = row + 1
            door_y = col + 1
            distance = min(distance,
                           self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))
        return distance <= self.radius

    # Function to perform the BFS traversal
    # Adapted from https://www.geeksforgeeks.org/breadth-first-traversal-bfs-on-a-2d-array/
    def BFS(self, state):

        # Stores indices of the matrix cells
        q = queue()
        vis = [[False for _ in range(constants.map_dim)] for _ in range(constants.map_dim)]
        is_end_visible = False

        # Mark the starting cell as visited
        # and push it into the queue
        q.append((self.cur_pos[0], self.cur_pos[1]))
        vis[self.cur_pos[0]][self.cur_pos[1]] = True

        if self.cur_pos[0] == self.end_pos[0] and self.cur_pos[1] == self.end_pos[1]:
            is_end_visible = True

        # Iterate while the queue
        # is not empty
        while len(q) > 0:
            cell = q.popleft()
            row = cell[0]
            col = cell[1]
            any_part_visible = False

            # Check for all the four doors,
            for door_type in range(4):
                # Get the distance of door from middle of current cell
                # If the distance is less than or equal to r then the door is a part of the drone visual
                # Otherwise it is not
                if not self.validate_distance_between_drone_and_door(row, col, door_type):
                    continue

                any_part_visible = True

                # if door is a part of the drone visual
                # add to state whether they are open, closed or at boundary
                if row == 0 and door_type == constants.LEFT:
                    state.append((row-self.cur_pos[0], col-self.cur_pos[1], door_type, constants.BOUNDARY))
                elif row == constants.map_dim-1 and door_type == constants.RIGHT:
                    state.append((row-self.cur_pos[0], col-self.cur_pos[1], door_type, constants.BOUNDARY))
                elif col == 0 and door_type == constants.UP:
                    state.append((row-self.cur_pos[0], col-self.cur_pos[1], door_type, constants.BOUNDARY))
                elif col == constants.map_dim-1 and door_type == constants.DOWN:
                    state.append((row-self.cur_pos[0], col-self.cur_pos[1], door_type, constants.BOUNDARY))
                elif self.map_state[row][col][door_type] == 1:
                    state.append((row-self.cur_pos[0], col-self.cur_pos[1], door_type, constants.OPEN))
                else:
                    state.append((row-self.cur_pos[0], col-self.cur_pos[1], door_type, constants.CLOSED))

            # Go to the adjacent cells if any part is visible
            if not any_part_visible:
                continue

            # Check if this is end point
            if row == self.end_pos[0] and col == self.end_pos[1]:
                is_end_visible = True

            for i in range(4):
                adj_x = row + self.dRow[i]
                adj_y = col + self.dCol[i]
                if self.is_valid(adj_x, adj_y, vis):
                    q.append((adj_x, adj_y))
                    vis[adj_x][adj_y] = True

        return is_end_visible

    def get_drone_visual(self):
        # use breadth first search to traverse all the cells in a radius of r of the current position
        # and get the information on cell boundaries for each of the cells if they are open or closed
        # write code here
        # create an empty state to store the information on the cell boundaries
        state = []
        is_end_visible = self.BFS(state)

        return state, is_end_visible

    # Verify the action returned by the player
    def check_action(self, action):
        if action is None:
            print("No action returned")
            return False
        if type(action) is not int:
            print("Invalid action type")
            return False
        if action < -1 or action > 3:
            print("Invalid action value")
            return False
        return True

    # Validate if the move is possible by checking if move will cross
    # grid boundary or the doors are closed
    def check_and_apply_move(self, move):
        cur_y = self.cur_pos[1]
        cur_x = self.cur_pos[0]
        if move == constants.LEFT:
            if (cur_x != 0 and self.map_state[cur_x][cur_y][constants.LEFT] == 1
                    and self.map_state[cur_x-1][cur_y][constants.RIGHT] == 1):
                self.cur_pos[0] -= 1
                return True
        elif move == constants.UP:
            if (cur_y != 0 and self.map_state[cur_x][cur_y][constants.UP] == 1
                    and self.map_state[cur_x][cur_y-1][constants.DOWN] == 1):
                self.cur_pos[1] -= 1
                return True
        elif move == constants.RIGHT:
            if (cur_x != constants.map_dim - 1 and self.map_state[cur_x][cur_y][constants.RIGHT] == 1
                    and self.map_state[cur_x+1][cur_y][constants.LEFT] == 1):
                self.cur_pos[0] += 1
                return True
        elif move == constants.DOWN:
            if (cur_y != constants.map_dim - 1 and self.map_state[cur_x][cur_y][constants.DOWN] == 1
                    and self.map_state[cur_x][cur_y+1][constants.UP] == 1):
                self.cur_pos[1] += 1
                return True
        elif move == constants.WAIT:
            return True
        return False

    def get_state(self):
        return_dict = dict()
        return_dict['map_state'] = self.map_state
        return_dict['cur_pos'] = self.cur_pos
        return return_dict

    def draw_grid(self):
        self.canvas.delete("all")  # Clear the canvas

        for i in range(constants.map_dim):
            for j in range(constants.map_dim):
                x1, y1 = self.x_offset + i * constants.CELL_SIZE, self.y_offset + j * constants.CELL_SIZE
                x2, y2 = x1 + constants.CELL_SIZE, y1 + constants.CELL_SIZE

                # Draw the cell's doors based on door_states
                if self.map_state[i][j][constants.UP] != 1:  # Top door
                    self.canvas.create_line(x1, y1+0.5, x2, y1+0.5, fill="blue", width = 0.5)
                if self.map_state[i][j][constants.RIGHT] != 1:  # Right door
                    self.canvas.create_line(x2-0.5, y1, x2-0.5, y2, fill="blue", width = 0.5)
                if self.map_state[i][j][constants.DOWN] != 1:  # Bottom door
                    self.canvas.create_line(x1, y2-0.5, x2, y2-0.5, fill="red", width = 0.5)
                if self.map_state[i][j][constants.LEFT] != 1:  # Left door
                    self.canvas.create_line(x1+0.5, y1, x1+0.5, y2, fill="red", width = 0.5)

        # Mark the start, cur, and end positions
        self.mark_position(self.start_pos, "green")
        self.mark_position(self.cur_pos, "orange", True)
        self.mark_position(self.end_pos, "red")
        self.create_buttons()
        self.canvas.create_text(650, 20, text="Turns: {}".format(self.turns), font=("Arial", 14), fill="black",
                                                  activefill="gray", tags="turns text")
        self.canvas.create_text(750, 20, text="Start Pos: {}".format(self.start_pos), font=("Arial", 14), fill="black",
                                activefill="gray", tags="turns text")
        self.canvas.create_text(900, 20, text="End Pos: {}".format(self.end_pos), font=("Arial", 14), fill="black",
                                activefill="gray", tags="turns text")
        self.canvas.create_text(1050, 20, text="Cur Pos: {}".format(self.cur_pos), font=("Arial", 14), fill="black",
                                activefill="gray", tags="turns text")

    def create_buttons(self):
        # Create text-based "Pause" button on the canvas
        self.pause_btn = self.canvas.create_text(250, 20, text="Pause", font=("Arial", 14), fill="black",
                                                 activefill="gray", tags="pause_button")
        self.canvas.tag_bind("pause_button", "<Button-1>", lambda e: self.pause())

        # Create a text-based "Reset" button on the canvas
        self.resume_btn = self.canvas.create_text(350, 20, text="Start/Resume", font=("Arial", 14), fill="black",
                                                  activefill="gray", tags="resume_button")
        self.canvas.tag_bind("resume_button", "<Button-1>", lambda e: self.resume())

        self.resume_btn = self.canvas.create_text(450, 20, text="1X/4X", font=("Arial", 14), fill="black",
                                                  activefill="gray", tags="speed_button")
        self.canvas.tag_bind("speed_button", "<Button-1>", lambda e: self.toggle_speed())

        self.step_btn = self.canvas.create_text(550, 20, text="Step", font=("Arial", 14), fill="black",
                                                  activefill="gray", tags="step_button")
        self.canvas.tag_bind("step_button", "<Button-1>", lambda e: self.step())

    def mark_position(self, pos, color, withCircle = False):
        x, y = pos

        x1, y1 = self.x_offset + x * constants.CELL_SIZE + constants.CELL_SIZE / 5, self.y_offset + y * constants.CELL_SIZE + constants.CELL_SIZE / 5
        x2, y2 = x1 + constants.CELL_SIZE * 2/3, y1 + constants.CELL_SIZE * 2/3
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

        if withCircle:
            cx, cy = self.x_offset + x * constants.CELL_SIZE + constants.CELL_SIZE/2, self.y_offset + y * constants.CELL_SIZE + constants.CELL_SIZE/2
            r = self.radius*constants.CELL_SIZE
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="", outline="blue", width=1)
            
