import json
import os
import sys
import threading
import time
import signal
import numpy as np
import math
import pygame
from timing_maze_state import TimingMazeState
from constants import *
import constants
from utils import *
from glob import glob
from players.default_player import Player as DefaultPlayer
from collections import deque as queue
import queue as queue2

class TimingMazeGame:
    # Direction vectors
    dRow = [0, 1, 0, -1]
    dCol = [-1, 0, 1, 0]

    # Colors
    COLORS = {
        'player': GREEN,
        'flag': RED,
        'door': BLACK
    }

    def __init__(self, args):
        self.cur_pos = None
        self.end_pos = None
        self.start_time = time.time()
        self.use_gui = not args.no_gui
        self.do_logging = not args.disable_logging
        self.is_paused = False

        if not self.use_gui:
            self.use_timeout = not args.disable_timeout
        else:
            self.use_timeout = False

            os.makedirs("render", exist_ok=True)

            old_files = glob("render/*.png")
            for f in old_files:
                os.remove(f)

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

    def draw_grid(self):
        for x in range(0, WINDOW_SIZE, CELL_SIZE):
            for y in range(0, WINDOW_SIZE, CELL_SIZE):
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, WHITE, rect, 1)

    def draw_player(self, cur_pos):
        x, y = cur_pos
        # Flip this value as the y-axis is flipped in pygame
        y = constants.map_dim - y - 1
        pygame.draw.rect(self.screen, self.COLORS['player'], pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def draw_flag(self):
        x, y = self.end_pos[0], self.end_pos[1]
        # Flip this value as the y-axis is flipped in pygame
        y = constants.map_dim - y - 1
        pygame.draw.rect(self.screen, self.COLORS['flag'], pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def draw_door(self, doors):
        for x in range(constants.map_dim):
            for col in range(constants.map_dim):
                # Flip this value as the y-axis is flipped in pygame
                y = constants.map_dim - col - 1
                print(x, y, doors[x][y])
                for direction in range(4):
                    if doors[x][y][direction] != 1:
                        if direction == constants.LEFT:
                            start_pos = (x * CELL_SIZE, y * CELL_SIZE)
                            end_pos = (x * CELL_SIZE, y * CELL_SIZE + CELL_SIZE)
                        elif direction == constants.RIGHT:
                            start_pos = (x * CELL_SIZE + CELL_SIZE, y * CELL_SIZE)
                            end_pos = (x * CELL_SIZE + CELL_SIZE, y * CELL_SIZE + CELL_SIZE)
                        elif direction == constants.UP:
                            start_pos = (x * CELL_SIZE, y * CELL_SIZE)
                            end_pos = (x * CELL_SIZE + CELL_SIZE, y * CELL_SIZE)
                        elif direction == constants.DOWN:
                            start_pos = (x * CELL_SIZE, y * CELL_SIZE + CELL_SIZE)
                            end_pos = (x * CELL_SIZE + CELL_SIZE, y * CELL_SIZE + CELL_SIZE)

                        pygame.draw.line(self.screen, self.COLORS['door'], start_pos, end_pos, 2)

    def draw_button(self):
        """Draw a 'Pause' button."""
        button_color = (0, 255, 0) if not self.is_paused else (255, 0, 0)
        pygame.draw.rect(self.screen, button_color, (700, 20, 80, 40))  # Draw button
        font = pygame.font.SysFont(None, 55)
        button_text = font.render("Pause" if not self.is_paused else "Resume", True, (0, 0, 0))
        self.screen.blit(button_text, (705, 25))  # Display button text

    def toggle_pause(self):
        """Toggle the game's paused state."""
        self.is_paused = not self.is_paused

    def check_button_click(self, pos):
        """Check if the 'Pause' button was clicked."""
        if 700 <= pos[0] <= 780 and 20 <= pos[1] <= 60:  # Button coordinates
            self.toggle_pause()

    def pygame_loop(self, q, cur_pos, map_state):
        pygame.init()
        # Initialize screen
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption('Timing Maze Game')
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.check_button_click(pygame.mouse.get_pos())

            while not q.empty():
                cur_pos, map_state, running = q.get()

            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_player(cur_pos)
            self.draw_flag()
            self.draw_door(map_state)
            self.draw_button()
            pygame.display.flip()
            # Add a small delay to reduce CPU usage
            time.sleep(constants.GUI_SLEEP/3)
        pygame.quit()
        sys.exit()

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
                while 1:
                    self.end_pos = np.array([self.rng.integers(0, constants.map_dim),
                                             self.rng.integers(0, constants.map_dim)])
                    if self.end_pos[0] != self.cur_pos[0] and self.end_pos[1] != self.cur_pos[1]:
                        break

                # Generate a random map
                for i in range(constants.map_dim):
                    for j in range(constants.map_dim):
                        for k in range(4):
                            if self.rng.random() < 0.05:
                                self.map_frequencies[i][j][k] = 0
                            else:
                                self.map_frequencies[i][j][k] = self.rng.integers(1, self.max_door_frequency)

                # Assign n=0 to all boundary doors
                for i in range (constants.map_dim):
                    self.map_frequencies[0][i][constants.DOWN] = 0
                    self.map_frequencies[constants.map_dim-1][i][constants.UP] = 0
                    self.map_frequencies[i][0][constants.LEFT] = 0
                    self.map_frequencies[i][constants.map_dim-1][constants.RIGHT] = 0

                if self.validate_maze():
                    break

                print("Retrying to generate a valid maze...")

        print("Maze created successfully...")

        # To save the maze in a json file
        # data = {
        #     "frequencies": self.map_frequencies.tolist(),
        #     "start_pos": self.cur_pos.tolist(),
        #     "end_pos": self.end_pos.tolist()
        # }
        # filename = 'data.json'
        # with open(filename, 'w') as json_file:
        #     json.dump(data, json_file, indent=4)
        #
        # print(f"JSON file '{filename}' created successfully.")

        self.map_state = self.map_frequencies.copy()

        if self.use_gui:
            q = queue2.Queue()
            bg_thread = threading.Thread(target=self.play_game, args=(q,), daemon=True)
            bg_thread.start()
            print("pygame", 0)

            self.pygame_loop(q, self.cur_pos.copy(), self.map_state.copy())
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
            if self.map_frequencies[0][i][constants.DOWN] != 0:
                print("Error with UP")
                return False
            if self.map_frequencies[constants.map_dim-1][i][constants.UP] != 0:
                print("Error with DOWN")
                return False
            if self.map_frequencies[i][0][constants.LEFT] != 0:
                print("Error with LEFT")
                return False
            if self.map_frequencies[i][constants.map_dim-1][constants.RIGHT] != 0:
                print("Error with RIGHT")
                return False

        # Check that map has a valid start and end position.
        if self.cur_pos[0] < 0 or self.cur_pos[0] >= constants.map_dim or self.cur_pos[1] < 0 or self.cur_pos[1] >= constants.map_dim:
            print("Error with start")
            return False

        if self.end_pos[0] < 0 or self.end_pos[0] >= constants.map_dim or self.end_pos[1] < 0 or self.end_pos[1] >= constants.map_dim:
            print("Error with end")
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
                            if j > 0 and self.map_frequencies[i][j-1][constants.RIGHT] != 0:
                                graph[i][j][k] = 1
                        elif k == constants.RIGHT:
                            if j < constants.map_dim-1 and self.map_frequencies[i][j+1][constants.LEFT] != 0:
                                graph[i][j][k] = 1
                        elif k == constants.UP:
                            if i < constants.map_dim-1 and self.map_frequencies[i+1][j][constants.DOWN] != 0:
                                graph[i][j][k] = 1
                        elif k == constants.DOWN:
                            if i > 0 and self.map_frequencies[i-1][j][constants.UP] != 0:
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

        return visited_count == constants.total_cells

    def play_game(self, q=None):
        # Sleep needed to let pygame initialize
        time.sleep(constants.GUI_SLEEP)

        while self.turns != self.max_turns:
            self.turns += 1
            self.play_turn(q)
            print("Turn {} complete".format(self.turns))

            if self.cur_pos[0] == self.end_pos[0] and self.cur_pos[1] == self.end_pos[1]:
                self.goal_reached = True
                print("Goal reached!\n\n Turns taken: {}\n".format(self.turns))
                break

        if not self.goal_reached:
            print("Goal not reached...\n\n")

        if self.use_gui:
            q.put((self.cur_pos.copy(), self.map_state.copy(), False))

        self.end_time = time.time()
        print("\nTime taken: {}\nValid moves: {}\n".format(self.end_time - self.start_time, self.valid_moves))


    # Check that the map has a start and end door.
    # Need to check if all cells are reachable from one another,
    # For this if a pair of doors between two cells are both non-zero. Then, This intersection between the two cells is a valid path otherwise not.
    # So we can create an undirected graph and check if the map is valid by looking for islands in the graph.

    def play_turn(self, q):
        # Get the drone visual for a radius of r
        maze_state, is_end_visible = self.get_drone_visual()

        # Create the state object for the player
        before_state = TimingMazeState(maze_state, is_end_visible,
                                       self.end_pos[0]-self.cur_pos[0], self.end_pos[1]-self.cur_pos[1])
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
            q.put((self.cur_pos.copy(), self.map_state.copy(), True))

        self.update_door_state()
        if self.use_gui:
            time.sleep(constants.GUI_SLEEP)

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
            door_x = row + 0.5
            door_y = col
            distance = self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y)

            # calculate the distance between the drone and the bottom end of the door
            door_x = row
            door_y = col
            distance = min(distance,
                self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))

            # calculate the distance between the drone and the top end of the door
            door_x = row + 1
            door_y = col
            distance = min(distance,
                self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))

        elif door_type == constants.RIGHT:
            # calculate the distance between the drone and the centre of the door
            door_x = row + 0.5
            door_y = col + 1
            distance = self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y)

            # calculate the distance between the drone and the bottom end of the door
            door_x = row
            door_y = col + 1
            distance = min(distance,
                           self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))

            # calculate the distance between the drone and the top end of the door
            door_x = row + 1
            door_y = col + 1
            distance = min(distance,
                           self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))

        elif door_type == constants.UP:
            # calculate the distance between the drone and the centre of the door
            door_x = row + 1
            door_y = col + 0.5
            distance = self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y)

            # calculate the distance between the drone and the left end of the door
            door_x = row + 1
            door_y = col
            distance = min(distance,
                           self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))

            # calculate the distance between the drone and the right end of the door
            door_x = row + 1
            door_y = col + 1
            distance = min(distance,
                           self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y))
        elif door_type == constants.DOWN:
            # calculate the distance between the drone and the centre of the door
            door_x = row
            door_y = col + 0.5
            distance = self.get_euclidean_distance_between_two_points(drone_x, drone_y, door_x, door_y)

            # calculate the distance between the drone and the left end of the door
            door_x = row
            door_y = col
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

            # Check if this is end point
            if row == self.end_pos[0] and col == self.end_pos[1]:
                is_end_visible = True

            # Check for all the four doors,
            for door_type in range(4):
                # Get the distance of door from middle of current cell
                # If the distance is less than or equal to r then the door is a part of the drone visual
                # Otherwise it is not
                if not self.validate_distance_between_drone_and_door(row, col, door_type):
                    continue

                # if door is a part of the drone visual
                # add to state whether they are open, closed or at boundary
                if row == 0 and door_type == constants.UP:
                    state.append((row-self.cur_pos[0], col-self.cur_pos[1], door_type, constants.BOUNDARY))
                elif row == constants.map_dim-1 and door_type == constants.DOWN:
                    state.append((row-self.cur_pos[0], col-self.cur_pos[1], door_type, constants.BOUNDARY))
                elif col == 0 and door_type == constants.LEFT:
                    state.append((row-self.cur_pos[0], col-self.cur_pos[1], door_type, constants.BOUNDARY))
                elif col == constants.map_dim-1 and door_type == constants.RIGHT:
                    state.append((row-self.cur_pos[0], col-self.cur_pos[1], door_type, constants.BOUNDARY))
                elif self.map_state[row][col][door_type] == 1:
                    state.append((row-self.cur_pos[0], col-self.cur_pos[1], door_type, constants.OPEN))
                else:
                    state.append((row-self.cur_pos[0], col-self.cur_pos[1], door_type, constants.CLOSED))

            # Go to the adjacent cells
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
        cur_y = self.cur_pos[0]
        cur_x = self.cur_pos[1]
        if move == constants.LEFT:
            if (cur_x != 0 and self.map_state[cur_y][cur_x][constants.LEFT] == 1
                    and self.map_state[cur_y][cur_x-1][constants.RIGHT] == 1):
                self.cur_pos[1] -= 1
                return True
        elif move == constants.UP:
            if (cur_y != constants.map_dim - 1 and self.map_state[cur_y][cur_x][constants.UP] == 1
                    and self.map_state[cur_y+1][cur_x][constants.DOWN] == 1):
                self.cur_pos[0] += 1
                return True
        elif move == constants.RIGHT:
            if (cur_x != constants.map_dim - 1 and self.map_state[cur_y][cur_x][constants.RIGHT] == 1
                    and self.map_state[cur_y][cur_x+1][constants.LEFT] == 1):
                self.cur_pos[1] += 1
                return True
        elif move == constants.DOWN:
            if (cur_y != 0 and self.map_state[cur_y][cur_x][constants.DOWN] == 1
                    and self.map_state[cur_y-1][cur_x][constants.UP] == 1):
                self.cur_pos[0] -= 1
                return True
        elif move == constants.WAIT:
            return True
        return False

    def get_state(self):
        return_dict = dict()
        return_dict['map_state'] = self.map_state
        return_dict['cur_pos'] = self.cur_pos
        return return_dict
