########################################## Frank (9/16):
import constants
import random
import numpy as np


class Experience:
    def __init__(self, L, r):
        # Hyper-parameters
        self.wait_penalty = 0.2  # penalty for waiting
        self.revisit_penalty = 0.1  # penalty for revisiting a cell
        self.revisit_max_penalty = 1  # maximum penalty for revisiting a cell
        self.direction_vector_max_weight = 2  # maximum weight of the direction vector
        self.direction_vector_multiplier = 0.01  # multiplier for the direction vector
        self.direction_vector_pov_radius = (
            30  # radius of the field of view for the direction vector
        )

        self.L = L
        self.r = r
        self.num_turns = 0
        self.cur_pos = (
            0,
            0,
        )  # (x, y) coordinates relative to the original start position
        self.maze_dimension = 100  # size of the maze
        self.seen_cells = (
            set()
        )  # set of tuples (x, y) storing coordinates of cells relative to the original start position
        self.walls = (
            float("inf"),
            float("inf"),
            float("-inf"),
            float("-inf"),
        )  # (right, top, left, bottom) coordinates relative to the original start position
        self.stays = {}  # key: (x, y), value: number of stays at the position
        self.direction_vector_weight = min(
            self.direction_vector_max_weight,
            self.direction_vector_multiplier * self.num_turns,
        )  # weight of the direction vector

    def move(self, current_percept):
        """Update experience with new cell seen in this move

        Args:
            current_percept(TimingMazeState): contains current state information
        """

        self.cur_pos = (-current_percept.start_x, -current_percept.start_y)
        self.stays[self.cur_pos] = self.stays.get(self.cur_pos, 0) + 1
        self.num_turns += 1
        self.direction_vector_weight = min(
            self.direction_vector_max_weight,
            self.direction_vector_multiplier * self.num_turns,
        )  # update direction vector weight

        # initialize coordinates for the maximum field of view relative to current position
        right, top, left, bottom = 0, 0, 0, 0

        for cell in current_percept.maze_state:

            # update field of view coordinates relative to current position
            right, top, left, bottom = (
                max(right, cell[0]),
                max(top, cell[1]),
                min(left, cell[0]),
                min(bottom, cell[1]),
            )

            cell = (
                self.cur_pos[0] + cell[0],
                self.cur_pos[1] + cell[1],
            )
            if cell not in self.seen_cells:
                self.seen_cells.add(cell)

        # update walls coordinates relative to the original start position
        # TODO: infer left wall from right wall, and bottom wall from top wall
        if right < self.r:
            self.walls = (
                right + self.cur_pos[0],
                self.walls[1],
                right + self.cur_pos[0] - self.maze_dimension,
                self.walls[3],
            )
        if top < self.r:
            self.walls = (
                self.walls[0],
                top + self.cur_pos[1],
                self.walls[2],
                top + self.cur_pos[1] - self.maze_dimension,
            )
        if left > -self.r:
            self.walls = (
                left + self.cur_pos[0] + self.maze_dimension,
                self.walls[1],
                left + self.cur_pos[0],
                self.walls[3],
            )
        if bottom > -self.r:
            self.walls = (
                self.walls[0],
                bottom + self.cur_pos[1] + self.maze_dimension,
                self.walls[2],
                bottom + self.cur_pos[1],
            )

        move = self.get_best_move(current_percept)

        # print(f"Current position: {self.cur_pos}")
        # print(
        #     f"Best move: {'WAIT' if move == constants.WAIT else 'LEFT' if move == constants.LEFT else 'UP' if move == constants.UP else 'RIGHT' if move == constants.RIGHT else 'DOWN'}"
        # )
        # print(f"Walls: {self.walls}")
        # print(f"Number of seen cells: {len(self.seen_cells)}")
        # print("\n")

        if self.is_valid_move(current_percept, move):
            return move
        else:
            self.wait()
            return constants.WAIT

    def wait(self):
        """Increment the number of times the player has waited"""
        self.stays[self.cur_pos] = self.stays.get(self.cur_pos, 0) + 1

    def get_direction_vector(self):
        direction_vector = [0, 0]  # [x, y]
        for x in range(
            max(self.cur_pos[0] - self.direction_vector_pov_radius, self.walls[2]),
            min(self.cur_pos[0] + self.direction_vector_pov_radius, self.walls[0]),
        ):
            for y in range(
                max(self.cur_pos[1] - self.direction_vector_pov_radius, self.walls[3]),
                min(
                    self.cur_pos[1] + self.direction_vector_pov_radius + 1,
                    self.walls[1],
                ),
            ):
                if (x, y) not in self.seen_cells:
                    direction = (x - self.cur_pos[0], y - self.cur_pos[1])
                    if direction[0] != 0:
                        direction_vector[0] += 1 / direction[0]
                    if direction[1] != 0:
                        direction_vector[1] += 1 / direction[1]

        # Normalize and add weight to direction vector
        norm = np.linalg.norm(direction_vector)
        if np.isfinite(norm) and norm > 0:
            direction_vector = (
                np.array(direction_vector) / norm * self.direction_vector_weight
            )
        else:
            direction_vector = np.zeros_like(direction_vector)

        return direction_vector

    def get_best_move(self, current_percept):
        """Evaluate best move

        Returns:
            int:
                WAIT = -1
                LEFT = 0
                UP = 1
                RIGHT = 2
                DOWN = 3
        """
        move_scores = self.get_move_scores()

        # Normalize move scores
        for i in range(4):
            move_scores[i] = move_scores[i] / max([1, max(move_scores)])

        direction_vector = self.get_direction_vector()

        for i in range(4):
            # Give penalty for waiting
            if not self.is_valid_move(current_percept, i):
                move_scores[i] = move_scores[i] - self.wait_penalty * self.stays.get(
                    self.cur_pos, 0
                )

            # Add direction vector to move scores
            if i == constants.LEFT:
                move_scores[i] -= direction_vector[0]
                move_scores[i] -= max(
                    self.stays.get((self.cur_pos[0] - 1, self.cur_pos[1]), 0)
                    * self.revisit_penalty,
                    self.revisit_max_penalty,
                )
            elif i == constants.UP:
                move_scores[i] -= direction_vector[1]
                move_scores[i] -= max(
                    self.stays.get((self.cur_pos[0], self.cur_pos[1] - 1), 0)
                    * self.revisit_penalty,
                    self.revisit_max_penalty,
                )
            elif i == constants.RIGHT:
                move_scores[i] += direction_vector[0]
                move_scores[i] -= max(
                    self.stays.get((self.cur_pos[0] + 1, self.cur_pos[1]), 0)
                    * self.revisit_penalty,
                    self.revisit_max_penalty,
                )
            elif i == constants.DOWN:
                move_scores[i] += direction_vector[1]
                move_scores[i] -= max(
                    self.stays.get((self.cur_pos[0], self.cur_pos[1] + 1), 0)
                    * self.revisit_penalty,
                    self.revisit_max_penalty,
                )

        max_score = max(move_scores)
        max_indices = [i for i, score in enumerate(move_scores) if score == max_score]
        move = random.choice(max_indices)

        # print(f"Direction vector: {direction_vector}")
        # print(f"Direction vector weight: {self.direction_vector_weight}")
        # print(f"Move scores: {move_scores}")
        return move

    def get_move_scores(self):
        """Score each move based on the number of new cells seen

        Returns:
            list: list of scores for each move (LEFT, UP, RIGHT, DOWN)
        """
        move_scores = [0, 0, 0, 0]

        # Define the corners based on walls
        corners = [
            (self.walls[2], self.walls[3]),  # Bottom-left corner
            (self.walls[2], self.walls[1]),  # Top-left corner
            (self.walls[0], self.walls[3]),  # Bottom-right corner
            (self.walls[0], self.walls[1]),  # Top-right corner
        ]

        for dx, dy, direction in [
            (-1, 0, constants.LEFT),
            (0, -1, constants.UP),
            (1, 0, constants.RIGHT),
            (0, 1, constants.DOWN),
        ]:
            new_x = self.cur_pos[0] + dx
            new_y = self.cur_pos[1] + dy
            num_new_cells = self.get_num_new_cells(new_x, new_y)

            # Score for the number of new cells seen
            move_scores[direction] = num_new_cells

            # Check if corners are visible within the radius and unvisited
            for corner in corners:
                corner_x, corner_y = corner
                if abs(corner_x - new_x) <= self.r and abs(corner_y - new_y) <= self.r:
                    if corner not in self.seen_cells:
                        move_scores[direction] += 5  # Extra score for visible unvisited corners

            # Adjust for distance to walls and hugging behavior
            distance_to_wall = -1
            if direction == constants.LEFT and self.walls[2] != float('inf'):
                distance_to_wall = abs(self.walls[2] - new_x)
            elif direction == constants.UP and self.walls[1] != float('inf'):
                distance_to_wall = abs(self.walls[1] - new_y)
            elif direction == constants.RIGHT and self.walls[0] != float('inf'):
                distance_to_wall = abs(self.walls[0] - new_x)
            elif direction == constants.DOWN and self.walls[3] != float('inf'):
                distance_to_wall = abs(self.walls[3] - new_y)

            # Score for hugging walls
            if distance_to_wall != -1:
                move_scores[direction] += (1 / (distance_to_wall + 1)) * 10
                if distance_to_wall < self.r:
                    move_scores[direction] += 1

        # Adjust scores if close to walls for hugging behavior
        if ((self.walls[0] != float('inf') and self.walls[0] <= self.cur_pos[0] + self.r) or
            (self.walls[2] != float('inf') and self.walls[2] >= self.cur_pos[0] + self.r)):
            move_scores[constants.DOWN] += 1
            move_scores[constants.UP] += 1
        elif ((self.walls[1] != float('inf') and self.walls[1] <= self.cur_pos[1] + self.r) or
            (self.walls[3] != float('inf') and self.walls[3] >= self.cur_pos[1] + self.r)):
            move_scores[constants.RIGHT] += 1
            move_scores[constants.LEFT] += 1

        return move_scores



    def get_num_new_cells(self, x, y):
        """Get the number of new cells seen at a new position

        Args:
            x (int): x-coordinate of the new position
            y (int): y-coordinate of the new position

        Returns:
            int: number of new cells seen at the new position
        """

        num_new_cells = 0
        for dx in range(-self.r, self.r + 1):
            for dy in range(-self.r, self.r + 1):
                if dx**2 + dy**2 <= self.r**2:
                    if (x + dx, y + dy) not in self.seen_cells and (
                        self.walls[2] <= x + dx <= self.walls[0]
                        and self.walls[3] <= y + dy <= self.walls[1]
                    ):
                        num_new_cells += 1
        return num_new_cells

    # TODO: This function can be sped up by decreasing the number iterations
    def is_valid_move(self, current_percept, move):
        direction = [0, 0, 0, 0]
        for maze_state in current_percept.maze_state:
            if maze_state[0] == 0 and maze_state[1] == 0:
                direction[maze_state[2]] = maze_state[3]

        if direction[move] != constants.OPEN:
            return False

        if move == constants.LEFT:
            for maze_state in current_percept.maze_state:
                if (
                    maze_state[0] == -1
                    and maze_state[1] == 0
                    and maze_state[2] == constants.RIGHT
                    and maze_state[3] == constants.OPEN
                ):
                    return True
        elif move == constants.UP:
            for maze_state in current_percept.maze_state:
                if (
                    maze_state[0] == 0
                    and maze_state[1] == -1
                    and maze_state[2] == constants.DOWN
                    and maze_state[3] == constants.OPEN
                ):
                    return True
        elif move == constants.RIGHT:
            for maze_state in current_percept.maze_state:
                if (
                    maze_state[0] == 1
                    and maze_state[1] == 0
                    and maze_state[2] == constants.LEFT
                    and maze_state[3] == constants.OPEN
                ):
                    return True
        elif move == constants.DOWN:
            for maze_state in current_percept.maze_state:
                if (
                    maze_state[0] == 0
                    and maze_state[1] == 1
                    and maze_state[2] == constants.UP
                    and maze_state[3] == constants.OPEN
                ):
                    return True

        return False


##########################################
