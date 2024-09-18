########################################## Frank (9/16):
import constants
import random


class Experience:
    def __init__(self, L, r):
        self.L = L
        self.r = r
        self.cur_pos = (
            0,
            0,
        )  # (x, y) coordinates relative to the original start position
        self.seen_cells = (
            set()
        )  # set of tuples (x, y) storing coordinates of cells relative to the original start position
        self.walls = (
            float("inf"),
            float("inf"),
            float("-inf"),
            float("-inf"),
        )  # (right, top, left, bottom) coordinates relative to the original start position
        self.wait_penalty = 0.2  # penalty for waiting
        self.wait_penalty_multiplier = 1  # number of times the player has waited

    def move(self, current_percept):
        """Update experience with new cell seen in this move

        Args:
            current_percept(TimingMazeState): contains current state information
        """

        self.cur_pos = (-current_percept.start_x, -current_percept.start_y)

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
        if right < self.r:
            self.walls = (
                right + self.cur_pos[0],
                self.walls[1],
                self.walls[2],
                self.walls[3],
            )
        if top < self.r:
            self.walls = (
                self.walls[0],
                top + self.cur_pos[1],
                self.walls[2],
                self.walls[3],
            )
        if left > -self.r:
            self.walls = (
                self.walls[0],
                self.walls[1],
                left + self.cur_pos[0],
                self.walls[3],
            )
        if bottom > -self.r:
            self.walls = (
                self.walls[0],
                self.walls[1],
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
        print("\n")

        return move

    def wait(self):
        """Increment the number of times the player has waited"""
        self.wait_penalty_multiplier += 0.5

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
            move_scores[i] = move_scores[i] / max(move_scores)

        # Give penalty for waiting
        for i in range(4):
            if not self.is_valid_move(current_percept, i):
                move_scores[i] = (
                    move_scores[i] - self.wait_penalty * self.wait_penalty_multiplier
                )

        max_score = max(move_scores)
        max_indices = [i for i, score in enumerate(move_scores) if score == max_score]
        move = random.choice(max_indices)

        print(f"Move scores: {move_scores}")
        print(f'Move: {move}')
        return move

    def get_move_scores(self):
        """Score each move based on the number of new cells seen

        Returns:
            list: list of scores for each move (LEFT, UP, RIGHT, DOWN)
        """
        move_scores = [0, 0, 0, 0]

        for dx, dy in [
            (1, 0),
            (0, -1),
            (-1, 0),
            (0, 1),
        ]:  # LEFT, UP, RIGHT, DOWN
            num_new_cells = self.get_num_new_cells(
                self.cur_pos[0] + dx, self.cur_pos[1] + dy
            )
            if dx == 1 and dy == 0:
                move_scores[constants.LEFT] = num_new_cells
            elif dx == 0 and dy == -1:
                move_scores[constants.UP] = num_new_cells
            elif dx == -1 and dy == 0:
                move_scores[constants.RIGHT] = num_new_cells
            elif dx == 0 and dy == 1:
                move_scores[constants.DOWN] = num_new_cells
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

    def is_valid_move(self, current_percept, move):
        direction = [0, 0, 0, 0]
        for maze_state in current_percept.maze_state:
            if maze_state[0] == 0 and maze_state[1] == 0:
                direction[maze_state[2]] = maze_state[3]

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