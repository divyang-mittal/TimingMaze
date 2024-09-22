import numpy as np
import logging
import random

from constants import UP, DOWN, LEFT, RIGHT, WAIT, BOUNDARY
from players.g6_player.classes.typed_timing_maze_state import (
    TypedTimingMazeState,
    convert,
)
from players.g6_player.data import Move
from timing_maze_state import TimingMazeState
from players.g6_player.classes.maze import Maze


class G6_Player:
    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        precomp_dir: str,
        maximum_door_frequency: int,
        radius: int,
    ) -> None:
        """Initialise the player with the basic amoeba information

        Args:
            rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
            logger (logging.Logger): logger use this like logger.info("message")
            maximum_door_frequency (int): the maximum frequency of doors
            radius (int): the radius of the drone
            precomp_dir (str): Directory path to store/load pre-computation
        """
        self.rng = rng
        self.logger = logger
        self.maximum_door_frequency = maximum_door_frequency
        self.radius = radius

        # Variables to facilitate knowing where the player has been and if they are trapped
        self.turn = 0
        self.stuck = 0
        self.move_history = []
        self.prev_move = None

        # Initialize Maze object to hold information about cells and doors perceived by the drone
        self.maze = Maze(self.maximum_door_frequency, self.radius)

        # an interim target which the agent tries to navigate towards
        self.search_target = None

        # variables to track inward spiral
        self.found_right_boundary = False
        self.found_down_boundary = False
        self.layer = 0
        self.phase = 4  # phases match directions in constants.py; 4 = find SE corner

    def move(self, current_percept: TimingMazeState) -> int:
        """
        Increments the turn count and updates the maze with the current percept. Calls
        __move() to determine the next move.
        """
        self.turn += 1
        current_percept: TypedTimingMazeState = convert(current_percept)

        self.maze.update(current_percept, self.turn)
        self.__update_history()
        player_move = self.__move(current_percept)

        return player_move.value

    def __update_history(self):
        """
        This function adjusts the move_history ordered list of coordinates that the player has already visited.
        """
        if len(self.move_history) == 0:
            return self.move_history.append(self.maze.curr_pos)
        elif self.move_history[-1] == self.maze.curr_pos:
            self.stuck += 1
            return
        else:
            self.stuck = 0
            self.prev_move = self.__get_prev_move()
            return self.move_history.append(self.maze.curr_pos)

    def __get_prev_move(self):
        delta = (
            self.maze.curr_pos[0] - self.move_history[-1][0],
            self.maze.curr_pos[1] - self.move_history[-1][1],
        )

        if delta == (-1, 0):
            return LEFT
        elif delta == (1, 0):
            return RIGHT
        elif delta == (0, -1):
            return DOWN
        else:
            return UP

    def __move(self, current_percept: TypedTimingMazeState) -> Move:
        """
        Helper function to move().
        """
        # Explore map to get target within drone's view
        if not current_percept.is_end_visible:
            return self.__explore()

        # Otherwise, go to target
        return self.__exploit(current_percept)

    def __explore(self) -> Move:
        """
        Move towards the southeast corner and perform inward spiral when right
        and down boundaries are visible by drone
        """
        if self.stuck >= (
            self.maximum_door_frequency * (self.maximum_door_frequency - 1)
        ):
            return self.__get_unstuck()

        if not self.found_right_boundary:
            self.found_right_boundary = self.__is_boundary_in_sight(RIGHT)
        if not self.found_down_boundary:
            self.found_down_boundary = self.__is_boundary_in_sight(DOWN)

        if not self.found_right_boundary and not self.found_down_boundary:
            return self.__greedy_move(directions=[RIGHT, DOWN])
        elif not self.found_right_boundary:
            return self.__greedy_move(directions=[RIGHT])
        elif not self.found_down_boundary:
            return self.__greedy_move(directions=[DOWN])

        return self.__inward_spiral()

    def __is_boundary_in_sight(self, direction: int) -> bool:
        """
        Check if boundary is in sight in the given direction
        """
        curr_x, curr_y = self.maze.curr_pos
        curr_cell = self.maze.get_cell(curr_x, curr_y)

        for _ in range(self.radius + 1):
            if direction == RIGHT:
                if curr_cell.e_door.state == BOUNDARY:
                    self.maze.update_boundary(curr_cell, RIGHT)
                    return True
                curr_cell = curr_cell.e_cell
            elif direction == DOWN:
                if curr_cell.s_door.state == BOUNDARY:
                    self.maze.update_boundary(curr_cell, DOWN)
                    return True
                curr_cell = curr_cell.s_cell
            elif direction == LEFT:
                if curr_cell.w_door.state == BOUNDARY:
                    self.maze.update_boundary(curr_cell, LEFT)
                    return True
                curr_cell = curr_cell.w_cell
            elif direction == UP:
                if curr_cell.n_door.state == BOUNDARY:
                    self.maze.update_boundary(curr_cell, UP)
                    return True
                curr_cell = curr_cell.n_cell

        return False

    def __inward_spiral(self):
        """
        Perform clockwise inward spiral starting from the southeast corner.
        """
        # Set initial search target so that radius touches southeast corner
        if self.layer == 0 and self.phase == 4:
            offset = int(np.floor(self.radius / np.sqrt(2)))
            self.search_target = (
                self.maze.east_end - offset,
                self.maze.south_end - offset,
            )

        # Set inward spiral phase and layer and update search target
        # [TODO] In case we are stuck getting to the exact target, we could consider
        # setting a max distance from target before we move on to the next phase
        if self.maze.curr_pos == self.search_target:
            self.__adjust_phase_and_target()

        # print(f'Phase: {self.phase}, Layer: {self.layer}, Target: {self.search_target}, curr_pos: {self.maze.curr_pos}')

        return self.__greedy_move(target=self.search_target)

    def __adjust_phase_and_target(self):
        """
        Adjust phase, layer and search target to perform inward spiral
        """
        self.phase = (self.phase + 1) % 5
        if self.phase == 4:
            self.layer += 1

        offset = int(np.floor(self.radius / np.sqrt(2)))
        cum_offset = (2 * self.layer + 1) * offset

        # Set search target so that radius touches southeast corner of previous layer
        if self.phase == 4:
            self.search_target = (
                self.maze.east_end - cum_offset,
                self.maze.south_end - cum_offset,
            )

        # Set search target so that radius touches southwest corner of previous layer
        elif self.phase == LEFT:
            self.search_target = (
                self.maze.west_end + cum_offset,
                self.maze.south_end - cum_offset,
            )

        # Set search target so that radius touches northwest corner of previous layer
        elif self.phase == UP:
            self.search_target = (
                self.maze.west_end + cum_offset,
                self.maze.north_end + cum_offset,
            )

        # Set search target so that radius touches northeast corner of previous layer
        elif self.phase == RIGHT:
            self.search_target = (
                self.maze.east_end - cum_offset,
                self.maze.north_end + cum_offset,
            )

        # Set search target so that radius touches southeast corner of previous layer
        # Extra offset from south border to avoid overlapping with previous layer
        elif self.phase == DOWN:
            self.search_target = (
                self.maze.east_end - cum_offset,
                self.maze.south_end - cum_offset - offset,
            )

    def __get_available_moves(self):
        curr_x, curr_y = self.maze.curr_pos
        curr_cell = self.maze.get_cell(curr_x, curr_y)
        curr_available_moves = []

        for move in Move:
            if curr_cell.is_move_available(move):
                curr_available_moves.append(move)
        return curr_available_moves

    def __get_unstuck(self):
        curr_available_moves = self.__get_available_moves()

        # [TODO] - this will not work if there is a full three-sided trap (like a maze with a dead end).
        print("Previous move: {}".format(self.prev_move))
        print("Available moves: {}".format(curr_available_moves))

        for available_move in curr_available_moves:
            if self.prev_move in [RIGHT, LEFT] and available_move in [
                Move.UP,
                Move.DOWN,
            ]:
                return available_move
            elif self.prev_move in [DOWN, UP] and available_move in [
                Move.LEFT,
                Move.RIGHT,
            ]:
                return available_move
            else:
                return Move.WAIT

    def __greedy_move(self, directions: list[int] = [], target: tuple = ()) -> Move:
        """
        Given a list of directions in order of priority or target coordinates, navigate
        towards the target direction in a greedy manner.
        [TODO] Consider tactics for avoiding walls and finding shortest paths
        """
        if directions:
            if directions[0] == RIGHT:
                return Move.RIGHT
            if directions[0] == DOWN:
                return Move.DOWN
            if directions[0] == LEFT:
                return Move.LEFT
            if directions[0] == UP:
                return Move.UP

        elif target:
            if self.maze.curr_pos[0] < target[0]:
                return Move.RIGHT
            if self.maze.curr_pos[0] > target[0]:
                return Move.LEFT
            if self.maze.curr_pos[1] < target[1]:
                return Move.DOWN
            if self.maze.curr_pos[1] > target[1]:
                return Move.UP

        return Move.WAIT

    def __panic_escape(self):
        curr_available_moves = self.__get_available_moves()
        return random.choice(curr_available_moves)

    def __exploit(self, current_state: TypedTimingMazeState) -> Move:
        """
        [TODO] Implement A star algorithm
        [TODO] Implement greedy algorithm if one of the following conditions is met:
        a) after a certain number of turns (e.g. 3x Manhattan distance to target)
        b) when we are not getting closer to the target after a certain number of turns
        c) 10% random chance for any given turn
        """
        if random.random() < 0.1:
            return random.choice(list(Move))

        if 0 > current_state.end_x:
            return Move.LEFT

        if 0 < current_state.end_x:
            return Move.RIGHT

        if 0 > current_state.end_y:
            return Move.UP

        if 0 < current_state.end_y:
            return Move.DOWN

        return Move.WAIT

    def __exploit_a_star(self, current_state: TypedTimingMazeState) -> Move:
        """
        [TODO] Use the A* shortest_path to generate moves towards the target.:
        """
        start = self.maze.curr_pos
        target = (self.maze.target_pos[0], self.maze.target_pos[1])
        shortest_path, path_length = self.maze.graph.astar_shortest_path(start, target)

        return Move.WAIT

    def __str__(self) -> str:
        # TODO: how do we get the current position
        return "G6_Player()"

    def __repr__(self) -> str:
        return str(self)
