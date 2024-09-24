import numpy as np
import logging

from constants import UP, DOWN, LEFT, RIGHT, WAIT, BOUNDARY
from players.g6_player.a_star import a_star
from players.g6_player.classes.typed_timing_maze_state import (
    TypedTimingMazeState,
    convert,
)
from players.g6_player.data import Move
from timing_maze_state import TimingMazeState
from players.g6_player.classes.maze import Maze

from players.g6_player.data import move_to_str


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

        # Initialize Maze object to hold information about cells and doors perceived by the drone
        self.maze = Maze()

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
        current_percept: TypedTimingMazeState = convert(current_percept)

        self.maze.update(current_percept)
        player_move = self.__move(current_percept)

        print(f"MOVE: {move_to_str(player_move)}")
        return player_move.value

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
        if not self.found_right_boundary:
            self.found_right_boundary = self.__is_boundary_in_sight(RIGHT)
        if not self.found_down_boundary:
            self.found_down_boundary = self.__is_boundary_in_sight(DOWN)

        if self.found_right_boundary and self.found_down_boundary:
            return self.__inward_spiral()    

        if not self.found_right_boundary and not self.found_down_boundary:
            self.search_target = (self.maze.east_end, self.maze.south_end)
        elif not self.found_right_boundary:
            self.search_target = (self.maze.east_end, self.maze.curr_pos[1])
        elif not self.found_down_boundary:
            self.search_target = (self.maze.curr_pos[0], self.maze.south_end)

        self.maze.target_pos = self.__set_target_on_radius()
        result, cost = a_star(self.maze.current_cell(), self.maze.target_cell())
        print(f"TARGET: {len(result)} moves - {cost} cost")
        return result[0]

    def __is_boundary_in_sight(self, direction: int) -> bool:
        """
        Check if boundary is in sight in the given direction
        """
        curr_cell = self.maze.current_cell()

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
    
    def __set_target_on_radius(self) -> tuple:
        """
        Set target position on drone radius towards search target
        """
        vec_x = self.search_target[0] - self.maze.curr_pos[0]
        vec_y = self.search_target[1] - self.maze.curr_pos[1]
        norm = np.sqrt(vec_x**2 + vec_y**2)
        if norm > self.radius:
            x = int(np.floor(vec_x / norm * self.radius)) + self.maze.curr_pos[0]
            y = int(np.floor(vec_y / norm * self.radius)) + self.maze.curr_pos[1]
        else:
            x, y = self.search_target
        return (x, y)

    def __inward_spiral(self) -> Move:
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
        if self.maze.curr_pos == self.search_target:
            self.__adjust_phase_and_target()

        # A* search to target
        self.maze.target_pos = self.__set_target_on_radius()
        result, cost = a_star(self.maze.current_cell(), self.maze.target_cell())
        print(f"TARGET: {len(result)} moves - {cost} cost")

        # Move in least obstructed direction towards target
        target_directions = self.__get_target_directions()     
        door1_freq = self.__get_door_freq(target_directions[0])
        door2_freq = self.__get_door_freq(target_directions[1])
        door3_freq = self.__get_door_freq(target_directions[2])
        door4_freq = self.__get_door_freq(target_directions[3])

        # [TODO] Can experiment with different cost and freq thresholds
        if cost == float("inf"):
            if door1_freq != 0:
                return Move.target_directions[0]
            elif door2_freq != 0:
                return Move.target_directions[1]
            elif door3_freq != 0:
                return Move.target_directions[2]
            elif door4_freq != 0:
                return Move.target_directions[3]

        return result[0]

    def __get_target_directions(self) -> list:
        """
        Get directions to target based on current and target positions
        """
        # Rank directions based on distance to target
        left_dist = max(self.maze.curr_pos[0] - self.maze.target_pos[0], 0)
        up_dist = max(self.maze.curr_pos[1] - self.maze.target_pos[1], 0)
        right_dist = max(self.maze.target_pos[0] - self.maze.curr_pos[0], 0)
        down_dist = max(self.maze.target_pos[1] - self.maze.curr_pos[1], 0)
        dist_arr = [left_dist, up_dist, right_dist, down_dist]

        # Sort in descending order
        rank = np.argsort(-dist_arr)

        # Rank bottom two directions based on distance from border
        dist1 = 0
        dist2 = 0
        if rank[2] == LEFT:
            dist1 = self.maze.curr_pos[0] - self.maze.west_end
        elif rank[2] == UP:
            dist1 = self.maze.curr_pos[1] - self.maze.north_end
        elif rank[2] == RIGHT:
            dist1 = self.maze.east_end - self.maze.curr_pos[0]
        elif rank[2] == DOWN:
            dist1 = self.maze.south_end - self.maze.curr_pos[1]
        
        if rank[3] == LEFT:
            dist2 = self.maze.curr_pos[0] - self.maze.west_end
        elif rank[3] == UP:
            dist2 = self.maze.curr_pos[1] - self.maze.north_end
        elif rank[3] == RIGHT:
            dist2 = self.maze.east_end - self.maze.curr_pos[0]
        elif rank[3] == DOWN:
            dist2 = self.maze.south_end - self.maze.curr_pos[1]
        
        if dist1 < dist2:
            rank[2], rank[3] = rank[3], rank[2]

        return rank

    def __get_door_freq(self, direction: int) -> int:
        """
        Get the frequency of the door in a given direction
        """
        curr_cell = self.maze.current_cell()
        if direction == UP:
            return curr_cell.n_door.freq
        elif direction == RIGHT:
            return curr_cell.e_door.freq
        elif direction == DOWN:
            return curr_cell.s_door.freq
        elif direction == LEFT:
            return curr_cell.w_door.freq

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

    def __exploit(self, current_state: TypedTimingMazeState) -> Move:
        """
        Use the A* shortest path to generate moves towards the target.
        """

        assert current_state.end_x is not None
        assert current_state.end_y is not None

        result, cost = a_star(self.maze.current_cell(), self.maze.target_cell())

        # this shouldn't happen
        if len(result) == 0:
            return Move.WAIT

        print(f"TARGET: {len(result)} moves - {cost} cost")

        # [TODO] IMPLEMENT SAME TACTICAL MOVE AS IN EXPLORE

        return result[0]

    def __str__(self) -> str:
        # TODO: how do we get the current position
        return "G6_Player()"

    def __repr__(self) -> str:
        return str(self)
