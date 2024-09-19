import numpy as np

from constants import *
from players.g6_player.data import Move
from timing_maze_state import TimingMazeState
from players.g6_player.classes.maze import Maze


def explore(self) -> Move:
    """
    Move towards the southeast corner and perform inward spiral
    """
    found_right_boundary = True if self.curr_pos[0] == self.edges[0] else False
    found_down_boundary = True if self.curr_pos[1] == self.edges[1] else False
    if not found_right_boundary:
        return self.find_boundary(RIGHT)
    if not found_down_boundary:
        return self.find_boundary(DOWN)
    return self.inward_spiral()

def find_boundary(self, direction) -> Move:
    """
    Traverse to right/down boundary by setting search target to the farthest right/down
    direction. If edge is detected between current position and search target, set
    search target to edge.
    """
    if direction == RIGHT:
        self.search_target = (map_dim-1, 0)
        if self.edges is not None and self.__border_between_target_and_curr(
            self.edges[0], self.search_target[0], self.curr_pos[0]
        ):
            self.search_target[0] = self.edges[0]
        return Move.RIGHT
    
    if direction == DOWN:
        self.search_target = (0, map_dim-1)
        if self.edges is not None and self.__border_between_target_and_curr(
            self.edges[1], self.search_target[1], self.curr_pos[1]
        ):
            self.search_target[1] = self.edges[1]
        return Move.DOWN

def inward_spiral(self):
    """
    Perform clockwise inward spiral from southeast corner.
    """
    # Set inward spiral phase and layer, and update search target
    if self.curr_pos == self.search_target:
        self.adjust_phase_and_target()
    
    return self.__find_best_move_towards_search_target()

def adjust_phase_and_target(self):
    if self.phase + 1 == 5:
        self.layer += 1
    self.phase = (self.phase + 1) % 5
    
    # Set search target so that radius touches southeast corner of previous layer
    if self.phase == 0:
        offset = int(np.floor(self.radius / np.sqrt(2)))
        offset = 2 * offset if self.layer > 0 else offset
        self.search_target = (self.curr_pos[0] - offset, self.curr_pos[1] - offset)    

    # Set search target so that radius touches southwest corner of previous layer
    elif self.phase == 1:
        offset = map_dim - (int(np.floor(self.radius / np.sqrt(2))) * 2 +
                                        4 * self.radius * self.layer)
        self.search_target[0] = self.curr_pos[0] - offset

    # Set search target so that radius touches northwest corner of previous layer
    elif self.phase == 2:
        offset = map_dim - (int(np.floor(self.radius / np.sqrt(2))) * 2 +
                                        4 * self.radius * self.layer)
        self.search_target[1] = self.curr_pos[1] - offset

    # Set search target so that radius touches northeast corner of previous layer
    elif self.phase == 3:
        offset = map_dim - (int(np.floor(self.radius / np.sqrt(2))) * 2 +
                                        4 * self.radius * self.layer)
        self.search_target[0] = self.curr_pos[0] + offset

    # Set search target so that radius touches southeast corner of previous layer
    elif self.phase == 4:
        offset = map_dim - (int(np.floor(self.radius / np.sqrt(2))) * 2 +
                                        4 * self.radius * self.layer)
        self.search_target[1] = self.curr_pos[1] + offset

def __find_best_move_towards_search_target(self) -> Move:
    """
    Given an unexplored search target at self.search_target,
    Finds the best move towards that target
    Considers borders and selects a nice pythagorean route instead of straight lines
    (even though it doesn't really matter)
    """
    horizontal_dist = self.search_target[0] - self.curr_pos[0]
    vertical_dist = self.search_target[1] - self.curr_pos[1]

    # prioritize the axis with a greater distance
    # for no real reason other than it looks nicer
    # (and might be better later on)
    if abs(horizontal_dist) >= abs(vertical_dist):
        # search target is to the right
        if horizontal_dist >= 0:
            # check whether the border is between the target and current,
            # if so, go the other direction
            if self.edges[0] is not None and self.__border_between_target_and_curr(
                self.edges[0], self.search_target[0], self.curr_pos[0]
            ):
                return Move.LEFT
            return Move.RIGHT
        else:
            if self.edges[0] is not None and self.__border_between_target_and_curr(
                self.edges[0], self.search_target[0], self.curr_pos[0]
            ):
                return Move.RIGHT
            return Move.LEFT
    else:
        # search target is to the south
        if vertical_dist >= 0:
            if self.edges[1] is not None and self.__border_between_target_and_curr(
                self.edges[1], self.search_target[1], self.curr_pos[1]
            ):
                return Move.UP
            return Move.DOWN
        else:
            if self.edges[1] is not None and self.__border_between_target_and_curr(
                self.edges[1], self.search_target[1], self.curr_pos[1]
            ):
                return Move.DOWN
            return Move.UP

def __border_between_target_and_curr(
    self, border: int, target: int, curr: int
) -> bool:
    """
    Takes a target coordinate, border coordinate and current coordinate.
    Returns True if the border is between the target and current

    Let's consider an example using the x-axis:
    you're at [25, x] and want to get to [45, x], but the horizontal border is at 35. Then your global coordinates (in x-axis)
    are [89, x], and the target is really at [9, x]. You must go the opposite way to approach the target, LEFT from [25, x] to [-55, x]
    """

    # check whether the border is between the target and current
    return sorted([target, border, curr])[1] == border