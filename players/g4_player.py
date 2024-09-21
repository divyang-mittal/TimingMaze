from collections import defaultdict
import heapq
import random
import time
from constants import WAIT, LEFT, UP, RIGHT, DOWN
import os
import pickle
import numpy as np
import logging
import math

import constants
from timing_maze_state import TimingMazeState
from players.g4.gridworld import GridWorld
from players.g4.mcts import MCTS

from sympy import divisors


class Player:
    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        precomp_dir: str,
        maximum_door_frequency: int,
        radius: int,
    ) -> None:
        """Initialize the player with the basic amoeba information

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
        self.frequencies_per_cell = defaultdict(
            lambda: set(range(maximum_door_frequency + 1))
        )
        self.lcm_cache = {}
        self.curr_turn = 0
        self.start = (0, 0)
        self.goal = None
        self.maze_graph = defaultdict(dict)

    def update_door_frequencies(self, curr_x, curr_y, current_percept):
        factors = set(divisors(self.curr_turn))
        for dX, dY, door, state in current_percept.maze_state:
            # update frequency dictionary
            if state == constants.CLOSED:
                self.frequencies_per_cell[(curr_x + dX, curr_y + dY, door)] -= factors
            elif state == constants.OPEN:
                self.frequencies_per_cell[(curr_x + dX, curr_y + dY, door)] &= factors
            elif (
                curr_x + dX,
                curr_y + dY,
                door,
            ) not in self.frequencies_per_cell.keys() and state == constants.BOUNDARY:
                self.frequencies_per_cell[(curr_x + dX, curr_y + dY, door)] = {0}

    def update_graph(self, curr_x, curr_y, current_percept):
        # Update maze graph with new information
        for dX, dY, door, state in current_percept.maze_state:
            cell_pos = (curr_x + dX, curr_y + dY)
            neighbor_pos = None

            # Determine the neighbor position based on the door direction
            if door == constants.LEFT:
                neighbor_pos = (cell_pos[0] - 1, cell_pos[1])
            elif door == constants.RIGHT:
                neighbor_pos = (cell_pos[0] + 1, cell_pos[1])
            elif door == constants.UP:
                neighbor_pos = (cell_pos[0], cell_pos[1] - 1)
            elif door == constants.DOWN:
                neighbor_pos = (cell_pos[0], cell_pos[1] + 1)
            else:
                continue  # Should not happen

            # Get the frequencies of the doors
            cell_door_freqs = self.frequencies_per_cell[
                (cell_pos[0], cell_pos[1], door)
            ]
            neighbor_door = self.opposite_door(door)
            neighbor_door_freqs = self.frequencies_per_cell[
                (neighbor_pos[0], neighbor_pos[1], neighbor_door)
            ]

            # Determine if the edge is passable
            if cell_door_freqs == {0} or neighbor_door_freqs == {0}:
                # Door is always closed or frequencies are unknown; remove edge if it exists
                self.maze_graph[cell_pos].pop(neighbor_pos, None)
                self.maze_graph[neighbor_pos].pop(cell_pos, None)
            else:
                # Compute expected cost using avg_time_for_both_doors_to_open
                expected_cost = self.avg_time_for_both_doors_to_open(
                    cell_door_freqs, neighbor_door_freqs
                )
                # Add edge to the graph with the expected cost as the weight
                self.maze_graph[cell_pos][neighbor_pos] = expected_cost
                self.maze_graph[neighbor_pos][cell_pos] = expected_cost

    def lcm(self, a, b):
        # Return 0 if one of the values is 0 (door never opens)
        if a == 0 or b == 0:
            return 0

        # Check both (a, b) and (b, a) in cache to avoid unnecessary sorting
        if (a, b) in self.lcm_cache:
            return self.lcm_cache[(a, b)]
        elif (b, a) in self.lcm_cache:
            return self.lcm_cache[(b, a)]

        # Compute the LCM and store it in the cache
        result = abs(a * b) // math.gcd(a, b)
        self.lcm_cache[(a, b)] = result

        return result

    """Function which returns an approximation of the number of turns from the current turn needed to
        wait before adjacent doors are open at the same time"""

    def avg_time_for_both_doors_to_open(self, door1_freq_set, door2_freq_set):
        total_sum = 0
        count = 0

        # Loop through all possible frequency pairs from set1 and set2
        for f1 in door1_freq_set:
            for f2 in door2_freq_set:
                if f1 != 0 and f2 != 0:  # Skip cases where one of the doors never opens
                    next_open_time = self.lcm(f1, f2)

                    # If we don't care about current cycle, just avg expected moves to wait
                    if next_open_time != 0:
                        total_sum += next_open_time
                        count += 1

                    # if next_open_time != 0:
                    #     # Calculate how far ahead from current turn both doors will be open
                    #     next_open_turn = (-self.curr_turn % next_open_time)
                    #     total_sum += next_open_turn
                    #     count += 1

        # Compute and return the average of all next open times
        return total_sum / count if count > 0 else float("inf")

    def opposite_door(self, door):
        if door == constants.LEFT:
            return constants.RIGHT
        elif door == constants.RIGHT:
            return constants.LEFT
        elif door == constants.UP:
            return constants.DOWN
        elif door == constants.DOWN:
            return constants.UP
        else:
            return None

    def heuristic(self, current, goal):
        dx = abs(current[0] - goal[0])
        dy = abs(current[1] - goal[1])
        avg_cost_per_move = 1
        return (dx + dy) * avg_cost_per_move

    def a_star_search(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (self.heuristic(start, goal), start))

        came_from = {start: None}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = []
                current_node = current
                while current_node is not None:
                    path.append(current_node)
                    current_node = came_from[current_node]
                # Don't need to deal with entire path if maximizing efficiency, but useful for debugging
                path.reverse()
                return path

            for neighbor in self.maze_graph[current]:
                tentative_g_score = (
                    g_score[current] + self.maze_graph[current][neighbor]
                )
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        # Goal not reachable
        return None

    def move(self, current_percept) -> int:
        """Function which retrieves the current state of the maze and returns a movement action.

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
        curr_x, curr_y = -current_percept.start_x, -current_percept.start_y
        self.curr_turn += 1

        # Update door frequencies based on current percept
        self.update_door_frequencies(curr_x, curr_y, current_percept)

        # Update graph based on current percept
        self.update_graph(curr_x, curr_y, current_percept)

        # Update goal if end is visible
        if current_percept.is_end_visible:
            self.goal = (curr_x + current_percept.end_x, curr_y + current_percept.end_y)

        if self.goal:
            start = (curr_x, curr_y)
            goal = self.goal

            # Recompute path every turn
            path = self.a_star_search(start, goal)
            if path and len(path) > 1:
                # Next position to move to
                next_pos = path[1]
                # Decide which direction to move
                dx, dy = next_pos[0] - curr_x, next_pos[1] - curr_y
                if dx == -1:
                    move = constants.LEFT
                elif dx == 1:
                    move = constants.RIGHT
                elif dy == -1:
                    move = constants.UP
                elif dy == 1:
                    move = constants.DOWN
                else:
                    move = constants.WAIT
                return move
            else:
                # No valid path found, or already at goal
                return constants.WAIT
        else:
            # Goal is not known; implement exploration strategy or wait
            return constants.WAIT
