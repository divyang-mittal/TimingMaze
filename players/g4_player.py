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

from collections import deque


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

        # Calculate the grid cell size based on the radius
        self.cell_size = max(int(self.radius / math.sqrt(2)), 1)

        # Initialize variables for the exploration strategy
        self.visited_grid_cells = set()
        self.frontier_positions = set()
        self.frontier_set = set()

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

    def avg_time_for_both_doors_to_open(self, door1_freq_set, door2_freq_set):
        """Function which returns an approximation of the number of turns from the current turn needed to
        wait before adjacent doors are open at the same time"""

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

    def a_star_search(self, start, goals):
        open_set = []
        heapq.heappush(open_set, (self.heuristic(start, start), 0, start))
        came_from = {start: None}
        g_score = {start: 0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current in goals:
                # Reconstruct path
                path = []
                current_node = current
                while current_node is not None:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.reverse()
                # print(path)
                return path

            for neighbor in self.maze_graph[current]:
                tentative_g_score = (
                    g_score[current] + self.maze_graph[current][neighbor]
                )
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(start, neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        # No path found
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
            # Use A* search to the goal
            return self.perform_a_star_and_get_next_move((curr_x, curr_y), {self.goal})
        else:
            # Exploration strategy
            self.update_visited_and_frontier(curr_x, curr_y)

            if self.frontier_positions:
                # Use A* search to any of the frontier positions
                return self.perform_a_star_and_get_next_move(
                    (curr_x, curr_y), self.frontier_positions
                )

            return constants.WAIT

    def get_grid_cell(self, x, y):
        grid_x = x // self.cell_size
        grid_y = y // self.cell_size
        return (grid_x, grid_y)

    def get_positions_in_grid_cell(self, grid_cell):
        grid_x, grid_y = grid_cell

        # Define the boundaries of the grid cell
        min_x = grid_x * self.cell_size
        max_x = (grid_x + 1) * self.cell_size - 1
        min_y = grid_y * self.cell_size
        max_y = (grid_y + 1) * self.cell_size - 1

        # Collect known positions within the grid cell
        known_positions = set(self.maze_graph.keys())
        positions = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) in known_positions:
                    positions.append((x, y))
        return positions

    def get_unvisited_neighbors(self, grid_cell):
        x, y = grid_cell
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if (
                neighbor not in self.visited_grid_cells
                and neighbor not in self.frontier_set
            ):
                neighbors.append(neighbor)
                self.frontier_set.add(neighbor)
        return neighbors

    def determine_move(self, dx, dy):
        if dx == -1:
            return constants.LEFT
        elif dx == 1:
            return constants.RIGHT
        elif dy == -1:
            return constants.UP
        elif dy == 1:
            return constants.DOWN
        else:
            return constants.WAIT

    def perform_a_star_and_get_next_move(self, start, goals):
        path = self.a_star_search(start, goals)
        if path and len(path) > 1:
            next_pos = path[1]
            dx, dy = next_pos[0] - start[0], next_pos[1] - start[1]
            return self.determine_move(dx, dy)
        return constants.WAIT

    def update_visited_and_frontier(self, curr_x, curr_y):
        current_grid_cell = self.get_grid_cell(curr_x, curr_y)

        if current_grid_cell not in self.visited_grid_cells:
            self.visited_grid_cells.add(current_grid_cell)

            # Remove positions in this grid cell from frontier_positions
            positions_in_current_grid_cell = set(
                self.get_positions_in_grid_cell(current_grid_cell)
            )
            self.frontier_positions -= positions_in_current_grid_cell

            # Expand the frontier
            neighbors = self.get_unvisited_neighbors(current_grid_cell)
            for neighbor in neighbors:
                self.frontier_set.add(neighbor)
                # Add positions in neighbor grid cell to frontier_positions
                positions_in_neighbor_grid_cell = set(
                    self.get_positions_in_grid_cell(neighbor)
                )
                self.frontier_positions |= positions_in_neighbor_grid_cell
