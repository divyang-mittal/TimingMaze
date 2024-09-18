import heapq
import logging
from os import system
import constants
from typing import List, Optional, Set, Tuple

from players.group5.player_map import PlayerMapInterface, SimplePlayerCentricMap, StartPosCentricPlayerMap
from players.group5.door import DoorIdentifier


def converge(current_pos : list, goal : list, turn : int, player_map: PlayerMapInterface) -> int:

	path = dyjkstra(current_pos, goal, turn, player_map)

	# print("Ending Dijkstra's algorithm...")
	# print("Valid Moves: ", player_map.get_valid_moves(turn))

	# print("Path: ", path)
	# print("Path[0]: ", path[0])

	return path[0]


def dyjkstra(current_pos : list, goal : list, turn : int, player_map: PlayerMapInterface) -> list:

	# Create a priority queue
	queue = []
	heapq.heappush(queue, (0, current_pos, turn))

	# Create a dictionary to store the cost of each position
	costs = {tuple(current_pos): 0}

	# Create a set to store visited positions
	visited = set()

	# Create a dictionary to store the path to each position
	paths = {tuple(current_pos): []}

	# Create a dictionary to store the turns for each position
	turns = {tuple(current_pos): turn}

	# While there are positions to explore
	while queue:
		# print("Starting while loop...")
		# Get the position with the lowest cost
		current_cost, current_pos, expected_turn  = heapq.heappop(queue)

		# print("Expected Turn: ", expected_turn)

		# If we have reached the goal, return the path
		if current_pos == goal:
			return paths[tuple(current_pos)]

		# If we have already visited this position, skip it
		if tuple(current_pos) in visited:
			continue

		# Mark the position as visited
		visited.add(tuple(current_pos))

		# Explore the neighbors
		for move in [constants.UP, constants.DOWN, constants.RIGHT, constants.LEFT]:

			if move == constants.LEFT:
				neighbor = [current_pos[0] - 1, current_pos[1]]
				door = DoorIdentifier([current_pos[0], current_pos[1]], constants.LEFT)
			elif move == constants.UP:
				neighbor = [current_pos[0], current_pos[1] - 1]
				door = DoorIdentifier([current_pos[0], current_pos[1]], constants.UP)
			elif move == constants.RIGHT:
				neighbor = [current_pos[0] + 1, current_pos[1]]
				door = DoorIdentifier([current_pos[0], current_pos[1]], constants.RIGHT)
			elif move == constants.DOWN:
				neighbor = [current_pos[0], current_pos[1] + 1]
				door = DoorIdentifier([current_pos[0], current_pos[1]], constants.DOWN)

			# Calculate the cost of the neighbor
			# TODO make a special function that calculates based on observations of wall intervals
			# weight, new_expected_turn = add_weight(current_pos, neighbor, player_map.get_wall_freq_candidates(door), expected_turn)

			# print("current_pos: ", current_pos)

			weight, new_expected_turn = calculate_weighted_average(expected_turn, player_map.get_wall_freq_candidates(door))
			if weight == 10000000000000000000001:
				# print("SKIPPED MOVE: ", move)
				continue  # Skip this move if the door is closed
			
			new_cost = current_cost + weight

			# print("Exploring Move: ", move)
			# print("Going from: ", current_pos)
			# print("To Neighbor: ", neighbor)
			# print("Current Cost: ", current_cost)
			# print("Weight: ", weight)
			# print("New Cost: ", new_cost)
			# print("New Expected Turn: ", new_expected_turn)

			# If the neighbor has not been visited or the new cost is lower, update the cost and add it to the queue
			if tuple(neighbor) not in visited and (tuple(neighbor) not in costs or new_cost < costs[tuple(neighbor)]):
				costs[tuple(neighbor)] = new_cost
				paths[tuple(neighbor)] = paths[tuple(current_pos)] + [move]
				heapq.heappush(queue, (new_cost, neighbor, new_expected_turn))

	# If we reach here, it means we could not find a path to the goal
	return None


# # NOTES:
# # - see if we can hold onto the calucalted values by algorithm and modify them slightly with each turn as we learn more
# # - the visited set should be modified to allow for backtracking if we find a better path
# # maybe for each set of touching doors, make a dictionary that stores the door state and the cost of passing through it on any given turn
# def compute_unobserved_door_weight() -> int:
	# calculates the likelihood of doors being open and average wait expected
	return 0

def calculate_weighted_average(current_turn, candidates):
    """
    Calculate a weighted average cost for traversing a door based on the current turn
    and the candidate turns when the door might open. Also return the expected turn.

    Parameters:
    - current_turn (int): The current turn.
    - candidates (list): A list of candidate turns when the door might open.

    Returns:
    - average_weight (float): The weighted average cost for passing through the door.
    - expected_turn (int): The next expected turn when the door will open.
    """

    print("Candidates: ", candidates)

    if all(candidate == 0 for candidate in candidates):
        print("All candidates are closed.")
		# TODO fix this to infinity (or change it to None)
        return 10000000000000000000001, current_turn + 10000000000000000000001  # No candidates means the door is closed indefinitely
	

    weights = []
    total_weight = 0
    weighted_sum = 0
    next_expected_turn = None

    for candidate in candidates:
        if candidate == 0:
            # Treat as always closed
            return float('inf'), float('inf')

        distance = current_turn % candidate
        if distance == 0:
            weight = 1  # Door is open now
        else:
            weight = 1 / (distance + 1)

        weights.append((weight, candidate))
        total_weight += weight
        weighted_sum += weight * candidate

	# Calculate the weighted average
    average_weight = weighted_sum / total_weight if total_weight else float('inf')

    next_expected_turn = round(average_weight) + current_turn

    return average_weight, next_expected_turn