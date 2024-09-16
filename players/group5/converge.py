import heapq
import constants

def converge_basic(start, goal):


	print("Start: ", start)
	print("Goal: ", goal)

	# Move across the x axis until the x coordinate matches the goal
	while start[0] != goal[0]:
		if start[0] < goal[0]:
			return constants.RIGHT
		else:
			return constants.LEFT
		
	# Move across the y axis until the y coordinate matches the goal
	while start[1] != goal[1]:
		if start[1] < goal[1]:
			return constants.UP
		else:
			return constants.DOWN

def get_weight(pos, direction, turn_num):
	# given the frequency candidates and the current turn number, return the weight of the move
	pass


def converge(current_pos : list, goal : list):
	# print("Current Position: ", current_pos)
	# print("Goal Position: ", goal)
	path = dyjkstra(current_pos, goal)
	# print("Path: ", path)
	return path[0] if path else constants.WAIT


def dyjkstra(current_pos : list, goal : list) -> list:

	# Create a priority queue
	queue = []
	heapq.heappush(queue, (0, current_pos))


	# Create a dictionary to store the cost of each position
	costs = {tuple(current_pos): 0}

	# Create a set to store visited positions
	visited = set()

	# Create a dictionary to store the path to each position
	paths = {tuple(current_pos): []}

	# While there are positions to explore
	while queue:

		# Get the position with the lowest cost
		current_cost, current_pos  = heapq.heappop(queue)

		# If we have reached the goal, return the path
		if current_pos == goal:
			print(paths[tuple(current_pos)])
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
			elif move == constants.UP:
				neighbor = [current_pos[0], current_pos[1] - 1]
			elif move == constants.RIGHT:
				neighbor = [current_pos[0] + 1, current_pos[1]]
			elif move == constants.DOWN:
				neighbor = [current_pos[0], current_pos[1] + 1]

			# Calculate the cost of the neighbor
			# TODO make a special function that calculates based on observations of wall intervals
			new_cost = current_cost + 1

			# If the neighbor has not been visited or the new cost is lower, update the cost and add it to the queue
			if tuple(neighbor) not in visited and (tuple(neighbor) not in costs or new_cost < costs[tuple(neighbor)]):
				costs[tuple(neighbor)] = new_cost
				paths[tuple(neighbor)] = paths[tuple(current_pos)] + [move]
				heapq.heappush(queue, (new_cost, neighbor))

	# If we reach here, it means we could not find a path to the goal
	return None
