import os

map_dim = 100

default_maze = os.path.join("maps", "default", "simple.json")

CLOSED_PROB = 0.05

possible_players = ["d"] + list(map(str, range(1, 10)))

# Directions for the maze
WAIT = -1
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Maze cell states
CLOSED = 1
OPEN = 2
BOUNDARY = 3

timeout = 12000

CELL_SIZE = 8

# two doors visible, drone radius