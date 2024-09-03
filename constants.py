import os

map_dim = 100
total_cells = 10000

vis_width = 960
vis_height = 720
default_maze = os.path.join("maps", "default", "simple.json")

CLOSED_PROB = 0.02

possible_players = ["d"] + list(map(str, range(1, 10)))

# Directions for the maze
WAIT = -1
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

INVISIBLE = 0
CLOSED = 1
OPEN = 2
BOUNDARY = 3

timeout = 60 * 10
