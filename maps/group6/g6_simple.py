import random
import json
import numpy as np

MAZE_DIMENSION = 100
L = 4

DISTRIBUTION = list(range(L + 1))
WEIGHTS = [1, 35, 35, 5, 24]

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

START = [12, 15]
END = [17, 31]


def initilize_maze(maze_dimension, cell_dimension):
    maze = {
        "frequencies": np.ones(
            (maze_dimension, maze_dimension, cell_dimension), dtype=int
        ),
        "start_pos": START,
        "end_pos": END,
    }

    # Build the edges of the maze
    for i in range(maze_dimension):
        maze["frequencies"][0][i][LEFT] = 0
        maze["frequencies"][maze_dimension - 1][i][RIGHT] = 0
        maze["frequencies"][i][0][UP] = 0
        maze["frequencies"][i][maze_dimension - 1][DOWN] = 0

    return maze


def structure_maze(maze):
    for row in range(MAZE_DIMENSION):
        for col in range(MAZE_DIMENSION):
            for dim in range(4):  # 4 directions
                if maze["frequencies"][row][col][dim] == 1:
                    maze["frequencies"][row][col][dim] = random.choices(
                        DISTRIBUTION, WEIGHTS
                    )[0]
    return maze


def save_maze(filename, maze):
    maze["frequencies"] = maze["frequencies"].tolist()
    with open(file=filename, mode="w", encoding="utf-8") as file:
        json.dump(maze, file)


initial_maze = initilize_maze(MAZE_DIMENSION, 4)
dynamic_maze = structure_maze(initial_maze)

save_maze("g6_simple.json", dynamic_maze)
