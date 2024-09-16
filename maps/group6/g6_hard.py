from random import sample
import json
import numpy as np

L = 11
TEXTURE_FREQ = [1, 2]

MAZE_DIMENSION = 100
# Took this from the TA's maze definition so each cell has [left, up, right, down]
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


EMPTY_MAZE = {
    "frequencies": np.ones((MAZE_DIMENSION, MAZE_DIMENSION, 4), dtype=int),
    "start_pos": None,
    "end_pos": None,
}

for i in range(MAZE_DIMENSION):
    EMPTY_MAZE["frequencies"][0][i][LEFT] = 0
    EMPTY_MAZE["frequencies"][MAZE_DIMENSION - 1][i][RIGHT] = 0
    EMPTY_MAZE["frequencies"][i][0][UP] = 0
    EMPTY_MAZE["frequencies"][i][MAZE_DIMENSION - 1][DOWN] = 0


# Set a texture in the full maze of randomly sampled values of 1 or 2
def create_texture_maze(maze, texture_freq):
    for r in range(MAZE_DIMENSION):
        for c in range(MAZE_DIMENSION):
            for d in range(4):
                if maze[r][c][d] == 1:
                    maze[r][c][d] = sample(texture_freq, 1)[0]

    return maze


# Adding ridges to maze
def build_ridge(maze, ridge_x_start, ridge_x_end, ridge_y_start, ridge_y_end):
    range_x = np.array(range(ridge_x_start, ridge_x_end))
    ridge_x_path = range_x[int(len(range_x) / 2)]
    range_y = np.array(range(ridge_y_start, ridge_y_end))

    for x in range_x:
        for y in range_y:
            # Add gradient edge on RIGHT / LEFT
            if x == ridge_x_path - 3:
                maze[x][y][LEFT] = L - 6
                maze[x][y][RIGHT] = L - 5
            elif x == ridge_x_path - 2:
                maze[x][y][LEFT] = L - 4
                maze[x][y][RIGHT] = L - 3
            elif x == ridge_x_path - 1:
                maze[x][y][LEFT] = L - 2
                maze[x][y][RIGHT] = L - 1
            elif x == ridge_x_path:
                maze[x][y][LEFT] = L
                maze[x][y][RIGHT] = L
            elif x == ridge_x_path + 1:
                maze[x][y][LEFT] = L - 1
                maze[x][y][RIGHT] = L - 2
            elif x == ridge_x_path + 2:
                maze[x][y][LEFT] = L - 3
                maze[x][y][RIGHT] = L - 4
            elif x == ridge_x_path + 3:
                maze[x][y][LEFT] = L - 5
                maze[x][y][RIGHT] = L - 6

            # Add ridge on BOTTOM
            if y == 0:
                maze[x][y][UP] = 0  # Should already be the case
                maze[x][y][DOWN] = 1
            # Closing off bottom of the ridge with a wall
            elif y == ridge_y_end - 1:
                maze[x][y][UP] = 1
                maze[x][y][DOWN] = 0
            else:
                maze[x][y][UP] = 1
                maze[x][y][DOWN] = 1

    return maze


# ridge_x_start = 10
# ridge_x_end = 17
# ridge_y_start = 0
# ridge_y_end = 96

textured_maze = create_texture_maze(EMPTY_MAZE["frequencies"], TEXTURE_FREQ)
maze_with_left_ridge = build_ridge(textured_maze, 10, 17, 0, 96)
maze_with_two_ridges = build_ridge(maze_with_left_ridge, 83, 90, 0, 96)

complete_maze = {
    "frequencies": maze_with_two_ridges.tolist(),
    "start_pos": [2, 2],
    "end_pos": [76, 2],
}

with open("g6_hard.json", "w", encoding="utf-8") as file:
    json.dump(complete_maze, file)
