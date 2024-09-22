from players.g6_player.classes.cell import Cell

from constants import UP, DOWN, RIGHT, LEFT, map_dim
from players.g6_player.classes.typed_timing_maze_state import TypedTimingMazeState

# 199 usually
GRID_DIM = map_dim * 2 - 1
# 99 usually
CENTER_POS = map_dim - 1


class Maze:
    def __init__(self, max_door_freq: int, radius: int, turn: int) -> None:
        self.grid = [[Cell(x=x, y=y) for y in range(GRID_DIM)] for x in range(GRID_DIM)]
        self.max_door_freq = max_door_freq
        self.radius = radius
        self.turn = turn
        self.curr_pos = (CENTER_POS, CENTER_POS)  # relative to 199x199 grid
        self.north_end = 0
        self.east_end = GRID_DIM - 1
        self.south_end = GRID_DIM - 1
        self.west_end = 0
        self.__init_edges()

    def __init_edges(self):
        for i in range(GRID_DIM):
            for j in range(GRID_DIM):
                self.grid[i][j].save_neighbors(
                    self.grid[i][j - 1] if j > 0 else None,
                    self.grid[i + 1][j] if i < GRID_DIM - 1 else None,
                    self.grid[i][j + 1] if j < GRID_DIM - 1 else None,
                    self.grid[i - 1][j] if i > 0 else None,
                )

    def get_cell(self, x: int, y: int) -> Cell:
        return self.grid[x][y]

    def update(self, current_percept: TypedTimingMazeState):
        """
        Update current maze with info from the drone
        """
        self.curr_pos = (
            CENTER_POS - current_percept.start_x,
            CENTER_POS - current_percept.start_y,
        )

        self.__update_maze_door_freq(current_percept)
        self.__update_maze_path_freq(current_percept)

    def __update_maze_door_freq(self, current_percept: TypedTimingMazeState):
        """
        Given door status information from the drone udpate door frequencies
        """
        for cell in current_percept.maze_state:
            # Iterating over cells seen by drone
            # cell[0]=x, cell[1]=y, cell[2]=door type, cell[3]=door state
            x = self.curr_pos[0] + cell[0]
            y = self.curr_pos[1] + cell[1]
            if cell[2] == UP:
                self.grid[x][y].n_door.update(cell[3], self.turn)
            elif cell[2] == RIGHT:
                self.grid[x][y].e_door.update(cell[3], self.turn)
            elif cell[2] == DOWN:
                self.grid[x][y].s_door.update(cell[3], self.turn)
            elif cell[2] == LEFT:
                self.grid[x][y].w_door.update(cell[3], self.turn)
        return

    def __update_maze_path_freq(self, current_percept: TypedTimingMazeState):
        """
        Using updated door frequencies calculate path frequencies
        """
        for cell in current_percept.maze_state:
            # cell[0]=x, cell[1]=y, cell[2]=door type, cell[3]=door state
            x = self.curr_pos[0] + cell[0]
            y = self.curr_pos[1] + cell[1]
            self.grid[x][y].update_paths()
        return

    def update_boundary(self, curr_cell: Cell, direction: int):
        """
        Given a border cell, update a boundary and its opposite boundary
        """
        if direction == RIGHT:
            self.east_end = curr_cell.x
            self.west_end = curr_cell.x - map_dim + 1
        elif direction == DOWN:
            self.south_end = curr_cell.y
            self.north_end = curr_cell.y - map_dim + 1
        elif direction == LEFT:
            self.west_end = curr_cell.x
            self.east_end = curr_cell.x + map_dim - 1
        elif direction == UP:
            self.north_end = curr_cell.y
            self.south_end = curr_cell.y + map_dim - 1

    def __str__(self) -> str:
        return f"Maze(turn: {self.turn})"

    def __repr__(self) -> str:
        return str(self)
