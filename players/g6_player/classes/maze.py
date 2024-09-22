from players.g6_player.classes.cell import Cell
from players.g6_player.classes.maze_graph import MazeGraph

from constants import UP, DOWN, RIGHT, LEFT, map_dim
from players.g6_player.classes.typed_timing_maze_state import TypedTimingMazeState

# 199 usually
GRID_DIM = map_dim * 2 - 1
# 99 usually
CENTER_POS = map_dim - 1


class Maze:
    def __init__(self) -> None:
        self.grid = [[Cell(x=x, y=y) for y in range(GRID_DIM)] for x in range(GRID_DIM)]
        self.graph = MazeGraph()
        self.turn = 0
        self.target_pos = None
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

    def current_cell(self) -> Cell:
        """
        Return current cell of player
        """
        x, y = self.curr_pos
        return self.grid[x][y]

    def get_cell(self, x: int, y: int) -> Cell:
        return self.grid[x][y]

    def update(self, current_percept: TypedTimingMazeState):
        """
        Update current maze with info from the drone
        """
        self.turn += 1
        self.curr_pos = (
            CENTER_POS - current_percept.start_x,
            CENTER_POS - current_percept.start_y,
        )

        if current_percept.is_end_visible:
            self.target_pos = (
                CENTER_POS - current_percept.end_x,
                CENTER_POS - current_percept.end_y,
            )

        self.__update_maze_door_freq(current_percept)
        self.__update_maze_path_freq(current_percept)
        self.__update_maze_graph(current_percept)

    def __update_maze_door_freq(self, current_percept: TypedTimingMazeState):
        """
        Given door status information from the drone udpate door frequencies
        """
        for cell in current_percept.maze_state:
            # Iterating over cells seen by drone
            x = self.curr_pos[0] + cell.row
            y = self.curr_pos[1] + cell.col
            if cell.door_type == UP:
                self.grid[x][y].n_door.update(cell.door_state, self.turn)
            elif cell.door_type == RIGHT:
                self.grid[x][y].e_door.update(cell.door_state, self.turn)
            elif cell.door_type == DOWN:
                self.grid[x][y].s_door.update(cell.door_state, self.turn)
            elif cell.door_type == LEFT:
                self.grid[x][y].w_door.update(cell.door_state, self.turn)

    def __update_maze_path_freq(self, current_percept: TypedTimingMazeState):
        """
        Using updated door frequencies calculate path frequencies
        """
        for cell in current_percept.maze_state:
            x = self.curr_pos[0] + cell.row
            y = self.curr_pos[1] + cell.col
            self.grid[x][y].update_paths()

    def __update_maze_graph(self, current_percept: TypedTimingMazeState):
        for cell in current_percept.maze_state:
            x = self.curr_pos[0] + cell.row
            y = self.curr_pos[1] + cell.col

            # Update edges that exist between cell and neighbors
            self.graph.add_edge((x, y), (x + 1, y), weight=self.grid[x][y].e_path)
            self.graph.add_edge((x, y), (x - 1, y), weight=self.grid[x][y].w_path)
            self.graph.add_edge((x, y), (x, y + 1), weight=self.grid[x][y].n_path)
            self.graph.add_edge((x, y), (x, y - 1), weight=self.grid[x][y].s_path)

            # self.graph.display_graph()

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
