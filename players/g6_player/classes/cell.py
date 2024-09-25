from players.g6_player.data import Move
from players.g6_player.classes.door import Door
from typing import Optional
from constants import OPEN
from math import lcm


class Cell:
    def __init__(self, x: int, y: int) -> None:
        # coordinates in 199x199 grid
        self.x = x
        self.y = y

        # whether the cell has been visited
        self.seen = False

        # doors of current cell
        self.n_door = Door(Move.UP)
        self.e_door = Door(Move.RIGHT)
        self.s_door = Door(Move.DOWN)
        self.w_door = Door(Move.LEFT)

        # neighbors
        self.n_cell = None
        self.e_cell = None
        self.s_cell = None
        self.w_cell = None

        # path frequencies to neighbors
        self.n_path = None
        self.e_path = None
        self.s_path = None
        self.w_path = None

    def save_neighbors(
        self,
        north: Optional["Cell"],
        east: Optional["Cell"],
        south: Optional["Cell"],
        west: Optional["Cell"],
    ):
        """Called during initialization by the Maze; can't be part of __init()__ as all
        other neighbor Cells must be defined before this step.

        Neighbors are optional because edge and corner cells don't have all four neighbors
        """
        self.n_cell = north
        self.e_cell = east
        self.s_cell = south
        self.w_cell = west

    def neighbours(self) -> list[tuple[int, "Cell"]]:
        """Gets all the cell's non-null neighbours"""
        neighbours = [
            c
            for c in [
                (self.n_path, self.n_cell, Move.UP),
                (self.w_path, self.w_cell, Move.LEFT),
                (self.e_path, self.e_cell, Move.RIGHT),
                (self.s_path, self.s_cell, Move.DOWN),
            ]
        ]

        # filter out None neighbours
        filtered: list[tuple[int, "Cell"]] = [
            (path, cell, move)
            for path, cell, move in neighbours
            if path is not None and cell is not None
        ]

        return filtered

    def is_move_available(self, move: Move) -> bool:
        match move:
            case Move.LEFT:
                return (
                    self.w_cell is not None
                    and self.w_cell.e_door.state == OPEN
                    and self.w_door.state == OPEN
                )
            case Move.UP:
                return (
                    self.n_cell is not None
                    and self.n_cell.s_door.state == OPEN
                    and self.n_door.state == OPEN
                )
            case Move.RIGHT:
                return (
                    self.e_cell is not None
                    and self.e_cell.w_door.state == OPEN
                    and self.e_door.state == OPEN
                )
            case Move.DOWN:
                return (
                    self.s_cell is not None
                    and self.s_cell.n_door.state == OPEN
                    and self.s_door.state == OPEN
                )
            case _:
                return False

    def update_paths(self):
        """
        Calculates the path frequency for each direction using the least common multiple
        """
        self.n_path = lcm(self.n_door.freq, self.n_cell.s_door.freq) if self.n_cell else 0
        self.s_path = lcm(self.s_door.freq, self.s_cell.n_door.freq) if self.s_cell else 0
        self.e_path = lcm(self.e_door.freq, self.e_cell.w_door.freq) if self.e_cell else 0
        self.w_path = lcm(self.w_door.freq, self.w_cell.e_door.freq) if self.w_cell else 0

    def __str__(self) -> str:
        return f"Cell({self.x}, {self.y})"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, value: "Cell") -> bool:
        return self.x == value.x and self.y == value.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __lt__(self, other):
        return True
