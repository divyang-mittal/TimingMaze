from players.g6_player.data import Move
from players.g6_player.classes.door import Door
from typing import Optional
from constants import *


class Cell:
    def __init__(self, x: int, y: int) -> None:
        # coordinates in 199x199 grid
        self.x = x
        self.y = y

        # doors of current cell
        self.n_door = Door(UP)
        self.e_door = Door(RIGHT)
        self.s_door = Door(DOWN)
        self.w_door = Door(LEFT)
        
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
        [TODO] To be used for A-star or Dijkstra's algorithm
        """
        pass

