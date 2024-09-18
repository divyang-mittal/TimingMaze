from players.g6_player.data import Move
from players.g6_player.classes.door import Door
from typing import Optional


class Cell:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        # NORTH, EAST, SOUTH, WEST
        self.n_door = Door()
        self.e_door = Door()
        self.s_door = Door()
        self.w_door = Door()
        self.n_neighbour = None
        self.e_neighbour = None
        self.s_neighbour = None
        self.w_neighbour = None

    def tick(self, some_info):
        """Updates the cell information -
        Called every tick when the cell is seen"""
        # TODO: only tick door if its open
        self.n_door.tick(some_info)
        self.s_door.tick(some_info)
        self.e_door.tick(some_info)
        self.w_door.tick(some_info)
        pass

    def save_neighbours(
        self,
        north: Optional["Cell"],
        east: Optional["Cell"],
        south: Optional["Cell"],
        west: Optional["Cell"],
    ):
        """Called during initialization by the Maze -
        Can't be apart of __init()__ as all other neighbour Cells must be defined before this step

        Neighbours are optional because edge and corner cells don't have all four neighbours
        """
        self.n_neighbour = north
        self.e_neighbour = east
        self.s_neighbour = south
        self.w_neighbour = west

    def is_path_available(self, tick: int, move: Move) -> bool:
        match move:
            case Move.UP:
                return (
                    self.n_neighbour is not None
                    and self.n_neighbour.s_door.last_open
                    == self.n_door.last_open
                    == tick
                )
        return True
