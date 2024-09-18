from players.g6_player.data import Move
from typing import Optional


class Cell:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        # NORTH, EAST, SOUTH, WEST
        self.doors = [0, 0, 0, 0]
        self.north = None
        self.east = None
        self.south = None
        self.west = None

    def save_neighbours(
        self,
        north: Optional["Cell"],
        east: Optional["Cell"],
        south: Optional["Cell"],
        west: Optional["Cell"],
    ):
        self.north = north
        self.east = east
        self.south = south
        self.west = west

    def is_path_available(self, move: Move) -> bool:
        match move:
            case Move.UP:
                return (
                    self.north is not None
                    and self.north.doors[2] == 1
                    and self.doors[0] == 1
                )
        return True
