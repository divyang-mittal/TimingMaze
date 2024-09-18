from players.g6_player.classes.cell import Cell


class Maze:
    def __init__(self) -> None:
        self.cells = [[Cell(x=x, y=y) for y in range(0, 100)] for x in range(0, 100)]

    def get_cell(self, x: int, _y: int) -> Cell:
        pass
