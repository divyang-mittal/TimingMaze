from enum import Enum


# should maybe be called Direction
class Move(Enum):
    WAIT = -1
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    def __lt__(self, other) -> bool:
        return True


def move_to_str(move: Move) -> str:
    match move:
        case Move.UP:
            return "UP"
        case Move.DOWN:
            return "DOWN"
        case Move.RIGHT:
            return "RIGHT"
        case Move.LEFT:
            return "LEFT"
