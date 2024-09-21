from constants import OPEN, CLOSED
from math import gcd


class Door:
    def __init__(self, door_type: int) -> None:
        self.door_type: int = door_type
        self.state: int = CLOSED
        self.turn: int = 0
        self.freq: int = 0
        self.last_open = 0

    def update_turn(self, state: int, turn: int):
        """
        Called on every turn
        """
        self.state = state
        self.turn = turn
        if state == OPEN:
            if self.freq != 0:
                self.freq = gcd(self.freq, turn)
            else:
                self.freq = turn
            self.last_open = turn
