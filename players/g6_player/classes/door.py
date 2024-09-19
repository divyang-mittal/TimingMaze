from constants import *
from math import gcd


class Door:
    def __init__(self, door_type: int) -> None:
        self.door_type: int = door_type
        self.state: int = CLOSED
        self.turn: int = 0
        self.freq: int = 0
        self.turns_open: list[int] = []
        self.weight: float = self.update_weight()

    def update_turn(self, state: int, turn: int):
        """
        Called on every turn
        """
        self.state = state
        self.turn = turn
        if state == OPEN:
            self.turns_open.append(turn)
            if len(self.turns_open) > 1:
                self.freq = self.update_freq()

    def update_freq(self):
        """
        Update frequency of door based on previous turns when open door was detected
        """
        frequencies = []
        for i in range(len(self.turns_open) - 1):
            for j in range(i, len(self.turns_open)):
                frequencies.append(abs(self.turns_open[j] - self.turns_open[i]))
        self.freq = gcd(*frequencies)

    def update_weight(self):
        """
        [TODO] To be used for A-star or Dijkstra's algorithm
        """
        pass

