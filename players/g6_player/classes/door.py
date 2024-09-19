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
                frequencies.append(self.turns_open[j] - self.turns_open[i])
        self.freq = gcd(*frequencies)

    def update_weight(self):
        """
        To be used for A-star or Dijkstra's algorithm
        """
        pass


# class Door:
#     def __init__(self) -> None:
#         # store when the door was last open
#         # We can think of this as a self.is_currently_open but
#         # removes the need to set that variable to false for all closed doors
#         self.last_open: int = -1

#         # record all ticks where the door was seen open
#         # for memory's sake, maybe we should only store those closest together
#         self.ticks_open: list[int] = []

#         # frequency starts off as 0, always closed
#         # but everytime its seen open, the frequency
#         # is updated to a newest worst case estimate
#         #
#         # 0 = always closed
#         # 4 = open every four ticks
#         self.freq = 0

#     def tick(self, tick: int):
#         """Updates the door information -
#         Called if the door is seen while open"""

#         self.last_open = tick

#         # TODO: once we have found the true frequency,
#         # return early

#         self.ticks_open.append(tick)

#         if len(self.ticks_open) == 1:
#             self.freq = self.ticks_open[0]
#         else:
#             # spread the array into a sequence before calling gcd
#             # also, maybe we get better results by also comparing the smallest distance between two ticks
#             # I'm not clear on the math on that, but feel like gcd should be sufficient
#             divisor = gcd(*self.ticks_open)
#             self.freq = divisor
