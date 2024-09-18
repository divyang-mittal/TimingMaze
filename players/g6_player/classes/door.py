from math import gcd


class Door:
    def __init__(self) -> None:
        # store when the door was last open
        # We can think of this as a self.is_currently_open but
        # removes the need to set that variable to false for all closed doors
        self.last_open: int = -1

        # record all ticks where the door was seen open
        # for memory's sake, maybe we should only store those closest together
        self.ticks_open: list[int] = []

        # frequency starts off as 0, always closed
        # but everytime its seen open, the frequency
        # is updated to a newest worst case estimate
        #
        # 0 = always closed
        # 4 = open every four ticks
        self.freq = 0

    def tick(self, tick: int):
        """Updates the door information -
        Called if the door is seen while open"""

        self.last_open = tick

        # TODO: once we have found the true frequency,
        # return early

        self.ticks_open.append(tick)

        if len(self.ticks_open) == 1:
            self.freq = self.ticks_open[0]
        else:
            # spread the array into a sequence before calling gcd
            # also, maybe we get better results by also comparing the smallest distance between two ticks
            # I'm not clear on the math on that, but feel like gcd should be sufficient
            divisor = gcd(*self.ticks_open)
            self.freq = divisor
