from constants import OPEN, CLOSED
from math import gcd
from players.g6_player.data import Move, move_to_str


class Door:
    def __init__(self, door_type: Move) -> None:
        self.door_type: Move = door_type
        self.state: int = CLOSED
        self.freq: int = 0
        self.last_open = 0

    def update(self, state: int, turn: int):
        """
        Updates the door's state and refines the frequency estimate on each turn.

        This method is called every turn to update the door's state and improve
        the frequency estimate. It uses the greatest common divisor (GCD) of the
        current frequency and the new turn number when the door is open.

        Parameters:
        state (int): The current state of the door (OPEN or CLOSED).
        turn (int): The current turn number.

        Example:
        1. Door opens at turn 40: freq = 40
        2. Door opens at turn 80: freq = gcd(40, 80) = 40
        3. Door opens at turn 85: freq = gcd(40, 85) = 5

        The frequency converges to the most accurate value available to the agent.

        Note: If the door is closed, only the state and turn are updated.
        """
        self.state = state
        if state == OPEN:
            if self.freq != 0:
                self.freq = gcd(self.freq, turn)
            else:
                self.freq = turn
            self.last_open = turn

    def __str__(self) -> str:
        return f"Door({move_to_str(self.door_type)}, freq: {self.freq})"

    def __repr__(self) -> str:
        return str(self)
