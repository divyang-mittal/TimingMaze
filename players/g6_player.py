import numpy as np
class Player:
    def __init__(self, rng: np.random.Generator,  maximum_door_frequency: int, radius: int) -> None:
        """Initialise the player with the basic amoeba information
            Args:
                maximum_door_frequency (int): the maximum frequency of doors
                radius (int): the radius of the drone
        """

        self.rng = rng
        self.maximum_door_frequency = maximum_door_frequency
        self.radius = radius
        self.known_target = FALSE

    def build_maze_map(self, curr_maze_map) -> dict[str, int]:

        return {}


    def turn(self) -> int:

        curr_maze_map = self.build_maze_map(curr_maze_map)

        if not self.known_target:
            # SEARCH FOR TARGET
            return

        else:
            # GO TO TARGET
            return

        return

