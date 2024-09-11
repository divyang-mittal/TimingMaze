import numpy as np
class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger,
                 precomp_dir: str, maximum_door_frequency: int, radius: int) -> None:
        """Initialise the player with the basic amoeba information

            Args:
                rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
                logger (logging.Logger): logger use this like logger.info("message")
                maximum_door_frequency (int): the maximum frequency of doors
                radius (int): the radius of the drone
                precomp_dir (str): Directory path to store/load pre-computation
        """

        self.rng = rng
        self.logger = logger
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

