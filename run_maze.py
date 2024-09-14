from timing_maze_game import TimingMazeGame


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":
    args = Namespace(
        **{
            "max_door_frequency": 5,
            "radius": 15,
            "seed": 2,
            "maze": None,
            "scale": 9,
            "no_gui": True,
            "log_path": "log",
            "disable_logging": False,
            "disable_timeout": True,
            "player": "1",
        }
    )

    root = tk.Tk()
    app = TimingMazeGame(args, root)
