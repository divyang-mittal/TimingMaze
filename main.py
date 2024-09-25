import argparse
from timing_maze_game import TimingMazeGame
import tkinter as tk

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_door_frequency", "-m", type=int, default=5,
                        help="Value between 1 and 100 (including 1)")
    parser.add_argument("--radius", "-r", type=int, default=15,
                        help="radius of the circle visible by the drone ""(min=1, max=150")
    parser.add_argument("--seed", "-s", type=int, default=4, help="Seed used by random number generator")
    parser.add_argument(
        "--maze", "-mz", help="Use the given map, if no map is given, Generate a maze using the seed provided"
    )
    parser.add_argument("--scale", "-sc", default=9, help="Scale")
    parser.add_argument("--no_gui", "-ng", action="store_true", help="Disable GUI")
    parser.add_argument("--log_path", default="log", help="Directory path to dump log files, filepath if "
                                                          "disable_logging is false")
    parser.add_argument("--disable_logging", action="store_true", help="Disable Logging, log_path becomes path to file")
    parser.add_argument("--disable_timeout", action="store_true", help="Disable timeouts for player code")
    parser.add_argument("--player", "-p", default="4", help="Specifying player")
    args = parser.parse_args()

    if args.disable_logging:
        if args.log_path == "log":
            args.log_path = "results.log"

    root = tk.Tk()
    app = TimingMazeGame(args, root)

