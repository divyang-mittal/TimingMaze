import argparse
from timing_maze_game import TimingMazeGame
import constants

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_door_frequency", "-m", type=float, default=1.0, help="Value between 0 and 1 (including 1) that "
                                                                            ""
                                                                            "")
    parser.add_argument("--radius", "-r", type=float, default=15, help="radius of the circle visible by the drone "
                                                                   "(min=0, max=150")
    parser.add_argument("--seed", "-s", type=int, default=2, help="Seed used by random number generator")
    parser.add_argument(
        "--maze", "-mz", help="Use the given map, if no map is given, Generate a maze using the seed provided"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to start, specify -1 to auto-assign")
    parser.add_argument("--address", "-a", type=str, default="127.0.0.1", help="Address")
    parser.add_argument("--no_browser", "-nb", action="store_true", help="Disable browser launching in GUI mode")
    parser.add_argument("--no_gui", "-ng", action="store_true", help="Disable GUI")
    parser.add_argument("--log_path", default="log", help="Directory path to dump log files, filepath if "
                                                          "disable_logging is false")
    parser.add_argument("--disable_logging", action="store_true", help="Disable Logging, log_path becomes path to file")
    parser.add_argument("--disable_timeout", action="store_true", help="Disable timeouts for player code")
    parser.add_argument("--player", "-p", default="d", help="Specifying player")
    parser.add_argument("--vid_name", "-v", default="game", help="Naming the video file")
    parser.add_argument("--no_vid", "-nv", action="store_true", help="Stops generating video of the session")
    args = parser.parse_args()

    if args.disable_logging:
        if args.log_path == "log":
            args.log_path = "results.log"

    amoeba_game = TimingMazeGame(args)
