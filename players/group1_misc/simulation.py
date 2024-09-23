import argparse
import os
import json
import time
import numpy as np
from timing_maze_game_simulation import TimingMazeGame
from collections import defaultdict
import tkinter as tk
import sys

sys.setrecursionlimit(10000)


def run_simulation(max_door_frequencies, radii, num_maps_per_config, wait_penalty):
    results = defaultdict(list)
    summary = []

    #### Tom (9/23):
    # for wait_penalty in wait_penalties: ??? 
    ################
    for max_door_frequency in max_door_frequencies:
        for radius in radii:
            for seed in range(num_maps_per_config):
                args = argparse.Namespace(
                    max_door_frequency=max_door_frequency,
                    radius=radius,
                    seed=seed,
                    maze=None,
                    scale=9,
                    no_gui=True,
                    log_path=f"logs/mdf{max_door_frequency}_r{radius}_s{seed}.log",
                    disable_logging=False,
                    disable_timeout=True,
                    player="1",
                    ############# Tom (9/23):
                    wait_penalty=wait_penalty,
                    #########################
                )

                root = tk.Tk()
                game = TimingMazeGame(args, root)

                start_time = time.time()
                try:
                    game.initialize(None)
                finally:
                    end_time = time.time()
                    result = {
                        "turns": game.turns,
                        "valid_moves": game.valid_moves,
                        "time_taken": end_time - start_time,
                        "goal_reached": game.cur_pos[0] == game.end_pos[0]
                        and game.cur_pos[1] == game.end_pos[1],
                    }
                    # Convert tuple to string for JSON compatibility
                    results[f"mdf_{max_door_frequency}_r_{radius}_s_{seed}"].append(
                        result
                    )

                    summary.append(
                        {
                            "max_door_frequency": max_door_frequency,
                            "radius": radius,
                            "seed": seed,
                            "turns": game.turns,
                            "goal_reached": result["goal_reached"],
                        }
                    )

    return results, summary
    results = defaultdict(list)
    summary = []

    for max_door_frequency in max_door_frequencies:
        for radius in radii:
            for seed in range(num_maps_per_config):
                args = argparse.Namespace(
                    max_door_frequency=max_door_frequency,
                    radius=radius,
                    seed=seed,
                    maze=None,
                    scale=9,
                    no_gui=True,
                    log_path=f"logs/mdf{max_door_frequency}_r{radius}_s{seed}.log",
                    disable_logging=False,
                    disable_timeout=True,
                    player="1",
                )

                root = tk.Tk()
                game = TimingMazeGame(args, root)

                start_time = time.time()
                try:
                    game.initialize(None)
                finally:
                    end_time = time.time()
                    result = {
                        "turns": game.turns,
                        "valid_moves": game.valid_moves,
                        "time_taken": end_time - start_time,
                        "goal_reached": game.cur_pos[0] == game.end_pos[0]
                        and game.cur_pos[1] == game.end_pos[1],
                    }
                    results[(max_door_frequency, radius, seed)].append(result)

                    summary.append(
                        {
                            "max_door_frequency": max_door_frequency,
                            "radius": radius,
                            "seed": seed,
                            "turns": game.turns,
                            "goal_reached": result["goal_reached"],
                        }
                    )

    return results, summary


def save_results(results, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert the results to ensure all data is JSON serializable
    cleaned_results = convert_numpy_types(results)

    output_file = os.path.join(output_dir, f"results.json")
    with open(output_file, "w") as f:
        json.dump(cleaned_results, f, indent=2)


def save_summary(summary, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "simulation_summary.csv")
    with open(output_file, "w") as f:
        f.write("max_door_frequency,radius,seed,turns,goal_reached\n")
        for entry in summary:
            f.write(
                f"{entry['max_door_frequency']},{entry['radius']},{entry['seed']},{entry['turns']},{entry['goal_reached']}\n"
            )


def convert_numpy_types(data):
    if isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    return data


def main():
    max_door_frequencies = [3]
    radii = [30]
    num_maps_per_config = 1
    output_dir = "simulation_results"

    all_summary = []

    results, summary = run_simulation(max_door_frequencies, radii, num_maps_per_config)
    save_results(results, output_dir)
    all_summary.extend(summary)
    print(f"Simulation complete")

    save_summary(all_summary, output_dir)
    print(
        "All simulations completed. Results and summary saved in the 'simulation_results' directory."
    )


if __name__ == "__main__":
    main()
