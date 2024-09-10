import json
import random
import os
import numpy as np

max_door_frequency = 15

map_frequencies = np.ones((100, 100, 4), dtype=int)

cur_pos = np.array([25, 25])
end_pos = np.array([75, 75])

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

for i in range(100):
    for j in range(100):
        for k in range(4):
            if np.random.default_rng().random() < 0.05:
                map_frequencies[i][j][k] = 0
            else:
                map_frequencies[i][j][k] = np.random.default_rng().integers(1, max_door_frequency)

for i in range (100):
    map_frequencies[0][i][LEFT] = 0
    map_frequencies[99][i][RIGHT] = 0
    map_frequencies[i][0][UP] = 0
    map_frequencies[i][99][DOWN] = 0

map_frequencies = np.array(map_frequencies)

data = {
    "frequencies": map_frequencies.tolist(),
    "start_pos": cur_pos.tolist(),
    "end_pos": end_pos.tolist()
}

filename = 'g1_hard.json'
with open(os.path.join(os.path.dirname(__file__), filename), 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"JSON file '{filename}' created successfully.")