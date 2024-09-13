import numpy as np
import json

# Constants
L = 5  # Maximum frequency, you can change it as needed
map_size = 100  # Size of the freq

# Initialize freq with zeros (meaning all doors are always closed)
freq = np.zeros((map_size, map_size, 4), dtype=int)

# Create a zig-zag pattern
for row in range(map_size):
    if row % 2 == 0:
        # Moving left to right in even rows
        for col in range(map_size):
            freq[row, col, 1] = L  # Open left door
            freq[row, col, 3] = L - 1  # Open right door

            if col == 0:
                freq[row, col, 1] = 0  # Close left door
                if row > 0:
                    freq[row, col, 0] = L - 1  # Open top door

            if col == map_size - 1:
                freq[row, col, 3] = 0  # Close right door
                if row < map_size - 1:
                    freq[row, col, 2] = L  # Open bottom door

    else:
        # Moving left to right in odd rows
        for col in range(map_size):
            freq[row, col, 1] = L  # Open left door
            freq[row, col, 3] = L - 1  # Open right door

            if col == 0:
                freq[row, col, 1] = 0  # Close left door
                if row < map_size - 1:
                    freq[row, col, 2] = L  # Open bottom door

            if col == map_size - 1:
                freq[row, col, 3] = 0  # Close right door
                freq[row, col, 0] = L - 1  # Open top door


# Save the freq to json file
map = {
    "frequencies": freq.tolist(),
    "start_pos": [0, 0],
    "end_pos": [map_size - 1, map_size - 1],
}

with open("hard_map.json", "w") as json_file:
    json.dump(map, json_file)
