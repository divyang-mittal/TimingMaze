import json
import os
import numpy as np

max_door_frequency = 10

map_frequencies = np.ones((100, 100, 4), dtype=int)

cur_pos = np.array([50, 50])
end_pos = np.array([24, 50])

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# these are all expressed as probabilities given that any above them is not chosen
PRIME_PROBABILITY = 0.75
MAX_DOOR_PROBABILITY = 0.5

def sieve_of_eratosthenes(n):
    prime = [True for _ in range(n+1)]
    p = 2
    while (p * p <= n):
        if (prime[p] == True):
            for i in range(p * p, n+1, p):
                prime[i] = False
        p += 1
    prime_numbers = [p for p in range(2, n) if prime[p]]
    return prime_numbers

primes = sieve_of_eratosthenes(max_door_frequency)
print(primes)

# general map (not including maze)
for i in range(100):
    for j in range(100):
        for k in range(4):
            if np.random.rand() < PRIME_PROBABILITY:
                map_frequencies[i][j][k] = np.random.choice(primes)
            elif np.random.rand() < MAX_DOOR_PROBABILITY:
                map_frequencies[i][j][k] = max_door_frequency
            else:
                map_frequencies[i][j][k] = np.random.randint(1, max_door_frequency)
            
# make divider
for i in range(40):
    map_frequencies[50][i][LEFT] = 0

for i in range(25, 50):
    map_frequencies[i][40][UP] = 0
    map_frequencies[i][60][UP] = 0

for i in range(40, 60):
    map_frequencies[25][i][LEFT] = 0

for i in range(60, 100):
    map_frequencies[50][i][LEFT] = 0

map_frequencies[50][0][LEFT] = max_door_frequency
map_frequencies[50][99][LEFT] = max_door_frequency - 1

map_frequencies[49][0][RIGHT] = max_door_frequency - 1
map_frequencies[49][99][RIGHT] = max_door_frequency

# borders
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

filename = 'g1_final.json'
with open(os.path.join(os.path.dirname(__file__), filename), 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"JSON file '{filename}' created successfully.")