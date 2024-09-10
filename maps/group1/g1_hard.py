import json
import random
import os
import numpy as np

max_door_frequency = 5

map_frequencies = np.ones((100, 100, 4), dtype=int)

cur_pos = np.array([0, 0])
end_pos = np.array([99, 99])

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


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

for i in range(100):
    for j in range(100):
        for k in range(4):
            if k ==0:
                map_frequencies[i][j][k] = np.random.default_rng().integers(1, max_door_frequency)
            else:
                map_frequencies[i][j][k] = np.random.choice(primes)

for i in range(0, 100, 4):
    if i%8 == 0:
        for j in range(1, 100):
            map_frequencies[j][i][UP] = 0
    else:
        for j in range(0, 99):
            map_frequencies[j][i][UP] = 0

# TODO: move to the bottom
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