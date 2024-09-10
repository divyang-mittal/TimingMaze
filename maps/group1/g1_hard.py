import json
import random
import os

data = {
    "frequencies": [[1 for _ in range(4)] for _ in range(10000)],
    "start_pos": [25, 25],
    "end_pos": [75, 75]
}

with open(os.path.join(os.path.dirname(__file__), "g1_hard.json"), "w") as f:
    json.dump(data, f)