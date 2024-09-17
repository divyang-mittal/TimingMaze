from dataclasses import dataclass
from typing import List, Set

import constants


@dataclass
class DoorIdentifier:
    absolute_coord: List[int]
    door_type: int
    relative_coord: List[int] | None = None


# TODO: doing this for every door would be time consuming; also make it more memory efficient (gc)
def get_updated_frequency_candidates(candidates: Set[int], turn_num: int, door_state: int) -> Set[int]:
    if door_state == constants.BOUNDARY:
        return {0}
    
    if door_state == constants.OPEN:
        candidates.discard(0)
    

    invalidated = set()
    for c in candidates:
        if c == 0:
            continue

        should_be_open, is_open = (turn_num % c == 0), (door_state == constants.OPEN)  # TODO: check this logic
        if should_be_open != is_open:
            invalidated.add(c)
        
    return candidates - invalidated
