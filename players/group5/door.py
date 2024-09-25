from dataclasses import dataclass
from typing import List, Set

import constants


@dataclass
class DoorIdentifier:
    absolute_coord: List[int]
    door_type: int
    relative_coord: List[int] | None = None


def get_updated_frequency_candidates(candidates: Set[int], turn_num: int, door_state: int) -> Set[int]:
    if door_state == constants.BOUNDARY:
        return {0}
    
    if door_state == constants.OPEN:
        candidates.discard(0)
    
    invalidated = {c for c in candidates if c != 0 and (turn_num % c == 0) != (door_state == constants.OPEN)}
    return candidates - invalidated
