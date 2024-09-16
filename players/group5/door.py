from dataclasses import dataclass
from typing import List, Set

import constants


@dataclass
class DoorIdentifier:
    relative_coord: List[int]
    door_type: int


# TODO: doing this for every door would be time consuming; also make it more memory efficient (gc)
def update_frequency_candidates(candidates: Set[int], turn_num: int, door_state: int, logger) -> Set[int]:
    try:
        if door_state == constants.BOUNDARY:
            logger.debug(f"boundary door found at turn {turn_num}, returning always closed set(0)")
            return {0}
        
        if door_state == constants.OPEN:
            logger.debug(f"open door found at turn {turn_num}, popping always closed (0)")
            candidates.discard(0)
        
        to_remove = set()
        for c in candidates:
            if c == 0:
                continue

            should_be_open = (turn_num % c == 0)
            is_open = (door_state == constants.OPEN)
            if should_be_open != is_open:
                to_remove.add(c)
        candidates = candidates - set(to_remove)
    except Exception as e:
        logger.debug(f"Error updating frequency candidates: {e}")
    return candidates