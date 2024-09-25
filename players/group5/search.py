

import copy
import logging
import constants
from typing import List, Optional, Tuple

from players.group5.converge import dyjkstra
from players.group5.player_map import PlayerMapInterface
from players.group5.util import setup_file_logger


class SearchStage:
    GO_TO_EDGE = 0
    TRAVERSE_CORRIDORS = 1

class RotationDirection:
    CLOCKWISE = 0
    COUNTERCLOCKWISE = 1


class Corridor:
    def __init__(self, boundaries: List[int], direction: int) -> None:
        self.boundaries: List[int] = boundaries
        self.direction: int = direction

        self.start_indices, self.end_indices = self._get_start_end_indices(boundaries, direction)
        self.strip_indices: List[int] = self._get_strip_indices(boundaries, direction)
        self.reached_start_indices: bool = False

        self.corridor_map = None

    def update_with_boundaries(self, boundaries: List[int]) -> None:
        self.boundaries = boundaries
        self.start_indices, self.end_indices = self._get_start_end_indices(boundaries, self.direction)

        self.strip_indices = self._get_strip_indices(boundaries, self.direction)
        """
        Strip Indices: Currently underutilized but can be used for optimizing search strategy 
        (e.g. keeping track of strips of the corridor previously explored during detours, enabling player to
        skip fully explored corridors or cut down big corridors into smaller ones)
        """

    def _get_start_end_indices(self, boundaries: List[int], direction: int) -> Tuple[List[List[int]], List[List[int]]]:
        if direction == constants.LEFT:
            return (
                [[boundaries[constants.RIGHT], y] for y in range(boundaries[constants.UP], boundaries[constants.DOWN] + 1)],
                [[boundaries[constants.LEFT], y] for y in range(boundaries[constants.UP], boundaries[constants.DOWN] + 1)],
            )
        elif direction == constants.RIGHT:
            return (
                [[boundaries[constants.LEFT], y] for y in range(boundaries[constants.UP], boundaries[constants.DOWN] + 1)],
                [[boundaries[constants.RIGHT], y] for y in range(boundaries[constants.UP], boundaries[constants.DOWN] + 1)],
            )
        elif direction == constants.UP:
            return (
                [[x, boundaries[constants.DOWN]] for x in range(boundaries[constants.LEFT], boundaries[constants.RIGHT] + 1)],
                [[x, boundaries[constants.UP]] for x in range(boundaries[constants.LEFT], boundaries[constants.RIGHT] + 1)],
            )
        elif direction == constants.DOWN:
            return (
                [[x, boundaries[constants.UP]] for x in range(boundaries[constants.LEFT], boundaries[constants.RIGHT] + 1)],
                [[x, boundaries[constants.DOWN]] for x in range(boundaries[constants.LEFT], boundaries[constants.RIGHT] + 1)],
            )

    def _get_strip_indices(self, boundaries: List[int], direction: int) -> List[int]:
        """
        Strip Indices: Currently underutilized but can be used for optimizing search strategy
        Represents the indices of the corridor that must be traversed to ensure full coverage of all cells of the corridor.
        """
        if direction in {constants.UP, constants.DOWN}:
            return range(boundaries[constants.LEFT], boundaries[constants.RIGHT] + 1)
        return range(boundaries[constants.UP], boundaries[constants.DOWN] + 1)
                                                                                     

class SearchStrategy:
    MINIMUM_SEARCH_SPEED_COEFFICIENT = 1
    G2E_STAGE_COMPLETE_SIGNAL = -100
    NO_CORRIDORS_SIGNAL = -200

    def __init__(self, player_map: PlayerMapInterface, radius: int, max_door_frequency: int, logger: logging.Logger) -> None:
        self.logger = self._setup_logger(logger)

        self.player_map = player_map
        self.radius = radius
        self.max_door_frequency = max_door_frequency
        self.stage = SearchStage.GO_TO_EDGE
        
        self._g2e_targets = None

        self.rotation_direction = RotationDirection.CLOCKWISE
        self.first_corridor_traversed = False
        self.corridors: List[Corridor] = []
        self.traversed_corridors: List[Corridor] = []

        self.is_detouring = False
        self.detour_target: Optional[List[List[int]]] = None

        self.search_speed_coefficient = SearchStrategy.MINIMUM_SEARCH_SPEED_COEFFICIENT
        """
        Search Speed Coefficient: This is a multiplier that determines how fast the player scans the entirety of the map. 
        1 is the minimum search speed coefficient. If set at 1, the player will scan all cells of the entire map in one go. 
        If set at a value greater than 1 (e.g. 2), the player will more quickly scan the entire map but may miss some cells,
        thereby requiring a second pass with search_speed_coefficient set at 1 (algorithm will still be robust and find the 
        target, but may take longer than a single run-through). 

        More research and proof of concept is needed to determine the optimal value of search_speed_coefficient. 
        Currently, the player is set to scan the entire map in one go.
        """

    def _setup_logger(self, logger):
        return setup_file_logger(logger, self.__class__.__name__, "./log")

    def move(self, turn: int) -> int:
        if self.stage == SearchStage.GO_TO_EDGE:
            move = self.go_to_edge(turn)
            if move != SearchStrategy.G2E_STAGE_COMPLETE_SIGNAL:
                return move
        
        if self.stage == SearchStage.TRAVERSE_CORRIDORS:
            move = self.traverse_corridors(turn)
            if move == SearchStrategy.NO_CORRIDORS_SIGNAL:
                if len(self.traversed_corridors) <= 1 and len(self.corridors) == 0:
                    self.corridors = self.create_corridors()
            return move

        raise Exception("Invalid stage")

    def _get_g2e_targets(self) -> List[List[int]]:
        boundaries = self.player_map.get_boundaries()
        
        edge_buffer = self.radius - 1
        left_edge_targets_x_idx = boundaries[constants.LEFT] + edge_buffer
        right_edge_targets_x_idx = boundaries[constants.RIGHT] - edge_buffer
        top_edge_targets_y_idx = boundaries[constants.UP] + edge_buffer
        bottom_edge_targets_y_idx = boundaries[constants.DOWN] - edge_buffer

        target = []
        for x in range(left_edge_targets_x_idx, right_edge_targets_x_idx + 1):
            target.append([x, top_edge_targets_y_idx])
            target.append([x, bottom_edge_targets_y_idx])
        for y in range(top_edge_targets_y_idx, bottom_edge_targets_y_idx + 1):
            target.append([left_edge_targets_x_idx, y])
            target.append([right_edge_targets_x_idx, y])        
        return target

    def go_to_edge(self, turn: int) -> int:
        self._g2e_targets = self._get_g2e_targets()
        cur_pos = self.player_map.get_cur_pos()

        # Check completion of G2E stage
        if cur_pos in self._g2e_targets or (self.player_map.is_boundary_found(constants.UP) and self.player_map.is_boundary_found(constants.LEFT)):
            self.stage = SearchStage.TRAVERSE_CORRIDORS
            return SearchStrategy.G2E_STAGE_COMPLETE_SIGNAL

        path = dyjkstra(cur_pos, self._g2e_targets, turn, self.player_map, self.max_door_frequency)
        return path[0] if path else None
    
    def get_first_corridor(self) -> Corridor:
        cur_pos = self.player_map.get_cur_pos()
        boundaries = self.player_map.get_boundaries()
        
        left_edge_x_idx = boundaries[constants.LEFT] + (self.search_speed_coefficient * self.radius - 1)
        right_edge_x_idx = boundaries[constants.RIGHT] - (self.search_speed_coefficient * self.radius - 1)
        top_edge_y_idx = boundaries[constants.UP] + (self.search_speed_coefficient * self.radius - 1)
        bottom_edge_y_idx = boundaries[constants.DOWN] - (self.search_speed_coefficient * self.radius - 1)
        
        my_edge = None
        if cur_pos[0] == left_edge_x_idx:
            my_edge = constants.LEFT
        elif cur_pos[0] == right_edge_x_idx:
            my_edge = constants.RIGHT
        elif cur_pos[1] == top_edge_y_idx:
            my_edge = constants.UP
        elif cur_pos[1] == bottom_edge_y_idx:
            my_edge = constants.DOWN

        if my_edge in {constants.LEFT, constants.RIGHT}:
            if boundaries[constants.DOWN] - cur_pos[1] >= cur_pos[1] - boundaries[constants.UP]:
                self.rotation_direction = RotationDirection.CLOCKWISE if my_edge == constants.LEFT else RotationDirection.COUNTERCLOCKWISE

                if my_edge == constants.LEFT:
                    return Corridor(
                        [
                            boundaries[constants.LEFT], 
                            boundaries[constants.UP], 
                            boundaries[constants.LEFT]+ self.radius * self.search_speed_coefficient, 
                            cur_pos[1],
                        ], 
                        constants.UP,
                    )
                else:
                    return Corridor(
                        [
                            boundaries[constants.RIGHT] - self.radius * self.search_speed_coefficient, 
                            boundaries[constants.UP], 
                            boundaries[constants.RIGHT], 
                            cur_pos[1],
                        ], 
                        constants.UP,
                    )
            else:
                self.rotation_direction = RotationDirection.COUNTERCLOCKWISE if my_edge == constants.LEFT else RotationDirection.CLOCKWISE

                if my_edge == constants.LEFT:
                    return Corridor(
                        [
                            boundaries[constants.LEFT], 
                            cur_pos[1], 
                            boundaries[constants.LEFT]+ self.radius * self.search_speed_coefficient, 
                            boundaries[constants.DOWN],
                        ], 
                        constants.DOWN,
                    )
                else:
                    return Corridor(
                        [
                            boundaries[constants.RIGHT] - self.radius * self.search_speed_coefficient, 
                            cur_pos[1], 
                            boundaries[constants.RIGHT], 
                            boundaries[constants.DOWN],
                        ], 
                        constants.DOWN,
                    )
                
        if my_edge in {constants.UP, constants.DOWN}:
            if boundaries[constants.RIGHT] - cur_pos[0] >= cur_pos[0] - boundaries[constants.LEFT]:
                self.rotation_direction = RotationDirection.COUNTERCLOCKWISE if my_edge == constants.UP else RotationDirection.CLOCKWISE

                if my_edge == constants.UP:
                    return Corridor(
                        [
                            boundaries[constants.LEFT], 
                            boundaries[constants.UP], 
                            cur_pos[0], 
                            boundaries[constants.UP] + self.radius * self.search_speed_coefficient,
                        ], 
                        constants.LEFT,
                    )
                else:
                    return Corridor(
                        [
                            cur_pos[0], 
                            boundaries[constants.DOWN] - self.radius * self.search_speed_coefficient, 
                            boundaries[constants.RIGHT], 
                            boundaries[constants.DOWN],
                        ], 
                        constants.RIGHT,
                    )
            else:
                self.rotation_direction = RotationDirection.COUNTERCLOCKWISE if my_edge == constants.UP else RotationDirection.CLOCKWISE

                if my_edge == constants.UP:
                    return Corridor(
                        [
                            cur_pos[0], 
                            boundaries[constants.UP], 
                            boundaries[constants.RIGHT], 
                            boundaries[constants.UP] + self.radius * self.search_speed_coefficient,
                        ], 
                        constants.RIGHT,
                    )
                else:
                    return Corridor(
                        [
                            boundaries[constants.LEFT], 
                            boundaries[constants.DOWN] - self.radius * self.search_speed_coefficient, 
                            cur_pos[0], 
                            boundaries[constants.DOWN],
                        ], 
                        constants.LEFT,
                    )
        return Corridor(
            [boundaries[constants.LEFT], boundaries[constants.UP], boundaries[constants.RIGHT], boundaries[constants.DOWN]], 
            constants.RIGHT,
        )
        
    def traverse_corridors(self, turn: int) -> int:
        if self.is_detouring:
            return self.detour(turn)
    
        cur_pos = self.player_map.get_cur_pos()
        if not self.first_corridor_traversed:
            if self.corridors == []:
                fst_corridor = self.get_first_corridor()
                self.corridors.append(fst_corridor)

            self.corridors[0].update_with_boundaries([
                max(self.player_map.get_boundaries()[constants.LEFT], self.corridors[0].boundaries[constants.LEFT]),
                max(self.player_map.get_boundaries()[constants.UP], self.corridors[0].boundaries[constants.UP]),
                min(self.player_map.get_boundaries()[constants.RIGHT], self.corridors[0].boundaries[constants.RIGHT]),
                min(self.player_map.get_boundaries()[constants.DOWN], self.corridors[0].boundaries[constants.DOWN]),
            ])

        if self.corridors == []:
            return SearchStrategy.NO_CORRIDORS_SIGNAL
        
        current_corridor = self.corridors[0]

        corridor_map = copy.copy(self.player_map)
        corridor_map.set_boundaries(current_corridor.boundaries)

        start_indices_list = [[int(x) for x in inner_list] for inner_list in current_corridor.start_indices]
        
        if cur_pos in start_indices_list:
            current_corridor.reached_start_indices = True

        if not current_corridor.reached_start_indices:
            path = dyjkstra(cur_pos, start_indices_list, turn, self.player_map, self.max_door_frequency)
            return path[0] if path else None

        if cur_pos in current_corridor.end_indices:
            self.first_corridor_traversed = True
            self.traversed_corridors.append(self.corridors.pop(0))
            if self.corridors == []:
                return SearchStrategy.NO_CORRIDORS_SIGNAL
            return constants.WAIT  # gather info about cells in path to next corridor; could remove for quicker traversal

        path = dyjkstra(cur_pos, current_corridor.end_indices, turn, corridor_map, self.max_door_frequency)
        if not path:
            self.is_detouring = True
            self.detour_target = [get_offset_cell_coordinate(cur_pos, current_corridor.direction, 1)]
            return self.detour(turn)

        return path[0] if path else None

    def create_corridors(self) -> List[Corridor]:
        boundaries = self.player_map.get_boundaries()

        prev_corridor = self.traversed_corridors[-1]
        prev_dir = prev_corridor.direction

        while boundaries[constants.RIGHT] - boundaries[constants.LEFT] > 2*self.radius:
            additional_corridors = {
                constants.RIGHT if self.rotation_direction == RotationDirection.CLOCKWISE else constants.LEFT: Corridor(
                    [
                        boundaries[constants.LEFT], 
                        boundaries[constants.UP], 
                        boundaries[constants.RIGHT] - self.radius * self.search_speed_coefficient, 
                        boundaries[constants.UP] + self.radius * self.search_speed_coefficient,
                    ],
                    constants.RIGHT if self.rotation_direction == RotationDirection.CLOCKWISE else constants.LEFT,
                ),
                constants.DOWN if self.rotation_direction == RotationDirection.CLOCKWISE else constants.UP: Corridor(
                    [
                        boundaries[constants.RIGHT] - self.radius * self.search_speed_coefficient, 
                        boundaries[constants.UP], 
                        boundaries[constants.RIGHT], 
                        boundaries[constants.DOWN] - self.radius * self.search_speed_coefficient,
                    ],
                    constants.DOWN if self.rotation_direction == RotationDirection.CLOCKWISE else constants.UP,
                ),
                constants.LEFT if self.rotation_direction == RotationDirection.CLOCKWISE else constants.RIGHT: Corridor(
                    [
                        boundaries[constants.LEFT] + self.radius * self.search_speed_coefficient, 
                        boundaries[constants.DOWN] - self.radius * self.search_speed_coefficient, 
                        boundaries[constants.RIGHT], 
                        boundaries[constants.DOWN],
                    ],
                    constants.LEFT if self.rotation_direction == RotationDirection.CLOCKWISE else constants.RIGHT,
                ),
                constants.UP if self.rotation_direction == RotationDirection.CLOCKWISE else constants.DOWN: Corridor(
                    [
                        boundaries[constants.LEFT], 
                        boundaries[constants.UP] + self.radius * self.search_speed_coefficient, 
                        boundaries[constants.LEFT] + self.radius * self.search_speed_coefficient, 
                        boundaries[constants.DOWN],
                    ],
                    constants.UP if self.rotation_direction == RotationDirection.CLOCKWISE else constants.DOWN,
                ),
            }
            for _ in range(4):
                nxt_dir = next_direction(self.rotation_direction == RotationDirection.CLOCKWISE, prev_dir)
                self.corridors.append(additional_corridors[nxt_dir])
                prev_dir = nxt_dir
            
            boundaries = [
                boundaries[constants.LEFT] + self.radius * self.search_speed_coefficient,
                boundaries[constants.UP] + self.radius * self.search_speed_coefficient,
                boundaries[constants.RIGHT] - self.radius * self.search_speed_coefficient,
                boundaries[constants.DOWN] - self.radius * self.search_speed_coefficient,
            ]
        
        if boundaries[constants.RIGHT] - boundaries[constants.LEFT] > 0:
            self.corridors.append(Corridor(
                [
                    boundaries[constants.LEFT], 
                    boundaries[constants.UP], 
                    boundaries[constants.LEFT] + (boundaries[constants.RIGHT] - boundaries[constants.LEFT])//2, 
                    boundaries[constants.DOWN],
                ],
                constants.RIGHT,
            ))
            self.corridors.append(Corridor(
                [
                    boundaries[constants.LEFT] + (boundaries[constants.RIGHT] - boundaries[constants.LEFT])//2 + 1,
                    boundaries[constants.UP], 
                    boundaries[constants.RIGHT], 
                    boundaries[constants.DOWN],
                ],
                constants.LEFT,
            ))

        return self.corridors
    
    def detour(self, turn: int) -> int:
        cur_pos = self.player_map.get_cur_pos()
        path = dyjkstra(cur_pos, self.detour_target, turn, self.player_map, self.max_door_frequency)

        if cur_pos in self.detour_target:
            self.is_detouring = False

        if path:
            return path[0]
        return None


"""
HELPER FUNCTIONS
"""        

CLOCKWISE_ORDER = [constants.RIGHT, constants.DOWN, constants.LEFT, constants.UP]
COUNTERCLOCKWISE_ORDER = [constants.RIGHT, constants.UP, constants.LEFT, constants.DOWN]


def next_direction(is_clockwise: bool, current_direction: int) -> int:
    order = CLOCKWISE_ORDER if is_clockwise else COUNTERCLOCKWISE_ORDER
    idx = order.index(current_direction)
    return order[(idx + 1) % 4]


def get_offset_cell_coordinate(cur_pos: List[int], direction: int, offset: int) -> List[int]:
    if direction == constants.LEFT:
        return [cur_pos[0] - offset, cur_pos[1]]
    elif direction == constants.RIGHT:
        return [cur_pos[0] + offset, cur_pos[1]]
    elif direction == constants.UP:
        return [cur_pos[0], cur_pos[1] - offset]
    elif direction == constants.DOWN:
        return [cur_pos[0], cur_pos[1] + offset]
    else:
        raise ValueError("Invalid direction")