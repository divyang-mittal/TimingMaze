

import copy
import logging
from typing import List, Tuple
import constants
from players.group5.converge import dyjkstra
from players.group5.player_map import PlayerMapInterface
from players.group5.util import setup_file_logger
from timing_maze_state import TimingMazeState


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
        if direction in {constants.UP, constants.DOWN}:
            return range(boundaries[constants.LEFT], boundaries[constants.RIGHT] + 1)
        return range(boundaries[constants.UP], boundaries[constants.DOWN] + 1)
                                                                                     

# SEARCH SPECIFIC FIELDS
class SearchStrategy:
    def __init__(self, player_map: PlayerMapInterface, radius: int, max_door_frequency: int, logger: logging.Logger) -> None:
        self.logger = self._setup_logger(logger)

        self.player_map = player_map
        self.radius = radius
        self.max_door_frequency = max_door_frequency
        self.stage = SearchStage.GO_TO_EDGE
        
        # GO_TO_EDGE SPECIFIC FIELDS
        self._g2e_targets = None

        self.rotation_direction = RotationDirection.CLOCKWISE
        self.first_corridor_traversed = False
        self.corridors: List[Corridor] = []
        self.traversed_corridors: List[Corridor] = []
        # self.current_corridor = None TODO: might not need if corridor[0] is the current corridor

    def _setup_logger(self, logger):
        return setup_file_logger(logger, self.__class__.__name__, "./log")

    def move(self, current_percept: TimingMazeState, turn: int) -> int:
        if self.stage == SearchStage.GO_TO_EDGE:
            move = self.go_to_edge(turn)
            if move != -100:  # TODO: refactor
                return move
        
        if self.stage == SearchStage.TRAVERSE_CORRIDORS:
            move = self.traverse_corridors(turn)
            self.logger.debug(f"Traversed corridors move: {move}")
            if move == -100:
                if len(self.traversed_corridors) == 1 and len(self.corridors) == 0:
                    self.corridors = self.create_corridors()  # TODO: rename
                    self.logger.debug(f"Corridors MADE: {self.corridors}")
            return move

        self.logger.debug(f"Invalid stage: {self.stage}")
        raise Exception("Invalid stage")

    def _get_g2e_targets(self) -> List[List[int]]:
        boundaries = self.player_map.get_boundaries()
        
        left_edge_targets_x_idx = boundaries[constants.LEFT] + (self.radius - 1)
        right_edge_targets_x_idx = boundaries[constants.RIGHT] - (self.radius - 1)
        top_edge_targets_y_idx = boundaries[constants.UP] + (self.radius - 1)
        bottom_edge_targets_y_idx = boundaries[constants.DOWN] - (self.radius - 1)

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
        if cur_pos in self._g2e_targets:
            # create first corridor (best estimate corridor)
            # first_corridor = self.get_first_corridor()
            # self.corridors.append(first_corridor)
            self.stage = SearchStage.TRAVERSE_CORRIDORS
            return -100

        self.logger.debug(f"edge targets: {self._g2e_targets}")
        
        self.logger.debug(f"found edge {self.player_map.get_boundaries()}")

        path = dyjkstra(cur_pos, self._g2e_targets, turn, self.player_map, self.max_door_frequency)
        # print("path: ", path)
        print("Direction: ", path[0]) if path else None
        return path[0] if path else None
    
    def get_first_corridor(self) -> Corridor:
        cur_pos = self.player_map.get_cur_pos()
        boundaries = self.player_map.get_boundaries()
        
        left_edge_x_idx = boundaries[constants.LEFT] + (self.radius - 1)
        right_edge_x_idx = boundaries[constants.RIGHT] - (self.radius - 1)
        top_edge_y_idx = boundaries[constants.UP] + (self.radius - 1)
        bottom_edge_y_idx = boundaries[constants.DOWN] - (self.radius - 1)
        
        my_edge = None
        if cur_pos[0] == left_edge_x_idx:
            my_edge = constants.LEFT
        elif cur_pos[0] == right_edge_x_idx:
            my_edge = constants.RIGHT
        elif cur_pos[1] == top_edge_y_idx:
            my_edge = constants.UP
        elif cur_pos[1] == bottom_edge_y_idx:
            my_edge = constants.DOWN

        #  determine direction to move towards
        if my_edge in {constants.LEFT, constants.RIGHT}:
            if boundaries[constants.DOWN] - cur_pos[1] >= cur_pos[1] - boundaries[constants.UP]:
                # move up. this is clockwise if my_edge is left, anticlockwise if my_edge is right
                self.rotation_direction = RotationDirection.CLOCKWISE if my_edge == constants.LEFT else RotationDirection.COUNTERCLOCKWISE

                if my_edge == constants.LEFT:
                    return Corridor(
                        [
                            boundaries[constants.LEFT], 
                            boundaries[constants.UP], 
                            boundaries[constants.LEFT]+ self.radius, 
                            cur_pos[1],
                        ], 
                        constants.UP,
                    )
                else:
                    return Corridor(
                        [
                            boundaries[constants.RIGHT] - self.radius, 
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
                            boundaries[constants.LEFT]+ self.radius, 
                            boundaries[constants.DOWN],
                        ], 
                        constants.DOWN,
                    )
                else:
                    return Corridor(
                        [
                            boundaries[constants.RIGHT] - self.radius, 
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
                            boundaries[constants.UP] + self.radius,
                        ], 
                        constants.LEFT,
                    )
                else:
                    return Corridor(
                        [
                            cur_pos[0], 
                            boundaries[constants.DOWN] - self.radius, 
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
                            boundaries[constants.UP] + self.radius,
                        ], 
                        constants.RIGHT,
                    )
                else:
                    return Corridor(
                        [
                            boundaries[constants.LEFT], 
                            boundaries[constants.DOWN] - self.radius, 
                            cur_pos[0], 
                            boundaries[constants.DOWN],
                        ], 
                        constants.LEFT,
                    )
        
    
    def traverse_corridors(self, turn: int) -> int:
        cur_pos = self.player_map.get_cur_pos()

        # self.logger.debug(f"지금 제일 위에 있는 건: {self.corridors[0].boundaries}, {self.corridors[0].direction=}, {self.corridors[0].start_indices=}, {self.corridors[0].end_indices=}") if self.corridors else None
        
        if not self.first_corridor_traversed:
            if self.corridors == []:
                fst_corridor = self.get_first_corridor()
                self.corridors.append(fst_corridor)

            # self.logger.debug(f"map's boundaries: {self.player_map.get_boundaries()}")
            self.corridors[0].update_with_boundaries([
                max(self.player_map.get_boundaries()[constants.LEFT], self.corridors[0].boundaries[constants.LEFT]),
                max(self.player_map.get_boundaries()[constants.UP], self.corridors[0].boundaries[constants.UP]),
                min(self.player_map.get_boundaries()[constants.RIGHT], self.corridors[0].boundaries[constants.RIGHT]),
                min(self.player_map.get_boundaries()[constants.DOWN], self.corridors[0].boundaries[constants.DOWN]),
            ])

        if self.corridors == []:
            self.logger.debug(f"Corridors empty")
            return -100  # TODO: handle better
        
        current_corridor = self.corridors[0]

        corridor_map = copy.copy(self.player_map)
        corridor_map.set_boundaries(current_corridor.boundaries)
        
        if cur_pos in current_corridor.start_indices:
            current_corridor.reached_start_indices = True

        if not current_corridor.reached_start_indices:
            path = dyjkstra(cur_pos, current_corridor.start_indices, turn, corridor_map, self.max_door_frequency)
            if not path:
                self.logger.debug(f"AHA 여기구나 {cur_pos} {current_corridor.start_indices}")
            return path[0] if path else None

        # self.logger.debug(f"Traversing endings {current_corridor.end_indices}; cur_pos: {cur_pos}")
        if cur_pos in current_corridor.end_indices:
            self.first_corridor_traversed = True
            self.logger.debug(f"We finished a corridor {current_corridor.end_indices} because we reached {cur_pos}")
            self.traversed_corridors.append(self.corridors.pop(0))
            # self.logger.debug(f"curr_ corridors : {self.corridors}")
            if self.corridors == []:
                return -100  # TODO: signal done
            return -1  # TODO: currently wasting a turn. make it iterative above to avoid this

        path = dyjkstra(cur_pos, current_corridor.end_indices, turn, corridor_map, self.max_door_frequency)
        # if not path:
            # self.logger.debug(f"AHA 여기구2 {cur_pos} {current_corridor.start_indices}")
        return path[0] if path else None

    def create_corridors(self) -> List[Corridor]:
        boundaries = self.player_map.get_boundaries()
        self.logger.debug(f"====boundaries: {boundaries}")

        prev_corridor = self.traversed_corridors[-1]
        prev_dir = prev_corridor.direction
        # while distance from boundary edge to edge is more than 2*radius
        while boundaries[constants.RIGHT] - boundaries[constants.LEFT] > 2*self.radius:
            additional_corridors = {
                constants.RIGHT if self.rotation_direction == RotationDirection.CLOCKWISE else constants.LEFT: Corridor(
                    [
                        boundaries[constants.LEFT], 
                        boundaries[constants.UP], 
                        boundaries[constants.RIGHT] - self.radius, 
                        boundaries[constants.UP] + self.radius,
                    ],
                    constants.RIGHT if self.rotation_direction == RotationDirection.CLOCKWISE else constants.LEFT,
                ),
                constants.DOWN if self.rotation_direction == RotationDirection.CLOCKWISE else constants.UP: Corridor(
                    [
                        boundaries[constants.RIGHT] - self.radius, 
                        boundaries[constants.UP], 
                        boundaries[constants.RIGHT], 
                        boundaries[constants.DOWN] - self.radius,
                    ],
                    constants.DOWN if self.rotation_direction == RotationDirection.CLOCKWISE else constants.UP,
                ),
                constants.LEFT if self.rotation_direction == RotationDirection.CLOCKWISE else constants.RIGHT: Corridor(
                    [
                        boundaries[constants.LEFT] + self.radius, 
                        boundaries[constants.DOWN] - self.radius, 
                        boundaries[constants.RIGHT], 
                        boundaries[constants.DOWN],
                    ],
                    constants.LEFT if self.rotation_direction == RotationDirection.CLOCKWISE else constants.RIGHT,
                ),
                constants.UP if self.rotation_direction == RotationDirection.CLOCKWISE else constants.DOWN: Corridor(
                    [
                        boundaries[constants.LEFT], 
                        boundaries[constants.UP] + self.radius, 
                        boundaries[constants.LEFT] + self.radius, 
                        boundaries[constants.DOWN],
                    ],
                    constants.UP if self.rotation_direction == RotationDirection.CLOCKWISE else constants.DOWN,
                ),
            }
            for i in range(4):
                nxt_dir = next_direction(self.rotation_direction == RotationDirection.CLOCKWISE, prev_dir)
                self.corridors.append(additional_corridors[nxt_dir])
                prev_dir = nxt_dir
            
            # break
            boundaries = [
                boundaries[constants.LEFT] + self.radius,
                boundaries[constants.UP] + self.radius,
                boundaries[constants.RIGHT] - self.radius,
                boundaries[constants.DOWN] - self.radius,
            ]
            # self.logger.debug(f"boundaries: {boundaries}")
        
        self.logger.debug(f"Corridors: {self.corridors}; {len(self.corridors)}")
        for corridor in self.corridors:
            self.logger.debug(f"Corridor: {corridor.boundaries}; {corridor.direction} {corridor.start_indices=} {corridor.end_indices}->")

        return self.corridors
        

CLOCKWISE_ORDER = [constants.RIGHT, constants.DOWN, constants.LEFT, constants.UP]
COUNTERCLOCKWISE_ORDER = [constants.RIGHT, constants.UP, constants.LEFT, constants.DOWN]

# split and repeat clockwise order starting from down or any other direction (function)
# split and repeat anticlockwise order starting from down or any other direction (function)
# get first corridor (function)
# get next corridor (function)

def next_direction(is_clockwise: bool, current_direction: int) -> int:
    order = CLOCKWISE_ORDER if is_clockwise else COUNTERCLOCKWISE_ORDER
    idx = order.index(current_direction)
    return order[(idx + 1) % 4]