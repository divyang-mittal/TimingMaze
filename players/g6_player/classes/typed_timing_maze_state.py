from typing import List, Optional
from timing_maze_state import TimingMazeState


class TypedTimingMazeState:
    def __init__(
        self,
        maze_state: List[List[int]],
        is_end_visible: bool,
        end_x: Optional[int],
        end_y: Optional[int],
        start_x: int,
        start_y: int,
    ):
        """
        Args:
            maze_state (List[List[int]]): 2D list of integers representing the maze state
            is_end_visible (bool): Boolean representing if the end is visible
            end_x (int): x-coordinate of the end cell
            end_y (int): y-coordinate of the end cell
            start_x (int): x-coordinate of the start cell
            start_y (int): y-coordinate of the start cell
        """
        self.maze_state: List[List[int]] = maze_state
        self.start_x: int = start_x
        self.start_y: int = start_y
        self.is_end_visible: bool = is_end_visible
        if is_end_visible:
            self.end_x = end_x
            self.end_y = end_y

    def __str__(self) -> str:
        return f"Is End Visible: {self.is_end_visible}\nStart: [{self.start_x},{self.start_y}]\n"


def convert(untyped_state: TimingMazeState) -> TypedTimingMazeState:
    # Map the untyped class's attributes to the typed class
    return TypedTimingMazeState(
        maze_state=untyped_state.maze_state,
        is_end_visible=untyped_state.is_end_visible,
        end_x=getattr(untyped_state, "end_x", None),
        end_y=getattr(untyped_state, "end_y", None),
        start_x=untyped_state.start_x,
        start_y=untyped_state.start_y,
    )
