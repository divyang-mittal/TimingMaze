from players.g6_player.classes.cell import Cell
from players.g6_player.data import Move
from players.g6_player.updatable_heap import UpdatableHeap


def a_star(start: Cell, target: Cell) -> list[Move]:
    """
    G: the cost from start to curr + curr to next
    H: the manhattan distance from next to target
    """
    frontier = UpdatableHeap()
    explored = set()

    # heapq.heappush(frontier, (heuristic(start, target), start))
    frontier.push(start, priority=heuristic(start, target), path=[])

    while len(frontier) > 0:
        # state = heapq.heappop(frontier)
        (cost, cell, path) = frontier.pop()
        explored.add(cell)

        print(f"item: {cell}")

        if cell == target:
            # Success
            return []

        for path, neighbour, move in cell.neighbours():
            if neighbour not in explored or neighbour not in frontier:

                frontier.push(
                    neighbour,
                    priority=(
                        cost - heuristic(cell, target)
                    )  # get only the real cost (g) of the current cell
                    + path
                    or float(
                        "inf"
                    )  # set path cost as infinity if path is closed (path == 0)
                    + heuristic(neighbour, target),
                    path=[*path, move],
                )

            elif neighbour in frontier:
                n_cost = cost + path + heuristic(neighbour, target)

                frontier.update(neighbour, n_cost)

    return [Move.WAIT]


def heuristic(start: Cell, target: Cell) -> float:
    """
    Manhattan distance (for now)
    TODO: Explore frequencies
    """
    return abs(start.x - target.x) + abs(start.y - target.y)
