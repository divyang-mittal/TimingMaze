from players.g6_player.classes.cell import Cell
from players.g6_player.data import Move
from players.g6_player.updatable_heap import UpdatableHeap

from time import sleep


def a_star(start: Cell, target: Cell) -> list[Move]:
    """
    G: the cost from start to curr + curr to next
    H: the manhattan distance from next to target
    """
    frontier = UpdatableHeap()
    explored = set()
    moves = []

    # heapq.heappush(frontier, (heuristic(start, target), start))
    frontier.push(start, priority=heuristic(start, target), moves=[])

    while len(frontier) > 0:
        # state = heapq.heappop(frontier)
        (cost, cell, moves) = frontier.pop()
        explored.add(cell)

        if cell == target:
            # Success
            return moves, cost

        for path, neighbour, move in cell.neighbours():
            if neighbour not in explored and neighbour not in frontier:
                priority = calc_priority(cost, path, cell, neighbour, target)

                frontier.push(
                    neighbour,
                    priority=priority,
                    moves=moves + [move],
                )

            elif neighbour in frontier:
                n_cost = cost + path + heuristic(neighbour, target)

                frontier.update(neighbour, n_cost, moves=moves + [move])

    raise Exception("Target should have been found")


def calc_priority(
    cost: float, path: int, curr: Cell, neighbour: Cell, target: Cell
) -> float:
    # get only the real cost (g) of the current cell
    org_h = heuristic(curr, target)
    new_h = heuristic(neighbour, target)

    # set path cost as infinity if path is closed (path == 0)
    path_freq = path or float("inf")

    return cost - org_h + path_freq + new_h


def heuristic(start: Cell, target: Cell) -> float:
    """
    Manhattan distance (for now)
    TODO: Explore frequencies
    """
    return abs(start.x - target.x) + abs(start.y - target.y)
