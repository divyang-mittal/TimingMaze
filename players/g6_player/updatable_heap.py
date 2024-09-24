import heapq

from players.g6_player.data import Move
from numpy.random import default_rng


gen = default_rng()


class UpdatableHeap:
    def __init__(self) -> None:
        self.heap = []
        self.entry_finder: dict = {}

    def push(self, item, priority: float, moves: list[Move]) -> None:
        heapq.heappush(self.heap, (priority, gen.random(), item, moves))
        self.entry_finder[item] = priority

    def pop(self) -> tuple[float, object, list[Move]]:
        priority, _, item, moves = heapq.heappop(self.heap)

        # TODO: check if heap is empty, do not attempt a pop
        while item not in self.entry_finder or self.entry_finder[item] != priority:
            priority, _, item, moves = heapq.heappop(self.heap)

        del self.entry_finder[item]

        return (priority, item, moves)

    def update(self, item, priority: float, moves: list[Move]) -> None:
        # do not update if the new priority is higher
        if self.entry_finder[item] < priority:
            return

        heapq.heappush(self.heap, (priority, gen.random(), item, moves))
        self.entry_finder[item] = priority

    def __len__(self) -> int:
        return len(self.entry_finder)

    def __str__(self) -> str:
        return f"Heap(heap_size: {len(self.heap)}, items: {len(self.entry_finder)})"

    def __repr__(self) -> str:
        return str(self)

    def __contains__(self, item) -> bool:
        return item in self.entry_finder
