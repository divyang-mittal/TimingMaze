import heapq

from players.g6_player.data import Move


class UpdatableHeap:
    def __init__(self) -> None:
        self.heap = []
        self.entry_finder = {}

    def push(self, item, priority: float, path: list[Move]) -> None:
        heapq.heappush(self.heap, (priority, item, path))
        self.entry_finder[item] = priority

    def pop(self) -> tuple:
        priority, item, path = heapq.heappop(self.heap)

        while self.entry_finder[item] != priority:
            priority, item, path = heapq.heappop(self.heap)

        del self.entry_finder[item]

        return (priority, item, path)

    def update(self, item, priority: float) -> None:
        # do not update if the new priority is higher
        if self.entry_finder[item] < priority:
            return

        heapq.heappush(self.heap, (priority, item))
        self.entry_finder[item] = priority

    def __len__(self) -> int:
        return len(self.heap)

    def __str__(self) -> str:
        return f"Heap(heap_size: {len(self.heap)}, items: {len(self.entry_finder)})"

    def __repr__(self) -> str:
        return str(self)

    def __contains__(self, item) -> bool:
        for _, cell, _ in self.heap:
            if item == cell:
                return True

        return False
