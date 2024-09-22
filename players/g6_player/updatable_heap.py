import heapq


class UpdatableHeap:
    def __init__(self) -> None:
        self.heap = []
        self.entry_finder = {}

    def push(self, item, priority: float) -> None:
        heapq.heappush(self.heap, (priority, item))
        self.entry_finder[item] = priority

    def pop(self) -> tuple:
        priority, item = heapq.heappop(self.heap)

        while self.entry_finder[item] != priority:
            priority, item = heapq.heappop(self.heap)

        return (priority, item)

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
        for priority, cell in self.heap:
            if item == cell:
                return True

        return False
