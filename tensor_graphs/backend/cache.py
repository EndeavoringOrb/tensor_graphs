import numpy as np
import torch
import heapq
from typing import Dict, Any, List
from ..ir.node import TensorNode


def node_score(node: TensorNode) -> float:
    """
    Calculates the score for a node based on its compute cost and stability.
    """
    runs = node.execution_count if node.execution_count > 0 else 1
    stability = (runs - node.dirty_count) / runs
    return node.compute_cost * stability


class CacheEntry:
    """Wrapper for cache entries to enable heap operations."""

    def __init__(self, node: TensorNode, data: Any, size: int):
        self.node = node
        self.data = data
        self.size = size
        self.score = node_score(node)

    def __lt__(self, other):
        # Min heap based on score
        return self.score < other.score

    def update_score(self):
        """Recalculate score based on current node state."""
        self.score = node_score(self.node)


class CacheManager:
    """
    Manages cache memory for TensorNodes.
    Implements knapsack-style eviction based on a scoring function:
    score = kernel_cost * ((num_runs - num_dirty) / num_runs)

    Uses a min-heap to efficiently find victims and compares incoming
    node value against potential victims before eviction.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.entries: Dict[str, CacheEntry] = {}  # name -> CacheEntry
        self.heap: List[CacheEntry] = []  # Min heap of cache entries

    def has(self, node: TensorNode) -> bool:
        """Checks if a valid entry exists for the node."""
        return node.name in self.entries

    def get(self, node: TensorNode) -> Any:
        """Retrieves data from cache and updates usage statistics."""
        if node.name in self.entries:
            # Update stats
            node.execution_count += 1
            # Note: Score changed, but we don't rebuild heap here for performance.
            # The heap may be slightly stale but will be cleaned on next eviction.
            return self.entries[node.name].data
        return None

    def _rebuild_heap(self):
        """Rebuild heap from current entries with updated scores."""
        for entry in self.entries.values():
            entry.update_score()
        self.heap = list(self.entries.values())
        heapq.heapify(self.heap)

    def _clean_heap(self):
        """Remove stale references from heap top."""
        while self.heap and self.heap[0].node.name not in self.entries:
            heapq.heappop(self.heap)

    def put(self, node: TensorNode, data: Any):
        """
        Tries to insert data into the cache using knapsack-style eviction.

        Strategy:
        1. Calculate size of incoming data
        2. If needed, collect minimum-score victims to free space
        3. Compare incoming node score with sum of victim scores
        4. Only evict if incoming score > victim scores (better value)
        5. Clone data to ensure independence from Executor buffers
        """
        size = 0
        if isinstance(data, np.ndarray):
            size = data.nbytes
        elif isinstance(data, torch.Tensor):
            size = data.numel() * data.element_size()
        else:
            # Fallback for scalars or lists
            return

        if size > self.max_bytes:
            return

        # Check if already exists (update)
        if node.name in self.entries:
            old_size = self.entries[node.name].size
            self.current_bytes -= old_size
            # Remove old entry from tracking (heap will be cleaned later)
            del self.entries[node.name]

        # Calculate how much space we need
        space_needed = size - (self.max_bytes - self.current_bytes)

        if space_needed > 0:
            # Need to evict - collect potential victims using knapsack approach
            self._rebuild_heap()  # Ensure scores are current

            victims: List[CacheEntry] = []
            freed_space = 0
            temp_heap = self.heap.copy()

            # Collect minimum number of lowest-score items to free enough space
            while freed_space < space_needed and temp_heap:
                victim = heapq.heappop(temp_heap)
                if victim.node.name in self.entries:  # Still valid
                    victims.append(victim)
                    freed_space += victim.size

            # If we can't free enough space, don't cache
            if freed_space < space_needed:
                return

            # Compare value: incoming score vs sum of victim scores
            incoming_score = node_score(node)
            victim_score_sum = sum(v.score for v in victims)

            # Only evict if incoming node is more valuable
            # Use a small epsilon for floating point comparison
            if incoming_score * 1.001 <= victim_score_sum:  # 0.1% tolerance, prioritize keeping existing nodes
                return  # Don't evict, incoming item not valuable enough

            # Perform eviction
            for victim in victims:
                if victim.node.name in self.entries:
                    self.current_bytes -= victim.size
                    del self.entries[victim.node.name]

            # Rebuild heap after evictions
            self._rebuild_heap()

        # Clone data to persist outside executor buffer
        if isinstance(data, np.ndarray):
            saved_data = data.copy()
        elif isinstance(data, torch.Tensor):
            saved_data = data.clone()
        else:
            saved_data = data

        # Insert new entry
        entry = CacheEntry(node, saved_data, size)
        self.entries[node.name] = entry
        heapq.heappush(self.heap, entry)
        self.current_bytes += size

    def invalidate(self, node: TensorNode):
        """Remove a node from cache."""
        if node.name in self.entries:
            size = self.entries[node.name].size
            del self.entries[node.name]
            self.current_bytes -= size
            # Note: Stale heap entry will be cleaned on next _clean_heap() call
            # We don't rebuild heap here for performance
