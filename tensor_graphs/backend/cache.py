# tensor_graphs/backend/cache.py
import numpy as np
import torch
import heapq
from typing import Dict, Any, List, Union
from ..ir.node import TensorNode


def node_score(node: TensorNode) -> float:
    """Score for eviction: kernel_cost * stability."""
    runs = node.execution_count if node.execution_count > 0 else 1
    stability = (runs - node.dirty_count) / runs
    return node.compute_cost * stability


class CacheEntry:
    """Wrapper for cache entries to enable heap operations."""

    def __init__(
        self,
        node: TensorNode,
        data: Union[np.ndarray, torch.Tensor],
        size: int,
        is_view: bool = False,
    ):
        self.node = node
        self.data = data
        self.size = size
        self.is_view = is_view  # True if data is a view (no clone needed on evict)
        self.score = node_score(node)

    def __lt__(self, other):
        return self.score < other.score

    def update_score(self):
        self.score = node_score(self.node)


class CacheManager:
    """
    Manages cache memory with knapsack-style eviction.

    Optimizations:
    - Stores references to views when possible (no copy overhead)
    - Only clones data when necessary (policy-driven)
    - Efficient heap-based victim selection
    """

    def __init__(self, max_bytes: int = 5 * 1024**3):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.entries: Dict[str, CacheEntry] = {}
        self.heap: List[CacheEntry] = []

    def has(self, node: TensorNode) -> bool:
        """Check if node is cached."""
        return node.name in self.entries

    def get(self, node: TensorNode) -> Any:
        """Retrieve and update stats. Returns reference (no copy)."""
        if node.name in self.entries:
            node.execution_count += 1
            return self.entries[node.name].data
        return None

    def _rebuild_heap(self):
        """Rebuild heap with current scores."""
        for entry in self.entries.values():
            entry.update_score()
        self.heap = list(self.entries.values())
        heapq.heapify(self.heap)

    def _clean_heap(self):
        """Remove stale entries from heap top."""
        while self.heap and self.heap[0].node.name not in self.entries:
            heapq.heappop(self.heap)

    def _get_size(self, data: Any) -> int:
        """Calculate size of data in bytes."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        return 0

    def put(self, node: TensorNode, data: Union[np.ndarray, torch.Tensor]):
        """
        Cache data using knapsack eviction.

        Optimizations:
        - Detect views (no copy needed)
        - Only clone if policy requires persistence
        """
        size = self._get_size(data)
        if size == 0 or size > self.max_bytes:
            return

        # Check if data is a view (shares memory with another array)
        is_view = False
        if isinstance(data, np.ndarray):
            is_view = not data.flags["OWNDATA"]
        elif isinstance(data, torch.Tensor):
            is_view = data.is_view()

        # Update existing entry
        if node.name in self.entries:
            old_size = self.entries[node.name].size
            self.current_bytes -= old_size
            del self.entries[node.name]

        # Calculate space needed
        space_needed = size - (self.max_bytes - self.current_bytes)

        if space_needed > 0:
            # Evict victims
            self._rebuild_heap()
            victims: List[CacheEntry] = []
            freed_space = 0
            temp_heap = self.heap.copy()

            while freed_space < space_needed and temp_heap:
                victim = heapq.heappop(temp_heap)
                if victim.node.name in self.entries:
                    victims.append(victim)
                    freed_space += victim.size

            if freed_space < space_needed:
                return  # Can't free enough space

            # Check value: incoming vs victims
            incoming_score = node_score(node)
            victim_score_sum = sum(v.score for v in victims)

            if incoming_score * 1.001 <= victim_score_sum:
                return  # Not valuable enough to evict

            # Perform evictions
            for victim in victims:
                if victim.node.name in self.entries:
                    self.current_bytes -= victim.size
                    del self.entries[victim.node.name]

            self._rebuild_heap()

        # Clone data only if it's not a view (views are transient anyway)
        if is_view:
            # Store reference to view (safe because Executor maintains buffer)
            saved_data = data
        else:
            # Clone non-views to ensure data persistence
            if isinstance(data, np.ndarray):
                saved_data = data.copy()
            elif isinstance(data, torch.Tensor):
                saved_data = data.clone()
            else:
                saved_data = data

        # Insert entry
        entry = CacheEntry(node, saved_data, size, is_view=is_view)
        self.entries[node.name] = entry
        heapq.heappush(self.heap, entry)
        self.current_bytes += size

    def invalidate(self, node: TensorNode):
        """Remove node from cache."""
        if node.name in self.entries:
            size = self.entries[node.name].size
            del self.entries[node.name]
            self.current_bytes -= size
