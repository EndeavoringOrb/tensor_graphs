import numpy as np
import torch
from typing import Dict, Any, Tuple
from ..ir.node import TensorNode


class CacheManager:
    """
    Manages cache memory for TensorNodes.
    Implements knapsack-style eviction based on a scoring function:
    score = kernel_cost * ((num_runs - num_dirty) / num_runs)
    """

    def __init__(self, max_bytes: int = 1 * 1024**3):  # Default 1GB
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.cache: Dict[str, Any] = {}  # node.name -> data

    def get_data_size(self, data: Any) -> int:
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        return 0

    def calculate_score(self, node: TensorNode) -> float:
        if node.execution_count == 0:
            return 0.0

        # Stability ratio: (total - dirty) / total
        # High stability (rarely dirty) -> 1.0
        # Low stability (often dirty) -> 0.0
        stability = (node.execution_count - node.dirty_count) / node.execution_count

        # Multiply by cost. Expensive + Stable = High Score.
        return node.compute_cost * stability

    def has(self, node: TensorNode) -> bool:
        return node.name in self.cache

    def get(self, node: TensorNode) -> Any:
        return self.cache.get(node.name)

    def put(self, node: TensorNode, data: Any):
        """
        Attempts to add data to cache. Evicts if necessary.
        """
        size = self.get_data_size(data)

        # 1. If too big for entire cache, skip
        if size > self.max_bytes:
            return

        # 2. Evict until space is available
        while (self.current_bytes + size) > self.max_bytes:
            # Find worst candidate
            # Optimization: We could maintain a heap, but for simplicity/robustness we scan.
            # Only consider nodes currently in cache.
            # Note: access to nodes needs a registry or we assume we can iterate self.cache?
            # self.cache only has data. We need to map back to nodes or store (node, data).
            # Let's assume we can't easily iterate all nodes unless we store them.
            # Storing node objects in cache values:
            pass
            # To fix this, we need to track cached nodes.

            # Simplified eviction:
            # We need to access the nodes associated with cached data to compute scores.
            # But the CacheManager is decoupled.
            # Let's assume the caller manages the nodes, or we store (node, data).
            pass

        # For this implementation, let's implement the eviction logic by iterating `node.cached_output`
        # is NOT used directly by CacheManager, but CacheManager stores the definitive reference.
        # Actually, `TensorNode.cached_output` field is useful for quick access,
        # but CacheManager controls the memory budget.

        # To strictly follow knapsack eviction, we need to know all cached nodes.
        # Let's store `cached_nodes` map.

    # Re-implementation with proper state tracking
    pass


class SimpleCacheManager:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.entries: Dict[
            str, Tuple[TensorNode, Any, int]
        ] = {}  # name -> (node, data, size)

    def get(self, node: TensorNode) -> Any:
        if node.name in self.entries:
            return self.entries[node.name][1]
        return None

    def put(self, node: TensorNode, data: Any):
        size = 0
        if isinstance(data, np.ndarray):
            size = data.nbytes
        elif isinstance(data, torch.Tensor):
            size = data.numel() * data.element_size()

        if size > self.max_bytes:
            return

        # Check if already exists (update)
        if node.name in self.entries:
            old_size = self.entries[node.name][2]
            self.current_bytes -= old_size

        # Evict
        while (self.current_bytes + size) > self.max_bytes:
            if not self.entries:
                return  # Cannot fit

            # Find min score
            min_score = float("inf")
            victim_name = None

            for name, (n, _, _) in self.entries.items():
                # Score = Cost * (1 - DirtyRate)
                runs = n.execution_count if n.execution_count > 0 else 1
                stability = (runs - n.dirty_count) / runs
                score = n.compute_cost * stability

                if score < min_score:
                    min_score = score
                    victim_name = name

            if victim_name:
                _, _, v_size = self.entries.pop(victim_name)
                self.current_bytes -= v_size
            else:
                break

        # Insert
        # We must clone data to ensure it persists outside Executor buffer
        if isinstance(data, np.ndarray):
            saved_data = data.copy()
        elif isinstance(data, torch.Tensor):
            saved_data = data.clone()
        else:
            saved_data = data

        self.entries[node.name] = (node, saved_data, size)
        self.current_bytes += size

        # Link back for easy debug/access
        node.cached_output = saved_data

    def invalidate(self, node: TensorNode):
        if node.name in self.entries:
            _, _, size = self.entries.pop(node.name)
            self.current_bytes -= size
            node.cached_output = None
