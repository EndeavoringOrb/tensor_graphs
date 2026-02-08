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
