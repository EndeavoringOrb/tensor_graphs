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

    def has(self, node: TensorNode) -> bool:
        """Checks if a valid entry exists for the node."""
        return node.name in self.entries

    def get(self, node: TensorNode) -> Any:
        """Retrieves data from cache and updates usage statistics."""
        if node.name in self.entries:
            # Update stats
            node.execution_count += 1
            return self.entries[node.name][1]
        return None

    def put(self, node: TensorNode, data: Any):
        """
        Inserts data into the cache, evicting other entries if necessary.
        Data is cloned to ensure independence from Executor buffers.
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
            old_size = self.entries[node.name][2]
            self.current_bytes -= old_size

        # Evict if needed
        while (self.current_bytes + size) > self.max_bytes:
            if not self.entries:
                return  # Cannot fit even if empty

            # Find min score victim
            min_score = float("inf")
            victim_name = None

            for name, (n, _, _) in self.entries.items():
                # Score = Cost * Stability
                # Stability = (Runs - Dirty) / Runs
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

        # Insert - Clone data to persist outside executor buffer
        if isinstance(data, np.ndarray):
            saved_data = data.copy()
        elif isinstance(data, torch.Tensor):
            saved_data = data.clone()
        else:
            saved_data = data

        self.entries[node.name] = (node, saved_data, size)
        self.current_bytes += size

        # Update node state
        node.cached_output = saved_data  # Optional: link back for debugging

    def invalidate(self, node: TensorNode):
        if node.name in self.entries:
            _, _, size = self.entries.pop(node.name)
            self.current_bytes -= size
            node.cached_output = None
