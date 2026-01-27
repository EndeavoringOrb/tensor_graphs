from typing import List, Dict, Tuple
from ..ir.node import TensorNode
from ..ir.graph import topological_sort


class LivenessAnalyzer:
    @staticmethod
    def analyze(nodes: List[TensorNode]) -> Dict[TensorNode, List[int]]:
        """
        Computes the live intervals for each node.
        Returns a dict: node -> [birth_time, death_time]
        """
        # Ensure nodes are topologically sorted to assign time steps
        # If the input list is already sorted, this is cheap.
        # But we must trust the caller or re-sort.
        # For safety, let's re-sort or assume the caller passes a valid topo list.
        # Let's assume the caller passes a list.

        node_to_time = {node: i for i, node in enumerate(nodes)}
        intervals = {node: [i, i] for i, node in enumerate(nodes)}

        # Inputs are born at time 0 because they are loaded at the start
        from ..ops.atomic_types import OpType

        for node in nodes:
            if node.op_type == OpType.INPUT:
                intervals[node][0] = 0

        # Update death times
        # A node's death is the time of its last consumer
        for i, node in enumerate(nodes):
            for parent in node.parents:
                if parent in intervals:
                    intervals[parent][1] = max(intervals[parent][1], i)

        # Ensure outputs live until the end
        # An output is a node that is not a parent of any other node
        # (Technically, the recipe root is the primary output)
        last_step = len(nodes) - 1
        has_children = set()
        for node in nodes:
            for parent in node.parents:
                has_children.add(parent)

        for node in nodes:
            if node not in has_children:
                intervals[node][1] = last_step

        return intervals
