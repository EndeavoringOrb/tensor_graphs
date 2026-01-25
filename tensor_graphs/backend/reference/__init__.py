"""
File: tensor_graphs/backend/reference.py
"""

import numpy as np
from typing import Dict
from ...ops.atomic import OpType
from ...ir.node import TensorNode
from ...backend.registry import KernelRegistry
from ...ops.registry import get_composite_op
from ..kernels import *


def evaluate_graph(root: TensorNode, inputs: Dict[str, np.ndarray]) -> np.ndarray:
    cache: Dict[TensorNode, np.ndarray] = {}

    def _eval(node: TensorNode):
        if node in cache:
            return cache[node]

        if node.op_type == OpType.INPUT:
            if node.name not in inputs:
                raise ValueError(f"Missing input data for node: {node.name}")
            val = np.asarray(inputs[node.name])
        elif node.op_type == OpType.CONSTANT:
            val = np.asarray(node.attrs.get("value"))
        else:
            # 1. Evaluate Parents
            parent_vals = [_eval(p) for p in node.parents]
            input_sigs = [p.signature for p in node.parents]

            # 2. Try Kernel
            kernel = KernelRegistry.select_best_kernel(node.op_type, input_sigs)

            if kernel:
                val = kernel(parent_vals, node.attrs)
            else:
                # 3. Try Decomposition
                composite = get_composite_op(node.op_type)
                if composite:
                    # Decompose using the PARENT NODES (not values)
                    # We need to construct the decomposition graph on the fly
                    decomp_root = composite.decompose(node.parents, node.attrs)
                    # Evaluate the decomposition
                    val = _eval(decomp_root)
                else:
                    raise NotImplementedError(
                        f"No valid kernel or decomposition for '{node.op_type}'\n{node.get_details()}"
                    )

        cache[node] = val
        return val

    return _eval(root)
