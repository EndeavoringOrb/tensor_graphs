"""
File: tensor_graphs/backend/reference.py
"""

import numpy as np
from typing import Dict
from ..ops.atomic_types import OpType
from ..ir.node import TensorNode
from ..backend.registry import KernelRegistry
from ..ops.registry import get_reference_factory
from .kernels import *


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
            kernel = KernelRegistry.select_best_kernel(
                node.op_type, input_sigs, target_dtype=node.dtype
            )

            if kernel:
                val = kernel(parent_vals, node.attrs)
            else:
                # 3. Try Decomposition via Factory
                ref_factory = get_reference_factory(node.op_type)
                if ref_factory:
                    decomp_root = ref_factory(node.parents, node.attrs)
                    val = _eval(decomp_root)
                else:
                    raise NotImplementedError(
                        f"No valid kernel or decomposition for '{node.op_type}'\n{node.get_details()}"
                    )

        cache[node] = val
        return val

    return _eval(root)
