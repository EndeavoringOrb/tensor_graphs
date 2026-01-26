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

# Debug flag - set to True to enable debug output
DEBUG = False


def evaluate_graph(root: TensorNode, inputs: Dict[str, np.ndarray]) -> np.ndarray:
    cache: Dict[TensorNode, np.ndarray] = {}

    def _eval(node: TensorNode):
        if DEBUG:
            print(f"[DEBUG] Evaluating node: {node}")

        if node in cache:
            if DEBUG:
                print(f"[DEBUG]  -> Cache hit: {node}")
            return cache[node]

        if node.op_type == OpType.INPUT:
            if node.name not in inputs:
                raise ValueError(f"Missing input data for node: {node.name}")
            val = np.asarray(inputs[node.name])
            if DEBUG:
                print(f"[DEBUG] INPUT: {node.name} -> {val.shape} {val.dtype}")
        elif node.op_type == OpType.CONSTANT:
            val = np.asarray(node.attrs.get("value"))
            if DEBUG:
                print(f"[DEBUG] CONSTANT: {node.name} -> {val.shape} {val.dtype}")
        else:
            if DEBUG:
                print(f"[DEBUG] Processing {node.op_type}: {node}")

            # 1. Evaluate Parents
            parent_vals = [_eval(p) for p in node.parents]
            input_sigs = [p.signature for p in node.parents]

            # 2. Try Kernel
            kernel = KernelRegistry.select_best_kernel(
                node.op_type, input_sigs, target_dtype=node.dtype
            )

            if kernel:
                if DEBUG:
                    print(f"[DEBUG]  -> Using kernel for {node.op_type}")
                val = kernel(parent_vals, node.attrs)
            else:
                if OpType.is_atomic(node.op_type):
                    raise NotImplementedError(
                        f"No registered kernel for atomic op '{node.op_type}' with "
                        f"inputs {[s for s in input_sigs]} and target {node.dtype}. "
                        "Check kernel registrations."
                    )
                # 3. Try Decomposition via Factory
                ref_factory = get_reference_factory(node.op_type)
                if ref_factory:
                    if DEBUG:
                        print(f"[DEBUG]  -> Using ref_factory for {node.op_type}")
                    decomp_root = ref_factory(node.parents, node.attrs)
                    val = _eval(decomp_root)
                else:
                    raise NotImplementedError(
                        f"No valid kernel or decomposition for '{node.op_type}'\n{node.get_details()}"
                    )

        cache[node] = val
        if DEBUG:
            print(
                f"[DEBUG]  -> Result: {val.shape} {val.dtype} (first few: {val.flat[:5]})"
            )
        return val

    return _eval(root)
