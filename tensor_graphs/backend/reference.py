"""
File: tensor_graphs/backend/reference.py
"""

import numpy as np
from typing import Dict, Optional, Any
from ..ops.atomic_types import OpType
from ..ir.node import TensorNode
from ..backend.registry import KernelRegistry
from ..ops.registry import get_reference_factory
from .kernels import *

# Debug flag - set to True to enable debug output
DEBUG = False


def evaluate_graph(
    root: TensorNode,
    inputs: Dict[str, np.ndarray],
    optimization_db=None,
    env_id: str = None,
) -> np.ndarray:

    cache: Dict[TensorNode, np.ndarray] = {}

    def _eval(node: TensorNode):
        if DEBUG:
            print(f"[DEBUG] Evaluating node: {node}")

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

            # 2. Optimization Check: Do we use Kernel or Atomic Decomposition?
            use_kernel = True

            # If optimization DB is present, check preference
            if optimization_db and env_id:
                # We use a simple shape approximation for the key
                shape_str = str(node.shape)
                pref = optimization_db.get_op_preference(
                    node.op_type, shape_str, env_id
                )

                if pref == "GRAPH_RECIPE":
                    # DB explicitly says decomposition is faster
                    use_kernel = False
                    if DEBUG:
                        print(
                            f"[OPT] Using Atomic Decomposition for {node.op_type} (DB Preference)"
                        )

            kernel = None
            if use_kernel:
                kernel = KernelRegistry.select_best_kernel(
                    node.op_type, input_sigs, target_dtype=node.dtype
                )

            if kernel:
                val = kernel(parent_vals, node.attrs)
            else:
                # 3. Fallback / Forced Decomposition
                ref_factory = get_reference_factory(node.op_type)
                if ref_factory:
                    if DEBUG:
                        print(f"[DEBUG]  -> Using ref_factory for {node.op_type}")

                    # Generate the decomposition subgraph
                    decomp_root = ref_factory(node.parents, node.attrs)

                    # Recursively evaluate the decomposition
                    # We pass the same DB context so sub-nodes can also be optimized
                    # Note: Since decomp_root is new, we recursively call _eval on it.
                    # _eval is closed over the 'cache', which is good.

                    # However, evaluate_graph creates a closure.
                    # We need to evaluate the NEW graph nodes which are not in the current cache scope?
                    # Actually, since _eval is recursive and accepts any node,
                    # passing decomp_root to _eval works fine.
                    val = _eval(decomp_root)
                else:
                    if OpType.is_atomic(node.op_type):
                        raise NotImplementedError(
                            f"No registered kernel for atomic op '{node.op_type}'"
                        )
                    raise NotImplementedError(
                        f"No valid kernel or decomposition for '{node.op_type}'"
                    )

        cache[node] = val
        return val

    return _eval(root)
