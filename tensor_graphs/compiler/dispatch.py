from typing import Optional, Dict
from ..ir.node import TensorNode
from ..ir.dtypes import DType, TensorSignature
from ..backend.registry import KernelRegistry
from ..ops.registry import get_composite_op

# Ensure kernels are registered
import tensor_graphs.backend.kernels


def _replace_node_content(dest: TensorNode, src: TensorNode):
    """Internal helper to swap node contents in-place."""
    if dest is src:
        return
    dest.__dict__.update(src.__dict__)
    dest.__class__ = src.__class__


def resolve_dispatch(
    node: TensorNode, memo: Optional[Dict[TensorNode, TensorNode]] = None
) -> TensorNode:
    """
    Analyzes a node and its parents. If a kernel implementation exists, returns the node.
    If not, attempts to decompose or inject Cast nodes to find a valid kernel.
    Performs in-place graph mutation.
    """
    if memo is None:
        memo = {}
    if node in memo:
        return memo[node]
    memo[node] = node

    # 1. Skip Source Nodes
    if node.op_type in ["Input", "Constant"]:
        return node

    # 2. Recursively resolve parents
    for i in range(len(node.parents)):
        node.parents[i] = resolve_dispatch(node.parents[i], memo)

    input_sigs = [p.signature for p in node.parents]

    # 3. Try Optimized Kernel
    if KernelRegistry.select_best_kernel(node.op_type, input_sigs, node.backend):
        return node

    # 4. Special Case: Cast (if no direct kernel, check converters)
    if node.op_type == "Cast" and len(node.parents) == 1:
        if KernelRegistry.find_conversion_path(
            node.parents[0].signature, node.signature
        ):
            return node

    # 5. Try Decomposition
    composite = get_composite_op(node.op_type)
    if composite:
        print(f"[Dispatch] No kernel for {node.op_type}, decomposing...")
        # Expand the node into a subgraph of atomic ops
        decomp_root = composite.decompose(node.parents, node.attrs)
        # Recursively resolve the new graph
        resolved_root = resolve_dispatch(decomp_root, memo)
        _replace_node_content(node, resolved_root)
        return node

    print(
        f"[Dispatch] MISSING KERNEL: {node.op_type} {input_sigs}. searching for conversions..."
    )

    # 5. Heuristic: Try to convert everything to FP32
    new_parents = []
    for p in node.parents:
        if p.dtype not in [DType.FP32, DType.INT32, DType.BOOL]:
            if KernelRegistry.find_conversion_path(
                p.signature, TensorSignature(DType.FP32, p.shape)
            ):
                print(f"   -> Injecting Cast({p.dtype.value} -> fp32)")
                cast_node = TensorNode(
                    op_type="Cast",
                    shape=p.shape,
                    dtype=DType.FP32,
                    parents=[p],
                    name=f"auto_cast_{p.name}",
                )
                # Ensure the new cast node is also resolved
                new_parents.append(resolve_dispatch(cast_node, memo))
            else:
                raise RuntimeError(
                    f"No kernel for {node.op_type} and no path to cast {p.dtype} to FP32."
                )
        else:
            new_parents.append(p)

    # Mutate in-place
    node.dtype = DType.FP32
    node.parents = new_parents

    # 6. Final Verification
    new_sigs = [p.signature for p in node.parents]
    if not KernelRegistry.select_best_kernel(node.op_type, new_sigs, node.backend):
        raise RuntimeError(
            f"Even after casting to FP32, no kernel found for {new_sigs}"
        )

    print(f"   -> Rewrite success. New op: {node}")
    return node
