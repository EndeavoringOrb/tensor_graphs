from ..ir.node import TensorNode
from ..ir.dtypes import DType, TensorSignature
from ..backend.registry import KernelRegistry
from ..ops.registry import get_composite_op

# Ensure kernels are registered
import tensor_graphs.backend.kernels


def resolve_dispatch(node: TensorNode) -> TensorNode:
    """
    Analyzes a node. If a kernel implementation exists, returns the node.
    If not, attempts to inject Cast nodes to find a valid kernel.
    """

    input_sigs = [p.signature for p in node.parents]

    # 1. Try Optimized Kernel
    if KernelRegistry.select_best_kernel(node.op_type, input_sigs):
        return node

    # 2. Try Decomposition
    composite = get_composite_op(node.op_type)
    if composite:
        print(f"[Dispatch] No kernel for {node.op_type}, decomposing...")
        # Expand the node into a subgraph of atomic ops
        decomp_root = composite.decompose(node.parents)
        # Recursively resolve the new graph
        # In a real compiler we would replace 'node' in the graph.
        # Here we return the new root.
        return resolve_dispatch(decomp_root)

    print(
        f"[Dispatch] MISSING KERNEL: {node.op_type} {input_sigs}. searching for conversions..."
    )

    # 3. Heuristic: Try to convert everything to FP32
    new_parents = []
    for p in node.parents:
        if p.dtype != DType.FP32:
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
                new_parents.append(cast_node)
            else:
                raise RuntimeError(
                    f"No kernel for {node.op_type} and no path to cast {p.dtype} to FP32."
                )
        else:
            new_parents.append(p)

    lowered_node = TensorNode(
        op_type=node.op_type,
        shape=node.shape,
        dtype=DType.FP32,
        parents=new_parents,
        name=node.name,
    )

    # 4. Final Verification
    new_sigs = [p.signature for p in lowered_node.parents]
    if not KernelRegistry.select_best_kernel(lowered_node.op_type, new_sigs):
        raise RuntimeError(
            f"Even after casting to FP32, no kernel found for {new_sigs}"
        )

    print(f"   -> Rewrite success. New op: {lowered_node}")
    return lowered_node
