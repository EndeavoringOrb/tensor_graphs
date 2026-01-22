from ..ir.node import TensorNode
from ..ir.dtypes import DType, TensorSignature
from ..backend.registry import KernelRegistry

# Ensure kernels are registered
import tensor_graphs.backend.kernels


def resolve_dispatch(node: TensorNode) -> TensorNode:
    """
    Analyzes a node. If a kernel implementation exists, returns the node.
    If not, attempts to inject Cast nodes to find a valid kernel.
    """

    # 1. Check if an kernel exists
    input_sigs = [p.signature for p in node.parents]
    # Changed from get_kernel to select_best_kernel
    kernel = KernelRegistry.select_best_kernel(node.op_type, input_sigs)

    if kernel:
        # print(f"[Dispatch] Found match for {node.op_type} {input_sigs}")
        return node

    print(
        f"[Dispatch] MISSING KERNEL: {node.op_type} {input_sigs}. searching for conversions..."
    )

    # 2. Heuristic: Try to convert everything to FP32
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
