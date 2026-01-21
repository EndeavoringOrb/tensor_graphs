# File: tensor_graphs/compiler/dispatch.py
from typing import List
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType, TensorSignature
from tensor_graphs.backend.registry import KernelRegistry
from tensor_graphs.ops.atomic import OpType

def resolve_dispatch(node: TensorNode) -> TensorNode:
    """
    Analyzes a node. If an exact kernel implementation exists, returns the node.
    If not, attempts to inject Cast nodes to find a valid kernel.
    """
    
    # 1. Check if an exact kernel exists
    input_sigs = [p.signature for p in node.parents]
    kernel = KernelRegistry.get_kernel(node.op_type, input_sigs)
    
    if kernel:
        print(f"[Dispatch] Found exact match for {node.op_type} {input_sigs}")
        return node # All good

    print(f"[Dispatch] MISSING KERNEL: {node.op_type} {input_sigs}. searching for conversions...")

    # 2. Heuristic: Try to convert everything to FP32 (The "Common Denominator")
    # In a real system (TVM/Triton), this would be a cost-based search.
    new_parents = []
    
    for p in node.parents:
        if p.dtype != DType.FP32:
            # Check if we have a converter
            if KernelRegistry.find_conversion_path(p.signature, TensorSignature(DType.FP32, p.shape)):
                print(f"   -> Injecting Cast({p.dtype.value} -> fp32)")
                
                # Insert Cast Node
                cast_node = TensorNode(
                    op_type="Cast",
                    shape=p.shape,
                    dtype=DType.FP32,
                    parents=[p],
                    name=f"auto_cast_{p.name}"
                )
                new_parents.append(cast_node)
            else:
                # No path to FP32, we are stuck
                raise RuntimeError(f"No kernel for {node.op_type} and no path to cast {p.dtype} to FP32.")
        else:
            new_parents.append(p)
            
    # 3. Create a new Node with the casted parents
    # We assume the output of the operation will be FP32 if inputs are FP32
    lowered_node = TensorNode(
        op_type=node.op_type,
        shape=node.shape,
        dtype=DType.FP32, # Result is promoted
        parents=new_parents,
        name=node.name
    )
    
    # 4. Final Verification: Do we have the FP32 kernel?
    new_sigs = [p.signature for p in lowered_node.parents]
    if not KernelRegistry.get_kernel(lowered_node.op_type, new_sigs):
         raise RuntimeError(f"Even after casting to FP32, no kernel found for {new_sigs}")

    print(f"   -> Rewrite success. New op: {lowered_node}")
    return lowered_node