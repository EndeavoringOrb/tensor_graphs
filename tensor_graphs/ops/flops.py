from typing import Tuple, List, Optional
from ..ops.atomic_types import OpType
import numpy as np

def get_region_shape(full_shape: Tuple[int, ...], region: Optional[Tuple[slice, ...]]) -> Tuple[int, ...]:
    if region is None:
        return full_shape
    
    if not full_shape:
        return ()

    shape = []
    for i, s in enumerate(region):
        if i >= len(full_shape):
            break
        
        dim_size = full_shape[i]
        if dim_size is None:
            # Symbolic?
            shape.append(None)
            continue

        start, stop, step = s.indices(dim_size)
        # Bounding box size (conservative)
        out_dim = (abs(stop - start) + abs(step) - 1) // abs(step)
        shape.append(max(0, out_dim))
    
    # Pad with remaining full_shape dims if region is shorter
    if len(region) < len(full_shape):
        shape.extend(full_shape[len(region):])
        
    return tuple(shape)

def calculate_flops(op_type: str, shape: Tuple[int, ...], input_shapes: List[Tuple[int, ...]], attrs: dict) -> int:
    """
    Calculates the number of floating point operations for a given op and output shape.
    Args:
        op_type: Type of operation
        shape: Shape of the computed region (output)
        input_shapes: Shapes of the input regions used for this computation
        attrs: Node attributes
    """
    if not shape:
        num_elements = 1
    else:
        num_elements = 1
        for d in shape:
            if d is not None:
                num_elements *= d
            else:
                return 0

    # Atomic Ops
    if op_type in [OpType.ADD, OpType.MUL, OpType.DIVIDE, OpType.POWER, OpType.NEGATE, OpType.EXP, OpType.SIN, OpType.COS, OpType.SQRT, OpType.CAST, OpType.TRIU, OpType.FILL, OpType.WHERE]:
        return num_elements

    if op_type == OpType.DOT:
        # A: (..., M, K), B: (..., K, N) -> (..., M, N)
        # FLOPS = 2 * M * N * K
        if len(input_shapes) < 2:
            return 0
        
        # Output shape matches (..., M, N)
        # Input 0 is (..., M, K)
        # K is the last dim of A (or second to last if B is transposed? 
        # but in tensor_graphs DOT usually follows standard matmul)
        K = input_shapes[0][-1]
        if K is None: return 0
        return 2 * num_elements * K

    if op_type == OpType.SUM:
        # Reduce: input_elements additions
        if not input_shapes: return 0
        p_elements = 1
        for d in input_shapes[0]:
            if d is not None:
                p_elements *= d
        return p_elements

    # Higher-level OPs (if they exist as atomic in current impl)
    if op_type == "RMSNorm":
        return 6 * num_elements
    
    if op_type == "Softmax":
        return 3 * num_elements
    
    if op_type == "GELU":
        return 8 * num_elements
        
    if op_type == "RoPE":
        # Broadly: few muls and adds per head_dim
        return 4 * num_elements

    return 0
