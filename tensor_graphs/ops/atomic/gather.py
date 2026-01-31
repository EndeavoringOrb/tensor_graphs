from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def gather_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Gather: data[indices] (axis=0 gather)
    inputs[0]: Data tensor (commonly (Vocab, Dim))
    inputs[1]: Indices tensor (any rank)
    """
    if len(inputs) != 2:
        raise ValueError("Gather requires exactly 2 inputs: data and indices")

    data, indices = inputs

    return TensorNode(
        OpType.GATHER,
        # Output shape: indices.shape + data.shape[1:] if len(data.shape) > 1 else indices.shape,
        indices.shape + data.shape[1:] if len(data.shape) > 1 else indices.shape,
        data.dtype,
        [data, indices],
        f"gather_{data.name}_{indices.name}",
        backend=data.backend,
    )
