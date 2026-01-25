from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def embedding_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    indices, weights = inputs
    return TensorNode(
        OpType.GATHER,
        (None, None),
        weights.dtype,
        [weights, indices],
        "embedding_gather",
    )
