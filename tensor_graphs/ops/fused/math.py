"""
File: tensor_graphs/ops/fused/math.py
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ...ir.node import TensorNode
from ...ir.dtypes import DType
from ..atomic import OpType
from ..interface import CompositeOp
from ..registry import register_composite


@register_composite
class FusedMulAdd(CompositeOp):
    op_type = "FusedMulAdd"

    def decompose(
        self, inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
    ) -> TensorNode:
        # Expected: A, B, C -> (A * B) + C
        if len(inputs) != 3:
            raise ValueError("FusedMulAdd requires 3 inputs")

        a, b, c = inputs
        mul_node = TensorNode(
            OpType.MUL, a.shape, a.dtype, [a, b], f"decomp_mul_{a.name}"
        )
        add_node = TensorNode(
            OpType.ADD, a.shape, a.dtype, [mul_node, c], f"decomp_fma_{a.name}"
        )
        return add_node

    def sample_inputs(self) -> List[Tuple[List[np.ndarray], Dict[str, Any]]]:
        # Case 1: Simple 1D vectors
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        c = np.array([10, 10, 10], dtype=np.float32)
        return [([a, b, c], {})]
