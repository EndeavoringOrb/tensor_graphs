"""
File: tensor_graphs/ops/fused/llm.py
"""

from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode, ConstantNode
from ..atomic import OpType
from ..interface import CompositeOp
from ..registry import register_composite
from ...ir.dtypes import DType


@register_composite
class RoPE(CompositeOp):
    op_type = "RoPE"

    def decompose(
        self, inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
    ) -> TensorNode:
        # inputs: x, cos, sin
        x, cos, sin = inputs

        # RoPE Logic:
        # x1 = x[..., :D/2]
        # x2 = x[..., D/2:]
        # rotated = cat(-x2, x1)
        # out = x*cos + rotated*sin

        D = x.shape[-1]
        half_d = D // 2 if D is not None else None

        # x1 = x[..., :half_d]
        x1_node = x[..., :half_d]
        x1_node.name = f"{x.name}_x1"

        # x2 = x[..., half_d:]
        x2_node = x[..., half_d:]
        x2_node.name = f"{x.name}_x2"

        # rotated = cat(-x2, x1)
        neg_x2_node = TensorNode(
            op_type=OpType.NEGATE,
            shape=x2_node.shape,
            dtype=x2_node.dtype,
            parents=[x2_node],
            name=f"{x.name}_neg_x2",
        )

        rotated_shape = x.shape
        axis_node = ConstantNode(
            OpType.CONSTANT, (1,), DType.INT32, [], f"{x.name}_axis", value=[-1]
        )
        rotated_node = TensorNode(
            op_type=OpType.CONCAT,
            shape=rotated_shape,
            dtype=x.dtype,
            parents=[neg_x2_node, x1_node, axis_node],
            name=f"{x.name}_rotated",
        )

        # out = x*cos + rotated*sin
        term1_node = TensorNode(
            op_type=OpType.MUL,
            shape=x.shape,
            dtype=x.dtype,
            parents=[x, cos],
            name=f"{x.name}_term1_mul_cos",
        )

        term2_node = TensorNode(
            op_type=OpType.MUL,
            shape=x.shape,
            dtype=x.dtype,
            parents=[rotated_node, sin],
            name=f"{x.name}_term2_mul_sin",
        )

        output_node = TensorNode(
            op_type=OpType.ADD,
            shape=x.shape,
            dtype=x.dtype,
            parents=[term1_node, term2_node],
            name=f"{x.name}_rope_out",
        )

        return output_node


@register_composite
class Embedding(CompositeOp):
    op_type = "Embedding"

    def decompose(
        self, inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
    ) -> TensorNode:
        # Lowering Embedding to Gather
        # inputs: indices, weights
        indices, weights = inputs
        # Gather(weights, indices)
        return TensorNode(
            OpType.GATHER,
            (None, None),  # Shape will be determined by indices and weights
            weights.dtype,
            [weights, indices],
            "embedding_gather",
        )
