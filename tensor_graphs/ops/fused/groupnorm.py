from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory
import numpy as np
from ...ir.dtypes import DType


def groupnorm_decomposition(inputs, attrs=None):
    """
    GroupNorm Decomposition.
    inputs: [x, weight, bias]
    attrs: num_groups, eps
    """
    x, weight, bias = inputs
    num_groups = attrs["num_groups"]
    eps = attrs["eps"]

    # Shape: (N, C, H, W)
    N, C, H, W = x.shape
    assert C % num_groups == 0, "Channels must be divisible by groups"

    # 1. Reshape to [N, G, C//G, H, W]
    G = num_groups
    new_shape_1 = [N, G, C // G, H, W]
    shape_node_1 = TensorNode(
        OpType.CONSTANT,
        DType.INT32,
        [],
        attrs={"value": np.array(new_shape_1, dtype=np.int32)},
    )
    x_g = TensorNode(OpType.RESHAPE, x.dtype, [x, shape_node_1])

    # 2. Compute Mean over [C//G, H, W] -> dims 2, 3, 4
    # Sum
    sum_x = TensorNode(
        OpType.SUM,
        x_g.dtype,
        [x_g],
        attrs={"axis": [2, 3, 4], "keepdims": True},
    )
    # Count = C//G * H * W
    count = (C // G) * H * W
    count_node = TensorNode(OpType.CONSTANT, x.dtype, [], attrs={"value": float(count)})
    mean = TensorNode(OpType.DIVIDE, x_g.dtype, [sum_x, count_node])

    # 3. Compute Var
    x_sub_mean = TensorNode(
        OpType.ADD,
        x_g.dtype,
        [x_g, TensorNode(OpType.NEGATE, mean.dtype, [mean])],
    )
    sq = TensorNode(OpType.MUL, x_sub_mean.dtype, [x_sub_mean, x_sub_mean])
    sum_sq = TensorNode(
        OpType.SUM,
        sq.dtype,
        [sq],
        attrs={"axis": [2, 3, 4], "keepdims": True},
    )
    var = TensorNode(OpType.DIVIDE, sum_sq.dtype, [sum_sq, count_node])

    # 4. Normalize
    eps_node = TensorNode(OpType.CONSTANT, x.dtype, [], attrs={"value": eps})
    var_eps = TensorNode(OpType.ADD, var.dtype, [var, eps_node])
    sqrt = TensorNode(OpType.SQRT, var_eps.dtype, [var_eps])
    one = TensorNode(OpType.CONSTANT, x.dtype, [], attrs={"value": 1.0})
    inv_std = TensorNode(OpType.DIVIDE, one.dtype, [one, sqrt])

    x_norm = TensorNode(OpType.MUL, x_g.dtype, [x_sub_mean, inv_std])

    # 5. Reshape back [N, C, H, W]
    # Actually we need to apply affine transform first?
    # Weight and Bias shape [C].
    # We can reshape weight/bias to [1, G, C//G, 1, 1] and multiply/add.

    w_shape = TensorNode(
        OpType.CONSTANT,
        DType.INT32,
        [],
        attrs={"value": np.array([1, G, C // G, 1, 1], dtype=np.int32)},
    )
    w_g = TensorNode(OpType.RESHAPE, weight.dtype, [weight, w_shape])
    b_g = TensorNode(OpType.RESHAPE, bias.dtype, [bias, w_shape])

    out_g = TensorNode(OpType.MUL, x_norm.dtype, [x_norm, w_g])
    out_g = TensorNode(OpType.ADD, out_g.dtype, [out_g, b_g])

    # Reshape back
    orig_shape = TensorNode(
        OpType.CONSTANT,
        DType.INT32,
        [],
        attrs={"value": np.array([N, C, H, W], dtype=np.int32)},
    )
    result = TensorNode(OpType.RESHAPE, out_g.dtype, [out_g, orig_shape])

    return result


register_reference_factory("GroupNorm", groupnorm_decomposition)
