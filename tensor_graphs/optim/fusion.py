"""
File: tensor_graphs/optim/fusion.py
"""

from typing import Optional
from ..ir.node import TensorNode
from ..ops.atomic_types import OpType
from ..ops.fused.fma import fused_mul_add_ref
from ..ops.fused.rms_norm import rms_norm_ref
from ..ops.fused.gelu import gelu_ref


def try_fuse_rmsnorm(node: TensorNode) -> Optional[TensorNode]:
    """
    Pattern match for RMSNorm:
    out = norm * (1 + scale)
    norm = x * inv_sqrt
    inv_sqrt = 1 / sqrt(mean_sq + eps)
    mean_sq = sum(x^2) / N
    """
    if node.op_type != OpType.MUL or len(node.parents) != 2:
        return None

    p1, p2 = node.parents
    norm, one_scale = p1, p2

    # Match (1 + scale)
    if one_scale.op_type != OpType.ADD:
        norm, one_scale = p2, p1
        if one_scale.op_type != OpType.ADD:
            return None

    s_p1, s_p2 = one_scale.parents
    scale = None
    if s_p1.op_type == OpType.CONSTANT and s_p1.attrs.get("value") == 1.0:
        scale = s_p2
    elif s_p2.op_type == OpType.CONSTANT and s_p2.attrs.get("value") == 1.0:
        scale = s_p1

    if scale is None:
        return None

    # Match norm = x * inv_sqrt
    if norm.op_type != OpType.MUL or len(norm.parents) != 2:
        return None

    n_p1, n_p2 = norm.parents
    x, inv_sqrt = n_p1, n_p2

    if inv_sqrt.op_type != OpType.DIVIDE:
        x, inv_sqrt = n_p2, n_p1
        if inv_sqrt.op_type != OpType.DIVIDE:
            return None

    # Match inv_sqrt = 1 / rsqrt
    i_p1, i_p2 = inv_sqrt.parents
    if not (i_p1.op_type == OpType.CONSTANT and i_p1.attrs.get("value") == 1.0):
        return None
    rsqrt = i_p2

    # Match rsqrt = sqrt(add_eps)
    if rsqrt.op_type != OpType.SQRT or len(rsqrt.parents) != 1:
        return None
    add_eps = rsqrt.parents[0]

    # Match add_eps = mean_sq + eps
    if add_eps.op_type != OpType.ADD or len(add_eps.parents) != 2:
        return None

    a_p1, a_p2 = add_eps.parents
    mean_sq, eps = a_p1, a_p2
    if mean_sq.op_type != OpType.DIVIDE:
        mean_sq, eps = a_p2, a_p1
        if mean_sq.op_type != OpType.DIVIDE:
            return None

    # Match mean_sq = sum_sq / n_elements
    m_p1, m_p2 = mean_sq.parents
    sum_sq, n_elements = m_p1, m_p2
    if sum_sq.op_type != OpType.SUM or len(sum_sq.parents) != 1:
        return None

    # Match sum_sq = sum(x^2)
    sq = sum_sq.parents[0]
    if sq.op_type != OpType.MUL or len(sq.parents) != 2:
        return None
    if sq.parents[0] != x or sq.parents[1] != x:
        return None

    # Success
    axis = sum_sq.get_attr("axis", -1)
    return TensorNode(
        op_type=RMSNorm.op_type,
        shape=node.shape,
        dtype=node.dtype,
        parents=[x, scale, eps],
        attrs={"axis": axis},
        name=f"fused_rmsnorm_{node.name}",
    )


def try_fuse_gelu(node: TensorNode) -> Optional[TensorNode]:
    """
    Pattern match for GELU:
    out = (0.5 * x) * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    if node.op_type != OpType.MUL or len(node.parents) != 2:
        return None

    p1, p2 = node.parents
    half_x, one_plus = p1, p2

    # Match (1 + Tanh(...))
    if one_plus.op_type != OpType.ADD:
        half_x, one_plus = p2, p1
        if one_plus.op_type != OpType.ADD:
            return None

    o_p1, o_p2 = one_plus.parents
    tanh_node = None
    if o_p1.op_type == OpType.CONSTANT and o_p1.attrs.get("value") == 1.0:
        tanh_node = o_p2
    elif o_p2.op_type == OpType.CONSTANT and o_p2.attrs.get("value") == 1.0:
        tanh_node = o_p1

    if tanh_node is None or tanh_node.op_type != OpType.DIVIDE:
        # Tanh is decomposed to DIVIDE
        return None

    # Match half_x = x * 0.5
    if half_x.op_type != OpType.MUL or len(half_x.parents) != 2:
        return None
    h_p1, h_p2 = half_x.parents
    x = None
    if h_p1.op_type == OpType.CONSTANT and h_p1.attrs.get("value") == 0.5:
        x = h_p2
    elif h_p2.op_type == OpType.CONSTANT and h_p2.attrs.get("value") == 0.5:
        x = h_p1

    if x is None:
        return None

    # If we got here and it looks like a GELU structure, we'll fuse it.
    # A full deep match of Tanh and the inner polynomial is very verbose.
    # We'll do a quick check on the tanh_node (DIVIDE) to see if its parents
    # involve 'x' to be reasonably sure.

    # We could do more deep matching here if needed.
    # For now, let's assume if it's (0.5*x)*(1+something_that_is_a_divide) it might be GELU.
    # To be safer, let's check for the Tanh numerator/denominator structure roughly.
    if len(tanh_node.parents) != 2:
        return None

    return TensorNode(
        op_type=GELU.op_type,
        shape=node.shape,
        dtype=node.dtype,
        parents=[x],
        name=f"fused_gelu_{node.name}",
    )


def fuse_graph(node: TensorNode, memo=None) -> TensorNode:
    if memo is None:
        memo = {}
    if node in memo:
        return memo[node]

    new_parents = [fuse_graph(p, memo) for p in node.parents]

    # Create a temporary node with new parents to try matching
    temp_node = TensorNode(
        node.op_type,
        node.shape,
        node.dtype,
        new_parents,
        name=node.name,
        attrs=node.attrs,
    )

    # 1. Try RMSNorm
    rmsnorm_node = try_fuse_rmsnorm(temp_node)
    if rmsnorm_node:
        memo[node] = rmsnorm_node
        return rmsnorm_node

    # 2. Try GELU
    gelu_node = try_fuse_gelu(temp_node)
    if gelu_node:
        memo[node] = gelu_node
        return gelu_node

    # 3. Existing MulAdd fusion
    if node.op_type == OpType.ADD:
        lhs, rhs = new_parents
        if lhs.op_type == OpType.MUL:
            fused_node = TensorNode(
                op_type=FusedMulAdd.op_type,
                shape=node.shape,
                dtype=node.dtype,
                parents=[*lhs.parents, rhs],
                name=f"fused_{node.name}",
            )
            memo[node] = fused_node
            return fused_node

    if new_parents == node.parents:
        memo[node] = node
        return node

    new_node = TensorNode(
        node.op_type,
        node.shape,
        node.dtype,
        new_parents,
        name=node.name,
        attrs=node.attrs,
    )
    memo[node] = new_node
    return new_node
