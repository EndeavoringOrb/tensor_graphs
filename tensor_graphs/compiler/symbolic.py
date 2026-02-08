"""
File: tensor_graphs/compiler/symbolic.py
"""

import sympy as sp
import numpy as np
import json
import functools
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from ..ir.node import TensorNode
from ..ops.atomic_types import OpType
from ..ops.registry import get_reference_factory
from ..ir.graph import topological_sort
from ..ir.dtypes import DType

# ==============================================================================
# Constants & Types
# ==============================================================================

# SymPy Infinity
S_INF = sp.oo

# Runtime constants for "Clean" and "Full" regions
RT_INF = np.inf
RT_NEG_INF = -np.inf


@dataclass
class SymbolicRegion:
    """
    Represents the dirty region of a tensor symbolically.
    ranges: A list of (start, stop) SymPy expressions for each dimension.
    """

    ranges: List[Tuple[sp.Expr, sp.Expr]]

    def __repr__(self):
        return f"SymReg({self.ranges})"


# ==============================================================================
# Registry
# ==============================================================================


class SymbolicPropagator:
    """
    Compiler that converts TensorNode graphs into fast symbolic functions
    for dirty region propagation.
    """

    _registry: Dict[str, Callable] = {}
    _cache: Dict[str, Callable] = {}

    @classmethod
    def register(cls, op_type: str):
        def decorator(func):
            cls._registry[op_type] = func
            return func

        return decorator

    @classmethod
    def get_propagator(cls, node: TensorNode) -> Callable:
        """
        Returns a compiled function that calculates the output dirty region
        given input dirty regions and shapes.

        Signature of returned function:
            f(*dirty_args, *shape_args) -> List[float] (flat start/stop pairs)
        """
        # 1. Generate Cache Key
        input_ranks = tuple(
            (len(p.shape) if p.shape is not None else 0) for p in node.parents
        )

        try:
            attrs_key = json.dumps(node.attrs, sort_keys=True, default=str)
        except TypeError:
            attrs_key = str(node.attrs)

        key = f"{node.op_type}|{input_ranks}|{attrs_key}"

        if key in cls._cache:
            return cls._cache[key]

        # 2. Compile if not found
        func = cls._compile(node)
        cls._cache[key] = func
        return func

    @classmethod
    def _compile(cls, node: TensorNode) -> Callable:
        # Create Symbols for Inputs: (start, stop) pairs
        input_symbols = []
        input_shape_symbols = []

        dirty_args = []
        shape_args = []

        for i, parent in enumerate(node.parents):
            rank = len(parent.shape) if parent.shape is not None else 0

            # Dirty Region Symbols
            p_ranges = []
            for d in range(rank):
                s = sp.Symbol(f"in{i}_d{d}_s")
                e = sp.Symbol(f"in{i}_d{d}_e")
                p_ranges.append((s, e))
                dirty_args.extend([s, e])
            input_symbols.append(SymbolicRegion(p_ranges))

            # Shape Symbols
            p_dims = []
            for d in range(rank):
                dim = sp.Symbol(f"in{i}_shape_d{d}")
                p_dims.append(dim)
                shape_args.extend([dim])
            input_shape_symbols.append(tuple(p_dims))

        # Trace
        context = {
            "input_shapes": input_shape_symbols,
            "attrs": node.attrs,
            "op_type": node.op_type,
        }

        try:
            final_region = cls._trace_node(node, input_symbols, context)
        except Exception as e:
            # Fallback for complex/unknown ops: Full Dirty
            final_region = cls._full_dirty_region(len(node.shape) if node.shape else 0)

        # Flatten Output
        out_exprs = []
        if final_region and final_region.ranges:
            for s, e in final_region.ranges:
                out_exprs.append(s)
                out_exprs.append(e)

        # Compile to Lambda
        all_args = dirty_args + shape_args

        if not out_exprs:
            lam = lambda *args: []
        else:
            # numpy used for Piecewise/min/max/inf handling
            lam = sp.lambdify(all_args, out_exprs, modules=["numpy", "math"])

        return lam

    @classmethod
    def _trace_node(
        cls, node: TensorNode, inputs: List[SymbolicRegion], context: Dict[str, Any]
    ) -> SymbolicRegion:
        # 1. Atomic Handler
        if node.op_type in cls._registry:
            return cls._registry[node.op_type](inputs, context)

        # 2. Decomposition
        factory = get_reference_factory(node.op_type)
        if not factory:
            raise ValueError(f"No symbolic handler or decomposition for {node.op_type}")

        # Instantiate decomposition
        sub_root = factory(node.parents, node.attrs)
        sub_nodes = topological_sort(sub_root)

        # Map original parents to symbolic inputs
        node_map: Dict[TensorNode, SymbolicRegion] = {}

        for p_node, p_sym in zip(node.parents, inputs):
            node_map[p_node] = p_sym

        # Propagate through subgraph
        for sub in sub_nodes:
            if sub in node_map:
                continue

            if sub.op_type == OpType.CONSTANT:
                rank = len(sub.shape) if sub.shape else 0
                node_map[sub] = cls._clean_region(rank)
                continue

            sub_inputs = [node_map[p] for p in sub.parents]

            # For internal nodes, we lack symbolic shapes unless we infer them.
            # We pass empty shapes context; handlers needing shapes (Concat) might fail or degrade.
            sub_ctx = {"attrs": sub.attrs, "input_shapes": [], "op_type": sub.op_type}

            res = cls._trace_node(sub, sub_inputs, sub_ctx)
            node_map[sub] = res

        return node_map[sub_root]

    @staticmethod
    def _clean_region(rank: int) -> SymbolicRegion:
        return SymbolicRegion([(S_INF, -S_INF) for _ in range(rank)])

    @staticmethod
    def _full_dirty_region(rank: int) -> SymbolicRegion:
        return SymbolicRegion([(sp.Integer(0), S_INF) for _ in range(rank)])


# ==============================================================================
# Helper Functions
# ==============================================================================


def _merge_dim(
    r1: Tuple[sp.Expr, sp.Expr], r2: Tuple[sp.Expr, sp.Expr]
) -> Tuple[sp.Expr, sp.Expr]:
    """Union of two 1D intervals."""
    s1, e1 = r1
    s2, e2 = r2
    return (sp.Min(s1, s2), sp.Max(e1, e2))


def _is_dirty(region: SymbolicRegion) -> sp.Expr:
    """Returns symbolic boolean: True if region is NOT empty (dirty)."""
    if not region.ranges:
        return sp.true  # Scalar dirty

    # A region is dirty if ALL dimensions have start < end
    # (Intersection of non-empty intervals)
    conditions = [s < e for s, e in region.ranges]
    # Use And
    return sp.And(*conditions)


# ==============================================================================
# Atomic Handlers
# ==============================================================================


@SymbolicPropagator.register(OpType.ADD)
@SymbolicPropagator.register(OpType.MUL)
@SymbolicPropagator.register(OpType.DIVIDE)
@SymbolicPropagator.register(OpType.POWER)
@SymbolicPropagator.register(OpType.WHERE)
def symbolic_elementwise(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    if not inputs:
        return SymbolicPropagator._clean_region(0)

    # Align ranks (broadcasting)
    max_rank = max(len(inp.ranges) for inp in inputs)
    aligned_ranges = []

    for inp in inputs:
        ranges = list(inp.ranges)
        pad = max_rank - len(ranges)
        if pad > 0:
            # If input is dirty, broadcasted dimensions are full (0, inf)
            is_d = _is_dirty(inp)
            pad_range = (
                sp.Piecewise((sp.Integer(0), is_d), (S_INF, True)),
                sp.Piecewise((S_INF, is_d), (RT_NEG_INF, True)),
            )
            ranges = [pad_range] * pad + ranges
        aligned_ranges.append(ranges)

    out_ranges = []
    for i in range(max_rank):
        dims = [ar[i] for ar in aligned_ranges]
        current = dims[0]
        for d in dims[1:]:
            current = _merge_dim(current, d)
        out_ranges.append(current)

    return SymbolicRegion(out_ranges)


@SymbolicPropagator.register(OpType.DOT)
def symbolic_dot(inputs: List[SymbolicRegion], ctx: Dict[str, Any]) -> SymbolicRegion:
    # A: (..., M, K), B: (..., K, N) -> (..., M, N)
    rA, rB = inputs[0], inputs[1]

    rank_a = len(rA.ranges)
    rank_b = len(rB.ranges)

    m_range = rA.ranges[-2] if rank_a >= 2 else (sp.Integer(0), S_INF)
    k_range_a = rA.ranges[-1] if rank_a >= 1 else (sp.Integer(0), S_INF)

    k_range_b = rB.ranges[-2] if rank_b >= 2 else (sp.Integer(0), S_INF)
    n_range = rB.ranges[-1] if rank_b >= 1 else (sp.Integer(0), S_INF)

    # K collision
    k_dirty_a = k_range_a[0] < k_range_a[1]
    k_dirty_b = k_range_b[0] < k_range_b[1]
    any_k_dirty = sp.Or(k_dirty_a, k_dirty_b)

    # If K is dirty, M and N are FULL. Else preserve M/N ranges.
    out_m_s = sp.Piecewise((sp.Integer(0), any_k_dirty), (m_range[0], True))
    out_m_e = sp.Piecewise((S_INF, any_k_dirty), (m_range[1], True))

    out_n_s = sp.Piecewise((sp.Integer(0), any_k_dirty), (n_range[0], True))
    out_n_e = sp.Piecewise((S_INF, any_k_dirty), (n_range[1], True))

    # Batch dims: simplified union of leading dims (assuming alignment for now)
    # Proper broadcasting would be similar to elementwise
    batch_ranges = []
    # TODO: Implement batch broadcasting if needed

    return SymbolicRegion(batch_ranges + [(out_m_s, out_m_e), (out_n_s, out_n_e)])


@SymbolicPropagator.register(OpType.SLICE)
def symbolic_slice(inputs: List[SymbolicRegion], ctx: Dict[str, Any]) -> SymbolicRegion:
    inp = inputs[0]
    attrs = ctx.get("attrs", {})
    starts = attrs.get("starts", [])
    ends = attrs.get("ends", [])
    steps = attrs.get("steps", [])

    out_ranges = []
    for i, (s_in, e_in) in enumerate(inp.ranges):
        if i >= len(starts):
            out_ranges.append((s_in, e_in))
            continue

        st = sp.Integer(starts[i])
        # Use a large number for None end if unknown, but better to use S_INF
        # Fixme: If ends[i] is None, use S_INF
        en = sp.Integer(ends[i]) if ends[i] is not None else S_INF
        step = sp.Integer(steps[i])

        # Intersection
        overlap_s = sp.Max(s_in, st)
        overlap_e = sp.Min(e_in, en)

        is_empty = overlap_s >= overlap_e

        # Map to output
        def sym_ceil(num, denom):
            return sp.floor((num + denom - 1) / denom)

        raw_s = sym_ceil(overlap_s - st, step)
        raw_e = sym_ceil(overlap_e - st, step)

        fin_s = sp.Piecewise((S_INF, is_empty), (sp.Max(0, raw_s), True))
        fin_e = sp.Piecewise((RT_NEG_INF, is_empty), (sp.Max(0, raw_e), True))

        out_ranges.append((fin_s, fin_e))

    return SymbolicRegion(out_ranges)


@SymbolicPropagator.register(OpType.CONCAT)
def symbolic_concat(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    attrs = ctx.get("attrs", {})
    axis = attrs.get("axis", 0)
    input_shapes = ctx.get("input_shapes", [])

    if not input_shapes:
        rank = len(inputs[0].ranges) if inputs else 1
        return SymbolicPropagator._full_dirty_region(rank)

    rank = len(inputs[0].ranges)
    if axis < 0:
        axis += rank

    out_non_concat = []
    for d in range(rank):
        if d == axis:
            out_non_concat.append(None)
            continue
        current = (S_INF, RT_NEG_INF)
        for inp in inputs:
            current = _merge_dim(current, inp.ranges[d])
        out_non_concat.append(current)

    concat_axis_range = (S_INF, RT_NEG_INF)
    current_offset = sp.Integer(0)

    for i, inp in enumerate(inputs):
        s, e = inp.ranges[axis]
        is_d = s < e

        s_shifted = s + current_offset
        e_shifted = e + current_offset

        s_term = sp.Piecewise((s_shifted, is_d), (S_INF, True))
        e_term = sp.Piecewise((e_shifted, is_d), (RT_NEG_INF, True))

        concat_axis_range = _merge_dim(concat_axis_range, (s_term, e_term))
        current_offset += input_shapes[i][axis]

    final_ranges = []
    for d in range(rank):
        if d == axis:
            final_ranges.append(concat_axis_range)
        else:
            final_ranges.append(out_non_concat[d])

    return SymbolicRegion(final_ranges)


@SymbolicPropagator.register(OpType.RESHAPE)
def symbolic_reshape(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    # Conservative Full Dirty
    return SymbolicRegion([(sp.Integer(0), S_INF)])


@SymbolicPropagator.register(OpType.PERMUTE)
def symbolic_permute(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    inp = inputs[0]
    dims = ctx.get("attrs", {}).get("dims", [])

    if not dims:
        return inp

    new_ranges = [None] * len(dims)
    for i, d in enumerate(dims):
        if d < len(inp.ranges):
            new_ranges[i] = inp.ranges[d]
        else:
            new_ranges[i] = (sp.Integer(0), S_INF)

    return SymbolicRegion(new_ranges)


@SymbolicPropagator.register(OpType.GATHER)
def symbolic_gather(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    ind_dirty = _is_dirty(inputs[1])
    dat_dirty = _is_dirty(inputs[0])

    is_any_dirty = sp.Or(ind_dirty, dat_dirty)
    s = sp.Piecewise((sp.Integer(0), is_any_dirty), (S_INF, True))
    e = sp.Piecewise((S_INF, is_any_dirty), (RT_NEG_INF, True))

    return SymbolicRegion([(s, e)])


@SymbolicPropagator.register(OpType.SUM)
@SymbolicPropagator.register(OpType.MAX)
def symbolic_reduce(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    # If dynamic axis (2 inputs for MAX), fallback to Full
    if len(inputs) > 1:
        # Conservative full dirty
        return SymbolicPropagator._full_dirty_region(1)

    inp = inputs[0]
    attrs = ctx.get("attrs", {})
    axis = attrs.get("axis", None)
    keepdims = attrs.get("keepdims", True)

    # 1. Check if input is dirty at all
    is_inp_dirty = _is_dirty(inp)

    # If input is clean, output is clean.
    # We construct regions that evaluate to clean if !is_inp_dirty

    # Resolve axes
    rank = len(inp.ranges)
    if axis is None:
        axes = list(range(rank))
    elif isinstance(axis, (list, tuple)):
        axes = [a if a >= 0 else a + rank for a in axis]
    else:
        axes = [axis if axis >= 0 else axis + rank]

    # Check if ANY reduction axis is dirty (affects value)
    # Logic: If reduction axis is dirty, the aggregated value changes.
    # Output dirty region on non-reduction axes matches input dirty region.
    # If reduction axis is clean, output is clean?
    # No, if input is dirty on non-reduction axis, output is dirty there.
    # So output region is just the projection of input dirty region onto non-reduced axes.
    # The only catch: if input is clean, output is clean. This is handled by projecting ranges.

    out_ranges = []
    for d in range(rank):
        s, e = inp.ranges[d]
        if d in axes:
            if keepdims:
                # Range becomes (0, 1) if dirty, else (inf, -inf)
                # But actually, if input is dirty anywhere, this output dim is dirty (size 1).
                # Wait, if this axis was clean in input, but other axes were dirty?
                # The reduction over this axis collapses it.
                # If keepdims=True, it's size 1.
                # If the *result* is dirty (which is true if inp is dirty), this dim is dirty.
                s_kd = sp.Piecewise((sp.Integer(0), is_inp_dirty), (S_INF, True))
                e_kd = sp.Piecewise((sp.Integer(1), is_inp_dirty), (RT_NEG_INF, True))
                out_ranges.append((s_kd, e_kd))
            else:
                pass  # Dropped
        else:
            # Preserve range
            # If !is_inp_dirty, this range might still look dirty (e.g. 0 to 10),
            # but we need to ensure the whole region is marked clean if input is clean.
            # However, _is_dirty checks ALL dimensions. If any is clean, result is clean.
            # So preserving ranges is safe.
            out_ranges.append((s, e))

    return SymbolicRegion(out_ranges)


@SymbolicPropagator.register(OpType.CAST)
@SymbolicPropagator.register(OpType.COPY_TO)
@SymbolicPropagator.register(OpType.NEGATE)
@SymbolicPropagator.register(OpType.EXP)
@SymbolicPropagator.register(OpType.SIN)
@SymbolicPropagator.register(OpType.COS)
@SymbolicPropagator.register(OpType.SQRT)
@SymbolicPropagator.register(OpType.TRIU)
@SymbolicPropagator.register("GELU")
@SymbolicPropagator.register("RoPE")
def symbolic_unary(inputs: List[SymbolicRegion], ctx: Dict[str, Any]) -> SymbolicRegion:
    return inputs[0]
