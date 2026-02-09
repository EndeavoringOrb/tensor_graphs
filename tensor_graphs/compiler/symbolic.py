from tensor_graphs.config import DEBUG_DETAILED
from tensor_graphs.config import DEBUG_EXECUTION
import sympy as sp
import json
from typing import List, Tuple, Dict, Any, Callable, Optional
from dataclasses import dataclass
from ..ir.node import TensorNode
from ..ops.atomic_types import OpType
from ..ops.registry import get_reference_factory
from ..ir.graph import topological_sort
from .shape_inference import ShapeInference
import numpy as np


def _safe_select(condlist, choicelist, default=np.nan):
    """Wrapper around np.select that ensures conditions are boolean.

    Sympy's lambdify emits nested select() calls where inner results
    (float arrays from default=nan) can end up as conditions in outer calls.
    """
    condlist = [np.asarray(c).astype(bool) for c in condlist]
    return np.select(condlist, choicelist, default=default)


_LAMBDIFY_MODULES = [{"select": _safe_select}, "numpy", "math"]

# ==============================================================================
# Constants & Types
# ==============================================================================

# We use a large integer for "Logical Infinity" when shapes are truly unknown,
# but we prioritize using the actual shape symbols.
LOGICAL_INF = sp.Integer(2147483647)  # Max Int32


@dataclass
class SymbolicRegion:
    ranges: List[Tuple[sp.Expr, sp.Expr]]
    is_dirty_expr: sp.Expr = None

    def __repr__(self):
        return f"SymReg({self.ranges})"


class SymbolicPropagator:
    _registry: Dict[str, Callable] = {}
    _backward_registry: Dict[str, Callable] = {}
    _cache: Dict[str, Callable] = {}

    @classmethod
    def register(cls, op_type: str):
        def decorator(func):
            cls._registry[op_type] = func
            return func

        return decorator

    @classmethod
    def register_backward(cls, op_type: str):
        def decorator(func):
            cls._backward_registry[op_type] = func
            return func

        return decorator

    @classmethod
    def get_propagator(
        cls, node: TensorNode, known_values: Optional[dict] = None
    ) -> Callable:
        input_ranks = tuple(
            (len(p.shape) if p.shape is not None else 0) for p in node.parents
        )
        attrs_key = json.dumps(node.attrs, sort_keys=True, default=str)
        key = f"{node.op_type}|{input_ranks}|{attrs_key}"
        if key in cls._cache:
            return cls._cache[key]
        func = cls._compile(node, known_values)
        cls._cache[key] = func
        return func

    @classmethod
    def get_backward_propagator(
        cls, node: TensorNode, known_values: Optional[dict] = None
    ) -> Callable:
        input_ranks = tuple(
            (len(p.shape) if p.shape is not None else 0) for p in node.parents
        )
        out_rank = len(node.shape) if node.shape is not None else 0
        attrs_key = json.dumps(node.attrs, sort_keys=True, default=str)
        key = f"BACKWARD|{node.op_type}|{input_ranks}|{out_rank}|{attrs_key}"
        if key in cls._cache:
            return cls._cache[key]
        func = cls._compile_backward(node, known_values)
        cls._cache[key] = func
        return func

    @classmethod
    def _compile(
        cls, node: TensorNode, known_values: Optional[dict] = None
    ) -> Callable:
        input_symbols = []
        input_shape_symbols = []
        dirty_args = []
        shape_args = []

        for i, parent in enumerate(node.parents):
            rank = len(parent.shape) if parent.shape is not None else 0
            p_ranges = []
            p_dims = []
            for d in range(rank):
                # Shape Symbols
                dim = sp.Symbol(f"in{i}_shape_d{d}")
                p_dims.append(dim)
                shape_args.append(dim)

                # Dirty Region Symbols
                s = sp.Symbol(f"in{i}_d{d}_s")
                e = sp.Symbol(f"in{i}_d{d}_e")
                p_ranges.append((s, e))
                dirty_args.extend([s, e])

            is_d = sp.And(*[s < e for s, e in p_ranges]) if p_ranges else sp.true
            input_symbols.append(SymbolicRegion(p_ranges, is_dirty_expr=is_d))
            input_shape_symbols.append(tuple(p_dims))

        # Output shape symbols
        out_dims = []
        out_rank = len(node.shape) if node.shape is not None else 0
        for d in range(out_rank):
            dim = sp.Symbol(f"out_shape_d{d}")
            out_dims.append(dim)
            shape_args.append(dim)

        context = {
            "input_shapes": input_shape_symbols,
            "output_shape": tuple(out_dims),
            "attrs": node.attrs,
            "op_type": node.op_type,
        }

        final_region = cls._trace_node(node, input_symbols, context, known_values)

        out_exprs = []
        if final_region and final_region.ranges:
            for s, e in final_region.ranges:
                out_exprs.append(s)
                out_exprs.append(e)

        all_args = dirty_args + shape_args
        if not out_exprs:
            return lambda *args: []

        if DEBUG_EXECUTION and DEBUG_DETAILED:
            print(f"Compiling {node.op_type}")
            print(f"Input shapes: {input_shape_symbols}")
            print(f"Output shape: {out_dims}")
            print(f"Dirty args: {dirty_args}")
            print(f"Shape args: {shape_args}")
            print(f"Out exprs: {out_exprs}")
            print(f"All args: {all_args}")
        return sp.lambdify(all_args, out_exprs, modules=_LAMBDIFY_MODULES, cse=True)

    @classmethod
    def _compile_backward(
        cls, node: TensorNode, known_values: Optional[dict] = None
    ) -> Callable:
        out_rank = len(node.shape) if node.shape is not None else 0
        out_ranges = []
        out_dirty_args = []
        out_dims = []
        out_shape_args = []

        for d in range(out_rank):
            dim = sp.Symbol(f"out_shape_d{d}")
            out_dims.append(dim)
            out_shape_args.append(dim)
            s = sp.Symbol(f"out_d{d}_s")
            e = sp.Symbol(f"out_d{d}_e")
            out_ranges.append((s, e))
            out_dirty_args.extend([s, e])

        output_region = SymbolicRegion(out_ranges)

        input_shape_symbols = []
        shape_args = []
        for i, parent in enumerate(node.parents):
            rank = len(parent.shape) if parent.shape is not None else 0
            p_dims = []
            for d in range(rank):
                dim = sp.Symbol(f"in{i}_shape_d{d}")
                p_dims.append(dim)
                shape_args.append(dim)
            input_shape_symbols.append(tuple(p_dims))

        context = {
            "input_shapes": input_shape_symbols,
            "output_shape": tuple(out_dims),
            "attrs": node.attrs,
            "op_type": node.op_type,
        }

        input_regions = cls._trace_backward_node(
            node, output_region, context, known_values
        )

        flat_results = []
        for reg in input_regions:
            for s, e in reg.ranges:
                flat_results.append(s)
                flat_results.append(e)

        all_args = out_dirty_args + shape_args + out_shape_args
        if not flat_results:
            return lambda *args: []

        if DEBUG_EXECUTION and DEBUG_DETAILED:
            print(f"Compiling backward {node.op_type}")
            print(f"Input shapes: {input_shape_symbols}")
            print(f"Output shape: {out_dims}")
            print(f"Dirty args: {out_dirty_args}")
            print(f"Shape args: {shape_args}")
            print(f"Out exprs: {flat_results}")
            print(f"All args: {all_args}")
        return sp.lambdify(all_args, flat_results, modules=_LAMBDIFY_MODULES, cse=True)

    @classmethod
    def _trace_node(
        cls,
        node: TensorNode,
        inputs: List[SymbolicRegion],
        context: Dict[str, Any],
        known_values: Optional[dict] = None,
    ) -> SymbolicRegion:
        if node.op_type in cls._registry:
            return cls._registry[node.op_type](inputs, context)

        factory = get_reference_factory(node.op_type)
        if not factory:
            raise ValueError(f"No symbolic handler or decomposition for {node.op_type}")

        leaf_parents = [
            TensorNode(
                op_type=p.op_type,
                dtype=p.dtype,
                parents=[],
                shape=p.shape,
                name=p.name,
                attrs=p.attrs,
                backend=p.backend,
                storage_type=p.storage_type,
            )
            for p in node.parents
        ]

        sub_root = factory(leaf_parents, node.attrs)
        sub_nodes = topological_sort(sub_root)
        ShapeInference.infer(sub_nodes, known_values, keep_cut_parent_shapes=True)

        node_map: Dict[TensorNode, SymbolicRegion] = {
            lp: p_sym for lp, p_sym in zip(leaf_parents, inputs)
        }

        for sub in sub_nodes:
            if sub in node_map:
                continue
            if sub.op_type == OpType.CONSTANT:
                node_map[sub] = cls._clean_region(sub.shape or (), context)
                continue

            sub_inputs = [node_map[p] for p in sub.parents]
            sub_ctx = {
                "attrs": sub.attrs,
                "input_shapes": [p.shape or () for p in sub.parents],
                "output_shape": sub.shape or (),
                "op_type": sub.op_type,
            }
            node_map[sub] = cls._trace_node(sub, sub_inputs, sub_ctx, known_values)

        return node_map[sub_root]

    @classmethod
    def _trace_backward_node(
        cls,
        node: TensorNode,
        output_region: SymbolicRegion,
        context: Dict[str, Any],
        known_values: Optional[dict] = None,
    ) -> List[SymbolicRegion]:
        if node.op_type in cls._backward_registry:
            return cls._backward_registry[node.op_type](output_region, context)

        factory = get_reference_factory(node.op_type)
        if not factory:
            raise ValueError(
                f"No symbolic backward handler or decomposition for {node.op_type}"
            )

        sub_root = factory(node.parents, node.attrs)

        # Prevent infinite recursion if the decomposition returns the same op type
        if sub_root.op_type == node.op_type:
            raise NotImplementedError(
                f"Atomic operation '{node.op_type}' is missing a symbolic backward handler "
                f"in SymbolicPropagator._backward_registry. This is required for "
                f"incremental dirty region propagation."
            )

        sub_nodes = topological_sort(sub_root)
        node_map: Dict[TensorNode, SymbolicRegion] = {sub_root: output_region}
        ShapeInference.infer(sub_nodes, known_values, keep_cut_parent_shapes=True)

        for sub in reversed(sub_nodes):
            if sub not in node_map or sub.op_type == OpType.CONSTANT or not sub.parents:
                continue

            sub_ctx = {
                "attrs": sub.attrs,
                "input_shapes": [p.shape or () for p in sub.parents],
                "output_shape": sub.shape or (),
                "op_type": sub.op_type,
            }
            sub_inputs = cls._trace_backward_node(
                sub, node_map[sub], sub_ctx, known_values
            )

            for p, p_reg in zip(sub.parents, sub_inputs):
                if p not in node_map:
                    node_map[p] = p_reg
                else:
                    new_ranges = [
                        _merge_dim(r1, r2)
                        for r1, r2 in zip(node_map[p].ranges, p_reg.ranges)
                    ]
                    node_map[p] = SymbolicRegion(new_ranges)

        results = []
        for p in node.parents:
            results.append(node_map.get(p, cls._clean_region(p.shape or (), context)))
        return results

    @staticmethod
    def _clean_region(
        shape: Tuple[Any, ...], context: Dict[str, Any]
    ) -> SymbolicRegion:
        # Clean region is one where start >= stop.
        # We use (dim+1, -1) to be safely beyond any valid index.
        ranges = []
        for i, dim in enumerate(shape):
            sym_dim = dim if isinstance(dim, sp.Basic) else sp.Integer(dim)
            ranges.append((0, 0))
        return SymbolicRegion(ranges)

    @staticmethod
    def _full_dirty_region(
        shape: Tuple[Any, ...], context: Dict[str, Any]
    ) -> SymbolicRegion:
        ranges = []
        for dim in shape:
            sym_dim = dim if isinstance(dim, sp.Basic) else sp.Integer(dim)
            ranges.append((sp.Integer(0), sym_dim))
        return SymbolicRegion(ranges)


def _merge_dim(
    r1: Tuple[sp.Expr, sp.Expr], r2: Tuple[sp.Expr, sp.Expr]
) -> Tuple[sp.Expr, sp.Expr]:
    return (sp.Min(r1[0], r2[0]), sp.Max(r1[1], r2[1]))


def _is_dirty(region: SymbolicRegion) -> sp.Expr:
    if region.is_dirty_expr is not None:
        return region.is_dirty_expr
    if not region.ranges:
        return sp.true
    return sp.And(*[s < e for s, e in region.ranges])


# ==============================================================================
# Atomic Handlers
# ==============================================================================


@SymbolicPropagator.register(OpType.ARANGE)
def symbolic_arange(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    out_shape = ctx.get("output_shape", (LOGICAL_INF,))
    is_dirty = sp.Or(*[_is_dirty(inp) for inp in inputs])
    clean = SymbolicPropagator._clean_region(out_shape, ctx).ranges[0]
    full = SymbolicPropagator._full_dirty_region(out_shape, ctx).ranges[0]
    out_s = sp.Piecewise((full[0], is_dirty), (clean[0], True))
    out_e = sp.Piecewise((full[1], is_dirty), (clean[1], True))
    return SymbolicRegion([(out_s, out_e)], is_dirty_expr=is_dirty)


@SymbolicPropagator.register(OpType.ADD)
@SymbolicPropagator.register(OpType.MUL)
@SymbolicPropagator.register(OpType.DIVIDE)
@SymbolicPropagator.register(OpType.POWER)
@SymbolicPropagator.register(OpType.WHERE)
def symbolic_elementwise(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    out_shape = ctx.get("output_shape", ())
    max_rank = len(out_shape)
    input_shapes = ctx.get("input_shapes", [])

    aligned_ranges = []
    for i, inp in enumerate(inputs):
        ranges = list(inp.ranges)
        is_d = _is_dirty(inp)
        p_shape = input_shapes[i] if i < len(input_shapes) else ()

        # Broadcasting logic
        for d in range(len(ranges)):
            s, e = ranges[d]
            dim_size = p_shape[d]
            full_d = (sp.Integer(0), out_shape[max_rank - len(ranges) + d])
            s = sp.Piecewise((full_d[0], sp.And(is_d, sp.Eq(dim_size, 1))), (s, True))
            e = sp.Piecewise((full_d[1], sp.And(is_d, sp.Eq(dim_size, 1))), (e, True))
            ranges[d] = (s, e)

        pad = max_rank - len(ranges)
        if pad > 0:
            for d in range(pad):
                full_d = (sp.Integer(0), out_shape[d])
                clean_d = (out_shape[d] + 1, sp.Integer(-1))
                ranges.insert(
                    0,
                    (
                        sp.Piecewise((full_d[0], is_d), (clean_d[0], True)),
                        sp.Piecewise((full_d[1], is_d), (clean_d[1], True)),
                    ),
                )
        aligned_ranges.append(ranges)

    out_ranges = []
    for i in range(max_rank):
        dims = [ar[i] for ar in aligned_ranges]
        current = dims[0]
        for d in dims[1:]:
            current = _merge_dim(current, d)
        out_ranges.append(current)

    return SymbolicRegion(
        out_ranges, is_dirty_expr=sp.Or(*[_is_dirty(inp) for inp in inputs])
    )


@SymbolicPropagator.register(OpType.DOT)
def symbolic_dot(inputs: List[SymbolicRegion], ctx: Dict[str, Any]) -> SymbolicRegion:
    rA, rB = inputs[0], inputs[1]
    out_shape = ctx.get("output_shape", ())

    # K collision: if K is dirty, M and N become full
    k_range_a = rA.ranges[-1]
    k_range_b = rB.ranges[-2]
    any_k_dirty = sp.Or(k_range_a[0] < k_range_a[1], k_range_b[0] < k_range_b[1])

    m_range = rA.ranges[-2]
    n_range = rB.ranges[-1]

    out_m_s = sp.Piecewise((sp.Integer(0), any_k_dirty), (m_range[0], True))
    out_m_e = sp.Piecewise((out_shape[-2], any_k_dirty), (m_range[1], True))
    out_n_s = sp.Piecewise((sp.Integer(0), any_k_dirty), (n_range[0], True))
    out_n_e = sp.Piecewise((out_shape[-1], any_k_dirty), (n_range[1], True))

    return SymbolicRegion(
        list(rA.ranges[:-2]) + [(out_m_s, out_m_e), (out_n_s, out_n_e)],
        is_dirty_expr=sp.Or(_is_dirty(rA), _is_dirty(rB)),
    )


@SymbolicPropagator.register(OpType.SLICE)
def symbolic_slice(inputs: List[SymbolicRegion], ctx: Dict[str, Any]) -> SymbolicRegion:
    inp = inputs[0]
    out_shape = ctx.get("output_shape", ())
    attrs = ctx.get("attrs", {})
    starts, ends, steps = (
        attrs.get("starts", []),
        attrs.get("ends", []),
        attrs.get("steps", []),
    )

    out_ranges = []
    for i, (s_in, e_in) in enumerate(inp.ranges):
        if i >= len(starts):
            out_ranges.append((s_in, e_in))
            continue

        st, en, step = (
            sp.Integer(starts[i]),
            sp.Integer(ends[i] or LOGICAL_INF),
            sp.Integer(steps[i] or 1),
        )
        overlap_s, overlap_e = sp.Max(s_in, st), sp.Min(e_in, en)
        is_empty = overlap_s >= overlap_e

        def sym_ceil(num, denom):
            return sp.floor((num + denom - 1) / denom)

        raw_s, raw_e = sym_ceil(overlap_s - st, step), sym_ceil(overlap_e - st, step)
        out_dim = out_shape[i]
        out_ranges.append(
            (
                sp.Piecewise((out_dim + 1, is_empty), (sp.Max(0, raw_s), True)),
                sp.Piecewise((sp.Integer(-1), is_empty), (sp.Max(0, raw_e), True)),
            )
        )

    return SymbolicRegion(out_ranges)


@SymbolicPropagator.register(OpType.CONCAT)
def symbolic_concat(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    axis = ctx.get("attrs", {}).get("axis", 0)
    input_shapes = ctx.get("input_shapes", [])
    out_shape = ctx.get("output_shape", ())
    rank = len(out_shape)
    if axis < 0:
        axis += rank

    concat_axis_range = (out_shape[axis] + 1, sp.Integer(-1))
    current_offset = sp.Integer(0)

    for i, inp in enumerate(inputs):
        s, e = inp.ranges[axis]
        is_d = s < e
        concat_axis_range = _merge_dim(
            concat_axis_range,
            (
                sp.Piecewise((s + current_offset, is_d), (out_shape[axis] + 1, True)),
                sp.Piecewise((e + current_offset, is_d), (sp.Integer(-1), True)),
            ),
        )
        current_offset += input_shapes[i][axis]

    final_ranges = []
    for d in range(rank):
        if d == axis:
            final_ranges.append(concat_axis_range)
        else:
            merged = (out_shape[d] + 1, sp.Integer(-1))
            for inp in inputs:
                merged = _merge_dim(merged, inp.ranges[d])
            final_ranges.append(merged)

    return SymbolicRegion(
        final_ranges, is_dirty_expr=sp.Or(*[_is_dirty(inp) for inp in inputs])
    )


@SymbolicPropagator.register(OpType.RESHAPE)
def symbolic_reshape(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    out_shape = ctx.get("output_shape", ())
    is_d = _is_dirty(inputs[0])
    full = SymbolicPropagator._full_dirty_region(out_shape, ctx)
    clean = SymbolicPropagator._clean_region(out_shape, ctx)
    ranges = [
        (
            sp.Piecewise((f[0], is_d), (c[0], True)),
            sp.Piecewise((f[1], is_d), (c[1], True)),
        )
        for f, c in zip(full.ranges, clean.ranges)
    ]
    return SymbolicRegion(ranges, is_dirty_expr=is_d)


@SymbolicPropagator.register(OpType.PERMUTE)
def symbolic_permute(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    dims = ctx.get("attrs", {}).get("dims", [])
    inp = inputs[0]
    if not dims:
        return inp
    return SymbolicRegion([inp.ranges[d] for d in dims], is_dirty_expr=_is_dirty(inp))


@SymbolicPropagator.register(OpType.GATHER)
def symbolic_gather(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    data_region, ind_region = inputs[0], inputs[1]
    out_shape = ctx.get("output_shape", ())
    ind_rank = len(ind_region.ranges)
    is_ind_dirty = _is_dirty(ind_region)
    gathered_axis_dirty = data_region.ranges[0][0] < data_region.ranges[0][1]

    out_ranges = []
    for i in range(ind_rank):
        s_in, e_in = ind_region.ranges[i]
        dim_dirty = s_in < e_in
        out_ranges.append(
            (
                sp.Piecewise(
                    (s_in, dim_dirty),
                    (sp.Integer(0), gathered_axis_dirty),
                    (out_shape[i] + 1, True),
                ),
                sp.Piecewise(
                    (e_in, dim_dirty),
                    (out_shape[i], gathered_axis_dirty),
                    (sp.Integer(-1), True),
                ),
            )
        )

    for i, (sd, ed) in enumerate(data_region.ranges[1:]):
        di = i + ind_rank
        dim_dirty = sd < ed
        out_ranges.append(
            (
                sp.Piecewise(
                    (sd, dim_dirty),
                    (sp.Integer(0), is_ind_dirty),
                    (out_shape[di] + 1, True),
                ),
                sp.Piecewise(
                    (ed, dim_dirty),
                    (out_shape[di], is_ind_dirty),
                    (sp.Integer(-1), True),
                ),
            )
        )
    return SymbolicRegion(out_ranges)


@SymbolicPropagator.register(OpType.SUM)
@SymbolicPropagator.register(OpType.MAX)
def symbolic_reduce(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    inp = inputs[0]
    is_inp_dirty = _is_dirty(inp)
    out_shape = ctx.get("output_shape", ())
    attrs = ctx.get("attrs", {})
    axis, keepdims = attrs.get("axis"), attrs.get("keepdims", True)
    rank_in = len(inp.ranges)

    if axis is None:
        axes = list(range(rank_in))
    else:
        axes = [
            a if a >= 0 else a + rank_in
            for a in (axis if isinstance(axis, (list, tuple)) else [axis])
        ]

    out_ranges = []
    out_idx = 0
    for d in range(rank_in):
        if d in axes:
            if keepdims:
                out_ranges.append(
                    (
                        sp.Piecewise(
                            (sp.Integer(0), is_inp_dirty), (sp.Integer(2), True)
                        ),
                        sp.Piecewise(
                            (sp.Integer(1), is_inp_dirty), (sp.Integer(-1), True)
                        ),
                    )
                )
                out_idx += 1
        else:
            out_ranges.append(inp.ranges[d])
            out_idx += 1
    return SymbolicRegion(out_ranges, is_dirty_expr=is_inp_dirty)


@SymbolicPropagator.register(OpType.CAST)
@SymbolicPropagator.register(OpType.COPY_TO)
@SymbolicPropagator.register(OpType.NEGATE)
@SymbolicPropagator.register(OpType.EXP)
@SymbolicPropagator.register(OpType.SIN)
@SymbolicPropagator.register(OpType.COS)
@SymbolicPropagator.register(OpType.SQRT)
@SymbolicPropagator.register(OpType.TRIU)
def symbolic_unary(inputs: List[SymbolicRegion], ctx: Dict[str, Any]) -> SymbolicRegion:
    return SymbolicRegion(inputs[0].ranges, is_dirty_expr=_is_dirty(inputs[0]))


# --- Backward Handlers (Updated for shapes) ---


@SymbolicPropagator.register_backward(OpType.ADD)
@SymbolicPropagator.register_backward(OpType.MUL)
@SymbolicPropagator.register_backward(OpType.DIVIDE)
@SymbolicPropagator.register_backward(OpType.POWER)
@SymbolicPropagator.register_backward(OpType.WHERE)
@SymbolicPropagator.register_backward(OpType.CAST)
@SymbolicPropagator.register_backward(OpType.COPY_TO)
@SymbolicPropagator.register_backward(OpType.NEGATE)
@SymbolicPropagator.register_backward(OpType.EXP)
@SymbolicPropagator.register_backward(OpType.SIN)
@SymbolicPropagator.register_backward(OpType.COS)
@SymbolicPropagator.register_backward(OpType.SQRT)
@SymbolicPropagator.register_backward(OpType.TRIU)
def backward_elementwise(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    out_ranges = out_region.ranges
    out_rank = len(out_ranges)
    input_shapes = ctx["input_shapes"]
    results = []
    for p_shape in input_shapes:
        p_rank = len(p_shape)
        if p_rank == 0:
            results.append(SymbolicRegion([]))
            continue
        p_ranges = list(out_ranges[max(0, out_rank - p_rank) :])
        if len(p_ranges) < p_rank:
            p_ranges = [
                (sp.Integer(0), p_shape[d]) for d in range(p_rank - len(p_ranges))
            ] + p_ranges
        final_p_ranges = []
        for d in range(p_rank):
            s, e, dim_size = p_ranges[d][0], p_ranges[d][1], p_shape[d]
            final_p_ranges.append(
                (
                    sp.Piecewise((sp.Integer(0), sp.Eq(dim_size, 1)), (s, True)),
                    sp.Piecewise((sp.Integer(1), sp.Eq(dim_size, 1)), (e, True)),
                )
            )
        results.append(SymbolicRegion(final_p_ranges))
    return results


@SymbolicPropagator.register_backward(OpType.CONCAT)
def backward_concat(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    axis = ctx["attrs"].get("axis", 0)
    input_shapes = ctx["input_shapes"]
    out_ranges = out_region.ranges
    rank = len(out_ranges)
    if axis < 0:
        axis += rank
    out_s, out_e = out_ranges[axis]
    results = []
    curr_offset = sp.Integer(0)
    for p_shape in input_shapes:
        p_dim = p_shape[axis]
        p_start, p_end = curr_offset, curr_offset + p_dim
        ov_start, ov_end = sp.Max(out_s, p_start), sp.Min(out_e, p_end)
        is_empty = ov_start >= ov_end
        rel_start, rel_end = ov_start - p_start, ov_end - p_start
        p_r = list(out_ranges)
        p_r[axis] = (
            sp.Piecewise((p_dim + 1, is_empty), (sp.Max(0, rel_start), True)),
            sp.Piecewise((sp.Integer(-1), is_empty), (sp.Max(0, rel_end), True)),
        )
        results.append(SymbolicRegion(p_r))
        curr_offset += p_dim
    return results


@SymbolicPropagator.register_backward(OpType.SLICE)
def backward_slice(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    out_ranges, input_shape = out_region.ranges, ctx["input_shapes"][0]
    attrs = ctx.get("attrs", {})
    starts, steps = attrs.get("starts", []), attrs.get("steps", [])
    in_ranges = []
    for i in range(len(input_shape)):
        os, oe = out_ranges[i]
        st, step, dim = (
            sp.Integer(starts[i] if i < len(starts) else 0),
            sp.Integer(steps[i] if i < len(steps) else 1),
            input_shape[i],
        )
        is_empty = os >= oe
        in_ranges.append(
            (
                sp.Piecewise((dim + 1, is_empty), (sp.Max(0, st + os * step), True)),
                sp.Piecewise(
                    (sp.Integer(-1), is_empty),
                    (sp.Max(0, st + (oe - 1) * step + 1), True),
                ),
            )
        )
    return [SymbolicRegion(in_ranges)]


@SymbolicPropagator.register_backward(OpType.DOT)
def backward_dot(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    input_shapes = ctx["input_shapes"]
    A_shape, B_shape = input_shapes[0], input_shapes[1]
    A_rank = len(A_shape)
    B_rank = len(B_shape)
    out_ranges = out_region.ranges

    # Output: (..., M, N)
    m_s, m_e = out_ranges[-2]
    n_s, n_e = out_ranges[-1]
    out_batch_ranges = out_ranges[:-2]

    # A: (..., M, K) — A's batch dims = A_rank - 2
    # Take the LAST (A_rank-2) batch ranges from the output (right-aligned broadcasting)
    a_batch_count = max(A_rank - 2, 0)
    a_batch_ranges = (
        out_batch_ranges[len(out_batch_ranges) - a_batch_count :]
        if a_batch_count > 0
        else []
    )
    a_ranges = list(a_batch_ranges) + [(m_s, m_e), (sp.Integer(0), A_shape[-1])]

    # B: (..., K, N) — B's batch dims = B_rank - 2
    # Take the LAST (B_rank-2) batch ranges from the output (right-aligned broadcasting)
    b_batch_count = max(B_rank - 2, 0)
    b_batch_ranges = (
        out_batch_ranges[len(out_batch_ranges) - b_batch_count :]
        if b_batch_count > 0
        else []
    )
    b_ranges = list(b_batch_ranges) + [(sp.Integer(0), B_shape[-2]), (n_s, n_e)]

    return [SymbolicRegion(a_ranges), SymbolicRegion(b_ranges)]


@SymbolicPropagator.register_backward(OpType.GATHER)
def backward_gather(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    # Input 0: Data (Vocab, Dim), Input 1: Indices (Rank N) -> Out (Rank N + Rank Data-1)
    data_shape = ctx["input_shapes"][0]
    indices_shape = ctx["input_shapes"][1]
    ind_rank = len(indices_shape)

    # Indices dirty region matches leading dims of output
    ind_ranges = out_region.ranges[:ind_rank]
    # Data is treated as full dirty on axis 0 if any index could have changed
    data_ranges = [(sp.Integer(0), d) for d in data_shape]
    return [SymbolicRegion(data_ranges), SymbolicRegion(ind_ranges)]


@SymbolicPropagator.register_backward(OpType.RESHAPE)
def backward_reshape(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    in_shape = ctx["input_shapes"][0]
    is_dirty = _is_dirty(out_region)
    full_in = SymbolicPropagator._full_dirty_region(in_shape, ctx)
    clean_in = SymbolicPropagator._clean_region(in_shape, ctx)

    ranges = []
    for f, c in zip(full_in.ranges, clean_in.ranges):
        ranges.append(
            (
                sp.Piecewise((f[0], is_dirty), (c[0], True)),
                sp.Piecewise((f[1], is_dirty), (c[1], True)),
            )
        )

    results = [SymbolicRegion(ranges)]

    # Shape tensor (parent 1): must emit ranges matching its rank.
    # If the output is dirty we need the full shape tensor; otherwise clean.
    if len(ctx["input_shapes"]) > 1:
        shape_input_shape = ctx["input_shapes"][1]
        full_s = SymbolicPropagator._full_dirty_region(shape_input_shape, ctx)
        clean_s = SymbolicPropagator._clean_region(shape_input_shape, ctx)
        shape_ranges = []
        for f, c in zip(full_s.ranges, clean_s.ranges):
            shape_ranges.append(
                (
                    sp.Piecewise((f[0], is_dirty), (c[0], True)),
                    sp.Piecewise((f[1], is_dirty), (c[1], True)),
                )
            )
        results.append(SymbolicRegion(shape_ranges))

    return results


@SymbolicPropagator.register_backward(OpType.ARANGE)
def backward_arange(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    input_shapes = ctx["input_shapes"]
    is_dirty = _is_dirty(out_region)
    results = []
    for p_shape in input_shapes:
        if len(p_shape) == 0:
            results.append(SymbolicRegion([]))
        else:
            full = SymbolicPropagator._full_dirty_region(p_shape, ctx)
            clean = SymbolicPropagator._clean_region(p_shape, ctx)
            ranges = [
                (
                    sp.Piecewise((f[0], is_dirty), (c[0], True)),
                    sp.Piecewise((f[1], is_dirty), (c[1], True)),
                )
                for f, c in zip(full.ranges, clean.ranges)
            ]
            results.append(SymbolicRegion(ranges))
    return results


@SymbolicPropagator.register_backward(OpType.FILL)
def backward_fill(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    # Input 0: Value, Input 1: Shape
    shape_in = ctx["input_shapes"][1]
    return [SymbolicRegion([]), SymbolicRegion([(sp.Integer(0), shape_in[0])])]


@SymbolicPropagator.register_backward(OpType.REPEAT)
def backward_repeat(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    axis = ctx["attrs"].get("axis", 0)
    repeats = sp.Integer(ctx["attrs"].get("repeats", 1))
    in_shape = ctx["input_shapes"][0]
    rank = len(in_shape)
    if axis < 0:
        axis += rank

    out_s, out_e = out_region.ranges[axis]
    # Input dirty range covers any part that could contribute to the dirty output range
    in_s = sp.floor(out_s / repeats)
    in_e = sp.floor((out_e + repeats - 1) / repeats)

    res_ranges = list(out_region.ranges)
    res_ranges[axis] = (in_s, in_e)
    return [SymbolicRegion(res_ranges)]


@SymbolicPropagator.register_backward(OpType.SUM)
@SymbolicPropagator.register_backward(OpType.MAX)
def backward_reduce(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    in_shape = ctx["input_shapes"][0]
    rank_in = len(in_shape)
    attrs = ctx.get("attrs", {})
    axis = attrs.get("axis")
    keepdims = attrs.get("keepdims", True)

    if axis is None:
        axes = list(range(rank_in))
    else:
        # Normalize axes to positive indices
        axes = [
            a if a >= 0 else a + rank_in
            for a in (axis if isinstance(axis, (list, tuple)) else [axis])
        ]

    in_ranges = []
    out_idx = 0

    # Check if the output region is dirty at all
    is_out_dirty = _is_dirty(out_region)

    for d in range(rank_in):
        if d in axes:
            # For a reduced dimension, if any part of the output is dirty,
            # we need the entire input dimension to perform the reduction.
            dim_size = in_shape[d]
            in_ranges.append(
                (
                    sp.Piecewise((sp.Integer(0), is_out_dirty), (dim_size + 1, True)),
                    sp.Piecewise((dim_size, is_out_dirty), (sp.Integer(-1), True)),
                )
            )
            if keepdims:
                out_idx += 1
        else:
            # For non-reduced dimensions, the dirty range maps 1:1
            in_ranges.append(out_region.ranges[out_idx])
            out_idx += 1

    return [SymbolicRegion(in_ranges)]


@SymbolicPropagator.register_backward(OpType.PERMUTE)
def backward_permute(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    dims = ctx.get("attrs", {}).get("dims", [])
    out_ranges = out_region.ranges
    rank = len(out_ranges)

    if not dims:
        # Default permutation is usually a full reversal
        dims = list(range(rank - 1, -1, -1))

    # Create a placeholder for input ranges
    in_ranges = [None] * rank

    # Map the output range at index 'i' back to the input axis it came from
    for i, original_dim_index in enumerate(dims):
        in_ranges[original_dim_index] = out_ranges[i]

    return [SymbolicRegion(in_ranges)]
