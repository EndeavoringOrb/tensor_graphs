"""
File: tensor_graphs/compiler/symbolic.py
"""

import sympy as sp
import time
import json
from typing import List, Tuple, Dict, Any, Callable, Optional
from dataclasses import dataclass
from ..ir.node import TensorNode
from ..ops.atomic_types import OpType
from ..ops.registry import get_reference_factory
from ..ir.graph import topological_sort
from .shape_inference import ShapeInference
from ..config import DEBUG_EXECUTION

# ==============================================================================
# Constants & Types
# ==============================================================================

# SymPy Infinity
S_INF = sp.oo


@dataclass
class SymbolicRegion:
    """
    Represents the dirty region of a tensor symbolically.
    ranges: A list of (start, stop) SymPy expressions for each dimension.
    is_dirty_expr: A symbolic expression that evaluates to True if the region is dirty.
    """

    ranges: List[Tuple[sp.Expr, sp.Expr]]
    is_dirty_expr: sp.Expr = None

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
    _backward_registry: Dict[str, Callable] = {}
    _cache: Dict[str, Callable] = {}

    @classmethod
    def register(cls, op_type: str):
        def decorator(func):
            cls._registry[op_type] = func
            return func

        return decorator

    @classmethod
    def get_propagator(
        cls, node: TensorNode, known_values: Optional[dict] = None
    ) -> Callable:
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

        attrs_key = json.dumps(node.attrs, sort_keys=True, default=str)

        key = f"{node.op_type}|{input_ranks}|{attrs_key}"

        if key in cls._cache:
            return cls._cache[key]

        # 2. Compile if not found
        func = cls._compile(node, known_values)
        cls._cache[key] = func
        return func

    @classmethod
    def get_backward_propagator(
        cls, node: TensorNode, known_values: Optional[dict] = None
    ) -> Callable:
        """
        Returns a compiled function that calculates input dirty regions
        given output dirty region and input shapes.

        Signature: f(*out_dirty_args, *input_shape_args, *out_shape_args) -> List[float]
        """
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
    def _trace_backward_node(
        cls,
        node: TensorNode,
        output_region: SymbolicRegion,
        context: Dict[str, Any],
        known_values: Optional[dict] = None,
    ) -> List[SymbolicRegion]:
        # 1. Atomic Handler
        if node.op_type in cls._backward_registry:
            return cls._backward_registry[node.op_type](output_region, context)

        # 2. Decomposition
        factory = get_reference_factory(node.op_type)
        if not factory:
            raise ValueError(
                f"No symbolic backward handler or decomposition for {node.op_type}"
            )

        # Instantiate decomposition
        sub_root = factory(node.parents, node.attrs)
        sub_nodes = topological_sort(sub_root)

        # Map sub_root to our output region
        node_map: Dict[TensorNode, SymbolicRegion] = {sub_root: output_region}

        # Perform shape inference on the decomposition subgraph
        ShapeInference.infer(sub_nodes, known_values, keep_cut_parent_shapes=True)

        # Propagate backward in reverse topological order
        for sub in reversed(sub_nodes):
            if sub not in node_map:
                continue

            if sub.op_type == OpType.CONSTANT or not sub.parents:
                continue

            # Use actual shapes from inference
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
                    # Merge existing and new
                    if len(node_map[p].ranges) == len(p_reg.ranges):
                        new_ranges = []
                        for r1, r2 in zip(node_map[p].ranges, p_reg.ranges):
                            new_ranges.append(_merge_dim(r1, r2))
                        node_map[p] = SymbolicRegion(new_ranges)
                    else:
                        # Rank mismatch fallback
                        node_map[p] = p_reg

        # Return regions for the original node parents
        results = []
        for p in node.parents:
            results.append(
                node_map.get(p, cls._clean_region(len(p.shape) if p.shape else 0))
            )

        return results

    @classmethod
    def register_backward(cls, op_type: str):
        def decorator(func):
            cls._backward_registry[op_type] = func
            return func

        return decorator

    @classmethod
    def _compile_backward(
        cls, node: TensorNode, known_values: Optional[dict] = None
    ) -> Callable:
        # 1. Symbols for Output Dirty Region
        out_ranges = []
        out_dirty_args = []
        out_rank = len(node.shape) if node.shape is not None else 0
        for d in range(out_rank):
            s = sp.Symbol(f"out_d{d}_s")
            e = sp.Symbol(f"out_d{d}_e")
            out_ranges.append((s, e))
            out_dirty_args.extend([s, e])
        output_region = SymbolicRegion(out_ranges)

        # 2. Symbols for Input Shapes
        input_shape_symbols = []
        shape_args = []
        for i, parent in enumerate(node.parents):
            rank = len(parent.shape) if parent.shape is not None else 0
            p_dims = []
            for d in range(rank):
                dim = sp.Symbol(f"in{i}_shape_d{d}")
                p_dims.append(dim)
                shape_args.extend([dim])
            input_shape_symbols.append(tuple(p_dims))

        # 3. Symbols for Output Shape
        out_dims = []
        out_shape_args = []
        for d in range(out_rank):
            dim = sp.Symbol(f"out_shape_d{d}")
            out_dims.append(dim)
            out_shape_args.extend([dim])

        context = {
            "input_shapes": input_shape_symbols,
            "output_shape": tuple(out_dims),
            "attrs": node.attrs,
            "op_type": node.op_type,
        }

        # 4. Trace
        input_regions = cls._trace_backward_node(
            node, output_region, context, known_values
        )

        # 5. Flatten Results
        flat_results = []
        for reg in input_regions:
            for s, e in reg.ranges:
                flat_results.append(s)
                flat_results.append(e)

        # 6. Compile to Lambda
        all_args = out_dirty_args + shape_args + out_shape_args

        if not flat_results:
            lam = lambda *args: []
        else:
            if DEBUG_EXECUTION:
                print(f"[SYMBOLIC] _compile_backward running lambdify for node {node.op_type}")
            # Use only "numpy" - including "math" causes Piecewise conversion errors
            start_time = time.perf_counter()
            lam = sp.lambdify(
                all_args, flat_results, modules=["numpy"], cse=True
            )
            end_time = time.perf_counter()
            if DEBUG_EXECUTION:
                print(f"[SYMBOLIC] _compile_backward lambdify for node {node.op_type} took {end_time - start_time} seconds")

        return lam

    @classmethod
    def _compile(
        cls, node: TensorNode, known_values: Optional[dict] = None
    ) -> Callable:
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

            # Initial is_dirty: all dims must have s < e
            is_d = sp.true
            if p_ranges:
                is_d = sp.And(*[s < e for s, e in p_ranges])

            input_symbols.append(SymbolicRegion(p_ranges, is_dirty_expr=is_d))

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

        final_region = cls._trace_node(node, input_symbols, context, known_values)

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
            if DEBUG_EXECUTION:
                print(f"[SYMBOLIC] _compile running lambdify for node {node.op_type}")
            start_time = time.perf_counter()
            lam = sp.lambdify(all_args, out_exprs, modules=["numpy"], cse=True)
            end_time = time.perf_counter()
            if DEBUG_EXECUTION:
                print(f"[SYMBOLIC] _compile lambdify for node {node.op_type} took {end_time - start_time} seconds")

        return lam

    @classmethod
    def _trace_node(
        cls,
        node: TensorNode,
        inputs: List[SymbolicRegion],
        context: Dict[str, Any],
        known_values: Optional[dict] = None,
    ) -> SymbolicRegion:
        # 1. Atomic Handler
        if node.op_type in cls._registry:
            return cls._registry[node.op_type](inputs, context)

        # 2. Decomposition
        factory = get_reference_factory(node.op_type)
        if not factory:
            raise ValueError(f"No symbolic handler or decomposition for {node.op_type}")

        # --- FIX: Create isolated leaf nodes instead of modifying in-place ---
        leaf_parents = [
            TensorNode(
                op_type=p.op_type,
                dtype=p.dtype,
                parents=[],  # Isolate from the rest of the graph
                shape=p.shape,
                name=p.name,  # Keep name for known_values lookup
                attrs=p.attrs,
                backend=p.backend,
                storage_type=p.storage_type,
            )
            for p in node.parents
        ]

        # Instantiate decomposition with isolated inputs
        sub_root = factory(leaf_parents, node.attrs)
        sub_nodes = topological_sort(sub_root)

        # Perform shape inference on the isolated decomposition subgraph
        ShapeInference.infer(sub_nodes, known_values, keep_cut_parent_shapes=True)

        # Map leaf parents to their symbolic input regions
        node_map: Dict[TensorNode, SymbolicRegion] = {}
        for lp, p_sym in zip(leaf_parents, inputs):
            node_map[lp] = p_sym

        # Propagate through subgraph
        for sub in sub_nodes:
            if sub in node_map:
                continue

            if sub.op_type == OpType.CONSTANT:
                rank = len(sub.shape) if sub.shape else 0
                node_map[sub] = cls._clean_region(rank)
                continue
            if sub.op_type == OpType.INPUT:
                pass

            sub_inputs = [node_map[p] for p in sub.parents]

            # Use actual shapes from inference
            sub_ctx = {
                "attrs": sub.attrs,
                "input_shapes": [p.shape or () for p in sub.parents],
                "output_shape": sub.shape or (),
                "op_type": sub.op_type,
            }

            res = cls._trace_node(sub, sub_inputs, sub_ctx, known_values)
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
    if region.is_dirty_expr is not None:
        return region.is_dirty_expr

    if not region.ranges:
        return sp.true  # Scalar dirty (fallback)

    # A region is dirty if ALL dimensions have start < end
    # (Intersection of non-empty intervals)
    conditions = [s < e for s, e in region.ranges]
    return sp.And(*conditions)


# ==============================================================================
# Atomic Handlers
# ==============================================================================


@SymbolicPropagator.register(OpType.ARANGE)
def symbolic_arange(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    """
    Arange operation: arange(start, stop, step) -> [start, start+step, ..., stop-1]
    Output is 1D with shape (ceil((stop-start)/step),)
    """
    if len(inputs) < 3:
        return SymbolicPropagator._clean_region(1)

    start_region, stop_region, step_region = inputs[0], inputs[1], inputs[2]

    # For arange, the output is dirty if any input is dirty
    # (since the output values depend on start/stop/step)
    is_dirty = sp.Or(
        _is_dirty(start_region), _is_dirty(stop_region), _is_dirty(step_region)
    )

    # Output is always 1D
    # The dirty region is (0, output_length) if dirty, else clean (inf, -inf)
    out_s = sp.Piecewise((sp.Integer(0), is_dirty), (S_INF, True))
    out_e = sp.Piecewise((S_INF, is_dirty), (-S_INF, True))

    return SymbolicRegion([(out_s, out_e)], is_dirty_expr=is_dirty)


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

    input_shapes = ctx.get("input_shapes", [])

    # Align ranks (broadcasting)
    max_rank = max(len(inp.ranges) for inp in inputs)
    aligned_ranges = []

    for i, inp in enumerate(inputs):
        ranges = list(inp.ranges)

        # Handle size-1 broadcasting in existing dims
        if i < len(input_shapes):
            p_shape = input_shapes[i]
            is_d = _is_dirty(inp)
            for d in range(len(ranges)):
                s, e = ranges[d]
                dim_size = p_shape[d]
                # If this dimension has size 1, it broadcasts to full if dirty
                s_final = sp.Piecewise(
                    (sp.Integer(0), sp.And(is_d, sp.Eq(dim_size, 1))), (s, True)
                )
                e_final = sp.Piecewise(
                    (S_INF, sp.And(is_d, sp.Eq(dim_size, 1))), (e, True)
                )
                ranges[d] = (s_final, e_final)

        pad = max_rank - len(ranges)
        if pad > 0:
            # If input is dirty, broadcasted dimensions are full (0, inf)
            is_d = _is_dirty(inp)
            pad_range = (
                sp.Piecewise((sp.Integer(0), is_d), (S_INF, True)),
                sp.Piecewise((S_INF, is_d), (-S_INF, True)),
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

    # Output is dirty if ANY input is dirty
    out_is_d = sp.false
    if inputs:
        out_is_d = sp.Or(*[_is_dirty(inp) for inp in inputs])

    return SymbolicRegion(out_ranges, is_dirty_expr=out_is_d)


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

    return SymbolicRegion(
        batch_ranges + [(out_m_s, out_m_e), (out_n_s, out_n_e)],
        is_dirty_expr=sp.Or(_is_dirty(rA), _is_dirty(rB)),
    )


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
        fin_e = sp.Piecewise((-S_INF, is_empty), (sp.Max(0, raw_e), True))

        out_ranges.append((fin_s, fin_e))

    # For slice, we re-evaluate dirty state from output ranges
    # because slicing can make a dirty region clean.
    out_is_d = sp.And(*[s < e for s, e in out_ranges]) if out_ranges else sp.true
    return SymbolicRegion(out_ranges, is_dirty_expr=out_is_d)


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
        current = (S_INF, -S_INF)
        for inp in inputs:
            current = _merge_dim(current, inp.ranges[d])
        out_non_concat.append(current)

    concat_axis_range = (S_INF, -S_INF)
    current_offset = sp.Integer(0)

    for i, inp in enumerate(inputs):
        s, e = inp.ranges[axis]
        is_d = s < e

        s_shifted = s + current_offset
        e_shifted = e + current_offset

        s_term = sp.Piecewise((s_shifted, is_d), (S_INF, True))
        e_term = sp.Piecewise((e_shifted, is_d), (-S_INF, True))

        concat_axis_range = _merge_dim(concat_axis_range, (s_term, e_term))
        current_offset += input_shapes[i][axis]

    final_ranges = []
    for d in range(rank):
        if d == axis:
            final_ranges.append(concat_axis_range)
        else:
            final_ranges.append(out_non_concat[d])

    out_is_d = sp.Or(*[_is_dirty(inp) for inp in inputs])
    return SymbolicRegion(final_ranges, is_dirty_expr=out_is_d)


@SymbolicPropagator.register(OpType.RESHAPE)
def symbolic_reshape(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    # Conservative Full Dirty
    return SymbolicRegion([(sp.Integer(0), S_INF)], is_dirty_expr=_is_dirty(inputs[0]))


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

    return SymbolicRegion(new_ranges, is_dirty_expr=_is_dirty(inp))


@SymbolicPropagator.register(OpType.GATHER)
def symbolic_gather(
    inputs: List[SymbolicRegion], ctx: Dict[str, Any]
) -> SymbolicRegion:
    """
    Gather operation: output[indices_shape, data_trailing_shape]
    data: (N, ...)
    indices: (M, ...)
    output: (M, ..., ...)
    """
    if len(inputs) < 2:
        return SymbolicPropagator._clean_region(0)

    data_region = inputs[0]
    ind_region = inputs[1]

    # Check if anything in indices is dirty
    is_ind_dirty = _is_dirty(ind_region)
    # Check if gathered axis of data is dirty
    gathered_axis_dirty = data_region.ranges[0][0] < data_region.ranges[0][1]

    out_ranges = []
    # Output dims corresponding to indices
    for s_in, e_in in ind_region.ranges:
        # This dimension of output is dirty if:
        # a) this dimension of indices is dirty
        # b) OR the gathered axis of data is dirty (making all indices potentially dirty)
        dim_dirty = s_in < e_in
        s = sp.Piecewise(
            (s_in, dim_dirty), (sp.Integer(0), gathered_axis_dirty), (S_INF, True)
        )
        e = sp.Piecewise(
            (e_in, dim_dirty), (S_INF, gathered_axis_dirty), (-S_INF, True)
        )
        out_ranges.append((s, e))

    # Output dims corresponding to data trailing dims
    for sd, ed in data_region.ranges[1:]:
        # These dims are dirty if they are dirty in data OR if indices shifted
        dim_dirty = sd < ed
        s = sp.Piecewise((sd, dim_dirty), (sp.Integer(0), is_ind_dirty), (S_INF, True))
        e = sp.Piecewise((ed, dim_dirty), (S_INF, is_ind_dirty), (-S_INF, True))
        out_ranges.append((s, e))

    return SymbolicRegion(out_ranges)


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
                e_kd = sp.Piecewise((sp.Integer(1), is_inp_dirty), (-S_INF, True))
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
    inp = inputs[0]
    return SymbolicRegion(inp.ranges, is_dirty_expr=_is_dirty(inp))


# ==============================================================================
# Backward Atomic Handlers
# ==============================================================================


@SymbolicPropagator.register_backward(OpType.ARANGE)
def backward_arange(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    """
    Backward for ARANGE: Since arange generates values from start/stop/step,
    if the output is dirty, all inputs must be considered dirty (full range).
    """
    input_shapes = ctx.get("input_shapes", [])
    is_out_dirty = _is_dirty(out_region)

    # ARANGE has 3 inputs: start, stop, step
    # Each is typically a scalar (rank 0 or shape [1])
    # If output is dirty, all inputs are dirty (full range 0 to 1 for scalars)
    results = []
    for p_shape in input_shapes:
        p_rank = len(p_shape) if p_shape else 0
        if p_rank == 0:
            # Scalar input
            s = sp.Piecewise((sp.Integer(0), is_out_dirty), (S_INF, True))
            e = sp.Piecewise((sp.Integer(1), is_out_dirty), (-S_INF, True))
            results.append(SymbolicRegion([(s, e)]))
        else:
            # Non-scalar input - use full dirty region
            results.append(SymbolicPropagator._full_dirty_region(p_rank))

    return results


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

        # Take trailing dimensions
        p_ranges = list(out_ranges[max(0, out_rank - p_rank) :])

        # Pad with full slices if p_rank > out_rank
        if len(p_ranges) < p_rank:
            p_ranges = [(sp.Integer(0), S_INF)] * (p_rank - len(p_ranges)) + p_ranges

        # Handle broadcasting (dim size 1)
        final_p_ranges = []
        for d in range(p_rank):
            s, e = p_ranges[d]
            dim_size = p_shape[d]
            # If dim_size == 1, input is full (0, 1) if output dirty
            # Otherwise use s, e
            s_final = sp.Piecewise((sp.Integer(0), sp.Eq(dim_size, 1)), (s, True))
            e_final = sp.Piecewise((sp.Integer(1), sp.Eq(dim_size, 1)), (e, True))
            final_p_ranges.append((s_final, e_final))

        results.append(SymbolicRegion(final_p_ranges))
    return results


@SymbolicPropagator.register_backward(OpType.DOT)
def backward_dot(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    out_ranges = out_region.ranges
    out_rank = len(out_ranges)
    input_shapes = ctx["input_shapes"]

    # A: (..., M, K), B: (..., K, N) -> (..., M, N)
    m_range = out_ranges[-2] if out_rank >= 2 else out_ranges[0]
    n_range = out_ranges[-1] if out_rank >= 2 else (sp.Integer(0), S_INF)

    # Batch dims
    batch_out = out_ranges[:-2] if out_rank > 2 else []

    def get_p_batch(p_shape, b_out):
        p_rank = len(p_shape)
        p_batch_rank = max(0, p_rank - 2)
        if p_batch_rank == 0:
            return []

        p_b_ranges = list(b_out[max(0, len(b_out) - p_batch_rank) :])
        if len(p_b_ranges) < p_batch_rank:
            p_b_ranges = [(sp.Integer(0), S_INF)] * (
                p_batch_rank - len(p_b_ranges)
            ) + p_b_ranges

        final = []
        for d in range(p_batch_rank):
            s, e = p_b_ranges[d]
            dim_size = p_shape[d]
            s_f = sp.Piecewise((sp.Integer(0), sp.Eq(dim_size, 1)), (s, True))
            e_f = sp.Piecewise((dim_size, sp.Eq(dim_size, 1)), (e, True))
            final.append((s_f, e_f))
        return final

    ranges_a = get_p_batch(input_shapes[0], batch_out) + [
        m_range,
        (sp.Integer(0), S_INF),
    ]
    ranges_b = get_p_batch(input_shapes[1], batch_out) + [
        (sp.Integer(0), S_INF),
        n_range,
    ]

    return [SymbolicRegion(ranges_a), SymbolicRegion(ranges_b)]


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
    for i, p_shape in enumerate(input_shapes):
        p_dim = p_shape[axis]
        p_start = curr_offset
        p_end = curr_offset + p_dim

        ov_start = sp.Max(out_s, p_start)
        ov_end = sp.Min(out_e, p_end)

        is_empty = ov_start >= ov_end

        rel_start = ov_start - p_start
        rel_end = ov_end - p_start

        fin_s = sp.Piecewise((sp.Integer(0), is_empty), (sp.Max(0, rel_start), True))
        fin_e = sp.Piecewise((sp.Integer(0), is_empty), (sp.Max(0, rel_end), True))

        p_ranges = list(out_ranges)
        p_ranges[axis] = (fin_s, fin_e)
        results.append(SymbolicRegion(p_ranges))

        curr_offset += p_dim

    return results


@SymbolicPropagator.register_backward(OpType.SLICE)
def backward_slice(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    out_ranges = out_region.ranges
    attrs = ctx.get("attrs", {})
    starts = attrs.get("starts", [])
    steps = attrs.get("steps", [])

    input_shape = ctx["input_shapes"][0]
    in_ranges = []

    for i in range(len(input_shape)):
        os, oe = out_ranges[i] if i < len(out_ranges) else (sp.Integer(0), S_INF)
        st = sp.Integer(starts[i]) if i < len(starts) else sp.Integer(0)
        step = sp.Integer(steps[i]) if i < len(steps) else sp.Integer(1)

        is_empty = os >= oe

        in_s = st + os * step
        # For slice, if output has N elements, input range spans st + os*step to st + (oe-1)*step + 1
        in_e = st + (oe - 1) * step + 1

        in_ranges.append(
            (
                sp.Piecewise((sp.Integer(0), is_empty), (sp.Max(0, in_s), True)),
                sp.Piecewise((sp.Integer(0), is_empty), (sp.Max(0, in_e), True)),
            )
        )

    return [SymbolicRegion(in_ranges)]


@SymbolicPropagator.register_backward(OpType.RESHAPE)
def backward_reshape(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    input_shapes = ctx["input_shapes"]
    return [SymbolicPropagator._full_dirty_region(len(p)) for p in input_shapes]


@SymbolicPropagator.register_backward(OpType.PERMUTE)
def backward_permute(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    dims = ctx.get("attrs", {}).get("dims", [])
    out_ranges = out_region.ranges
    if not dims:
        return [out_region]

    # Find inverse permutation
    p_inv = [0] * len(dims)
    for i, d in enumerate(dims):
        p_inv[d] = i

    in_ranges = [out_ranges[p_inv[j]] for j in range(len(dims))]
    return [SymbolicRegion(in_ranges)]


@SymbolicPropagator.register_backward(OpType.GATHER)
def backward_gather(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    # data, indices = inputs
    # To compute out_region, we need matching indices
    # The first N dimensions of out_region correspond to indices
    input_shapes = ctx.get("input_shapes", [])
    if not input_shapes:
        return [SymbolicPropagator._full_dirty_region(1), out_region]

    indices_rank = len(input_shapes[1])
    indices_region = SymbolicRegion(out_region.ranges[:indices_rank])

    # data needs its gathered axis full, and trailing dims matching out_region
    data_rank = len(input_shapes[0])
    data_ranges = [(sp.Integer(0), S_INF)]  # Full gathered axis
    data_ranges.extend(out_region.ranges[indices_rank:])

    return [SymbolicRegion(data_ranges), indices_region]


@SymbolicPropagator.register_backward(OpType.SUM)
@SymbolicPropagator.register_backward(OpType.MAX)
def backward_reduce(
    out_region: SymbolicRegion, ctx: Dict[str, Any]
) -> List[SymbolicRegion]:
    input_shapes = ctx.get("input_shapes", [])
    if not input_shapes:
        return [SymbolicPropagator._full_dirty_region(1)]

    p_shape = input_shapes[0]
    rank_in = len(p_shape)
    attrs = ctx.get("attrs", {})
    axis = attrs.get("axis", None)
    keepdims = attrs.get("keepdims", True)

    if axis is None:
        axes = list(range(rank_in))
    elif isinstance(axis, (list, tuple)):
        axes = [a if a >= 0 else a + rank_in for a in axis]
    else:
        axes = [axis if axis >= 0 else axis + rank_in]

    in_ranges = []
    out_idx = 0
    for d in range(rank_in):
        if d in axes:
            # Need whole dimension
            in_ranges.append((sp.Integer(0), S_INF))
            if keepdims:
                out_idx += 1
        else:
            # Need matching slice
            if out_idx < len(out_region.ranges):
                in_ranges.append(out_region.ranges[out_idx])
            else:
                in_ranges.append((sp.Integer(0), S_INF))
            out_idx += 1

    return [SymbolicRegion(in_ranges)]
