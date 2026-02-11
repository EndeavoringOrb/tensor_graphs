"""
Unified Graph Propagation

Handles both shape inference and dirty region propagation through a single
per-op handler registry. Each op registers a PropagationHandler that computes:
  1. Output shape from input shapes + attrs
  2. Forward dirty region from input dirty regions
  3. Backward dirty regions (output → required input regions)

Decomposition fallback is shared: if no handler is registered for an op,
the reference factory is used to decompose it into a subgraph, and propagation
recurses through that subgraph.
"""

from typing import Dict, List, Tuple, Optional, Callable, Any, NamedTuple
import math
import numpy as np

from ..ir.node import TensorNode
from ..ir.dtypes import DType, TensorSignature, Backend
from ..ir.graph import topological_sort
from ..ops.atomic_types import OpType
from ..ops.registry import get_reference_factory
from ..backend.registry import KernelRegistry

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A dirty region is None (clean) or a tuple of (start, stop) per dimension.
NumericRegion = Optional[Tuple[Tuple[int, int], ...]]


class PropagationHandler(NamedTuple):
    """Bundle of callbacks for a single op type."""

    shape: Optional[Callable] = None  # (node, get_val) -> None  (mutates node.shape)
    forward: Optional[Callable] = (
        None  # (input_regions, input_shapes, output_shape, attrs) -> NumericRegion
    )
    backward: Optional[Callable] = (
        None  # (output_region, input_shapes, output_shape, attrs) -> List[NumericRegion]
    )


# ---------------------------------------------------------------------------
# Region helpers
# ---------------------------------------------------------------------------


def _is_clean(region: NumericRegion) -> bool:
    if region is None:
        return True
    return all(start >= stop for start, stop in region)


def _make_full(shape: Tuple[int, ...]) -> NumericRegion:
    if not shape:
        return None
    return tuple((0, dim) for dim in shape)


def _merge_regions(
    r1: NumericRegion, r2: NumericRegion, shape: Tuple[int, ...]
) -> NumericRegion:
    if r1 is None and r2 is None:
        return None
    if r1 is None:
        return r2
    if r2 is None:
        return r1
    return tuple((min(a[0], b[0]), max(a[1], b[1])) for a, b in zip(r1, r2))


def _to_slices(region: NumericRegion) -> Optional[Tuple[slice, ...]]:
    if region is None:
        return None
    if all(start >= stop for start, stop in region):
        return None
    return tuple(slice(start, stop) for start, stop in region)


def _from_slices(
    slices: Optional[Tuple[slice, ...]], shape: Tuple[int, ...]
) -> NumericRegion:
    if slices is None:
        return None
    result = []
    for i, s in enumerate(slices):
        dim = shape[i] if i < len(shape) else 1
        start, stop, _ = s.indices(dim)
        result.append((start, stop))
    return tuple(result)


# ---------------------------------------------------------------------------
# Shared shape utilities
# ---------------------------------------------------------------------------


def _broadcast_shapes(
    shape1: Tuple[Optional[int], ...], shape2: Tuple[Optional[int], ...]
) -> Tuple[Optional[int], ...]:
    ndim1, ndim2 = len(shape1), len(shape2)
    out_ndim = max(ndim1, ndim2)
    s1 = (1,) * (out_ndim - ndim1) + shape1
    s2 = (1,) * (out_ndim - ndim2) + shape2
    out_shape = []
    for d1, d2 in zip(s1, s2):
        if d1 == 1:
            out_shape.append(d2)
        elif d2 == 1:
            out_shape.append(d1)
        elif d1 == d2:
            out_shape.append(d1)
        elif d1 is None or d2 is None:
            out_shape.append(None)
        else:
            out_shape.append(d1)
    return tuple(out_shape)


def _prod_shape(shape: Tuple[Optional[int], ...]) -> Optional[int]:
    if not shape:
        return 1
    result = 1
    for d in shape:
        if d is None:
            return None
        result *= d
    return result


def _map_dtype_np(dtype: DType) -> Any:
    return {
        DType.FP32: np.float32,
        DType.INT32: np.int32,
        DType.FP16: np.float16,
        DType.BOOL: np.bool_,
    }.get(dtype, np.float32)


# ---------------------------------------------------------------------------
# GraphPropagator – unified registry and dispatch
# ---------------------------------------------------------------------------


class GraphPropagator:
    """
    Central registry that stores per-op shape, forward-dirty, and backward-dirty
    handlers and exposes three entry-points:

    * ``infer_shapes``   – static shape inference over a topo-sorted node list
    * ``propagate``      – forward dirty region for one node
    * ``get_input_slices`` – backward dirty region for one node
    """

    _handlers: Dict[str, PropagationHandler] = {}

    # -- registration helpers ------------------------------------------------

    @classmethod
    def _ensure(cls, op_type: str) -> PropagationHandler:
        if op_type not in cls._handlers:
            cls._handlers[op_type] = PropagationHandler()
        return cls._handlers[op_type]

    @classmethod
    def register_shape(cls, *op_types: str):
        """Decorator: register a shape handler for one or more op types."""

        def decorator(func):
            for op in op_types:
                old = cls._ensure(op)
                cls._handlers[op] = old._replace(shape=func)
            return func

        return decorator

    @classmethod
    def register_forward(cls, *op_types: str):
        """Decorator: register a forward dirty-region handler."""

        def decorator(func):
            for op in op_types:
                old = cls._ensure(op)
                cls._handlers[op] = old._replace(forward=func)
            return func

        return decorator

    @classmethod
    def register_backward(cls, *op_types: str):
        """Decorator: register a backward dirty-region handler."""

        def decorator(func):
            for op in op_types:
                old = cls._ensure(op)
                cls._handlers[op] = old._replace(backward=func)
            return func

        return decorator

    # -- shape inference -----------------------------------------------------

    @classmethod
    def infer_shapes(
        cls,
        nodes: List[TensorNode],
        known_values: Dict[str, Any],
        keep_cut_parent_shapes: bool = False,
    ):
        """
        Update ``node.shape`` in-place for every node in *nodes*
        (which must be in topological order).
        """
        computed_values = dict(known_values)

        def get_val(node):
            if node.name in computed_values:
                return computed_values[node.name]
            if node.op_type == OpType.CONSTANT:
                val = node.attrs.get("value")
                if isinstance(val, (int, float, bool)):
                    val = np.array(val)
                elif isinstance(val, (list, tuple)):
                    val = np.array(val)
                computed_values[node.name] = val
                return val
            return None

        for node in nodes:
            # Normalize existing shape
            if node.shape is not None:
                node.shape = tuple(
                    d if isinstance(d, int) else None for d in node.shape
                )

            # Skip pre-filled leaf shapes when requested
            if (
                keep_cut_parent_shapes
                and node.shape
                and all(isinstance(d, int) for d in node.shape)
                and not node.parents
            ):
                continue

            # Dispatch to registered shape handler
            handler = cls._handlers.get(node.op_type)
            if handler and handler.shape:
                handler.shape(node, get_val)
            elif node.op_type not in (OpType.INPUT, OpType.CONSTANT):
                # Decomposition fallback
                factory = get_reference_factory(node.op_type)
                if factory:
                    sub_root = factory(node.parents, node.attrs)
                    sub_nodes = topological_sort(sub_root)
                    cls.infer_shapes(sub_nodes, computed_values)
                    node.shape = sub_root.shape
                elif node.parents and node.parents[0].shape:
                    node.shape = node.parents[0].shape

            if node.shape is None or any(item is None for item in node.shape):
                raise ValueError(f"Cannot resolve shape for node {node}")

            # Value resolution for shape-relevant ops
            if node.op_type in (
                OpType.ADD,
                OpType.MUL,
                OpType.CONCAT,
                OpType.CAST,
                OpType.SLICE,
                OpType.RESHAPE,
            ):
                if all(p.name in computed_values for p in node.parents):
                    input_sigs = [
                        TensorSignature(p.dtype, p.shape, Backend.CPU_NUMPY)
                        for p in node.parents
                    ]
                    kernel = KernelRegistry.select_best_kernel(
                        node.op_type, input_sigs, Backend.CPU_NUMPY, node.dtype
                    )
                    if kernel:
                        out_np = np.zeros(node.shape, dtype=_map_dtype_np(node.dtype))
                        parent_vals = [computed_values[p.name] for p in node.parents]
                        kernel(parent_vals, [out_np], node.attrs)
                        computed_values[node.name] = out_np

    # -- forward dirty propagation -------------------------------------------

    @classmethod
    def propagate(
        cls,
        node: TensorNode,
        known_values: Optional[dict] = None,
    ) -> NumericRegion:
        if not node.parents:
            return _from_slices(node.dirty_region, node.shape or ())

        input_regions = [
            _from_slices(p.dirty_region, p.shape or ()) for p in node.parents
        ]
        input_shapes = [p.shape or () for p in node.parents]
        output_shape = node.shape or ()

        handler = cls._handlers.get(node.op_type)
        if handler and handler.forward:
            return handler.forward(
                input_regions, input_shapes, output_shape, node.attrs
            )

        return cls._propagate_subgraph(node, input_regions, known_values)

    @classmethod
    def _propagate_subgraph(
        cls,
        node: TensorNode,
        input_regions: List[NumericRegion],
        known_values: Optional[dict],
    ) -> NumericRegion:
        factory = get_reference_factory(node.op_type)
        if not factory:
            raise ValueError(f"No handler or decomposition for {node.op_type}")

        leaf_parents = [
            TensorNode(
                op_type=p.op_type,
                dtype=p.dtype,
                parents=[],
                shape=p.shape,
                name=f"_leaf_{i}",
                attrs=p.attrs,
                backend=p.backend,
                storage_type=p.storage_type,
            )
            for i, p in enumerate(node.parents)
        ]
        sub_root = factory(leaf_parents, node.attrs)

        if sub_root.op_type == node.op_type and len(sub_root.parents) == len(
            node.parents
        ):
            raise NotImplementedError(
                f"Atomic op {node.op_type} has no forward propagation handler"
            )

        sub_nodes = topological_sort(sub_root)
        cls.infer_shapes(sub_nodes, known_values or {}, keep_cut_parent_shapes=True)

        region_map: Dict[int, NumericRegion] = {
            id(lp): reg for lp, reg in zip(leaf_parents, input_regions)
        }

        for sub in sub_nodes:
            nid = id(sub)
            if nid in region_map:
                continue
            if sub.op_type == OpType.CONSTANT:
                region_map[nid] = None
                continue

            sub_in = [region_map[id(p)] for p in sub.parents]
            sub_in_shapes = [p.shape or () for p in sub.parents]
            sub_out_shape = sub.shape or ()

            h = cls._handlers.get(sub.op_type)
            if h and h.forward:
                region_map[nid] = h.forward(
                    sub_in, sub_in_shapes, sub_out_shape, sub.attrs
                )
            else:
                region_map[nid] = cls._propagate_subgraph(sub, sub_in, known_values)

        return region_map[id(sub_root)]

    # -- backward dirty propagation ------------------------------------------

    @classmethod
    def get_input_slices(
        cls,
        node: TensorNode,
        output_region: NumericRegion,
        known_values: Optional[dict] = None,
    ) -> List[NumericRegion]:
        if output_region is None or _is_clean(output_region):
            return [None] * len(node.parents)

        input_shapes = [p.shape or () for p in node.parents]
        output_shape = node.shape or ()

        handler = cls._handlers.get(node.op_type)
        if handler and handler.backward:
            return handler.backward(
                output_region, input_shapes, output_shape, node.attrs
            )

        return cls._backward_subgraph(node, output_region, known_values)

    @classmethod
    def _backward_subgraph(
        cls,
        node: TensorNode,
        output_region: NumericRegion,
        known_values: Optional[dict],
    ) -> List[NumericRegion]:
        factory = get_reference_factory(node.op_type)
        if not factory:
            raise ValueError(f"No backward handler or decomposition for {node.op_type}")

        leaf_parents = [
            TensorNode(
                op_type=p.op_type,
                dtype=p.dtype,
                parents=[],
                shape=p.shape,
                name=f"_leaf_{i}",
                attrs=p.attrs,
                backend=p.backend,
                storage_type=p.storage_type,
            )
            for i, p in enumerate(node.parents)
        ]
        sub_root = factory(leaf_parents, node.attrs)

        if sub_root.op_type == node.op_type and len(sub_root.parents) == len(
            node.parents
        ):
            raise NotImplementedError(
                f"Atomic op {node.op_type} has no backward propagation handler"
            )

        sub_nodes = topological_sort(sub_root)
        cls.infer_shapes(sub_nodes, known_values or {}, keep_cut_parent_shapes=True)

        region_map: Dict[int, NumericRegion] = {id(sub_root): output_region}

        for sub in reversed(sub_nodes):
            nid = id(sub)
            if nid not in region_map:
                continue
            if sub.op_type == OpType.CONSTANT or not sub.parents:
                continue

            sub_out_region = region_map[nid]
            sub_in_shapes = [p.shape or () for p in sub.parents]
            sub_out_shape = sub.shape or ()

            h = cls._handlers.get(sub.op_type)
            if h and h.backward:
                sub_in_regions = h.backward(
                    sub_out_region, sub_in_shapes, sub_out_shape, sub.attrs
                )
            else:
                sub_in_regions = cls._backward_subgraph(
                    sub, sub_out_region, known_values
                )

            for p, p_reg in zip(sub.parents, sub_in_regions):
                pid = id(p)
                if pid not in region_map:
                    region_map[pid] = p_reg
                else:
                    region_map[pid] = _merge_regions(
                        region_map[pid], p_reg, p.shape or ()
                    )

        return [region_map.get(id(lp), None) for lp in leaf_parents]


# ===========================================================================
# Op handlers – shape, forward, backward
# ===========================================================================

# ---------------------------------------------------------------------------
# INPUT / CONSTANT
# ---------------------------------------------------------------------------


@GraphPropagator.register_shape(OpType.INPUT)
def _shape_input(node: TensorNode, get_val):
    val = get_val(node)
    if val is not None and hasattr(val, "shape"):
        node.shape = tuple(int(d) for d in val.shape)
        return
    if node.shape:
        node.shape = tuple(d if isinstance(d, int) else None for d in node.shape)
    else:
        node.shape = None


@GraphPropagator.register_shape(OpType.CONSTANT)
def _shape_constant(node: TensorNode, get_val):
    val = get_val(node)
    if val is not None:
        if hasattr(val, "shape"):
            node.shape = tuple(int(d) for d in val.shape)
        elif isinstance(val, (list, tuple)):
            node.shape = (len(val),)
        else:
            node.shape = None
    elif node.shape is None:
        node.shape = ()
    else:
        node.shape = tuple(d if isinstance(d, int) else None for d in node.shape)


# ---------------------------------------------------------------------------
# Elementwise (broadcast) ops – ADD, MUL, DIVIDE, POWER, WHERE
# ---------------------------------------------------------------------------

_ELEMENTWISE_OPS = (OpType.ADD, OpType.MUL, OpType.DIVIDE, OpType.POWER, OpType.WHERE)


@GraphPropagator.register_shape(*_ELEMENTWISE_OPS)
def _shape_broadcast(node: TensorNode, get_val):
    shapes = [p.shape for p in node.parents if p.shape is not None]
    if not shapes:
        return
    current = shapes[0]
    for s in shapes[1:]:
        current = _broadcast_shapes(current, s)
    node.shape = current


@GraphPropagator.register_forward(*_ELEMENTWISE_OPS)
def _fwd_elementwise(input_regions, input_shapes, output_shape, attrs):
    if all(_is_clean(r) for r in input_regions):
        return None
    out_rank = len(output_shape)
    if out_rank == 0:
        return ()

    result = [(output_shape[d], 0) for d in range(out_rank)]
    has_dirty = False

    for reg, shape in zip(input_regions, input_shapes):
        if _is_clean(reg):
            continue
        has_dirty = True
        in_rank = len(shape)
        pad = out_rank - in_rank

        for d in range(in_rank):
            out_d = d + pad
            start, stop = reg[d]
            if shape[d] == 1:
                start, stop = 0, output_shape[out_d]
            result[out_d] = (min(result[out_d][0], start), max(result[out_d][1], stop))

        for d in range(pad):
            result[d] = (0, output_shape[d])

    return tuple(result) if has_dirty else None


@GraphPropagator.register_backward(*_ELEMENTWISE_OPS)
def _bwd_elementwise(output_region, input_shapes, output_shape, attrs):
    if _is_clean(output_region):
        return [None] * len(input_shapes)
    out_rank = len(output_shape)
    results = []
    for in_shape in input_shapes:
        in_rank = len(in_shape)
        if in_rank == 0:
            results.append(None)
            continue
        pad = out_rank - in_rank
        in_region = []
        for d in range(in_rank):
            out_d = d + pad
            start, stop = output_region[out_d]
            if in_shape[d] == 1:
                in_region.append((0, 1))
            else:
                in_region.append((start, stop))
        results.append(tuple(in_region))
    return results


# ---------------------------------------------------------------------------
# Unary (shape-preserving) ops
# ---------------------------------------------------------------------------

_UNARY_OPS = (
    OpType.CAST,
    OpType.COPY_TO,
    OpType.NEGATE,
    OpType.EXP,
    OpType.SIN,
    OpType.COS,
    OpType.SQRT,
    OpType.TRIU,
)


@GraphPropagator.register_shape(*_UNARY_OPS)
def _shape_unary(node: TensorNode, get_val):
    if node.parents and node.parents[0].shape is not None:
        node.shape = node.parents[0].shape


@GraphPropagator.register_forward(*_UNARY_OPS)
def _fwd_unary(input_regions, input_shapes, output_shape, attrs):
    return input_regions[0] if input_regions else None


@GraphPropagator.register_backward(*_UNARY_OPS)
def _bwd_unary(output_region, input_shapes, output_shape, attrs):
    return [output_region]


# ---------------------------------------------------------------------------
# SLICE
# ---------------------------------------------------------------------------


@GraphPropagator.register_shape(OpType.SLICE)
def _shape_slice(node: TensorNode, get_val):
    if not node.parents or node.parents[0].shape is None:
        return
    data_shape = node.parents[0].shape
    starts = node.attrs.get("starts", [])
    ends = node.attrs.get("ends", [])
    steps = node.attrs.get("steps", [])

    new_shape = []
    for i, dim in enumerate(data_shape):
        if dim is None:
            new_shape.append(None)
            continue

        s = int(starts[i]) if i < len(starts) and starts[i] is not None else None
        e = int(ends[i]) if i < len(ends) and ends[i] is not None else None
        st = int(steps[i]) if i < len(steps) and steps[i] is not None else 1

        # Let Python compute the canonical (start, stop, step) for this dim
        start, stop, step = slice(s, e, st).indices(dim)
        new_shape.append(len(range(start, stop, step)))

    node.shape = tuple(new_shape)


@GraphPropagator.register_forward(OpType.SLICE)
def _fwd_slice(input_regions, input_shapes, output_shape, attrs):
    inp = input_regions[0]
    if _is_clean(inp):
        return None
    starts = attrs.get("starts", [])
    ends = attrs.get("ends", [])
    steps = attrs.get("steps", [])
    in_shape = input_shapes[0]
    result = []
    for d in range(len(output_shape)):
        in_start, in_stop = inp[d] if d < len(inp) else (0, in_shape[d])
        sl_start = starts[d] if d < len(starts) else 0
        sl_end = ends[d] if d < len(ends) and ends[d] is not None else in_shape[d]
        sl_step = steps[d] if d < len(steps) and steps[d] is not None else 1
        ov_start = max(in_start, sl_start)
        ov_end = min(in_stop, sl_end)
        if ov_start >= ov_end:
            result.append((0, 0))
        else:
            out_start = (ov_start - sl_start + sl_step - 1) // sl_step
            out_end = (ov_end - sl_start + sl_step - 1) // sl_step
            result.append((max(0, out_start), min(out_end, output_shape[d])))
    if all(s >= e for s, e in result):
        return None
    return tuple(result)


@GraphPropagator.register_backward(OpType.SLICE)
def _bwd_slice(output_region, input_shapes, output_shape, attrs):
    return [None]


# ---------------------------------------------------------------------------
# CONCAT
# ---------------------------------------------------------------------------


@GraphPropagator.register_shape(OpType.CONCAT)
def _shape_concat(node: TensorNode, get_val):
    if not node.parents or any(p.shape is None for p in node.parents):
        return
    axis = node.attrs.get("axis", 0)
    shapes = [p.shape for p in node.parents]
    if len(shapes) < 2:
        if len(shapes) == 1:
            node.shape = shapes[0]
        return
    rank = len(shapes[0])
    if axis < 0:
        axis += rank
    if 0 <= axis < rank:
        new_shape = list(shapes[0])
        total_dim = shapes[0][axis]
        for s in shapes[1:]:
            if len(s) == rank:
                other_dim = s[axis]
                if total_dim is not None and other_dim is not None:
                    total_dim += other_dim
                else:
                    total_dim = None
                new_shape[axis] = total_dim
        node.shape = tuple(new_shape)


@GraphPropagator.register_forward(OpType.CONCAT)
def _fwd_concat(input_regions, input_shapes, output_shape, attrs):
    axis = attrs.get("axis", 0)
    rank = len(output_shape)
    if axis < 0:
        axis += rank
    if all(_is_clean(r) for r in input_regions):
        return None
    result = [(output_shape[d], 0) for d in range(rank)]
    current_offset = 0
    for reg, shape in zip(input_regions, input_shapes):
        if not _is_clean(reg):
            for d in range(rank):
                start, stop = reg[d]
                if d == axis:
                    start += current_offset
                    stop += current_offset
                result[d] = (min(result[d][0], start), max(result[d][1], stop))
        current_offset += shape[axis]
    if all(r[0] >= r[1] for r in result):
        return None
    return tuple(
        (max(0, r[0]), min(r[1], output_shape[i])) for i, r in enumerate(result)
    )


@GraphPropagator.register_backward(OpType.CONCAT)
def _bwd_concat(output_region, input_shapes, output_shape, attrs):
    if _is_clean(output_region):
        return [None] * len(input_shapes)
    axis = attrs.get("axis", 0)
    rank = len(output_shape)
    if axis < 0:
        axis += rank
    out_start, out_stop = output_region[axis]
    results = []
    current_offset = 0
    for in_shape in input_shapes:
        in_dim = in_shape[axis]
        in_end = current_offset + in_dim
        ov_start = max(out_start, current_offset)
        ov_stop = min(out_stop, in_end)
        if ov_start >= ov_stop:
            results.append(
                (
                    (
                        0,
                        0,
                    ),
                )
            )
        else:
            in_region = list(output_region)
            in_region[axis] = (ov_start - current_offset, ov_stop - current_offset)
            results.append(tuple(in_region))
        current_offset = in_end
    return results


# ---------------------------------------------------------------------------
# RESHAPE
# ---------------------------------------------------------------------------


@GraphPropagator.register_shape(OpType.RESHAPE)
def _shape_reshape(node: TensorNode, get_val):
    data = node.parents[0]
    shape_val = get_val(node.parents[1])
    if shape_val is not None:
        if hasattr(shape_val, "cpu"):
            shape_arr = shape_val.cpu().numpy()
        else:
            shape_arr = np.asarray(shape_val)
        target_dims_raw = shape_arr.astype(int).flatten().tolist()
        if -1 in target_dims_raw:
            if data.shape is None:
                node.shape = tuple(d if d != -1 else None for d in target_dims_raw)
                return
            total_vol = _prod_shape(data.shape)
            known_vol = 1
            idx_neg = -1
            target_dims = []
            for i, d in enumerate(target_dims_raw):
                if d == -1:
                    idx_neg = i
                    target_dims.append(None)
                else:
                    known_vol *= d
                    target_dims.append(d)
            if total_vol is not None and known_vol is not None:
                target_dims[idx_neg] = total_vol // known_vol
            node.shape = tuple(target_dims)
        else:
            node.shape = tuple(target_dims_raw)


@GraphPropagator.register_forward(OpType.RESHAPE)
def _fwd_reshape(input_regions, input_shapes, output_shape, attrs):
    if _is_clean(input_regions[0]):
        return None
    return _make_full(output_shape)


@GraphPropagator.register_backward(OpType.RESHAPE)
def _bwd_reshape(output_region, input_shapes, output_shape, attrs):
    if _is_clean(output_region):
        return [None] * len(input_shapes)
    return [_make_full(s) for s in input_shapes]


# ---------------------------------------------------------------------------
# PERMUTE
# ---------------------------------------------------------------------------


@GraphPropagator.register_shape(OpType.PERMUTE)
def _shape_permute(node: TensorNode, get_val):
    if node.parents and node.parents[0].shape is not None:
        dims = node.attrs.get("dims")
        if dims:
            input_shape = node.parents[0].shape
            if len(input_shape) == len(dims):
                node.shape = tuple(input_shape[d] for d in dims)


@GraphPropagator.register_forward(OpType.PERMUTE)
def _fwd_permute(input_regions, input_shapes, output_shape, attrs):
    inp = input_regions[0]
    if _is_clean(inp):
        return None
    dims = attrs.get("dims", list(range(len(output_shape))))
    return tuple(inp[dims[d]] for d in range(len(dims)))


@GraphPropagator.register_backward(OpType.PERMUTE)
def _bwd_permute(output_region, input_shapes, output_shape, attrs):
    if _is_clean(output_region):
        return [None]
    dims = attrs.get("dims", list(range(len(output_shape))))
    inv_dims = [0] * len(dims)
    for i, d in enumerate(dims):
        inv_dims[d] = i
    return [tuple(output_region[inv_dims[d]] for d in range(len(dims)))]


# ---------------------------------------------------------------------------
# DOT (matmul)
# ---------------------------------------------------------------------------


@GraphPropagator.register_shape(OpType.DOT)
def _shape_dot(node: TensorNode, get_val):
    if len(node.parents) != 2:
        return
    s0, s1 = node.parents[0].shape, node.parents[1].shape
    if not s0 or not s1:
        return
    if len(s0) == 3 and len(s1) == 2:
        node.shape = (
            s0[0] if s0[0] is not None else None,
            s0[1] if s0[1] is not None else None,
            s1[1] if s1[1] is not None else None,
        )
    elif len(s0) >= 2 and len(s1) >= 2:
        batch_out = _broadcast_shapes(s0[:-2], s1[:-2])
        node.shape = batch_out + (s0[-2], s1[-1])
    elif len(s0) == 2 and len(s1) == 2:
        node.shape = (s0[0], s1[1])


@GraphPropagator.register_forward(OpType.DOT)
def _fwd_dot(input_regions, input_shapes, output_shape, attrs):
    rA, rB = input_regions[0], input_regions[1]
    A_shape, B_shape = input_shapes[0], input_shapes[1]
    if _is_clean(rA) and _is_clean(rB):
        return None
    out_rank = len(output_shape)

    k_dirty = False
    if not _is_clean(rA):
        ks, ke = rA[-1]
        if ks < ke:
            k_dirty = True
    if not _is_clean(rB):
        ks, ke = rB[-2] if len(rB) >= 2 else (0, 0)
        if ks < ke:
            k_dirty = True

    result = []
    for d in range(out_rank - 2):
        start, stop = output_shape[d], 0
        if not _is_clean(rA) and d < len(rA) - 2:
            start = min(start, rA[d][0])
            stop = max(stop, rA[d][1])
        if not _is_clean(rB) and d < len(rB) - 2:
            start = min(start, rB[d][0])
            stop = max(stop, rB[d][1])
        if start >= stop:
            start, stop = 0, 0
        result.append((start, stop))

    if k_dirty:
        result.append((0, output_shape[-2]))
    elif not _is_clean(rA):
        m_s, m_e = rA[-2] if len(rA) >= 2 else (0, A_shape[-2])
        result.append((m_s, m_e))
    else:
        result.append((0, 0))

    if k_dirty:
        result.append((0, output_shape[-1]))
    elif not _is_clean(rB):
        result.append(rB[-1])
    else:
        result.append((0, 0))

    if all(s >= e for s, e in result):
        return None
    return tuple(result)


@GraphPropagator.register_backward(OpType.DOT)
def _bwd_dot(output_region, input_shapes, output_shape, attrs):
    if _is_clean(output_region):
        return [None, None]
    A_shape, B_shape = input_shapes[0], input_shapes[1]
    out_rank = len(output_shape)
    m_start, m_stop = output_region[-2] if out_rank >= 2 else (0, 1)
    n_start, n_stop = output_region[-1] if out_rank >= 1 else (0, 1)

    A_region = list(output_region[:-2]) if out_rank > 2 else []
    while len(A_region) < len(A_shape) - 2:
        A_region.insert(0, (0, A_shape[len(A_region)]))
    A_region = A_region[-(len(A_shape) - 2) :] if len(A_shape) > 2 else []
    A_region.append((m_start, m_stop))
    A_region.append((0, A_shape[-1]))

    B_region = list(output_region[:-2]) if out_rank > 2 else []
    while len(B_region) < len(B_shape) - 2:
        B_region.insert(0, (0, B_shape[len(B_region)]))
    B_region = B_region[-(len(B_shape) - 2) :] if len(B_shape) > 2 else []
    B_region.append((0, B_shape[-2]))
    B_region.append((n_start, n_stop))

    return [tuple(A_region), tuple(B_region)]


# ---------------------------------------------------------------------------
# GATHER
# ---------------------------------------------------------------------------


@GraphPropagator.register_shape(OpType.GATHER)
def _shape_gather(node: TensorNode, get_val):
    if len(node.parents) >= 2:
        data_shape = node.parents[0].shape
        idx_shape = node.parents[1].shape
        if data_shape and idx_shape:
            if data_shape[1:] is not None:
                node.shape = idx_shape + data_shape[1:]
            else:
                node.shape = (None,) * (len(idx_shape) + len(data_shape[1:]))


@GraphPropagator.register_forward(OpType.GATHER)
def _fwd_gather(input_regions, input_shapes, output_shape, attrs):
    data_reg, idx_reg = input_regions[0], input_regions[1]
    data_shape, idx_shape = input_shapes[0], input_shapes[1]
    if _is_clean(data_reg) and _is_clean(idx_reg):
        return None
    idx_rank = len(idx_shape)
    result = []
    for d in range(idx_rank):
        if not _is_clean(idx_reg):
            result.append(idx_reg[d])
        elif not _is_clean(data_reg):
            result.append((0, output_shape[d]))
        else:
            result.append((0, 0))
    for d in range(1, len(data_shape)):
        out_d = idx_rank + d - 1
        if not _is_clean(data_reg) and d < len(data_reg):
            result.append(data_reg[d])
        elif not _is_clean(idx_reg):
            result.append((0, output_shape[out_d]))
        else:
            result.append((0, 0))
    if all(s >= e for s, e in result):
        return None
    return tuple(result)


@GraphPropagator.register_backward(OpType.GATHER)
def _bwd_gather(output_region, input_shapes, output_shape, attrs):
    if _is_clean(output_region):
        return [None, None]
    data_shape, idx_shape = input_shapes[0], input_shapes[1]
    idx_rank = len(idx_shape)
    data_region = [(0, data_shape[0])]
    for d in range(1, len(data_shape)):
        out_d = idx_rank + d - 1
        if out_d < len(output_region):
            data_region.append(output_region[out_d])
        else:
            data_region.append((0, data_shape[d]))
    idx_region = [output_region[d] for d in range(idx_rank)]
    return [tuple(data_region), tuple(idx_region)]


# ---------------------------------------------------------------------------
# Reductions – SUM, MAX
# ---------------------------------------------------------------------------

_REDUCE_OPS = (OpType.SUM, OpType.MAX)


def _resolve_axes(attrs, in_rank):
    axis = attrs.get("axis")
    if axis is None:
        return set(range(in_rank))
    if isinstance(axis, (list, tuple)):
        return set(int(a) + in_rank if int(a) < 0 else int(a) for a in axis)
    a = int(axis)
    return {a + in_rank if a < 0 else a}


@GraphPropagator.register_shape(*_REDUCE_OPS)
def _shape_reduce(node: TensorNode, get_val):
    if not node.parents or node.parents[0].shape is None:
        return
    data_shape = list(node.parents[0].shape)
    ndim = len(data_shape)
    axis = node.attrs.get("axis")
    if axis is None and len(node.parents) > 1:
        val = get_val(node.parents[1])
        if val is not None:
            axis = int(val.item()) if hasattr(val, "item") else int(val)
    keepdims = node.attrs.get("keepdims", True)
    if axis is None:
        if keepdims:
            node.shape = tuple(1 for _ in range(ndim))
        else:
            node.shape = (1,)
    else:
        if hasattr(axis, "__iter__") and not isinstance(axis, (str, bytes)):
            axes = list(axis)
        else:
            axes = [axis]
        axes = [int(a) + ndim if int(a) < 0 else int(a) for a in axes]
        new_shape = []
        for i, d in enumerate(data_shape):
            if i in axes:
                if keepdims:
                    new_shape.append(1)
            else:
                new_shape.append(d)
        node.shape = tuple(new_shape)


@GraphPropagator.register_forward(*_REDUCE_OPS)
def _fwd_reduce(input_regions, input_shapes, output_shape, attrs):
    inp = input_regions[0]
    if _is_clean(inp):
        return None
    in_shape = input_shapes[0]
    in_rank = len(in_shape)
    axes = _resolve_axes(attrs, in_rank)
    keepdims = attrs.get("keepdims", True)
    result = []
    for d in range(in_rank):
        if d in axes:
            if keepdims:
                if inp[d][0] < inp[d][1]:
                    result.append((0, 1))
                else:
                    result.append((0, 0))
        else:
            result.append(inp[d])
    if all(s >= e for s, e in result):
        return None
    return tuple(result)


@GraphPropagator.register_backward(*_REDUCE_OPS)
def _bwd_reduce(output_region, input_shapes, output_shape, attrs):
    if _is_clean(output_region):
        return [None]
    in_shape = input_shapes[0]
    in_rank = len(in_shape)
    axes = _resolve_axes(attrs, in_rank)
    keepdims = attrs.get("keepdims", True)
    result = []
    out_idx = 0
    for d in range(in_rank):
        if d in axes:
            if keepdims:
                if output_region[out_idx][0] < output_region[out_idx][1]:
                    result.append((0, in_shape[d]))
                else:
                    result.append((0, 0))
                out_idx += 1
            else:
                result.append((0, in_shape[d]))
        else:
            result.append(output_region[out_idx])
            out_idx += 1
    return [tuple(result)]


# ---------------------------------------------------------------------------
# ARANGE
# ---------------------------------------------------------------------------


@GraphPropagator.register_shape(OpType.ARANGE)
def _shape_arange(node: TensorNode, get_val):
    start = get_val(node.parents[0])
    stop = get_val(node.parents[1])
    step = get_val(node.parents[2])
    if start is not None and stop is not None and step is not None:
        s = float(start) if not isinstance(start, np.ndarray) else float(start.item())
        e = float(stop) if not isinstance(stop, np.ndarray) else float(stop.item())
        st = float(step) if not isinstance(step, np.ndarray) else float(step.item())
        if st != 0:
            node.shape = (max(0, int(math.ceil((e - s) / st))),)
    else:
        node.shape = (None,)


@GraphPropagator.register_forward(OpType.ARANGE)
def _fwd_arange(input_regions, input_shapes, output_shape, attrs):
    if any(not _is_clean(r) for r in input_regions):
        return _make_full(output_shape)
    return None


@GraphPropagator.register_backward(OpType.ARANGE)
def _bwd_arange(output_region, input_shapes, output_shape, attrs):
    if _is_clean(output_region):
        return [None] * len(input_shapes)
    return [_make_full(s) if s else () for s in input_shapes]


# ---------------------------------------------------------------------------
# FILL
# ---------------------------------------------------------------------------


@GraphPropagator.register_shape(OpType.FILL)
def _shape_fill(node: TensorNode, get_val):
    if len(node.parents) >= 2:
        shape_val = get_val(node.parents[1])
        if shape_val is not None:
            if hasattr(shape_val, "cpu"):
                shape_arr = shape_val.cpu().numpy()
            else:
                shape_arr = np.asarray(shape_val)
            node.shape = tuple(int(x) for x in shape_arr.astype(int).flatten())


@GraphPropagator.register_forward(OpType.FILL)
def _fwd_fill(input_regions, input_shapes, output_shape, attrs):
    if any(not _is_clean(r) for r in input_regions):
        return _make_full(output_shape)
    return None


@GraphPropagator.register_backward(OpType.FILL)
def _bwd_fill(output_region, input_shapes, output_shape, attrs):
    if _is_clean(output_region):
        return [None] * len(input_shapes)
    return [_make_full(s) if s else () for s in input_shapes]


# ---------------------------------------------------------------------------
# REPEAT
# ---------------------------------------------------------------------------


@GraphPropagator.register_shape(OpType.REPEAT)
def _shape_repeat(node: TensorNode, get_val):
    if node.parents and node.parents[0].shape:
        data_shape = list(node.parents[0].shape)
        repeats = int(node.attrs.get("repeats", 1))
        axis = int(node.attrs.get("axis", 0))
        ndim = len(data_shape)
        if axis < 0:
            axis += ndim
        if 0 <= axis < ndim:
            dim_val = data_shape[axis]
            if dim_val is not None:
                data_shape[axis] = dim_val * repeats
            else:
                data_shape[axis] = None
        node.shape = tuple(data_shape)


@GraphPropagator.register_forward(OpType.REPEAT)
def _fwd_repeat(input_regions, input_shapes, output_shape, attrs):
    inp = input_regions[0]
    if _is_clean(inp):
        return None
    axis = attrs.get("axis", 0)
    repeats = attrs.get("repeats", 1)
    rank = len(output_shape)
    if axis < 0:
        axis += rank
    result = list(inp)
    start, stop = result[axis]
    result[axis] = (start * repeats, stop * repeats)
    return tuple(result)


@GraphPropagator.register_backward(OpType.REPEAT)
def _bwd_repeat(output_region, input_shapes, output_shape, attrs):
    if _is_clean(output_region):
        return [None]
    axis = attrs.get("axis", 0)
    repeats = attrs.get("repeats", 1)
    in_shape = input_shapes[0]
    rank = len(in_shape)
    if axis < 0:
        axis += rank
    result = list(output_region)
    start, stop = result[axis]
    result[axis] = (start // repeats, (stop + repeats - 1) // repeats)
    return [tuple(result)]


# ---------------------------------------------------------------------------
# Fused ops – shape-preserving (RoPE, GELU, Softmax, RMSNorm)
# ---------------------------------------------------------------------------


@GraphPropagator.register_shape("RoPE", "GELU", "Softmax", "RMSNorm")
def _shape_fused_preserving(node: TensorNode, get_val):
    if node.parents and node.parents[0].shape:
        node.shape = node.parents[0].shape


# ---------------------------------------------------------------------------
# Normalization Ops (Softmax, RMSNorm)
# ---------------------------------------------------------------------------


@GraphPropagator.register_forward("Softmax", "RMSNorm")
def _fwd_norm(input_regions, input_shapes, output_shape, attrs):
    """
    Forward: If any element in a vector is dirty, the entire output vector
    along the normalization axis becomes dirty.
    """
    inp = input_regions[0]
    if _is_clean(inp):
        return None

    axis = attrs.get("axis", -1)
    rank = len(output_shape)
    if axis < 0:
        axis += rank

    result = list(inp)
    # Any change in the row makes the entire output row dirty
    # because the normalization constant changes for everyone.
    result[axis] = (0, output_shape[axis])
    return tuple(result)


@GraphPropagator.register_backward("Softmax")
def _bwd_softmax(output_region, input_shapes, output_shape, attrs):
    """
    Backward: To compute even one element of the output, we need the
    entire input vector along the normalization axis.
    """
    if _is_clean(output_region):
        return [None]

    axis = attrs.get("axis", -1)
    rank = len(output_shape)
    if axis < 0:
        axis += rank

    result = list(output_region)
    # We need the full row from the input to compute the sum/mean
    result[axis] = (0, input_shapes[0][axis])
    return [tuple(result)]


@GraphPropagator.register_backward("RMSNorm")
def _bwd_norm(output_region, input_shapes, output_shape, attrs):
    """
    Backward: To compute even one element of the output, we need the
    entire input vector along the normalization axis.
    """
    if _is_clean(output_region):
        return [None] * len(input_shapes)

    axis = attrs.get("axis", -1)
    rank = len(output_shape)
    if axis < 0:
        axis += rank

    result = list(output_region)
    # We need the full row from the input to compute the sum/mean
    result[axis] = (0, input_shapes[0][axis])

    # Return regions for all inputs
    input_regions = [tuple(result)]

    # For additional inputs (like scale in RMSNorm), propagate the region
    for i in range(1, len(input_shapes)):
        if input_shapes[i] is not None:
            # Scale parameter typically matches the normalized dimension
            # Just take the slice along the normalization axis
            input_regions.append((result[axis],))
        else:
            input_regions.append(None)

    return input_regions
