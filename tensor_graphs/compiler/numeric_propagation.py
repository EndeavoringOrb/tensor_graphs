"""
Numeric Dirty Region Propagation

Instead of building symbolic expressions and lambdifying them (exponential blowup),
we propagate concrete integer bounds through the operation graph at runtime.

This is O(nodes × max_rank) per propagation call, which is negligible compared
to the actual kernel execution time.
"""

from typing import Dict, List, Tuple, Optional, Callable

from ..ir.node import TensorNode
from ..ops.atomic_types import OpType
from ..ops.registry import get_reference_factory
from ..ir.graph import topological_sort
from .shape_inference import ShapeInference


# A dirty region is None (clean) or a tuple of (start, stop) per dimension
NumericRegion = Optional[Tuple[Tuple[int, int], ...]]


def _is_clean(region: NumericRegion) -> bool:
    """Check if a region is clean (None or all empty slices)."""
    if region is None:
        return True
    return all(start >= stop for start, stop in region)


def _make_full(shape: Tuple[int, ...]) -> NumericRegion:
    """Create a fully-dirty region covering the entire shape."""
    if not shape:
        return ()
    return tuple((0, dim) for dim in shape)


def _make_clean(shape: Tuple[int, ...]) -> NumericRegion:
    """Create a clean region (empty bounds)."""
    return None


def _merge_regions(
    r1: NumericRegion, r2: NumericRegion, shape: Tuple[int, ...]
) -> NumericRegion:
    """Union of two regions (bounding box)."""
    if r1 is None and r2 is None:
        return None
    if r1 is None:
        return r2
    if r2 is None:
        return r1

    return tuple((min(a[0], b[0]), max(a[1], b[1])) for a, b in zip(r1, r2))


def _to_slices(region: NumericRegion) -> Optional[Tuple[slice, ...]]:
    """Convert numeric region to slice tuple for executor compatibility."""
    if region is None:
        return None
    if all(start >= stop for start, stop in region):
        return None
    return tuple(slice(start, stop) for start, stop in region)


def _from_slices(
    slices: Optional[Tuple[slice, ...]], shape: Tuple[int, ...]
) -> NumericRegion:
    """Convert slice tuple to numeric region."""
    if slices is None:
        return None
    result = []
    for i, s in enumerate(slices):
        dim = shape[i] if i < len(shape) else 1
        start, stop, _ = s.indices(dim)
        result.append((start, stop))
    return tuple(result)


class NumericPropagator:
    """
    Runtime propagator that computes dirty regions using interval arithmetic.

    For atomic ops: uses registered handlers (simple integer math).
    For composite ops: decomposes and propagates through subgraph.
    """

    _forward_registry: Dict[str, Callable[..., NumericRegion]] = {}
    _backward_registry: Dict[str, Callable[..., List[NumericRegion]]] = {}

    @classmethod
    def register(cls, op_type: str):
        """Decorator to register a forward propagation handler."""

        def decorator(func: Callable[..., NumericRegion]):
            cls._forward_registry[op_type] = func
            return func

        return decorator

    @classmethod
    def register_backward(cls, op_type: str):
        """Decorator to register a backward propagation handler."""

        def decorator(func: Callable[..., List[NumericRegion]]):
            cls._backward_registry[op_type] = func
            return func

        return decorator

    @classmethod
    def propagate(
        cls,
        node: TensorNode,
        known_values: Optional[dict] = None,
    ) -> NumericRegion:
        """
        Propagate dirty regions forward through a node.

        Args:
            node: The node to propagate through
            known_values: Optional dict of known constant values

        Returns:
            The output dirty region (None if clean)
        """
        if not node.parents:
            return _from_slices(node.dirty_region, node.shape or ())

        input_regions = [
            _from_slices(p.dirty_region, p.shape or ()) for p in node.parents
        ]
        input_shapes = [p.shape or () for p in node.parents]
        output_shape = node.shape or ()

        # Direct handler available?
        if node.op_type in cls._forward_registry:
            return cls._forward_registry[node.op_type](
                input_regions, input_shapes, output_shape, node.attrs
            )

        # Decompose and propagate through subgraph
        return cls._propagate_subgraph(node, input_regions, known_values)

    @classmethod
    def _propagate_subgraph(
        cls,
        node: TensorNode,
        input_regions: List[NumericRegion],
        known_values: Optional[dict],
    ) -> NumericRegion:
        """Propagate through a decomposed subgraph."""
        factory = get_reference_factory(node.op_type)
        if not factory:
            raise ValueError(f"No handler or decomposition for {node.op_type}")

        # Create proxy leaf nodes
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

        # Prevent infinite recursion
        if sub_root.op_type == node.op_type and len(sub_root.parents) == len(
            node.parents
        ):
            raise NotImplementedError(
                f"Atomic op {node.op_type} has no numeric propagation handler"
            )

        sub_nodes = topological_sort(sub_root)
        ShapeInference.infer(sub_nodes, known_values, keep_cut_parent_shapes=True)

        # Map node id -> dirty region
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

            sub_input_regions = [region_map[id(p)] for p in sub.parents]
            sub_input_shapes = [p.shape or () for p in sub.parents]
            sub_output_shape = sub.shape or ()

            if sub.op_type in cls._forward_registry:
                region_map[nid] = cls._forward_registry[sub.op_type](
                    sub_input_regions, sub_input_shapes, sub_output_shape, sub.attrs
                )
            else:
                # Recursive decomposition
                region_map[nid] = cls._propagate_subgraph(
                    sub,
                    sub_input_regions,
                    known_values,
                )

        return region_map[id(sub_root)]

    @classmethod
    def get_input_slices(
        cls,
        node: TensorNode,
        output_region: NumericRegion,
        known_values: Optional[dict] = None,
    ) -> List[NumericRegion]:
        """
        Backward propagate: given output dirty region, compute required input regions.
        """
        if output_region is None or _is_clean(output_region):
            return [None] * len(node.parents)

        input_shapes = [p.shape or () for p in node.parents]
        output_shape = node.shape or ()

        if node.op_type in cls._backward_registry:
            return cls._backward_registry[node.op_type](
                output_region, input_shapes, output_shape, node.attrs
            )

        # Decompose and backward propagate
        return cls._backward_subgraph(node, output_region, known_values)

    @classmethod
    def _backward_subgraph(
        cls,
        node: TensorNode,
        output_region: NumericRegion,
        known_values: Optional[dict],
    ) -> List[NumericRegion]:
        """Backward propagate through a decomposed subgraph."""
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
                f"Atomic op {node.op_type} has no numeric backward handler"
            )

        sub_nodes = topological_sort(sub_root)
        ShapeInference.infer(sub_nodes, known_values, keep_cut_parent_shapes=True)

        # Backward: start from output, propagate to inputs
        region_map: Dict[int, NumericRegion] = {id(sub_root): output_region}

        for sub in reversed(sub_nodes):
            nid = id(sub)
            if nid not in region_map:
                continue
            if sub.op_type == OpType.CONSTANT or not sub.parents:
                continue

            sub_output_region = region_map[nid]
            sub_input_shapes = [p.shape or () for p in sub.parents]
            sub_output_shape = sub.shape or ()

            if sub.op_type in cls._backward_registry:
                sub_input_regions = cls._backward_registry[sub.op_type](
                    sub_output_region, sub_input_shapes, sub_output_shape, sub.attrs
                )
            else:
                sub_input_regions = cls._backward_subgraph(
                    sub, sub_output_region, known_values
                )

            # Merge into existing regions (union)
            for p, p_reg in zip(sub.parents, sub_input_regions):
                pid = id(p)
                if pid not in region_map:
                    region_map[pid] = p_reg
                else:
                    region_map[pid] = _merge_regions(
                        region_map[pid], p_reg, p.shape or ()
                    )

        return [region_map.get(id(lp), None) for lp in leaf_parents]


# =============================================================================
# Forward Handlers (Atomic Ops)
# =============================================================================


@NumericPropagator.register(OpType.ADD)
@NumericPropagator.register(OpType.MUL)
@NumericPropagator.register(OpType.DIVIDE)
@NumericPropagator.register(OpType.POWER)
@NumericPropagator.register(OpType.WHERE)
def propagate_elementwise(
    input_regions: List[NumericRegion],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> NumericRegion:
    """Elementwise ops: union of inputs with broadcasting."""
    # All clean → output clean
    if all(_is_clean(r) for r in input_regions):
        return None

    out_rank = len(output_shape)
    if out_rank == 0:
        return ()

    # Start with empty bounds
    result = [(output_shape[d], 0) for d in range(out_rank)]  # (max, min) inverted
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

            # Broadcasting: dim=1 expands to full output dim
            if shape[d] == 1:
                start, stop = 0, output_shape[out_d]

            result[out_d] = (
                min(result[out_d][0], start),
                max(result[out_d][1], stop),
            )

        # Leading broadcast dims become fully dirty
        for d in range(pad):
            result[d] = (0, output_shape[d])

    if not has_dirty:
        return None

    return tuple(result)


@NumericPropagator.register(OpType.CAST)
@NumericPropagator.register(OpType.COPY_TO)
@NumericPropagator.register(OpType.NEGATE)
@NumericPropagator.register(OpType.EXP)
@NumericPropagator.register(OpType.SIN)
@NumericPropagator.register(OpType.COS)
@NumericPropagator.register(OpType.SQRT)
@NumericPropagator.register(OpType.TRIU)
def propagate_unary(
    input_regions: List[NumericRegion],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> NumericRegion:
    """Unary ops: pass through unchanged."""
    return input_regions[0] if input_regions else None


@NumericPropagator.register(OpType.SLICE)
def propagate_slice(
    input_regions: List[NumericRegion],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> NumericRegion:
    """Slice: intersect with slice window and adjust coordinates."""
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

        # Intersect dirty region with slice window
        overlap_start = max(in_start, sl_start)
        overlap_end = min(in_stop, sl_end)

        if overlap_start >= overlap_end:
            # No overlap → this dim is clean in output
            result.append((0, 0))
        else:
            # Transform to output coordinates
            out_start = (overlap_start - sl_start + sl_step - 1) // sl_step
            out_end = (overlap_end - sl_start + sl_step - 1) // sl_step
            result.append((max(0, out_start), min(out_end, output_shape[d])))

    if all(start >= stop for start, stop in result):
        return None
    return tuple(result)


@NumericPropagator.register(OpType.CONCAT)
def propagate_concat(
    input_regions: List[NumericRegion],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> NumericRegion:
    """Concat: offset each input's region on the concat axis."""
    axis = attrs.get("axis", 0)
    rank = len(output_shape)
    if axis < 0:
        axis += rank

    if all(_is_clean(r) for r in input_regions):
        return None

    # Start with empty bounds
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

    # Check if still empty
    if all(r[0] >= r[1] for r in result):
        return None

    return tuple(
        (max(0, r[0]), min(r[1], output_shape[i])) for i, r in enumerate(result)
    )


@NumericPropagator.register(OpType.RESHAPE)
def propagate_reshape(
    input_regions: List[NumericRegion],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> NumericRegion:
    """Reshape: any dirty → full dirty (conservative but correct)."""
    if _is_clean(input_regions[0]):
        return None
    return _make_full(output_shape)


@NumericPropagator.register(OpType.PERMUTE)
def propagate_permute(
    input_regions: List[NumericRegion],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> NumericRegion:
    """Permute: reorder dimensions according to perm."""
    inp = input_regions[0]
    if _is_clean(inp):
        return None

    dims = attrs.get("dims", list(range(len(output_shape))))
    return tuple(inp[dims[d]] for d in range(len(dims)))


@NumericPropagator.register(OpType.DOT)
def propagate_dot(
    input_regions: List[NumericRegion],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> NumericRegion:
    """Matmul: K collision → full M×N, else propagate M and N ranges."""
    rA, rB = input_regions[0], input_regions[1]
    A_shape, B_shape = input_shapes[0], input_shapes[1]

    if _is_clean(rA) and _is_clean(rB):
        return None

    out_rank = len(output_shape)

    # Check K dimension collision
    k_dirty = False
    if not _is_clean(rA):
        k_start, k_stop = rA[-1]
        if k_start < k_stop:
            k_dirty = True
    if not _is_clean(rB):
        k_start, k_stop = rB[-2] if len(rB) >= 2 else (0, 0)
        if k_start < k_stop:
            k_dirty = True

    result = []

    # Batch dims (union)
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

    # M dimension
    if k_dirty:
        result.append((0, output_shape[-2]))
    elif not _is_clean(rA):
        m_start, m_stop = rA[-2] if len(rA) >= 2 else (0, A_shape[-2])
        result.append((m_start, m_stop))
    else:
        result.append((0, 0))

    # N dimension
    if k_dirty:
        result.append((0, output_shape[-1]))
    elif not _is_clean(rB):
        n_start, n_stop = rB[-1]
        result.append((n_start, n_stop))
    else:
        result.append((0, 0))

    if all(start >= stop for start, stop in result):
        return None

    return tuple(result)


@NumericPropagator.register(OpType.GATHER)
def propagate_gather(
    input_regions: List[NumericRegion],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> NumericRegion:
    """Gather: index changes → full recompute on gathered dims."""
    data_reg, idx_reg = input_regions[0], input_regions[1]
    data_shape, idx_shape = input_shapes[0], input_shapes[1]

    if _is_clean(data_reg) and _is_clean(idx_reg):
        return None

    idx_rank = len(idx_shape)
    result = []

    # Leading dims from indices
    for d in range(idx_rank):
        if not _is_clean(idx_reg):
            result.append(idx_reg[d])
        elif not _is_clean(data_reg):
            # If data axis 0 is dirty, all gathered positions are dirty
            result.append((0, output_shape[d]))
        else:
            result.append((0, 0))

    # Trailing dims from data
    for d in range(1, len(data_shape)):
        out_d = idx_rank + d - 1
        if not _is_clean(data_reg) and d < len(data_reg):
            result.append(data_reg[d])
        elif not _is_clean(idx_reg):
            result.append((0, output_shape[out_d]))
        else:
            result.append((0, 0))

    if all(start >= stop for start, stop in result):
        return None

    return tuple(result)


@NumericPropagator.register(OpType.SUM)
@NumericPropagator.register(OpType.MAX)
def propagate_reduce(
    input_regions: List[NumericRegion],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> NumericRegion:
    """Reduction: reduced dims become full, others pass through."""
    inp = input_regions[0]
    if _is_clean(inp):
        return None

    in_shape = input_shapes[0]
    axis = attrs.get("axis")
    keepdims = attrs.get("keepdims", True)
    in_rank = len(in_shape)

    if axis is None:
        axes = set(range(in_rank))
    else:
        axes = set(
            a if a >= 0 else a + in_rank
            for a in (axis if isinstance(axis, (list, tuple)) else [axis])
        )

    result = []
    out_idx = 0

    for d in range(in_rank):
        if d in axes:
            if keepdims:
                # If any part of reduced dim is dirty, output is dirty
                if inp[d][0] < inp[d][1]:
                    result.append((0, 1))
                else:
                    result.append((0, 0))
                out_idx += 1
            # else: dimension is removed
        else:
            result.append(inp[d])
            out_idx += 1

    if all(start >= stop for start, stop in result):
        return None

    return tuple(result)


@NumericPropagator.register(OpType.ARANGE)
def propagate_arange(
    input_regions: List[NumericRegion],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> NumericRegion:
    """Arange: if any scalar input is dirty, output is fully dirty."""
    if any(not _is_clean(r) for r in input_regions):
        return _make_full(output_shape)
    return None


@NumericPropagator.register(OpType.FILL)
def propagate_fill(
    input_regions: List[NumericRegion],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> NumericRegion:
    """Fill: if value or shape changes, entire output is dirty."""
    if any(not _is_clean(r) for r in input_regions):
        return _make_full(output_shape)
    return None


@NumericPropagator.register(OpType.REPEAT)
def propagate_repeat(
    input_regions: List[NumericRegion],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> NumericRegion:
    """Repeat: expand dirty region by repeat factor on the axis."""
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


# =============================================================================
# Backward Handlers (Atomic Ops)
# =============================================================================


@NumericPropagator.register_backward(OpType.ADD)
@NumericPropagator.register_backward(OpType.MUL)
@NumericPropagator.register_backward(OpType.DIVIDE)
@NumericPropagator.register_backward(OpType.POWER)
@NumericPropagator.register_backward(OpType.WHERE)
def backward_elementwise(
    output_region: NumericRegion,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> List[NumericRegion]:
    """Backward elementwise: project output region to each input with broadcasting."""
    if _is_clean(output_region):
        return [None] * len(input_shapes)

    out_rank = len(output_shape)
    results = []

    for in_shape in input_shapes:
        in_rank = len(in_shape)
        if in_rank == 0:
            results.append(())
            continue

        pad = out_rank - in_rank
        in_region = []

        for d in range(in_rank):
            out_d = d + pad
            start, stop = output_region[out_d]

            # If input dim is 1 (broadcast source), need full input
            if in_shape[d] == 1:
                in_region.append((0, 1))
            else:
                in_region.append((start, stop))

        results.append(tuple(in_region))

    return results


@NumericPropagator.register_backward(OpType.CAST)
@NumericPropagator.register_backward(OpType.COPY_TO)
@NumericPropagator.register_backward(OpType.NEGATE)
@NumericPropagator.register_backward(OpType.EXP)
@NumericPropagator.register_backward(OpType.SIN)
@NumericPropagator.register_backward(OpType.COS)
@NumericPropagator.register_backward(OpType.SQRT)
@NumericPropagator.register_backward(OpType.TRIU)
def backward_unary(
    output_region: NumericRegion,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> List[NumericRegion]:
    """Backward unary: pass through unchanged."""
    return [output_region]


@NumericPropagator.register_backward(OpType.SLICE)
def backward_slice(
    output_region: NumericRegion,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> List[NumericRegion]:
    """Backward slice: map output coords back to input coords."""
    if _is_clean(output_region):
        return [None]

    in_shape = input_shapes[0]
    starts = attrs.get("starts", [])
    steps = attrs.get("steps", [])

    result = []
    for d in range(len(in_shape)):
        out_start, out_stop = (
            output_region[d] if d < len(output_region) else (0, output_shape[d])
        )
        sl_start = starts[d] if d < len(starts) else 0
        sl_step = steps[d] if d < len(steps) and steps[d] else 1

        in_start = sl_start + out_start * sl_step
        in_stop = sl_start + (out_stop - 1) * sl_step + 1

        result.append((max(0, in_start), min(in_stop, in_shape[d])))

    return [tuple(result)]


@NumericPropagator.register_backward(OpType.CONCAT)
def backward_concat(
    output_region: NumericRegion,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> List[NumericRegion]:
    """Backward concat: split output region by input boundaries."""
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

        # Intersect with this input's range
        ov_start = max(out_start, current_offset)
        ov_stop = min(out_stop, in_end)

        if ov_start >= ov_stop:
            results.append(None)
        else:
            in_region = list(output_region)
            in_region[axis] = (ov_start - current_offset, ov_stop - current_offset)
            results.append(tuple(in_region))

        current_offset = in_end

    return results


@NumericPropagator.register_backward(OpType.RESHAPE)
def backward_reshape(
    output_region: NumericRegion,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> List[NumericRegion]:
    """Backward reshape: any dirty → full input dirty."""
    if _is_clean(output_region):
        return [None] * len(input_shapes)
    results = [_make_full(input_shapes[0])]
    # Handle shape tensor input if present
    for i in range(1, len(input_shapes)):
        results.append(_make_full(input_shapes[i]))
    return results


@NumericPropagator.register_backward(OpType.PERMUTE)
def backward_permute(
    output_region: NumericRegion,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> List[NumericRegion]:
    """Backward permute: inverse permutation."""
    if _is_clean(output_region):
        return [None]

    dims = attrs.get("dims", list(range(len(output_shape))))

    # Compute inverse permutation
    inv_dims = [0] * len(dims)
    for i, d in enumerate(dims):
        inv_dims[d] = i

    return [tuple(output_region[inv_dims[d]] for d in range(len(dims)))]


@NumericPropagator.register_backward(OpType.DOT)
def backward_dot(
    output_region: NumericRegion,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> List[NumericRegion]:
    """Backward dot: A needs full K, B needs full K."""
    if _is_clean(output_region):
        return [None, None]

    A_shape, B_shape = input_shapes[0], input_shapes[1]
    out_rank = len(output_shape)

    m_start, m_stop = output_region[-2] if out_rank >= 2 else (0, 1)
    n_start, n_stop = output_region[-1] if out_rank >= 1 else (0, 1)

    # A: (..., M, K) - need full K
    A_region = list(output_region[:-2]) if out_rank > 2 else []
    # Pad to match A's batch dims
    while len(A_region) < len(A_shape) - 2:
        A_region.insert(0, (0, A_shape[len(A_region)]))
    A_region = A_region[-(len(A_shape) - 2) :] if len(A_shape) > 2 else []
    A_region.append((m_start, m_stop))
    A_region.append((0, A_shape[-1]))  # Full K

    # B: (..., K, N) - need full K
    B_region = list(output_region[:-2]) if out_rank > 2 else []
    while len(B_region) < len(B_shape) - 2:
        B_region.insert(0, (0, B_shape[len(B_region)]))
    B_region = B_region[-(len(B_shape) - 2) :] if len(B_shape) > 2 else []
    B_region.append((0, B_shape[-2]))  # Full K
    B_region.append((n_start, n_stop))

    return [tuple(A_region), tuple(B_region)]


@NumericPropagator.register_backward(OpType.GATHER)
def backward_gather(
    output_region: NumericRegion,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> List[NumericRegion]:
    """Backward gather: data needs full axis 0, indices match output prefix."""
    if _is_clean(output_region):
        return [None, None]

    data_shape, idx_shape = input_shapes[0], input_shapes[1]
    idx_rank = len(idx_shape)

    # Data: full on axis 0, match trailing dims from output
    data_region = [(0, data_shape[0])]
    for d in range(1, len(data_shape)):
        out_d = idx_rank + d - 1
        if out_d < len(output_region):
            data_region.append(output_region[out_d])
        else:
            data_region.append((0, data_shape[d]))

    # Indices: match leading dims of output
    idx_region = [output_region[d] for d in range(idx_rank)]

    return [tuple(data_region), tuple(idx_region)]


@NumericPropagator.register_backward(OpType.SUM)
@NumericPropagator.register_backward(OpType.MAX)
def backward_reduce(
    output_region: NumericRegion,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> List[NumericRegion]:
    """Backward reduce: reduced dims need full input, others pass through."""
    if _is_clean(output_region):
        return [None]

    in_shape = input_shapes[0]
    axis = attrs.get("axis")
    keepdims = attrs.get("keepdims", True)
    in_rank = len(in_shape)

    if axis is None:
        axes = set(range(in_rank))
    else:
        axes = set(
            a if a >= 0 else a + in_rank
            for a in (axis if isinstance(axis, (list, tuple)) else [axis])
        )

    result = []
    out_idx = 0

    for d in range(in_rank):
        if d in axes:
            # Reduced dimension needs full input
            if (
                output_region[out_idx][0] < output_region[out_idx][1]
                if keepdims
                else True
            ):
                result.append((0, in_shape[d]))
            else:
                result.append((0, 0))
            if keepdims:
                out_idx += 1
        else:
            result.append(output_region[out_idx])
            out_idx += 1

    return [tuple(result)]


@NumericPropagator.register_backward(OpType.ARANGE)
def backward_arange(
    output_region: NumericRegion,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> List[NumericRegion]:
    """Backward arange: if output dirty, all scalar inputs are dirty."""
    if _is_clean(output_region):
        return [None] * len(input_shapes)
    return [_make_full(s) if s else () for s in input_shapes]


@NumericPropagator.register_backward(OpType.FILL)
def backward_fill(
    output_region: NumericRegion,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> List[NumericRegion]:
    """Backward fill: if output dirty, value and shape are dirty."""
    if _is_clean(output_region):
        return [None] * len(input_shapes)
    return [_make_full(s) if s else () for s in input_shapes]


@NumericPropagator.register_backward(OpType.REPEAT)
def backward_repeat(
    output_region: NumericRegion,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    attrs: dict,
) -> List[NumericRegion]:
    """Backward repeat: shrink by repeat factor."""
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
