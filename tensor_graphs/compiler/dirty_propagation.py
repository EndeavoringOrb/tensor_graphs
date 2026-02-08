from typing import Tuple, Optional, List, Any
from ..ir.node import TensorNode
from ..ops.atomic_types import OpType

# A DirtyRegion is a tuple of slices, one per dimension.
# None implies the node is CLEAN.
DirtyRegion = Optional[Tuple[slice, ...]]


class DirtyPropagator:
    """
    Handles the logic for incremental execution:
    1. Calculating diffs for Inputs.
    2. Propagating dirty regions forward through Atomic Ops.
    3. Mapping Output Dirty Regions back to Input Slices for execution.
    """

    @staticmethod
    def get_diff(old_data: Any, new_data: Any) -> DirtyRegion:
        """
        Compares old_data and new_data buffers to find the minimal dirty slice.
        Assumes data is contiguous and changes are generally append-like or block updates.
        Returns a tuple of slices covering the changed region.
        """
        if old_data is None:
            # First run, fully dirty
            ndim = new_data.ndim if hasattr(new_data, "ndim") else 0
            if ndim == 0:
                return (slice(None),)
            return tuple(slice(None) for _ in range(ndim))

        # Check for shape change (re-allocation)
        if hasattr(new_data, "shape") and hasattr(old_data, "shape"):
            if new_data.shape != old_data.shape:
                return tuple(slice(None) for _ in range(new_data.ndim))

        # Check content diff
        # We assume numpy-like or torch-like interface
        try:
            # Optimization: If hashes match (handled in Executor), this isn't called.
            # Here we do exact diffing.
            import numpy as np

            # Convert to numpy for diffing if needed
            n_new = new_data
            n_old = old_data

            if hasattr(new_data, "cpu"):
                n_new = new_data.cpu().numpy()
            if hasattr(old_data, "cpu"):
                n_old = old_data.cpu().numpy()

            if not isinstance(n_new, np.ndarray):
                n_new = np.array(n_new)
            if not isinstance(n_old, np.ndarray):
                n_old = np.array(n_old)

            if n_new.shape != n_old.shape:
                return tuple(slice(None) for _ in range(n_new.ndim))

            diff = n_new != n_old
            if not np.any(diff):
                return None

            # Find bounding box of changes
            slices = []
            for dim in range(diff.ndim):
                # Project diff to this dimension
                # any() over all axes except dim
                axes_to_reduce = tuple(i for i in range(diff.ndim) if i != dim)
                if axes_to_reduce:
                    dim_diff = np.any(diff, axis=axes_to_reduce)
                else:
                    dim_diff = diff  # 1D case

                # Find indices
                indices = np.where(dim_diff)[0]
                if len(indices) == 0:
                    slices.append(
                        slice(0, 0)
                    )  # Should not happen if np.any(diff) is true
                else:
                    start = indices[0]
                    stop = indices[-1] + 1
                    slices.append(slice(int(start), int(stop)))

            return tuple(slices)

        except Exception:
            # Fallback to full dirty if diff fails
            ndim = new_data.ndim if hasattr(new_data, "ndim") else 0
            return tuple(slice(None) for _ in range(ndim))

    @staticmethod
    def propagate(node: TensorNode) -> DirtyRegion:
        """
        Calculates the dirty region of 'node' based on the dirty regions of its parents.
        """
        # 1. Check if all parents are clean
        parents_dirty = False
        input_regions = []

        # If node has no parents (and not Input), it's static/constant, thus clean
        if not node.parents and node.op_type != OpType.INPUT:
            return None

        for p in node.parents:
            if p.dirty_region is not None:
                parents_dirty = True
            input_regions.append(p.dirty_region)

        if not parents_dirty and node.op_type != OpType.INPUT:
            return None

        # 2. Dispatch to specific logic
        handler = _PROPAGATION_HANDLERS.get(node.op_type, _propagate_elementwise)

        # Helper: resolve slice(None) to actual shape if possible, or keep as slice(None)
        # We keep slice(None) as it handles broadcasting generically.

        return handler(node, input_regions)

    @staticmethod
    def get_input_slices(node: TensorNode, output_region: DirtyRegion) -> List[Any]:
        """
        Given that 'node' needs to compute 'output_region', calculate the
        slices required from each parent.
        Returns a list of keys (tuples of slices) to index into parent buffers.
        """
        if output_region is None:
            return []

        handler = _SLICE_MAPPING_HANDLERS.get(node.op_type, _map_slice_elementwise)
        return handler(node, output_region)


# --- Helper Functions ---


def _merge_regions(r1: DirtyRegion, r2: DirtyRegion) -> DirtyRegion:
    if r1 is None:
        return r2
    if r2 is None:
        return r1

    # Check dimensionality mismatch (broadcasting)
    ndim = max(len(r1), len(r2))

    # Prepend slice(None) (which acts as 0:1 broadcasted usually, or full)
    # Actually for dirty propagation, if a dim is missing, it implies broadcasting.
    # If a broadcasted dim is dirty, it dirties the whole expanded dim.

    # Normalize to same rank by prepending None (newaxis logic)
    # But slice objects don't store rank. We assume slice(None) covers everything.

    # Simple Union Logic:
    # If ranks match: bounding box.
    if len(r1) == len(r2):
        new_slices = []
        for s1, s2 in zip(r1, r2):
            if s1 == slice(None) or s2 == slice(None):
                new_slices.append(slice(None))
            else:
                start = min(s1.start, s2.start)
                stop = max(s1.stop, s2.stop)
                new_slices.append(slice(start, stop))
        return tuple(new_slices)
    else:
        # Broadcasting case: Conservative fallback -> Fully Dirty on dimensions that exist
        # Or try to align from right?
        # A dirty region in a scalar input dirties the whole output.
        # So we take the "wider" region.
        # For simplicity in this implementation: if ranks differ, return full dirty on max rank.
        return tuple(slice(None) for _ in range(ndim))


# --- Propagation Handlers ---


def _propagate_elementwise(
    node: TensorNode, input_regions: List[DirtyRegion]
) -> DirtyRegion:
    """Propagates union of all input regions."""
    if not input_regions:
        return None
    combined = input_regions[0]
    for r in input_regions[1:]:
        combined = _merge_regions(combined, r)
    return combined


def _propagate_matmul(
    node: TensorNode, input_regions: List[DirtyRegion]
) -> DirtyRegion:
    """
    A @ B
    A: (..., M, K), B: (..., K, N)
    Out: (..., M, N)
    """
    if len(input_regions) != 2:
        return _propagate_elementwise(node, input_regions)

    rA = input_regions[0]
    rB = input_regions[1]

    # If either is clean, treat as empty region, but we need structure.
    # If rA is None, it contributes nothing to dirtyness.

    # Standard 2D Logic:
    # rA = (rows_a, cols_a)
    # rB = (rows_b, cols_b)
    # Out = (rows_a OR rows_b_if_inner_dirty, cols_b OR cols_a_if_inner_dirty)

    rows_dirty = slice(0, 0)  # Empty
    cols_dirty = slice(0, 0)

    # Handle A
    if rA is not None:
        # If A is > 2D, handle batch dims conservatively (union)
        if len(rA) > 2:
            return tuple(slice(None) for _ in range(len(rA)))

        row_slice = rA[0]
        inner_slice = rA[1] if len(rA) > 1 else slice(None)

        # A makes rows dirty directly
        rows_dirty = row_slice

        # If A's inner dim is dirty, it dirties ALL cols of output
        if inner_slice != slice(None) and (inner_slice.stop > inner_slice.start):
            cols_dirty = slice(None)
        elif inner_slice == slice(None):
            cols_dirty = slice(None)

    # Handle B
    if rB is not None:
        if len(rB) > 2:
            return tuple(slice(None) for _ in range(len(rB)))

        inner_slice = rB[0]
        col_slice = rB[1] if len(rB) > 1 else slice(None)

        # B makes cols dirty directly
        # Merge with existing cols_dirty
        if cols_dirty == slice(None) or col_slice == slice(None):
            cols_dirty = slice(None)
        else:
            # Union
            start = min(cols_dirty.start, col_slice.start)
            stop = max(cols_dirty.stop, col_slice.stop)
            cols_dirty = slice(start, stop)

        # If B's inner dim is dirty, it dirties ALL rows of output
        if inner_slice == slice(None) or (inner_slice.stop > inner_slice.start):
            rows_dirty = slice(None)

    if rows_dirty == slice(0, 0) and cols_dirty == slice(0, 0):
        return None

    return (rows_dirty, cols_dirty)


def _propagate_concat(
    node: TensorNode, input_regions: List[DirtyRegion]
) -> DirtyRegion:
    axis = node.attrs.get("axis", 0)
    if axis < 0:
        # Cannot resolve negative axis without rank. Assume conservative.
        return tuple(slice(None) for _ in range(4))  # Arbitrary max rank fallback

    # We need shapes to calculate offsets.
    # Since we don't have shapes here easily without querying nodes,
    # and the prompt asks for propagation logic...
    # We must access node.parents shapes.

    out_slices = []
    current_offset = 0

    # Construct a full bounding box over the Concat axis
    dirty_start = None
    dirty_end = None

    # For non-concat axes, it's just a union (they must match)
    other_axes_dirty = None

    for i, p in enumerate(node.parents):
        shape_dim = p.shape[axis] if p.shape else 0
        if shape_dim is None:
            shape_dim = 0  # Dynamic? Warning.

        r = input_regions[i]

        if r is not None:
            # Check non-concat axes
            p_other = tuple(s for j, s in enumerate(r) if j != axis)
            if other_axes_dirty is None:
                other_axes_dirty = p_other
            else:
                # Union logic for tuples
                pass  # Simplified: assume elementwise union for others

            # Concat axis logic
            if len(r) > axis:
                s = r[axis]
                if s == slice(None):
                    s_start, s_end = 0, shape_dim
                else:
                    s_start, s_end = s.start, s.stop

                # Offset
                abs_start = current_offset + s_start
                abs_end = current_offset + s_end

                if dirty_start is None or abs_start < dirty_start:
                    dirty_start = abs_start
                if dirty_end is None or abs_end > dirty_end:
                    dirty_end = abs_end

        current_offset += shape_dim

    if dirty_start is None:
        return None

    # Construct result
    # Assuming rank is known from first parent?
    rank = (
        len(node.shape)
        if node.shape
        else len(input_regions[0])
        if input_regions[0]
        else 1
    )

    res = []
    for j in range(rank):
        if j == axis:
            res.append(slice(dirty_start, dirty_end))
        else:
            # Use union of other axes or slice(None) if unknown
            res.append(slice(None))
    return tuple(res)


def _propagate_slice(node: TensorNode, input_regions: List[DirtyRegion]) -> DirtyRegion:
    # If input changed within the window, output is dirty.
    # Map input indices to output indices.
    r = input_regions[0]
    if r is None:
        return None

    starts = node.attrs.get("starts")
    ends = node.attrs.get("ends")
    steps = node.attrs.get("steps")

    new_slices = []
    for i, s in enumerate(r):
        if i < len(starts):
            st, en, step = starts[i], ends[i], steps[i]
            if s == slice(None):
                new_slices.append(slice(None))
                continue

            # Intersection of dirty region (s) and slice window (st, en)
            # Input dirty: [s.start, s.stop)
            # Window: [st, en)

            # Overlap
            ov_start = max(s.start, st)
            ov_end = min(s.stop, en)

            if ov_start >= ov_end:
                # No overlap, this dimension is clean in output?
                # If a dimension is clean in output, does it imply the whole tensor is clean?
                # Ideally yes, but for now we mark it empty.
                return None

            # Map back to output coordinates
            # out_idx = (in_idx - st) // step
            out_start = (ov_start - st + step - 1) // step
            out_end = (ov_end - st + step - 1) // step

            new_slices.append(slice(out_start, out_end))
        else:
            new_slices.append(s)

    return tuple(new_slices)


def _propagate_reshape(
    node: TensorNode, input_regions: List[DirtyRegion]
) -> DirtyRegion:
    # Complex mapping. Fallback to full dirty if input is dirty.
    if any(r is not None for r in input_regions):
        return (
            tuple(slice(None) for _ in range(len(node.shape)))
            if node.shape
            else (slice(None),)
        )
    return None


def _propagate_reduce(
    node: TensorNode, input_regions: List[DirtyRegion]
) -> DirtyRegion:
    # SUM/MAX
    if not input_regions or input_regions[0] is None:
        return None
    r = input_regions[0]
    axis = node.attrs.get("axis")
    keepdims = node.attrs.get("keepdims", True)

    if axis is None:
        # Global reduction. Any dirty input -> Full dirty output
        return (
            tuple(slice(None) for _ in range(len(node.shape)))
            if node.shape
            else (slice(None),)
        )

    # Handle scalar axis
    if isinstance(axis, int):
        axis = [axis]

    # Check if any reduction axis is dirty
    is_reduction_dirty = False
    for a in axis:
        if a < len(r):
            s = r[a]
            if s == slice(None) or (s.stop > s.start):
                is_reduction_dirty = True
                break

    if is_reduction_dirty:
        # Full output dirty
        return (
            tuple(slice(None) for _ in range(len(node.shape)))
            if node.shape
            else (slice(None),)
        )
    else:
        # Reduction axis is clean, dirtyness is in other axes.
        # Propagate slices for non-reduction axes.
        res = []
        for i, s in enumerate(r):
            if i not in axis:
                res.append(s)
            elif keepdims:
                res.append(
                    slice(None)
                )  # Dimension becomes 1, so full "slice" of that dim
        return tuple(res)


_PROPAGATION_HANDLERS = {
    OpType.ADD: _propagate_elementwise,
    OpType.MUL: _propagate_elementwise,
    OpType.DIVIDE: _propagate_elementwise,
    OpType.POWER: _propagate_elementwise,
    OpType.WHERE: _propagate_elementwise,
    OpType.DOT: _propagate_matmul,
    OpType.CONCAT: _propagate_concat,
    OpType.SLICE: _propagate_slice,
    OpType.RESHAPE: _propagate_reshape,
    OpType.SUM: _propagate_reduce,
    OpType.MAX: _propagate_reduce,
}


# --- Input Slice Mapping Handlers ---


def _map_slice_elementwise(
    node: TensorNode, out_region: DirtyRegion
) -> List[DirtyRegion]:
    """
    Maps output dirty region back to inputs for elementwise ops,
    accounting for broadcasting and rank differences.
    """
    if out_region is None:
        return []

    res = []
    out_rank = len(out_region)

    for p in node.parents:
        p_shape = p.shape if p.shape is not None else ()
        p_rank = len(p_shape)

        if p_rank == 0:
            res.append(())  # Scalar
            continue

        # Align from the right (standard broadcasting rules)
        # If output is (Batch, Seq, Dim) and parent is (Dim,),
        # parent only cares about the last slice.
        p_slices = list(out_region[max(0, out_rank - p_rank) :])

        # If parent rank is higher than output rank (should not happen in valid graphs),
        # or if parent rank is lower, ensure we have the correct number of slices.
        if len(p_slices) < p_rank:
            # Prepend full slices for leading broadcasted dims
            p_slices = [slice(None)] * (p_rank - len(p_slices)) + p_slices

        # Handle dimensions of size 1 (broadcasted)
        # If a parent dim is 1, it provides the same value for all indices
        # in the output dim. Therefore, the "slice" for the parent is always the full 1-element.
        for i in range(p_rank):
            if p_shape[i] == 1:
                p_slices[i] = slice(None)

        res.append(tuple(p_slices))

    return res


def _map_slice_matmul(node: TensorNode, out_region: DirtyRegion) -> List[DirtyRegion]:
    # A @ B = C
    # Handle Batch Matmul (..., M, K) @ (..., K, N) -> (..., M, N)
    if out_region is None:
        return []

    out_rank = len(out_region)
    rows = out_region[-2] if out_rank >= 2 else out_region[0]
    cols = out_region[-1] if out_rank >= 2 else slice(None)

    # Batch dimensions (everything before M, N)
    batch_slices = out_region[:-2] if out_rank > 2 else ()

    # A needs (Batch, rows, full_K)
    # B needs (Batch, full_K, cols)

    # We apply the same rank-aware logic for the batch dimensions
    def _get_input_batch_slices(parent, out_batch_region):
        p_shape = parent.shape if parent.shape else ()
        p_batch_rank = max(0, len(p_shape) - 2)
        if p_batch_rank == 0:
            return []

        p_batch_slices = list(
            out_batch_region[max(0, len(out_batch_region) - p_batch_rank) :]
        )
        if len(p_batch_slices) < p_batch_rank:
            p_batch_slices = [slice(None)] * (
                p_batch_rank - len(p_batch_slices)
            ) + p_batch_slices

        for i in range(p_batch_rank):
            if p_shape[i] == 1:
                p_batch_slices[i] = slice(None)
        return p_batch_slices

    slice_a = tuple(_get_input_batch_slices(node.parents[0], batch_slices)) + (
        rows,
        slice(None),
    )
    slice_b = tuple(_get_input_batch_slices(node.parents[1], batch_slices)) + (
        slice(None),
        cols,
    )

    return [slice_a, slice_b]


def _map_slice_concat(node: TensorNode, out_region: DirtyRegion) -> List[DirtyRegion]:
    # Reverse of propagate concat.
    # If output is dirty at [S, E], find which inputs overlap and slice them.
    # If input doesn't overlap, slice is None (don't need it? Kernel expects list?)
    # Wait, kernel expects specific inputs.
    # If we pass a subset of inputs to Concat, the logic breaks.
    # We must pass ALL inputs, but sliced appropriately.
    # If an input is not involved in the dirty output region, we can pass a 0-size slice?
    # Or just pass the intersection.

    axis = node.attrs.get("axis", 0)
    if out_region is None:
        return []

    out_s = out_region[axis]
    if out_s == slice(None):
        # Full dirty, return full inputs
        return [
            (
                tuple(slice(None) for _ in range(len(p.shape)))
                if p.shape
                else (slice(None),)
            )
            for p in node.parents
        ]

    start, stop = out_s.start, out_s.stop

    res = []
    curr = 0
    for p in node.parents:
        dim = p.shape[axis] if p.shape else 0
        p_start = curr
        p_end = curr + dim
        curr += dim

        # Intersection [start, stop) with [p_start, p_end)
        ov_start = max(start, p_start)
        ov_end = min(stop, p_end)

        if ov_start < ov_end:
            # Relative slice
            rel_start = ov_start - p_start
            rel_end = ov_end - p_start

            # Construct full slice tuple
            p_slice = list(out_region)  # Copy
            p_slice[axis] = slice(rel_start, rel_end)
            res.append(tuple(p_slice))
        else:
            # No overlap -> Empty slice
            p_slice = list(out_region)
            p_slice[axis] = slice(0, 0)
            res.append(tuple(p_slice))

    return res


def _map_slice_full(node: TensorNode, out_region: DirtyRegion) -> List[DirtyRegion]:
    # Fallback: Just return full slices for all inputs
    return [
        tuple(slice(None) for _ in range(len(p.shape))) if p.shape else (slice(None),)
        for p in node.parents
    ]


_SLICE_MAPPING_HANDLERS = {
    OpType.ADD: _map_slice_elementwise,
    OpType.MUL: _map_slice_elementwise,
    OpType.DIVIDE: _map_slice_elementwise,
    OpType.POWER: _map_slice_elementwise,
    OpType.WHERE: _map_slice_elementwise,
    OpType.DOT: _map_slice_matmul,
    OpType.CONCAT: _map_slice_concat,
    OpType.SLICE: _map_slice_elementwise,  # Slice usually just needs the corresponding input region
    OpType.RESHAPE: _map_slice_full,  # Complex mapping, use full
    "GELU": _map_slice_elementwise,
    "RoPE": _map_slice_elementwise,
    "RMSNorm": _map_slice_elementwise,
}
