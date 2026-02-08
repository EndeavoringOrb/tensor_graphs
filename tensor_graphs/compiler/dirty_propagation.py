"""
File: tensor_graphs/compiler/dirty_propagation.py
"""

from typing import Tuple, Optional, List, Any
import numpy as np
from ..ir.node import TensorNode
from ..ops.atomic_types import OpType
from .symbolic import SymbolicPropagator

# A DirtyRegion is a tuple of slices, one per dimension.
# None implies the node is CLEAN.
DirtyRegion = Optional[Tuple[slice, ...]]


class DirtyPropagator:
    """
    Handles the logic for incremental execution:
    1. Calculating diffs for Inputs.
    2. Propagating dirty regions forward through Atomic Ops using SymbolicPropagator.
    3. Mapping Output Dirty Regions back to Input Slices for execution.
    """

    @staticmethod
    def get_diff(old_data: Any, new_data: Any) -> DirtyRegion:
        """
        Compares old_data and new_data buffers to find the minimal dirty slice.
        """
        if old_data is None:
            ndim = new_data.ndim if hasattr(new_data, "ndim") else 0
            if ndim == 0:
                return (slice(None),)
            return tuple(slice(None) for _ in range(ndim))

        if hasattr(new_data, "shape") and hasattr(old_data, "shape"):
            if new_data.shape != old_data.shape:
                return tuple(slice(None) for _ in range(new_data.ndim))

        try:
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

            slices = []
            for dim in range(diff.ndim):
                axes_to_reduce = tuple(i for i in range(diff.ndim) if i != dim)
                if axes_to_reduce:
                    dim_diff = np.any(diff, axis=axes_to_reduce)
                else:
                    dim_diff = diff

                indices = np.where(dim_diff)[0]
                if len(indices) == 0:
                    slices.append(slice(0, 0))
                else:
                    start = indices[0]
                    stop = indices[-1] + 1
                    slices.append(slice(int(start), int(stop)))

            return tuple(slices)

        except Exception:
            ndim = new_data.ndim if hasattr(new_data, "ndim") else 0
            return tuple(slice(None) for _ in range(ndim))

    @staticmethod
    def propagate(node: TensorNode) -> DirtyRegion:
        """
        Calculates the dirty region of 'node' based on the dirty regions of its parents
        using the SymbolicPropagator.
        """
        # Constants and Inputs (without parents) don't propagate
        if not node.parents:
            # If INPUT, it should have dirty_region set by Executor.
            # If CONSTANT, it is clean (None).
            return node.dirty_region

        # 1. Get Compiled Propagator
        propagator = SymbolicPropagator.get_propagator(node)

        # 2. Prepare Arguments
        dirty_args = []
        shape_args = []

        # Iterate parents to collect args (must match _compile order in symbolic.py)
        # First pass: Dirty Ranges
        for p in node.parents:
            p_shape = p.shape if p.shape is not None else ()
            rank = len(p_shape)

            if p.dirty_region is None:
                # Clean: (inf, -inf)
                for _ in range(rank):
                    dirty_args.extend([np.inf, -np.inf])
            else:
                # Dirty: Convert slices to (start, stop)
                # Handle full slice(None) or partial
                for d in range(rank):
                    if d < len(p.dirty_region):
                        s = p.dirty_region[d]
                        dim_size = (
                            p_shape[d]
                            if d < len(p_shape) and p_shape[d] is not None
                            else np.inf
                        )

                        start, stop, step = s.indices(
                            dim_size if dim_size != np.inf else 1_000_000_000
                        )

                        if step != 1:
                            # Conservative bounding box for stepped slice
                            # If step > 0: [start, start + ceil((stop-start)/step)*step] ?
                            # Indices produces correct range bounds [start, stop) usually
                            # But slice(10, 0, -1) is tricky.
                            # We assume standard positive steps for dirty regions usually.
                            pass

                        dirty_args.extend([start, stop])
                    else:
                        # Should not happen if dirty_region matches rank
                        dirty_args.extend([0, np.inf])

        # Second pass: Shapes
        for p in node.parents:
            p_shape = p.shape if p.shape is not None else ()
            for d in p_shape:
                shape_args.extend([d if d is not None else np.inf])

        # 3. Execute Propagator
        # Returns flat list [s0, e0, s1, e1...]
        res_flat = propagator(*(dirty_args + shape_args))

        if not res_flat:
            # Scalar clean? Or empty?
            if node.shape == ():
                # Scalar
                # If list is empty, symbolic says nothing dirty?
                return None
            # If node has rank but res empty, means clean?
            # Symbolic logic returns pairs. If empty, maybe scalar?
            pass

        # 4. Convert back to DirtyRegion
        out_slices = []
        # Result is [s0, e0, s1, e1, ...]

        # Check if clean (any dim has start >= end)
        is_clean = False
        num_dims = len(res_flat) // 2

        # If output is scalar (rank 0), num_dims is 0.
        # But symbolic_elementwise returns clean region (empty list) if inputs clean?
        # symbolic_elementwise checks "if not inputs: return clean".
        # clean region has ranges [(inf, -inf)].
        # So scalar clean -> [(inf, -inf)].

        if num_dims == 0:
            # Check if scalar dirty?
            # How does symbolic represent scalar dirty?
            # SymbolicRegion([]) means scalar.
            # _is_dirty checks if ranges empty -> True.
            # So if [] returned, it implies Dirty Scalar if logic holds.
            # But wait, _clean_region(0) returns [].
            # _full_dirty_region(0) returns [].
            # Ambiguity for scalars?
            # symbolic_elementwise: if clean inputs, returns Clean(0).
            # Clean(0) is [].
            # So [] implies Clean?
            # But Dirty(0) is also [].
            # Actually, SymbolicPropagator logic for scalars needs care.
            # For now, if node is scalar:
            # Assume dirty unless parents clean?
            # If parents clean, dirty_args would be (inf, -inf).
            pass

        for i in range(num_dims):
            s, e = res_flat[2 * i], res_flat[2 * i + 1]

            # Check for cleanliness
            if s >= e:
                is_clean = True
                break

            # Clamp and Convert
            # s, e are numpy scalars or inf
            if s == -np.inf:
                s = 0  # boundless start
            if s == np.inf:
                is_clean = True
                break

            start = int(max(0, s))

            if e == np.inf:
                stop = None  # Slice end
            elif e == -np.inf:
                is_clean = True
                break
            else:
                stop = int(max(0, e))

            out_slices.append(slice(start, stop))

        if is_clean:
            return None

        return tuple(out_slices) if out_slices else None

    @staticmethod
    def get_input_slices(node: TensorNode, output_region: DirtyRegion) -> List[Any]:
        """
        Given that 'node' needs to compute 'output_region', calculate the
        slices required from each parent.
        """
        if output_region is None:
            return []

        handler = _SLICE_MAPPING_HANDLERS.get(node.op_type, _map_slice_elementwise)
        return handler(node, output_region)


# --- Helper Functions for Reverse Mapping (unchanged) ---


def _map_slice_elementwise(
    node: TensorNode, out_region: DirtyRegion
) -> List[DirtyRegion]:
    if out_region is None:
        return []
    res = []
    out_rank = len(out_region)
    for p in node.parents:
        p_shape = p.shape if p.shape is not None else ()
        p_rank = len(p_shape)
        if p_rank == 0:
            res.append(())
            continue
        p_slices = list(out_region[max(0, out_rank - p_rank) :])
        if len(p_slices) < p_rank:
            p_slices = [slice(None)] * (p_rank - len(p_slices)) + p_slices
        for i in range(p_rank):
            if p_shape[i] == 1:
                p_slices[i] = slice(None)
        res.append(tuple(p_slices))
    return res


def _map_slice_matmul(node: TensorNode, out_region: DirtyRegion) -> List[DirtyRegion]:
    if out_region is None:
        return []
    out_rank = len(out_region)
    rows = out_region[-2] if out_rank >= 2 else out_region[0]
    cols = out_region[-1] if out_rank >= 2 else slice(None)
    batch_slices = out_region[:-2] if out_rank > 2 else ()

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
    axis = node.attrs.get("axis", 0)
    if out_region is None:
        return []
    out_s = out_region[axis]
    if out_s == slice(None):
        return [
            tuple(slice(None) for _ in range(len(p.shape)))
            if p.shape
            else (slice(None),)
            for p in node.parents
        ]

    start, stop = out_s.start, out_s.stop
    if start is None:
        start = 0
    # Stop None needs handling based on total shape, but here we do relative

    res = []
    curr = 0
    for p in node.parents:
        dim = p.shape[axis] if p.shape else 0
        p_start = curr
        p_end = curr + dim
        curr += dim

        # If stop is None, it means end of concat
        eff_stop = stop if stop is not None else 1_000_000_000

        ov_start = max(start, p_start)
        ov_end = min(eff_stop, p_end)

        if ov_start < ov_end:
            rel_start = ov_start - p_start
            rel_end = ov_end - p_start
            p_slice = list(out_region)
            p_slice[axis] = slice(rel_start, rel_end)
            res.append(tuple(p_slice))
        else:
            p_slice = list(out_region)
            p_slice[axis] = slice(0, 0)
            res.append(tuple(p_slice))
    return res


def _map_slice_full(node: TensorNode, out_region: DirtyRegion) -> List[DirtyRegion]:
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
    OpType.SLICE: _map_slice_elementwise,
    OpType.RESHAPE: _map_slice_full,
}
