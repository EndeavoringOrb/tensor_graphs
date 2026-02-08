"""
File: tensor_graphs/compiler/dirty_propagation.py
"""

from typing import Tuple, Optional, List, Any
import numpy as np
from ..ir.node import TensorNode
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
                stop = node.shape[i] if node.shape and i < len(node.shape) else None
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
    def get_input_slices(
        node: TensorNode, output_region: DirtyRegion
    ) -> List[DirtyRegion]:
        """
        Given that 'node' needs to compute 'output_region', calculate the
        slices required from each parent using the SymbolicPropagator.
        """
        if output_region is None:
            return []

        # 1. Get Compiled Backward Propagator
        propagator = SymbolicPropagator.get_backward_propagator(node)

        # 2. Prepare Arguments
        out_rank = len(node.shape) if node.shape is not None else 0
        out_args = []
        for d in range(out_rank):
            if d < len(output_region):
                s = output_region[d]
                dim_size = node.shape[d] if node.shape[d] is not None else 1_000_000_000
                # s.indices() handles None and negative indices correctly
                start, stop, _ = s.indices(dim_size)
                out_args.extend([float(start), float(stop)])
            else:
                # If output_region is smaller than rank (shouldn't happen), assume full
                out_args.extend([0.0, float(node.shape[d] or 1_000_000_000)])

        shape_args = []
        for p in node.parents:
            p_shape = p.shape if p.shape is not None else ()
            for d in p_shape:
                shape_args.append(float(d if d is not None else 1_000_000_000))

        out_shape_args = [
            float(d if d is not None else 1_000_000_000) for d in (node.shape or ())
        ]

        # 3. Execute
        res_flat = propagator(*(out_args + shape_args + out_shape_args))

        # 4. Map back to DirtyRegion list
        input_regions = []
        idx = 0
        for p in node.parents:
            p_shape = p.shape if p.shape is not None else ()
            rank = len(p_shape)

            p_slices = []
            for d in range(rank):
                s, e = res_flat[idx], res_flat[idx + 1]
                idx += 2

                # Convert to slice
                # We use (0, 0) for empty ranges to avoid Executor using full buffer
                if s >= e or s == np.inf or e == -np.inf:
                    p_slices.append(slice(0, 0))
                    continue

                start = int(max(0, s)) if s != -np.inf else 0
                stop = int(max(0, e)) if e != np.inf else p_shape[d]
                p_slices.append(slice(start, stop))

            if rank == 0:
                input_regions.append(())
            else:
                input_regions.append(tuple(p_slices))

        return input_regions
