from typing import Tuple, Optional, List, Any
import numpy as np
from ..ir.node import TensorNode
from .propagation import GraphPropagator, _to_slices, _from_slices

DirtyRegion = Optional[List[Tuple[slice, ...]]]


class DirtyPropagator:
    @staticmethod
    def get_diff(old_data: Any, new_data: Any) -> DirtyRegion:
        """Compute the bounding-box dirty region. Returns a list containing one box."""
        if old_data is None:
            ndim = new_data.ndim if hasattr(new_data, "ndim") else 0
            return [(slice(None),) * ndim] if ndim > 0 else [(slice(None),)]

        if hasattr(new_data, "shape") and hasattr(old_data, "shape"):
            if new_data.shape != old_data.shape:
                return [(slice(None),) * new_data.ndim]

        n_new = (
            new_data.cpu().numpy() if hasattr(new_data, "cpu") else np.array(new_data)
        )
        n_old = (
            old_data.cpu().numpy() if hasattr(old_data, "cpu") else np.array(old_data)
        )

        if n_new.shape != n_old.shape:
            return [(slice(None),) * n_new.ndim]

        diff = n_new != n_old
        if not np.any(diff):
            return None

        slices: list = []
        for dim in range(diff.ndim):
            axes_to_reduce = tuple(i for i in range(diff.ndim) if i != dim)
            dim_diff = np.any(diff, axis=axes_to_reduce) if axes_to_reduce else diff
            indices = np.where(dim_diff)[0]
            if len(indices) == 0:
                slices.append(slice(0, 0))
            else:
                slices.append(slice(int(indices[0]), int(indices[-1] + 1)))

        # Return as a list of one box
        return [tuple(slices)]

    @staticmethod
    def propagate(node: TensorNode, known_values: Optional[dict] = None) -> DirtyRegion:
        result = GraphPropagator.propagate(node, known_values)
        return _to_slices(result)

    @staticmethod
    def get_input_slices(
        node: TensorNode,
        output_region: DirtyRegion,
        known_values: Optional[dict] = None,
    ) -> List[DirtyRegion]:
        numeric_output = _from_slices(output_region, node.shape or ())
        results = GraphPropagator.get_input_slices(node, numeric_output, known_values)
        return [_to_slices(r) for r in results]
