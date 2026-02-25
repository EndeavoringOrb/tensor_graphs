from typing import Tuple, Optional, List, Any
import numpy as np
from ..ir.node import TensorNode
from .propagation import GraphPropagator

DirtyRegion = Optional[List[Tuple[Tuple[int, int], ...]]]


class DirtyPropagator:
    @staticmethod
    def get_diff(old_data: Any, new_data: Any) -> DirtyRegion:
        """Compute the bounding-box dirty region. Returns a list containing one box."""
        if old_data is None:
            ndim = new_data.ndim if hasattr(new_data, "ndim") else 0
            if ndim > 0:
                return [tuple((0, int(new_data.shape[i])) for i in range(ndim))]
            else:
                return [((0, 1),)]

        if hasattr(new_data, "shape") and hasattr(old_data, "shape"):
            if new_data.shape != old_data.shape:
                return [
                    tuple((0, int(new_data.shape[i])) for i in range(new_data.ndim))
                ]

        n_new = (
            new_data.cpu().numpy() if hasattr(new_data, "cpu") else np.array(new_data)
        )
        n_old = (
            old_data.cpu().numpy() if hasattr(old_data, "cpu") else np.array(old_data)
        )

        if n_new.shape != n_old.shape:
            return [tuple((0, int(n_new.shape[i])) for i in range(n_new.ndim))]

        diff = n_new != n_old
        if not np.any(diff):
            return None

        box: list = []
        for dim in range(diff.ndim):
            axes_to_reduce = tuple(i for i in range(diff.ndim) if i != dim)
            dim_diff = np.any(diff, axis=axes_to_reduce) if axes_to_reduce else diff
            indices = np.where(dim_diff)[0]
            if len(indices) == 0:
                box.append((0, 0))
            else:
                box.append((int(indices[0]), int(indices[-1] + 1)))

        return [tuple(box)]

    @staticmethod
    def propagate(node: TensorNode, known_values: Optional[dict] = None) -> DirtyRegion:
        return GraphPropagator.propagate(node, known_values)

    @staticmethod
    def get_input_slices(
        node: TensorNode,
        output_region: DirtyRegion,
        known_values: Optional[dict] = None,
    ) -> List[DirtyRegion]:
        return GraphPropagator.get_input_slices(node, output_region, known_values)
