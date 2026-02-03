from typing import List, Dict, Any, Tuple, Optional, cast, Iterable
import math
import numpy as np
from ..ir.node import TensorNode
from ..ops.atomic_types import OpType
from ..ops.registry import get_reference_factory
from ..ir.graph import topological_sort


class ShapeInference:
    @staticmethod
    def infer(nodes: List[TensorNode], known_values: Dict[str, Any]):
        """
        Updates the shapes of nodes in-place based on known input values.
        """
        computed_values = known_values.copy()

        def get_val(node):
            if node.name in computed_values:
                return computed_values[node.name]
            if node.op_type == OpType.CONSTANT:
                val = node.attrs.get("value")
                # Normalize scalar constants
                if isinstance(val, (int, float, bool)):
                    val = np.array([val])
                elif isinstance(val, list):
                    val = np.array(val)

                computed_values[node.name] = val
                return val
            return None

        for node in nodes:
            # OpType.ARANGE
            if node.op_type == OpType.ARANGE:
                start = get_val(node.parents[0])
                stop = get_val(node.parents[1])
                step = get_val(node.parents[2])

                if start is not None and stop is not None and step is not None:
                    s = float(start.item()) if hasattr(start, "item") else float(start)
                    e = float(stop.item()) if hasattr(stop, "item") else float(stop)
                    st = float(step.item()) if hasattr(step, "item") else float(step)

                    if st != 0:
                        length = math.ceil((e - s) / st)
                        length = max(0, int(length))
                        node.shape = (length,)

            # OpType.FILL
            elif node.op_type == OpType.FILL:
                if len(node.parents) >= 2:
                    shape_val = get_val(node.parents[1])
                    if shape_val is not None:
                        if hasattr(shape_val, "cpu"):
                            shape_arr = cast(Any, shape_val).cpu().numpy()
                        else:
                            shape_arr = np.asarray(shape_val)

                        shape_tuple = tuple(
                            int(x) for x in shape_arr.astype(int).flatten()
                        )
                        node.shape = shape_tuple

            # OpType.REPEAT
            elif node.op_type == OpType.REPEAT:
                if node.parents and node.parents[0].shape:
                    data_shape = list(node.parents[0].shape)
                    repeats = int(node.attrs.get("repeats", 1))
                    axis = int(node.attrs.get("axis", 0))

                    ndim = len(data_shape)
                    # Handle negative axis
                    if axis < 0:
                        axis += ndim

                    if 0 <= axis < ndim:
                        dim_val = data_shape[axis]
                        if dim_val is not None:
                            data_shape[axis] = dim_val * repeats
                        node.shape = tuple(data_shape)
                    else:
                        # Fallback or invalid axis; preserve shape to avoid crash
                        node.shape = tuple(data_shape)

            # OpType.CONCAT
            elif node.op_type == OpType.CONCAT:
                if len(node.parents) >= 2:
                    a = node.parents[0]
                    # We only support same-rank concat for now
                    axis = node.attrs.get("axis", 0)

                    if a.shape is not None:
                        out_shape = list(a.shape)
                        ndim = len(a.shape)
                        if axis < 0:
                            axis += ndim

                        # Try to get dynamic axis from 3rd input if present
                        if len(node.parents) >= 3:
                            axis_val = get_val(node.parents[2])
                            if axis_val is not None:
                                axis = (
                                    int(axis_val.item())
                                    if hasattr(axis_val, "item")
                                    else int(axis_val)
                                )

                        # Fallback to static dim logic if simple concat
                        total_dim = 0
                        valid = True
                        for p in node.parents:
                            if p.shape is None or len(p.shape) <= axis:
                                valid = False
                                break

                            dim_p = p.shape[axis]
                            if dim_p is None:
                                valid = False
                                break
                            total_dim += dim_p

                        if valid and 0 <= axis < ndim:
                            out_shape[axis] = total_dim
                            node.shape = tuple(out_shape)

            # OpType.PERMUTE
            elif node.op_type == OpType.PERMUTE:
                if node.parents and node.parents[0].shape is not None:
                    dims = node.attrs.get("dims")
                    if dims:
                        input_shape = node.parents[0].shape
                        if len(input_shape) == len(dims):
                            node.shape = tuple(input_shape[d] for d in dims)

            # OpType.TRIU
            elif node.op_type == OpType.TRIU:
                if node.parents and node.parents[0].shape is not None:
                    node.shape = node.parents[0].shape

            # OpType.GATHER
            elif node.op_type == OpType.GATHER:
                if len(node.parents) >= 2:
                    data_shape = node.parents[0].shape
                    indices_shape = node.parents[1].shape
                    if data_shape and indices_shape:
                        # Standard gather behavior on axis 0
                        # Output shape = indices_shape + data_shape[1:]
                        node.shape = indices_shape + data_shape[1:]

            # OpType.DOT
            elif node.op_type == OpType.DOT:
                if len(node.parents) == 2:
                    s0 = node.parents[0].shape
                    s1 = node.parents[1].shape
                    if s0 and s1:
                        if any(not isinstance(d, int) for d in s0) or any(
                            not isinstance(d, int) for d in s1
                        ):
                            raise ValueError(
                                f"DOT requires concrete shapes, found None in: {s0}, {s1}"
                            )

                        # Cast to concrete tuples for the type checker
                        s0 = cast(Tuple[int, ...], s0)
                        s1 = cast(Tuple[int, ...], s1)

                        # Handle basic MatMul logic
                        # If 2D: (M, K) @ (K, N) -> (M, N)
                        if len(s0) == 2 and len(s1) == 2:
                            node.shape = (s0[0], s1[1])
                        # Handle broadcasting (batch dims)
                        elif len(s0) >= 2 and len(s1) >= 2:
                            batch_shape = np.broadcast_shapes(s0[:-2], s1[:-2])
                            node.shape = batch_shape + (s0[-2], s1[-1])

            # OpType.RESHAPE
            elif node.op_type == OpType.RESHAPE:
                data = node.parents[0]
                shape_tensor_val = get_val(node.parents[1])

                if shape_tensor_val is not None:
                    if hasattr(shape_tensor_val, "cpu"):
                        shape_arr = cast(
                            np.ndarray, cast(Any, shape_tensor_val).cpu().numpy()
                        )
                    else:
                        shape_arr = np.asarray(shape_tensor_val)

                    target_dims = shape_arr.astype(int).flatten().tolist()

                    if -1 in target_dims:
                        if data.shape and all(d is not None for d in data.shape):
                            total_elems = int(
                                np.prod([d for d in data.shape if d is not None])
                            )
                            known_prod = 1
                            for d in target_dims:
                                if d != -1:
                                    known_prod *= d

                            if known_prod != 0 and total_elems % known_prod == 0:
                                missing_dim = total_elems // known_prod
                                target_dims = [
                                    d if d != -1 else missing_dim for d in target_dims
                                ]

                    node.shape = tuple(target_dims)

            # OpType.SLICE
            elif node.op_type == OpType.SLICE:
                if len(node.parents) == 1 and node.parents[0].shape is not None:
                    data_shape = node.parents[0].shape
                    starts = node.attrs.get("starts", [0] * len(data_shape))
                    ends = node.attrs.get("ends", [d for d in data_shape])
                    steps = node.attrs.get("steps", [1] * len(data_shape))

                    new_shape = []
                    for i, dim in enumerate(data_shape):
                        if dim is None:
                            new_shape.append(None)
                            continue

                        s = starts[i] if i < len(starts) else 0
                        e = ends[i] if i < len(ends) else dim
                        st = steps[i] if i < len(steps) else 1

                        if e is None:
                            e = dim
                        if s is None:
                            s = 0
                        if st is None or st == 0:
                            st = 1

                        # Handle negative indices
                        if s < 0:
                            s += dim
                        if e < 0:
                            e += dim

                        s = max(0, min(s, dim))
                        e = max(0, min(e, dim))

                        length = 0
                        if st > 0:
                            if s < e:
                                length = math.ceil((e - s) / st)
                        else:
                            if s > e:
                                length = math.ceil((s - e) / abs(st))

                        new_shape.append(int(length))
                    node.shape = tuple(new_shape)

            # OpType.SUM / MAX
            elif node.op_type in (OpType.SUM, OpType.MAX):
                if len(node.parents) > 0 and node.parents[0].shape is not None:
                    data_shape = list(node.parents[0].shape)
                    ndim = len(data_shape)

                    axis = None
                    if node.attrs and "axis" in node.attrs:
                        axis = node.attrs["axis"]
                    elif len(node.parents) > 1:
                        axis_val = get_val(node.parents[1])
                        if axis_val is not None:
                            axis = (
                                int(axis_val.item())
                                if hasattr(axis_val, "item")
                                else int(axis_val)
                            )

                    keepdims = True
                    if node.op_type == OpType.SUM:
                        keepdims = node.attrs.get("keepdims", True)
                    elif node.op_type == OpType.MAX:
                        # Max kernel implementation implies keepdims=True usually (based on tests)
                        # but let's check attributes
                        keepdims = node.attrs.get("keepdims", True)

                    if axis is None:
                        if keepdims:
                            node.shape = tuple(1 for _ in range(ndim))
                        else:
                            node.shape = (1,)
                    else:
                        if hasattr(axis, "__iter__") and not isinstance(
                            axis, (str, bytes)
                        ):
                            axes = list(cast(Iterable[Any], axis))
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

            # OpType.CONSTANT
            elif node.op_type == OpType.CONSTANT:
                node.shape = (1,)

            # Unary Propagators (Generic)
            elif node.op_type in (
                OpType.CAST,
                OpType.COPY_TO,
                OpType.NEGATE,
                OpType.EXP,
                OpType.SIN,
                OpType.COS,
                OpType.SQRT,
            ):
                if node.parents and node.parents[0].shape:
                    node.shape = node.parents[0].shape

            # Broadcasting Ops
            elif node.op_type in (
                OpType.ADD,
                OpType.MUL,
                OpType.DIVIDE,
                OpType.POWER,
                OpType.WHERE,
            ):
                parents_to_check = node.parents
                shapes = [p.shape for p in parents_to_check if p.shape is not None]
                if len(shapes) == len(parents_to_check) and len(shapes) > 0:
                    if hasattr(np, "broadcast_shapes"):
                        # Verify no dynamic dimensions and cast for type checker
                        for sh in shapes:
                            if any(d is None for d in sh):
                                raise ValueError(
                                    f"Dynamic dimensions in broadcasting op '{node.name}' not supported."
                                )

                        node.shape = cast(
                            Tuple[int, ...],
                            np.broadcast_shapes(*cast(List[Tuple[int, ...]], shapes)),
                        )

            elif node.op_type != OpType.INPUT:

                factory = get_reference_factory(node.op_type)
                if factory:
                    # Decompose the node into its reference implementation
                    sub_root = factory(node.parents, node.attrs)
                    # Infer shapes for the entire decomposed subgraph
                    sub_nodes = topological_sort(sub_root)
                    # Recursively process sub-nodes using the same value context
                    ShapeInference.infer(sub_nodes, known_values)
                    # High-level node inherits the shape of its decomposition's root
                    node.shape = sub_root.shape
