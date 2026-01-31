import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, cast
from ..ir.node import TensorNode
from ..ops.atomic_types import OpType


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
                    try:
                        s = (
                            float(start.item())
                            if hasattr(start, "item")
                            else float(start)
                        )
                        e = float(stop.item()) if hasattr(stop, "item") else float(stop)
                        st = (
                            float(step.item()) if hasattr(step, "item") else float(step)
                        )

                        if st != 0:
                            length = math.ceil((e - s) / st)
                            length = max(0, int(length))
                            node.shape = (length,)
                    except Exception:
                        pass

            # OpType.FILL
            elif node.op_type == OpType.FILL:
                # Fill operation has two inputs: value tensor and shape tensor.
                # The shape tensor may be a constant (provided via known values).
                if len(node.parents) >= 2:
                    shape_val = get_val(node.parents[1])
                    if shape_val is not None:
                        try:
                            # Accept both torch tensors and numpy arrays
                            if hasattr(shape_val, "cpu"):
                                shape_arr = shape_val.cpu().numpy()
                            else:
                                shape_arr = np.asarray(shape_val)

                            # Cast to int, flatten, and build a tuple
                            shape_tuple = tuple(
                                int(x) for x in shape_arr.astype(int).flatten()
                            )
                            node.shape = shape_tuple
                        except Exception:
                            # If conversion fails, keep existing shape (likely (None,))
                            pass

            # OpType.REPEAT
            elif node.op_type == OpType.REPEAT:
                data = node.parents[0] if node.parents else None
                if data and data.shape and node.attrs and "repeats" in node.attrs:
                    repeats = int(node.attrs["repeats"])
                    # Assuming 1D / Flattened repeat for simple shape logic unless axis is handled
                    # (Codebase implementation was rudimentary, keeping it simple here)
                    input_shape = data.shape
                    if input_shape and input_shape[0] is not None:
                        node.shape = (input_shape[0] * repeats,)

            # OpType.CONCAT
            elif node.op_type == OpType.CONCAT:
                if len(node.parents) >= 2:
                    a = node.parents[0]
                    # We only support same-rank concat for now
                    axis = node.attrs.get("axis", 0)
                    
                    if a.shape is not None:
                         out_shape = list(a.shape)
                         ndim = len(a.shape)
                         if axis < 0: axis += ndim
                         
                         total_dim = 0
                         valid = True
                         for p in node.parents:
                             if p.shape is None or p.shape[axis] is None:
                                 valid = False; break
                             total_dim += p.shape[axis]
                        
                         if valid and 0 <= axis < ndim:
                             out_shape[axis] = total_dim
                             node.shape = tuple(out_shape)

            # OpType.RESHAPE
            elif node.op_type == OpType.RESHAPE:
                data = node.parents[0]
                shape_tensor_val = get_val(node.parents[1])

                if shape_tensor_val is not None:
                    try:
                        # Handle both torch tensors and numpy arrays
                        if hasattr(shape_tensor_val, "cpu"):
                            # Type cast to handle torch tensor
                            shape_arr = cast(np.ndarray, shape_tensor_val.cpu().numpy())
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
                                        d if d != -1 else missing_dim
                                        for d in target_dims
                                    ]

                        node.shape = tuple(target_dims)
                    except Exception:
                        pass

            # Unary Propagators (Generic)
            elif node.op_type in (
                OpType.CAST,
                OpType.COPY_TO,
                OpType.NEGATE,
                OpType.EXP,
                OpType.SIN,
                OpType.COS,
                OpType.SQRT,
                "GELU",
                "Softmax",
                "RMSNorm",
                "RoPE",
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
                "FusedMulAdd",
            ):
                parents_to_check = node.parents
                shapes = [p.shape for p in parents_to_check if p.shape is not None]
                if len(shapes) == len(parents_to_check) and len(shapes) > 0:
                    try:
                        if hasattr(np, "broadcast_shapes"):
                            # Filter out None values from shapes for broadcast_shapes
                            # Convert to int to satisfy type checker
                            filtered_shapes: Tuple[int, ...] = tuple(
                                int(s) if s is not None else 1
                                for s in shapes
                                if s is not None
                            )
                            node.shape = cast(
                                Tuple[int, ...], np.broadcast_shapes(*filtered_shapes)
                            )
                    except:
                        pass

            # OpType.CONCAT
            elif node.op_type == OpType.CONCAT:
                # Concatenate A and B along specified axis
                if len(node.parents) >= 2:
                    a, b = node.parents[0], node.parents[1]

                    axis = 0

                    # 1. Try to get axis from 3rd input (Runtime Value)
                    axis_val = None
                    if len(node.parents) >= 3:
                        axis_val = get_val(node.parents[2])

                    if axis_val is not None:
                        try:
                            axis = (
                                int(axis_val.item())
                                if hasattr(axis_val, "item")
                                else int(axis_val)
                            )
                        except Exception:
                            pass
                    # 2. Try to get axis from attributes (Static Value)
                    elif node.attrs and "axis" in node.attrs:
                        val = node.attrs["axis"]
                        if isinstance(val, list) and len(val) > 0:
                            axis = val[0]
                        else:
                            axis = val

                    if a.shape is not None and b.shape is not None:
                        if len(a.shape) == len(b.shape):
                            out_shape = list(a.shape)

                            ndim = len(a.shape)
                            if axis < 0:
                                axis += ndim

                            if 0 <= axis < ndim:
                                da = a.shape[axis]
                                db = b.shape[axis]
                                if da is not None and db is not None:
                                    out_shape[axis] = da + db
                                else:
                                    out_shape[axis] = None
                                node.shape = tuple(out_shape)
