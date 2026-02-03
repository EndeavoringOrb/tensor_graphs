from typing import List, Dict, Any, Tuple, Optional, cast, Iterable, Union, Callable
import math
import numpy as np
import sympy
import uuid
from ..ir.node import TensorNode
from ..ops.atomic_types import OpType
from ..ops.registry import get_reference_factory
from ..ir.graph import topological_sort
from tqdm import tqdm
from ..config import *

class _SymShape:
    _symbol_cache = {}

    @staticmethod
    def to_symbolic(dim: Optional[Union[int, sympy.Expr]], prefix: str = "d") -> sympy.Expr:
        if dim is None:
            # Deterministic naming based on prefix to prevent UUID bloat
            count = _SymShape._symbol_cache.get(prefix, 0)
            _SymShape._symbol_cache[prefix] = count + 1
            return sympy.Symbol(f"{prefix}_{count}", integer=True, positive=True)
        if isinstance(dim, int):
            return sympy.Integer(dim)
        return dim

    @staticmethod
    def prod(iterable: Iterable[Union[int, sympy.Expr]]) -> sympy.Expr:
        """Symbolic product."""
        res = sympy.Integer(1)
        for x in iterable:
            res = res * x
        return res

    @staticmethod
    def broadcast(
        shape1: Tuple[sympy.Expr, ...], shape2: Tuple[sympy.Expr, ...]
    ) -> Tuple[sympy.Expr, ...]:
        """Broadcasts two symbolic shapes according to numpy rules."""
        ndim1, ndim2 = len(shape1), len(shape2)
        out_ndim = max(ndim1, ndim2)

        # Prepend 1s
        s1 = (sympy.Integer(1),) * (out_ndim - ndim1) + shape1
        s2 = (sympy.Integer(1),) * (out_ndim - ndim2) + shape2

        out_shape = []
        for d1, d2 in zip(s1, s2):
            if d1 == 1:
                out_shape.append(d2)
            elif d2 == 1:
                out_shape.append(d1)
            elif d1 == d2:
                out_shape.append(d1)
            else:
                # Symbolic Equality Check
                # In a dynamic graph, we assume they are compatible.
                # We prioritize the non-integer symbol if one is concrete 1 and the other isn't,
                # but here d1 != 1 and d2 != 1. We usually pick d1.
                out_shape.append(d1)

        return tuple(out_shape)


class ShapeInference:
    _handlers: Dict[str, Callable] = {}

    @classmethod
    def register_handler(cls, op_type: str):
        def decorator(func):
            cls._handlers[op_type] = func
            return func

        return decorator

    @staticmethod
    def infer(nodes: List[TensorNode], known_values: Dict[str, Any]):
        """
        Updates the shapes of nodes in-place based on symbolic inference.
        If known_values are provided, attempts to resolve symbols to concrete integers.
        """
        # 1. Normalize Constants in known_values
        computed_values = known_values.copy()

        def get_val(node):
            if node.name in computed_values:
                return computed_values[node.name]
            if node.op_type == OpType.CONSTANT:
                val = node.attrs.get("value")
                if isinstance(val, (int, float, bool)):
                    val = np.array([val])
                elif isinstance(val, list):
                    val = np.array(val)
                computed_values[node.name] = val
                return val
            return None

        # 2. Inference Loop
        for node in tqdm(nodes, desc="Inferring symbolic shapes", disable=not DEBUG_EXECUTION):

            # Ensure shape is a tuple of Sympy Expressions or None (initially)
            if node.shape is None:
                pass
            else:
                # Normalize existing int/None shape to Sympy
                node.shape = tuple(
                    _SymShape.to_symbolic(d, prefix=f"in_{node.name}")
                    for d in node.shape
                )

            # --- Dispatch ---
            if node.op_type in ShapeInference._handlers:
                ShapeInference._handlers[node.op_type](node, get_val)

            # --- Fallback: Decomposition ---
            elif node.op_type != OpType.INPUT and node.op_type != OpType.CONSTANT:
                factory = get_reference_factory(node.op_type)
                if factory:
                    # Decompose
                    sub_root = factory(node.parents, node.attrs)
                    sub_nodes = topological_sort(sub_root)
                    ShapeInference.infer(sub_nodes, computed_values)
                    node.shape = sub_root.shape
                else:
                    # Default: Propagate from 0th input if available (unary-like)
                    if node.parents and node.parents[0].shape:
                        node.shape = node.parents[0].shape

        # 3. Final Resolution (Optional) could go here
        pass


# ==============================================================================
# Op Handlers
# ==============================================================================


@ShapeInference.register_handler(OpType.INPUT)
def handle_input(node: TensorNode, get_val):
    val = get_val(node)
    if val is not None and hasattr(val, "shape"):
        node.shape = tuple(sympy.Integer(d) for d in val.shape)
        return
    if node.shape:
        new_shape = []
        for i, d in enumerate(node.shape):
            if d is None or (isinstance(d, sympy.Symbol) and d.name.startswith("in_")):
                # Generate a clean symbol for the input dimension
                sym_name = f"{node.name}_dim{i}"
                new_shape.append(sympy.Symbol(sym_name, integer=True, positive=True))
            else:
                new_shape.append(_SymShape.to_symbolic(d))
        node.shape = tuple(new_shape)
    else:
        node.shape = ()


@ShapeInference.register_handler(OpType.CONSTANT)
def handle_constant(node: TensorNode, get_val):
    val = get_val(node)
    if val is not None:
        if hasattr(val, "shape"):
            node.shape = tuple(sympy.Integer(d) for d in val.shape)
        elif isinstance(val, (list, tuple)):
            node.shape = (sympy.Integer(len(val)),)
        else:
            node.shape = ()
    elif node.shape is None:
        node.shape = (sympy.Integer(1),)
    else:
        node.shape = tuple(_SymShape.to_symbolic(d) for d in node.shape)


# --- Arithmetic & Broadcasting ---
def _handle_broadcast(node: TensorNode, get_val):
    shapes = [p.shape for p in node.parents if p.shape is not None]
    if not shapes:
        return

    current_shape = shapes[0]
    for s in shapes[1:]:
        current_shape = _SymShape.broadcast(current_shape, s)
    node.shape = current_shape


for op in [OpType.ADD, OpType.MUL, OpType.DIVIDE, OpType.POWER, OpType.WHERE]:
    ShapeInference.register_handler(op)(_handle_broadcast)


# --- Unary / Elementwise Preserving ---
def _handle_unary(node: TensorNode, get_val):
    if node.parents and node.parents[0].shape is not None:
        node.shape = node.parents[0].shape


for op in [
    OpType.CAST,
    OpType.COPY_TO,
    OpType.NEGATE,
    OpType.EXP,
    OpType.SIN,
    OpType.COS,
    OpType.SQRT,
    OpType.TRIU,
    "GELU",
    "Softmax",
    "RoPE",
]:
    ShapeInference.register_handler(op)(_handle_unary)


# --- Generation & Structure ---


@ShapeInference.register_handler(OpType.FILL)
def handle_fill(node: TensorNode, get_val):
    if len(node.parents) >= 2:
        shape_val = get_val(node.parents[1])
        if shape_val is not None:
            if hasattr(shape_val, "cpu"):
                shape_arr = shape_val.cpu().numpy()
            else:
                shape_arr = np.asarray(shape_val)

            shape_tuple = tuple(
                sympy.Integer(int(x)) for x in shape_arr.astype(int).flatten()
            )
            node.shape = shape_tuple


@ShapeInference.register_handler(OpType.ARANGE)
def handle_arange(node: TensorNode, get_val):
    start = get_val(node.parents[0])
    stop = get_val(node.parents[1])
    step = get_val(node.parents[2])

    if start is not None and stop is not None and step is not None:
        s = float(start) if not isinstance(start, np.ndarray) else float(start.item())
        e = float(stop) if not isinstance(stop, np.ndarray) else float(stop.item())
        st = float(step) if not isinstance(step, np.ndarray) else float(step.item())
        if st != 0:
            length = math.ceil((e - s) / st)
            length = max(0, int(length))
            node.shape = (sympy.Integer(length),)
    else:
        # Symbolic length
        node.shape = (
            sympy.Symbol(f"arange_{node.name}_len", integer=True, positive=True),
        )


@ShapeInference.register_handler(OpType.REPEAT)
def handle_repeat(node: TensorNode, get_val):
    if node.parents and node.parents[0].shape:
        data_shape = list(node.parents[0].shape)
        repeats = int(node.attrs.get("repeats", 1))
        axis = int(node.attrs.get("axis", 0))

        ndim = len(data_shape)
        if axis < 0:
            axis += ndim

        if 0 <= axis < ndim:
            data_shape[axis] = data_shape[axis] * repeats

        node.shape = tuple(data_shape)


@ShapeInference.register_handler(OpType.PERMUTE)
def handle_permute(node: TensorNode, get_val):
    if node.parents and node.parents[0].shape is not None:
        dims = node.attrs.get("dims")
        if dims:
            input_shape = node.parents[0].shape
            if len(input_shape) == len(dims):
                node.shape = tuple(input_shape[d] for d in dims)


@ShapeInference.register_handler(OpType.CONCAT)
def handle_concat(node: TensorNode, get_val):
    if not node.parents:
        return

    axis = node.attrs.get("axis", 0)
    # Check if we can get axis dynamically
    if len(node.parents) >= 3:
        axis_val = get_val(node.parents[2])
        if axis_val is not None:
            axis = int(axis_val.item()) if hasattr(axis_val, "item") else int(axis_val)

    shapes = [p.shape for p in node.parents if p.shape is not None]
    if len(shapes) < 2:
        if len(shapes) == 1:
            node.shape = shapes[0]
        return

    # Normalize axis
    rank = len(shapes[0])
    if axis < 0:
        axis += rank

    if 0 <= axis < rank:
        new_shape = list(shapes[0])
        total_dim = shapes[0][axis]
        for s in shapes[1:]:
            # We assume ranks match for valid concat
            if len(s) == rank:
                total_dim = total_dim + s[axis]

        new_shape[axis] = total_dim
        node.shape = tuple(new_shape)


@ShapeInference.register_handler(OpType.GATHER)
def handle_gather(node: TensorNode, get_val):
    if len(node.parents) >= 2:
        data_shape = node.parents[0].shape
        indices_shape = node.parents[1].shape
        if data_shape and indices_shape:
            # Standard gather behavior on axis 0
            node.shape = indices_shape + data_shape[1:]


# --- Mathematical Ops ---


@ShapeInference.register_handler(OpType.DOT)
def handle_dot(node: TensorNode, get_val):
    if len(node.parents) == 2:
        s0 = node.parents[0].shape
        s1 = node.parents[1].shape
        if s0 and s1:
            # Handle Rank 3 @ Rank 2 specifically (Batch Matmul)
            if len(s0) == 3 and len(s1) == 2:
                # (B, M, K) @ (K, N) -> (B, M, N)
                node.shape = (s0[0], s0[1], s1[1])
            elif len(s0) >= 2 and len(s1) >= 2:
                batch0 = s0[:-2]
                batch1 = s1[:-2]
                batch_out = _SymShape.broadcast(batch0, batch1)
                m = s0[-2]
                n = s1[-1]
                node.shape = batch_out + (m, n)
            elif len(s0) == 2 and len(s1) == 2:
                node.shape = (s0[0], s1[1])


@ShapeInference.register_handler(OpType.SUM)
@ShapeInference.register_handler(OpType.MAX)
def handle_reduction(node: TensorNode, get_val):
    if not node.parents or node.parents[0].shape is None:
        return

    data_shape = list(node.parents[0].shape)
    ndim = len(data_shape)

    axis = node.attrs.get("axis")
    if axis is None and len(node.parents) > 1:
        # Try dynamic axis from input 1
        val = get_val(node.parents[1])
        if val is not None:
            axis = int(val.item()) if hasattr(val, "item") else int(val)

    keepdims = node.attrs.get("keepdims", True)

    # If op is MAX, standard impl usually keeps dims, but check attr
    # Defaults handled above.

    if axis is None:
        # Global reduction
        if keepdims:
            node.shape = tuple(sympy.Integer(1) for _ in range(ndim))
        else:
            node.shape = (sympy.Integer(1),)
    else:
        # Resolve axes
        if hasattr(axis, "__iter__") and not isinstance(axis, (str, bytes)):
            axes = list(axis)
        else:
            axes = [axis]

        # Handle negative axes
        axes = [int(a) + ndim if int(a) < 0 else int(a) for a in axes]

        new_shape = []
        for i, d in enumerate(data_shape):
            if i in axes:
                if keepdims:
                    new_shape.append(sympy.Integer(1))
            else:
                new_shape.append(d)
        node.shape = tuple(new_shape)


# --- Reshape & Slice ---


@ShapeInference.register_handler(OpType.RESHAPE)
def handle_reshape(node: TensorNode, get_val):
    data = node.parents[0]
    shape_val = get_val(node.parents[1])

    if shape_val is not None:
        # Concrete target shape found
        if hasattr(shape_val, "cpu"):
            shape_arr = shape_val.cpu().numpy()
        else:
            shape_arr = np.asarray(shape_val)

        target_dims_raw = shape_arr.astype(int).flatten().tolist()

        if -1 in target_dims_raw:
            if data.shape is None:
                # Can't resolve -1 without input shape
                node.shape = tuple(
                    sympy.Integer(d) if d != -1 else _SymShape.to_symbolic(None)
                    for d in target_dims_raw
                )
                return

            total_vol = _SymShape.prod(data.shape)
            known_vol = sympy.Integer(1)
            idx_neg = -1

            target_dims_sym = []
            for i, d in enumerate(target_dims_raw):
                if d == -1:
                    idx_neg = i
                    target_dims_sym.append(None)
                else:
                    known_vol *= d
                    target_dims_sym.append(sympy.Integer(d))

            missing = total_vol / known_vol
            target_dims_sym[idx_neg] = missing
            node.shape = tuple(target_dims_sym)
        else:
            node.shape = tuple(sympy.Integer(d) for d in target_dims_raw)


@ShapeInference.register_handler(OpType.SLICE)
def handle_slice(node: TensorNode, get_val):
    if not node.parents or node.parents[0].shape is None:
        return

    data_shape = node.parents[0].shape
    starts = node.attrs.get("starts", [0] * len(data_shape))
    ends = node.attrs.get(
        "ends", [None] * len(data_shape)
    )  # None acts as dim placeholder
    steps = node.attrs.get("steps", [1] * len(data_shape))

    new_shape = []
    for i, dim in enumerate(data_shape):
        s_val = starts[i] if i < len(starts) else 0
        e_val = ends[i] if i < len(ends) else None
        st_val = steps[i] if i < len(steps) else 1

        # Handle explicit ends
        if e_val is None:
            e = dim
        else:
            e = _SymShape.to_symbolic(e_val)

        s = _SymShape.to_symbolic(s_val)
        st = _SymShape.to_symbolic(st_val)

        # Simple negative index handling for constants (symbolic is harder)
        if isinstance(s, sympy.Integer) and isinstance(dim, sympy.Integer) and s < 0:
            s += dim
        if isinstance(e, sympy.Integer) and isinstance(dim, sympy.Integer) and e < 0:
            e += dim

        # Calculation: ceil((end - start) / step)
        diff = e - s
        dim_len = sympy.ceiling(diff / st)
        new_shape.append(dim_len)

    node.shape = tuple(new_shape)


# --- Fused Ops Handlers ---


@ShapeInference.register_handler("RoPE")
@ShapeInference.register_handler("GELU")
@ShapeInference.register_handler("Softmax")
@ShapeInference.register_handler("RMSNorm")
def handle_fused_preserving(node: TensorNode, get_val):
    if node.parents and node.parents[0].shape:
        node.shape = node.parents[0].shape