from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import uuid
from .dtypes import DType, TensorSignature, Backend


@dataclass(eq=False)
class TensorNode:
    op_type: str
    shape: Tuple[Optional[int], ...]
    dtype: DType  # Strict Typing
    parents: List["TensorNode"]
    name: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    attrs: Dict[str, Any] = field(default_factory=dict)
    backend: Backend = Backend.CPU_NUMPY

    def get_attr(self, key: str, default: Any = None) -> Any:
        return self.attrs.get(key, default)

    @property
    def signature(self) -> TensorSignature:
        return TensorSignature(self.dtype, self.shape)

    def __getitem__(self, key) -> "TensorNode":
        """
        Syntactic sugar for OpType.SLICE.
        Supports standard Python slicing logic.
        """
        if not isinstance(key, tuple):
            key = (key,)

        # 1. Resolve Ellipsis and handle missing dimensions
        if Ellipsis in key:
            idx = key.index(Ellipsis)
            num_missing = len(self.shape) - (len(key) - 1)
            key = key[:idx] + (slice(None),) * num_missing + key[idx + 1 :]

        if len(key) < len(self.shape):
            key = key + (slice(None),) * (len(self.shape) - len(key))

        key = key[: len(self.shape)]

        new_shape = []
        starts = []
        ends = []
        steps = []

        for i, (k, dim_size) in enumerate(zip(key, self.shape)):
            if isinstance(k, int):
                # Treat int as slice(k, k+1) to preserve rank for now
                start = k if k >= 0 or dim_size is None else k + dim_size

                if start == -1 and dim_size is None:
                    stop = None
                else:
                    stop = start + 1

                starts.append(start)
                ends.append(stop)
                steps.append(1)
                new_shape.append(1)
            elif isinstance(k, slice):
                start = k.start if k.start is not None else 0
                stop = k.stop
                step = k.step if k.step is not None else 1

                if dim_size is not None:
                    if start < 0:
                        start += dim_size
                    if stop is not None and stop < 0:
                        stop += dim_size

                    # Compute output dim size
                    e = stop if stop is not None else dim_size
                    out_dim = (e - start + step - (1 if step > 0 else -1)) // step
                    new_shape.append(max(0, out_dim))
                else:
                    new_shape.append(None)

                starts.append(start)
                ends.append(stop)
                steps.append(step)
            else:
                raise ValueError(f"Unsupported index type: {type(k)}")

        return TensorNode(
            "Slice",
            tuple(new_shape),
            self.dtype,
            [self],
            f"{self.name}_slice",
            attrs={"starts": starts, "ends": ends, "steps": steps},
            backend=self.backend,
        )

    def __repr__(self):
        attr_str = f" | {self.attrs}" if self.attrs else ""
        return f"[{self.dtype.value}|{self.shape}{attr_str}] {self.op_type}"


@dataclass(eq=False)
class ConstantNode(TensorNode):
    """A node representing a constant value stored within the graph."""

    value: Any = None

    def __repr__(self):
        return f"[CONST|{self.dtype.value}|{self.shape}] = {self.value}"
