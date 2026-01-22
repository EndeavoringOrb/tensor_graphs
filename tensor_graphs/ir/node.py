from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import uuid
from .dtypes import DType, TensorSignature


@dataclass(eq=False)
class TensorNode:
    op_type: str
    shape: Tuple[Optional[int], ...]
    dtype: DType  # Strict Typing
    parents: List["TensorNode"]
    name: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    attrs: Dict[str, Any] = field(default_factory=dict)

    def get_attr(self, key: str, default: Any = None) -> Any:
        return self.attrs.get(key, default)

    @property
    def signature(self) -> TensorSignature:
        return TensorSignature(self.dtype, self.shape)

    def __repr__(self):
        attr_str = f" | {self.attrs}" if self.attrs else ""
        return f"[{self.dtype.value}|{self.shape}{attr_str}] {self.op_type}"


@dataclass(eq=False)
class ConstantNode(TensorNode):
    """A node representing a constant value stored within the graph."""

    value: Any = None

    def __repr__(self):
        return f"[CONST|{self.dtype.value}|{self.shape}] = {self.value}"
