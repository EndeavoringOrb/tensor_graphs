from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import uuid
from .dtypes import DType, TensorSignature


@dataclass
class TensorNode:
    op_type: str
    shape: Tuple[Optional[int], ...]
    dtype: DType  # Strict Typing
    parents: List["TensorNode"]
    name: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def signature(self) -> TensorSignature:
        return TensorSignature(self.dtype, self.shape)

    def __repr__(self):
        return f"[{self.dtype.value}|{self.shape}] {self.op_type}"

    def __hash__(self):
        return id(self)
