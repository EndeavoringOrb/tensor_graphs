from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional

class DType(Enum):
    FP32 = "float32"
    FP16 = "float16"
    FP8E4M3 = "fp8e4m3"
    INT32 = "int32"
    BOOL = "bool"

@dataclass(frozen=True)
class TensorSignature:
    """
    Represents the Type and Shape state of a tensor for kernel matching.
    'shape' entries can be None to indicate a wildcard (generic size).
    If 'shape' is entirely None, it matches any rank/shape.
    """
    dtype: DType
    shape: Optional[Tuple[Optional[int], ...]]
    
    def __repr__(self):
        if self.shape is None:
            return f"<{self.dtype.value} (*)>"
        shape_str = ",".join(str(d) if d is not None else "*" for d in self.shape)
        return f"<{self.dtype.value} ({shape_str})>"

    def is_scalar(self):
        if self.shape is None: return False
        return len(self.shape) == 0 or (len(self.shape) == 1 and self.shape[0] == 1)