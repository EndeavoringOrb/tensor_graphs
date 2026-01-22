from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional

class DType(Enum):
    FP32 = "float32"
    FP16 = "float16"
    FP8E4M3 = "fp8e4m3"
    INT32 = "int32"

@dataclass(frozen=True)
class TensorSignature:
    """
    Represents the Type and Shape state of a tensor for kernel matching.
    'shape' entries can be None to indicate a wildcard (generic size).
    """
    dtype: DType
    shape: Tuple[Optional[int], ...]
    
    def __repr__(self):
        shape_str = ",".join(str(d) if d is not None else "*" for d in self.shape)
        return f"<{self.dtype.value} ({shape_str})>"

    def is_scalar(self):
        return len(self.shape) == 0 or (len(self.shape) == 1 and self.shape[0] == 1)