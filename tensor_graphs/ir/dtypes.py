# File: tensor_graphs/ir/dtypes.py
from enum import Enum
from dataclasses import dataclass
from typing import Tuple

class DType(Enum):
    FP32 = "float32"
    FP16 = "float16"
    FP8E4M3 = "fp8e4m3"  # The specific type you requested
    INT32 = "int32"

@dataclass(frozen=True)
class TensorSignature:
    """Represents the Type and Shape state of a tensor for kernel matching."""
    dtype: DType
    shape: Tuple[int, ...]
    
    def __repr__(self):
        return f"<{self.dtype.value} {self.shape}>"

    def is_scalar(self):
        return len(self.shape) == 0 or (len(self.shape) == 1 and self.shape[0] == 1)