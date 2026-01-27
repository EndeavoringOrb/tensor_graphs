from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional


class KernelUnavailableError(RuntimeError):
    """Raised when a kernel is not available for the requested backend."""


class DType(Enum):
    FP32 = "float32"
    FP16 = "float16"
    FP8E4M3 = "fp8e4m3"
    INT32 = "int32"
    BOOL = "bool"


class Backend(Enum):
    CPU_NUMPY = "cpu_numpy"
    CPU_TORCH = "cpu_torch"
    GPU_TORCH = "gpu_torch"


@dataclass(frozen=True)
class TensorSignature:
    """
    Represents the Type, Shape, and Backend state of a tensor for kernel matching.

    - shape=None: Wildcard (matches any shape)
    - backend=None: Wildcard (matches any backend)
    """

    dtype: DType
    shape: Optional[Tuple[Optional[int], ...]] = None
    backend: Optional[Backend] = None

    def __repr__(self):
        shape_str = "*"
        if self.shape is not None:
            shape_str = ",".join(str(d) if d is not None else "*" for d in self.shape)

        backend_str = self.backend.value if self.backend else "*"
        return f"<{self.dtype.value} [{shape_str}] @ {backend_str}>"

    def is_scalar(self):
        if self.shape is None:
            return False
        return len(self.shape) == 0 or (len(self.shape) == 1 and self.shape[0] == 1)
