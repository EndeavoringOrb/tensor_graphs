import math
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Any


class KernelUnavailableError(RuntimeError):
    """Raised when a kernel is not available for the requested backend."""


class DType(Enum):
    FP32 = "float32"
    INT32 = "int32"
    BOOL = "bool"

    @property
    def itemsize(self) -> int:
        """Returns the number of bytes per element."""
        return {
            DType.FP32: 4,
            DType.INT32: 4,
            DType.BOOL: 1,
        }[self]


def get_size_bytes(shape: Optional[Tuple[Optional[int], ...]], dtype: DType) -> int:
    """
    Centralized logic for calculating total byte size.
    Raises ValueError for dynamic shapes (containing None).
    """
    if shape is None or any(d is None for d in shape):
        raise ValueError(f"Cannot calculate byte size for dynamic shape: {shape}")

    # Handle scalar shapes ()
    if len(shape) == 0:
        return dtype.itemsize

    return math.prod(shape) * dtype.itemsize


def get_buffer_size(
    dtype: DType, shape: Optional[Tuple[int, ...]] = None, data: Any = None
) -> int:
    """
    Unified calculation for memory allocation.
    Priority:
    1. If data is provided, use its count of elements * internal itemsize.
    2. If shape is provided, use prod(shape) * internal itemsize.
    """
    if data is not None:
        # Handle numpy, torch, or raw scalars
        if hasattr(data, "shape") and len(data.shape) > 0:
            count = math.prod(data.shape)
        elif hasattr(data, "numel"):  # Torch fallback
            count = data.numel()
        else:
            count = 1  # Scalar
        return count * dtype.itemsize

    if shape is not None:
        if any(d is None for d in shape):
            raise ValueError(f"Cannot size a dynamic shape: {shape}")
        return math.prod(shape) * dtype.itemsize

    raise ValueError("Must provide either shape or data to calculate size.")


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
