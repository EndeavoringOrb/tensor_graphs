import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.fill import fill_ref


@KernelRegistry.register(
    OpType.FILL,
    [
        TensorSignature(
            DType.FP32, shape=(1,), backend=Backend.CPU_NUMPY
        ),  # Value (Scalar)
        TensorSignature(DType.INT32, shape=(None,), backend=Backend.CPU_NUMPY),  # Shape
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=fill_ref,
)
def fill_fp32(inputs, attrs=None):
    value = inputs[0]
    shape_tensor = inputs[1]

    # Extract scalar value
    if value.size != 1:
        # Allow for (1, 1, ...) if it essentially scalar?
        # For now, strict (1,) or scalar check.
        # But numpy often handles (1,) as scalar in item().
        pass

    val = value.item()

    # Extract shape tuple
    target_shape = tuple(shape_tensor.astype(int))

    return np.full(target_shape, val, dtype=np.float32)


@KernelRegistry.register(
    OpType.FILL,
    [
        TensorSignature(
            DType.INT32, shape=(1,), backend=Backend.CPU_NUMPY
        ),  # Value (Scalar)
        TensorSignature(DType.INT32, shape=(None,), backend=Backend.CPU_NUMPY),  # Shape
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.INT32,
    reference_factory=fill_ref,
)
def fill_int32(inputs, attrs=None):
    value = inputs[0]
    shape_tensor = inputs[1]

    val = value.item()
    target_shape = tuple(shape_tensor.astype(int))

    return np.full(target_shape, val, dtype=np.int32)
