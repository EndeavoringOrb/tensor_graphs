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
def fill_fp32(inputs, outputs, attrs):
    value = inputs[0]
    shape_tensor = inputs[1]

    val = value.item()
    target_shape = tuple(shape_tensor.astype(int))

    result = np.full(target_shape, val, dtype=np.float32)
    outputs[0][:] = result


@KernelRegistry.register(
    OpType.FILL,
    [
        TensorSignature(DType.INT32, shape=(1,), backend=Backend.CPU_NUMPY),
        TensorSignature(DType.INT32, shape=(None,), backend=Backend.CPU_NUMPY),
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.INT32,
    reference_factory=fill_ref,
)
def fill_int32(inputs, outputs, attrs):
    value = inputs[0]
    shape_tensor = inputs[1]

    val = value.item()
    target_shape = tuple(shape_tensor.astype(int))

    result = np.full(target_shape, val, dtype=np.int32)
    outputs[0][:] = result
