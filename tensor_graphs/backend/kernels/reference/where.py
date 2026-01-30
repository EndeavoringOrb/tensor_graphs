import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.where import where_ref


@KernelRegistry.register(
    OpType.WHERE,
    [
        TensorSignature(DType.BOOL, shape=None, backend=Backend.CPU_NUMPY),  # Condition
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),  # X
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),  # Y
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=where_ref,
)
def where_bool_fp32(inputs, attrs=None, outputs=None):
    if outputs is None:
        return np.where(inputs[0], inputs[1], inputs[2])
    np.where(inputs[0], inputs[1], inputs[2], out=outputs[0])
    return outputs[0]


@KernelRegistry.register(
    OpType.WHERE,
    [
        TensorSignature(
            DType.INT32, shape=None, backend=Backend.CPU_NUMPY
        ),  # Condition
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),  # X
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),  # Y
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=where_ref,
)
def where_int32_fp32(inputs, attrs=None, outputs=None):
    if outputs is None:
        return np.where(inputs[0], inputs[1], inputs[2])
    np.where(inputs[0], inputs[1], inputs[2], out=outputs[0])
    return outputs[0]


@KernelRegistry.register(
    OpType.WHERE,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),  # Condition
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),  # X
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),  # Y
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=where_ref,
)
def where_fp32_fp32(inputs, attrs=None, outputs=None):
    if outputs is None:
        return np.where(inputs[0], inputs[1], inputs[2])
    np.where(inputs[0], inputs[1], inputs[2], out=outputs[0])
    return outputs[0]
