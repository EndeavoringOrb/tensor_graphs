import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType


@KernelRegistry.register(
    OpType.WHERE,
    [
        TensorSignature(DType.BOOL, shape=None),  # Condition
        TensorSignature(DType.FP32, shape=None),  # X
        TensorSignature(DType.FP32, shape=None),  # Y
    ],
)
def where_bool_fp32(inputs):
    return np.where(inputs[0], inputs[1], inputs[2])


@KernelRegistry.register(
    OpType.WHERE,
    [
        TensorSignature(DType.INT32, shape=None),  # Condition
        TensorSignature(DType.FP32, shape=None),  # X
        TensorSignature(DType.FP32, shape=None),  # Y
    ],
)
def where_int32_fp32(inputs):
    return np.where(inputs[0], inputs[1], inputs[2])

@KernelRegistry.register(
    OpType.WHERE,
    [
        TensorSignature(DType.FP32, shape=None),  # Condition
        TensorSignature(DType.FP32, shape=None),  # X
        TensorSignature(DType.FP32, shape=None),  # Y
    ],
)
def where_fp32_fp32(inputs):
    return np.where(inputs[0], inputs[1], inputs[2])
