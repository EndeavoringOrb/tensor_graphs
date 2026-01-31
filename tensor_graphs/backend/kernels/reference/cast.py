"""
File: tensor_graphs/backend/kernels/reference/cast.py
"""

import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic_types import OpType
from ....ops.atomic.cast import cast_ref


def _dtype_to_numpy(dtype_enum):
    if dtype_enum == DType.FP32:
        return np.float32
    elif dtype_enum == DType.FP16:
        return np.float16
    elif dtype_enum == DType.INT32:
        return np.int32
    elif dtype_enum == DType.BOOL:
        return bool
    return np.float32


def cast_implementation(inputs, outputs, attrs):
    if attrs is None:
        attrs = {}
    target_dtype = attrs.get("to", DType.FP32)
    result = inputs[0].astype(_dtype_to_numpy(target_dtype))
    outputs[0][:] = result


@KernelRegistry.register(
    OpType.CAST,
    [TensorSignature(DType.INT32, shape=None)],
    target_dtype=DType.FP32,
    reference_factory=cast_ref,
)
@KernelRegistry.register(
    OpType.CAST,
    [TensorSignature(DType.FP32, shape=None)],
    target_dtype=DType.INT32,
    reference_factory=cast_ref,
)
@KernelRegistry.register(
    OpType.CAST,
    [TensorSignature(DType.BOOL, shape=None)],
    target_dtype=DType.FP32,
    reference_factory=cast_ref,
)
@KernelRegistry.register(
    OpType.CAST,
    [TensorSignature(DType.FP32, shape=None)],
    target_dtype=DType.FP16,
    reference_factory=cast_ref,
)
def cast_wrappers(inputs, outputs, attrs):
    cast_implementation(inputs, outputs, attrs)
