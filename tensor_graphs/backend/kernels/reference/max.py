"""
File: tensor_graphs/backend/kernels/atomic/max.py
"""

import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic_types import OpType


@KernelRegistry.register(
    OpType.MAX,
    [TensorSignature(DType.FP32, shape=None)],
)
@KernelRegistry.register(
    OpType.MAX,
    [TensorSignature(DType.FP32, shape=None), TensorSignature(DType.INT32, shape=(1,))],
)
def max_generic(inputs, attrs=None):
    data = inputs[0]

    # 1. Determine Axis
    if attrs and "axis" in attrs:
        axis = attrs["axis"]
    elif len(inputs) > 1:
        axis = int(inputs[1][0])
    else:
        axis = None

    return np.max(data, axis=axis, keepdims=True)
