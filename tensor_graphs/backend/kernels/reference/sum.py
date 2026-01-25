"""
File: tensor_graphs/backend/kernels/atomic/sum.py
"""

import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic_types import OpType


@KernelRegistry.register(
    OpType.SUM,
    [TensorSignature(DType.FP32, shape=None)],
)
@KernelRegistry.register(
    OpType.SUM,
    [TensorSignature(DType.FP32, shape=None), TensorSignature(DType.INT32, shape=(1,))],
)
def sum_generic(inputs, attrs=None):
    data = inputs[0]

    # 1. Determine Axis
    if attrs and "axis" in attrs:
        axis = attrs["axis"]
    elif len(inputs) > 1:
        # Backward compatibility for when axis was an input node
        axis = int(inputs[1][0])
    else:
        axis = None  # Global sum

    # 2. Determine Keepdims
    keepdims = True
    if attrs and "keepdims" in attrs:
        keepdims = attrs["keepdims"]

    return np.sum(data, axis=axis, keepdims=keepdims)
