"""
File: tensor_graphs/backend/kernels/atomic/max.py
"""

import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.max import max_ref


@KernelRegistry.register(
    OpType.MAX,
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.CPU_NUMPY,
    reference_factory=max_ref,
)
@KernelRegistry.register(
    OpType.MAX,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
        TensorSignature(DType.INT32, shape=(1,), backend=Backend.CPU_NUMPY),
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=max_ref,
)
def max_generic(inputs, outputs, attrs):
    data = inputs[0]

    if attrs and "axis" in attrs:
        axis = attrs["axis"]
    elif len(inputs) > 1:
        axis = int(inputs[1][0])
    else:
        axis = None

    result = np.max(data, axis=axis, keepdims=True)
    outputs[0][:] = result
