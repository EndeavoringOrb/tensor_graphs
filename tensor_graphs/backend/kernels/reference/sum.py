import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.sum import sum_ref


@KernelRegistry.register(
    OpType.SUM,
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=sum_ref,
)
@KernelRegistry.register(
    OpType.SUM,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
        TensorSignature(DType.INT32, shape=(1,), backend=Backend.CPU_NUMPY),
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=sum_ref,
)
def sum_generic(inputs, outputs, attrs):
    data = inputs[0]

    if attrs and "axis" in attrs:
        axis = attrs["axis"]
        # NumPy requires a tuple for multiple axes, lists can cause TypeErrors in some versions
        if isinstance(axis, list):
            axis = tuple(axis)
    elif len(inputs) > 1:
        axis = int(inputs[1][0])
    else:
        axis = None

    keepdims = True
    if attrs and "keepdims" in attrs:
        keepdims = attrs["keepdims"]

    result = np.sum(data, axis=axis, keepdims=keepdims)
    outputs[0][:] = result
