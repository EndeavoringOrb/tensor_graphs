import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic_types import OpType
from ....ops.atomic.concat import concat_ref


@KernelRegistry.register(
    OpType.CONCAT,
    [
        TensorSignature(DType.FP32, shape=None),  # Input A
        TensorSignature(DType.FP32, shape=None),  # Input B
    ],
    target_dtype=DType.FP32,
    reference_factory=concat_ref,
)
def concat_generic(inputs, outputs, attrs):
    if attrs is None or "axis" not in attrs:
        raise ValueError("Concat kernel requires 'axis' attribute")

    axis = attrs["axis"]
    result = np.concatenate(inputs, axis=axis)
    outputs[0][:] = result