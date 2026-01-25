import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic_types import OpType


@KernelRegistry.register(
    OpType.REPEAT,
    [TensorSignature(DType.FP32, shape=None)],
)
@KernelRegistry.register(
    OpType.REPEAT,
    [
        TensorSignature(DType.FP32, shape=None),  # Input Data
        TensorSignature(DType.INT32, shape=(1,)),  # Repeats
    ],
)
def repeat_generic(inputs, attrs=None):
    """
    Generic Repeat (Interleave).
    Repeats elements of an array.
    """
    data = inputs[0]

    # 1. Determine Repeats
    if attrs and "repeats" in attrs:
        repeats = attrs["repeats"]
    elif len(inputs) > 1:
        repeats = int(inputs[1][0])
    else:
        raise ValueError("Repeat op requires 'repeats' either as an input or attribute")

    # 2. Determine Axis
    axis = attrs.get("axis", 0) if attrs else 0

    # Validation for demo safety
    if len(data.shape) < 2:
        # Fallback for flat arrays
        return np.repeat(data, repeats)

    return np.repeat(data, repeats, axis=axis)
