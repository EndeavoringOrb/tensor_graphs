import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.repeat import repeat_ref


@KernelRegistry.register(
    OpType.REPEAT,
    [
        TensorSignature(
            DType.FP32, shape=None, backend=Backend.CPU_NUMPY
        ),  # Input Data
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=repeat_ref,
)
@KernelRegistry.register(
    OpType.REPEAT,
    [
        TensorSignature(
            DType.FP32, shape=None, backend=Backend.CPU_NUMPY
        ),  # Input Data
        TensorSignature(DType.INT32, shape=(1,), backend=Backend.CPU_NUMPY),  # Repeats
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=repeat_ref,
)
def repeat_generic(inputs, attrs=None, outputs=None):
    """
    Generic Repeat (Interleave).
    Repeats elements of an array.
    """
    data = inputs[0]
    if len(inputs) > 1:
        repeats = int(inputs[1][0])
    else:
        if attrs is None or "repeats" not in attrs:
            raise ValueError(
                "Repeat requires 'repeats' as either an input or an attribute"
            )
        repeats = int(attrs["repeats"])

    # 2. Determine Axis
    axis = attrs.get("axis", 0) if attrs else 0

    # Validation for demo safety
    if len(data.shape) < 2:
        # Fallback for flat arrays
        result = np.repeat(data, repeats)
    else:
        result = np.repeat(data, repeats, axis=axis)

    if outputs is not None:
        outputs[0][:] = result
        return outputs[0]
    return result
