import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType


@KernelRegistry.register(
    OpType.REPEAT,
    [
        TensorSignature(DType.FP32, shape=None),  # Input Data
        TensorSignature(DType.INT32, shape=(1,)),  # Repeats
    ],
)
def repeat_generic(inputs):
    """
    Generic Repeat (Interleave).
    Repeats elements of an array.
    Note: For GQA (KV replication), usually axis=1 is used.
    Since Atomic Ops are generic, we assume axis=1 for this specific signature/usage context
    or we would need an 'axis' input.
    For simplicity in this demo, we hardcode axis=1 to match Gemma usage.
    """
    data = inputs[0]
    repeats = int(inputs[1][0])

    # Validation for demo safety
    if len(data.shape) < 2:
        # Fallback for flat arrays
        return np.repeat(data, repeats)

    return np.repeat(data, repeats, axis=1)
