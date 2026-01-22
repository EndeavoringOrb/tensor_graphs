import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType


@KernelRegistry.register(
    OpType.SLICE,
    [
        TensorSignature(DType.FP32, shape=None),  # Data
        TensorSignature(DType.INT32, shape=(None,)),  # Starts
        TensorSignature(DType.INT32, shape=(None,)),  # Ends
        TensorSignature(DType.INT32, shape=(None,)),  # Steps
    ],
)
def slice_generic(inputs, attrs=None):
    """
    Generic Slice Implementation.
    inputs[0]: Data tensor (Any Rank)
    inputs[1]: Starts (1D INT32)
    inputs[2]: Ends (1D INT32)
    inputs[3]: Steps (1D INT32)
    """
    data = inputs[0]
    starts = inputs[1].astype(int)
    ends = inputs[2].astype(int)
    steps = inputs[3].astype(int)

    # Validation
    if len(starts) != len(ends) or len(starts) != len(steps):
        raise ValueError("Starts, Ends, and Steps must be same length")

    slices = []
    for i in range(len(starts)):
        slices.append(slice(starts[i], ends[i], steps[i]))

    return data[tuple(slices)]
