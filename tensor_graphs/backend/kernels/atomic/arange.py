import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType


@KernelRegistry.register(
    OpType.ARANGE,
    [
        TensorSignature(DType.INT32, shape=(1,)),  # Start
        TensorSignature(DType.INT32, shape=(1,)),  # Stop
        TensorSignature(DType.INT32, shape=(1,)),  # Step
    ],
)
def arange_int32(inputs, attrs=None):
    start = int(inputs[0][0])
    stop = int(inputs[1][0])
    step = int(inputs[2][0])
    return np.arange(start, stop, step, dtype=np.int32)
