import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic_types import OpType
from ....ops.atomic.arange import arange_ref


@KernelRegistry.register(
    OpType.ARANGE,
    [
        TensorSignature(DType.INT32, shape=(1,)),  # Start
        TensorSignature(DType.INT32, shape=(1,)),  # Stop
        TensorSignature(DType.INT32, shape=(1,)),  # Step
    ],
    reference_factory=arange_ref,
)
def arange_int32(inputs, attrs=None, outputs=None):
    start = int(inputs[0][0])
    stop = int(inputs[1][0])
    step = int(inputs[2][0])

    # np.arange does not support 'out'
    res = np.arange(start, stop, step, dtype=np.int32)

    if outputs is not None:
        # Internal copy to the framework-provided buffer
        # We use [:] to write into the view
        outputs[0][:] = res
        return outputs[0]

    return res
