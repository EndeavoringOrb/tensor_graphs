import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType
from ....ir.dtypes import Backend

# --- Generic CopyTo for CPU_NUMPY ---
# This kernel is executed when the destination node is on CPU_NUMPY.
# It handles copying from any source (conceptually) to CPU_NUMPY.
# Since we are just simulating for now, we assume inputs are numpy-compatible or can be coerced.


@KernelRegistry.register(
    OpType.COPY_TO, [TensorSignature(DType.FP32, shape=None)], backend=Backend.CPU_NUMPY
)
def copy_to_cpu_numpy_fp32(inputs, attrs=None):
    # inputs[0] is the source tensor data.
    # In a real scenario, we would check inputs[0].device/backend and perform transfer.
    # For now, we assume it's already a numpy array or compatible.
    return np.array(inputs[0], dtype=np.float32)


@KernelRegistry.register(
    OpType.COPY_TO,
    [TensorSignature(DType.INT32, shape=None)],
    backend=Backend.CPU_NUMPY,
)
def copy_to_cpu_numpy_int32(inputs, attrs=None):
    return np.array(inputs[0], dtype=np.int32)
