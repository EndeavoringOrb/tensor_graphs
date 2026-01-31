import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.power import power_ref


@KernelRegistry.register(
    OpType.POWER,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
    ],
    backend=Backend.CPU_NUMPY,
    reference_factory=power_ref,
)
def power_generic(inputs, outputs, attrs):
    result = np.power(inputs[0], inputs[1])
    outputs[0][:] = result
