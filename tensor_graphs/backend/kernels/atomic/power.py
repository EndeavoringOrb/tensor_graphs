import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType


@KernelRegistry.register(
    OpType.POWER,
    [TensorSignature(DType.FP32, shape=None), TensorSignature(DType.FP32, shape=None)],
)
def power_generic(inputs, attrs=None):
    return np.power(inputs[0], inputs[1])
