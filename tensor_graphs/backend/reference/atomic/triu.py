import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType


@KernelRegistry.register(
    OpType.TRIU,
    [
        TensorSignature(DType.FP32, shape=None),  # Data
        TensorSignature(DType.INT32, shape=(1,)),  # k
    ],
)
def triu_generic(inputs, attrs=None):
    data = inputs[0]
    k = int(inputs[1][0])
    return np.triu(data, k=k)
