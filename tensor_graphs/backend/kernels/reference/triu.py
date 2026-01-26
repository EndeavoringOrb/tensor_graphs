import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.triu import triu_ref


@KernelRegistry.register(
    OpType.TRIU,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),  # Data
        TensorSignature(DType.INT32, shape=(1,), backend=Backend.CPU_NUMPY),  # k
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=triu_ref,
)
def triu_generic(inputs, attrs=None):
    data = inputs[0]
    k = int(inputs[1].item() if inputs[1].ndim == 0 else inputs[1][0])
    return np.triu(data, k=k)
