# tensor_graphs/backend/kernels/cpu_numpy/add.py
import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.add import add_ref


@KernelRegistry.register(
    OpType.ADD,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=add_ref,
    inplace=True,
)
def add_inplace(inputs, outputs, attrs):
    np.add(inputs[0], inputs[1], out=outputs[0])
