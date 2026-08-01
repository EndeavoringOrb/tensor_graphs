# tensor_graphs/backend/kernels/cpu_numpy/mul.py
import numpy as np

from ....backend.registry import KernelRegistry
from ....ir.dtypes import Backend, DType, TensorSignature
from ....ops.atomic.mul import mul_ref
from ....ops.atomic_types import OpType


@KernelRegistry.register(
    OpType.MUL,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=mul_ref,
    inplace=True,
)
def mul_inplace(inputs, outputs, attrs):
    np.multiply(inputs[0], inputs[1], out=outputs[0])
