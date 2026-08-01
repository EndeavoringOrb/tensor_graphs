import numpy as np

from ....backend.registry import KernelRegistry
from ....ir.dtypes import Backend, DType, TensorSignature
from ....ops.atomic.dot import dot_ref
from ....ops.atomic_types import OpType


# Generic Matrix Multiplication
@KernelRegistry.register(
    OpType.DOT,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
    ],
    reference_factory=dot_ref,
)
def dot_generic(inputs, outputs, attrs):
    np.matmul(inputs[0], inputs[1], out=outputs[0])
