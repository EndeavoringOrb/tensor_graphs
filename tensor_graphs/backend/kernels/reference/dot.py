import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, Backend, TensorSignature
from ....ops.atomic_types import OpType
from ....ops.atomic.dot import dot_ref


# Generic Matrix Multiplication
@KernelRegistry.register(
    OpType.DOT,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
    ],
    reference_factory=dot_ref,
)
def dot_generic(inputs, attrs=None, outputs=None):
    if outputs is None:
        return np.matmul(inputs[0], inputs[1])
    np.matmul(inputs[0], inputs[1], out=outputs[0])
    return outputs[0]
