import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.mul import mul_ref


@KernelRegistry.register(
    OpType.MUL,
    [
        TensorSignature(DType.FP32, (None,), backend=Backend.CPU_NUMPY),
        TensorSignature(DType.FP32, (None,), backend=Backend.CPU_NUMPY),
    ],
    backend=Backend.CPU_NUMPY,
    reference_factory=mul_ref,
)
def mul_generic_vector(inputs, attrs=None, outputs=None):
    if outputs is None:
        return inputs[0] * inputs[1]
    np.multiply(inputs[0], inputs[1], out=outputs[0])
    return outputs[0]


@KernelRegistry.register(
    OpType.MUL,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
        TensorSignature(DType.FP32, shape=(1,), backend=Backend.CPU_NUMPY),
    ],
    backend=Backend.CPU_NUMPY,
    reference_factory=mul_ref,
)
def mul_tensor_generic_scalar(inputs, attrs=None, outputs=None):
    if outputs is None:
        return inputs[0] * inputs[1]
    np.multiply(inputs[0], inputs[1], out=outputs[0])
    return outputs[0]


# Generic Tensor (Any Rank)
@KernelRegistry.register(
    OpType.MUL,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
    ],
    backend=Backend.CPU_NUMPY,
    reference_factory=mul_ref,
)
def mul_generic_tensor(inputs, attrs=None, outputs=None):
    if outputs is None:
        return inputs[0] * inputs[1]
    np.multiply(inputs[0], inputs[1], out=outputs[0])
    return outputs[0]
