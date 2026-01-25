import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic_types import OpType


# --- 1. Generic Vector ---
@KernelRegistry.register(
    OpType.DIVIDE,
    [TensorSignature(DType.FP32, (None,)), TensorSignature(DType.FP32, (None,))],
)
def div_generic_vector(inputs, attrs=None):
    return inputs[0] / inputs[1]


# --- 2. Generic Matrix ---
@KernelRegistry.register(
    OpType.DIVIDE,
    [
        TensorSignature(DType.FP32, (None, None)),
        TensorSignature(DType.FP32, (None, None)),
    ],
)
def div_generic_matrix(inputs, attrs=None):
    return inputs[0] / inputs[1]


# --- 3. Scalar Broadcast (Scalar / Matrix) ---
@KernelRegistry.register(
    OpType.DIVIDE,
    [TensorSignature(DType.FP32, (1,)), TensorSignature(DType.FP32, (None, None))],
)
def div_scalar_broadcast(inputs, attrs=None):
    # Scalar / Matrix
    return inputs[0] / inputs[1]


# --- 4. Matrix / Scalar ---
@KernelRegistry.register(
    OpType.DIVIDE,
    [TensorSignature(DType.FP32, (None, None)), TensorSignature(DType.FP32, (1,))],
)
def div_matrix_scalar(inputs, attrs=None):
    # Matrix / Scalar
    return inputs[0] / inputs[1]


# --- 5. Generic Tensor (Any Rank) ---
@KernelRegistry.register(
    OpType.DIVIDE,
    [TensorSignature(DType.FP32, shape=None), TensorSignature(DType.FP32, shape=None)],
)
def div_generic_tensor(inputs, attrs=None):
    return inputs[0] / inputs[1]
