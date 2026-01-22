import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType


# --- 1. Generic Fallback ---
# Matches any FP32 vector + vector
@KernelRegistry.register(
    OpType.ADD,
    [TensorSignature(DType.FP32, (None,)), TensorSignature(DType.FP32, (None,))],
)
def add_generic_vector(inputs):
    # print("DEBUG: Using Generic ADD Kernel")
    return inputs[0] + inputs[1]


# --- 2. Generic Matrix ---
@KernelRegistry.register(
    OpType.ADD,
    [
        TensorSignature(DType.FP32, (None, None)),
        TensorSignature(DType.FP32, (None, None)),
    ],
)
def add_generic_matrix(inputs):
    return inputs[0] + inputs[1]


# --- 3. Optimized Vec32 ---
# Matches exactly (32,) + (32,). Should win score tie-breaks over generic.
@KernelRegistry.register(
    OpType.ADD, [TensorSignature(DType.FP32, (32,)), TensorSignature(DType.FP32, (32,))]
)
def add_vec32_optimized(inputs):
    # print("DEBUG: Using Optimized Vec32 ADD Kernel")
    return inputs[0] + inputs[1]


# --- 4. Scalar Broadcast ---
# Scalar + Generic Matrix
@KernelRegistry.register(
    OpType.ADD,
    [TensorSignature(DType.FP32, (1,)), TensorSignature(DType.FP32, (None, None))],
)
def add_scalar_broadcast(inputs):
    return inputs[0] + inputs[1]


# --- 5. Generic Tensor (Any Rank) ---
@KernelRegistry.register(
    OpType.ADD,
    [TensorSignature(DType.FP32, shape=None), TensorSignature(DType.FP32, shape=None)],
)
def add_generic_tensor(inputs):
    return inputs[0] + inputs[1]
