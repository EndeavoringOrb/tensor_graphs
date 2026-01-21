import numpy as np
from tensor_graphs.backend.registry import KernelRegistry
from tensor_graphs.ir.dtypes import DType, TensorSignature
from tensor_graphs.ops.atomic import OpType

# --- 1. Converters ---
@KernelRegistry.register_cast(src=DType.FP8E4M3, dst=DType.FP32)
def cast_fp8_to_fp32(inputs):
    val = inputs[0]
    return val.astype(np.float32)

# --- 2. ADD Kernels ---

# Scalar + Scalar
sig_scalar = TensorSignature(DType.FP32, (1,))
@KernelRegistry.register(OpType.ADD, [sig_scalar, sig_scalar])
def add_scalar(inputs):
    return inputs[0] + inputs[1]

# Vector(32) + Vector(32)
sig_vec32 = TensorSignature(DType.FP32, (32,))
@KernelRegistry.register(OpType.ADD, [sig_vec32, sig_vec32])
def add_vec32(inputs):
    return inputs[0] + inputs[1]

# Vector(4,4) + Vector(4,4)
sig_mat4 = TensorSignature(DType.FP32, (4, 4))
@KernelRegistry.register(OpType.ADD, [sig_mat4, sig_mat4])
def add_mat4(inputs):
    return inputs[0] + inputs[1]

# Scalar + Vector(4,4) (Broadcast)
@KernelRegistry.register(OpType.ADD, [sig_scalar, sig_mat4])
def add_scalar_broadcast(inputs):
    return inputs[0] + inputs[1]

# Vector(2) + Vector(2) (For tests)
sig_vec2 = TensorSignature(DType.FP32, (2,))
@KernelRegistry.register(OpType.ADD, [sig_vec2, sig_vec2])
def add_vec2(inputs):
    return inputs[0] + inputs[1]

# --- 3. MUL Kernels ---

@KernelRegistry.register(OpType.MUL, [sig_scalar, sig_scalar])
def mul_scalar(inputs):
    return inputs[0] * inputs[1]

@KernelRegistry.register(OpType.MUL, [sig_vec32, sig_vec32])
def mul_vec32(inputs):
    return inputs[0] * inputs[1]

@KernelRegistry.register(OpType.MUL, [sig_vec2, sig_vec2])
def mul_vec2(inputs):
    return inputs[0] * inputs[1]

# --- 4. DOT Kernels ---

# Matrix(2,2) @ Matrix(2,2)
sig_mat2 = TensorSignature(DType.FP32, (2, 2))
@KernelRegistry.register(OpType.DOT, [sig_mat2, sig_mat2])
def dot_mat2(inputs):
    return np.matmul(inputs[0], inputs[1])

# --- 5. FUSED Kernels ---

@KernelRegistry.register(OpType.FUSED_MUL_ADD, [sig_vec32, sig_vec32, sig_vec32])
def fma_vec32(inputs):
    # (A * B) + C
    return inputs[0] * inputs[1] + inputs[2]

@KernelRegistry.register(OpType.FUSED_MUL_ADD, [sig_vec2, sig_vec2, sig_vec2])
def fma_vec2(inputs):
    return inputs[0] * inputs[1] + inputs[2]