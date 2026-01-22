import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.fused.norm import RMSNorm
from ....ops.fused.activation import GELU, Softmax
from ....ops.fused.llm import RoPE, Embedding


# --- RMSNorm ---
@KernelRegistry.register(
    RMSNorm.op_type,
    [
        TensorSignature(DType.FP32, shape=None),
        TensorSignature(DType.FP32, (None,)),
        TensorSignature(DType.FP32, (1,)),
    ],
)
def rms_norm_kernel(inputs):
    x, scale, eps = inputs
    eps_val = eps[0]
    var = np.mean(x**2, axis=-1, keepdims=True)
    x_norm = x * (1.0 / np.sqrt(var + eps_val))
    # Gemma 3 uses (1 + scale)
    return x_norm * (1.0 + scale)


# --- GELU ---
@KernelRegistry.register(GELU.op_type, [TensorSignature(DType.FP32, shape=None)])
def gelu_kernel(inputs):
    x = inputs[0]
    c1 = np.sqrt(2.0 / np.pi)
    c2 = 0.044715
    inner = c1 * (x + c2 * np.power(x, 3))
    return 0.5 * x * (1.0 + np.tanh(inner))


# --- Softmax ---
@KernelRegistry.register(Softmax.op_type, [TensorSignature(DType.FP32, shape=None)])
def softmax_kernel(inputs):
    x = inputs[0]
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# --- RoPE ---
@KernelRegistry.register(
    RoPE.op_type,
    [
        TensorSignature(DType.FP32, shape=None),
        TensorSignature(DType.FP32, shape=None),
        TensorSignature(DType.FP32, shape=None),
    ],
)
def rope_kernel(inputs):
    x, cos, sin = inputs
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotated = np.concatenate((-x2, x1), axis=-1)
    return (x * cos) + (rotated * sin)


# --- Embedding ---
@KernelRegistry.register(
    Embedding.op_type,
    [
        TensorSignature(DType.INT32, (None, None)),
        TensorSignature(DType.FP32, (None, None)),
    ],
)
def embedding_kernel(inputs):
    indices, weights = inputs
    return weights[indices.astype(int)]
