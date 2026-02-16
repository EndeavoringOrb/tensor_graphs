import os
import torch
from typing import Any, cast
from torch.utils.cpp_extension import load
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend, KernelUnavailableError
from ....ops.atomic.dot import dot_ref
from ....ops.atomic_types import OpType

# JIT Compile
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_CUDA_SRC_DIR = os.path.join(os.path.dirname(_CUR_DIR), "cuda_src")
_SRC_FILE = os.path.join(_CUDA_SRC_DIR, "dot.cu")

_DOT_OPS = None
if torch.cuda.is_available():
    try:
        # Note: If dot.cu was already compiled, load will reuse the cached version
        _DOT_OPS = load(name="dot_cuda_kernel", sources=[_SRC_FILE], verbose=False)
    except Exception as e:
        print(f"❌ Failed to compile Dot CUDA kernel: {e}")

if _DOT_OPS is not None:
    # Register 4D @ 4D (Batch MatMul)
    @KernelRegistry.register(
        OpType.DOT,
        [
            TensorSignature(DType.FP32, shape=(None, None, None, None), backend=Backend.GPU_TORCH),
            TensorSignature(DType.FP32, shape=(None, None, None, None), backend=Backend.GPU_TORCH),
        ],
        backend=Backend.GPU_TORCH,
        reference_factory=dot_ref,
    )
    # Register 3D @ 2D (Linear Projection)
    @KernelRegistry.register(
        OpType.DOT,
        [
            TensorSignature(DType.FP32, shape=(None, None, None), backend=Backend.GPU_TORCH),
            TensorSignature(DType.FP32, shape=(None, None), backend=Backend.GPU_TORCH),
        ],
        backend=Backend.GPU_TORCH,
        reference_factory=dot_ref,
    )
    def dot_cuda(inputs, outputs, attrs):
        ops = cast(Any, _DOT_OPS)
        a, b = inputs
        out = outputs[0]
        
        # Performance: avoid repeated checks if possible, but required for safety
        if not a.is_cuda: a = a.cuda()
        if not b.is_cuda: b = b.cuda()

        # Kernels assume C-contiguous pointers
        ops.dot(a.contiguous(), b.contiguous(), out)
else:
    @KernelRegistry.register(
        OpType.DOT,
        [
            TensorSignature(DType.FP32, shape=None, backend=Backend.GPU_TORCH),
            TensorSignature(DType.FP32, shape=None, backend=Backend.GPU_TORCH),
        ],
        backend=Backend.GPU_TORCH,
        reference_factory=dot_ref,
    )
    def dot_cuda_fallback(inputs, outputs, attrs):
        raise KernelUnavailableError("Dot CUDA kernel unavailable. Check JIT logs.")