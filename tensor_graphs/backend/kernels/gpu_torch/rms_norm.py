# tensor_graphs/backend/kernels/gpu_torch/rms_norm.py

import os
import torch
from typing import Any, cast
from torch.utils.cpp_extension import load
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend, KernelUnavailableError
from ....ops.fused.rms_norm import rms_norm_decomposition, rms_norm

# 1. JIT Compile the Kernel
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_CUDA_SRC_DIR = os.path.join(os.path.dirname(_CUR_DIR), "cuda_src")
_SRC_FILE = os.path.join(_CUDA_SRC_DIR, "rms_norm.cu")

_RMS_NORM_OPS = None
if torch.cuda.is_available():
    try:
        _RMS_NORM_OPS = load(
            name="rms_norm_cuda_kernel", sources=[_SRC_FILE], verbose=False
        )
        print("✅ RMSNorm CUDA kernel compiled successfully.")
    except Exception as e:
        print(f"❌ Failed to compile RMSNorm CUDA kernel: {e}")
        # Set to None to indicate failure
        _RMS_NORM_OPS = None


# 2. Conditionally Register
# Only register if the JIT compilation succeeded
if _RMS_NORM_OPS is not None:

    @KernelRegistry.register(
        "RMSNorm",
        [
            TensorSignature(DType.FP32, shape=None, backend=Backend.GPU_TORCH),
            TensorSignature(DType.FP32, shape=(None,), backend=Backend.GPU_TORCH),
            TensorSignature(DType.FP32, shape=(1,), backend=Backend.GPU_TORCH),
        ],
        backend=Backend.GPU_TORCH,
        reference_factory=rms_norm_decomposition,
    )
    def rms_norm_cuda(inputs, outputs, attrs):
        # Static analysis (pyright) doesn't know _RMS_NORM_OPS is not None here,
        # nor does it know the attributes of the JIT-compiled module.
        # We cast to Any to suppress these errors.
        ops = cast(Any, _RMS_NORM_OPS)

        # inputs: x, weight, eps
        x, weight, eps_tensor = inputs

        # Ensure inputs are contiguous and on CUDA
        if not x.is_cuda or not weight.is_cuda:
            raise ValueError("Inputs to RMSNorm CUDA kernel must be on GPU.")

        x = x.contiguous()
        weight = weight.contiguous()

        # Extract epsilon value
        if isinstance(eps_tensor, torch.Tensor):
            epsilon = float(eps_tensor.item())
        else:
            epsilon = float(eps_tensor)

        # Use pre-allocated output
        output = outputs[0]

        # Launch Kernel
        ops.rms_norm(x, weight, output, epsilon)

else:
    # If compilation failed, register a dummy kernel that raises an error to force CPU fallback
    @KernelRegistry.register(
        "RMSNorm",
        [
            TensorSignature(DType.FP32, shape=None, backend=Backend.GPU_TORCH),
            TensorSignature(DType.FP32, shape=(None,), backend=Backend.GPU_TORCH),
            TensorSignature(DType.FP32, shape=(1,), backend=Backend.GPU_TORCH),
        ],
        backend=Backend.GPU_TORCH,
        reference_factory=rms_norm_decomposition,
    )
    def rms_norm_cuda_fallback(inputs, outputs, attrs):
        raise KernelUnavailableError(
            "RMSNorm CUDA kernel is not available. Falling back to CPU execution."
        )
