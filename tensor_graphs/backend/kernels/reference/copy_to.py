import numpy as np
import torch
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend, KernelUnavailableError
from ....ops.atomic_types import OpType
from ....ops.atomic.copy_to import copy_to_ref


# --- 1. Same-Backend Copy (CPU_NUMPY -> CPU_NUMPY) ---
@KernelRegistry.register(
    OpType.COPY_TO,
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.CPU_NUMPY,
    reference_factory=copy_to_ref,
)
def copy_numpy_to_numpy(inputs, attrs=None):
    # Just ensure it's a new array, contiguous
    return np.ascontiguousarray(inputs[0], dtype=np.float32)


# --- 2. Cross-Backend Copy (GPU_TORCH -> CPU_NUMPY) ---
# This kernel lives on CPU_NUMPY (the destination), but accepts GPU_TORCH input.
@KernelRegistry.register(
    OpType.COPY_TO,
    [TensorSignature(DType.FP32, shape=None, backend=Backend.GPU_TORCH)],
    backend=Backend.CPU_NUMPY,
    reference_factory=copy_to_ref,  # Re-using ref factory is fine
)
def copy_torch_gpu_to_numpy(inputs, attrs=None):
    tensor = inputs[0]
    # Handle torch tensor -> numpy conversion
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)
    # Fallback if simulation passed a numpy array masquerading as GPU
    return np.array(tensor, dtype=np.float32)


# --- 3. Cross-Backend Copy (CPU_NUMPY -> GPU_TORCH) ---
@KernelRegistry.register(
    OpType.COPY_TO,
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.GPU_TORCH,
    reference_factory=copy_to_ref,
)
def copy_numpy_to_torch_gpu(inputs, attrs=None):
    data = inputs[0]

    if not torch.cuda.is_available():
        raise KernelUnavailableError("Cannot copy to GPU_TORCH: CUDA is not available.")

    # Ensure input is numpy
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)

    # Create torch tensor on GPU
    # torch.from_numpy creates a tensor sharing memory on CPU, then we move it.
    return torch.from_numpy(data).to(device="cuda", dtype=torch.float32)


# --- 4. Int32 Support ---
@KernelRegistry.register(
    OpType.COPY_TO,
    [TensorSignature(DType.INT32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.CPU_NUMPY,
)
def copy_numpy_to_numpy_int(inputs, attrs=None):
    return np.ascontiguousarray(inputs[0], dtype=np.int32)
