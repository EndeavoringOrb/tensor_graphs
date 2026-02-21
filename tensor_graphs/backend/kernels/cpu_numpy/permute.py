# tensor_graphs/backend/kernels/cpu_numpy/permute.py
import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.permute import permute_ref


@KernelRegistry.register(
    OpType.PERMUTE,
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=permute_ref,
    inplace=True,
)
def permute_inplace(inputs, outputs, attrs):
    if attrs is None or "dims" not in attrs:
        raise ValueError("Permute kernel requires 'dims' attribute")

    data: np.ndarray = inputs[0]
    dims = attrs["dims"]

    # In-place for view ops simply changes the view mapping
    outputs[0] = data.transpose(dims)
