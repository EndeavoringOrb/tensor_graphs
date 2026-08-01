import numpy as np

from ....backend.registry import KernelRegistry
from ....ir.dtypes import Backend, DType, TensorSignature
from ....ops.atomic.permute import permute_ref
from ....ops.atomic_types import OpType


@KernelRegistry.register(
    OpType.PERMUTE,
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=permute_ref,
)
def permute_generic(inputs, outputs, attrs):
    if attrs is None or "dims" not in attrs:
        raise ValueError("Permute kernel requires 'dims' attribute")

    data = inputs[0]
    dims = attrs["dims"]

    result = np.transpose(data, axes=dims)
    outputs[0][:] = result
