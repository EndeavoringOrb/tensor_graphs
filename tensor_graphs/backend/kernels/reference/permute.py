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
)
def permute_generic(inputs, attrs=None, outputs=None):
    if attrs is None or "dims" not in attrs:
        raise ValueError("Permute kernel requires 'dims' attribute")
        
    data = inputs[0]
    dims = attrs["dims"] # e.g. (1, 0)
    
    result = np.transpose(data, axes=dims)
    if outputs is not None:
        outputs[0][:] = result
        return outputs[0]
    return result