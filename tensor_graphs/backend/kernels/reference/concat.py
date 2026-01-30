import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic_types import OpType
from ....ops.atomic.concat import concat_ref


@KernelRegistry.register(
    OpType.CONCAT,
    [
        TensorSignature(DType.FP32, shape=None),  # Input A (Any Rank)
        TensorSignature(DType.FP32, shape=None),  # Input B (Any Rank)
        TensorSignature(DType.INT32, shape=(1,)),  # Axis (Scalar)
    ],
    target_dtype=DType.FP32,
    reference_factory=concat_ref,
)
def concat_generic(inputs, attrs=None, outputs=None):
    """
    Generic Concatenation of two tensors.
    inputs[0]: Tensor A
    inputs[1]: Tensor B
    inputs[2]: Axis (1-element 1D INT32 tensor)
    """
    a = inputs[0]
    b = inputs[1]
    axis = int(inputs[2][0])
    result = np.concatenate((a, b), axis=axis)
    if outputs is not None:
        outputs[0][:] = result
        return outputs[0]
    return result
