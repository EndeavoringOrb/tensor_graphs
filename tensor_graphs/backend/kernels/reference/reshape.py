import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.reshape import reshape_ref


@KernelRegistry.register(
    OpType.RESHAPE,
    [
        TensorSignature(
            DType.FP32, shape=None, backend=Backend.CPU_NUMPY
        ),  # Matches any rank/shape
        TensorSignature(
            DType.INT32, shape=(None,), backend=Backend.CPU_NUMPY
        ),  # Shape tensor is always 1D
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=reshape_ref,
)
def reshape_generic(inputs, attrs=None, outputs=None):
    """
    Generic Reshape Implementation.
    inputs[0]: Data tensor (Any Rank)
    inputs[1]: Target shape (1D tensor of integers)
    """
    data = inputs[0]
    target_shape = inputs[1]

    # Convert numpy array of shape dims to tuple of ints
    shape_tuple = tuple(target_shape.astype(int))
    result = np.reshape(data, shape_tuple)
    if outputs is not None:
        outputs[0][:] = result
        return outputs[0]
    return result
