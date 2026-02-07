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
            DType.INT32, shape=None, backend=Backend.CPU_NUMPY
        ),  # Shape tensor is any shape (flattened internally)
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=reshape_ref,
)
def reshape_generic(inputs, outputs, attrs):
    """
    Generic Reshape Implementation.
    inputs[0]: Data tensor (Any Rank)
    inputs[1]: Target shape (1D tensor of integers, possibly with extra dims)
    """
    data = inputs[0]
    target_shape = inputs[1]

    # Flatten shape tensor if it has more than 1 dimension
    shape_array = target_shape.astype(int)
    if shape_array.ndim > 1:
        print(f"WARNING: flattening shape array in reshape: {target_shape}")
        shape_array = shape_array.flatten()
    shape_tuple = tuple(shape_array)
    result = np.reshape(data, shape_tuple)
    outputs[0][:] = result
