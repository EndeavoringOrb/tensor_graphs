import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic_types import OpType


def _dtype_to_numpy(dtype_enum):
    if dtype_enum == DType.FP32:
        return np.float32
    elif dtype_enum == DType.FP16:
        return np.float16
    elif dtype_enum == DType.INT32:
        return np.int32
    elif dtype_enum == DType.BOOL:
        return bool
    return np.float32


@KernelRegistry.register(OpType.CAST, [TensorSignature(DType.INT32, shape=None)])
@KernelRegistry.register(OpType.CAST, [TensorSignature(DType.FP32, shape=None)])
@KernelRegistry.register(OpType.CAST, [TensorSignature(DType.BOOL, shape=None)])
def cast_generic(inputs, attrs=None):
    if attrs is None:
        attrs = {}

    target_dtype = attrs.get("to", DType.FP32)
    return inputs[0].astype(_dtype_to_numpy(target_dtype))
