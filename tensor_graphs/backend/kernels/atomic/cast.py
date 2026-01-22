import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType


@KernelRegistry.register_cast(src=DType.FP8E4M3, dst=DType.FP32)
def cast_fp8_to_fp32(inputs):
    val = inputs[0]
    return val.astype(np.float32)


@KernelRegistry.register_cast(src=DType.INT32, dst=DType.FP32)
def cast_int32_to_fp32(inputs):
    val = inputs[0]
    return val.astype(np.float32)


# Generic Cast Kernel (Used when OpType.CAST is explicitly in the graph)
@KernelRegistry.register(OpType.CAST, [TensorSignature(DType.INT32, shape=None)])
def cast_generic_int32(inputs):
    # We assume target is FP32 for this specific signature/context
    return inputs[0].astype(np.float32)


@KernelRegistry.register(OpType.CAST, [TensorSignature(DType.FP32, shape=None)])
def cast_generic_fp32(inputs):
    # Identity cast for FP32->FP32 (if it occurs)
    return inputs[0]
