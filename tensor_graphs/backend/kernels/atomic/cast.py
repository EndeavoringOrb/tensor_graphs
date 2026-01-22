import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType

@KernelRegistry.register_cast(src=DType.FP8E4M3, dst=DType.FP32)
def cast_fp8_to_fp32(inputs):
    val = inputs[0]
    return val.astype(np.float32)