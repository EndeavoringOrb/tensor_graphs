import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.fused.fma import fma_decomposition


@KernelRegistry.register(
    "FusedMulAdd",
    [
        TensorSignature(DType.FP32, (None,)),
        TensorSignature(DType.FP32, (None,)),
        TensorSignature(DType.FP32, (None,)),
    ],
    reference_factory=fma_decomposition,
)
def fma_generic(inputs, attrs=None, outputs=None):
    # Multiply first two elements, then add the third
    result = inputs[0] * inputs[1] + inputs[2]
    if outputs is not None:
        outputs[0][:] = result
        return outputs[0]
    return result
