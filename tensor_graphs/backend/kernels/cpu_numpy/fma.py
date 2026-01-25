from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.fused.fma import fused_mul_add_ref


@KernelRegistry.register(
    "FusedMulAdd",
    [
        TensorSignature(DType.FP32, (None,)),
        TensorSignature(DType.FP32, (None,)),
        TensorSignature(DType.FP32, (None,)),
    ],
    reference_factory=fused_mul_add_ref,
)
def fma_generic(inputs, attrs=None):
    return inputs[0] * inputs[1] + inputs[2]
