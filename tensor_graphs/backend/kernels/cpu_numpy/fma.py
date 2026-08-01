from ....ir.dtypes import DType, TensorSignature
from ....ops.fused.fma import fma_decomposition
from ...registry import KernelRegistry


@KernelRegistry.register(
    "FusedMulAdd",
    [
        TensorSignature(DType.FP32, shape=None),
        TensorSignature(DType.FP32, shape=None),
        TensorSignature(DType.FP32, shape=None),
    ],
    reference_factory=fma_decomposition,
)
def fma_generic(inputs, outputs, attrs):
    # Multiply first two elements, then add the third
    result = inputs[0] * inputs[1] + inputs[2]
    outputs[0][:] = result
