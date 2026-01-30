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
    # Multiply first two elements
    np.multiply(inputs[0], inputs[1], out=outputs[0])
    # Then add the third element
    outputs[0] += inputs[2]
    return outputs[0]
