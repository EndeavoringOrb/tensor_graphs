from ...backend.registry import KernelRegistry
from ...ir.dtypes import DType, TensorSignature
from ...ops.atomic import OpType

# Fused Op typically implies generic shapes unless specific HW unit exists
@KernelRegistry.register(OpType.FUSED_MUL_ADD, [
    TensorSignature(DType.FP32, (None,)), 
    TensorSignature(DType.FP32, (None,)),
    TensorSignature(DType.FP32, (None,))
])
def fma_generic(inputs):
    # (A * B) + C
    return inputs[0] * inputs[1] + inputs[2]