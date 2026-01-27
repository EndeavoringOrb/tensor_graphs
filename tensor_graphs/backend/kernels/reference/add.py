import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic.add import add_ref
from ....ops.atomic_types import OpType


# --- Generic Tensor (Any Rank) ---
@KernelRegistry.register(
    OpType.ADD,
    [TensorSignature(DType.FP32, shape=None), TensorSignature(DType.FP32, shape=None)],
    reference_factory=add_ref,
)
def add_generic_tensor(inputs, attrs=None, outputs=None):
    """
    inputs: [a, b]
    outputs: [out_view]
    """
    if outputs is None:
        # Fallback for non-static contexts (if any)
        return np.add(inputs[0], inputs[1])

    # Explicit output writing
    np.add(inputs[0], inputs[1], out=outputs[0])
    return outputs[0]
