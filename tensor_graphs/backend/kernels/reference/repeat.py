import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.repeat import repeat_ref


@KernelRegistry.register(
    OpType.REPEAT,
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=repeat_ref,
)
def repeat_generic(inputs, outputs, attrs):
    if attrs is None or "repeats" not in attrs:
        raise ValueError("Repeat kernel requires 'repeats' attribute")

    data = inputs[0]
    repeats = int(attrs["repeats"])
    axis = attrs.get("axis", 0)

    if len(data.shape) < 2:
        result = np.repeat(data, repeats)
    else:
        result = np.repeat(data, repeats, axis=axis)

    outputs[0][:] = result
