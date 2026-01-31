import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.triu import triu_ref


@KernelRegistry.register(
    OpType.TRIU,
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=triu_ref,
)
def triu_generic(inputs, outputs, attrs):
    data = inputs[0]
    k = attrs.get("k", 0) if attrs else 0

    result = np.triu(data, k=k)
    outputs[0][:] = result