# tensor_graphs/backend/kernels/cpu_numpy/reshape.py
from ....backend.registry import KernelRegistry
from ....ir.dtypes import Backend, DType, TensorSignature
from ....ops.atomic.reshape import reshape_ref
from ....ops.atomic_types import OpType


@KernelRegistry.register(
    OpType.RESHAPE,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
        TensorSignature(DType.INT32, shape=None, backend=Backend.CPU_NUMPY),
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=reshape_ref,
    inplace=True,
)
def reshape_inplace(inputs, outputs, attrs):
    pass
