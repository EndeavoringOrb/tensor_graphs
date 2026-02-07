from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.gather import gather_ref


@KernelRegistry.register(
    OpType.GATHER,
    [
        TensorSignature(DType.FP32, shape=(None, None), backend=Backend.CPU_NUMPY),
        TensorSignature(DType.INT32, shape=None, backend=Backend.CPU_NUMPY),
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=gather_ref,
)
def gather_embedding(inputs, outputs, attrs):
    """
    Gather / Embedding Lookup Implementation.
    inputs[0]: Data matrix - commonly (Vocab, Dim)
    inputs[1]: Indices (Any Rank)
    """
    data = inputs[0]
    indices = inputs[1]

    result = data[indices]
    outputs[0][:] = result
