import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic_types import OpType


@KernelRegistry.register(
    OpType.GATHER,
    [
        TensorSignature(DType.FP32, shape=(None, None)),  # Data / Embedding Matrix
        TensorSignature(DType.INT32, shape=(None,)),  # Indices
    ],
)
def gather_embedding(inputs, attrs=None):
    """
    Gather / Embedding Lookup Implementation.
    inputs[0]: Data matrix - commonly (Vocab, Dim)
    inputs[1]: Indices (Any Rank)

    Performs data[indices] (axis=0 gather).
    """
    data = inputs[0]
    indices = inputs[1]

    return data[indices]
