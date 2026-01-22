import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType


@KernelRegistry.register(
    OpType.GATHER,
    [
        TensorSignature(DType.FP32, shape=None),  # Data / Embedding Matrix
        TensorSignature(DType.INT32, shape=None),  # Indices
    ],
)
def gather_embedding(inputs, attrs=None):
    """
    Gather / Embedding Lookup Implementation.
    inputs[0]: Data tensor (Any Rank) - commonly (Vocab, Dim)
    inputs[1]: Indices (Any Rank)

    Performs data[indices] (axis=0 gather).
    """
    data = inputs[0]
    indices = inputs[1].astype(int)  # Ensure indices are int

    return data[indices]
