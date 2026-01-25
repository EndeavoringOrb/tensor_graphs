import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.fused.embedding import embedding_ref


# --- Embedding ---
@KernelRegistry.register(
    "Embedding",
    [
        TensorSignature(DType.INT32, (None, None)),
        TensorSignature(DType.FP32, (None, None)),
    ],
    reference_factory=embedding_ref,
)
def embedding_kernel(inputs, attrs=None):
    indices, weights = inputs
    return weights[indices.astype(int)]
