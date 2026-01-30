import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.slice import slice_ref


@KernelRegistry.register(
    OpType.SLICE,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),  # Data
        TensorSignature(
            DType.INT32, shape=(None,), backend=Backend.CPU_NUMPY
        ),  # Starts
        TensorSignature(DType.INT32, shape=(None,), backend=Backend.CPU_NUMPY),  # Ends
        TensorSignature(DType.INT32, shape=(None,), backend=Backend.CPU_NUMPY),  # Steps
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=slice_ref,
)
@KernelRegistry.register(
    OpType.SLICE,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),  # Data
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=slice_ref,
)
def slice_generic(inputs, attrs=None, outputs=None):
    """
    Generic Slice Implementation.
    inputs[0]: Data tensor (Any Rank)

    If using attributes:
    attrs["starts"]: list[int]
    attrs["ends"]: list[int]
    attrs["steps"]: list[int]

    If using input nodes (Legacy):
    inputs[1]: Starts (1D INT32)
    inputs[2]: Ends (1D INT32)
    inputs[3]: Steps (1D INT32)
    """
    data = inputs[0]

    if attrs and "starts" in attrs:
        starts = attrs["starts"]
        ends = attrs["ends"]
        steps = attrs.get("steps", [1] * len(starts))
    elif len(inputs) >= 4:
        starts = inputs[1].astype(int)
        ends = inputs[2].astype(int)
        steps = inputs[3].astype(int)
    else:
        raise ValueError(
            "Slice requires either attributes or 4 inputs (data, starts, ends, steps)"
        )

    # Validation
    if len(starts) != len(ends) or len(starts) != len(steps):
        raise ValueError("Starts, Ends, and Steps must be same length")

    slices = []
    for i in range(len(starts)):
        s = int(starts[i]) if starts[i] is not None else None
        e = int(ends[i]) if ends[i] is not None else None
        st = int(steps[i]) if steps[i] is not None else 1
        slices.append(slice(s, e, st))

    result = data[tuple(slices)]
    if outputs is not None:
        outputs[0][:] = result
        return outputs[0]
    return result
