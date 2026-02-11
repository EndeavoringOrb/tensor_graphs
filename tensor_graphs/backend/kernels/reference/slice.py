from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.slice import slice_ref


@KernelRegistry.register(
    OpType.SLICE,
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=slice_ref,
)
def slice_generic(inputs, outputs, attrs):
    print("DEBUG slice_generic:")
    print(f"  input shape:  {inputs[0].shape}")
    print(f"  output shape: {outputs[0].shape}")
    print(f"  attrs:        {attrs}")
    if not attrs or "starts" not in attrs:
        raise ValueError("Slice kernel requires attributes (starts, ends)")

    data = inputs[0]
    starts = attrs["starts"]
    ends = attrs["ends"]
    steps = attrs.get("steps", [1] * len(starts))

    slices = []
    for i in range(len(starts)):
        s = int(starts[i]) if starts[i] is not None else None
        e = int(ends[i]) if ends[i] is not None else None
        st = int(steps[i]) if steps[i] is not None else 1
        slices.append(slice(s, e, st))

    result = data[tuple(slices)]
    outputs[0][:] = result
