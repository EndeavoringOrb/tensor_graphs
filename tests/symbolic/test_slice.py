from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.dirty_propagation import DirtyPropagator


def test_slice_forward():
    # out = a[2:10:2]
    # a: (20,) -> out: (4,)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (20,), "a")
    slc = TensorNode(
        OpType.SLICE,
        DType.FP32,
        [a],
        (4,),
        "slice",
        attrs={"starts": [2], "ends": [10], "steps": [2]},
    )

    # a dirty [4:6]
    # touches out index (4-2)//2 = 1 to (6-2)//2 = 2
    a.dirty_region = (slice(4, 6),)
    out_dirty = DirtyPropagator.propagate(slc)
    assert out_dirty == (slice(1, 2),)


def test_slice_backward():
    a = TensorNode(OpType.INPUT, DType.FP32, [], (20,), "a")
    slc = TensorNode(
        OpType.SLICE,
        DType.FP32,
        [a],
        (4,),
        "slice",
        attrs={"starts": [2], "ends": [10], "steps": [2]},
    )

    res = DirtyPropagator.get_input_slices(slc, (slice(0, 2),))
    # out[0:2] touches a[2 + 0*2] = a[2], a[2 + 1*2] = a[4].
    # range in a: [2, 5)
    assert res[0] is None
