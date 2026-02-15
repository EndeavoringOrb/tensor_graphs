from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.dirty_propagation import DirtyPropagator


def test_concat_forward():
    # out = concat([a, b], axis=0)
    # a: (5,), b: (5,) -> out: (10,)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (5,), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (5,), "b")
    concat = TensorNode(
        OpType.CONCAT, DType.FP32, [a, b], (10,), "concat", attrs={"axis": 0}
    )

    # a dirty [1:3], b clean
    a.dirty_region = [(slice(1, 3),)]
    b.dirty_region = None
    out_dirty = DirtyPropagator.propagate(concat)
    assert out_dirty == (slice(1, 3),)

    # b dirty [1:3], a clean
    a.dirty_region = None
    b.dirty_region = (slice(1, 3),)
    out_dirty = DirtyPropagator.propagate(concat)
    # Shifted by len(a)=5 -> [6, 8]
    assert out_dirty == (slice(6, 8),)


test_concat_forward()


def test_concat_backward():
    a = TensorNode(OpType.INPUT, DType.FP32, [], (5,), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (5,), "b")
    concat = TensorNode(
        OpType.CONCAT, DType.FP32, [a, b], (10,), "concat", attrs={"axis": 0}
    )

    # Case 1: Region entirely in 'a'
    res1 = DirtyPropagator.get_input_slices(concat, (slice(2, 4),))
    assert res1[0] == (slice(2, 4),)
    assert res1[1] is None

    # Case 2: Region spanning both
    res2 = DirtyPropagator.get_input_slices(concat, (slice(4, 7),))
    assert res2[0] == (slice(4, 5),)
    assert res2[1] == (slice(0, 2),)
