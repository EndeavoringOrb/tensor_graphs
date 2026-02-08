from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.dirty_propagation import DirtyPropagator


def test_reduce_forward():
    # out = sum(a, axis=0)
    # a: (5, 10) -> out: (10,)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (5, 10), "a")
    reduce_sum = TensorNode(
        OpType.SUM, DType.FP32, [a], (10,), "sum", attrs={"axis": 0, "keepdims": False}
    )

    # a dirty rows [1:3], all cols [0:10]
    # Since we reduce over axis 0, and axis 0 is dirty, the whole result might be dirty?
    # No, if only cols [2:4] were dirty across some rows, only those output cols are dirty.
    a.dirty_region = (slice(1, 3), slice(2, 4))
    out_dirty = DirtyPropagator.propagate(reduce_sum)
    assert out_dirty == (slice(2, 4),)


def test_reduce_backward():
    a = TensorNode(OpType.INPUT, DType.FP32, [], (5, 10), "a")
    reduce_sum = TensorNode(
        OpType.SUM, DType.FP32, [a], (10,), "sum", attrs={"axis": 0, "keepdims": False}
    )

    # Backward reduce fallback to full
    in_slices = DirtyPropagator.get_input_slices(reduce_sum, (slice(2, 4),))
    assert in_slices[0] == (slice(0, 5), slice(0, 10))
