from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.dirty_propagation import DirtyPropagator


def test_unary_forward():
    # out = exp(a)
    # a: (10,) -> out: (10,)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (10,), "a")
    exp = TensorNode(OpType.EXP, DType.FP32, [a], (10,), "exp")

    # Unary ops propagate dirty region as is
    a.dirty_region = (slice(2, 5),)
    out_dirty = DirtyPropagator.propagate(exp)
    assert out_dirty == (slice(2, 5),)


def test_unary_backward():
    a = TensorNode(OpType.INPUT, DType.FP32, [], (10,), "a")
    exp = TensorNode(OpType.EXP, DType.FP32, [a], (10,), "exp")

    # Backward elementwise: out region maps directly to in region
    in_slices = DirtyPropagator.get_input_slices(exp, (slice(2, 5),))
    assert in_slices[0] == (slice(2, 5),)
