from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.dirty_propagation import DirtyPropagator


def test_dot_forward():
    # out = a @ b
    # a: (2, 3), b: (3, 4) -> out: (2, 4)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 3), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (3, 4), "b")
    dot = TensorNode(OpType.DOT, DType.FP32, [a, b], (2, 4), "dot")

    # a dirty in row 1, b clean
    a.dirty_region = [(slice(1, 2), slice(0, 3))]
    b.dirty_region = None

    out_dirty = DirtyPropagator.propagate(dot)
    assert out_dirty == [(slice(1, 2, None), slice(0, 4, None))]

def test_dot_backward():
    # a: (2, 3), b: (3, 4) -> out: (2, 4)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 3), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (3, 4), "b")
    dot = TensorNode(OpType.DOT, DType.FP32, [a, b], (2, 4), "dot")

    # Request row 1, col 2:3
    res = DirtyPropagator.get_input_slices(dot, [(slice(1, 2), slice(2, 3))])
    # a needs row 1, all K (3)
    # b needs all K (3), col 2:3
    assert res[0] == [(slice(1, 2), slice(0, 3))]
    assert res[1] == [(slice(0, 3), slice(2, 3))]
