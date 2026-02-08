import pytest
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.dirty_propagation import DirtyPropagator

def test_elementwise_forward():
    # out = a + b
    # a: (10,), b: (1,) (broadcast)
    # a dirty: [2:5], b clean
    a = TensorNode(OpType.INPUT, DType.FP32, [], (10,), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (1,), "b")
    add = TensorNode(OpType.ADD, DType.FP32, [a, b], (10,), "add")

    a.dirty_region = (slice(2, 5),)
    b.dirty_region = None

    out_dirty = DirtyPropagator.propagate(add)
    # Since b is clean, output dirty is same as a (broadcasted)
    assert out_dirty == (slice(2, 5),)

    # Both dirty
    b.dirty_region = (slice(0, 1),)
    out_dirty = DirtyPropagator.propagate(add)
    # Union of a:[2:5] and b:[0:10] (broadcasted)
    # Wait, b:[0:1] broadcasts to [0:10] in output
    # symbolic_elementwise implementation:
    # aligned_ranges for b: [(sp.Integer(0), S_INF)] if is_d else clean
    # Wait, symbolic_elementwise says:
    # is_d = _is_dirty(inp)
    # pad_range = (sp.Piecewise((sp.Integer(0), is_d), (S_INF, True)), ...)
    # If b is dirty [0:1], it's dirty, so it broadcasts to (0, INF)
    assert out_dirty == (slice(0, None),)

def test_elementwise_backward():
    # out = a + b
    # a: (10,), b: (1,) (broadcast)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (10,), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (1,), "b")
    add = TensorNode(OpType.ADD, DType.FP32, [a, b], (10,), "add")

    # Request output region [2:5]
    out_region = (slice(2, 5),)
    in_slices = DirtyPropagator.get_input_slices(add, out_region)

    # Expected:
    # a needs [2:5]
    # b needs [0:1] (because it's size 1, it broadcasts to everyone)
    assert in_slices[0] == (slice(2, 5),)
    assert in_slices[1] == (slice(0, 1),)
