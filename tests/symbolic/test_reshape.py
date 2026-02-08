import pytest
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.dirty_propagation import DirtyPropagator

def test_reshape_forward():
    # out = a.reshape(10)
    # a: (2, 5) -> out: (10,)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 5), "a")
    res = TensorNode(OpType.RESHAPE, DType.FP32, [a], (10,), "reshape")

    # Reshape is currently conservative: any dirt makes it full dirty
    a.dirty_region = (slice(0, 1), slice(0, 1))
    out_dirty = DirtyPropagator.propagate(res)
    assert out_dirty == (slice(0, None),)

def test_reshape_backward():
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 5), "a")
    res = TensorNode(OpType.RESHAPE, DType.FP32, [a], (10,), "reshape")

    # Backward reshape is also conservative (fallback to full)
    # Wait, check if backward_reshape is implemented.
    # symbolic.py says:
    # @SymbolicPropagator.register_backward(OpType.RESHAPE) is NOT there.
    # Fallback is full.
    in_slices = DirtyPropagator.get_input_slices(res, (slice(2, 5),))
    assert in_slices[0] == (slice(0, 2), slice(0, 5))
