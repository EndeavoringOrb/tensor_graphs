from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.dirty_propagation import DirtyPropagator


def test_gather_forward():
    # out = gather(data, indices)
    # data: (10, 5), indices: (3,) -> out: (3, 5)
    data = TensorNode(OpType.INPUT, DType.FP32, [], (10, 5), "data")
    indices = TensorNode(OpType.INPUT, DType.INT32, [], (3,), "indices")
    gather = TensorNode(OpType.GATHER, DType.FP32, [data, indices], (3, 5), "gather")

    # If data is dirty, output is likely full dirty on the gather axis?
    # Actually symbolic.py implementation for GATHER:
    # gather only checks indices dirtyness for output ranges.
    # Logic: if indices[i] is dirty, out[i] is dirty.
    indices.dirty_region = [(slice(1, 2),)]
    data.dirty_region = None
    out_dirty = DirtyPropagator.propagate(gather)
    assert out_dirty == [(slice(1, 2), slice(0, 5))]


def test_gather_backward():
    data = TensorNode(OpType.INPUT, DType.FP32, [], (10, 5), "data")
    indices = TensorNode(OpType.INPUT, DType.INT32, [], (3,), "indices")
    gather = TensorNode(OpType.GATHER, DType.FP32, [data, indices], (3, 5), "gather")

    # Gather backward precision
    in_slices = DirtyPropagator.get_input_slices(gather, [(slice(0, 1), slice(0, 5))])
    assert in_slices[0] == [(slice(0, 10), slice(0, 5))]
    assert in_slices[1] == [(slice(0, 1),)]
