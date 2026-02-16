from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.dirty_propagation import DirtyPropagator


def test_permute_forward():
    # out = a.permute(1, 0)
    # a: (2, 3) -> out: (3, 2)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 3), "a")
    perm = TensorNode(
        OpType.PERMUTE, DType.FP32, [a], (3, 2), "perm", attrs={"dims": [1, 0]}
    )

    # a dirty rows [1:2], all cols [0:3]
    a.dirty_region = [(slice(1, 2), slice(0, 3))]
    out_dirty = DirtyPropagator.propagate(perm)
    # Result should be dirty in out[0:3, 1:2]
    assert out_dirty == [(slice(0, 3), slice(1, 2))]


def test_permute_backward():
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 3), "a")
    perm = TensorNode(
        OpType.PERMUTE, DType.FP32, [a], (3, 2), "perm", attrs={"dims": [1, 0]}
    )

    # Request out[0:2, 1:2] -> a[1:2, 0:2]
    res = DirtyPropagator.get_input_slices(perm, [(slice(0, 2), slice(1, 2))])
    assert res[0] == [(slice(1, 2), slice(0, 2))]
