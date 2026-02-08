import numpy as np
import pytest
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.dirty_propagation import DirtyPropagator


def test_elementwise_input_slices():
    # out = a + b
    # a: (10,), b: (1,) (broadcast)
    # output: (10,)
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


def test_concat_input_slices():
    # out = concat([a, b], axis=0)
    # a: (5,), b: (5,)
    # output: (10,)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (5,), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (5,), "b")
    concat = TensorNode(
        OpType.CONCAT, DType.FP32, [a, b], (10,), "concat", attrs={"axis": 0}
    )

    # Case 1: Region entirely in 'a'
    res1 = DirtyPropagator.get_input_slices(concat, (slice(2, 4),))
    assert res1[0] == (slice(2, 4),)
    assert res1[1] == (slice(0, 0),)

    # Case 2: Region spanning both
    res2 = DirtyPropagator.get_input_slices(concat, (slice(4, 7),))
    # output [4, 7) ->
    # [4, 5) in 'a' -> rel [4, 5)
    # [5, 7) in 'b' -> rel [0, 2)
    assert res2[0] == (slice(4, 5),)
    assert res2[1] == (slice(0, 2),)


def test_slice_input_slices():
    # out = a[2:10:2]
    # a: (20,)
    # out: (4,)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (20,), "a")
    # out[i] = a[2 + i*2]
    # out[0:2] touches a[2 + 0*2] = a[2], a[2 + 1*2] = a[4].
    # range in a: [2, 5)
    slc = TensorNode(
        OpType.SLICE,
        DType.FP32,
        [a],
        (4,),
        "slice",
        attrs={"starts": [2], "ends": [10], "steps": [2]},
    )

    res = DirtyPropagator.get_input_slices(slc, (slice(0, 2),))
    # Correct range in 'a': from index of os (0) to index of oe-1 (1).
    # in_s = 2 + 0*2 = 2
    # in_e = 2 + (1)*2 + 1 = 5
    assert res[0] == (slice(2, 5),)


def test_matmul_input_slices():
    # out = dot(a, b)
    # a: (2, 3), b: (3, 4)
    # out: (2, 4)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 3), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (3, 4), "b")
    dot = TensorNode(OpType.DOT, DType.FP32, [a, b], (2, 4), "dot")

    # Request row 1, col 2:3
    res = DirtyPropagator.get_input_slices(dot, (slice(1, 2), slice(2, 3)))
    # a needs row 1, all K (3)
    # b needs all K (3), col 2:3
    assert res[0] == (slice(1, 2), slice(0, 3))
    assert res[1] == (slice(0, 3), slice(2, 3))


def test_permute_input_slices():
    # out = a.permute(1, 0)
    # a: (2, 3) -> out: (3, 2)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 3), "a")
    perm = TensorNode(
        OpType.PERMUTE, DType.FP32, [a], (3, 2), "perm", attrs={"dims": [1, 0]}
    )

    # Request out[0:2, 1:2]
    # This corresponds to a[1:2, 0:2]
    res = DirtyPropagator.get_input_slices(perm, (slice(0, 2), slice(1, 2)))
    assert res[0] == (slice(1, 2), slice(0, 2))
