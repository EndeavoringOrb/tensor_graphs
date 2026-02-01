import numpy as np
import pytest
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ir.buffer import StorageType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.planner import Planner
from tensor_graphs.compiler.compiler import Compiler
from tensor_graphs.backend.executor import Executor


def test_static_execution_basic():
    # Build a simple graph: (a * b) + c
    a = TensorNode(
        OpType.INPUT, DType.FP32, [], (2, 2), "a", storage_type=StorageType.TRANSIENT
    )
    b = TensorNode(
        OpType.INPUT, DType.FP32, [], (2, 2), "b", storage_type=StorageType.TRANSIENT
    )
    c = TensorNode(
        OpType.INPUT, DType.FP32, [], (2, 2), "c", storage_type=StorageType.TRANSIENT
    )

    mul = TensorNode(OpType.MUL, DType.FP32, [a, b], (2, 2), "mul")
    add = TensorNode(OpType.ADD, DType.FP32, [mul, c], (2, 2), "add")

    planner = Planner()
    recipe = planner.plan(add)

    compiler = Compiler()
    compiled_graph = compiler.compile(recipe)

    executor = Executor(compiled_graph)

    # Prepare inputs
    a_val = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_val = np.array([[5, 6], [7, 8]], dtype=np.float32)
    c_val = np.array([[10, 10], [10, 10]], dtype=np.float32)

    feed_dict = {"a": a_val, "b": b_val, "c": c_val}

    result = executor.run(feed_dict)

    expected = (a_val * b_val) + c_val
    np.testing.assert_allclose(result, expected)


def test_static_execution_persistent():
    # Build a graph with persistent weights: (x * weight) + bias
    x = TensorNode(
        OpType.INPUT, DType.FP32, [], (1, 4), "x", storage_type=StorageType.TRANSIENT
    )
    w = TensorNode(
        OpType.INPUT, DType.FP32, [], (1, 4), "w", storage_type=StorageType.PERSISTENT
    )
    b = TensorNode(
        OpType.INPUT, DType.FP32, [], (1, 4), "b", storage_type=StorageType.PERSISTENT
    )

    mul = TensorNode(OpType.MUL, DType.FP32, [x, w], (1, 4), "mul")
    add = TensorNode(OpType.ADD, DType.FP32, [mul, b], (1, 4), "add")

    planner = Planner()
    recipe = planner.plan(add)

    compiler = Compiler()
    compiled_graph = compiler.compile(recipe)

    executor = Executor(compiled_graph)

    # Prepare data
    x_val = np.array([[1, 2, 3, 4]], dtype=np.float32)
    w_val = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    b_val = np.array([[1, 1, 1, 1]], dtype=np.float32)

    # Load persistent weights once
    executor.load_weights({"w": w_val, "b": b_val})

    # Run with transient input
    result = executor.run({"x": x_val})

    expected = (x_val * w_val) + b_val
    np.testing.assert_allclose(result, expected)
