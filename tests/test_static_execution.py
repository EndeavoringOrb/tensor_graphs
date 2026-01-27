import numpy as np
import pytest
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ir.buffer import StorageType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.planner import Planner
from tensor_graphs.compiler.compiler import Compiler
from tensor_graphs.backend.static_executor import StaticExecutor

def test_static_execution_basic():
    # Build a simple graph: (a * b) + c
    a = TensorNode(OpType.INPUT, (2, 2), DType.FP32, [], "a", storage_type=StorageType.TRANSIENT)
    b = TensorNode(OpType.INPUT, (2, 2), DType.FP32, [], "b", storage_type=StorageType.TRANSIENT)
    c = TensorNode(OpType.INPUT, (2, 2), DType.FP32, [], "c", storage_type=StorageType.TRANSIENT)

    mul = TensorNode(OpType.MUL, (2, 2), DType.FP32, [a, b], "mul")
    add = TensorNode(OpType.ADD, (2, 2), DType.FP32, [mul, c], "add")

    planner = Planner()
    recipe = planner.plan(add)

    compiler = Compiler()
    compiled_graph = compiler.compile(recipe)

    executor = StaticExecutor(compiled_graph)

    # Prepare inputs
    a_val = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_val = np.array([[5, 6], [7, 8]], dtype=np.float32)
    c_val = np.array([[10, 10], [10, 10]], dtype=np.float32)

    feed_dict = {
        "a": a_val,
        "b": b_val,
        "c": c_val
    }

    result = executor.run(feed_dict)

    expected = (a_val * b_val) + c_val
    np.testing.assert_allclose(result, expected)

def test_static_execution_persistent():
    # Build a graph with persistent weights: (x * weight) + bias
    x = TensorNode(OpType.INPUT, (1, 4), DType.FP32, [], "x", storage_type=StorageType.TRANSIENT)
    w = TensorNode(OpType.INPUT, (1, 4), DType.FP32, [], "w", storage_type=StorageType.PERSISTENT)
    b = TensorNode(OpType.INPUT, (1, 4), DType.FP32, [], "b", storage_type=StorageType.PERSISTENT)

    mul = TensorNode(OpType.MUL, (1, 4), DType.FP32, [x, w], "mul")
    add = TensorNode(OpType.ADD, (1, 4), DType.FP32, [mul, b], "add")

    planner = Planner()
    recipe = planner.plan(add)

    compiler = Compiler()
    compiled_graph = compiler.compile(recipe)

    executor = StaticExecutor(compiled_graph)

    # Prepare data
    x_val = np.array([[1, 2, 3, 4]], dtype=np.float32)
    w_val = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    b_val = np.array([[1, 1, 1, 1]], dtype=np.float32)

    # Load persistent weights once
    executor.load_weights({
        "w": w_val,
        "b": b_val
    })

    # Run with transient input
    result = executor.run({"x": x_val})

    expected = (x_val * w_val) + b_val
    np.testing.assert_allclose(result, expected)
