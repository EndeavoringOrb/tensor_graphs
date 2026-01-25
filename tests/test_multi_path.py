import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType, Backend
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.smart_executor import SmartExecutor
from tensor_graphs.ops.fused.math import FusedMulAdd


def test_smart_executor():
    # Build a simple graph: (A * B) + C
    a = TensorNode(OpType.INPUT, (4, 4), DType.FP32, [], name="A")
    b = TensorNode(OpType.INPUT, (4, 4), DType.FP32, [], name="B")
    c = TensorNode(OpType.INPUT, (4, 4), DType.FP32, [], name="C")

    # Use a CompositeOp
    fma = TensorNode("FusedMulAdd", (4, 4), DType.FP32, parents=[a, b, c], name="FMA")

    # Inputs
    inputs = {
        "A": np.random.randn(4, 4).astype(np.float32),
        "B": np.random.randn(4, 4).astype(np.float32),
        "C": np.random.randn(4, 4).astype(np.float32),
    }

    # Run with SmartExecutor in EXPLORE mode
    executor = SmartExecutor(db_path="test_benchmarks.db", policy="EXPLORE")
    print("Executing with EXPLORE policy...")
    result = executor.run(fma, inputs)

    # Verification
    expected = (inputs["A"] * inputs["B"]) + inputs["C"]
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print("Result matches expected NumPy output!")


if __name__ == "__main__":
    test_smart_executor()
