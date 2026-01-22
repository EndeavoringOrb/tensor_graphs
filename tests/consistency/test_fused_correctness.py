import pytest
import numpy as np
from typing import Dict
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.backend.reference import evaluate_graph
from tensor_graphs.ops.atomic import OpType

# Import Fused Ops
from tensor_graphs.ops.fused.activation import GELU
from tensor_graphs.ops.fused.norm import RMSNorm


def create_input(name, shape, val_fill=1.0):
    node = TensorNode(OpType.INPUT, shape, DType.FP32, [], name)
    data = np.full(shape, val_fill, dtype=np.float32)
    return node, data


def test_gelu_consistency():
    """Verify GELU Kernel matches Decomposition."""
    # 1. Setup
    x_node, x_val = create_input("x", (10, 10), val_fill=0.5)
    inputs_map = {"x": x_val}

    # 2. Kernel Execution
    gelu_op = GELU()
    fused_node = TensorNode(gelu_op.op_type, (10, 10), DType.FP32, [x_node], "fused")
    res_kernel = evaluate_graph(fused_node, inputs_map)

    # 3. Decomposition Execution
    # Explicitly decompose
    inputs_map.update(
        {
            "c_0.044": np.full((1,), 0.044715, dtype=np.float32),
            "c_sqrt_2_pi": np.full((1,), np.sqrt(2 / np.pi), dtype=np.float32),
            "c_0.5": np.full((1,), 0.5, dtype=np.float32),
            "c_1.0": np.full((1,), 1.0, dtype=np.float32),
        }
    )
    decomp_root = gelu_op.decompose([x_node])
    res_decomp = evaluate_graph(decomp_root, inputs_map)

    # 4. Compare
    np.testing.assert_allclose(
        res_kernel, res_decomp, rtol=1e-5, err_msg="GELU Kernel != Decomposition"
    )


def test_rmsnorm_consistency():
    """Verify RMSNorm Kernel matches Decomposition."""
    # 1. Setup
    shape = (4, 16)
    x_node, x_val = create_input("x", shape, 2.0)
    scale_node, scale_val = create_input("scale", (16,), 0.5)
    eps_node, eps_val = create_input("eps", (1,), 1e-6)

    inputs_map: Dict[str, np.ndarray] = {"x": x_val, "scale": scale_val, "eps": eps_val}

    # 2. Kernel Execution
    norm_op = RMSNorm()
    fused_node = TensorNode(
        norm_op.op_type, shape, DType.FP32, [x_node, scale_node, eps_node], "fused"
    )
    res_kernel = evaluate_graph(fused_node, inputs_map)

    # 3. Decomposition Execution
    inputs_map.update(
        {
            "axis_last": np.array([-1], dtype=np.int32),
            "one_const": np.full((1,), 1.0, dtype=np.float32),
            "n_elements": np.full((1,), 16, dtype=np.float32),
        }
    )
    decomp_root = norm_op.decompose([x_node, scale_node, eps_node])
    res_decomp = evaluate_graph(decomp_root, inputs_map)

    # 4. Compare
    np.testing.assert_allclose(
        res_kernel, res_decomp, rtol=1e-5, err_msg="RMSNorm Kernel != Decomposition"
    )
