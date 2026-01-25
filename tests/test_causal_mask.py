import unittest
import numpy as np
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.backend.reference import evaluate_graph


# Simple GraphBuilder similar to the one in examples
class SimpleGraphBuilder:
    def __init__(self):
        self.constant_inputs = {}

    def constant(self, value, name, dtype=DType.INT32):
        val_arr = np.array(
            value, dtype=np.int32 if dtype == DType.INT32 else np.float32
        )
        if val_arr.ndim == 0:
            val_arr = val_arr.reshape(1)
        node = TensorNode(
            op_type=OpType.CONSTANT,
            shape=val_arr.shape,
            dtype=dtype,
            parents=[],
            name=name,
            attrs={"value": val_arr},
        )
        self.constant_inputs[name] = val_arr
        return node

    def fill(self, value_node, shape_node, target_shape):
        return TensorNode(
            OpType.FILL,
            target_shape,
            value_node.dtype,
            [value_node, shape_node],
            "fill",
        )

    def triu(self, x, k_node):
        return TensorNode(OpType.TRIU, x.shape, DType.FP32, [x, k_node], "triu")

    def mul(self, a, b):
        return TensorNode(OpType.MUL, a.shape, DType.FP32, [a, b], f"mul_{a.name}")

    def reshape(self, x, target_shape, shape_node):
        return TensorNode(
            OpType.RESHAPE,
            target_shape,
            DType.FP32,
            [x, shape_node],
            f"reshape_{x.name}",
        )


class TestCausalMask(unittest.TestCase):
    def test_causal_mask_gen(self):
        b = SimpleGraphBuilder()
        S = 4

        # 1. Ones Matrix (S, S) using Fill
        one_val = b.constant([1.0], "mask_one_val", DType.FP32)
        shape_ss = b.constant([S, S], f"shape_ss_{S}")
        ones_node = b.fill(one_val, shape_ss, (S, S))

        # 2. Triu(ones, k=1) -> Upper triangle (strictly upper) are 1s
        k_node = b.constant([1], "k_triu_mask")
        mask_tri = b.triu(ones_node, k_node)

        # 3. Mask * -1e9
        neg_inf = b.constant([-1e9], "neg_inf_mask", DType.FP32)
        mask_scaled = b.mul(mask_tri, neg_inf)

        # 4. Reshape to (1, 1, S, S)
        shape_final = b.constant([1, 1, S, S], f"shape_mask_final_{S}")
        mask_out = b.reshape(mask_scaled, (1, 1, S, S), shape_final)

        feed_dict = {**b.constant_inputs}
        mask_val = evaluate_graph(mask_out, feed_dict)

        expected_mask = np.triu(np.ones((S, S), dtype=np.float32), k=1) * -1e9
        expected_mask = expected_mask.reshape((1, 1, S, S))

        np.testing.assert_allclose(mask_val, expected_mask)
        print("Causal mask generation test passed!")


if __name__ == "__main__":
    unittest.main()
