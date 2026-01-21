# File: demo_autocast.py
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.compiler.dispatch import resolve_dispatch
# Import implementations to register them
import tensor_graphs.backend.ops.implementations 

print("=== Setting up Graph ===")
# Scenario: 
# Input A: FP32 Scalar (1,)
# Input B: FP8 Matrix (4, 4)
# Op: Add

node_a_fp32 = TensorNode(OpType.INPUT, (1,), DType.FP32, [], "scalar_fp32")
node_b_fp8  = TensorNode(OpType.INPUT, (4, 4), DType.FP8E4M3, [], "matrix_fp8")

# User tries to create an Add node.
# Note: The output implies we *want* to add them, even if we don't have the kernel yet.
target_node = TensorNode(OpType.ADD, (4, 4), DType.FP32, [node_a_fp32, node_b_fp8], "add_op")

print(f"Original Request: {target_node.signature} = {node_a_fp32.signature} + {node_b_fp8.signature}")

print("\n=== Running Dispatcher ===")
# This should fail to find (FP32, FP8) kernel, and rewrite to (FP32, Cast(FP8)->FP32)
executable_node = resolve_dispatch(target_node)

print("\n=== Resulting Graph Structure ===")
print(f"Root Op: {executable_node.op_type}")
print(f"Parent 1: {executable_node.parents[0]}") # Should be Original Scalar
print(f"Parent 2: {executable_node.parents[1]}") # Should be CAST node