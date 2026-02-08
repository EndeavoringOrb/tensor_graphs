
import time
import numpy as np
import sympy as sp
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.symbolic import SymbolicPropagator
from tensor_graphs.compiler.shape_inference import ShapeInference

def test_rmsnorm_compilation():
    print("Creating RMSNorm graph...")
    x = TensorNode(OpType.INPUT, DType.FP32, [], (1, 128, 640), name="x")
    scale = TensorNode(OpType.INPUT, DType.FP32, [], (640,), name="scale")
    eps = TensorNode(OpType.CONSTANT, DType.FP32, [], (1,), name="eps", attrs={"value": np.array([1e-6], dtype=np.float32)})
    
    rmsnorm = TensorNode("RMSNorm", DType.FP32, [x, scale, eps], (1, 128, 640), name="rmsnorm")
    
    # We need to make sure shapes are inferred for the decomposition to work
    # ShapeInference.infer([rmsnorm], {}) # Actually traceback showed it's during SymbolicPropagator.get_propagator
    
    print("Compiling symbolic propagator for RMSNorm...")
    start_time = time.time()
    try:
        func = SymbolicPropagator.get_propagator(rmsnorm)
        end_time = time.time()
        print(f"Compilation took {end_time - start_time:.4f} seconds")
    except Exception as e:
        print(f"Compilation failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rmsnorm_compilation()
