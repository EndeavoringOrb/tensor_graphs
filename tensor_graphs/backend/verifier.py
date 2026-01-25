import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..ir.node import TensorNode
from ..ir.dtypes import DType, TensorSignature
from ..ops.atomic import OpType
from ..ops.registry import get_composite_op
from .registry import KernelRegistry
from .reference import evaluate_graph


class VerificationError(Exception):
    pass


class KernelVerifier:
    @staticmethod
    def verify_all_composite_kernels() -> Dict[str, List[str]]:
        """
        Iterates over all registered kernels. If the op is Composite,
        verifies the kernel output matches the decomposed graph output.
        """
        kernels = KernelRegistry.get_all_kernels()

        results = {
            "passed": [],
            "failed": [],
            "skipped": [],  # Atomic ops or ops with no composite def
        }

        for op_type, backends in kernels.items():
            # 1. Skip Atomic Ops (They are the ground truth)
            if OpType.is_atomic(op_type):
                results["skipped"].append(f"{op_type} (Atomic)")
                continue

            # 2. Check for TensorNode Definition (CompositeOp)
            composite_op = get_composite_op(op_type)
            if not composite_op:
                error_msg = f"Op '{op_type}' has registered kernels but NO CompositeOp definition."
                print(f"[FAIL] {error_msg}")
                results["failed"].append((op_type, error_msg))
                continue

            # 3. Verify Each Kernel for this Composite Op
            samples = composite_op.sample_inputs()
            if not samples:
                msg = f"Op '{op_type}' has no test samples defined in sample_inputs()."
                print(f"[WARN] {msg}")
                results["skipped"].append(f"{op_type} (No Samples)")
                continue

            for backend, kernel_entries in backends.items():
                for entry in kernel_entries:
                    _, signatures, kernel_func = entry
                    try:
                        KernelVerifier._verify_single_kernel(
                            op_type, composite_op, kernel_func, samples
                        )
                        results["passed"].append(f"{op_type}::{backend}")
                    except Exception as e:
                        print(f"[FAIL] {op_type}::{backend} - {e}")
                        results["failed"].append((f"{op_type}::{backend}", str(e)))

        return results

    @staticmethod
    def _verify_single_kernel(op_type, composite_op, kernel_func, samples):
        for inputs_data, attrs in samples:
            # 1. Create Input Nodes
            input_nodes = []
            data_map = {}
            for i, val in enumerate(inputs_data):
                # Infer DType
                if val.dtype == np.float32:
                    dtype = DType.FP32
                elif val.dtype == np.int32:
                    dtype = DType.INT32
                elif val.dtype == bool:
                    dtype = DType.BOOL
                else:
                    dtype = DType.FP32  # Fallback

                name = f"in_{i}"
                node = TensorNode(OpType.INPUT, val.shape, dtype, [], name)
                input_nodes.append(node)
                data_map[name] = val

            # 2. Run Kernel
            try:
                kernel_out = kernel_func(inputs_data, attrs)
            except Exception as e:
                raise VerificationError(f"Kernel execution failed: {e}")

            # 3. Run Decomposition (Reference)
            try:
                decomp_root = composite_op.decompose(input_nodes, attrs)
            except Exception as e:
                raise VerificationError(f"Decomposition failed: {e}")

            # 4. Evaluate Reference Graph
            try:
                ref_out = evaluate_graph(decomp_root, data_map)
            except Exception as e:
                raise VerificationError(f"Reference graph execution failed: {e}")

            # 5. Compare
            try:
                np.testing.assert_allclose(
                    kernel_out, ref_out, rtol=1e-4, atol=1e-5, err_msg="Output mismatch"
                )
            except AssertionError as e:
                raise VerificationError(str(e))
