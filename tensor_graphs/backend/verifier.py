import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..ir.node import TensorNode
from ..ir.dtypes import DType, TensorSignature
from ..ops.atomic_types import OpType
from ..ops.registry import get_reference_factory
from .registry import KernelRegistry
from .reference import evaluate_graph


class VerificationError(Exception):
    pass


class KernelVerifier:
    @staticmethod
    def verify_all_composite_kernels() -> Dict[str, List[str]]:
        kernels = KernelRegistry.get_all_kernels()

        results = {
            "passed": [],
            "failed": [],
            "skipped": [],
        }

        for op_type, backends in kernels.items():
            if OpType.is_atomic(op_type):
                results["skipped"].append(f"{op_type} (Atomic)")
                continue

            ref_factory = get_reference_factory(op_type)
            if not ref_factory:
                error_msg = (
                    f"Op '{op_type}' has registered kernels but NO Reference Factory."
                )
                print(f"[FAIL] {error_msg}")
                results["failed"].append((op_type, error_msg))
                continue

            # Note: CompositeOp previously had sample_inputs().
            # With factories, we'd need to manually define samples or attach them to the function.
            # For now, we skip auto-sample generation unless attached.
            samples = getattr(ref_factory, "samples", [])

            if not samples:
                # Attempt to generate simple dummy inputs if possible or skip
                results["skipped"].append(f"{op_type} (No Samples Attached)")
                continue

            for backend, kernel_entries in backends.items():
                for entry in kernel_entries:
                    _, signatures, kernel_func = entry
                    try:
                        KernelVerifier._verify_single_kernel(
                            op_type, ref_factory, kernel_func, samples
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
