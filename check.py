import os
import sys
import glob
import numpy as np
import inspect
from typing import List, Any
import importlib

# Ensure the package is in path
sys.path.append(os.getcwd())

from tensor_graphs.backend.registry import KernelRegistry
from tensor_graphs.ops.registry import get_composite_op
from tensor_graphs.ir.dtypes import DType, TensorSignature
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.backend.reference import evaluate_graph

# Import all kernels to ensure registry is populated
import tensor_graphs.backend.kernels


def color_print(msg, color="green"):
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "reset": "\033[0m",
    }
    if os.name == "nt" and not os.getenv("ANSICON"):
        print(msg)
    else:
        print(f"{colors.get(color, '')}{msg}{colors['reset']}")


def fail(msg):
    color_print(f"FAILED: {msg}", "red")
    sys.exit(1)


# ==============================================================================
# Rule 1: Complain if there are untested reference kernels
# ==============================================================================
def check_reference_coverage():
    print("----------------------------------------------------------------")
    print("Checking Reference Kernel Test Coverage...")
    print("----------------------------------------------------------------")

    ref_dir = os.path.join("tensor_graphs", "backend", "reference", "atomic")
    test_dir = os.path.join("tests", "ops")

    ref_files = [
        f for f in os.listdir(ref_dir) if f.endswith(".py") and f != "__init__.py"
    ]

    missing_tests = []

    for ref_file in ref_files:
        op_name = ref_file[:-3]  # remove .py
        expected_test = f"test_{op_name}.py"
        expected_path = os.path.join(test_dir, expected_test)

        if not os.path.exists(expected_path):
            missing_tests.append(op_name)

    if missing_tests:
        color_print(
            f"Missing tests for reference kernels: {', '.join(missing_tests)}", "red"
        )
        print("\nPlease create the following files:")
        for m in missing_tests:
            print(f"  - tests/ops/test_{m}.py")
        return False

    color_print("All reference kernels have corresponding test files.", "green")
    return True


# ==============================================================================
# Rule 2: Verify Non-Reference Kernels against Decomposition/Reference
# ==============================================================================


def generate_dummy_data(sig: TensorSignature, rng: np.random.Generator) -> np.ndarray:
    shape = []
    if sig.shape is None:
        shape = [16]  # Arbitrary vector for generic inputs
    else:
        for dim in sig.shape:
            if dim is None:
                shape.append(16)
            else:
                shape.append(dim)

    safe_shape = tuple(shape)

    if sig.dtype == DType.FP32:
        return rng.standard_normal(safe_shape).astype(np.float32)
    elif sig.dtype == DType.INT32:
        return rng.integers(0, 100, size=safe_shape).astype(np.int32)
    elif sig.dtype == DType.BOOL:
        return rng.choice([True, False], size=safe_shape)
    else:
        return np.zeros(safe_shape)


def get_reference_implementation(op_type: str, inputs: List[Any]):
    """
    Executes the operation using ONLY reference logic.
    """
    composite = get_composite_op(op_type)

    if composite:
        # Build a mini-graph to decompose
        input_nodes = []
        input_map = {}
        for i, val in enumerate(inputs):
            name = f"in_{i}"
            dtype = DType.FP32
            if val.dtype == np.int32:
                dtype = DType.INT32
            elif val.dtype == bool:
                dtype = DType.BOOL

            node = TensorNode(OpType.INPUT, val.shape, dtype, [], name)
            input_nodes.append(node)
            input_map[name] = val

        try:
            decomp_root = composite.decompose(input_nodes)
        except Exception as e:
            return None, f"Decomposition failed: {e}"

        # Temporary monkeypatch to force reference kernels only
        original_selector = KernelRegistry.select_best_kernel

        def reference_only_selector(op_t, sigs, backend=None):
            # 1. Get candidates for this backend
            candidates = KernelRegistry._kernels.get(op_t, {}).get(backend, [])

            best_score = -1
            best_kernel = None

            for cand_backend, pattern_sigs, kernel_func in candidates:
                # 2. FILTER: Only allow reference kernels
                if not kernel_func.__module__.startswith(
                    "tensor_graphs.backend.reference"
                ):
                    continue

                # 3. SCORE: Use standard signature matching logic
                score = KernelRegistry._score_candidate(pattern_sigs, sigs)

                if score > best_score:
                    best_score = score
                    best_kernel = kernel_func

            return best_kernel

        try:
            KernelRegistry.select_best_kernel = reference_only_selector
            res = evaluate_graph(decomp_root, input_map)
        except NotImplementedError as e:
            return None, f"Ref Kernel Missing: {e}"
        except Exception as e:
            return None, f"Graph Eval Error: {e}"
        finally:
            KernelRegistry.select_best_kernel = original_selector

        return res, None

    else:
        # Atomic Op: Find the generic reference kernel directly
        candidates = []

        # Search all backends (usually CPU_NUMPY) for a reference implementation
        best_ref_kernel = None
        best_score = -1

        # Create signatures for current inputs to score against
        concrete_sigs = []
        for inp in inputs:
            dtype = DType.FP32
            if inp.dtype == np.int32:
                dtype = DType.INT32
            elif inp.dtype == bool:
                dtype = DType.BOOL
            concrete_sigs.append(TensorSignature(dtype, inp.shape))

        for backend, entries in KernelRegistry._kernels.get(op_type, {}).items():
            for _, sigs, func in entries:
                if func.__module__.startswith("tensor_graphs.backend.reference"):
                    score = KernelRegistry._score_candidate(sigs, concrete_sigs)
                    if score > best_score:
                        best_score = score
                        best_ref_kernel = func

        if not best_ref_kernel:
            return None, f"No reference atomic kernel found for {op_type}"

        return best_ref_kernel(inputs), None


def check_kernel_consistency():
    print("\n----------------------------------------------------------------")
    print("Checking Non-Reference Kernel Consistency...")
    print("----------------------------------------------------------------")

    rng = np.random.default_rng(42)
    failures = []
    checked_count = 0

    for op_type, backend_map in KernelRegistry._kernels.items():
        for backend, kernels in backend_map.items():
            for _, sigs, kernel_func in kernels:

                # Skip Reference Kernels
                if kernel_func.__module__.startswith("tensor_graphs.backend.reference"):
                    continue

                kernel_name = kernel_func.__name__
                print(f"Validating {op_type} :: {kernel_name}...", end=" ")

                try:
                    inputs = [generate_dummy_data(sig, rng) for sig in sigs]
                except Exception as e:
                    print(f"SKIPPED (Input Gen Error: {e})")
                    continue

                # Optimized Run
                try:
                    res_opt = kernel_func(inputs)
                except Exception as e:
                    print("ERROR")
                    color_print(f"  Kernel execution failed: {e}", "red")
                    failures.append(f"{kernel_name}: Runtime Error")
                    continue

                # Reference Run
                res_ref, err = get_reference_implementation(op_type, inputs)

                if err:
                    print("SKIPPED")
                    color_print(f"  Reference error: {err}", "yellow")
                    continue

                # Comparison
                try:
                    np.testing.assert_allclose(res_opt, res_ref, rtol=1e-3, atol=1e-4)
                    color_print("PASS", "green")
                    checked_count += 1
                except AssertionError as e:
                    print("FAIL")
                    color_print(f"  Mismatch found: {e}", "red")
                    failures.append(f"{kernel_name}: Mismatch")

    print("----------------------------------------------------------------")
    if failures:
        return False

    if checked_count == 0:
        color_print("Warning: No non-reference kernels found to test.", "yellow")
    else:
        color_print(
            f"All {checked_count} non-reference kernels verified against reference.",
            "green",
        )

    return True


if __name__ == "__main__":
    success_coverage = check_reference_coverage()
    success_consistency = check_kernel_consistency()

    if not success_coverage or not success_consistency:
        sys.exit(1)

    sys.exit(0)
