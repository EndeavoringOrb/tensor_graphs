import pytest
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, cast, Optional

from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType, Backend, TensorSignature
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.registry import KernelRegistry
from tensor_graphs.backend.executor import evaluate_graph
from tensor_graphs.ops.registry import get_reference_factory
from tensor_graphs.benchmark.data_gen import DataGenerator  # Import DataGenerator

# Ensure all kernels are loaded
import tensor_graphs.backend.kernels

# ==============================================================================
# 2. Test Discovery
# ==============================================================================


def get_all_test_kernels():
    """
    Yields tuples of (op_type, backend, signatures, target_dtype, kernel_func, kernel_name)
    """
    registry = KernelRegistry.get_all_kernels()

    for op_type, backends in registry.items():
        for backend, kernels in backends.items():
            for entry in kernels:
                # Entry: (backend, signatures, target_dtype, func)
                _, signatures, target_dtype, func = entry

                # Create a readable ID for pytest
                sig_str = ",".join([str(s.dtype.value) for s in signatures])
                name = f"{op_type}-{backend.value}-[{sig_str}]"

                yield (op_type, backend, signatures, target_dtype, func, name)


# ==============================================================================
# 3. Correctness Test
# ==============================================================================


@pytest.mark.parametrize(
    "op_type, backend, signatures, target_dtype, kernel_func, kernel_name",
    get_all_test_kernels(),
    ids=lambda x: x[5] if isinstance(x, tuple) and len(x) > 5 else None,
)
def test_kernel_correctness(
    op_type, backend, signatures, target_dtype, kernel_func, kernel_name
):
    """
    Validates that a specific kernel implementation produces the same output
    as the reference graph decomposition (evaluated on CPU NumPy).
    """

    # 1. Skip if GPU backend required but not available
    if backend == Backend.GPU_TORCH and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # 2. Check if a reference factory exists
    ref_factory = get_reference_factory(op_type)
    if ref_factory is None:
        if not OpType.is_atomic(op_type):
            pytest.skip(
                f"No reference factory for {op_type}, cannot verify correctness."
            )

    # 3. Generate Inputs
    try:
        inputs_np, attrs = DataGenerator.generate(op_type, signatures, backend)
    except Exception as e:
        pytest.skip(f"Failed to generate inputs for {op_type}: {e}")

    # Update attrs if target_dtype was part of registration (e.g. Cast)
    if target_dtype and op_type == OpType.CAST:
        attrs["to"] = target_dtype

    # 4. Execute Candidate Kernel
    actual_output = None

    # Prepare inputs based on their individual registered signature backends
    prepared_inputs = []
    for i, x in enumerate(inputs_np):
        # Handle cases where input generation produces fewer inputs than signature (attributes)
        if i >= len(signatures):
            break

        sig_backend = signatures[i].backend
        if sig_backend == Backend.GPU_TORCH:
            # Ensure CUDA is available if the signature explicitly requires it
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available for required GPU input")
            t = torch.from_numpy(x).cuda()
            prepared_inputs.append(t)
        elif sig_backend == Backend.CPU_TORCH:
            t = torch.from_numpy(x)
            prepared_inputs.append(t)
        else:
            # Default to CPU_NUMPY
            prepared_inputs.append(x)

    try:
        # Execute the kernel with correctly placed inputs
        out = kernel_func(prepared_inputs, attrs)

        # Move output back to CPU Numpy for comparison against reference
        if isinstance(out, torch.Tensor):
            actual_output = out.detach().cpu().numpy()
        else:
            actual_output = out
    except Exception as e:
        pytest.fail(f"Kernel Execution Failed ({backend.value}): {e}")

    # 5. Execute Reference (Golden Truth)
    input_nodes = []
    feed_dict = {}

    for i, data in enumerate(inputs_np):
        name = f"in_{i}"
        dt = DType.FP32  # Default
        if data.dtype == np.float32:
            dt = DType.FP32
        elif data.dtype == np.int32:
            dt = DType.INT32
        elif data.dtype == bool:
            dt = DType.BOOL

        node = TensorNode(
            OpType.INPUT, data.shape, dt, [], name=name, backend=Backend.CPU_NUMPY
        )
        input_nodes.append(node)
        feed_dict[name] = data

    try:
        if ref_factory:
            graph_root = ref_factory(input_nodes, attrs)
        else:
            graph_root = TensorNode(
                op_type,
                input_nodes[0].shape,
                DType.FP32,
                input_nodes,
                "ref_atomic",
                attrs=attrs,
            )

        expected_output = evaluate_graph(graph_root, feed_dict)

    except Exception as e:
        pytest.fail(f"Reference Graph Evaluation Failed: {e}")

    # 6. Compare
    if np.isscalar(actual_output) and isinstance(expected_output, np.ndarray):
        actual_output = np.array(actual_output)
    if np.isscalar(expected_output) and isinstance(actual_output, np.ndarray):
        expected_output = np.array(expected_output)

    actual_arr = cast(np.ndarray, np.asarray(actual_output))
    expected_arr = cast(np.ndarray, np.asarray(expected_output))

    rtol = 1e-4
    atol = 1e-5

    if backend == Backend.GPU_TORCH:
        rtol = 1e-3
        atol = 1e-4

    np.testing.assert_allclose(
        actual_arr,
        expected_arr,
        rtol=rtol,
        atol=atol,
        err_msg=f"Kernel {kernel_name} output mismatch against reference graph",
    )
