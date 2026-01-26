import pytest
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, cast, Optional
import copy

from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType, Backend, TensorSignature
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.registry import KernelRegistry
from tensor_graphs.backend.reference import evaluate_graph
from tensor_graphs.ops.registry import get_reference_factory

# Ensure all kernels are loaded
import tensor_graphs.backend.kernels


# ==============================================================================
# 1. Input Generation Helpers
# ==============================================================================


class DataGenerator:
    """Generates valid random inputs for specific OpTypes and Signatures."""

    @staticmethod
    def get_numpy_dtype(dtype: DType):
        if dtype == DType.FP32:
            return np.float32
        elif dtype == DType.FP16:
            return np.float16
        elif dtype == DType.INT32:
            return np.int32
        elif dtype == DType.BOOL:
            return bool
        return np.float32

    @staticmethod
    def random_tensor(shape, dtype: DType):
        np_dtype = DataGenerator.get_numpy_dtype(dtype)
        if dtype == DType.BOOL:
            return np.random.choice([True, False], size=shape)
        elif dtype == DType.INT32:
            return np.random.randint(-10, 10, size=shape).astype(np_dtype)
        else:
            return np.random.randn(*shape).astype(np_dtype)

    @staticmethod
    def generate(
        op_type: str,
        signatures: Tuple[TensorSignature, ...],
        backend: Optional[Backend] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Returns (inputs_list, attributes_dict)
        """
        attrs = {}
        inputs = []

        # --- Special Handling for Shape-dependent Ops ---

        if op_type == OpType.DOT:
            # Matrix Multiplication: (M, K) @ (K, N)
            m, k, n = 4, 8, 4
            inputs.append(DataGenerator.random_tensor((m, k), DType.FP32))
            inputs.append(DataGenerator.random_tensor((k, n), DType.FP32))
            return inputs, attrs

        elif op_type == OpType.RESHAPE:
            # Data: (2, 3, 4), Shape: [24] or [6, 4]
            # Signature: [Data, ShapeTensor]
            data = DataGenerator.random_tensor((2, 3, 4), DType.FP32)
            # Target shape tensor
            target_shape = np.array([6, 4], dtype=np.int32)
            inputs = [data, target_shape]
            return inputs, attrs

        elif op_type == OpType.PERMUTE:
            # Data: (2, 3, 4), Perm: [2, 0, 1]
            data = DataGenerator.random_tensor((2, 3, 4), DType.FP32)
            perm = np.array([2, 0, 1], dtype=np.int32)
            inputs = [data, perm]
            return inputs, attrs

        elif op_type == OpType.CONCAT:
            # Data A: (2, 4), Data B: (2, 4), Axis: 0 or 1
            a = DataGenerator.random_tensor((2, 4), DType.FP32)
            b = DataGenerator.random_tensor((2, 4), DType.FP32)
            axis = np.array([1], dtype=np.int32)
            inputs = [a, b, axis]
            return inputs, attrs

        elif op_type == OpType.SLICE:
            # Fixed slice scenario for robustness
            data = DataGenerator.random_tensor((10, 10), DType.FP32)
            # Standard params for slice_generic kernel: data, starts, ends, steps
            if len(signatures) == 4:
                starts = np.array([0, 0], dtype=np.int32)
                ends = np.array([5, 5], dtype=np.int32)
                steps = np.array([1, 1], dtype=np.int32)
                inputs = [data, starts, ends, steps]
            else:
                inputs = [data]
                attrs = {"starts": [0, 0], "ends": [5, 5], "steps": [1, 1]}
            return inputs, attrs

        elif op_type == OpType.ARANGE:
            # Start, Stop, Step
            inputs = [
                np.array([0], dtype=np.int32),
                np.array([10], dtype=np.int32),
                np.array([1], dtype=np.int32),
            ]
            return inputs, attrs

        elif op_type == OpType.TRIU:
            data = DataGenerator.random_tensor((4, 4), DType.FP32)
            k = np.array([1], dtype=np.int32)
            inputs = [data, k]
            return inputs, attrs

        elif op_type == OpType.FILL:
            # Value, Shape
            target_dtype = signatures[0].dtype if signatures else DType.FP32
            if target_dtype == DType.INT32:
                val = np.array([7], dtype=np.int32)
            else:
                val = np.array([3.14], dtype=np.float32)

            shape = np.array([2, 2], dtype=np.int32)
            inputs = [val, shape]
            attrs = {"target_shape": (2, 2)}
            return inputs, attrs

        elif op_type == OpType.GATHER:
            # Data: (Vocab, Dim), Indices: (Any)
            vocab_size = 10
            dim = 4
            data = DataGenerator.random_tensor((vocab_size, dim), DType.FP32)
            # Indices must be within [0, vocab_size)
            indices = np.random.randint(0, vocab_size, size=(3,), dtype=np.int32)
            inputs = [data, indices]
            return inputs, attrs

        elif op_type in (OpType.MAX, OpType.SUM):
            # inputs[0] is data
            data = DataGenerator.random_tensor((4, 8), DType.FP32)
            inputs.append(data)

            # Check if there is a second input (axis)
            if len(signatures) > 1:
                # Axis must be valid for (4, 8). Rank 2. Valid: 0, 1.
                axis_val = np.array([0], dtype=np.int32)
                inputs.append(axis_val)

            return inputs, attrs

        elif op_type == OpType.REPEAT:
            data = DataGenerator.random_tensor((4, 4), DType.FP32)
            inputs.append(data)
            if len(signatures) > 1:
                # Input 2 is repeats
                inputs.append(np.array([2], dtype=np.int32))
            else:
                # Attribute repeats
                attrs["repeats"] = 2
            return inputs, attrs

        elif op_type == OpType.COPY_TO:
            data = DataGenerator.random_tensor((4, 4), signatures[0].dtype)
            inputs = [data]

            # Guess target backend based on where the kernel is likely running or intended result
            if backend:
                attrs["target_backend"] = backend.value
            else:
                attrs["target_backend"] = "cpu_numpy"
            return inputs, attrs

        elif op_type == OpType.CAST:
            # Input, Attrs[to]
            data = DataGenerator.random_tensor((4, 4), signatures[0].dtype)
            inputs = [data]
            attrs = {"to": DType.FP32}
            return inputs, attrs

        # --- Generic Element-wise / Broadcast handling ---
        # Default shape for generic tests
        base_shape = (4, 8)

        for sig in signatures:
            if sig.is_scalar() or (sig.shape and sig.shape == (1,)):
                inputs.append(DataGenerator.random_tensor((1,), sig.dtype))
            elif sig.shape and sig.shape == (None,):
                inputs.append(DataGenerator.random_tensor((base_shape[0],), sig.dtype))
            else:
                inputs.append(DataGenerator.random_tensor(base_shape, sig.dtype))

        # Special Logic: For RMSNorm, Last input is eps (scalar)
        if op_type == "RMSNorm":
            # x, weight, eps
            inputs[0] = DataGenerator.random_tensor(base_shape, DType.FP32)
            inputs[1] = DataGenerator.random_tensor((base_shape[-1],), DType.FP32)
            inputs[2] = np.array([1e-5], dtype=np.float32)

        # Special Logic: RoPE (x, cos, sin)
        if op_type == "RoPE":
            inputs[0] = DataGenerator.random_tensor(base_shape, DType.FP32)
            inputs[1] = DataGenerator.random_tensor(base_shape, DType.FP32)
            inputs[2] = DataGenerator.random_tensor(base_shape, DType.FP32)

        return inputs, attrs


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

    if backend == Backend.GPU_TORCH:
        # Move inputs to GPU Torch
        inputs_gpu = []
        for x in inputs_np:
            if isinstance(x, np.ndarray):
                t = torch.from_numpy(x).cuda()
                inputs_gpu.append(t)
            else:
                inputs_gpu.append(x)  # scalars

        try:
            out_gpu = kernel_func(inputs_gpu, attrs)
            # Move output back to CPU Numpy
            if isinstance(out_gpu, torch.Tensor):
                actual_output = out_gpu.detach().cpu().numpy()
            else:
                actual_output = out_gpu
        except Exception as e:
            pytest.fail(f"Kernel Execution Failed on GPU: {e}")

    else:
        # Execute on CPU (NumPy)
        try:
            actual_output = kernel_func(inputs_np, attrs)
        except Exception as e:
            pytest.fail(f"Kernel Execution Failed on CPU: {e}")

    # 5. Execute Reference (Golden Truth)
    # We build the graph using the reference factory and evaluate it.

    input_nodes = []
    feed_dict = {}

    for i, data in enumerate(inputs_np):
        name = f"in_{i}"
        # Determine dtype
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
