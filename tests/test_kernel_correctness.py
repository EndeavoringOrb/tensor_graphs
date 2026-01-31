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
from tensor_graphs.benchmark.data_gen import DataGenerator

# Ensure all kernels are loaded
import tensor_graphs.backend.kernels


def get_all_test_kernels():
    registry = KernelRegistry.get_all_kernels()
    for op_type, backends in registry.items():
        for backend, kernels in backends.items():
            for entry in kernels:
                _, signatures, target_dtype, func = entry
                sig_str = ",".join([str(s.dtype.value) for s in signatures])
                name = f"{op_type}-{backend.value}-[{sig_str}]"
                yield (op_type, backend, signatures, target_dtype, func, name)


@pytest.mark.parametrize(
    "op_type, backend, signatures, target_dtype, kernel_func, kernel_name",
    get_all_test_kernels(),
    ids=lambda x: x[5] if isinstance(x, tuple) and len(x) > 5 else None,
)
def test_kernel_correctness(
    op_type, backend, signatures, target_dtype, kernel_func, kernel_name
):
    # 1. Skip if GPU backend required but not available
    if backend == Backend.GPU_TORCH and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # 2. Check if a reference factory exists
    ref_factory = get_reference_factory(op_type)
    if ref_factory is None and not OpType.is_atomic(op_type):
        pytest.skip(f"No reference factory for {op_type}, cannot verify correctness.")

    # 3. Generate Inputs (DataGenerator handles backend conversion)
    try:
        inputs_list, attrs = DataGenerator.generate(op_type, signatures, backend)
    except Exception as e:
        pytest.skip(f"Failed to generate inputs for {op_type}: {e}")

    if target_dtype and op_type == OpType.CAST:
        attrs["to"] = target_dtype

    # 4. Prepare inputs for Candidate Kernel
    # We must ensure they are on the specific device/type required by the signature
    prepared_inputs = []
    for i, x in enumerate(inputs_list):
        if i >= len(signatures):
            break

        sig_backend = signatures[i].backend
        if sig_backend == Backend.GPU_TORCH:
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available for required GPU input")
            # Convert to tensor if it isn't one, then move to CUDA
            t = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
            prepared_inputs.append(t.cuda())
        elif sig_backend == Backend.CPU_TORCH:
            t = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
            prepared_inputs.append(t.cpu())
        else:
            # For CPU_NUMPY, ensure it is a numpy array
            if isinstance(x, torch.Tensor):
                prepared_inputs.append(x.detach().cpu().numpy())
            else:
                prepared_inputs.append(x)

    # 5. Execute Candidate Kernel
    actual_output = None
    try:
        # Create pre-allocated output buffer (empty, same shape as first input for unary, or infer from op)
        first_input = prepared_inputs[0]
        if isinstance(first_input, torch.Tensor):
            output_shape = first_input.shape
            output_dtype = torch.float32
        else:
            output_shape = first_input.shape
            output_dtype = np.float32

        # Handle special cases for output shape
        if op_type == OpType.ADD:
            if len(prepared_inputs) >= 2:
                output_shape = np.broadcast_shapes(prepared_inputs[0].shape, prepared_inputs[1].shape)
        elif op_type == OpType.MUL:
            if len(prepared_inputs) >= 2:
                output_shape = np.broadcast_shapes(prepared_inputs[0].shape, prepared_inputs[1].shape)
        elif op_type == OpType.DIVIDE:
            if len(prepared_inputs) >= 2:
                output_shape = np.broadcast_shapes(prepared_inputs[0].shape, prepared_inputs[1].shape)
        elif op_type == OpType.DOT:
            # Output is matrix product result
            pass  # Keep first input shape as starting point
        elif op_type == OpType.CONCAT:
            # Concatenation - need to compute from attrs
            axis = attrs.get("axis", 0) if attrs else 0
            concat_dim = sum(arr.shape[axis] for arr in prepared_inputs)
            new_shape = list(prepared_inputs[0].shape)
            new_shape[axis] = concat_dim
            output_shape = tuple(new_shape)
        elif op_type == OpType.RESHAPE:
            if attrs and "shape" in attrs:
                output_shape = tuple(attrs["shape"])
        elif op_type == OpType.PERMUTE:
            if attrs and "axes" in attrs:
                output_shape = tuple(first_input.shape[ax] for ax in attrs["axes"])
        elif op_type == OpType.SLICE:
            # Slice output shape depends on starts/ends
            pass  # Keep original shape
        elif op_type == OpType.REPEAT:
            if attrs and "repeats" in attrs:
                repeats = attrs["repeats"]
                output_shape = tuple(s * r for s, r in zip(first_input.shape, repeats))
        elif op_type == OpType.ARANGE:
            start = attrs.get("start", 0) if attrs else 0
            step = attrs.get("step", 1) if attrs else 1
            n = attrs.get("n") or len(prepared_inputs[1]) if len(prepared_inputs) > 1 else 10
            output_shape = (n,)
        elif op_type == OpType.FILL:
            if len(prepared_inputs) >= 2:
                output_shape = tuple(int(x) for x in prepared_inputs[1])
        elif op_type == OpType.GATHER:
            # Output shape from indices
            pass  # Keep original
        elif op_type == OpType.WHERE:
            pass  # Keep first input shape
        elif op_type == OpType.COPY_TO:
            output_shape = first_input.shape
        elif op_type == OpType.CAST:
            if target_dtype == DType.FP16:
                output_dtype = np.float16
            elif target_dtype == DType.FP32:
                output_dtype = np.float32
            elif target_dtype == DType.INT32:
                output_dtype = np.int32
            elif target_dtype == DType.BOOL:
                output_dtype = np.bool_
        elif op_type == OpType.TRIU:
            output_shape = first_input.shape

        # Allocate output buffer
        if isinstance(first_input, torch.Tensor):
            output = torch.empty(output_shape, dtype=output_dtype, device="cuda" if first_input.is_cuda else "cpu")
        else:
            output = np.empty(output_shape, dtype=output_dtype)

        # Call kernel with new signature: (inputs, outputs, attrs)
        kernel_func(prepared_inputs, [output], attrs)

        # Output is now in 'output'
        if isinstance(output, torch.Tensor):
            actual_output = output.detach().cpu().numpy()
        else:
            actual_output = output
    except Exception as e:
        pytest.fail(f"Kernel Execution Failed ({backend.value}): {e}")

    # 6. Execute Reference (Golden Truth) on CPU NumPy
    input_nodes = []
    feed_dict = {}

    for i, data in enumerate(inputs_list):
        name = f"in_{i}"

        # CRITICAL: Reference evaluation MUST use NumPy arrays
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data

        dt = DType.FP32
        if data_np.dtype == np.float32:
            dt = DType.FP32
        elif data_np.dtype == np.int32:
            dt = DType.INT32
        elif data_np.dtype == bool:
            dt = DType.BOOL

        node = TensorNode(
            OpType.INPUT, data_np.shape, dt, [], name=name, backend=Backend.CPU_NUMPY
        )
        input_nodes.append(node)
        feed_dict[name] = data_np

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
        raw_expected = evaluate_graph(graph_root, feed_dict)
        # Ensure reference output is moved to CPU/NumPy
        if isinstance(raw_expected, torch.Tensor):
            expected_output = raw_expected.detach().cpu().numpy()
        else:
            expected_output = raw_expected
    except Exception as e:
        pytest.fail(f"Reference Graph Evaluation Failed: {e}")

    # 7. Comparison
    actual_arr = np.asarray(actual_output)
    expected_arr = np.asarray(expected_output)

    # Allow slightly higher tolerance for GPU kernels
    rtol, atol = (1e-3, 1e-4) if backend == Backend.GPU_TORCH else (1e-4, 1e-5)

    np.testing.assert_allclose(
        actual_arr,
        expected_arr,
        rtol=rtol,
        atol=atol,
        err_msg=f"Kernel {kernel_name} output mismatch",
    )
