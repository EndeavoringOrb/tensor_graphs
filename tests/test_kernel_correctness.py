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
from tensor_graphs.compiler.shape_inference import ShapeInference
from tensor_graphs.ir.graph import topological_sort

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

    # --- Architecture-based Output Shape/Type Determination ---
    input_nodes_for_shape = []
    known_values = {}

    def _map_to_dtype(obj):
        if isinstance(obj, torch.Tensor):
            dt = obj.dtype
            if dt == torch.float32:
                return DType.FP32
            if dt == torch.float16:
                return DType.FP16
            if dt == torch.int32:
                return DType.INT32
            if dt == torch.bool:
                return DType.BOOL
            return DType.FP32
        else:
            dt = obj.dtype
            # handle numpy types
            if dt == np.float32:
                return DType.FP32
            if dt == np.float16:
                return DType.FP16
            if dt == np.int32:
                return DType.INT32
            if dt == bool or dt == np.bool_:
                return DType.BOOL
            return DType.FP32

    for i, val in enumerate(prepared_inputs):
        node = TensorNode(
            OpType.INPUT, _map_to_dtype(val), [], tuple(val.shape), name=f"in_{i}"
        )
        input_nodes_for_shape.append(node)
        known_values[f"in_{i}"] = val

    # 1. Build the reference/decomposed subgraph to determine output shape
    if ref_factory:
        subgraph_root = ref_factory(input_nodes_for_shape, attrs)
    else:
        # Fallback for atomic ops without a factory in the registry
        subgraph_root = TensorNode(
            op_type,
            input_nodes_for_shape[0].dtype if input_nodes_for_shape else DType.FP32,
            input_nodes_for_shape,
            (None,),
            "temp_out",
            attrs=attrs,
        )

    # 2. Run Shape Inference on the entire decomposed subgraph
    subgraph_nodes = topological_sort(subgraph_root)
    # We include input nodes to ensure their shapes are available to the inference engine
    inference_nodes = input_nodes_for_shape + [
        n for n in subgraph_nodes if n not in input_nodes_for_shape
    ]

    try:
        ShapeInference.infer(inference_nodes, known_values)
    except Exception as e:
        print(f"Warning: Shape inference failed for {op_type}: {e}")

    # 3. The true output shape is the inferred shape of the subgraph root
    output_shape = subgraph_root.shape
    out_dt_enum = subgraph_root.dtype
    if not (
        isinstance(output_shape, Tuple)
        and len(output_shape) > 0
        and all([isinstance(item, int) for item in output_shape])
    ):
        raise ValueError(f"output_shape could not be inferred")

    first_input = prepared_inputs[0] if prepared_inputs else None
    is_torch = isinstance(first_input, torch.Tensor)

    if is_torch:
        if out_dt_enum == DType.FP16:
            output_dtype = torch.float16
        elif out_dt_enum == DType.INT32:
            output_dtype = torch.int32
        elif out_dt_enum == DType.BOOL:
            output_dtype = torch.bool
        else:
            output_dtype = torch.float32
    else:
        if out_dt_enum == DType.FP16:
            output_dtype = np.float16
        elif out_dt_enum == DType.INT32:
            output_dtype = np.int32
        elif out_dt_enum == DType.BOOL:
            output_dtype = np.bool_
        else:
            output_dtype = np.float32

    # Allocate output buffer
    if is_torch:
        safe_shape = cast(Tuple[int], output_shape)
        output_dtype = cast(torch.dtype, output_dtype)
        output = torch.empty(
            safe_shape,
            dtype=output_dtype,
            device="cuda" if first_input.is_cuda else "cpu",
        )
    else:
        safe_shape = cast(Tuple[int], output_shape)
        output_dtype = cast(np.dtype, output_dtype)
        output = np.empty(safe_shape, dtype=output_dtype)

    # Call kernel with new signature: (inputs, outputs, attrs)
    kernel_func(prepared_inputs, [output], attrs)

    # Output is now in 'output'
    if isinstance(output, torch.Tensor):
        actual_output = output.detach().cpu().numpy()
    else:
        actual_output = output

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
        elif data_np.dtype == bool or data_np.dtype == np.bool_:
            dt = DType.BOOL

        node = TensorNode(
            OpType.INPUT, dt, [], data_np.shape, name=name, backend=Backend.CPU_NUMPY
        )
        input_nodes.append(node)
        feed_dict[name] = data_np

    if ref_factory:
        graph_root = ref_factory(input_nodes, attrs)
    else:
        graph_root = TensorNode(
            op_type,
            DType.FP32,
            input_nodes,
            name="ref_atomic",
            attrs=attrs,
        )
    raw_expected = evaluate_graph(graph_root, feed_dict)
    # Ensure reference output is moved to CPU/NumPy
    if isinstance(raw_expected, torch.Tensor):
        expected_output = raw_expected.detach().cpu().numpy()
    else:
        expected_output = raw_expected

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


# For debugging with vscode debugger
# for item in get_all_test_kernels():
#     if item[0] == "GELU":
#         test_kernel_correctness(*item)
