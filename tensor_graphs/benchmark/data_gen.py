import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
import torch
from ..ir.dtypes import DType, Backend, TensorSignature
from ..ops.atomic_types import OpType
from ..ir.node import TensorNode
from ..ops.atomic.copy_to import copy_to_ref


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

        # Handle scalar / empty shape normalization
        if shape is None:
            shape = (1,)

        if dtype == DType.BOOL:
            return np.random.choice([True, False], size=shape)
        elif dtype == DType.INT32:
            return np.random.randint(-10, 10, size=shape).astype(np_dtype)
        else:
            return np.random.randn(*shape).astype(np_dtype)

    @staticmethod
    def _prepare_inputs_for_backend(
        inputs_with_backends: List[Tuple[np.ndarray, Optional[Backend]]],
        global_backend: Optional[Backend] = None,
    ) -> List[Any]:
        """
        Converts inputs to the specific backends specified in their signatures.
        If a signature does not specify a backend, it defaults to the global_backend
        or CPU_NUMPY if global_backend is None.
        """
        prepared = []
        for inp, target_backend in inputs_with_backends:
            # Determine the actual backend to use for this input
            if target_backend is None:
                # Fallback to global backend if signature doesn't specify
                target_backend = global_backend

            # If global backend is also None, default to CPU_NUMPY
            if target_backend is None:
                target_backend = Backend.CPU_NUMPY

            if target_backend == Backend.CPU_NUMPY:
                # Already numpy, just ensure it's contiguous
                if isinstance(inp, np.ndarray):
                    prepared.append(np.ascontiguousarray(inp))
                else:
                    prepared.append(np.array(inp, dtype=np.float32))

            elif target_backend == Backend.GPU_TORCH:
                if not torch.cuda.is_available():
                    raise ValueError(
                        f"Cannot prepare inputs for {target_backend.value}: CUDA is not available."
                    )

                # Convert numpy arrays to GPU tensors
                if isinstance(inp, np.ndarray):
                    prepared.append(
                        torch.from_numpy(inp).to(device="cuda", dtype=torch.float32)
                    )
                elif isinstance(inp, torch.Tensor):
                    if not inp.is_cuda:
                        prepared.append(inp.to(device="cuda", dtype=torch.float32))
                    else:
                        prepared.append(inp)
                else:
                    # Fallback
                    prepared.append(
                        torch.tensor(inp, device="cuda", dtype=torch.float32)
                    )
            else:
                # Fallback for other backends (e.g., CPU_TORCH) or unexpected types
                prepared.append(inp)

        return prepared

    @staticmethod
    def generate_from_shapes(
        nodes: List[TensorNode], axes_map: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Generates dummy data for a list of input nodes, using specific shapes
        provided in axes_map (name -> shape_tuple).
        """
        inputs = {}
        for node in nodes:
            if node.name in axes_map:
                # Use stored shape
                shape = tuple(axes_map[node.name])
            else:
                # Fallback to node definition shape or default
                shape = node.shape if node.shape is not None else (1,)

            inputs[node.name] = DataGenerator.random_tensor(shape, node.dtype)
        return inputs

    @staticmethod
    def _random_shape(min_rank=1, max_rank=4, min_dim=1, max_dim=64) -> Tuple[int, ...]:
        rank = random.randint(min_rank, max_rank)
        return tuple(random.randint(min_dim, max_dim) for _ in range(rank))

    @staticmethod
    def _generate_internal(
        op_type: str,
        signatures: Tuple[TensorSignature, ...],
        backend: Optional[Backend] = None,
    ) -> Tuple[List[Tuple[np.ndarray, Optional[Backend]]], Dict[str, Any]]:
        """
        Internal method to generate inputs and attributes based on op_type.
        Returns a list of (numpy_array, target_backend) tuples.
        """
        attrs = {}
        inputs_with_backends = []

        # --- Special Handling for Shape-dependent Ops ---

        if op_type == OpType.DOT:
            # Matrix Multiplication: (M, K) @ (K, N)
            if len(signatures) >= 2:
                s0 = signatures[0].shape or (32, 32)
                s1 = signatures[1].shape or (32, 32)

                m = s0[0] if len(s0) > 0 and s0[0] is not None else 32
                sig0_k = s0[1] if len(s0) > 1 and s0[1] is not None else 32
                sig1_k = s1[0] if len(s1) > 0 and s1[0] is not None else 32
                n = s1[1] if len(s1) > 1 and s1[1] is not None else 32

                k = max(int(sig0_k), int(sig1_k))
            else:
                m, k, n = 32, 32, 32

            if len(signatures) >= 2:
                inputs_with_backends.append(
                    (
                        DataGenerator.random_tensor((m, k), signatures[0].dtype),
                        signatures[0].backend or backend,
                    )
                )
                inputs_with_backends.append(
                    (
                        DataGenerator.random_tensor((k, n), signatures[1].dtype),
                        signatures[1].backend or backend,
                    )
                )
            return inputs_with_backends, attrs

        elif op_type == OpType.RESHAPE:
            if len(signatures) >= 1 and signatures[0].shape:
                in_shape = tuple(
                    d if d is not None else 32 for d in signatures[0].shape
                )
                dtype = signatures[0].dtype
            else:
                in_shape = DataGenerator._random_shape()
                dtype = DType.FP32

            data = DataGenerator.random_tensor(in_shape, dtype)
            total_elements = int(np.prod([d for d in in_shape if d is not None]))

            # Simple factorization strategy for valid reshape
            if random.random() < 0.5:
                # Flatten
                target_shape = np.array([total_elements], dtype=np.int32)
            else:
                # Attempt to split into 2 dims
                dim1 = 1
                # Find a divisor roughly near sqrt(total)
                for i in range(int(total_elements**0.5), 0, -1):
                    if total_elements % i == 0:
                        dim1 = i
                        break
                dim2 = total_elements // dim1
                target_shape = np.array([dim1, dim2], dtype=np.int32)

            # Data input follows signature, Shape tensor is CPU
            if signatures:
                inputs_with_backends.append(
                    (data, signatures[0].backend if len(signatures) > 0 else backend)
                )
            inputs_with_backends.append((target_shape, Backend.CPU_NUMPY))
            return inputs_with_backends, attrs

        elif op_type == OpType.PERMUTE:
            # Data: Random Shape, Perm: Random Shuffle of axes
            if len(signatures) >= 1:
                shape = (
                    signatures[0].shape
                    if signatures[0].shape
                    else DataGenerator._random_shape(min_rank=2)
                )
                dtype = signatures[0].dtype
            else:
                shape = DataGenerator._random_shape(min_rank=2)
                dtype = DType.FP32

            data = DataGenerator.random_tensor(shape, dtype)

            axes = list(range(len(shape)))
            random.shuffle(axes)
            perm = np.array(axes, dtype=np.int32)

            if signatures:
                inputs_with_backends.append(
                    (data, signatures[0].backend if len(signatures) > 0 else backend)
                )
            inputs_with_backends.append((perm, Backend.CPU_NUMPY))
            return inputs_with_backends, attrs

        elif op_type == OpType.CONCAT:
            # Data A/B: Must have matching rank and matching dims except on axis
            if len(signatures) >= 2:
                shape_a = list(
                    signatures[0].shape
                    if signatures[0].shape
                    else DataGenerator._random_shape(min_rank=2)
                )
                # Force B to match A's rank
                rank = len(shape_a)
                shape_b = list(
                    signatures[1].shape
                    if (signatures[1].shape and len(signatures[1].shape) == rank)
                    else DataGenerator._random_shape(min_rank=rank, max_rank=rank)
                )
                dtype_a, dtype_b = signatures[0].dtype, signatures[1].dtype
            else:
                shape_a = list(DataGenerator._random_shape(min_rank=2))
                shape_b = shape_a.copy()
                dtype_a, dtype_b = DType.FP32, DType.FP32

            # Clean up Nones in shapes
            shape_a = [d if d is not None else 32 for d in shape_a]
            shape_b = [d if d is not None else 32 for d in shape_b]

            axis_idx = random.randint(0, len(shape_a) - 1)
            # Ensure dimensions match on all axes except concat axis
            for i in range(len(shape_a)):
                if i != axis_idx:
                    shape_b[i] = shape_a[i]

            a = DataGenerator.random_tensor(tuple(shape_a), dtype_a)
            b = DataGenerator.random_tensor(tuple(shape_b), dtype_b)
            axis = np.array([axis_idx], dtype=np.int32)

            inputs_with_backends.append(
                (a, signatures[0].backend if len(signatures) > 0 else backend)
            )
            inputs_with_backends.append(
                (b, signatures[1].backend if len(signatures) > 1 else backend)
            )
            inputs_with_backends.append((axis, Backend.CPU_NUMPY))
            return inputs_with_backends, attrs

        elif op_type == OpType.SLICE:
            if len(signatures) >= 1:
                shape = (
                    signatures[0].shape
                    if signatures[0].shape
                    else DataGenerator._random_shape()
                )
                dtype = signatures[0].dtype
            else:
                shape = DataGenerator._random_shape()
                dtype = DType.FP32

            data = DataGenerator.random_tensor(shape, dtype)

            starts, ends, steps = [], [], []
            for dim in shape:
                dim_int = int(dim) if dim is not None else 32
                start = random.randint(0, dim_int - 1)
                end = random.randint(start + 1, dim_int)
                step = random.randint(1, 2)
                starts.append(start)
                ends.append(end)
                steps.append(step)

            if len(signatures) == 4:
                inputs_with_backends.append((data, signatures[0].backend or backend))
                inputs_with_backends.append(
                    (np.array(starts, dtype=np.int32), Backend.CPU_NUMPY)
                )
                inputs_with_backends.append(
                    (np.array(ends, dtype=np.int32), Backend.CPU_NUMPY)
                )
                inputs_with_backends.append(
                    (np.array(steps, dtype=np.int32), Backend.CPU_NUMPY)
                )
            else:
                if len(signatures) > 0:
                    inputs_with_backends.append(
                        (data, signatures[0].backend or backend)
                    )
                attrs = {"starts": starts, "ends": ends, "steps": steps}
            return inputs_with_backends, attrs

        elif op_type == OpType.ARANGE:
            start = random.randint(0, 10)
            stop = start + random.randint(10, 100)
            step = random.choice([1, 2, 5])

            inputs_with_backends.append(
                (np.array([start], dtype=np.int32), Backend.CPU_NUMPY)
            )
            inputs_with_backends.append(
                (np.array([stop], dtype=np.int32), Backend.CPU_NUMPY)
            )
            inputs_with_backends.append(
                (np.array([step], dtype=np.int32), Backend.CPU_NUMPY)
            )
            return inputs_with_backends, attrs

        elif op_type == OpType.TRIU:
            if len(signatures) >= 1:
                shape = (
                    signatures[0].shape
                    if signatures[0].shape
                    else DataGenerator._random_shape(min_rank=2)
                )
                dtype = signatures[0].dtype
            else:
                shape = DataGenerator._random_shape(min_rank=2)
                dtype = DType.FP32

            data = DataGenerator.random_tensor(shape, dtype)
            k = np.array([random.randint(-2, 2)], dtype=np.int32)

            inputs_with_backends.append(
                (data, signatures[0].backend if len(signatures) > 0 else backend)
            )
            inputs_with_backends.append((k, Backend.CPU_NUMPY))
            return inputs_with_backends, attrs

        elif op_type == OpType.FILL:
            if len(signatures) >= 1:
                target_dtype = signatures[0].dtype
            else:
                target_dtype = DType.FP32

            # Value
            if target_dtype == DType.INT32:
                val = np.array([random.randint(0, 10)], dtype=np.int32)
            else:
                val = np.array([random.random()], dtype=np.float32)

            # Shape
            shape_tuple = DataGenerator._random_shape()
            shape_tensor = np.array(shape_tuple, dtype=np.int32)

            inputs_with_backends.append((val, Backend.CPU_NUMPY))
            inputs_with_backends.append((shape_tensor, Backend.CPU_NUMPY))
            attrs = {"target_shape": shape_tuple}
            return inputs_with_backends, attrs

        elif op_type == OpType.GATHER:
            vocab_size = random.randint(50, 200)
            dim = random.randint(32, 128)
            data = DataGenerator.random_tensor((vocab_size, dim), DType.FP32)

            indices_shape = DataGenerator._random_shape(max_rank=2)
            indices = np.random.randint(
                0, vocab_size, size=indices_shape, dtype=np.int32
            )

            inputs_with_backends.append((data, Backend.CPU_NUMPY))
            inputs_with_backends.append((indices, Backend.CPU_NUMPY))
            return inputs_with_backends, attrs

        elif op_type in (OpType.MAX, OpType.SUM):
            if len(signatures) >= 1:
                shape = (
                    signatures[0].shape
                    if signatures[0].shape
                    else DataGenerator._random_shape(min_rank=2)
                )
                dtype = signatures[0].dtype
            else:
                shape = DataGenerator._random_shape(min_rank=2)
                dtype = DType.FP32

            data = DataGenerator.random_tensor(shape, dtype)
            inputs_with_backends.append(
                (data, signatures[0].backend if len(signatures) > 0 else backend)
            )

            axis_idx = random.randint(0, len(shape) - 1)

            if len(signatures) > 1:
                inputs_with_backends.append(
                    (np.array([axis_idx], dtype=np.int32), Backend.CPU_NUMPY)
                )
            else:
                attrs["axis"] = axis_idx
                attrs["keepdims"] = True
            return inputs_with_backends, attrs

        elif op_type == OpType.REPEAT:
            if len(signatures) >= 1:
                shape = (
                    signatures[0].shape
                    if signatures[0].shape
                    else DataGenerator._random_shape()
                )
                dtype = signatures[0].dtype
            else:
                shape = DataGenerator._random_shape()
                dtype = DType.FP32

            data = DataGenerator.random_tensor(shape, dtype)
            inputs_with_backends.append(
                (data, signatures[0].backend if len(signatures) > 0 else backend)
            )

            repeats = random.randint(2, 5)

            if len(signatures) > 1:
                inputs_with_backends.append(
                    (np.array([repeats], dtype=np.int32), Backend.CPU_NUMPY)
                )
            else:
                attrs["repeats"] = repeats
                # Pick valid axis
                attrs["axis"] = random.randint(0, len(shape) - 1)
            return inputs_with_backends, attrs

        elif op_type == OpType.COPY_TO:
            if len(signatures) >= 1:
                shape = (
                    signatures[0].shape
                    if signatures[0].shape
                    else DataGenerator._random_shape()
                )
                dtype = signatures[0].dtype
            else:
                shape = DataGenerator._random_shape()
                dtype = DType.FP32

            data = DataGenerator.random_tensor(shape, dtype)
            # Input follows signature backend
            inputs_with_backends.append(
                (data, signatures[0].backend if len(signatures) > 0 else backend)
            )

            if backend:
                attrs["target_backend"] = backend.value
            else:
                attrs["target_backend"] = "cpu_numpy"
            return inputs_with_backends, attrs

        elif op_type == OpType.CAST:
            if len(signatures) >= 1:
                shape = (
                    signatures[0].shape
                    if signatures[0].shape
                    else DataGenerator._random_shape()
                )
                dtype = signatures[0].dtype
            else:
                shape = DataGenerator._random_shape()
                dtype = DType.FP32

            data = DataGenerator.random_tensor(shape, dtype)
            inputs_with_backends.append(
                (data, signatures[0].backend if len(signatures) > 0 else backend)
            )
            attrs = {"to": DType.FP32}  # Default target
            return inputs_with_backends, attrs

        # --- Generic Element-wise / Broadcast / Fused ---

        # Generate a random base shape for element-wise operations
        base_shape = DataGenerator._random_shape()

        # Fused Op Handling with random shapes
        if op_type == "RMSNorm":
            # x, weight, eps
            dim = base_shape[-1]
            inputs_with_backends.append(
                (
                    DataGenerator.random_tensor(base_shape, DType.FP32),
                    signatures[0].backend or backend,
                )
            )
            inputs_with_backends.append(
                (
                    DataGenerator.random_tensor((dim,), DType.FP32),
                    signatures[1].backend or backend,
                )
            )
            inputs_with_backends.append(
                (np.array([1e-5], dtype=np.float32), signatures[2].backend or backend)
            )
            return inputs_with_backends, attrs

        if op_type == "RoPE":
            inputs_with_backends.append(
                (
                    DataGenerator.random_tensor(base_shape, DType.FP32),
                    signatures[0].backend or backend,
                )
            )
            inputs_with_backends.append(
                (
                    DataGenerator.random_tensor(base_shape, DType.FP32),
                    signatures[1].backend or backend,
                )
            )
            inputs_with_backends.append(
                (
                    DataGenerator.random_tensor(base_shape, DType.FP32),
                    signatures[2].backend or backend,
                )
            )
            return inputs_with_backends, attrs

        if op_type == "GELU" or op_type == "Softmax":
            inputs_with_backends.append(
                (
                    DataGenerator.random_tensor(base_shape, DType.FP32),
                    signatures[0].backend or backend,
                )
            )
            return inputs_with_backends, attrs

        if op_type == "FusedMulAdd":
            inputs_with_backends.append(
                (
                    DataGenerator.random_tensor(base_shape, DType.FP32),
                    signatures[0].backend or backend,
                )
            )
            inputs_with_backends.append(
                (
                    DataGenerator.random_tensor(base_shape, DType.FP32),
                    signatures[1].backend or backend,
                )
            )
            inputs_with_backends.append(
                (
                    DataGenerator.random_tensor(base_shape, DType.FP32),
                    signatures[2].backend or backend,
                )
            )
            return inputs_with_backends, attrs

        # Fallback for generic atomic element-wise based on signature
        for sig in signatures:
            sig_shape = sig.shape

            # 1. Scalar signature
            if sig.is_scalar() or (sig_shape and sig_shape == (1,)):
                inputs_with_backends.append(
                    (
                        DataGenerator.random_tensor((1,), sig.dtype),
                        sig.backend or backend,
                    )
                )
            # 2. Vector-specific signature
            elif sig_shape and sig_shape == (None,):
                # Use flattened base shape dimension or random 1D
                inputs_with_backends.append(
                    (
                        DataGenerator.random_tensor((base_shape[0],), sig.dtype),
                        sig.backend or backend,
                    )
                )
            # 3. Explicit shape in signature (rare for atomic ops, but possible)
            elif sig_shape and None not in sig_shape:
                inputs_with_backends.append(
                    (
                        DataGenerator.random_tensor(sig_shape, sig.dtype),
                        sig.backend or backend,
                    )
                )
            # 4. Standard Broadcastable Input
            else:
                inputs_with_backends.append(
                    (
                        DataGenerator.random_tensor(base_shape, sig.dtype),
                        sig.backend or backend,
                    )
                )

        return inputs_with_backends, attrs

    @staticmethod
    def generate(
        op_type: str,
        signatures: Tuple[TensorSignature, ...],
        backend: Optional[Backend] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Returns (inputs_list, attributes_dict)
        For GPU_TORCH backend, inputs will be torch tensors on CUDA.
        For CPU_NUMPY backend, inputs will be numpy arrays.
        Inputs are prepared according to the backend specified in their respective signatures.
        """
        # 1. Generate raw inputs and attributes with backend metadata
        inputs_with_backends, attrs = DataGenerator._generate_internal(
            op_type, signatures, backend
        )

        # 2. Prepare inputs for the specific backends
        return (
            DataGenerator._prepare_inputs_for_backend(inputs_with_backends, backend),
            attrs,
        )
