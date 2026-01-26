import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
from ..ir.dtypes import DType, Backend, TensorSignature
from ..ops.atomic_types import OpType
from ..ir.node import TensorNode


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
            m = random.randint(32, 256)
            k = random.randint(32, 256)
            n = random.randint(32, 256)
            inputs.append(DataGenerator.random_tensor((m, k), DType.FP32))
            inputs.append(DataGenerator.random_tensor((k, n), DType.FP32))
            return inputs, attrs

        elif op_type == OpType.RESHAPE:
            # Data: Random Shape -> Reshape to 1D or 2D preserving elements
            in_shape = DataGenerator._random_shape()
            data = DataGenerator.random_tensor(in_shape, DType.FP32)
            total_elements = int(np.prod(in_shape))

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

            inputs = [data, target_shape]
            return inputs, attrs

        elif op_type == OpType.PERMUTE:
            # Data: Random Shape, Perm: Random Shuffle of axes
            shape = DataGenerator._random_shape(min_rank=2)
            data = DataGenerator.random_tensor(shape, DType.FP32)

            axes = list(range(len(shape)))
            random.shuffle(axes)
            perm = np.array(axes, dtype=np.int32)

            inputs = [data, perm]
            return inputs, attrs

        elif op_type == OpType.CONCAT:
            # Data A: Random, Data B: Same rank, matching dims except axis
            shape_a = list(DataGenerator._random_shape(min_rank=2))
            axis_idx = random.randint(0, len(shape_a) - 1)

            shape_b = shape_a.copy()
            shape_b[axis_idx] = random.randint(1, 64)  # Vary the concat axis size

            a = DataGenerator.random_tensor(tuple(shape_a), DType.FP32)
            b = DataGenerator.random_tensor(tuple(shape_b), DType.FP32)
            axis = np.array([axis_idx], dtype=np.int32)

            inputs = [a, b, axis]
            return inputs, attrs

        elif op_type == OpType.SLICE:
            shape = DataGenerator._random_shape()
            data = DataGenerator.random_tensor(shape, DType.FP32)

            starts, ends, steps = [], [], []
            for dim in shape:
                start = random.randint(0, dim - 1)
                end = random.randint(start + 1, dim)
                step = random.randint(1, 2)
                starts.append(start)
                ends.append(end)
                steps.append(step)

            if len(signatures) == 4:
                inputs = [
                    data,
                    np.array(starts, dtype=np.int32),
                    np.array(ends, dtype=np.int32),
                    np.array(steps, dtype=np.int32),
                ]
            else:
                inputs = [data]
                attrs = {"starts": starts, "ends": ends, "steps": steps}
            return inputs, attrs

        elif op_type == OpType.ARANGE:
            start = random.randint(0, 10)
            stop = start + random.randint(10, 100)
            step = random.choice([1, 2, 5])

            inputs = [
                np.array([start], dtype=np.int32),
                np.array([stop], dtype=np.int32),
                np.array([step], dtype=np.int32),
            ]
            return inputs, attrs

        elif op_type == OpType.TRIU:
            shape = DataGenerator._random_shape(min_rank=2)
            data = DataGenerator.random_tensor(shape, DType.FP32)
            k = np.array([random.randint(-2, 2)], dtype=np.int32)
            inputs = [data, k]
            return inputs, attrs

        elif op_type == OpType.FILL:
            target_dtype = signatures[0].dtype if signatures else DType.FP32

            # Value
            if target_dtype == DType.INT32:
                val = np.array([random.randint(0, 10)], dtype=np.int32)
            else:
                val = np.array([random.random()], dtype=np.float32)

            # Shape
            shape_tuple = DataGenerator._random_shape()
            shape_tensor = np.array(shape_tuple, dtype=np.int32)

            inputs = [val, shape_tensor]
            attrs = {"target_shape": shape_tuple}
            return inputs, attrs

        elif op_type == OpType.GATHER:
            vocab_size = random.randint(50, 200)
            dim = random.randint(32, 128)
            data = DataGenerator.random_tensor((vocab_size, dim), DType.FP32)

            indices_shape = DataGenerator._random_shape(max_rank=2)
            indices = np.random.randint(
                0, vocab_size, size=indices_shape, dtype=np.int32
            )

            inputs = [data, indices]
            return inputs, attrs

        elif op_type in (OpType.MAX, OpType.SUM):
            shape = DataGenerator._random_shape(min_rank=2)
            data = DataGenerator.random_tensor(shape, DType.FP32)
            inputs.append(data)

            axis_idx = random.randint(0, len(shape) - 1)

            if len(signatures) > 1:
                axis_val = np.array([axis_idx], dtype=np.int32)
                inputs.append(axis_val)
            else:
                attrs["axis"] = axis_idx
                attrs["keepdims"] = True
            return inputs, attrs

        elif op_type == OpType.REPEAT:
            shape = DataGenerator._random_shape()
            data = DataGenerator.random_tensor(shape, DType.FP32)
            inputs.append(data)

            repeats = random.randint(2, 5)

            if len(signatures) > 1:
                inputs.append(np.array([repeats], dtype=np.int32))
            else:
                attrs["repeats"] = repeats
                # Pick valid axis
                attrs["axis"] = random.randint(0, len(shape) - 1)
            return inputs, attrs

        elif op_type == OpType.COPY_TO:
            shape = DataGenerator._random_shape()
            data = DataGenerator.random_tensor(shape, signatures[0].dtype)
            inputs = [data]
            if backend:
                attrs["target_backend"] = backend.value
            else:
                attrs["target_backend"] = "cpu_numpy"
            return inputs, attrs

        elif op_type == OpType.CAST:
            shape = DataGenerator._random_shape()
            data = DataGenerator.random_tensor(shape, signatures[0].dtype)
            inputs = [data]
            attrs = {"to": DType.FP32}  # Default target
            return inputs, attrs

        # --- Generic Element-wise / Broadcast / Fused ---

        # Generate a random base shape for element-wise operations
        base_shape = DataGenerator._random_shape()

        # Fused Op Handling with random shapes
        if op_type == "RMSNorm":
            # x, weight, eps
            dim = base_shape[-1]
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            inputs.append(DataGenerator.random_tensor((dim,), DType.FP32))
            inputs.append(np.array([1e-5], dtype=np.float32))
            return inputs, attrs

        if op_type == "RoPE":
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            return inputs, attrs

        if op_type == "GELU" or op_type == "Softmax":
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            return inputs, attrs

        if op_type == "FusedMulAdd":
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            return inputs, attrs

        # Fallback for generic atomic element-wise based on signature
        for sig in signatures:
            sig_shape = sig.shape

            # 1. Scalar signature
            if sig.is_scalar() or (sig_shape and sig_shape == (1,)):
                inputs.append(DataGenerator.random_tensor((1,), sig.dtype))
            # 2. Vector-specific signature
            elif sig_shape and sig_shape == (None,):
                # Use flattened base shape dimension or random 1D
                inputs.append(DataGenerator.random_tensor((base_shape[0],), sig.dtype))
            # 3. Explicit shape in signature (rare for atomic ops, but possible)
            elif sig_shape and None not in sig_shape:
                inputs.append(DataGenerator.random_tensor(sig_shape, sig.dtype))
            # 4. Standard Broadcastable Input
            else:
                inputs.append(DataGenerator.random_tensor(base_shape, sig.dtype))

        return inputs, attrs
