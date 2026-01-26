"""
File: tensor_graphs/benchmark/data_gen.py
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ..ir.dtypes import DType, Backend, TensorSignature
from ..ops.atomic_types import OpType


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
            m, k, n = 128, 128, 128
            inputs.append(DataGenerator.random_tensor((m, k), DType.FP32))
            inputs.append(DataGenerator.random_tensor((k, n), DType.FP32))
            return inputs, attrs

        elif op_type == OpType.RESHAPE:
            # Data: (2, 3, 4), Shape: [24] or [6, 4]
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
            data = DataGenerator.random_tensor((10, 10), DType.FP32)
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
            inputs = [
                np.array([0], dtype=np.int32),
                np.array([10], dtype=np.int32),
                np.array([1], dtype=np.int32),
            ]
            return inputs, attrs

        elif op_type == OpType.TRIU:
            data = DataGenerator.random_tensor((32, 32), DType.FP32)
            k = np.array([1], dtype=np.int32)
            inputs = [data, k]
            return inputs, attrs

        elif op_type == OpType.FILL:
            target_dtype = signatures[0].dtype if signatures else DType.FP32
            if target_dtype == DType.INT32:
                val = np.array([7], dtype=np.int32)
            else:
                val = np.array([3.14], dtype=np.float32)

            shape = np.array([32, 32], dtype=np.int32)
            inputs = [val, shape]
            attrs = {"target_shape": (32, 32)}
            return inputs, attrs

        elif op_type == OpType.GATHER:
            vocab_size = 100
            dim = 64
            data = DataGenerator.random_tensor((vocab_size, dim), DType.FP32)
            indices = np.random.randint(0, vocab_size, size=(32,), dtype=np.int32)
            inputs = [data, indices]
            return inputs, attrs

        elif op_type in (OpType.MAX, OpType.SUM):
            data = DataGenerator.random_tensor((64, 128), DType.FP32)
            inputs.append(data)
            if len(signatures) > 1:
                axis_val = np.array([0], dtype=np.int32)
                inputs.append(axis_val)
            return inputs, attrs

        elif op_type == OpType.REPEAT:
            data = DataGenerator.random_tensor((32, 32), DType.FP32)
            inputs.append(data)
            if len(signatures) > 1:
                inputs.append(np.array([2], dtype=np.int32))
            else:
                attrs["repeats"] = 2
            return inputs, attrs

        elif op_type == OpType.COPY_TO:
            data = DataGenerator.random_tensor((32, 32), signatures[0].dtype)
            inputs = [data]
            if backend:
                attrs["target_backend"] = backend.value
            else:
                attrs["target_backend"] = "cpu_numpy"
            return inputs, attrs

        elif op_type == OpType.CAST:
            data = DataGenerator.random_tensor((32, 32), signatures[0].dtype)
            inputs = [data]
            attrs = {"to": DType.FP32}
            return inputs, attrs

        # --- Generic Element-wise / Broadcast / Fused ---
        base_shape = (128, 128)

        # Fused Op Handling
        if op_type == "RMSNorm":
            # x, weight, eps
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            inputs.append(DataGenerator.random_tensor((base_shape[-1],), DType.FP32))
            inputs.append(np.array([1e-5], dtype=np.float32))
            return inputs, attrs

        if op_type == "RoPE":
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            return inputs, attrs

        if op_type == "GELU":
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            return inputs, attrs

        if op_type == "Softmax":
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            return inputs, attrs

        if op_type == "FusedMulAdd":
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            inputs.append(DataGenerator.random_tensor(base_shape, DType.FP32))
            return inputs, attrs

        # Fallback for generic atomic element-wise
        for sig in signatures:
            if sig.is_scalar() or (sig.shape and sig.shape == (1,)):
                inputs.append(DataGenerator.random_tensor((1,), sig.dtype))
            elif sig.shape and sig.shape == (None,):
                inputs.append(DataGenerator.random_tensor((base_shape[0],), sig.dtype))
            else:
                inputs.append(DataGenerator.random_tensor(base_shape, sig.dtype))

        return inputs, attrs
