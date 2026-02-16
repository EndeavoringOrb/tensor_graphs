# tensor_graphs/weights/safetensors_source.py
import os
import torch
from safetensors import safe_open
from typing import Any, Tuple, List, Dict

from .interface import WeightSource


class SafetensorsSource(WeightSource):
    def __init__(self, path: str):
        self.path = path
        self._handles: Dict[str, Any] = {}  # filepath -> handle
        self._key_to_handle: Dict[str, Any] = {}  # tensor_name -> handle

        # Determine if path is a file or directory
        if os.path.isdir(path):
            # Discover all safetensors files in the directory
            safetensors_files = sorted(
                [f for f in os.listdir(path) if f.endswith(".safetensors")]
            )
            if not safetensors_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")
            filepaths = [os.path.join(path, f) for f in safetensors_files]
        else:
            # Single file
            if not os.path.exists(path):
                raise FileNotFoundError(f"Safetensors file not found: {path}")
            filepaths = [path]

        # Open all files and build key mapping
        for fp in filepaths:
            handle = safe_open(fp, framework="pt", device="cpu")
            self._handles[fp] = handle

            # Map each tensor name to this handle
            for key in handle.keys():
                if key in self._key_to_handle:
                    raise ValueError(
                        f"Duplicate tensor name '{key}' found in multiple shards "
                        f"({path})"
                    )
                self._key_to_handle[key] = handle

    def keys(self) -> List[str]:
        return list(self._key_to_handle.keys())

    def get_tensor_metadata(self, name: str) -> Tuple[Tuple[int, ...], str]:
        if name not in self._key_to_handle:
            raise KeyError(f"Tensor '{name}' not found in safetensors source")
        handle = self._key_to_handle[name]
        return tuple(handle.get_tensor(name).shape), handle.get_tensor(name).dtype

    def get_tensor(self, name: str) -> Any:
        if name not in self._key_to_handle:
            raise KeyError(f"Tensor '{name}' not found in safetensors source")

        handle = self._key_to_handle[name]

        # 1. Get PyTorch tensor (Memory mapped on CPU)
        tensor = handle.get_tensor(name)

        # 2. Handle Dtype Conversion (Framework currently expects FP32/INT32/BOOL)
        if tensor.dtype == torch.bfloat16:
            # bfloat16 is not natively supported by NumPy, must cast.
            # This triggers a copy.
            tensor = tensor.to(torch.float32)
        elif tensor.dtype == torch.float16:
            # float16 IS supported by NumPy, but framework might expect FP32 for Math ops.
            # We assume FP16 is acceptable for now, but typically we cast to FP32 for broad kernel support.
            # Let's cast to FP32 for safety in this framework version.
            tensor = tensor.to(torch.float32)

        if tensor.dtype not in (torch.float32, torch.int32, torch.bool):
            raise TypeError(f"tensor.dtype {tensor.dtype} not supported")

        # 3. Convert to NumPy (Zero-copy if dtype was compatible, Copy if we casted)
        return tensor.numpy()

    def close(self):
        self._handles.clear()
        self._key_to_handle.clear()
