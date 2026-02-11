import torch
from safetensors import safe_open
from typing import Any, Tuple, List

from .interface import WeightSource


class SafetensorsSource(WeightSource):
    def __init__(self, filepath: str):
        self.filepath = filepath
        # Use framework="pt" to support bfloat16 and robust memory mapping
        self._handle = safe_open(filepath, framework="pt", device="cpu")

    def keys(self) -> List[str]:
        return self._handle.keys()

    def get_tensor_metadata(self, name: str) -> Tuple[Tuple[int, ...], str]:
        info = self._handle.get_tensor_info(name)
        return tuple(info.shape), info.dtype

    def get_tensor(self, name: str) -> Any:
        # 1. Get PyTorch tensor (Memory mapped on CPU)
        tensor = self._handle.get_tensor(name)

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

        # 3. Convert to NumPy (Zero-copy if dtype was compatible, Copy if we casted)
        return tensor.numpy()

    def close(self):
        self._handle = None
