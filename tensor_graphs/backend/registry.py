from typing import Dict, List, Callable, Optional, Tuple, Any
from tensor_graphs.ir.dtypes import DType, TensorSignature

# A Kernel is identified by: (OpName, InputSignatures)
KernelKey = Tuple[str, Tuple[TensorSignature, ...]]
KernelImpl = Callable[[List[Any]], Any]


class KernelRegistry:
    _kernels: Dict[KernelKey, KernelImpl] = {}
    _converters: Dict[Tuple[DType, DType], Callable] = {}

    @classmethod
    def register(cls, op_type: str, input_sigs: List[TensorSignature]):
        """Decorator to register a specific hardware implementation."""

        def decorator(func):
            key = (op_type, tuple(input_sigs))
            cls._kernels[key] = func
            return func

        return decorator

    @classmethod
    def register_cast(cls, src: DType, dst: DType):
        """Register a caster (conversion) function."""

        def decorator(func):
            cls._converters[(src, dst)] = func
            return func

        return decorator

    @classmethod
    def get_kernel(cls, op_type: str, input_sigs: List[TensorSignature]) -> Optional[KernelImpl]:
        """Exact match lookup."""
        return cls._kernels.get((op_type, tuple(input_sigs)))

    @classmethod
    def find_conversion_path(cls, src_sig: TensorSignature, target_sig: TensorSignature) -> Optional[str]:
        """Checks if a converter exists (returns op name 'Cast' if yes)."""
        # We simplify checks to just DType for now, assuming Cast handles shapes implicitly or strict shape match
        if (src_sig.dtype, target_sig.dtype) in cls._converters:
            return "Cast"
        return None
