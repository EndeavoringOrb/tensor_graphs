from typing import Dict, List, Callable, Optional, Tuple, Any
from ..ir.dtypes import DType, TensorSignature, Backend

# A Kernel is identified by its OpType, Backend, and the list of input signatures it accepts.
# We store: (Backend, Signatures, ImplementationFunction)
KernelEntry = Tuple[Backend, Tuple[TensorSignature, ...], Callable]


class KernelRegistry:
    # OpType -> Backend -> List of Candidate Kernels
    _kernels: Dict[str, Dict[Backend, List[KernelEntry]]] = {}
    _converters: Dict[Tuple[DType, DType], Callable] = {}

    @classmethod
    def register(
        cls,
        op_type: str,
        input_sigs: List[TensorSignature],
        backend: Backend = Backend.CPU_NUMPY,
    ):
        """Decorator to register a hardware implementation."""

        def decorator(func):
            if op_type not in cls._kernels:
                cls._kernels[op_type] = {}

            if backend not in cls._kernels[op_type]:
                cls._kernels[op_type][backend] = []

            cls._kernels[op_type][backend].append((backend, tuple(input_sigs), func))
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
    def get_all_kernels(cls) -> Dict[str, Dict[Backend, List[KernelEntry]]]:
        """Returns the entire kernel registry."""
        return cls._kernels

    @classmethod
    def select_best_kernel(
        cls,
        op_type: str,
        concrete_inputs: List[TensorSignature],
        backend: Backend = Backend.CPU_NUMPY,
    ) -> Optional[Callable]:
        """
        Finds the best matching kernel for the given concrete input signatures and backend.
        Uses a scoring system:
        - Exact dimension match: +10 points
        - Wildcard (None) match: +1 point
        - Mismatch: -1 (Disqualified)
        """
        candidates = cls._kernels.get(op_type, {}).get(backend, [])
        best_score = -1
        best_kernel = None

        for cand_backend, pattern_sigs, kernel_func in candidates:
            # Sanity check
            if cand_backend != backend:
                continue

            score = cls._score_candidate(pattern_sigs, concrete_inputs)
            if score > best_score:
                best_score = score
                best_kernel = kernel_func

        return best_kernel

    @staticmethod
    def _score_candidate(
        patterns: Tuple[TensorSignature, ...], concrete: List[TensorSignature]
    ) -> int:
        if len(patterns) != len(concrete):
            return -1

        total_score = 0

        for pat, con in zip(patterns, concrete):
            # 1. Check DType (Strict Match Required)
            if pat.dtype != con.dtype:
                return -1

            # NEW: Check for Any-Rank Wildcard
            if pat.shape is None:
                total_score += 1  # Low score, but matches any rank
                continue

            # Check if concrete shape is None (cannot match specific pattern)
            if con.shape is None:
                return -1

            # 2. Check Rank (Strict Match Required otherwise)
            if len(pat.shape) != len(con.shape):
                return -1

            # 3. Check Dimensions
            for p_dim, c_dim in zip(pat.shape, con.shape):
                if p_dim is not None:
                    if p_dim == c_dim:
                        total_score += 10  # Strong Match
                    else:
                        return -1  # Dimension Mismatch
                else:
                    total_score += 1  # Weak Match (Generic)

        return total_score

    @classmethod
    def find_conversion_path(
        cls, src_sig: TensorSignature, target_sig: TensorSignature
    ) -> Optional[str]:
        if (src_sig.dtype, target_sig.dtype) in cls._converters:
            return "Cast"
        return None
