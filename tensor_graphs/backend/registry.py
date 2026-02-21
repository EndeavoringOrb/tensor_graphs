# tensor_graphs/backend/registry.py
from typing import Dict, List, Callable, Optional, Tuple
from ..ir.dtypes import DType, TensorSignature, Backend
from ..ops.registry import register_reference_factory, get_reference_factory

# Backend, Input Signatures, Target DType, Inplace Flag, Kernel Function
KernelEntry = Tuple[
    Backend, Tuple[TensorSignature, ...], Optional[DType], bool, Callable
]


class KernelRegistry:
    # OpType -> Backend -> List of Candidate Kernels
    _kernels: Dict[str, Dict[Backend, List[KernelEntry]]] = {}

    @classmethod
    def get_all_kernels(cls):
        """Returns the entire kernel registry."""
        return cls._kernels

    @classmethod
    def has_kernel(cls, op_type: str, backend: Backend) -> bool:
        """
        Checks if a kernel exists for the given OpType on the specific Backend.
        Used by the Planner to prune search space.
        """
        if op_type not in cls._kernels:
            return False
        if backend not in cls._kernels[op_type]:
            return False
        return len(cls._kernels[op_type][backend]) > 0

    @classmethod
    def register(
        cls,
        op_type: str,
        input_sigs: List[TensorSignature],
        backend: Backend = Backend.CPU_NUMPY,
        target_dtype: Optional[DType] = None,
        reference_factory: Optional[Callable] = None,
        inplace: bool = False,
    ):
        def decorator(func):
            # 1. Register Reference if provided
            if reference_factory:
                register_reference_factory(op_type, reference_factory)

            # 2. Check existence of reference (Atomic or Fused)
            if not get_reference_factory(op_type):
                pass  # Warning or Error logic here

            if op_type not in cls._kernels:
                cls._kernels[op_type] = {}
            if backend not in cls._kernels[op_type]:
                cls._kernels[op_type][backend] = []

            # 3. Resolve Target DType
            resolved_target_dtype = target_dtype

            if resolved_target_dtype is None and input_sigs:
                input_dtypes = {sig.dtype for sig in input_sigs}
                if len(input_dtypes) == 1:
                    resolved_target_dtype = list(input_dtypes)[0]
                elif len(input_dtypes) > 1:
                    dtypes_str = ", ".join([str(d.value) for d in input_dtypes])
                    raise ValueError(
                        f"Kernel registration error for '{op_type}' on backend '{backend.value}': "
                        f"Input signatures have mixed dtypes ({dtypes_str}). "
                        f"You must explicitly specify 'target_dtype' in the @register decorator."
                    )

            final_sigs = []
            for sig in input_sigs:
                if sig.backend is None:
                    final_sigs.append(TensorSignature(sig.dtype, sig.shape, backend))
                else:
                    final_sigs.append(sig)

            # Store the inplace flag in the entry
            cls._kernels[op_type][backend].append(
                (backend, tuple(final_sigs), resolved_target_dtype, inplace, func)
            )
            return func

        return decorator

    @classmethod
    def select_best_kernel(
        cls,
        op_type: str,
        concrete_inputs: List[TensorSignature],
        backend: Backend = Backend.CPU_NUMPY,
        target_dtype: Optional[DType] = None,
        allow_inplace: bool = True,
    ) -> Optional[Tuple[Callable, bool]]:

        candidates = cls._kernels.get(op_type, {}).get(backend, [])
        best_score = -1
        best_kernel = None
        best_inplace = False

        for (
            cand_backend,
            pattern_sigs,
            cand_target_dtype,
            cand_inplace,
            kernel_func,
        ) in candidates:
            # 1. Execution Backend Check
            if cand_backend != backend:
                continue

            # 2. Inplace Filter (For fallback re-querying)
            if cand_inplace and not allow_inplace:
                continue

            # 3. Output DType Check
            if cand_target_dtype is not None and target_dtype is not None:
                if cand_target_dtype != target_dtype:
                    continue

            # 4. Input Signature Scoring
            score = cls._score_candidate(pattern_sigs, concrete_inputs)
            if score < 0:
                continue

            # Prefer inplace if it's allowed and available
            if cand_inplace and allow_inplace:
                score += 1

            if score > best_score:
                best_score = score
                best_kernel = kernel_func
                best_inplace = cand_inplace

        if best_kernel:
            return best_kernel, best_inplace
        return None

    @staticmethod
    def _score_candidate(
        patterns: Tuple[TensorSignature, ...], concrete: List[TensorSignature]
    ) -> int:
        if len(patterns) != len(concrete):
            return -1

        total_score = 0
        for pat, con in zip(patterns, concrete):
            if pat.backend is not None:
                if pat.backend != con.backend:
                    return -1
                total_score += 10

            if pat.dtype != con.dtype:
                return -1

            if pat.shape is None:
                total_score += 1
            elif con.shape is None:
                return -1
            elif pat.is_scalar() and con.is_scalar():
                # Treat () and (1,) as equivalent scalars.
                # Use a high score (8) to prefer this over a generic wildcard (1),
                # but slightly lower than an exact dim match (10).
                total_score += 8
            elif len(pat.shape) != len(con.shape):
                return -1
            else:
                for p_dim, c_dim in zip(pat.shape, con.shape):
                    if p_dim is not None:
                        if p_dim == c_dim:
                            total_score += 10
                        else:
                            return -1
                    else:
                        total_score += 1
        return total_score

    @classmethod
    def find_conversion_path(
        cls,
        src_sig: TensorSignature,
        dest_sig: TensorSignature,
        backend: Backend = Backend.CPU_NUMPY,
    ) -> bool:
        return (
            cls.select_best_kernel(
                "Cast", [src_sig], backend, target_dtype=dest_sig.dtype
            )
            is not None
        )


from .kernels import *
