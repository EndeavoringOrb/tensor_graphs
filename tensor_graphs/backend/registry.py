from typing import Dict, List, Callable, Optional, Tuple, Any
from ..ir.dtypes import DType, TensorSignature, Backend
from ..ops.registry import register_reference_factory, get_reference_factory

KernelEntry = Tuple[Backend, Tuple[TensorSignature, ...], Optional[DType], Callable]


class KernelRegistry:
    # OpType -> Backend -> List of Candidate Kernels
    _kernels: Dict[str, Dict[Backend, List[KernelEntry]]] = {}

    @classmethod
    def get_all_kernels(cls):
        """Returns the entire kernel registry."""
        return cls._kernels

    @classmethod
    def register(
        cls,
        op_type: str,
        input_sigs: List[TensorSignature],
        backend: Backend = Backend.CPU_NUMPY,
        target_dtype: Optional[DType] = None,
        reference_factory: Optional[Callable] = None,
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
            # Logic:
            # - If target_dtype is explicit, use it.
            # - If inputs are all same dtype, infer target_dtype = input_dtype.
            # - If inputs are mixed dtypes, target_dtype MUST be specified (raise Error).
            resolved_target_dtype = target_dtype

            if resolved_target_dtype is None and input_sigs:
                input_dtypes = {sig.dtype for sig in input_sigs}
                if len(input_dtypes) == 1:
                    # Constraint 1: All inputs same -> default to that dtype
                    resolved_target_dtype = list(input_dtypes)[0]
                elif len(input_dtypes) > 1:
                    # Constraint 2: Mixed inputs -> must specify target
                    dtypes_str = ", ".join([str(d.value) for d in input_dtypes])
                    raise ValueError(
                        f"Kernel registration error for '{op_type}' on backend '{backend.value}': "
                        f"Input signatures have mixed dtypes ({dtypes_str}). "
                        f"You must explicitly specify 'target_dtype' in the @register decorator."
                    )

            # If input signatures didn't specify backend, default to the execution backend
            # This maintains backward compatibility with existing generic kernels
            final_sigs = []
            for sig in input_sigs:
                if sig.backend is None:
                    # Assume input is on same backend as execution if not specified
                    final_sigs.append(TensorSignature(sig.dtype, sig.shape, backend))
                else:
                    final_sigs.append(sig)

            cls._kernels[op_type][backend].append(
                (backend, tuple(final_sigs), resolved_target_dtype, func)
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
    ) -> Optional[Callable]:

        candidates = cls._kernels.get(op_type, {}).get(backend, [])
        best_score = -1
        best_kernel = None

        for cand_backend, pattern_sigs, cand_target_dtype, kernel_func in candidates:
            # 1. Execution Backend Check
            if cand_backend != backend:
                continue

            # 2. Output DType Check
            if cand_target_dtype is not None and target_dtype is not None:
                if cand_target_dtype != target_dtype:
                    continue

            # 3. Input Signature Scoring
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

            # 1. Backend Match
            # If pattern specifies a backend, it MUST match.
            # If pattern is None (wildcard), it matches anything (score +0)
            if pat.backend is not None:
                if pat.backend != con.backend:
                    return -1
                total_score += 10  # Strong match for explicit backend

            # 2. DType Match
            if pat.dtype != con.dtype:
                return -1

            # 3. Shape Match
            if pat.shape is None:
                total_score += 1
            elif con.shape is None:
                return -1
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
        """
        Planner helper: Checks if a CAST kernel exists for src_dtype -> dest_dtype.
        """
        return (
            cls.select_best_kernel(
                "Cast", [src_sig], backend, target_dtype=dest_sig.dtype
            )
            is not None
        )


from .kernels import *
