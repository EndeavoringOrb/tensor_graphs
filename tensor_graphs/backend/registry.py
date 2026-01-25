from typing import Dict, List, Callable, Optional, Tuple, Any
from ..ir.dtypes import DType, TensorSignature, Backend
from ..ops.registry import register_reference_factory, get_reference_factory

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
        reference_factory: Optional[Callable] = None,
    ):
        """
        Decorator to register a hardware implementation.

        Args:
            op_type: The string identifier for the operation.
            input_sigs: List of input signatures this kernel supports.
            backend: The backend this kernel targets.
            reference_factory: A function (inputs, attrs) -> TensorNode that defines
                               the canonical graph for this Op.
                               MUST be provided if this OpType has not been registered yet.
        """

        def decorator(func):
            # 1. Register the Reference Factory if provided
            if reference_factory:
                register_reference_factory(op_type, reference_factory)

            # 2. Check if a reference exists (Enforce the rule)
            if not get_reference_factory(op_type):
                raise ValueError(
                    f"Cannot register kernel for '{op_type}': No reference graph factory provided "
                    f"and none exists in registry. You must provide 'reference_factory'."
                )

            # 3. Register the Kernel
            if op_type not in cls._kernels:
                cls._kernels[op_type] = {}

            if backend not in cls._kernels[op_type]:
                cls._kernels[op_type][backend] = []

            cls._kernels[op_type][backend].append((backend, tuple(input_sigs), func))
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
        """
        candidates = cls._kernels.get(op_type, {}).get(backend, [])
        best_score = -1
        best_kernel = None

        for cand_backend, pattern_sigs, kernel_func in candidates:
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
            if pat.dtype != con.dtype:
                return -1

            if pat.shape is None:
                total_score += 1
                continue

            if con.shape is None:
                return -1

            if len(pat.shape) != len(con.shape):
                return -1

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
        cls, src_sig: TensorSignature, target_sig: TensorSignature
    ) -> Optional[str]:
        if (src_sig.dtype, target_sig.dtype) in cls._converters:
            return "Cast"
        return None
