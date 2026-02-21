# tensor_graphs/compiler/cost_model.py
from ..benchmark.db import BenchmarkDB
from ..ir.dtypes import Backend, get_size_bytes


class CostModel:
    def __init__(self, db: BenchmarkDB):
        self.db = db
        # Base overheads (ms)
        self.transfer_overhead = {
            (Backend.CPU_NUMPY, Backend.GPU_TORCH): 0.05,
            (Backend.GPU_TORCH, Backend.CPU_NUMPY): 0.05,
        }
        self.default_kernel_cost = 0.01

    def estimate_kernel_cost(
        self, op_type, backend, dtype, shape, attrs, inplace=False
    ) -> float:
        est = self.db.estimate_latency(
            op_type, backend.value, dtype.value, shape, attrs, inplace=inplace
        )
        if est is not None:
            return est

        # Fallback: If inplace is requested but missing, try standard out-of-place cost
        if inplace:
            est = self.db.estimate_latency(
                op_type, backend.value, dtype.value, shape, attrs, inplace=False
            )
            if est is not None:
                return est

        # Heuristic fallback if DB is empty
        # Assume generic cost based on shape volume
        # Shapes are now plain int/None; None means dynamic (unknown at compile time)
        import math

        vol = math.prod(shape) if shape and all(shape) else 1

        # Penalize unknown operations slightly more to prefer optimized kernels if data exists
        return self.default_kernel_cost + (vol * 1e-7)

    def estimate_transfer_cost(
        self, src_backend: Backend, dst_backend: Backend, shape, dtype
    ) -> float:
        if src_backend == dst_backend:
            return 0.0

        base = self.transfer_overhead.get((src_backend, dst_backend), 0.1)
        try:
            bytes_transferred = get_size_bytes(shape, dtype)
        except ValueError:
            # Dynamic shape fallback
            bytes_transferred = 1024 * 1024  # 1 MB placeholder

        # Assume PCIe gen3/4 speeds roughly ~10GB/s
        transfer_time = bytes_transferred / (10 * 1024**3) * 1000
        return base + transfer_time
