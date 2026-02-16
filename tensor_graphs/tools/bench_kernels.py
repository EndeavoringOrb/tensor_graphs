import time
from tensor_graphs.backend.registry import KernelRegistry
from tensor_graphs.benchmark.db import BenchmarkDB
from tensor_graphs.benchmark.data_gen import DataGenerator
from tensor_graphs.ir.dtypes import (
    KernelUnavailableError,
)


def bench_all(db_path="benchmarks.db"):
    db = BenchmarkDB(db_path)
    registry = KernelRegistry.get_all_kernels()

    print("Starting Kernel Benchmarking...")

    for op_type, backends in registry.items():
        for backend, kernels in backends.items():
            print(f"Benchmarking {op_type} on {backend.value}...")

            for entry in kernels:
                _, sigs, _, inplace, func = entry

                # Let DataGenerator handle shape generation based on signatures
                inputs, attrs = DataGenerator.generate(op_type, tuple(sigs), backend)

                try:
                    # Warmup
                    for _ in range(5):
                        func(inputs, attrs)

                    # Timed
                    start = time.perf_counter()
                    for _ in range(20):
                        func(inputs, attrs)
                    end = time.perf_counter()
                except KernelUnavailableError as e:
                    print(f"  {op_type} on {backend.value} kernel unavailable: {e}")
                    continue

                avg_ms = ((end - start) / 20.0) * 1000

                # Consistency Fix: The Planner queries using the primary node shape.
                # For most ops, this is inputs[0].shape.
                # For 'Fill', the important shape is in attrs['target_shape'].
                if op_type == "Fill" and attrs and "target_shape" in attrs:
                    primary_shape = list(attrs["target_shape"])
                else:
                    primary_shape = list(inputs[0].shape)

                db.add_benchmark(
                    op_type, backend.value, "float32", primary_shape, attrs, avg_ms
                )
                print(f"  Shape {primary_shape}: {avg_ms:.4f} ms")

    print("Benchmarking Complete. DB populated.")


if __name__ == "__main__":
    bench_all()
