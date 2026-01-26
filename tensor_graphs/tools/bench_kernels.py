import time
import numpy as np
import itertools
from tensor_graphs.backend.registry import KernelRegistry
from tensor_graphs.benchmark.db import BenchmarkDB
from tensor_graphs.benchmark.data_gen import DataGenerator
from tensor_graphs.ir.dtypes import Backend, DType, TensorSignature


def bench_all(db_path="benchmarks.db"):
    db = BenchmarkDB(db_path)
    registry = KernelRegistry.get_all_kernels()

    print("Starting Kernel Benchmarking...")

    for op_type, backends in registry.items():
        for backend, kernels in backends.items():
            print(f"Benchmarking {op_type} on {backend.value}...")

            for entry in kernels:
                _, sigs, _, func = entry

                # Create a few test shapes
                shapes_to_test = [(128,), (1024,), (4096,), (128, 128), (512, 512)]

                for shape in shapes_to_test:
                    # Construct dummy signature to guide generator
                    # We override the shape in the signature for generation
                    test_sigs = []
                    for s in sigs:
                        new_shape = shape if s.shape != (1,) else (1,)
                        test_sigs.append(TensorSignature(s.dtype, new_shape, backend))

                    try:
                        inputs, attrs = DataGenerator.generate(
                            op_type, tuple(test_sigs), backend
                        )

                        # Warmup
                        for _ in range(5):
                            func(inputs, attrs)

                        # Timed
                        start = time.perf_counter()
                        for _ in range(20):
                            func(inputs, attrs)
                        end = time.perf_counter()

                        avg_ms = ((end - start) / 20.0) * 1000

                        db.add_benchmark(
                            op_type, backend.value, "float32", shape, attrs, avg_ms
                        )
                        # print(f"  Shape {shape}: {avg_ms:.4f} ms")

                    except Exception as e:
                        # Skip invalid shapes for certain ops
                        pass

    print("Benchmarking Complete. DB populated.")


if __name__ == "__main__":
    bench_all()
