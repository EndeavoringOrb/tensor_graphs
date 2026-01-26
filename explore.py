"""
explore.py

This script explores the performance landscape of registered operations.
It benchmarks:
1. The monolithic Kernel implementation (if available).
2. The Atomic Graph decomposition (if available).

Results are saved to benchmarks.db.
"""

import time
import json
import hashlib
import numpy as np
from typing import Dict, Any, List

from tensor_graphs.benchmark.db import BenchmarkDB
from tensor_graphs.benchmark.env import EnvironmentSniffer
from tensor_graphs.benchmark.profiler import Profiler
from tensor_graphs.benchmark.data_gen import DataGenerator
from tensor_graphs.backend.registry import KernelRegistry
from tensor_graphs.ops.registry import get_reference_factory
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType, Backend, TensorSignature
from tensor_graphs.ir.hashing import compute_structural_hash
from tensor_graphs.backend.executor import Executor
from tensor_graphs.compiler.planner import ExecutionRecipe


def get_structural_hash(node: TensorNode):
    return compute_structural_hash(node)


def get_axes_hash(inputs: Dict[str, np.ndarray]):
    axes_json = {k: v.shape for k, v in inputs.items()}
    return (
        hashlib.sha256(json.dumps(axes_json, sort_keys=True).encode()).hexdigest(),
        axes_json,
    )


def run_exploration():
    print("================================================================")
    print("  Tensor Graphs: Exploration Mode")
    print("================================================================")

    # 1. Initialize DB & Environment
    db = BenchmarkDB()
    profiler = Profiler(db)
    env_info = EnvironmentSniffer.sniff()
    print(f"Environment: {env_info['hardware_name']}")

    env_id = profiler.env_id

    # 2. Ops to Explore
    all_kernels = KernelRegistry.get_all_kernels()

    # We prioritize fused ops mentioned in README, but we'll iterate over all registered ops
    ops_to_explore = list(all_kernels.keys())
    # Sort to have a consistent order
    ops_to_explore.sort()

    # 3. Iterate
    for op_type in ops_to_explore:
        print(f"\n--- Exploring: {op_type} ---")

        # Get factory
        ref_factory = get_reference_factory(op_type)
        if not ref_factory:
            print(f"  [Skip] No reference factory found for {op_type}")
            continue

        # Get available kernels by backend
        kernels_by_backend = all_kernels.get(op_type, {})
        if not kernels_by_backend:
            print(f"  [Skip] No kernels registered for {op_type}")
            continue

        for backend, entries in kernels_by_backend.items():
            # Check if backend is available
            if backend == Backend.GPU_TORCH:
                import torch

                if not torch.cuda.is_available():
                    print(f"  [Skip] {backend.value} not available")
                    continue

            for entry in entries:
                # entry is (backend, sigs, target_dtype, func)
                _, sigs, target_dtype, kernel_func = entry

                print(
                    f"  --- Kernel: {backend.value} | Sigs: {sigs} -> {target_dtype} ---"
                )

                # Generate Input Data
                try:
                    inputs_data, attrs = DataGenerator.generate(
                        op_type, sigs, backend=backend
                    )
                except Exception as e:
                    print(f"    [Error] Data gen failed: {e}")
                    continue

                # Create Input Nodes
                input_nodes = []
                feed_dict = {}
                for i, val in enumerate(inputs_data):
                    name = f"in_{i}"
                    # Use appropriate backend
                    dt = DType.FP32
                    if val.dtype == np.int32:
                        dt = DType.INT32
                    elif val.dtype == np.float16:
                        dt = DType.FP16
                    elif val.dtype == bool:
                        dt = DType.BOOL

                    # Inputs start on CPU_NUMPY for simplicity of the reference graph
                    node = TensorNode(
                        "Input", val.shape, dt, [], name, backend=Backend.CPU_NUMPY
                    )
                    input_nodes.append(node)
                    feed_dict[name] = val

                # ---------------------------------------------------------
                # Path B (Prep): Atomic Decomposition (to get ground truth shape and structural hash)
                # ---------------------------------------------------------
                try:
                    root_atomic = ref_factory(input_nodes, attrs)
                    output_shape = root_atomic.shape
                    output_dtype = root_atomic.dtype
                except Exception as e:
                    print(f"    [Error] Atomic decomposition failed: {e}")
                    continue

                # ---------------------------------------------------------
                # Path A: Monolithic Kernel
                # ---------------------------------------------------------
                # Build a single node graph, inserting CopyTo if inputs are on different backend
                kernel_input_nodes = []
                assignments = {}

                for n in input_nodes:
                    if backend != n.backend:
                        copy_node = TensorNode(
                            "CopyTo",
                            n.shape,
                            n.dtype,
                            [n],
                            f"copy_{n.name}_to_{backend.value}",
                            attrs={"target_backend": backend.value},
                            backend=backend,
                        )
                        kernel_input_nodes.append(copy_node)
                        assignments[copy_node] = backend
                        assignments[n] = n.backend
                    else:
                        kernel_input_nodes.append(n)
                        assignments[n] = n.backend

                root_kernel = TensorNode(
                    op_type,
                    output_shape,
                    target_dtype or output_dtype,
                    kernel_input_nodes,
                    "kernel_node",
                    attrs=attrs,
                    backend=backend,
                )
                assignments[root_kernel] = backend

                recipe_kernel = ExecutionRecipe(root_kernel, assignments)

                print(f"    Benchmarking KERNEL implementation...")
                try:
                    latency_kernel = profiler.benchmark_recipe(recipe_kernel, feed_dict)
                    print(f"      Latency: {latency_kernel:.4f} ms")

                    # Save to DB
                    graph_hash = get_structural_hash(root_kernel)
                    axes_hash, axes_json = get_axes_hash(feed_dict)

                    # Canonical Graph
                    graph_id = db.add_canonical_graph(graph_hash, human_name=op_type)

                    # Implementation
                    impl_id = db.add_implementation(
                        graph_id,
                        "KERNEL",
                        f"{op_type}_Monolithic",
                        backend.value,
                        "hash_src",
                    )

                    # Workload
                    workload_id = db.add_workload(graph_id, axes_hash, axes_json)

                    # Trace
                    db.add_benchmark_trace(
                        impl_id, workload_id, env_id, "PASSED", latency_kernel
                    )

                except Exception as e:
                    print(f"      Failed: {e}")
                    latency_kernel = float("inf")
                    graph_id = None

                # ---------------------------------------------------------
                # Path B: Atomic Decomposition
                # ---------------------------------------------------------
                if graph_id is None:
                    continue

                print(f"    Benchmarking ATOMIC implementation...")
                try:
                    # Helper to get all nodes
                    def get_nodes(node, s):
                        s.add(node)
                        for p in node.parents:
                            get_nodes(p, s)

                    atomic_nodes = set()
                    get_nodes(root_atomic, atomic_nodes)

                    # Assign all to CPU_NUMPY for reference decomposition
                    assignments_atomic = {n: Backend.CPU_NUMPY for n in atomic_nodes}
                    recipe_atomic = ExecutionRecipe(root_atomic, assignments_atomic)

                    latency_atomic = profiler.benchmark_recipe(recipe_atomic, feed_dict)
                    print(f"      Latency: {latency_atomic:.4f} ms")

                    # Compare
                    if latency_kernel < float("inf"):
                        diff = latency_atomic - latency_kernel
                        winner = (
                            "KERNEL" if latency_kernel < latency_atomic else "ATOMIC"
                        )
                        print(f"      Winner: {winner} (Diff: {abs(diff):.4f} ms)")

                    # Save to DB
                    impl_id_atomic = db.add_implementation(
                        graph_id,
                        "GRAPH_RECIPE",
                        f"{op_type}_Decomposed",
                        "cpu_numpy",
                        "hash_src",
                    )

                    db.add_benchmark_trace(
                        impl_id_atomic, workload_id, env_id, "PASSED", latency_atomic
                    )

                except Exception as e:
                    print(f"      Failed: {e}")

    print("\nExploration Complete. Results saved to benchmarks.db")


if __name__ == "__main__":
    run_exploration()
