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
import torch
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
                if not torch.cuda.is_available():
                    print(f"  [Skip] {backend.value} not available")
                    continue

            for entry in entries:
                # entry is (backend, sigs, target_dtype, func)
                _, sigs, target_dtype, kernel_func = entry

                print(
                    f"  --- Kernel: {backend.value} | Sigs: {sigs} -> {target_dtype} ---"
                )

                # Generate Input Data (Usually returns NumPy arrays)
                try:
                    inputs_data, attrs = DataGenerator.generate(
                        op_type, sigs, backend=backend
                    )
                except Exception as e:
                    print(f"    [Error] Data gen failed: {e}")
                    continue

                # Create Input Nodes for Graph
                input_nodes = []
                feed_dict = {}

                for i, val in enumerate(inputs_data):
                    name = f"in_{i}"
                    dt = DType.FP32
                    if val.dtype == np.int32:
                        dt = DType.INT32
                    elif val.dtype == np.float16:
                        dt = DType.FP16
                    elif val.dtype == bool:
                        dt = DType.BOOL

                    # NOTE: For the Monolithic kernel test, we want the inputs to match the backend
                    # so we don't measure copy time.
                    node = TensorNode("Input", val.shape, dt, [], name, backend=backend)
                    input_nodes.append(node)
                    feed_dict[name] = val

                # ---------------------------------------------------------
                # Path B (Prep): Atomic Decomposition (to get ground truth shape and structural hash)
                # ---------------------------------------------------------
                try:
                    # We generate atomic nodes on CPU_NUMPY for structure analysis
                    # (Reference factories often assume CPU parents for logic, though structure is backend-agnostic)
                    dummy_cpu_inputs = [
                        TensorNode(
                            "Input",
                            n.shape,
                            n.dtype,
                            [],
                            n.name,
                            backend=Backend.CPU_NUMPY,
                        )
                        for n in input_nodes
                    ]
                    root_atomic = ref_factory(dummy_cpu_inputs, attrs)
                    output_shape = root_atomic.shape
                    output_dtype = root_atomic.dtype
                except Exception as e:
                    print(f"    [Error] Atomic decomposition failed: {e}")
                    continue

                # ---------------------------------------------------------
                # Path A: Monolithic Kernel
                # ---------------------------------------------------------
                # Build a single node graph.
                # UPDATED: We assume inputs are ALREADY on the target backend (strict mode).
                # We do not insert CopyTo nodes here. We move data in feed_dict below.

                root_kernel = TensorNode(
                    op_type,
                    output_shape,
                    target_dtype or output_dtype,
                    input_nodes,
                    "kernel_node",
                    attrs=attrs,
                    backend=backend,
                )

                assignments = {n: backend for n in input_nodes}
                assignments[root_kernel] = backend
                recipe_kernel = ExecutionRecipe(root_kernel, assignments)

                # --- PRE-MOVE DATA TO TARGET BACKEND ---
                # This ensures benchmarking measures pure kernel time, not PCI-e transfer.
                strict_feed_dict = {}
                if backend == Backend.GPU_TORCH:
                    for k, v in feed_dict.items():
                        # Convert numpy to torch, move to GPU
                        if isinstance(v, np.ndarray):
                            t = torch.from_numpy(v)
                            strict_feed_dict[k] = t.cuda()
                        else:
                            # Already torch/other?
                            strict_feed_dict[k] = v
                else:
                    strict_feed_dict = feed_dict

                print(
                    f"    Benchmarking KERNEL implementation (Strict: Data pre-moved)..."
                )
                try:
                    latency_kernel = profiler.benchmark_recipe(
                        recipe_kernel, strict_feed_dict
                    )
                    print(f"      Latency: {latency_kernel:.4f} ms")

                    # Save to DB
                    graph_hash = get_structural_hash(root_kernel)
                    axes_hash, axes_json = get_axes_hash(
                        feed_dict
                    )  # Use original numpy dict for axis stats

                    # Canonical Graph
                    graph_id = db.add_canonical_graph(graph_hash, human_name=op_type)

                    # Implementation
                    # Serialize recipe for FASTEST executor lookup
                    recipe_json = json.dumps(recipe_kernel.to_dict())

                    impl_id = db.add_implementation(
                        graph_id,
                        "KERNEL",
                        f"{op_type}_Monolithic_{backend.value}",
                        backend.value,
                        "hash_src_placeholder",
                        recipe_json=recipe_json,
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

                    # Atomic decomposition runs on CPU, so use standard numpy feed_dict
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
                    recipe_atomic_json = json.dumps(recipe_atomic.to_dict())
                    impl_id_atomic = db.add_implementation(
                        graph_id,
                        "GRAPH_RECIPE",
                        f"{op_type}_Decomposed_CPU",
                        "cpu_numpy",
                        "hash_src_placeholder",
                        recipe_json=recipe_atomic_json,
                    )

                    db.add_benchmark_trace(
                        impl_id_atomic, workload_id, env_id, "PASSED", latency_atomic
                    )

                except Exception as e:
                    print(f"      Failed: {e}")

    print("\nExploration Complete. Results saved to benchmarks.db")


if __name__ == "__main__":
    run_exploration()
