"""
explore.py

This script explores the performance landscape of registered operations.
It benchmarks:
1. (Priority) Flagged graphs in the profiling queue.
2. (Fallback) All registered operations (synthetic exploration), skipping already profiled ones.

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
from tensor_graphs.benchmark.profiler import Profiler, DummyDataGenerator
from tensor_graphs.benchmark.data_gen import DataGenerator
from tensor_graphs.backend.registry import KernelRegistry
from tensor_graphs.ops.registry import get_reference_factory
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType, Backend, TensorSignature
from tensor_graphs.ir.hashing import compute_structural_hash
from tensor_graphs.ir.graph import graph_from_json, get_inputs
from tensor_graphs.backend.executor import Executor
from tensor_graphs.compiler.planner import ExecutionRecipe, PathGenerator
from tensor_graphs.benchmark.offline_profiler import profile_root, get_axes_hash_info


def run_exploration():
    print("================================================================")
    print("  Tensor Graphs: Exploration Mode")
    print("================================================================")

    # 1. Initialize DB & Environment
    db = BenchmarkDB()
    profiler = Profiler(db)
    env_info = EnvironmentSniffer.sniff()
    print(f"Environment: {env_info['hardware_name']}")

    # ==========================================================================
    # PHASE 1: Process Profiling Queue (Flagged items)
    # ==========================================================================
    queue_items = db.get_queue_items()

    if queue_items:
        print(f"\n[Queue] Found {len(queue_items)} flagged graphs. Profiling now...")

        for item in queue_items:
            print(
                f"\n--- Processing Flagged Graph: {item['human_name']} ({item['structural_hash'][:8]}) ---"
            )
            try:
                # 1. Reconstruct Graph
                if not item["atomic_graph_json"]:
                    print(
                        f"  [Error] No JSON definition found for graph. Removing from queue."
                    )
                    db.remove_from_queue(item["queue_id"])
                    continue

                root = graph_from_json(item["atomic_graph_json"])

                # 2. Reconstruct Inputs (Shapes)
                axes_json = json.loads(item["axes_json"])
                inputs = {}

                # We need to map the axes_json (name -> shape) to the input nodes of the reconstructed graph.
                # Since graph reconstruction preserves names, this should work.
                input_nodes = get_inputs(root)
                valid_inputs = True

                for node in input_nodes:
                    if node.name in axes_json:
                        # Create dummy data with the specified shape
                        shape = axes_json[node.name]
                        # Fix: tuple conversion from list
                        node.shape = tuple(shape)
                        inputs[node.name] = DummyDataGenerator.generate_for_node(node)
                    else:
                        print(
                            f"  [Warning] Input '{node.name}' not found in workload stats. Generating generic dummy."
                        )
                        inputs[node.name] = DummyDataGenerator.generate_for_node(node)

                # 3. Profile all strategies (Kernel vs Atomic vs Decomposed)
                # We reuse the offline profiler logic which enumerates strategies
                profile_root(root, db, profiler, inputs)

                # 4. Cleanup
                db.remove_from_queue(item["queue_id"])
                print(f"  [Success] Removed from queue.")

            except Exception as e:
                print(f"  [Failed] Error processing queue item: {e}")
                # We optionally remove it so we don't crash loop, or keep it for retry.
                # For this prototype, we remove it.
                db.remove_from_queue(item["queue_id"])

        print("\n[Queue] Processing complete.")
        # If we processed the queue, we can exit or continue to general exploration.
        # User prompt implies "if there are no flags... profile all".
        # So if we had flags, we might stop? Or just continue.
        # "just want to prioritize the flagged things". We'll continue to general, but usually
        # offline profilers might want to exit after the queue is empty.
        # I'll continue for robustness.

    # ==========================================================================
    # PHASE 2: General Synthetic Exploration (Unprofiled)
    # ==========================================================================

    # Check if we should skip this phase (e.g. if we just did a bunch of work)
    # For now, let's run it but skip items that already have a "PASSED" trace for this env.

    print("\n--- General Exploration (Synthetic) ---")

    all_kernels = KernelRegistry.get_all_kernels()
    ops_to_explore = list(all_kernels.keys())
    ops_to_explore.sort()

    env_id = profiler.env_id

    for op_type in ops_to_explore:
        # Check if we should skip?
        # It's hard to know if we've profiled *all* variants, so we'll do a quick check
        # if a 'KERNEL' implementation exists for this op in the DB.

        # Heuristic: If we have a preferred implementation for a generic shape, maybe skip?
        # But exploration generates random shapes.
        # We will modify the loop to check `db.get_op_preference` for the generated shape.

        print(f"\n  Checking: {op_type}")

        # Get factory
        ref_factory = get_reference_factory(op_type)
        if not ref_factory:
            continue

        kernels_by_backend = all_kernels.get(op_type, {})
        for backend, entries in kernels_by_backend.items():
            if backend == Backend.GPU_TORCH and not torch.cuda.is_available():
                continue

            for entry in entries:
                _, sigs, target_dtype, kernel_func = entry

                # Generate Input Data to get a Concrete Shape
                try:
                    inputs_data, attrs = DataGenerator.generate(
                        op_type, sigs, backend=backend
                    )
                except Exception:
                    continue

                # Create Nodes
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

                    node = TensorNode("Input", val.shape, dt, [], name, backend=backend)
                    input_nodes.append(node)
                    feed_dict[name] = val

                # Build dummy root to check if we've profiled this shape
                try:
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

                    # CHECK DB: Have we profiled this op + shape + env?
                    shape_str = str(root_atomic.shape)
                    existing_pref = db.get_op_preference(op_type, shape_str, env_id)

                    if existing_pref:
                        print(
                            f"    [Skip] Already profiled {op_type} {shape_str} (Best: {existing_pref})"
                        )
                        continue

                except Exception:
                    # If we can't build the check graph, just proceed to run it
                    pass

                print(f"    Benchmarking {backend.value} | Sigs: {sigs}")

                # ... (Rest of the existing logic in explore.py goes here,
                # effectively running the monolith vs atomic benchmark)

                # Copying the core benchmark logic from original explore.py for continuity
                # ---------------------------------------------------------
                # Path A: Monolithic Kernel
                # ---------------------------------------------------------
                try:
                    root_kernel = TensorNode(
                        op_type,
                        root_atomic.shape,
                        target_dtype or root_atomic.dtype,
                        input_nodes,
                        "kernel_node",
                        attrs=attrs,
                        backend=backend,
                    )
                    assignments = {n: backend for n in input_nodes}
                    assignments[root_kernel] = backend
                    recipe_kernel = ExecutionRecipe(root_kernel, assignments)

                    # Pre-move data
                    strict_feed_dict = {}
                    if backend == Backend.GPU_TORCH:
                        for k, v in feed_dict.items():
                            if isinstance(v, np.ndarray):
                                strict_feed_dict[k] = torch.from_numpy(v).cuda()
                            else:
                                strict_feed_dict[k] = v
                    else:
                        strict_feed_dict = feed_dict

                    latency_kernel = profiler.benchmark_recipe(
                        recipe_kernel, strict_feed_dict
                    )

                    # Save Kernel Result
                    graph_hash = compute_structural_hash(root_kernel)
                    axes_hash, axes_json = get_axes_hash_info(feed_dict)

                    graph_id = db.add_canonical_graph(graph_hash, human_name=op_type)
                    impl_id = db.add_implementation(
                        graph_id,
                        "KERNEL",
                        f"{op_type}_Monolithic_{backend.value}",
                        backend.value,
                        "src_hash",
                        recipe_json=json.dumps(recipe_kernel.to_dict()),
                    )
                    workload_id = db.add_workload(graph_id, axes_hash, axes_json)
                    db.add_benchmark_trace(
                        impl_id, workload_id, env_id, "PASSED", latency_kernel
                    )

                except Exception as e:
                    print(f"      Kernel Failed: {e}")
                    latency_kernel = float("inf")
                    graph_id = None

                # ---------------------------------------------------------
                # Path B: Atomic Decomposition
                # ---------------------------------------------------------
                if graph_id:
                    try:
                        # Get atomic nodes
                        def get_nodes(node, s):
                            s.add(node)
                            for p in node.parents:
                                get_nodes(p, s)

                        atomic_nodes = set()
                        get_nodes(root_atomic, atomic_nodes)

                        assignments_atomic = {
                            n: Backend.CPU_NUMPY for n in atomic_nodes
                        }
                        recipe_atomic = ExecutionRecipe(root_atomic, assignments_atomic)

                        latency_atomic = profiler.benchmark_recipe(
                            recipe_atomic, feed_dict
                        )

                        # Save Atomic Result
                        impl_id_atomic = db.add_implementation(
                            graph_id,
                            "GRAPH_RECIPE",
                            f"{op_type}_Decomposed_CPU",
                            "cpu_numpy",
                            "src_hash",
                            recipe_json=json.dumps(recipe_atomic.to_dict()),
                        )
                        db.add_benchmark_trace(
                            impl_id_atomic,
                            workload_id,
                            env_id,
                            "PASSED",
                            latency_atomic,
                        )

                        winner = (
                            "KERNEL" if latency_kernel < latency_atomic else "ATOMIC"
                        )
                        print(
                            f"      Result: K={latency_kernel:.3f}ms vs A={latency_atomic:.3f}ms -> Winner: {winner}"
                        )

                    except Exception as e:
                        print(f"      Atomic Failed: {e}")

    print("\nExploration Complete.")


def get_axes_hash_info(inputs: Dict[str, np.ndarray]):
    # Helper duplicated from offline_profiler to make this script standalone
    axes_json = {k: v.shape for k, v in inputs.items()}
    axes_hash = hashlib.sha256(
        json.dumps(axes_json, sort_keys=True).encode()
    ).hexdigest()
    return axes_hash, axes_json


if __name__ == "__main__":
    run_exploration()
