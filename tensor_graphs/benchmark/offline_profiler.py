from .db import BenchmarkDB
from .profiler import Profiler, DummyDataGenerator
from ..compiler.planner import PathGenerator
from ..ir.hashing import compute_structural_hash
from ..ir.dtypes import Backend
import json
import hashlib
import numpy as np
from typing import Optional, Dict, Any


def run_offline_profiling(db_path: str = "benchmarks.db"):
    """
    Scans the registry and profiles all possible paths for all registered ops.
    """
    from ..backend.registry import KernelRegistry
    from ..ops.registry import get_reference_factory
    from ..ir.node import TensorNode
    from ..ir.dtypes import DType, Backend
    from .data_gen import DataGenerator

    db = BenchmarkDB(db_path)
    profiler = Profiler(db)

    all_kernels = KernelRegistry.get_all_kernels()

    for op_type, backends in all_kernels.items():
        print(f"--- Offline Profiling: {op_type} ---")
        ref_factory = get_reference_factory(op_type)
        if not ref_factory:
            print(f"  [Skip] No reference factory")
            continue

        for backend, entries in backends.items():
            for entry in entries:
                sigs = entry[1]
                target_dtype = entry[2]

                # 1. Generate standard inputs for this signature
                try:
                    inputs_data, attrs = DataGenerator.generate(
                        op_type, sigs, backend=backend
                    )
                except Exception as e:
                    continue

                # 2. Create input nodes
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

                    node = TensorNode(
                        "Input", val.shape, dt, [], name, backend=Backend.CPU_NUMPY
                    )
                    input_nodes.append(node)
                    feed_dict[name] = val

                # 3. Create a representative root (monolithic)
                try:
                    # We use ref_factory to get the output shape/dtype correctly
                    root_ref = ref_factory(input_nodes, attrs)
                    output_shape = root_ref.shape
                    output_dtype = root_ref.dtype
                except Exception:
                    continue

                root = TensorNode(
                    op_type,
                    output_shape,
                    target_dtype or output_dtype,
                    input_nodes,
                    "root",
                    attrs=attrs,
                    backend=backend,
                )

                # 4. Profile all paths
                profile_root(root, db, profiler, feed_dict)


def profile_root(
    root,
    db: BenchmarkDB,
    profiler: Profiler,
    feed_dict: Optional[Dict[str, Any]] = None,
):
    structural_hash = compute_structural_hash(root)
    planner = PathGenerator(root)

    # Canonical Graph
    graph_id = db.add_canonical_graph(structural_hash, human_name=root.op_type)

    # Workload info
    if feed_dict:
        axes_hash, axes_json = get_axes_hash_info(feed_dict)
        workload_id = db.add_workload(graph_id, axes_hash, axes_json)
    else:
        # Fallback if no feed_dict provided (though usually it should be)
        workload_id = None

    # Generate all possible execution recipes
    for recipe in planner.generate_all_strategies():
        # 1. Generate dummy inputs if not provided
        if feed_dict:
            inputs = feed_dict
        else:
            input_nodes = profiler._get_inputs(recipe.root)
            inputs = {
                node.name: DummyDataGenerator.generate_for_node(node)
                for node in input_nodes
            }
            axes_hash, axes_json = get_axes_hash_info(inputs)
            workload_id = db.add_workload(graph_id, axes_hash, axes_json)

        print(
            f"  Profiling recipe: {[f'{n.op_type}@{b.value}' for n, b in recipe.assignments.items()]}"
        )
        try:
            latency = profiler.benchmark_recipe(recipe, inputs)
            print(f"    Latency: {latency:.4f} ms")

            # Save Implementation
            # Use a more descriptive name for the implementation
            recipe_desc = "_".join(
                [
                    f"{n.op_type[:3]}={b.value[:3]}"
                    for n, b in recipe.assignments.items()
                    if n.op_type != "Input"
                ]
            )
            impl_name = f"{root.op_type}_{recipe_desc}_{hashlib.md5(json.dumps(recipe.to_dict()).encode()).hexdigest()[:8]}"

            # Identify main backend
            main_backend = recipe.assignments.get(recipe.root, Backend.CPU_NUMPY).value

            impl_id = db.add_implementation(
                graph_id,
                "GRAPH_RECIPE",
                impl_name,
                main_backend,
                hashlib.md5(json.dumps(recipe.to_dict()).encode()).hexdigest(),
            )

            db.add_benchmark_trace(
                impl_id, workload_id, profiler.env_id, "PASSED", latency
            )
        except Exception as e:
            print(f"    Failed: {e}")


def get_axes_hash_info(inputs: Dict[str, np.ndarray]):
    axes_json = {k: v.shape for k, v in inputs.items()}
    axes_hash = hashlib.sha256(
        json.dumps(axes_json, sort_keys=True).encode()
    ).hexdigest()
    return axes_hash, axes_json
