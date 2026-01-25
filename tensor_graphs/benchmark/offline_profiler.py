from .db import BenchmarkDB
from .profiler import Profiler, DummyDataGenerator
from ..compiler.planner import PathGenerator
from ..ir.hashing import compute_structural_hash
import json
import hashlib


def run_offline_profiling(db_path: str = "benchmarks.db"):
    db = BenchmarkDB(db_path)
    profiler = Profiler(db)

    # In a real scenario, we'd query for canonical graphs that need profiling
    # For now, let's just show how it would work if we had a list of roots to profile
    pass


def profile_root(root, db: BenchmarkDB, profiler: Profiler):
    structural_hash = compute_structural_hash(root)
    planner = PathGenerator(root)

    # Generate all possible execution recipes
    for recipe in planner.generate_all_strategies():
        # Generate dummy inputs
        input_nodes = profiler._get_inputs(recipe.root)
        inputs = {
            node.name: DummyDataGenerator.generate_for_node(node)
            for node in input_nodes
        }

        # We need a workload axes hash. For dummy data, we can just use the shapes.
        axes_json = {node.name: node.shape for node in input_nodes}
        axes_hash = hashlib.sha256(
            json.dumps(axes_json, sort_keys=True).encode()
        ).hexdigest()

        print(f"Profiling recipe for graph {structural_hash}...")
        latency = profiler.benchmark_recipe(recipe, inputs)
        print(f"  Latency: {latency:.4f} ms")

        # Save to DB
        graph_id = db.add_canonical_graph(structural_hash)
        impl_id = db.add_implementation(
            graph_id,
            "GRAPH_RECIPE",
            f"recipe_{hash(json.dumps(recipe.to_dict()))}",
            "MIXED",
            hashlib.md5(json.dumps(recipe.to_dict()).encode()).hexdigest(),
        )
        workload_id = db.add_workload(graph_id, axes_hash, axes_json)
        db.add_benchmark_trace(impl_id, workload_id, profiler.env_id, "PASSED", latency)
