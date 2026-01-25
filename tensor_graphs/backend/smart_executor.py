from typing import Dict, Any, Optional
from ..ir.node import TensorNode
from ..ir.hashing import compute_structural_hash
from ..compiler.planner import PathGenerator
from ..benchmark.db import BenchmarkDB
from ..benchmark.profiler import Profiler, DummyDataGenerator
from .executor import Executor
import hashlib
import json


class SmartExecutor:
    def __init__(self, db_path: str = "benchmarks.db", policy: str = "FASTEST"):
        self.db = BenchmarkDB(db_path)
        self.profiler = Profiler(self.db)
        self.policy = policy  # FASTEST, EXPLORE, HEURISTIC

    def run(self, root: TensorNode, inputs: Dict[str, Any]) -> Any:
        recipe = self.select_recipe(root, inputs)
        executor = Executor(recipe)
        return executor.run(inputs)

    def select_recipe(self, root: TensorNode, inputs: Dict[str, Any]):
        structural_hash = compute_structural_hash(root)

        # Calculate workload axes hash
        axes_json = {name: val.shape for name, val in inputs.items()}
        axes_hash = hashlib.sha256(
            json.dumps(axes_json, sort_keys=True).encode()
        ).hexdigest()

        if self.policy == "FASTEST":
            best_impl = self.db.get_best_implementation(
                structural_hash, axes_hash, self.profiler.env_id
            )
            if best_impl:
                # In a real system, we'd deserialize the recipe from best_impl
                # For this prototype, we'll fallback if we can't easily recreate the recipe from DB
                pass

        if self.policy == "EXPLORE":
            # Profile all variants and pick the best one
            planner = PathGenerator(root)
            best_latency = float("inf")
            best_recipe = None

            for recipe in planner.generate_all_strategies():
                try:
                    latency = self.profiler.benchmark_recipe(recipe, inputs)
                    if latency < best_latency:
                        best_latency = latency
                        best_recipe = recipe
                except RuntimeError as e:
                    # Skip recipes that can't be executed (e.g. missing kernels)
                    print(f"Skipping recipe: {e}")
                    continue

            if best_recipe is None:
                raise RuntimeError("No executable recipe found for graph")
            return best_recipe

        # Default / HEURISTIC Fallback
        # Just pick the first strategy generated (usually monolithic if possible)
        planner = PathGenerator(root)
        return next(planner.generate_all_strategies())
