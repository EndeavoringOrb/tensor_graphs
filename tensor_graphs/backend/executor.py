import time
import json
import hashlib
import numpy as np
from typing import Dict, Any, Optional, Union, Set

from ..ir.node import TensorNode
from ..ir.dtypes import Backend, DType, TensorSignature
from ..ir.hashing import compute_structural_hash
from ..ops.atomic_types import OpType
from ..ops.registry import get_reference_factory
from ..backend.registry import KernelRegistry
from ..compiler.planner import ExecutionRecipe, PathGenerator
from ..benchmark.db import BenchmarkDB
from ..benchmark.env import EnvironmentSniffer

# Type checking import to avoid circular dependency with Profiler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..benchmark.profiler import Profiler


class Executor:
    """
    Unified execution engine for Tensor Graphs.
    """

    def __init__(
        self,
        recipe: Optional[ExecutionRecipe] = None,
        optimization_db: Optional[BenchmarkDB] = None,
        env_id: Optional[str] = None,
    ):
        self.recipe = recipe
        self.db = optimization_db
        self.env_id = env_id
        self.cache: Dict[TensorNode, Any] = {}

    def run(self, inputs: Dict[str, Any]) -> Any:
        """Executes the graph or recipe with the provided inputs."""
        self.cache = {}
        # If we don't have a root in the recipe (or no recipe), we might need the user to pass the root
        # explicitly, but standard usage implies the Executor is built with a recipe that has a root,
        # or we use the convenience function evaluate_graph which passes the root to _eval directly.
        if self.recipe:
            return self._eval(self.recipe.root, inputs)
        else:
            raise ValueError(
                "Executor.run() requires a recipe. Use evaluate_graph() for dynamic single-node execution."
            )

    def _find_root_from_inputs(self, inputs):
        pass

    def _eval(self, node: TensorNode, inputs: Dict[str, Any]) -> Any:
        # Check cache
        if node in self.cache:
            return self.cache[node]

        # 1. Base Cases: Input and Constant
        if node.op_type == OpType.INPUT:
            if node.name not in inputs:
                raise ValueError(f"Missing input data for node: {node.name}")
            val = inputs[node.name]
            self.cache[node] = val
            return val

        elif node.op_type == OpType.CONSTANT:
            val = node.attrs.get("value")
            self.cache[node] = val
            return val

        # 2. Evaluate Parents
        parent_vals = [self._eval(p, inputs) for p in node.parents]
        input_sigs = [p.signature for p in node.parents]

        # 3. Determine Execution Strategy
        if self.recipe:
            # --- RECIPE MODE (Strict) ---
            backend = self.recipe.assignments.get(node, Backend.CPU_NUMPY)

            # Select strict kernel
            kernel = KernelRegistry.select_best_kernel(
                node.op_type, input_sigs, backend, target_dtype=node.dtype
            )

            if not kernel:
                # Debug info
                sigs_str = ", ".join([str(s) for s in input_sigs])
                raise RuntimeError(
                    f"Execution Plan Failed: No kernel found for {node.op_type} "
                    f"on backend {backend.value}. Inputs: [{sigs_str}]."
                )

            val = kernel(parent_vals, node.attrs)

        else:
            # --- DYNAMIC MODE (Reference/Fallback) ---
            use_kernel = True

            # A. Check DB Preference if available
            if self.db and self.env_id:
                shape_str = str(node.shape)
                pref = self.db.get_op_preference(node.op_type, shape_str, self.env_id)
                if pref == "GRAPH_RECIPE":
                    use_kernel = False  # DB says decomposition is faster

            # B. Try to find a kernel
            kernel = None
            if use_kernel:
                kernel = KernelRegistry.select_best_kernel(
                    node.op_type, input_sigs, node.backend, target_dtype=node.dtype
                )

            # C. Execute or Decompose
            if kernel:
                val = kernel(parent_vals, node.attrs)
            else:
                # Fallback: Atomic Decomposition
                ref_factory = get_reference_factory(node.op_type)
                if ref_factory:
                    decomp_root = ref_factory(node.parents, node.attrs)
                    val = self._eval(decomp_root, inputs)
                else:
                    if OpType.is_atomic(node.op_type):
                        raise NotImplementedError(
                            f"No registered kernel for atomic op '{node.op_type}' on {node.backend}"
                        )
                    raise NotImplementedError(
                        f"No kernel or decomposition found for '{node.op_type}'"
                    )

        self.cache[node] = val
        return val


class SmartExecutor:
    """
    Intelligent executor that selects the best execution recipe based on a policy.
    Policies:
    - FASTEST: Look up best recipe in DB. If not found, use Heuristic.
    - EXPLORE: Flag graph for offline profiling and fallback to Heuristic immediately.
    - HEURISTIC: Use the default planner strategy.
    """

    def __init__(self, db_path: str = "benchmarks.db", policy: str = "FASTEST"):
        self.db = BenchmarkDB(db_path)
        # Import Profiler locally to avoid circular import
        from ..benchmark.profiler import Profiler

        self.profiler = Profiler(self.db)
        self.policy = policy

    def run(self, root: TensorNode, inputs: Dict[str, Any]) -> Any:
        recipe = self.select_recipe(root, inputs)
        executor = Executor(recipe=recipe)
        return executor.run(inputs)

    def select_recipe(
        self, root: TensorNode, inputs: Dict[str, Any]
    ) -> ExecutionRecipe:
        structural_hash = compute_structural_hash(root)

        # Calculate workload axes hash
        axes_json = {
            name: (val.shape if hasattr(val, "shape") else ())
            for name, val in inputs.items()
        }
        axes_hash = hashlib.sha256(
            json.dumps(axes_json, sort_keys=True, default=str).encode()
        ).hexdigest()

        # 1. FASTEST Policy
        if self.policy == "FASTEST":
            best_impl = self.db.get_best_implementation(
                structural_hash, axes_hash, self.profiler.env_id
            )
            if best_impl and best_impl["recipe_json"]:
                try:
                    recipe_data = json.loads(best_impl["recipe_json"])
                    # Reconstruct assignments from node name -> backend
                    # Note: This relies on node names being unique and stable within the session
                    # In a production system, we'd use topological indexing or persistent IDs.
                    node_map = {n.name: n for n in self._get_all_nodes(root)}
                    assignments = {}

                    # Load root name mapping
                    # (Simple reconstruction: assumes node names in DB match current graph instance)
                    # For dynamic graphs, this mapping requires structural matching,
                    # but for this prototype, we assume the graph passed in matches the structure stored.
                    for node_name, backend_val in recipe_data["assignments"].items():
                        if node_name in node_map:
                            assignments[node_map[node_name]] = Backend(backend_val)

                    # If mapping was successful for the root, return
                    if root in assignments:
                        return ExecutionRecipe(root, assignments)
                except Exception as e:
                    print(f"[SmartExecutor] Failed to deserialize recipe: {e}")
                    # Fallback to Heuristic

        # 2. EXPLORE Policy
        if self.policy == "EXPLORE":
            # UPDATED: Offline profiling flag logic
            self._flag_for_offline_profiling(root, structural_hash)
            # Fallback to Heuristic immediately so we don't block execution

        # 3. HEURISTIC / Default Fallback
        planner = PathGenerator(root)
        # Just pick the first strategy generated (usually monolithic/as-is if possible)
        return next(planner.generate_all_strategies())

    def _get_all_nodes(self, root: TensorNode, visited=None) -> Set[TensorNode]:
        if visited is None:
            visited = set()
        if root in visited:
            return visited
        visited.add(root)
        for p in root.parents:
            self._get_all_nodes(p, visited)
        return visited

    def _flag_for_offline_profiling(self, root: TensorNode, graph_hash: str):
        """
        Simulate adding this graph to a job queue for offline benchmarking.
        """
        print(
            f"[SmartExecutor] 🚩 Graph {graph_hash} ({root.op_type}) flagged for offline profiling."
        )


def evaluate_graph(
    root: TensorNode,
    inputs: Dict[str, np.ndarray],
    optimization_db: Optional[BenchmarkDB] = None,
    env_id: Optional[str] = None,
) -> Any:
    """
    Convenience function for evaluating a graph node dynamically.
    Used primarily by tests and reference implementations.
    """
    executor = Executor(recipe=None, optimization_db=optimization_db, env_id=env_id)
    return executor._eval(root, inputs)
