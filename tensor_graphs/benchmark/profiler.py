import time
import numpy as np
import json
from typing import Dict, Any, List, Set
from ..ir.node import TensorNode
from ..ir.dtypes import DType, Backend
from ..ops.atomic_types import OpType
from ..backend.executor import Executor
from .db import BenchmarkDB
from .env import EnvironmentSniffer


class DummyDataGenerator:
    @staticmethod
    def generate_for_node(node: TensorNode) -> np.ndarray:
        shape = node.shape
        if shape is None:
            shape = (1,)  # Default

        # Replace None in shape with 1 for dummy data
        safe_shape = tuple(d if d is not None else 1 for d in shape)

        if node.dtype == DType.FP32:
            return np.random.randn(*safe_shape).astype(np.float32)
        elif node.dtype == DType.INT32:
            return np.random.randint(0, 100, size=safe_shape).astype(np.int32)
        elif node.dtype == DType.BOOL:
            return np.random.choice([True, False], size=safe_shape)
        else:
            return np.zeros(safe_shape)


class Profiler:
    def __init__(self, db: BenchmarkDB):
        self.db = db
        self.env_id = self._get_current_env_id()

    def _get_current_env_id(self) -> str:
        env_info = EnvironmentSniffer.sniff()
        return self.db.add_environment(
            env_info["hardware_name"],
            env_info["memory_bytes"],
            env_info["platform_info"],
            env_info["libs_info"],
        )

    def benchmark_recipe(
        self, recipe, inputs: Dict[str, Any], warmups: int = 3, repeats: int = 10
    ) -> float:
        executor = Executor(recipe)

        # Warmup
        for _ in range(warmups):
            executor.run(inputs)

        # Benchmark
        start = time.perf_counter()
        for _ in range(repeats):
            executor.run(inputs)
        end = time.perf_counter()

        avg_latency_ms = (end - start) * 1000 / repeats
        return avg_latency_ms

    def verify_accuracy(
        self, recipe, reference_recipe, inputs: Dict[str, Any]
    ) -> float:
        executor = Executor(recipe)
        ref_executor = Executor(reference_recipe)

        out = executor.run(inputs)
        ref_out = ref_executor.run(inputs)

        # Max relative error
        abs_diff = np.abs(out - ref_out)
        rel_diff = abs_diff / (np.abs(ref_out) + 1e-8)
        return float(np.max(rel_diff))

    def _get_inputs(self, root: TensorNode) -> List[TensorNode]:
        inputs = []
        visited = set()

        def walk(node):
            if node in visited:
                return
            visited.add(node)
            if node.op_type == OpType.INPUT:
                inputs.append(node)
            for p in node.parents:
                walk(p)

        walk(root)
        return inputs

    def profile_and_save(
        self,
        recipe,
        structural_hash: str,
        workload_axes_hash: str,
        axes_json: Dict[str, Any],
    ):
        # 1. Generate inputs
        input_nodes = self._get_inputs(recipe.root)
        inputs = {
            node.name: DummyDataGenerator.generate_for_node(node)
            for node in input_nodes
        }

        # 2. Benchmark
        latency = self.benchmark_recipe(recipe, inputs)

        # 3. Save to DB
        # We need implementation_id and workload_id
        # For implementation_id, we need to know what canonical_graph this implementation belongs to
        graph_id = self.db.add_canonical_graph(structural_hash)

        # Here we assume recipe.to_dict() is enough to identify implementation
        impl_name = f"recipe_{time.time()}"  # In reality should be more descriptive
        impl_id = self.db.add_implementation(
            graph_id,
            "GRAPH_RECIPE",
            impl_name,
            "MIXED",  # Backend might be mixed
            str(hash(json.dumps(recipe.to_dict()))),  # Source hash as recipe hash
            requirements={},
        )

        workload_id = self.db.add_workload(graph_id, workload_axes_hash, axes_json)

        self.db.add_benchmark_trace(
            impl_id, workload_id, self.env_id, "PASSED", latency
        )
        return latency
