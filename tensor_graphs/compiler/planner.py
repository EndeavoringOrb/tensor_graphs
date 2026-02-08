import copy
from typing import Dict, Tuple, Optional, FrozenSet, Any
from ..ir.node import TensorNode
from ..ir.dtypes import Backend, TensorSignature
from ..ir.hashing import get_structural_hash
from ..benchmark.db import BenchmarkDB
from .cost_model import CostModel
from ..backend.registry import KernelRegistry
from ..ops.registry import get_reference_factory
from ..ops.atomic_types import OpType
from ..ir.graph import topological_sort
from .shape_inference import ShapeInference


class ExecutionRecipe:
    def __init__(self, root: TensorNode, assignments: Dict[TensorNode, Backend]):
        self.root = root
        self.assignments = assignments


class Planner:
    def __init__(self, db_path="benchmarks.db", greedy: bool = True):
        self.db = BenchmarkDB(db_path)
        self.cost_model = CostModel(self.db)
        # Memoization: (node_hash, target_backend) -> (cost, rewritten_node, backend_assignments)
        self.memo = {}
        self.greedy = greedy

    def plan(
        self, root: TensorNode, known_values: Optional[Dict[str, Any]] = None
    ) -> ExecutionRecipe:
        """
        Generates an optimized execution plan for the given graph root.
        """
        # 1. Run Shape Inference before planning.
        # This ensures constants get shapes, preventing hash collisions.
        nodes = topological_sort(root)
        ShapeInference.infer(nodes, known_values)

        # 2. Proceed with cost-based planning
        _, new_root, assignments = self._min_cost(
            root, root.backend, expansion_stack=frozenset(), known_values=known_values
        )
        return ExecutionRecipe(new_root, assignments)

    def _min_cost(
        self,
        node: TensorNode,
        target_backend: Backend,
        expansion_stack: FrozenSet,
        known_values: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, TensorNode, Dict[TensorNode, Backend]]:

        node_hash = get_structural_hash(node) + f"|{target_backend.value}"
        if node_hash in self.memo:
            return self.memo[node_hash]

        candidates = []

        # 1. Base Cases
        if node.op_type in (OpType.INPUT, OpType.CONSTANT):
            transfer_cost = self.cost_model.estimate_transfer_cost(
                node.backend, target_backend, node.shape, node.dtype
            )
            if node.backend != target_backend:
                new_node = TensorNode(
                    OpType.COPY_TO,
                    node.dtype,
                    [node],
                    node.shape,
                    f"copy_{node.name}",
                    attrs={"target_backend": target_backend.value},
                    backend=target_backend,
                )
                assignments = {node: node.backend, new_node: target_backend}
                res = (transfer_cost, new_node, assignments)
            else:
                res = (0.0, node, {node: node.backend})

            self.memo[node_hash] = res
            return res

        # 2. Strategy C: Fusion (Prioritized in Greedy Mode)
        # We check fusion before direct execution because fused kernels are generally more efficient.
        if node.op_type == OpType.ADD and len(node.parents) == 2:
            for i, p in enumerate(node.parents):
                if p.op_type == OpType.MUL and len(p.parents) == 2:
                    mul_node = p
                    other_node = node.parents[1 - i]
                    fma_parents = mul_node.parents + [other_node]
                    fma_sigs = [
                        TensorSignature(parent.dtype, parent.shape, target_backend)
                        for parent in fma_parents
                    ]
                    if KernelRegistry.select_best_kernel(
                        "FusedMulAdd", fma_sigs, target_backend, node.dtype
                    ):
                        fma_node = TensorNode(
                            "FusedMulAdd",
                            node.dtype,
                            fma_parents,
                            node.shape,
                            f"fused_{node.name}",
                            backend=target_backend,
                        )
                        fma_cost, fma_root, fma_assigns = self._min_cost(
                            fma_node, target_backend, expansion_stack, known_values
                        )
                        res = (fma_cost, fma_root, fma_assigns)

                        if self.greedy:
                            self.memo[node_hash] = res
                            return res

                        candidates.append(res)
                        break

        # 3. Strategy A: Direct Kernel Execution
        parent_results = [
            self._min_cost(p, target_backend, expansion_stack, known_values)
            for p in node.parents
        ]
        parent_cost = sum(r[0] for r in parent_results)
        parent_nodes = [r[1] for r in parent_results]
        parent_assigns = {}
        for r in parent_results:
            parent_assigns.update(r[2])

        input_sigs = [p.signature for p in parent_nodes]
        kernel = KernelRegistry.select_best_kernel(
            node.op_type, input_sigs, target_backend, node.dtype
        )

        if kernel:
            k_cost = self.cost_model.estimate_kernel_cost(
                node.op_type, target_backend, node.dtype, node.shape, node.attrs
            )
            new_node = copy.copy(node)
            new_node.parents = parent_nodes
            new_node.backend = target_backend
            assignments = parent_assigns.copy()
            assignments[new_node] = target_backend

            res = (parent_cost + k_cost, new_node, assignments)

            if self.greedy:
                self.memo[node_hash] = res
                return res

            candidates.append(res)

        # 4. Strategy B: Decomposition (Expansion)
        # In greedy mode, we only try decomposition if Direct Kernel failed (was not returned above).
        if node.op_type not in expansion_stack:
            ref_factory = get_reference_factory(node.op_type)
            if ref_factory:
                new_stack = expansion_stack | {node.op_type}
                subgraph_root = ref_factory(node.parents, node.attrs)

                # Run shape inference on the decomposition.
                subgraph_nodes = topological_sort(subgraph_root)
                ShapeInference.infer(subgraph_nodes, known_values)

                if node.shape and not subgraph_root.shape:
                    subgraph_root.shape = node.shape

                decomp_cost, decomp_root, decomp_assigns = self._min_cost(
                    subgraph_root, target_backend, new_stack, known_values
                )

                res = (decomp_cost, decomp_root, decomp_assigns)

                if self.greedy:
                    self.memo[node_hash] = res
                    return res

                candidates.append(res)

        # 5. Final Evaluation / Fallback
        if not candidates:
            if target_backend != Backend.CPU_NUMPY:
                cpu_cost, cpu_root, cpu_assigns = self._min_cost(
                    node, Backend.CPU_NUMPY, expansion_stack, known_values
                )
                transfer_back_cost = self.cost_model.estimate_transfer_cost(
                    Backend.CPU_NUMPY, target_backend, node.shape, node.dtype
                )
                copy_back = TensorNode(
                    OpType.COPY_TO,
                    node.dtype,
                    [cpu_root],
                    node.shape,
                    f"copy_back_{node.name}",
                    attrs={"target_backend": target_backend.value},
                    backend=target_backend,
                )
                final_assigns = cpu_assigns.copy()
                final_assigns[copy_back] = target_backend
                res = (cpu_cost + transfer_back_cost, copy_back, final_assigns)
                self.memo[node_hash] = res
                return res
            else:
                raise RuntimeError(
                    f"No execution strategy found for {node.op_type} on {target_backend}.\nnode: {node}\nnode.parents: {node.parents}"
                )

        best = min(candidates, key=lambda x: x[0])
        self.memo[node_hash] = best
        return best
