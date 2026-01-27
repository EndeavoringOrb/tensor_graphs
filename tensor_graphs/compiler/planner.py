import copy
from typing import Dict, Tuple, Optional
from ..ir.node import TensorNode
from ..ir.dtypes import Backend, TensorSignature
from ..ir.hashing import get_structural_hash
from ..benchmark.db import BenchmarkDB
from .cost_model import CostModel
from ..backend.registry import KernelRegistry
from ..ops.registry import get_reference_factory
from ..ops.atomic_types import OpType


class ExecutionRecipe:
    def __init__(self, root: TensorNode, assignments: Dict[TensorNode, Backend]):
        self.root = root
        self.assignments = assignments


class Planner:
    def __init__(self, db_path="benchmarks.db"):
        self.db = BenchmarkDB(db_path)
        self.cost_model = CostModel(self.db)
        # Memoization: hash -> (cost, rewritten_node, backend_assignments)
        self.memo = {}

    def plan(self, root: TensorNode) -> ExecutionRecipe:
        _, new_root, assignments = self._min_cost(root, root.backend)
        return ExecutionRecipe(new_root, assignments)

    def _min_cost(
        self, node: TensorNode, target_backend: Backend, depth=0
    ) -> Tuple[float, TensorNode, Dict]:
        # Hash including target backend because transferring logic is specific to destination
        node_hash = get_structural_hash(node) + f"|{target_backend.value}"

        if node_hash in self.memo:
            return self.memo[node_hash]

        candidates = []

        # 1. Base Cases (Input/Const)
        if node.op_type in (OpType.INPUT, OpType.CONSTANT):
            # Input transfer cost
            transfer_cost = self.cost_model.estimate_transfer_cost(
                node.backend, target_backend, node.shape, node.dtype
            )
            # If backend differs, inject CopyTo
            if node.backend != target_backend:
                new_node = TensorNode(
                    OpType.COPY_TO,
                    node.shape,
                    node.dtype,
                    [node],
                    f"copy_{node.name}",
                    attrs={"target_backend": target_backend.value},
                    backend=target_backend,
                )
                # Assign logic
                assignments = {node: node.backend, new_node: target_backend}
                res = (transfer_cost, new_node, assignments)
                self.memo[node_hash] = res
                return res
            else:
                res = (0.0, node, {node: node.backend})
                self.memo[node_hash] = res
                return res

        # 2. Strategy A: Direct Kernel on Target Backend
        # We process parents recursively first to get their costs and rewritten forms on THIS backend
        parent_results = [
            self._min_cost(p, target_backend, depth + 1) for p in node.parents
        ]
        parent_cost = sum(r[0] for r in parent_results)
        parent_nodes = [r[1] for r in parent_results]
        parent_assigns = {}
        for r in parent_results:
            parent_assigns.update(r[2])

        input_sigs = [p.signature for p in parent_nodes]

        # Check if kernel exists
        kernel = KernelRegistry.select_best_kernel(
            node.op_type, input_sigs, target_backend, node.dtype
        )
        if kernel:
            k_cost = self.cost_model.estimate_kernel_cost(
                node.op_type, target_backend, node.dtype, node.shape, node.attrs
            )
            total_cost = parent_cost + k_cost

            # Construct rewritten node using rewritten parents
            new_node = copy.copy(node)
            new_node.parents = parent_nodes
            new_node.backend = target_backend

            assignments = parent_assigns.copy()
            assignments[new_node] = target_backend

            candidates.append((total_cost, new_node, assignments))

        # 3. Strategy B: Decomposition
        if (
            depth < 10
        ):  # Prevent infinite recursion. TODO: remove this, we shouldn't have ops calling themselves (i don't think), so this shouldn't be needed
            ref_factory = get_reference_factory(node.op_type)
            if ref_factory:
                # Decompose into subgraph
                # Important: Factory uses original parents.
                # But we need to plan the decomposed graph.
                # The decomposition might introduce new internal nodes.
                subgraph_root = ref_factory(node.parents, node.attrs)

                # Recursively plan the subgraph
                decomp_cost, decomp_root, decomp_assigns = self._min_cost(
                    subgraph_root, target_backend, depth + 1
                )

                candidates.append((decomp_cost, decomp_root, decomp_assigns))

        # 4. Strategy C: Fusion (Simple FusedMulAdd)
        # Check matching Mul->Add
        if node.op_type == OpType.ADD and len(node.parents) == 2:
            # Try to match a*b + c
            # We iterate parents to find Mul
            for i, p in enumerate(node.parents):
                if p.op_type == OpType.MUL and len(p.parents) == 2:
                    # Found candidate: (A*B) + C
                    mul_node = p
                    other_node = node.parents[1 - i]

                    # Construct high-level Fused Node
                    fma_parents = mul_node.parents + [other_node]

                    # Check if a kernel actually exists for this FMA on target backend
                    # This prevents infinite recursion/RuntimeErrors when no kernel supports the specific shapes
                    fma_sigs = [
                        TensorSignature(parent.dtype, parent.shape, target_backend)
                        for parent in fma_parents
                    ]
                    if not KernelRegistry.select_best_kernel(
                        "FusedMulAdd", fma_sigs, target_backend, node.dtype
                    ):
                        continue

                    fma_node = TensorNode(
                        "FusedMulAdd",
                        node.shape,
                        node.dtype,
                        fma_parents,
                        f"fused_{node.name}",
                        backend=node.backend,
                    )

                    # Recursive plan for FMA
                    fma_cost, fma_root, fma_assigns = self._min_cost(
                        fma_node, target_backend, depth + 1
                    )
                    candidates.append((fma_cost, fma_root, fma_assigns))
                    break

        if not candidates:
            # Fallback: If no kernel and no decomposition, try to transfer to CPU_NUMPY if we are not there?
            if target_backend != Backend.CPU_NUMPY:
                # Try planning on CPU
                cpu_cost, cpu_root, cpu_assigns = self._min_cost(
                    node, Backend.CPU_NUMPY, depth + 1
                )

                # Add transfer back cost
                transfer_back = self.cost_model.estimate_transfer_cost(
                    Backend.CPU_NUMPY, target_backend, node.shape, node.dtype
                )
                copy_back = TensorNode(
                    OpType.COPY_TO,
                    node.shape,
                    node.dtype,
                    [cpu_root],
                    f"copy_back_{node.name}",
                    attrs={"target_backend": target_backend.value},
                    backend=target_backend,
                )

                final_assigns = cpu_assigns.copy()
                final_assigns[copy_back] = target_backend

                res = (cpu_cost + transfer_back, copy_back, final_assigns)
                self.memo[node_hash] = res
                return res
            else:
                raise RuntimeError(
                    f"No execution strategy found for {node.op_type} on {target_backend}"
                )

        # Pick Best
        best = min(candidates, key=lambda x: x[0])
        self.memo[node_hash] = best
        return best
