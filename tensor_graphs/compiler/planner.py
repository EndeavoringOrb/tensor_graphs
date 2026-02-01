import copy
from typing import Dict, Tuple, Optional, FrozenSet
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
        # Memoization: (node_hash, target_backend) -> (cost, rewritten_node, backend_assignments)
        self.memo = {}

    def plan(self, root: TensorNode) -> ExecutionRecipe:
        """
        Generates an optimized execution plan for the given graph root.
        """
        # We start with an empty expansion_stack. This stack tracks OpTypes
        # currently being decomposed to prevent infinite expansion loops.
        _, new_root, assignments = self._min_cost(
            root, root.backend, expansion_stack=frozenset()
        )
        return ExecutionRecipe(new_root, assignments)

    def _min_cost(
        self, node: TensorNode, target_backend: Backend, expansion_stack: FrozenSet
    ) -> Tuple[float, TensorNode, Dict[TensorNode, Backend]]:

        # 1. Memoization Lookup
        # We include the target_backend in the key because the same node
        # might have different optimal strategies depending on where it needs to end up.
        node_hash = get_structural_hash(node) + f"|{target_backend.value}"
        if node_hash in self.memo:
            return self.memo[node_hash]

        candidates = []

        # 2. Base Cases: Input and Constant
        if node.op_type in (OpType.INPUT, OpType.CONSTANT):
            transfer_cost = self.cost_model.estimate_transfer_cost(
                node.backend, target_backend, node.shape, node.dtype
            )

            if node.backend != target_backend:
                # If backend differs, inject a CopyTo node
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
                # No transfer needed
                res = (0.0, node, {node: node.backend})

            self.memo[node_hash] = res
            return res

        # 3. Strategy A: Direct Kernel Execution
        # First, we find the optimal way to get all parents onto the target_backend.
        # There is NO depth limit here, as we are simply traversing the existing DAG.
        parent_results = [
            self._min_cost(p, target_backend, expansion_stack) for p in node.parents
        ]
        parent_cost = sum(r[0] for r in parent_results)
        parent_nodes = [r[1] for r in parent_results]
        parent_assigns = {}
        for r in parent_results:
            parent_assigns.update(r[2])

        input_sigs = [p.signature for p in parent_nodes]

        # Check if a kernel exists for this op on this backend
        kernel = KernelRegistry.select_best_kernel(
            node.op_type, input_sigs, target_backend, node.dtype
        )

        if kernel:
            k_cost = self.cost_model.estimate_kernel_cost(
                node.op_type, target_backend, node.dtype, node.shape, node.attrs
            )

            # Construct a version of the node that uses the optimized parent subgraphs
            new_node = copy.copy(node)
            new_node.parents = parent_nodes
            new_node.backend = target_backend

            assignments = parent_assigns.copy()
            assignments[new_node] = target_backend
            candidates.append((parent_cost + k_cost, new_node, assignments))

        # 4. Strategy B: Decomposition (Expansion)
        # We only decompose if we aren't already in the middle of decomposing this OpType.
        if node.op_type not in expansion_stack:
            ref_factory = get_reference_factory(node.op_type)
            if ref_factory:
                # Add this op to the stack to prevent infinite recursion (e.g. A -> B -> A)
                new_stack = expansion_stack | {node.op_type}

                # Expand the high-level op into its atomic/reference subgraph
                subgraph_root = ref_factory(node.parents, node.attrs)

                # Transfer shape from original node if available and missing in subgraph
                if node.shape and not subgraph_root.shape:
                    subgraph_root.shape = node.shape

                # Recursively plan the resulting subgraph
                decomp_cost, decomp_root, decomp_assigns = self._min_cost(
                    subgraph_root, target_backend, new_stack
                )
                candidates.append((decomp_cost, decomp_root, decomp_assigns))

        # 5. Strategy C: Fusion (Pattern Matching)
        # Example: Match (Mul + Add) -> FusedMulAdd
        if node.op_type == OpType.ADD and len(node.parents) == 2:
            for i, p in enumerate(node.parents):
                if p.op_type == OpType.MUL and len(p.parents) == 2:
                    mul_node = p
                    other_node = node.parents[1 - i]
                    fma_parents = mul_node.parents + [other_node]

                    # Verify a fused kernel actually exists for these signatures
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
                            fma_node, target_backend, expansion_stack
                        )
                        candidates.append((fma_cost, fma_root, fma_assigns))
                        break

        # 6. Final Evaluation
        if not candidates:
            # Fallback: If no way to run on target_backend, try running on CPU and copying back
            if target_backend != Backend.CPU_NUMPY:
                cpu_cost, cpu_root, cpu_assigns = self._min_cost(
                    node, Backend.CPU_NUMPY, expansion_stack
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
                # If we are already on CPU and have no strategies, we are stuck
                raise RuntimeError(
                    f"No execution strategy found for {node.op_type} on {target_backend}. "
                    "Ensure a kernel or reference decomposition is registered."
                )

        # Pick the candidate with the lowest estimated cost
        best = min(candidates, key=lambda x: x[0])
        self.memo[node_hash] = best
        return best
