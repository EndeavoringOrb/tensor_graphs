import copy
from typing import Dict, Optional, List, Any, Set
from dataclasses import dataclass
import itertools
from tqdm import tqdm
import torch  # Import torch to check for CUDA availability

from ..ir.node import TensorNode
from ..ir.dtypes import Backend, TensorSignature
from ..ir.hashing import get_structural_hash
from ..benchmark.db import BenchmarkDB
from .cost_model import CostModel
from ..backend.registry import KernelRegistry
from ..ops.registry import get_reference_factory
from ..ops.atomic_types import OpType
from ..ir.graph import topological_sort
from .propagation import GraphPropagator
from ..config import DEBUG_EXECUTION, PLANNER_BEAM_WIDTH
from .compiled_graph import CompiledGraph, OpInstruction
from ..ir.buffer import StorageType


@dataclass
class BeamStrategy:
    cost: float
    node: TensorNode
    assignments: Dict[TensorNode, Backend]

    def __lt__(self, other):
        return self.cost < other.cost


class Planner:
    def __init__(self, db_path="benchmarks.db"):
        self.db = BenchmarkDB(db_path)
        self.cost_model = CostModel(self.db)
        self.memo: Dict[str, List[BeamStrategy]] = {}
        self.hash_memo: Dict[TensorNode, str] = {}
        self.beam_width = PLANNER_BEAM_WIDTH
        self.decomp_map: Dict[str, TensorNode] = {}
        self.fusion_map: Dict[str, List[TensorNode]] = {}

    def plan(
        self, root: TensorNode, known_values: Optional[Dict[str, Any]] = None
    ) -> CompiledGraph:
        """
        Plans the execution graph, selects best kernels, and compiles it into an executable graph.
        Merges Planning, Compilation, and Optimization phases.
        """
        # ---------------------------------------------------------------------
        # PHASE 1: Expansion & Shape Inference
        # ---------------------------------------------------------------------
        all_potential_nodes = self._expand_and_infer(root, known_values)

        # ---------------------------------------------------------------------
        # PHASE 2: Kernel Selection (Beam Search)
        # ---------------------------------------------------------------------
        sorted_nodes = self._get_augmented_topological_sort(all_potential_nodes)

        for node in tqdm(sorted_nodes, disable=not DEBUG_EXECUTION, desc="Planning"):
            self._plan_node_iterative(node, known_values)

        # ---------------------------------------------------------------------
        # PHASE 3: Graph Reconstruction
        # ---------------------------------------------------------------------
        root_hash = get_structural_hash(root)
        strategies = self.memo.get(root_hash)

        if not strategies:
            raise RuntimeError(
                f"Planner failed to find any execution strategy for {root}"
            )

        # Select the best strategy overall
        target_backend = root.backend
        candidates = []
        for strat in strategies:
            if strat.node.backend == target_backend:
                candidates.append(strat)
            else:
                transfer_cost = self.cost_model.estimate_transfer_cost(
                    strat.node.backend,
                    target_backend,
                    strat.node.shape,
                    strat.node.dtype,
                )
                copy_node = TensorNode(
                    OpType.COPY_TO,
                    strat.node.dtype,
                    [strat.node],
                    strat.node.shape,
                    f"final_copy_{strat.node.name}",
                    attrs={"target_backend": target_backend.value},
                    backend=target_backend,
                )
                new_assigns = strat.assignments.copy()
                new_assigns[copy_node] = target_backend
                candidates.append(
                    BeamStrategy(strat.cost + transfer_cost, copy_node, new_assigns)
                )

        best_recipe = min(candidates, key=lambda s: s.cost)

        # ---------------------------------------------------------------------
        # PHASE 4: Compilation & Instruction Generation
        # ---------------------------------------------------------------------
        final_root = best_recipe.node
        nodes = topological_sort(final_root)

        # Ref Counting
        ref_counts = {n.name: 0 for n in nodes}
        for node in nodes:
            for p in node.parents:
                ref_counts[p.name] = ref_counts.get(p.name, 0) + 1
        ref_counts[final_root.name] += 1

        instructions: List[OpInstruction] = []
        nodes_map: Dict[str, TensorNode] = {}

        for node in nodes:
            if node.name in nodes_map:
                continue  # Should not happen in topo sort of tree->dag
            nodes_map[node.name] = node

            if node.op_type in (OpType.INPUT, OpType.CONSTANT):
                continue

            # Resolve backend from recipe or node default
            backend = best_recipe.assignments.get(node, node.backend)
            input_sigs = [p.signature for p in node.parents]

            # Initial Selection: Allow any kernel (inplace or not) to ensure we find *something*
            kernel_result = KernelRegistry.select_best_kernel(
                node.op_type,
                input_sigs,
                backend,
                target_dtype=node.dtype,
                allow_inplace=True,
            )

            if not kernel_result:
                raise RuntimeError(
                    f"Kernel not found for {node.op_type} on {backend} with inputs {input_sigs}"
                )

            kernel, kernel_is_inplace = kernel_result
            inplace_input_index = None

            # -----------------------------------------------------------------
            # PHASE 5: In-Place Optimization
            # -----------------------------------------------------------------
            # Check if we can enable in-place execution safely

            # Criteria for In-Place Safety on Input 0 (Standard convention):
            # 1. Input 0 exists.
            # 2. This node is the *last* consumer of Input 0 (ref_count == 1).
            # 3. Input 0 is TRANSIENT (not a persistent weight/constant).
            # 4. Sizes match exactly.

            is_safe = False
            if node.parents:
                p0 = node.parents[0]
                # Check if p0 is in our current graph (it should be)
                if p0.name in ref_counts:
                    # Note: We check ref_counts[p0.name] == 1.
                    # Since we are processing in topological order, if ref_count is 1,
                    # it means this current node is the ONLY remaining consumer.
                    if (
                        ref_counts[p0.name] == 1
                        and p0.storage_type == StorageType.TRANSIENT
                        and p0.size_bytes == node.size_bytes
                    ):
                        is_safe = True

            if kernel_is_inplace:
                if is_safe:
                    # Safe to use the inplace kernel in inplace mode
                    inplace_input_index = 0
                else:
                    # Kernel is inplace-capable, but usage is unsafe.
                    # We must run it out-of-place (inplace_input_index = None).
                    # Executor handles this by allocating separate output.
                    inplace_input_index = None
            else:
                # Kernel is NOT inplace.
                # Optimization: Can we find an inplace kernel if we look specifically?
                if is_safe:
                    inplace_result = KernelRegistry.select_best_kernel(
                        node.op_type,
                        input_sigs,
                        backend,
                        target_dtype=node.dtype,
                        allow_inplace=True,
                    )
                    # If the registry returns an inplace kernel now (it might have preferred a non-inplace one before due to score/order)
                    # We check if it's actually inplace
                    if inplace_result:
                        ip_kernel, ip_bool = inplace_result
                        if ip_bool:
                            kernel = ip_kernel
                            inplace_input_index = 0

            input_names = [p.name for p in node.parents]

            instr = OpInstruction(
                node_name=node.name,
                kernel=kernel,
                input_node_names=input_names,
                attrs=node.attrs,
                inplace_input_index=inplace_input_index,
            )
            instructions.append(instr)

        return CompiledGraph(
            instructions=instructions,
            ref_counts=ref_counts,
            nodes_map=nodes_map,
        )

    def _expand_and_infer(
        self, root: TensorNode, known_values: Optional[Dict[str, Any]]
    ) -> List[TensorNode]:
        """
        Iteratively expands the graph and infers shapes.
        Ensures parents have shapes before calling decomposition factories.
        """
        all_nodes_discovered: Set[TensorNode] = set()

        # Use a worklist starting with the original graph in topological order
        worklist = topological_sort(root)
        # Initial shape inference for the primary graph
        GraphPropagator.infer_shapes(worklist, known_values, disable_pbar=True)

        index = 0
        with tqdm(
            disable=not DEBUG_EXECUTION, total=len(worklist), desc="expand&infer"
        ) as pbar:
            while index < len(worklist):
                node = worklist[index]
                index += 1
                if DEBUG_EXECUTION:
                    pbar.update(1)
                    pbar.set_description_str(f"expand&infer [{index}/{len(worklist)}]")

                node_hash = get_structural_hash(node, memo=self.hash_memo)
                if node in all_nodes_discovered:
                    continue
                all_nodes_discovered.add(node)

                # --- 1. Handle Decompositions ---
                if not OpType.is_atomic(node.op_type):
                    ref_factory = get_reference_factory(node.op_type)
                    if ref_factory:
                        # Because worklist is sorted, node.parents already have shapes.
                        subgraph_root = ref_factory(node.parents, node.attrs)
                        self.decomp_map[node_hash] = subgraph_root

                        # Discover sub-graph nodes and infer their shapes
                        sub_nodes = topological_sort(subgraph_root)
                        GraphPropagator.infer_shapes(
                            sub_nodes, known_values, disable_pbar=True
                        )

                        # Add new nodes to worklist to check if THEY need decomposition
                        for sn in sub_nodes:
                            if sn not in all_nodes_discovered:
                                worklist.append(sn)

                # --- 2. Handle Fusions ---
                if node.op_type == OpType.ADD and len(node.parents) == 2:
                    for i, p in enumerate(node.parents):
                        if p.op_type == OpType.MUL and len(p.parents) == 2:
                            fma_parents = p.parents + [node.parents[1 - i]]
                            fma_node = TensorNode(
                                "FusedMulAdd",
                                node.dtype,
                                fma_parents,
                                node.shape,
                                f"fused_{node.name}",
                                backend=node.backend,
                            )
                            # Fused node shape is same as original node
                            fma_node.shape = node.shape

                            if node_hash not in self.fusion_map:
                                self.fusion_map[node_hash] = []
                            self.fusion_map[node_hash].append(fma_node)

                            if fma_node not in all_nodes_discovered:
                                all_nodes_discovered.add(node)
                            break

        return list(all_nodes_discovered)

    def _get_augmented_topological_sort(
        self, nodes: List[TensorNode]
    ) -> List[TensorNode]:
        visited = set()
        sort = []

        def visit(n):
            n_hash = get_structural_hash(n, memo=self.hash_memo)
            if n_hash in visited:
                return
            visited.add(n_hash)

            # 1. Depend on parents
            for p in n.parents:
                visit(p)

            # 2. Depend on decomposition sub-roots (must plan impl before container)
            if n_hash in self.decomp_map:
                visit(self.decomp_map[n_hash])

            # 3. Depend on fusion candidates
            if n_hash in self.fusion_map:
                for fused_node in self.fusion_map[n_hash]:
                    visit(fused_node)

            sort.append(n)

        for node in tqdm(nodes, disable=not DEBUG_EXECUTION, desc="topo sort"):
            visit(node)
        return sort

    def _plan_node_iterative(
        self, node: TensorNode, known_values: Optional[Dict[str, Any]]
    ):
        node_hash = get_structural_hash(node)
        if node_hash in self.memo:
            return

        candidates: List[BeamStrategy] = []

        # --- 1. Base Cases ---
        if node.op_type in (OpType.INPUT, OpType.CONSTANT):
            self.memo[node_hash] = [
                BeamStrategy(cost=0.0, node=node, assignments={node: node.backend})
            ]
            return

        # --- 2. Direct Kernel Execution ---
        # Note: All parents guaranteed to be in memo due to augmented sort
        parent_hashes = [get_structural_hash(p) for p in node.parents]
        parent_strats = [self.memo[ph] for ph in parent_hashes]

        available_backends = [Backend.CPU_NUMPY]

        # Check if GPU is available in BOTH registry AND hardware
        if torch.cuda.is_available():
            if KernelRegistry.has_kernel(node.op_type, Backend.GPU_TORCH) or any(
                KernelRegistry.has_kernel(op, Backend.GPU_TORCH)
                for op in KernelRegistry.get_all_kernels().keys()
            ):
                if Backend.GPU_TORCH not in available_backends:
                    available_backends.append(Backend.GPU_TORCH)

        for backend in available_backends:
            # STRICT KERNEL CHECKING:
            # Instead of generic has_kernel, we verify if a kernel exists
            # for the *specific* shapes/dtypes of the inputs.
            # We reconstruct the input signatures for this check.

            # Since parents might have different backends in their strategies,
            # we check if *this* backend can support the op assuming inputs are converted.
            # The signatures passed to select_best_kernel should reflect the *target* backend
            # because we insert copies if needed.

            dummy_sigs = []
            for p in node.parents:
                dummy_sigs.append(TensorSignature(p.dtype, p.shape, backend))

            # Check if kernel exists
            if not KernelRegistry.select_best_kernel(
                node.op_type, dummy_sigs, backend, node.dtype
            ):
                continue

            for p_strat_combo in itertools.product(*parent_strats):
                current_cost = 0.0
                new_parents = []
                current_assigns = {}

                for p_strat in p_strat_combo:
                    current_cost += p_strat.cost
                    current_assigns.update(p_strat.assignments)

                    if p_strat.node.backend != backend:
                        t_cost = self.cost_model.estimate_transfer_cost(
                            p_strat.node.backend,
                            backend,
                            p_strat.node.shape,
                            p_strat.node.dtype,
                        )
                        current_cost += t_cost
                        copy_node = TensorNode(
                            OpType.COPY_TO,
                            p_strat.node.dtype,
                            [p_strat.node],
                            p_strat.node.shape,
                            f"auto_copy_{p_strat.node.name}",
                            attrs={"target_backend": backend.value},
                            backend=backend,
                        )
                        new_parents.append(copy_node)
                        current_assigns[copy_node] = backend
                    else:
                        new_parents.append(p_strat.node)

                exec_cost = self.cost_model.estimate_kernel_cost(
                    node.op_type, backend, node.dtype, node.shape, node.attrs
                )
                current_cost += exec_cost
                new_node = copy.copy(node)
                new_node.parents = new_parents
                new_node.backend = backend
                current_assigns[new_node] = backend
                candidates.append(BeamStrategy(current_cost, new_node, current_assigns))

        # --- 3. Decomposition Strategies ---
        if node_hash in self.decomp_map:
            decomp_root_hash = get_structural_hash(self.decomp_map[node_hash])
            if decomp_root_hash in self.memo:
                candidates.extend(self.memo[decomp_root_hash])

        # --- 4. Fusion Strategies ---
        if node_hash in self.fusion_map:
            for fused_node in self.fusion_map[node_hash]:
                fused_hash = get_structural_hash(fused_node)
                if fused_hash in self.memo:
                    candidates.extend(self.memo[fused_hash])

        # --- 5. Pruning ---
        candidates.sort()
        best_k = candidates[: self.beam_width]

        if not best_k:
            # Final fallback: if no kernel exists and no decomposition worked,
            # the graph is technically un-executable.
            raise ValueError(
                f"No execution path found for node {node.name} ({node.op_type}) with shape {node.shape}"
            )

        self.memo[node_hash] = best_k
