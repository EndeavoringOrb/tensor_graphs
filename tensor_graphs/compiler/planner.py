import copy
from typing import Dict, Optional, List, Any
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
from ..ir.rewrite import (
    CommutativeRule,
    DistributiveRule,
    FactoringRule,
    AssociativeRule,
    DoubleNegationRule,
    NegateAddRule,
    DivMulRule,
    DivAddRule,
    ExpAddRule,
    ExpAddReverseRule,
    generate_all_equivalents,
    match_pattern,
)


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
        atomic_root = self._expand_and_infer(root, known_values)
        atomic_nodes_topo = topological_sort(atomic_root)

        # ---------------------------------------------------------------------
        # PHASE 2: Pattern Matching & Fusion
        # ---------------------------------------------------------------------
        self._populate_fusion_map_generalized(atomic_nodes_topo)

        # ---------------------------------------------------------------------
        # PHASE 3: Kernel Selection (Beam Search)
        # ---------------------------------------------------------------------
        sorted_nodes = self._get_augmented_topological_sort(atomic_nodes_topo)

        for node in tqdm(sorted_nodes, disable=not DEBUG_EXECUTION, desc="Planning"):
            self._plan_node_iterative(node, known_values)

        # ---------------------------------------------------------------------
        # PHASE 4: Graph Reconstruction
        # ---------------------------------------------------------------------
        root_hash = get_structural_hash(atomic_root, memo=self.hash_memo)
        strategies = self.memo.get(root_hash)

        if not strategies:
            raise RuntimeError(
                f"Planner failed to find any execution strategy for {atomic_root}"
            )

        # Select the best strategy overall
        target_backend = atomic_root.backend
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
        # PHASE 5: Compilation & Instruction Generation
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

            # -----------------------------------------------------------------
            # PHASE 6: In-Place Optimization
            # -----------------------------------------------------------------
            is_inplace_safe = False
            if node.parents:
                p0 = node.parents[0]
                if p0.name in ref_counts:
                    if (
                        ref_counts[p0.name] == 1
                        and p0.storage_type == StorageType.TRANSIENT
                        and p0.size_bytes == node.size_bytes
                    ):
                        is_inplace_safe = True

            kernel_result = KernelRegistry.select_best_kernel(
                node.op_type,
                input_sigs,
                backend,
                target_dtype=node.dtype,
                allow_inplace=is_inplace_safe,
            )

            if not kernel_result:
                raise RuntimeError(
                    f"Kernel not found for {node.op_type} on {backend} with inputs {input_sigs}"
                )

            kernel, kernel_is_inplace = kernel_result
            inplace_input_index = None

            if kernel_is_inplace:
                if is_inplace_safe:
                    inplace_input_index = 0
                else:
                    raise ValueError(
                        "Chose inplace kernel when it is not safe. This should not be possible."
                    )

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

    def _make_atomic(
        self,
        root: TensorNode,
        known_values: Optional[Dict[str, Any]],
        atomic_map: Optional[Dict[TensorNode, TensorNode]] = None,
        keep_cut_parent_shapes=False,
    ) -> TensorNode:
        nodes = topological_sort(root)
        GraphPropagator.infer_shapes(
            nodes,
            known_values,
            keep_cut_parent_shapes=keep_cut_parent_shapes,
            disable_pbar=True,
        )

        is_top_level = atomic_map is None
        if is_top_level:
            atomic_map = {}

        iterable = (
            tqdm(nodes, disable=not DEBUG_EXECUTION, desc="making atomic graph")
            if is_top_level
            else nodes
        )

        for node in iterable:
            if node in atomic_map:
                continue

            if node.op_type in (OpType.INPUT, OpType.CONSTANT):
                atomic_map[node] = node
                continue

            atomic_parents = [atomic_map[p] for p in node.parents]

            if not OpType.is_atomic(node.op_type):
                ref_factory = get_reference_factory(node.op_type)
                if ref_factory:
                    proxy_parents = [
                        TensorNode(
                            op_type=p.op_type,
                            dtype=p.dtype,
                            parents=[],  # Cut off history
                            shape=p.shape,
                            name=p.name,
                            attrs=p.attrs,
                            backend=p.backend,
                            storage_type=p.storage_type,
                        )
                        for p in atomic_parents
                    ]
                    subgraph_root = ref_factory(proxy_parents, node.attrs)
                    sub_atomic_map = {
                        proxy: actual
                        for proxy, actual in zip(proxy_parents, atomic_parents)
                    }
                    atomic_map[node] = self._make_atomic(
                        subgraph_root, known_values, sub_atomic_map, True
                    )
                else:
                    raise ValueError(
                        f"No reference factory for fused op {node.op_type}"
                    )
            else:
                new_node = copy.copy(node)
                new_node.parents = atomic_parents
                atomic_map[node] = new_node

        return atomic_map[root]

    def _expand_and_infer(
        self, root: TensorNode, known_values: Optional[Dict[str, Any]]
    ) -> TensorNode:
        """
        Decomposes the graph until all nodes are atomic.
        Returns the root of the new fully atomic graph.
        """
        atomic_root = self._make_atomic(root, known_values)
        atomic_nodes = topological_sort(atomic_root)
        GraphPropagator.infer_shapes(atomic_nodes, known_values, disable_pbar=True)
        return atomic_root

    def _populate_fusion_map_generalized(self, atomic_nodes_topo: List[TensorNode]):
        rules = [
            CommutativeRule(),
            DistributiveRule(),
            FactoringRule(),
            AssociativeRule(),
            DoubleNegationRule(),
            NegateAddRule(),
            DivMulRule(),
            DivAddRule(),
            ExpAddRule(),
            ExpAddReverseRule(),
        ]

        class DummyAttrs(dict):
            def __missing__(self, key):
                return 1

        def collect_attrs(concrete_node, pattern_node, collected_attrs):
            if pattern_node.op_type == OpType.INPUT:
                return
            for k, v in concrete_node.attrs.items():
                if k not in collected_attrs:
                    collected_attrs[k] = v
            for c_p, p_p in zip(concrete_node.parents, pattern_node.parents):
                collect_attrs(c_p, p_p, collected_attrs)

        # --- 1. PRE-COMPUTE FUSED PATTERNS ---
        fused_patterns = []
        for op_type in tqdm(
            KernelRegistry.get_all_kernels().keys(),
            disable=not DEBUG_EXECUTION,
            desc="populating fused patterns",
        ):
            if not OpType.is_atomic(op_type):
                ref_factory = get_reference_factory(op_type)
                if not ref_factory:
                    continue

                first_backend = list(KernelRegistry.get_all_kernels()[op_type].keys())[
                    0
                ]
                signatures = KernelRegistry.get_all_kernels()[op_type][first_backend][
                    0
                ][1]

                # TODO: do this better somehow, this feels finicky
                dummy_inputs = []
                for i, sig in enumerate(signatures):
                    if sig.shape is None:
                        shape = (1, 32)
                    else:
                        shape = (d if d is not None else 32 for d in sig.shape)
                    dummy = TensorNode(
                        OpType.INPUT, sig.dtype, [], shape, name=f"dummy_{i}"
                    )
                    dummy_inputs.append(dummy)

                pattern_root = ref_factory(dummy_inputs, DummyAttrs())

                pattern_atomic_root = self._make_atomic(pattern_root, {})

                fused_patterns.append(
                    {
                        "op_type": op_type,
                        "variables": dummy_inputs,
                        "root": pattern_atomic_root,
                    }
                )

        # --- 2. MATCH IN MAIN GRAPH ---
        n_fused = 0
        for node in tqdm(
            reversed(atomic_nodes_topo),
            disable=not DEBUG_EXECUTION,
            desc="matching fused patterns",
            total=len(atomic_nodes_topo),
        ):
            node_hash = get_structural_hash(node, memo=self.hash_memo)

            equivalents = generate_all_equivalents(node, rules)

            for eq_node in equivalents:
                if eq_node is not node:
                    self.fusion_map.setdefault(node_hash, []).append(eq_node)

                for fp in fused_patterns:
                    binding = {}
                    if match_pattern(eq_node, fp["root"], fp["variables"], binding):
                        if not all(var in binding for var in fp["variables"]):
                            continue

                        concrete_inputs = [binding[var] for var in fp["variables"]]
                        collected_attrs = {}
                        collect_attrs(eq_node, fp["root"], collected_attrs)

                        fused_candidate = TensorNode(
                            op_type=fp["op_type"],
                            dtype=node.dtype,
                            parents=concrete_inputs,
                            shape=node.shape,
                            backend=node.backend,
                            attrs=collected_attrs,
                        )
                        fused_candidate.shape = node.shape

                        self.fusion_map.setdefault(node_hash, []).append(
                            fused_candidate
                        )
                        n_fused += 1
        if DEBUG_EXECUTION:
            print(f"# Fused operation matches: {n_fused}")

    def _get_augmented_topological_sort(
        self, nodes: List[TensorNode]
    ) -> List[TensorNode]:
        """
        Returns a topological sort that includes the primary atomic graph
        as well as all alternative fusion/rewrite candidates.
        """
        visited = set()
        sort = []

        def visit(n: TensorNode):
            n_hash = get_structural_hash(n, memo=self.hash_memo)
            if n_hash in visited:
                return
            visited.add(n_hash)

            # 1. Depend on parents of the current node
            for p in n.parents:
                visit(p)

            # 2. Depend on parents of all alternative fusion/topology candidates
            # This ensures that if we choose a fused op, its dependencies are already planned.
            if n_hash in self.fusion_map:
                for alt_node in self.fusion_map[n_hash]:
                    for p in alt_node.parents:
                        visit(p)

            sort.append(n)

        # Start from the root(s) of the atomic graph
        for node in nodes:
            visit(node)

        return sort

    def _plan_node_iterative(
        self, node: TensorNode, known_values: Optional[Dict[str, Any]]
    ):
        """
        Evaluates all possible execution strategies for a node (and its fusions).
        Stores the top K strategies in self.memo[node_hash].
        """
        node_hash = get_structural_hash(node, self.hash_memo)
        if node_hash in self.memo:
            return

        candidates: List[BeamStrategy] = []

        # --- 1. Base Cases: Inputs & Constants ---
        if node.op_type in (OpType.INPUT, OpType.CONSTANT):
            self.memo[node_hash] = [
                BeamStrategy(cost=0.0, node=node, assignments={node: node.backend})
            ]
            return

        # --- 2. Collection of Implementation Targets ---
        # We evaluate the "original" atomic node AND all fused/permuted alternatives
        targets = [node]
        if node_hash in self.fusion_map:
            targets.extend(self.fusion_map[node_hash])

        # --- 3. Evaluate each Target on available backends ---
        available_backends = [Backend.CPU_NUMPY]
        if torch.cuda.is_available():
            available_backends.append(Backend.GPU_TORCH)

        for target in targets:
            parent_hashes = [
                get_structural_hash(p, self.hash_memo) for p in target.parents
            ]
            # Get the beam strategies for every parent
            parent_beam_sets = [self.memo[ph] for ph in parent_hashes]

            for backend in available_backends:
                # Signature check for the kernel
                dummy_sigs = [
                    TensorSignature(p.dtype, p.shape, backend) for p in target.parents
                ]

                # Check if a kernel exists for this OpType/Backend/DType
                kernel_result = KernelRegistry.select_best_kernel(
                    target.op_type, dummy_sigs, backend, target.dtype
                )
                if not kernel_result:
                    continue

                _, kernel_inplace = kernel_result

                # Combine parent strategies (Cartesian product of parent beams)
                for p_strat_combo in itertools.product(*parent_beam_sets):
                    current_cost = 0.0
                    new_parents = []
                    current_assigns = {}

                    for i, p_strat in enumerate(p_strat_combo):
                        current_cost += p_strat.cost
                        current_assigns.update(p_strat.assignments)

                        # Logic for cross-backend data transfer
                        if p_strat.node.backend != backend:
                            t_cost = self.cost_model.estimate_transfer_cost(
                                p_strat.node.backend,
                                backend,
                                p_strat.node.shape,
                                p_strat.node.dtype,
                            )
                            current_cost += t_cost

                            # Insert an implicit CopyTo node
                            copy_node = TensorNode(
                                OpType.COPY_TO,
                                p_strat.node.dtype,
                                [p_strat.node],
                                p_strat.node.shape,
                                name=f"auto_copy_{p_strat.node.name}_to_{backend.value}",
                                attrs={"target_backend": backend.value},
                                backend=backend,
                            )
                            new_parents.append(copy_node)
                            current_assigns[copy_node] = backend
                        else:
                            new_parents.append(p_strat.node)

                    # Estimate the kernel execution cost
                    exec_cost = self.cost_model.estimate_kernel_cost(
                        target.op_type,
                        backend,
                        target.dtype,
                        target.shape,
                        target.attrs,
                        inplace=kernel_inplace,
                    )
                    current_cost += exec_cost

                    # Create a concrete implementation node for this strategy
                    impl_node = copy.copy(target)
                    impl_node.parents = new_parents
                    impl_node.backend = backend
                    current_assigns[impl_node] = backend

                    candidates.append(
                        BeamStrategy(current_cost, impl_node, current_assigns)
                    )

        # --- 4. Pruning (Keep Top K) ---
        candidates.sort()
        best_k = candidates[: self.beam_width]

        if not best_k:
            # Fallback for debugging: help identify which nodes fail planning
            print(f"FAILED TO PLAN NODE: {node.op_type} ({node_hash[:8]})")
            print(f"Parents: {[p.op_type for p in node.parents]}")
            raise ValueError(f"No execution path found for node {node.op_type}")

        self.memo[node_hash] = best_k
