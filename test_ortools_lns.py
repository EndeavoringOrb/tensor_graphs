"""Unit tests for iterative subgraph LNS solver and modular neighborhood selectors.

Run with:
.venv/Scripts/python.exe -m unittest test_ortools_lns
"""

import unittest
from typing import Any, Dict, List

from ortools_lns import (
    CompositeNeighborhoodSelector,
    CriticalPathSelector,
    MultiChoiceSelector,
    NeighborhoodSelector,
    NeuralNeighborhoodSelector,
    OrtoolsLnsSolver,
    RandomSubgraphSelector,
    cp_model,
    solveOrtools,
)

CPU = {"type": 1, "idx": 1}
PAGE = 4096


def makeNode(index=0, children=(), cost=1.0, **kwargs):
    return dict(enode_idx=index, children=list(children), cost=cost, **kwargs)


def makeClass(cid, enodes, pages=1, mem_space=None):
    return {
        "id": cid,
        "base_eclass_id": cid,
        "enodes": enodes,
        "size_bytes": pages * PAGE,
        "raw_size_bytes": pages * PAGE,
        "mem_space": mem_space or CPU,
    }


def makeProblem(classes, root, pages=10, hints=None):
    prob = {
        "use_ortools_lns": True,
        "disable_caching": False,
        "print_progress": False,
        "max_time_seconds": 2.0,
        "mem_caps": {"1:1": pages * PAGE},
        "full_bucket_idx": 0,
        "buckets": [{"bucket_idx": 0, "classes": classes, "root_eclass_id": root}],
    }
    if hints:
        prob["cpu_hints"] = hints
    return prob


class CustomDummySelector(NeighborhoodSelector):
    """Custom selector demonstrating the modular extension API."""

    def __init__(self, fixed_nodes):
        self.fixed_nodes = set(fixed_nodes)

    def selectUnfrozenNodes(
        self,
        bucket,
        selection_map,
        order,
        start_times,
        end_times,
        slack,
        critical_nodes,
        classes_by_id,
        iteration,
    ):
        return self.fixed_nodes


class TestOrtoolsLns(unittest.TestCase):
    def testRandomSubgraphSelector(self):
        classes = [
            makeClass(1, [makeNode(0, cost=1.0)]),
            makeClass(2, [makeNode(0, [1], cost=2.0), makeNode(1, [1], cost=1.0)]),
            makeClass(3, [makeNode(0, [2], cost=1.0)]),
        ]
        classes_by_id = {c["id"]: c for c in classes}
        selection_map = {1: 0, 2: 0, 3: 0}
        order = [1, 2, 3]
        start_times = {1: 0, 2: 1000, 3: 3000}
        end_times = {1: 1000, 2: 3000, 3: 4000}
        slack = {1: 0, 2: 0, 3: 0}
        critical_nodes = [1, 2, 3]

        selector = RandomSubgraphSelector(target_size=2)
        unfrozen = selector.selectUnfrozenNodes(
            {"classes": classes},
            selection_map,
            order,
            start_times,
            end_times,
            slack,
            critical_nodes,
            classes_by_id,
            1,
        )
        self.assertIsInstance(unfrozen, set)
        self.assertGreater(len(unfrozen), 0)
        self.assertTrue(unfrozen.issubset({1, 2, 3}))

    def testCriticalPathSelector(self):
        classes = [
            makeClass(1, [makeNode(0, cost=1.0)]),
            makeClass(2, [makeNode(0, [1], cost=5.0), makeNode(1, [1], cost=2.0)]),
            makeClass(3, [makeNode(0, [1], cost=1.0)]),
            makeClass(4, [makeNode(0, [2, 3], cost=1.0)]),
        ]
        classes_by_id = {c["id"]: c for c in classes}
        selection_map = {1: 0, 2: 0, 3: 0, 4: 0}
        order = [1, 2, 3, 4]
        start_times = {1: 0, 2: 1000, 3: 1000, 4: 6000}
        end_times = {1: 1000, 2: 6000, 3: 2000, 4: 7000}
        # Node 3 finishes at 2000 while node 4 starts at 6000 -> node 3 has slack 4000
        slack = {1: 0, 2: 0, 3: 4000, 4: 0}
        critical_nodes = [1, 2, 4]

        selector = CriticalPathSelector(target_size=2)
        unfrozen = selector.selectUnfrozenNodes(
            {"classes": classes},
            selection_map,
            order,
            start_times,
            end_times,
            slack,
            critical_nodes,
            classes_by_id,
            1,
        )
        self.assertIsInstance(unfrozen, set)
        # Should prioritize critical path nodes (1, 2, 4)
        self.assertTrue(any(c in critical_nodes for c in unfrozen))

    def testMultiChoiceSelector(self):
        classes = [
            makeClass(1, [makeNode(0, cost=1.0)]),
            makeClass(2, [makeNode(0, [1], cost=4.0), makeNode(1, [1], cost=1.0)]),
            makeClass(3, [makeNode(0, [2], cost=2.0)]),
        ]
        classes_by_id = {c["id"]: c for c in classes}
        selection_map = {1: 0, 2: 0, 3: 0}
        order = [1, 2, 3]

        selector = MultiChoiceSelector(target_size=2)
        unfrozen = selector.selectUnfrozenNodes(
            {"classes": classes},
            selection_map,
            order,
            {1: 0, 2: 1000, 3: 5000},
            {1: 1000, 2: 5000, 3: 7000},
            {1: 0, 2: 0, 3: 0},
            [1, 2, 3],
            classes_by_id,
            1,
        )
        self.assertIn(2, unfrozen)

    def testCompositeSelector(self):
        classes = [
            makeClass(1, [makeNode(0, cost=1.0)]),
            makeClass(2, [makeNode(0, [1], cost=2.0)]),
        ]
        classes_by_id = {c["id"]: c for c in classes}
        selector = CompositeNeighborhoodSelector()
        unfrozen = selector.selectUnfrozenNodes(
            {"classes": classes},
            {1: 0, 2: 0},
            [1, 2],
            {1: 0, 2: 1000},
            {1: 1000, 2: 3000},
            {1: 0, 2: 0},
            [1, 2],
            classes_by_id,
            1,
        )
        self.assertIsInstance(unfrozen, set)

    def testNeuralSelectorFallback(self):
        classes = [makeClass(1, [makeNode(0, cost=1.0)])]
        classes_by_id = {1: classes[0]}
        selector = NeuralNeighborhoodSelector(model_path=None)
        unfrozen = selector.selectUnfrozenNodes(
            {"classes": classes},
            {1: 0},
            [1],
            {1: 0},
            {1: 1000},
            {1: 0},
            [1],
            classes_by_id,
            1,
        )
        self.assertIn(1, unfrozen)

    def testLnsSolverFindsFasterKernel(self):
        # Node 2 has slow enode 0 (cost=10) and fast enode 1 (cost=2)
        classes = [
            makeClass(1, [makeNode(0, cost=1.0)]),
            makeClass(2, [makeNode(0, [1], cost=10.0), makeNode(1, [1], cost=2.0)]),
            makeClass(3, [makeNode(0, [2], cost=1.0)]),
        ]
        # Initial witness starts with the slow choice (cost=10.0)
        witness_hint = [
            {
                "cost": 12.0,
                "selection_map": {"1": 0, "2": 0, "3": 0},
                "order": [1, 2, 3],
                "eclass_to_buf": {"1": 1, "2": 2, "3": 3},
                "buffers": [],
            }
        ]
        problem = makeProblem(classes, 3, hints=witness_hint)
        # Custom selector that unfreezes node 2
        if cp_model is not None:
            selector = CustomDummySelector({2})
            solver = OrtoolsLnsSolver(problem, selector=selector)
            sol = solver.solve()
        else:
            sol = solveOrtools(problem)

        self.assertEqual(sol["solver"], "ortools_full")
        self.assertEqual(len(sol["extractions"]), 1)
        ext = sol["extractions"][0]
        # Verify LNS switched node 2 to fast enode (index 1)
        self.assertEqual(ext["selection_map"]["2"], 1)
        # Verify total cost decreased from 12 to 4 (1 + 2 + 1)
        self.assertLess(ext["cost"], 12.0)
        self.assertEqual(ext["cost"], 4.0)

    def testSolveOrtoolsDispatcher(self):
        classes = [
            makeClass(1, [makeNode(0, cost=1.0)]),
            makeClass(2, [makeNode(0, [1], cost=1.0)]),
        ]
        problem = makeProblem(classes, 2)
        sol = solveOrtools(problem)
        self.assertIn("extractions", sol)
        self.assertEqual(len(sol["extractions"]), 1)


if __name__ == "__main__":
    unittest.main()
