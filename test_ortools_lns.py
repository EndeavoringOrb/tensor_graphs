"""Unit tests for iterative subgraph LNS solver and modular neighborhood selectors.

Run with:
.venv/Scripts/python.exe -m unittest test_ortools_lns
"""

import unittest
from unittest.mock import patch
from typing import Any, Dict, List

from ortools_lns import (
    CacheNeighborhoodSelector,
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
    def testCacheSelectorDiamondClosureOmitsCommonChild(self):
        classes = [
            makeClass(
                1,
                [makeNode(0, [2]), makeNode(1, [3])],
            ),
            makeClass(2, [makeNode(0, [4])]),
            makeClass(3, [makeNode(0, [4])]),
            makeClass(4, [makeNode(0)]),
        ]
        problem = makeProblem(classes, 1)
        problem["disable_caching"] = True
        assignments = {
            "selection:0:1": 0,
            "selection:0:2": 0,
            "selection:0:3": None,
            "selection:0:4": 0,
        }
        groups = {
            f"selection:0:{eclass_id}": {
                "kind": "selection",
                "selectable": eclass_id == 1,
                "choice_count": 2 if eclass_id == 1 else 1,
            }
            for eclass_id in range(1, 5)
        }
        context = {
            "model": problem,
            "incumbent": {
                "primary_assignments": assignments,
                "extractions": [{"selection_map": {"1": 0, "2": 0, "4": 0}}],
            },
            "metadata": {"groups": groups},
        }

        selector = CacheNeighborhoodSelector()
        neighborhood = selector.selectNeighborhood(context)

        self.assertEqual(
            neighborhood,
            {"selection:0:1", "selection:0:2", "selection:0:3"},
        )
        self.assertNotIn("selection:0:4", neighborhood)

    def testCacheSelectorRanksScatterAndConsumesOneCandidate(self):
        scatter = makeClass(
            1,
            [
                makeNode(0),
                makeNode(1, [3], is_scatter=True, cost=0),
            ],
        )
        scatter.update(shape=[100])
        update = makeClass(3, [makeNode(0)])
        update.update(shape=[10])
        pure = makeClass(
            2,
            [makeNode(0), makeNode(1, cost=0, is_cache=True)],
        )
        pure.update(shape=[100])
        problem = makeProblem([scatter, pure, update], 1)
        problem["candidates"] = [
            {
                "base_eclass_id": 1,
                "mem_space": CPU,
                "size_bytes": PAGE,
                "raw_size_bytes": PAGE,
            },
            {
                "base_eclass_id": 2,
                "mem_space": CPU,
                "size_bytes": PAGE,
                "raw_size_bytes": PAGE,
            },
        ]
        assignments = {
            "cache:1": 0,
            "cache:2": 0,
            "selection:0:1": 0,
            "selection:0:2": 0,
            "selection:0:3": 0,
        }
        groups = {
            "cache:1": {"kind": "cache", "base_eclass_id": 1},
            "cache:2": {"kind": "cache", "base_eclass_id": 2},
            "selection:0:1": {"kind": "selection", "selectable": True},
            "selection:0:2": {"kind": "selection", "selectable": True},
            "selection:0:3": {"kind": "selection", "selectable": False},
        }
        context = {
            "model": problem,
            "incumbent": {
                "primary_assignments": assignments,
                "extractions": [{"selection_map": {"1": 0, "2": 0, "3": 0}}],
            },
            "metadata": {"groups": groups},
        }

        selector = CacheNeighborhoodSelector()
        first = selector.selectNeighborhood(context)
        first_targets = selector.getCandidateFixings()
        second = selector.selectNeighborhood(context)
        second_targets = selector.getCandidateFixings()

        self.assertIn("cache:1", first)
        self.assertEqual(first_targets, {"cache:1": 1})
        self.assertIn("cache:2", second)
        self.assertEqual(second_targets, {"cache:2": 1})

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

    def testLnsWithoutMaxTimeRunsNeighborhoodsAfterFirstFeasibleSolve(self):
        problem = makeProblem([makeClass(1, [makeNode(cost=1.0)])], 1)
        problem.pop("max_time_seconds")

        with patch("ortools_lns.OrtoolsSolver") as solver_type:
            solver_type.return_value.solve.return_value = {
                "solver": "ortools_full",
                "objective": 1.0,
            }
            solver_type.return_value.getNeighborhoodMetadata.return_value = {
                "groups": {}
            }
            solution = OrtoolsLnsSolver(
                problem, selector=CustomDummySelector(set())
            ).solve()

        solver_problem = solver_type.call_args.args[0]
        self.assertNotIn("max_time_seconds", solver_problem)
        self.assertTrue(solver_problem["stop_after_first_solution"])
        self.assertEqual(solution["lns"]["iterations"], 1)
        self.assertEqual(solution["lns"]["outcomes"], {"empty": 1})

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
