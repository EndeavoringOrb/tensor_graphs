"""Unit tests for the pure schedule CP-SAT solver."""

import unittest
from ortools_pure_schedule import PureScheduleSolver, solveSchedule


CPU = {"type": 1, "idx": 1}
GPU = {"type": 3, "idx": 0}
PAGE = 4096


def makeNode(index=0, children=(), cost=1.0, **kwargs):
    return dict(enode_idx=index, children=list(children), cost=cost, **kwargs)


def makeClass(cid, enodes, base_id=None, mem_space=None):
    b_id = base_id if base_id is not None else cid
    enodes = [
        dict(enode, base_eclass_id=b_id) if enode.get("is_cache") else enode
        for enode in enodes
    ]
    return {
        "id": cid,
        "base_eclass_id": b_id,
        "enodes": enodes,
        "size_bytes": PAGE,
        "raw_size_bytes": PAGE,
        "mem_space": mem_space or CPU,
    }


def makeProblem(classes, root, candidates=None, full_idx=0, buckets=None):
    if buckets is None:
        buckets = [{"bucket_idx": 0, "classes": classes, "root_eclass_id": root}]
    return {
        "disable_caching": False,
        "print_progress": False,
        "max_time_seconds": 5,
        "num_workers": 2,
        "full_bucket_idx": full_idx,
        "candidates": candidates or [],
        "buckets": buckets,
    }


class TestPureScheduleSolver(unittest.TestCase):
    def testSingleNode(self):
        classes = [makeClass(1, [makeNode(cost=5.0)])]
        prob = makeProblem(classes, 1)
        res = solveSchedule(prob)
        self.assertEqual(res["status"], "OPTIMAL")
        self.assertAlmostEqual(res["objective_ms"], 5.0, places=2)
        self.assertEqual(res["extractions"][0]["order"], [1])

    def testPrecedenceChain(self):
        # 1 -> 2 -> 3
        classes = [
            makeClass(1, [makeNode(cost=3.0)]),
            makeClass(2, [makeNode(children=[1], cost=4.0)]),
            makeClass(3, [makeNode(children=[2], cost=2.0)]),
        ]
        prob = makeProblem(classes, 3)
        res = solveSchedule(prob)
        self.assertEqual(res["status"], "OPTIMAL")
        # On single engine (CPU): 3 + 4 + 2 = 9
        self.assertAlmostEqual(res["objective_ms"], 9.0, places=2)
        order = res["extractions"][0]["order"]
        self.assertLess(order.index(1), order.index(2))
        self.assertLess(order.index(2), order.index(3))

    def testMultiEngineParallelism(self):
        # Two independent nodes on different engines
        # Node 1 on Engine A (duration 10), Node 2 on Engine B (duration 7)
        # Root 3 joins them (duration 1)
        engine_a = [{"type": 1, "idx": 0}]
        engine_b = [{"type": 2, "idx": 0}]
        classes = [
            makeClass(1, [makeNode(cost=10.0, engines=engine_a)]),
            makeClass(2, [makeNode(cost=7.0, engines=engine_b)]),
            makeClass(3, [makeNode(children=[1, 2], cost=1.0)]),
        ]
        prob = makeProblem(classes, 3)
        res = solveSchedule(prob)
        self.assertEqual(res["status"], "OPTIMAL")
        # Node 1 and Node 2 should run concurrently: max(10, 7) + 1 = 11.0 ms!
        self.assertAlmostEqual(res["objective_ms"], 11.0, places=2)

    def testAlternativeEnodeSelection(self):
        # Class 1 has two enode implementations: fast (cost 2.0) vs slow (cost 8.0)
        classes = [
            makeClass(1, [makeNode(index=0, cost=8.0), makeNode(index=1, cost=2.0)])
        ]
        prob = makeProblem(classes, 1)
        res = solveSchedule(prob)
        self.assertEqual(res["status"], "OPTIMAL")
        self.assertAlmostEqual(res["objective_ms"], 2.0, places=2)
        self.assertEqual(res["extractions"][0]["selection_map"]["1"], 1)

    def testCacheReadSelection(self):
        # Candidate base class 10 can be cached.
        # Bucket 0 is full_bucket (computes class 1, cost 10.0)
        # Bucket 1 is decode bucket (has choice: compute class 1 cost 10.0, or read cache cost 0.0)
        cand = {"base_eclass_id": 10, "clean_buckets": [1]}
        b0_classes = [makeClass(1, [makeNode(cost=10.0)], base_id=10)]
        b1_classes = [
            makeClass(
                1,
                [
                    makeNode(index=0, cost=10.0),
                    makeNode(index=1, cost=0.0, is_cache=True, base_eclass_id=10),
                ],
                base_id=10,
            )
        ]
        prob = makeProblem(
            [],
            1,
            candidates=[cand],
            full_idx=0,
            buckets=[
                {"bucket_idx": 0, "classes": b0_classes, "root_eclass_id": 1, "clean_eclasses": []},
                {"bucket_idx": 1, "classes": b1_classes, "root_eclass_id": 1, "clean_eclasses": [1], "weight": 1.0},
            ],
        )
        res = solveSchedule(prob)
        self.assertEqual(res["status"], "OPTIMAL")
        # Bucket 1 should pick cache (enode_idx 1, cost 0.0)
        self.assertIn(10, res["cached_base_eclass_ids"])
        self.assertAlmostEqual(res["objective_ms"], 0.0, places=2)
        self.assertEqual(res["extractions"][1]["selection_map"]["1"], 1)

    def testViewNodesHaveZeroDuration(self):
        # 1 (cost 5.0) -> 2 (view, cost 1.0, is_view=True) -> 3 (cost 3.0, consumes 2)
        classes = [
            makeClass(1, [makeNode(cost=5.0)]),
            makeClass(2, [makeNode(children=[1], cost=1.0, is_view=True)]),
            makeClass(3, [makeNode(children=[2], cost=3.0)]),
        ]
        prob = makeProblem(classes, 3)
        res = solveSchedule(prob)
        self.assertEqual(res["status"], "OPTIMAL")
        # View has 0 duration, so total = 5.0 + 0.0 + 3.0 = 8.0 ms
        self.assertAlmostEqual(res["objective_ms"], 8.0, places=2)
        schedule = res["extractions"][0]["schedule"]
        self.assertEqual(schedule["2"]["duration_ms"], 0.0)
        self.assertEqual(schedule["2"]["start_ms"], 5.0)
        self.assertEqual(schedule["2"]["end_ms"], 5.0)
        self.assertEqual(schedule["3"]["start_ms"], 5.0)
        self.assertEqual(schedule["3"]["end_ms"], 8.0)

    def testDeadClassPruning(self):
        # Class 99 is disconnected and dead, shouldn't be in schedule
        classes = [
            makeClass(1, [makeNode(cost=4.0)]),
            makeClass(2, [makeNode(children=[1], cost=2.0)]),
            makeClass(99, [makeNode(cost=100.0)]),
        ]
        prob = makeProblem(classes, 2)
        res = solveSchedule(prob)
        self.assertEqual(res["status"], "OPTIMAL")
        self.assertAlmostEqual(res["objective_ms"], 6.0, places=2)
        self.assertNotIn("99", res["extractions"][0]["selection_map"])


if __name__ == "__main__":
    unittest.main()
