"""Run with .venv/Scripts/python.exe -m unittest test_ortools_full.

With OR-Tools in the active environment these exercise the model directly.
Otherwise build test_ortools_full and use its production solver process bridge.
"""

import copy
import itertools
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    from ortools_full import OrtoolsSolver
except ImportError:
    OrtoolsSolver = None

CPU = {"type": 1, "idx": 1}
GPU = {"type": 3, "idx": 0}
PAGE = 4096


def makeNode(index=0, children=(), cost=1.0, **kwargs):
    return dict(enode_idx=index, children=list(children), cost=cost, **kwargs)


def makeClass(cid, enodes, pages=1, mem_space=None):
    return {
        "id": cid,
        "enodes": enodes,
        "size_bytes": pages * PAGE,
        "mem_space": mem_space or CPU,
    }


def makeProblem(classes, root, pages=10):
    return {
        "use_ortools_full": True,
        "print_progress": False,
        "max_time_seconds": 10,
        "num_workers": 1,
        "mem_caps": {"1:1": pages * PAGE},
        "full_bucket_idx": 0,
        "buckets": [{"bucket_idx": 0, "classes": classes, "root_eclass_id": root}],
    }


class FullSolverTests(unittest.TestCase):
    def solveProblem(self, problem):
        if OrtoolsSolver is not None:
            return OrtoolsSolver(problem).solve()
        binary = Path("tensor_graphs_cpp") / (
            "test_ortools_full.exe" if os.name == "nt" else "test_ortools_full"
        )
        if not binary.exists():
            self.fail(
                "Install the ortools extra or build test_ortools_full to run these tests"
            )
        with tempfile.TemporaryDirectory(prefix="ortools_full_") as directory:
            problem_path, solution_path = (
                Path(directory) / "problem.json",
                Path(directory) / "solution.json",
            )
            problem_path.write_text(json.dumps(problem), encoding="utf-8")
            process = subprocess.run(
                [
                    str(binary.resolve()),
                    "--solve-json",
                    str(problem_path),
                    str(solution_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if process.returncode:
                raise RuntimeError(process.stdout + process.stderr)
            return json.loads(solution_path.read_text(encoding="utf-8"))

    def checkSolution(self, problem, solution):
        self.assertEqual(solution["solver"], "ortools_full")
        self.assertEqual(len(solution["extractions"]), len(problem["buckets"]))
        for bucket, result in zip(problem["buckets"], solution["extractions"]):
            classes = {cls["id"]: cls for cls in bucket["classes"]}
            buffers = {buf["id"]: buf for buf in result["buffers"]}
            positions = {cid: pos for pos, cid in enumerate(result["order"])}
            self.assertIn(bucket["root_eclass_id"], positions)
            self.assertEqual(set(map(int, result["selection_map"])), set(positions))
            for cid, position in positions.items():
                enode = next(
                    e
                    for e in classes[cid]["enodes"]
                    if e["enode_idx"] == result["selection_map"][str(cid)]
                )
                for child in enode["children"]:
                    self.assertLess(positions[child], position)
                self.assertIn(result["eclass_to_buf"][str(cid)], buffers)
            for buf in buffers.values():
                if buf["mem_space"]["type"] == 0:
                    continue
                key = f"{buf['mem_space']['type']}:{buf['mem_space']['idx']}"
                self.assertLessEqual(
                    buf["offset"] + buf["size"], problem["mem_caps"].get(key, 2**60)
                )
                self.assertEqual(buf["offset"] % PAGE, 0)
            # Independently reconstruct actual accesses to each allocation and
            # each engine, including readers through aliased views.
            spans = {buf_id: [float("inf"), 0] for buf_id in buffers}
            engines = {}
            for cid in result["order"]:
                enode = next(
                    e
                    for e in classes[cid]["enodes"]
                    if e["enode_idx"] == result["selection_map"][str(cid)]
                )
                timing = result["schedule"][str(cid)]
                owner = result["eclass_to_buf"][str(cid)]
                spans[owner][0] = min(spans[owner][0], timing["start"])
                for used_id in [cid] + enode["children"]:
                    used_owner = result["eclass_to_buf"][str(used_id)]
                    spans[used_owner][1] = max(spans[used_owner][1], timing["end"])
                if timing["end"] > timing["start"]:
                    for engine in enode.get("engines", [{"type": 0, "idx": 0}]):
                        engines.setdefault((engine["type"], engine["idx"]), []).append(
                            (timing["start"], timing["end"])
                        )
            for buf_id, buf in buffers.items():
                if buf["end"] == 2**32 - 1:
                    spans[buf_id] = [0, float("inf")]
            spans[result["eclass_to_buf"][str(bucket["root_eclass_id"])]][1] = float(
                "inf"
            )
            for first, second in itertools.combinations(buffers.values(), 2):
                if (
                    first["mem_space"] != second["mem_space"]
                    or first["mem_space"]["type"] == 0
                ):
                    continue
                if max(first["offset"], second["offset"]) < min(
                    first["offset"] + first["size"], second["offset"] + second["size"]
                ):
                    a, b = spans[first["id"]], spans[second["id"]]
                    self.assertGreaterEqual(
                        max(a[0], b[0]),
                        min(a[1], b[1]),
                        "Live buffers overlap in memory",
                    )
            for intervals in engines.values():
                for first, second in itertools.combinations(intervals, 2):
                    self.assertGreaterEqual(
                        max(first[0], second[0]),
                        min(first[1], second[1]),
                        "Engine double-booked",
                    )

    def testWorkspacePressureChangesCacheSelection(self):
        problem = makeProblem(
            [
                makeClass(1, [makeNode(cost=8), makeNode(1, cost=0, is_cache=True)]),
                makeClass(2, [makeNode(children=[1])]),
                makeClass(3, [makeNode(children=[2])], pages=3),
                makeClass(4, [makeNode(children=[3])]),
                makeClass(5, [makeNode(children=[2, 4])]),
            ],
            5,
            pages=5,
        )
        problem["candidates"] = [
            {"logical_id": 10, "mem_space": CPU, "size_bytes": PAGE}
        ]
        problem["buckets"][0]["eclass_to_logical"] = {"1": 10}
        second = copy.deepcopy(problem["buckets"][0])
        second.update(bucket_idx=1, weight=100, clean_eclasses=[1])
        problem["buckets"].append(second)
        solution = self.solveProblem(problem)
        self.checkSolution(problem, solution)
        self.assertEqual(solution["cached_nodes"], [])

    def testDirtyScatterUsesPersistentDeviceCache(self):
        problem = makeProblem(
            [
                makeClass(2, [makeNode(cost=8)], mem_space=GPU),
                makeClass(3, [makeNode(children=[2])], mem_space=GPU),
            ],
            3,
        )
        problem["mem_caps"]["3:0"] = 3 * PAGE
        problem["candidates"] = [
            {
                "logical_id": 20,
                "mem_space": CPU,
                "mem_spaces": [CPU, GPU],
                "size_bytes": PAGE,
            }
        ]
        problem["buckets"][0]["eclass_to_logical"] = {"2": 20}
        second = copy.deepcopy(problem["buckets"][0])
        second.update(bucket_idx=1, weight=10)
        second["classes"].append(makeClass(1, [makeNode()], mem_space=GPU))
        second["classes"][0]["enodes"].extend(
            [
                makeNode(1, cost=0, is_cache=True),
                makeNode(2, [1], is_scatter=True),
            ]
        )
        problem["buckets"].append(second)
        solution = self.solveProblem(problem)
        self.checkSolution(problem, solution)
        self.assertEqual(
            solution["cached_nodes"], [{"logical_id": 20, "mem_space": GPU}]
        )
        first, second = solution["extractions"]
        self.assertEqual(second["selection_map"]["2"], 2)
        self.assertEqual(first["eclass_to_buf"]["2"], second["eclass_to_buf"]["2"])

    def testStorageViewsDoNotConsumeArena(self):
        storage = {"type": 0, "idx": 0}
        problem = makeProblem(
            [
                makeClass(1, [makeNode(cost=0, is_input=True)], mem_space=storage),
                makeClass(
                    2, [makeNode(children=[1], cost=0, is_view=True)], mem_space=storage
                ),
                makeClass(3, [makeNode(children=[2])]),
            ],
            3,
            pages=1,
        )
        problem["mem_caps"]["0:0"] = 0
        solution = self.solveProblem(problem)
        self.checkSolution(problem, solution)
        owners = solution["extractions"][0]["eclass_to_buf"]
        self.assertEqual(owners["1"], owners["2"])

    def testMemoryChangesExtraction(self):
        # The fast branch requires 4 pages at once; only the slower branch fits.
        problem = makeProblem(
            [
                makeClass(1, [makeNode(cost=1)], pages=3),
                makeClass(2, [makeNode(cost=3)]),
                makeClass(3, [makeNode(4, [1]), makeNode(9, [2])]),
            ],
            3,
            pages=2,
        )
        solution = self.solveProblem(problem)
        self.checkSolution(problem, solution)
        self.assertEqual(solution["extractions"][0]["selection_map"]["3"], 9)

    def testRejectsCyclesAndInvalidCosts(self):
        problem = makeProblem(
            [
                makeClass(
                    1,
                    [
                        makeNode(0, [2], cost=0),
                        makeNode(2, cost=2),
                        makeNode(8, cost=None),
                    ],
                ),
                makeClass(2, [makeNode(0, [1], cost=0)]),
            ],
            1,
        )
        solution = self.solveProblem(problem)
        self.checkSolution(problem, solution)
        self.assertEqual(solution["extractions"][0]["selection_map"], {"1": 2})

    def testInfeasibleFailsWithoutPartialPlan(self):
        problem = makeProblem([makeClass(1, [makeNode()], pages=2)], 1, pages=1)
        with self.assertRaisesRegex(RuntimeError, "INFEASIBLE"):
            self.solveProblem(problem)

    def testInplaceIsJointDecision(self):
        problem = makeProblem(
            [
                makeClass(1, [makeNode()]),
                makeClass(2, [makeNode(children=[1], safe_inplace_idxs=[0])]),
            ],
            2,
            pages=1,
        )
        solution = self.solveProblem(problem)
        self.checkSolution(problem, solution)
        owners = solution["extractions"][0]["eclass_to_buf"]
        self.assertEqual(owners["1"], owners["2"])

    def testInputCannotBeOverwritten(self):
        problem = makeProblem(
            [
                makeClass(1, [makeNode(cost=0, is_input=True)]),
                makeClass(2, [makeNode(children=[1], safe_inplace_idxs=[0])]),
            ],
            2,
            pages=1,
        )
        with self.assertRaisesRegex(RuntimeError, "INFEASIBLE"):
            self.solveProblem(problem)

    def testViewKeepsBaseAliveForLaterReader(self):
        problem = makeProblem(
            [
                makeClass(1, [makeNode()]),
                makeClass(2, [makeNode(children=[1], cost=0, is_view=True)], pages=8),
                makeClass(3, [makeNode(children=[1], safe_inplace_idxs=[0])]),
                makeClass(4, [makeNode(children=[2, 3])]),
            ],
            4,
            pages=3,
        )
        solution = self.solveProblem(problem)
        self.checkSolution(problem, solution)
        owners = solution["extractions"][0]["eclass_to_buf"]
        self.assertEqual(owners["1"], owners["2"])
        self.assertNotEqual(owners["1"], owners["3"])

    def testScheduleExploitsIndependentEngines(self):
        problem = makeProblem(
            [
                makeClass(1, [makeNode(cost=4, engines=[{"type": 0, "idx": 0}])]),
                makeClass(2, [makeNode(cost=4, engines=[{"type": 2, "idx": 0}])]),
                makeClass(3, [makeNode(children=[1, 2], cost=1)]),
            ],
            3,
        )
        solution = self.solveProblem(problem)
        self.checkSolution(problem, solution)
        self.assertEqual(solution["extractions"][0]["cost"], 5)

    def testMemoryForcesSerialization(self):
        problem = makeProblem(
            [
                makeClass(
                    1, [makeNode(cost=4, engines=[{"type": 0, "idx": 0}])], pages=3
                ),
                makeClass(2, [makeNode(children=[1], cost=1)]),
                makeClass(
                    3, [makeNode(cost=4, engines=[{"type": 2, "idx": 0}])], pages=3
                ),
                makeClass(
                    4, [makeNode(children=[3], cost=1, engines=[{"type": 2, "idx": 0}])]
                ),
                makeClass(5, [makeNode(children=[2, 4])]),
            ],
            5,
            pages=5,
        )
        solution = self.solveProblem(problem)
        self.checkSolution(problem, solution)
        result = solution["extractions"][0]
        self.assertEqual(result["cost"], 11)
        buffers = {buf["id"]: buf for buf in result["buffers"]}
        first, second = (buffers[result["eclass_to_buf"][str(cid)]] for cid in (1, 3))
        self.assertLess(
            max(first["offset"], second["offset"]),
            min(first["offset"] + first["size"], second["offset"] + second["size"]),
        )

    def testPersistentCacheAndPreallocatedReservations(self):
        problem = makeProblem(
            [
                makeClass(1, [makeNode(cost=0, is_input=True)]),
                makeClass(
                    2,
                    [
                        makeNode(children=[1], cost=5),
                        makeNode(1, cost=0, is_cache=True),
                    ],
                ),
                makeClass(3, [makeNode(children=[2])]),
            ],
            3,
            pages=3,
        )
        problem["preallocated_buffers"] = [
            {
                "logical_id": 10,
                "buffer_id": 71,
                "mem_space": CPU,
                "size": PAGE,
                "offset": 0,
            }
        ]
        problem["candidates"] = [
            {"logical_id": 20, "mem_space": CPU, "size_bytes": PAGE}
        ]
        problem["buckets"][0]["eclass_to_logical"] = {"1": 10, "2": 20}
        problem["buckets"][0]["clean_eclasses"] = [
            2
        ]  # Still cannot read an uninitialized cache.
        second = copy.deepcopy(problem["buckets"][0])
        second.update(bucket_idx=1, weight=10)
        problem["buckets"].append(second)
        solution = self.solveProblem(problem)
        self.checkSolution(problem, solution)
        self.assertEqual(
            solution["cached_nodes"], [{"logical_id": 20, "mem_space": CPU}]
        )
        first, second = solution["extractions"]
        self.assertEqual(first["selection_map"]["2"], 0)
        self.assertEqual(second["selection_map"]["2"], 1)
        self.assertEqual(first["eclass_to_buf"]["2"], second["eclass_to_buf"]["2"])
        cache_id = first["eclass_to_buf"]["2"]
        for result in solution["extractions"]:
            cache = next(buf for buf in result["buffers"] if buf["id"] == cache_id)
            self.assertEqual(cache["end"], 2**32 - 1)
            self.assertGreaterEqual(cache["offset"], PAGE)

    def testInactiveInputStillReservesMemory(self):
        problem = makeProblem([makeClass(1, [makeNode()], pages=2)], 1, pages=2)
        problem["preallocated_buffers"] = [
            {
                "logical_id": 10,
                "buffer_id": 71,
                "mem_space": CPU,
                "size": PAGE,
                "offset": 0,
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "INFEASIBLE"):
            self.solveProblem(problem)

    def testIndependentMemorySpaces(self):
        problem = makeProblem(
            [
                makeClass(1, [makeNode()], mem_space=GPU),
                makeClass(2, [makeNode(children=[1])]),
            ],
            2,
            pages=1,
        )
        problem["mem_caps"]["3:0"] = PAGE
        solution = self.solveProblem(problem)
        self.checkSolution(problem, solution)
        self.assertEqual(
            [buf["offset"] for buf in solution["extractions"][0]["buffers"]], [0, 0]
        )


if __name__ == "__main__":
    unittest.main()
