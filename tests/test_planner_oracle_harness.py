"""Check the independent oracle gate and its cache, including deliberate lies."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import planner_oracle_reference as reference
from tests.test_planner_oracle import checkCycleStrip, checkPropagation, checkSearch


class PlannerOracleHarnessTest(unittest.TestCase):
    def setUp(self):
        reference.configureCache(None)
        self.problem = reference.normalizeCase({
            "name": "harness", "root": "out",
            "nodes": {"out": {"alternatives": [{"cost": 2}, {"cost": 2}], "offset": [0, 1]}},
        })
        self.optimum = reference.enumerateOptimum(self.problem)

    def tearDown(self):
        reference.configureCache(None)

    def testPruneEverythingIsRejected(self):
        with patch("tests.test_planner_oracle.nativeCall", return_value={"consistent": False, "conflict": "prune all"}):
            with self.assertRaisesRegex(ValueError, "removed every optimum"):
                checkPropagation(self.problem, self.optimum, 1000)

    def testIncorrectBoundIsRejected(self):
        with patch("tests.test_planner_oracle.nativeCall", return_value={
            "consistent": True, "conflict": "", "lower_bound": 3, "problem": self.problem,
        }):
            with self.assertRaisesRegex(ValueError, "Invalid lower bound"):
                checkPropagation(self.problem, self.optimum, 1000)

    def testEqualOptimumRepresentativeIsAccepted(self):
        narrowed = deepcopy(self.problem)
        narrowed["buckets"][0]["nodes"][0]["selected"] = [2]
        narrowed["buckets"][0]["nodes"][0]["offset"] = [1]
        with patch("tests.test_planner_oracle.nativeCall", return_value={
            "consistent": True, "conflict": "", "lower_bound": 2, "problem": narrowed,
        }):
            checkPropagation(self.problem, self.optimum, 1000)

    def testOptimumLosingCycleStripIsRejected(self):
        stripped = deepcopy(self.problem)
        stripped["buckets"][0]["nodes"][0]["selected"] = []
        with patch("tests.test_planner_oracle.nativeCall", return_value={"problem": stripped, "removed": [2]}):
            with self.assertRaisesRegex(ValueError, "changed the optimum"):
                checkCycleStrip(self.problem, self.optimum, 1000, "fake")

    def testFakeIncumbentIsRejected(self):
        with patch("tests.test_planner_oracle.nativeCall", return_value={
            "solved": True, "exhausted": True, "cost": 2,
            "root_lower_bound": 2, "max_certified_bound": 2,
            "witness": {"cached": [], "buckets": [{"selected": [1], "order": [0], "offset": [999]}]},
        }):
            with self.assertRaisesRegex(ValueError, "Native incumbent is infeasible"):
                checkSearch(self.problem, self.optimum, 1000)

    def testCacheInvalidationAndFreshNativeChecks(self):
        with tempfile.TemporaryDirectory() as directory:
            reference.configureCache(Path(directory))
            first = reference.enumerateOptimum(self.problem)
            second = reference.enumerateOptimum(self.problem)
            self.assertEqual(first, second)
            self.assertEqual(reference.cacheStats(), {"hits": 1, "misses": 1})
            changed = deepcopy(self.problem)
            changed["buckets"][0]["nodes"][0]["alternatives"][0]["cost"] = 1
            self.assertEqual(reference.enumerateOptimum(changed)["cost"], 1)
            with patch.object(reference, "ORACLE_SOURCE_HASH", "changed-oracle"):
                self.assertEqual(reference.enumerateOptimum(self.problem)["cost"], 2)
            self.assertEqual(reference.cacheStats()["misses"], 3)
            # Cached truth does not cause a new candidate's verdict to be reused.
            with patch("tests.test_planner_oracle.nativeCall", return_value={"consistent": False, "conflict": "new bad candidate"}) as native:
                with self.assertRaisesRegex(ValueError, "removed every optimum"):
                    checkPropagation(self.problem, reference.enumerateOptimum(self.problem), 1000)
                self.assertEqual(native.call_count, 1)

    def testEmptySuiteCannotPass(self):
        from tests.test_planner_oracle import main
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.argv", ["oracle", "--cases-dir", directory]):
                with self.assertRaisesRegex(ValueError, "No oracle cases"):
                    main()

    def testUnknownFieldsAreRejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown fields"):
            reference.normalizeCase({"name": "typo", "root": "out", "nodes": {"out": {"engine": 1}}})


if __name__ == "__main__":
    unittest.main()
