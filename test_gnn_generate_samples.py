"""Tests for GNN neighborhood sample generation."""

import unittest
from unittest.mock import patch

from lns_selectors.gnn.generate_samples import generateSamples


class FakeOrtoolsSolver:
    """Small solver double for testing repair outcomes."""

    init_kwargs = []

    def __init__(self, problem_data, **kwargs):
        self.init_kwargs.append(kwargs)

    def getNeighborhoodMetadata(self):
        return {
            "feature_dim": 1,
            "groups": {
                "selection:0:1": {
                    "selectable": True,
                    "choice_count": 2,
                    "features": [2.0],
                }
            },
            "adjacency": {"selection:0:1": []},
        }

    def solve(self):
        if len(self.init_kwargs) == 1:
            return {
                "status": "OPTIMAL",
                "objective": 10.0,
                "primary_assignments": {"selection:0:1": 0},
            }
        raise RuntimeError("OR-Tools full found no feasible joint plan: INFEASIBLE")


class GnnGenerateSamplesTests(unittest.TestCase):
    def testInfeasibleRepairIsRecordedAsNoImprovement(self):
        FakeOrtoolsSolver.init_kwargs = []
        problem_data = {"buckets": [{"bucket_idx": 0}]}

        with patch(
            "lns_selectors.gnn.generate_samples.OrtoolsSolver",
            FakeOrtoolsSolver,
        ):
            samples = generateSamples(
                problem_data,
                sample_count=1,
                neighborhood_size=1,
                repair_time_seconds=1.0,
                seed=0,
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["current_runtime"], 10.0)
        self.assertEqual(samples[0]["solution_runtime"], 10.0)
        self.assertEqual(samples[0]["relative_runtime"], 1.0)
        self.assertNotIn("must_change_groups", FakeOrtoolsSolver.init_kwargs[1])


if __name__ == "__main__":
    unittest.main()
