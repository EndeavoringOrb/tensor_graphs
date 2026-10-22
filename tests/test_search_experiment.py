import unittest

from tests.search_experiment import parseMetrics


class SearchExperimentTests(unittest.TestCase):
    def testSupervisorElapsedTakesPrecedence(self):
        output = (
            "[SearchEngine] New best cost: 7 at iteration 0 "
            "event-epoch=1011 program-deadline=1027\n"
            "[Program gap] first-incumbent-elapsed=11 final-incumbent=7\n"
        )
        self.assertEqual(parseMetrics(output, 30)["first_incumbent"],
                         {"cost": 7, "program_elapsed": 11})

    def testProgramDeadlineWithoutSupervisor(self):
        output = ("[SearchEngine] New best cost: 7 at iteration 0 "
                  "event-epoch=1011 program-deadline=1030\n")
        self.assertEqual(parseMetrics(output, 30)["first_incumbent"],
                         {"cost": 7, "program_elapsed": 11})

    def testKilledCompileHasNoElapsedOrigin(self):
        output = (
            "[SearchEngine] New best cost: 7 at iteration 0 "
            "event-epoch=1029 program-deadline=1027\n"
            "[Gemma compile] supervisor elapsed=30.1 budget=30 exit=-9 timed_out=true\n"
        )
        self.assertEqual(parseMetrics(output, 30)["first_incumbent"],
                         {"cost": 7, "program_elapsed": None})

    def testFinalCostUsesSupervisor(self):
        output = (
            "[SearchEngine] New best cost: 7 at iteration 0 "
            "event-epoch=1011 program-deadline=1027\n"
            "[Program gap] first-incumbent-elapsed=11 final-incumbent=5\n"
        )
        self.assertEqual(parseMetrics(output, 30)["best_incumbent_cost"], 5)


if __name__ == "__main__":
    unittest.main()
