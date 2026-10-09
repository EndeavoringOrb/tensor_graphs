import os
import sys
import unittest

# Ensure repo root and utils are importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "utils"))

import tensor_graphs
from analyze_performance import calculate_makespan as analyze_calculate_makespan


class TestMakespan(unittest.TestCase):
    def test_bindings_exist_and_consistent(self):
        """Test all binding aliases exist on module and CompiledGraph."""
        self.assertTrue(hasattr(tensor_graphs, "calculate_makespan"))
        self.assertTrue(hasattr(tensor_graphs, "calculateMakespan"))
        self.assertTrue(hasattr(tensor_graphs, "evaluate_makespan"))
        self.assertTrue(hasattr(tensor_graphs, "evaluateMakespan"))
        self.assertTrue(hasattr(tensor_graphs, "get_cost"))
        self.assertTrue(hasattr(tensor_graphs, "getCost"))

        cg = tensor_graphs.CompiledGraph()
        self.assertTrue(hasattr(cg, "cost"))
        self.assertTrue(hasattr(cg, "get_cost"))
        self.assertTrue(hasattr(cg, "getCost"))
        self.assertTrue(hasattr(cg, "evaluate_makespan"))
        self.assertTrue(hasattr(cg, "evaluateMakespan"))
        self.assertTrue(hasattr(cg, "calculate_makespan"))
        self.assertTrue(hasattr(cg, "calculateMakespan"))

    def test_empty_instructions_fallback(self):
        """Empty instructions should fallback to sum of node_costs."""
        node_costs = {1: 3.5, 2: 4.5}
        self.assertAlmostEqual(tensor_graphs.calculate_makespan([], node_costs), 8.0)
        self.assertAlmostEqual(tensor_graphs.get_cost([], node_costs), 8.0)
        self.assertAlmostEqual(tensor_graphs.evaluate_makespan([], node_costs), 8.0)
        self.assertAlmostEqual(analyze_calculate_makespan([], node_costs), 8.0)

    def test_sequential_single_engine(self):
        """Sequential execution on a single engine."""
        inst1 = {"eclassId": 1, "children": [], "engines": [{"idx": 0, "type": 0}], "outBuffer": {"id": 10}}
        inst2 = {"eclassId": 2, "children": [1], "engines": [{"idx": 0, "type": 0}], "inBuffers": [{"id": 10}], "outBuffer": {"id": 20}}
        instructions = [inst1, inst2]
        node_costs = {1: 10.0, 2: 5.0}

        cost_tg = tensor_graphs.calculate_makespan(instructions, node_costs)
        cost_eval = tensor_graphs.evaluate_makespan(instructions, node_costs)
        cost_get = tensor_graphs.get_cost(instructions, node_costs)
        cost_an = analyze_calculate_makespan(instructions, node_costs)

        self.assertAlmostEqual(cost_tg, 15.0)
        self.assertAlmostEqual(cost_eval, 15.0)
        self.assertAlmostEqual(cost_get, 15.0)
        self.assertAlmostEqual(cost_an, 15.0)

    def test_parallel_engines(self):
        """Parallel execution across two engines with a join."""
        # inst_1 on CPU: 10.0ms
        inst1 = {"eclassId": 1, "children": [], "engines": [{"idx": 0, "type": 0}], "outBuffer": {"id": 10}}
        # inst_2 on GPU: 6.0ms
        inst2 = {"eclassId": 2, "children": [], "engines": [{"idx": 0, "type": 1}], "outBuffer": {"id": 20}}
        # inst_3 on CPU: 4.0ms, depends on inst_1 and inst_2
        inst3 = {"eclassId": 3, "children": [1, 2], "engines": [{"idx": 0, "type": 0}], "inBuffers": [{"id": 10}, {"id": 20}], "outBuffer": {"id": 30}}
        instructions = [inst1, inst2, inst3]
        node_costs = {1: 10.0, 2: 6.0, 3: 4.0}

        cost_tg = tensor_graphs.calculate_makespan(instructions, node_costs)
        cost_eval = tensor_graphs.evaluate_makespan(instructions, node_costs)
        cost_get = tensor_graphs.get_cost(instructions, node_costs)
        cost_an = analyze_calculate_makespan(instructions, node_costs)

        # inst_1: 0..10 on CPU
        # inst_2: 0..6 on GPU
        # inst_3: max(10, 6, engine_free(10)) => starts at 10, finishes at 14 on CPU
        self.assertAlmostEqual(cost_tg, 14.0)
        self.assertAlmostEqual(cost_eval, 14.0)
        self.assertAlmostEqual(cost_get, 14.0)
        self.assertAlmostEqual(cost_an, 14.0)

    def test_independent_work_on_producer_engine_no_phantom_blocking(self):
        """
        Critical regression: Independent task on producer engine must NOT
        falsely serialize a downstream consumer on another engine.
        """
        # inst_1 on CPU: 10.0ms (t=0..10), produces Buffer 10
        inst1 = {"eclassId": 1, "children": [], "engines": [{"idx": 0, "type": 0}], "outBuffer": {"id": 10}}
        # inst_2 on CPU: 20.0ms (t=10..30), independent of inst_1, produces Buffer 20
        inst2 = {"eclassId": 2, "children": [], "engines": [{"idx": 0, "type": 0}], "outBuffer": {"id": 20}}
        # inst_3 on GPU: 5.0ms, depends ONLY on inst_1
        inst3 = {"eclassId": 3, "children": [1], "engines": [{"idx": 0, "type": 1}], "inBuffers": [{"id": 10}], "outBuffer": {"id": 30}}

        instructions = [inst1, inst2, inst3]
        node_costs = {1: 10.0, 2: 20.0, 3: 5.0}

        cost_tg = tensor_graphs.calculate_makespan(instructions, node_costs)
        cost_eval = tensor_graphs.evaluate_makespan(instructions, node_costs)
        cost_get = tensor_graphs.get_cost(instructions, node_costs)
        cost_an = analyze_calculate_makespan(instructions, node_costs)

        # inst_1 finishes at 10.0 on CPU.
        # inst_3 on GPU only depends on inst_1, so starts at 10.0 and finishes at 15.0 on GPU.
        # inst_2 finishes at 30.0 on CPU.
        # Total makespan should be max(30.0, 15.0) = 30.0ms.
        # (The buggy model gave 35.0ms because it looked at CPU engine_finish which was 30.0).
        self.assertAlmostEqual(cost_tg, 30.0)
        self.assertAlmostEqual(cost_eval, 30.0)
        self.assertAlmostEqual(cost_get, 30.0)
        self.assertAlmostEqual(cost_an, 30.0)

    def test_view_buffer_aliasing_with_interleaved_producer_work(self):
        """
        View dependency via buffer aliasing with interleaved work on producer engine.
        """
        # inst_1 on CPU: 10.0ms, produces Buffer 100
        inst1 = {"eclassId": 1, "children": [], "engines": [{"idx": 0, "type": 0}], "outBuffer": {"id": 100}}
        # inst_2 on CPU: 20.0ms, independent, produces Buffer 200
        inst2 = {"eclassId": 2, "children": [], "engines": [{"idx": 0, "type": 0}], "outBuffer": {"id": 200}}
        # inst_3 on GPU: 5.0ms, child is EClass 4 (view of EClass 1, shares Buffer 100, not in instructions)
        inst3 = {"eclassId": 3, "children": [4], "engines": [{"idx": 0, "type": 1}], "inBuffers": [{"id": 100}], "outBuffer": {"id": 300}}

        instructions = [inst1, inst2, inst3]
        node_costs = {1: 10.0, 2: 20.0, 3: 5.0}

        cost_tg = tensor_graphs.calculate_makespan(instructions, node_costs)
        cost_eval = tensor_graphs.evaluate_makespan(instructions, node_costs)
        cost_get = tensor_graphs.get_cost(instructions, node_costs)
        cost_an = analyze_calculate_makespan(instructions, node_costs)

        # Buffer 100 was finished at 10.0ms.
        # inst_3 starts at 10.0 and finishes at 15.0ms on GPU.
        # Total makespan is 30.0ms.
        self.assertAlmostEqual(cost_tg, 30.0)
        self.assertAlmostEqual(cost_eval, 30.0)
        self.assertAlmostEqual(cost_get, 30.0)
        self.assertAlmostEqual(cost_an, 30.0)

    def test_compiled_graph_methods(self):
        """CompiledGraph methods (cost, get_cost, evaluate_makespan) match each other."""
        cg = tensor_graphs.CompiledGraph()
        cg.nodeCosts = {tensor_graphs.EClassId(1): 10.0, tensor_graphs.EClassId(2): 20.0, tensor_graphs.EClassId(3): 5.0}
        # With empty instructions, cost is sum of nodeCosts
        self.assertAlmostEqual(cg.cost(), 35.0)
        self.assertAlmostEqual(cg.get_cost(), 35.0)
        self.assertAlmostEqual(cg.evaluate_makespan(), 35.0)
        self.assertAlmostEqual(cg.calculate_makespan(), 35.0)
        self.assertAlmostEqual(tensor_graphs.calculate_makespan(cg), 35.0)

    def test_cache_file_makespan_consistency(self):
        """Verify real cached compiled graphs have identical makespan across all interfaces."""
        cache_file = os.path.join(REPO_ROOT, "versions", "2", "bench_gemma-3-270m-pp512-tg128.bin")
        if not os.path.exists(cache_file):
            self.skipTest(f"Cache file {cache_file} does not exist")

        from binary import load_cache_file
        cache_entries = load_cache_file(cache_file)
        buckets = [e for e in cache_entries if e.get("type") == "compiled_bucket"]
        self.assertGreater(len(buckets), 0)

        for idx, entry in enumerate(buckets):
            graph = entry["graph"]
            instructions = graph["instructions"]
            node_costs = graph.get("nodeCosts", {})

            cost_tg = tensor_graphs.calculate_makespan(instructions, node_costs)
            cost_eval = tensor_graphs.evaluate_makespan(instructions, node_costs)
            cost_get = tensor_graphs.get_cost(instructions, node_costs)
            cost_an = analyze_calculate_makespan(instructions, node_costs)

            self.assertAlmostEqual(cost_tg, cost_eval, places=4)
            self.assertAlmostEqual(cost_tg, cost_get, places=4)
            self.assertAlmostEqual(cost_tg, cost_an, places=4)


if __name__ == "__main__":
    unittest.main()
