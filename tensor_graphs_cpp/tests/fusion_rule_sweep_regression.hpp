#pragma once

#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/plan/planner.hpp"
#include "core/rewrite.hpp"
#include "core/settings.hpp"
#include "tests/common.hpp"

// Regression test verifying why FusionRule cannot have a visited set across sweeps.
// When a parent e-node (e.g. root mul / div) is evaluated in sweep 1, its children
// may not yet match the fusion pattern because an intermediate op (like CONTIGUOUS)
// is present. During sweep 1, RemoveContiguous unwraps the child. In sweep 2, the
// parent e-node MUST be re-evaluated so it can observe the newly added unwrapped
// child and successfully discover the fused kernel.
// If FusionRule had a visited set, the parent would be skipped in sweep 2 and the
// fused kernel would never be discovered.

namespace FusionRuleSweepRegression
{

// A mock version of FusionRule that tracks visited e-nodes in match() to demonstrate
// the failure mode that occurs if visited sets are added to FusionRule.
struct FusionRuleWithVisitedSet : public Rule
{
    FusionRule inner;
    std::unordered_set<uint32_t> visited_enodes;

    std::string name() const override
    {
        return "FusionRuleWithVisitedSet";
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (visited_enodes.count(eNodeIdx))
            return false;
        visited_enodes.insert(eNodeIdx);
        return inner.match(eNodeIdx, ctx);
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        inner.apply(eNodeIdx, ctx);
    }
};

inline void runFusionRuleSweepRegressionTests()
{
    std::cout << "fusion rule sweep regression tests" << std::endl << std::flush;

    CostModel costModel(false, "");
    Settings settings;
    settings.do_saturate = true;

    // Build a graph containing the SiLU decomposition where an intermediate
    // CONTIGUOUS op separates pow from add.
    Graph graph;
    const LogicalId x = graph.input({1, 1, 2048}, DType::FLOAT32);
    const auto &target_shape = graph.getNode(x).getShape();

    auto broadcast = [&graph, &target_shape](LogicalId scalar_id) {
        std::vector<int32_t> ones(target_shape.size(), 1);
        LogicalId out = graph.reshape(scalar_id,
                                      graph.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));
        for (uint64_t i = 0; i < target_shape.size(); ++i)
        {
            if (target_shape[i] > 1)
            {
                int32_t repeat = (int32_t)target_shape[i];
                int32_t axis = (int32_t)i;
                out = graph.repeat(out, graph.constant({1}, &repeat, DType::INT32),
                                   graph.constant({1}, &axis, DType::INT32));
            }
        }
        return out;
    };

    LogicalId neg_x = graph.neg(x);
    float e_value = 2.7182818f;
    LogicalId e_node = broadcast(graph.constant({1}, &e_value, DType::FLOAT32));
    LogicalId exp_neg = graph.pow(e_node, neg_x);

    // Intentionally wrap exp_neg with two nested CONTIGUOUS ops.
    // In Sweep 1, RemoveContiguous unwraps the outer CONTIGUOUS op into add(one, contig1).
    // Because contig1 is still contiguous, root (mul) CANNOT match Silu in Sweep 1,
    // and is visited by FusionRule.
    // In Sweep 2, RemoveContiguous unwraps the inner CONTIGUOUS op into add(one, pow).
    // FusionRule MUST re-evaluate root in Sweep 2 to discover the Silu_3D_1 fused kernel.
    // If FusionRule had a visited set, root would be skipped in Sweep 2 and Silu_3D_1
    // would never be discovered.
    LogicalId exp_neg_contig1 = graph.contiguous(exp_neg);
    LogicalId exp_neg_contig2 = graph.contiguous(exp_neg_contig1);

    float one_value = 1.0f;
    LogicalId one_node = broadcast(graph.constant({1}, &one_value, DType::FLOAT32));
    LogicalId denominator = graph.add(one_node, exp_neg_contig2);
    LogicalId sigmoid = graph.div(one_node, denominator);
    LogicalId root = graph.mul(x, sigmoid);

    Bucket fullBucket;
    fullBucket.inputDirtyRegions[x] = {makeRegion({{0, 1}, {0, 1}, {0, 2048}})};
    fullBucket.outputNeededRegion = {makeRegion({{0, 1}, {0, 1}, {0, 2048}})};

    // -------------------------------------------------------------------------
    // Part 1: Standard Planner saturation (FusionRule has NO visited set).
    // In Sweep 1, RemoveContiguous removes the CONTIGUOUS wrapper from the child.
    // In Sweep 2, FusionRule re-evaluates root and successfully matches Silu_3D_1.
    // -------------------------------------------------------------------------
    Planner planner(costModel, settings);
    const EGraph result = planner.saturateBucket(root, graph, fullBucket, {}, true);

    bool found_fused_silu = false;
    for (const ENode &enode : result.getENodes())
    {
        if (enode.getOpType() == OpType::FUSED && enode.getOpName() == "Silu_3D_1")
        {
            found_fused_silu = true;
            break;
        }
    }

    if (!found_fused_silu)
    {
        Error::throw_err(
            "[FusionRuleSweepRegression] Expected Silu_3D_1 to be discovered across saturation sweeps, but none was found!");
    }

    // -------------------------------------------------------------------------
    // Part 2: Demonstrate that if FusionRule HAD a visited set, the exact same
    // graph FAILS to discover the fused kernel because root is skipped in Sweep 2.
    // -------------------------------------------------------------------------
    {
        std::vector<LogicalId> topo = topologicalSort({root}, graph);
        Planner visited_planner(costModel, settings);
        visited_planner.initBaseEGraph(root, graph, topo, nullptr, false);
        EGraph test_egraph = visited_planner.baseState;

        std::unordered_set<EClassId> protectedClasses;
        RuleCtx ctx{test_egraph, protectedClasses, nullptr, &costModel};

        std::vector<std::unique_ptr<Rule>> rules;
        rules.emplace_back(std::make_unique<FusionRuleWithVisitedSet>());
        rules.emplace_back(std::make_unique<RemoveContiguous>());

        bool changed = true;
        uint32_t nMatches = 0;
        while (changed)
        {
            uint32_t preMatches = nMatches;
            for (uint32_t eNodeIdx = 0; eNodeIdx < test_egraph.getENodes().size(); eNodeIdx++)
            {
                for (const auto &rule : rules)
                {
                    if (!rule->match(eNodeIdx, ctx))
                        continue;
                    rule->apply(eNodeIdx, ctx);
                    nMatches++;
                }
            }
            if (nMatches == preMatches)
                break;
            test_egraph.rebuild();
        }

        bool visited_found_fused_silu = false;
        for (const ENode &enode : test_egraph.getENodes())
        {
            if (enode.getOpType() == OpType::FUSED && enode.getOpName() == "Silu_3D_1")
            {
                visited_found_fused_silu = true;
                break;
            }
        }

        if (visited_found_fused_silu)
        {
            Error::throw_err(
                "[FusionRuleSweepRegression] A visited-set FusionRule was unexpectedly able to discover the fused kernel!");
        }
    }

    std::cout << "  [PASS] FusionRule sweep regression test passed (verified multi-sweep re-evaluation when child changes)."
              << std::endl;
}

} // namespace FusionRuleSweepRegression
