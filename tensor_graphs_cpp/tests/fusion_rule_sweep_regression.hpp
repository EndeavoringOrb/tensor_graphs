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

// Regression test verifying why an unconditional FusionRule visited set is invalid.
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

    std::unique_ptr<Rule> clone() const override
    {
        return std::make_unique<FusionRuleWithVisitedSet>(*this);
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

inline void runPatternVisitRegressionTests()
{
    FusionRule::Pattern pattern;
    auto &graph = pattern.graph;
    const auto x = graph.input({1}, DType::FLOAT32);
    const auto y = graph.input({1}, DType::FLOAT32);
    pattern.variables = {x, y};
    pattern.dtypes = {DType::FLOAT32, DType::FLOAT32};
    pattern.rootId = graph.pow(graph.mul(graph.add(x, y), y), x);
    pattern.varIndex.assign(pattern.rootId.value + 1, -1);
    pattern.varIndex[x.value] = 0;
    pattern.varIndex[y.value] = 1;

    EGraph egraph;
    const MemSpace memory{1, HandleType::CPP};
    auto node = [&](OpType op, std::vector<EClassId> children, const std::string &tag = "") {
        return ENode(KernelId{0}, op, "", std::move(children), {1}, {1}, DType::FLOAT32, memory, {}, tag);
    };
    auto add = [&](OpType op, std::vector<EClassId> children, const std::string &tag = "") {
        const auto owner = egraph.addEClass({1}, {1}, DType::FLOAT32, memory);
        return egraph.addENode(owner, node(op, std::move(children), tag));
    };
    const auto input_x = add(OpType::INPUT, {}, "x");
    const auto input_y = add(OpType::INPUT, {}, "y");
    const auto deep = add(OpType::INPUT, {}, "not-an-add-yet");
    const auto middle = add(OpType::MUL, {deep, input_y});
    const ENodeId root{static_cast<uint32_t>(egraph.enodes.size())};
    add(OpType::POWER, {middle, input_x});
    const auto unread = add(OpType::INPUT, {}, "unread");
    std::unordered_set<EClassId> protected_classes;
    FusionRule rule;
    auto matches = [&]() {
        FusionRule::Binding binding(pattern.variables.size());
        return rule.matchPatternCached(root, egraph, pattern, binding, protected_classes);
    };
    if (matches() || matches() || rule.match_visit_hits != 1)
        Error::throw_err("[FusionRule visits] unchanged failed match was not cached");
    egraph.addENode(unread, node(OpType::ADD, {input_y, input_y}));
    if (matches() || rule.match_visit_hits != 2)
        Error::throw_err("[FusionRule visits] unrelated descendants invalidated a failed match");
    egraph.addENode(deep, node(OpType::ADD, {input_x, input_y}));
    if (!matches() || rule.match_visit_hits != 2)
        Error::throw_err("[FusionRule visits] failed branch did not notice a matching grandchild");
    egraph.addENode(input_x, node(OpType::ADD, {unread, input_y}));
    if (!matches() || rule.match_visit_hits != 3)
        Error::throw_err("[FusionRule visits] traversal continued below a pattern variable");
    protected_classes.insert(deep);
    if (matches() || rule.match_visit_hits != 3)
        Error::throw_err("[FusionRule visits] changed protection did not invalidate a match");
    egraph.addENode(deep, node(OpType::ADD, {input_y, input_x}));
    if (matches() || rule.match_visit_hits != 4)
        Error::throw_err("[FusionRule visits] inspected expressions below a protected stop");
    protected_classes.clear();
    if (!matches() || rule.match_visit_hits != 4)
        Error::throw_err("[FusionRule visits] removing protection did not revisit its pattern position");

    auto cloned_rule = rule.clone();
    auto *fusion_clone = dynamic_cast<FusionRule *>(cloned_rule.get());
    if (!fusion_clone)
        Error::throw_err("[FusionRule visits] clone changed rule type");
    EGraph cloned_graph = egraph;
    cloned_graph.addENode(deep, node(OpType::MUL, {input_y, input_x}));
    FusionRule::Binding cloned_binding(pattern.variables.size());
    if (!fusion_clone->matchPatternCached(root, cloned_graph, pattern, cloned_binding, protected_classes) ||
        fusion_clone->match_visit_hits != 4 || !matches() || rule.match_visit_hits != 5)
        Error::throw_err("[FusionRule visits] one bucket's invalidation changed another bucket's visit");

    const auto replacement = add(OpType::INPUT, {}, "new-canonical-x");
    egraph.merge(replacement, input_x);
    FusionRule::Binding merged_binding(pattern.variables.size());
    if (!rule.matchPatternCached(root, egraph, pattern, merged_binding, protected_classes) ||
        rule.match_visit_hits != 5 || merged_binding.slots[0] != egraph.findConst(input_x))
        Error::throw_err("[FusionRule visits] merge retained stale variable bindings");

    FusionRule::Pattern constant_pattern;
    const auto variable = constant_pattern.graph.input({1}, DType::FLOAT32);
    const float value = 2.0f;
    const auto literal = constant_pattern.graph.constant({1}, &value, DType::FLOAT32);
    constant_pattern.variables = {variable};
    constant_pattern.dtypes = {DType::FLOAT32};
    constant_pattern.rootId = constant_pattern.graph.add(variable, literal);
    constant_pattern.varIndex.assign(constant_pattern.rootId.value + 1, -1);
    constant_pattern.varIndex[variable.value] = 0;
    const auto literal_class = add(OpType::INPUT, {}, "late-constant");
    const ENodeId constant_root{static_cast<uint32_t>(egraph.enodes.size())};
    add(OpType::ADD, {input_x, literal_class});
    auto constant_matches = [&]() {
        FusionRule::Binding binding(1);
        return rule.matchPatternCached(constant_root, egraph, constant_pattern, binding, protected_classes);
    };
    if (constant_matches() || constant_matches())
        Error::throw_err("[FusionRule visits] unstaged literal unexpectedly matched");
    const auto hits_before = rule.match_visit_hits;
    egraph.constantStaging[literal_class] = std::make_shared<std::vector<uint8_t>>(
        *constant_pattern.graph.constantStaging.at(literal));
    if (!constant_matches() || rule.match_visit_hits != hits_before)
        Error::throw_err("[FusionRule visits] late constant staging did not invalidate its literal position");

    const ENodeId variant{static_cast<uint32_t>(egraph.enodes.size())};
    egraph.addENode(egraph.getENodeEClass(root),
        ENode(KernelId{12345}, OpType::POWER, "different-kernel", {middle, input_x}, {1}, {1},
              DType::FLOAT32, memory, {}));
    FusionRule::Binding variant_binding(pattern.variables.size());
    const auto variant_hits_before = rule.match_visit_hits;
    if (!rule.matchPatternCached(variant, egraph, pattern, variant_binding, protected_classes) ||
        rule.match_visit_hits != variant_hits_before + 1)
        Error::throw_err("[FusionRule visits] identical root operations did not share pattern visits");
}

inline void runFusionRuleSweepRegressionTests()
{
    std::cout << "fusion rule sweep regression tests" << std::endl << std::flush;
    runPatternVisitRegressionTests();

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
    // Part 1: Standard Planner saturation with dependency-aware visits.
    // In Sweep 1, RemoveContiguous removes the CONTIGUOUS wrapper from the child.
    // In Sweep 2, FusionRule re-evaluates root and successfully matches Silu_3D_1.
    // -------------------------------------------------------------------------
    Planner planner(costModel, settings);
    const EGraph result = planner.saturateBucket(root, graph, fullBucket, true);

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
