#pragma once

#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>

#include "core/misc.hpp"
#include "core/plan/search_engine.hpp"

namespace selection_reachability_test
{

using namespace plan;
using GraphSpec = std::vector<std::vector<std::vector<uint32_t>>>;

inline void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

inline std::vector<VarId> addBucket(SearchState &state, const GraphSpec &spec)
{
    const uint32_t bucket_idx = static_cast<uint32_t>(state.buckets.size());
    state.buckets.emplace_back();
    state.bucket_egraphs.emplace_back();
    state.selected_vars.emplace_back();
    state.start_vars.emplace_back();
    state.offset_vars.emplace_back();
    state.reachable_cids.emplace_back();
    state.bucket_enode_infos.emplace_back();
    EGraph &egraph = state.bucket_egraphs.back();
    std::vector<EClassId> cids;
    const MemSpace mem_space{1, HandleType::CPP};
    for (size_t i = 0; i < spec.size(); ++i)
        cids.push_back(egraph.addEClass({1}, {1}, DType::FLOAT32, mem_space));
    state.bucket_root_ids.push_back(cids[0]);
    state.reachable_cids.back() = cids;
    std::vector<VarId> vars;
    for (uint32_t i = 0; i < spec.size(); ++i)
    {
        for (uint32_t en_idx = 0; en_idx < spec[i].size(); ++en_idx)
        {
            std::vector<EClassId> children;
            for (uint32_t child : spec[i][en_idx])
                children.push_back(cids[child]);
            egraph.addENode(cids[i], ENode(KernelId{0}, OpType::INPUT,
                "reachability_" + std::to_string(i) + "_" + std::to_string(en_idx),
                children, {1}, {1}, DType::FLOAT32, mem_space, {}));
        }
        VarInfo info;
        info.type = VarType::SELECTED;
        info.bucket_idx = bucket_idx;
        info.eclass_id = cids[i];
        uint32_t mask = (1u << (spec[i].size() + 1)) - 1;
        if (i == 0)
            mask &= ~1u;
        vars.push_back(state.addVar(info, Domain::makeMask(mask)));
        state.selected_vars.back()[cids[i]] = vars.back();
    }
    return vars;
}

// Deliberately recompute from the root for an independent reference result.
inline bool propagateReference(SearchState &state)
{
    bool changed;
    do
    {
        changed = false;
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            const EGraph &egraph = state.bucket_egraphs[b];
            const auto &vars = state.selected_vars[b];
            for (const auto &[cid, var_id] : vars)
            {
                const Domain selection = state.domains[var_id];
                if (selection.isEmpty())
                    return false;
                if (!selection.isFixed() || selection.fixedValue() == 0)
                    continue;
                const ENode &enode = egraph.getENode(egraph.getEClass(cid).enodes[selection.fixedValue() - 1]);
                for (EClassId child : enode.getChildren())
                {
                    Domain &child_domain = state.domains[vars.at(egraph.findConst(child))];
                    changed |= child_domain.remove(0);
                    if (child_domain.isEmpty())
                        return false;
                }
            }
            std::unordered_set<EClassId> reached{state.bucket_root_ids[b]};
            std::vector<EClassId> frontier{state.bucket_root_ids[b]};
            for (size_t head = 0; head < frontier.size(); ++head)
            {
                EClassId cid = frontier[head];
                const EClass &cls = egraph.getEClass(cid);
                for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
                {
                    if (!state.domains[vars.at(cid)].contains(en_idx + 1))
                        continue;
                    for (EClassId child : egraph.getENode(cls.enodes[en_idx]).getChildren())
                    {
                        child = egraph.findConst(child);
                        if (reached.insert(child).second)
                            frontier.push_back(child);
                    }
                }
            }
            for (const auto &[cid, var_id] : vars)
            {
                if (reached.count(cid))
                    continue;
                Domain &selection = state.domains[var_id];
                if (!selection.contains(0))
                    return false;
                if (!selection.isFixed())
                {
                    selection = Domain::makeFixed(0, selection.is_mask);
                    changed = true;
                }
            }
        }
    } while (changed);
    return true;
}

inline bool checkPropagation(SearchEngine &engine)
{
    SearchState reference = engine.state;
    const bool expected = propagateReference(reference);
    const bool actual = engine.runPropagators(kInvalidVarId);
    require(actual == expected, "Incremental reachability disagrees with BFS on feasibility");
    if (actual)
        require(engine.state.domains == reference.domains, "Incremental reachability disagrees with BFS on domains");
    return actual;
}

inline void testStructuralVariables()
{
    SearchState source;
    // The third root alternative will be pruned, disconnecting a cycle.
    // Merge IDs only after adding edges so traversal must canonicalize them.
    const GraphSpec spec = {{{6}, {2}, {3}}, {{}}, {{}}, {{4}}, {{3}}, {{}}, {{}}};
    addBucket(source, spec);
    EGraph &egraph = source.bucket_egraphs[0];
    egraph.merge(EClassId{1}, EClassId{6});
    egraph.merge(EClassId{0}, EClassId{5});
    auto &root_enodes = egraph.getEClass(EClassId{0}).enodes;
    root_enodes.erase(root_enodes.begin() + 2);

    SearchState state;
    state.buckets.resize(2);
    state.bucket_egraphs = {egraph, egraph};
    state.bucket_root_ids = {EClassId{5}, EClassId{2}};
    VarInfo cache_info;
    cache_info.type = VarType::CACHED;
    cache_info.base_eclass_id = BaseEClassId{10};
    const VarId cache_var = state.addVar(cache_info, Domain::makeMask(3));
    state.cached_vars[cache_info.base_eclass_id] = cache_var;
    state.addBucketVariables(0);
    state.addBucketVariables(1);

    require(state.bucket_root_ids[0] == EClassId{0}, "Merged root was not canonicalized");
    require(state.reachable_cids[0] == std::vector<EClassId>{EClassId{0}, EClassId{1}, EClassId{2}},
            "Structural closure lost an alternative or included a disconnected class");
    require(state.reachable_cids[1] == std::vector<EClassId>{EClassId{2}},
            "Reachability must be computed independently for each bucket");
    require(state.numVars() == 16, "Unexpected number of variables in compact state");
    require(state.cached_vars.at(cache_info.base_eclass_id) == cache_var && state.domains[cache_var].size() == 2,
            "Shared cache variable changed during bucket initialization");
    for (uint32_t b = 0; b < state.buckets.size(); ++b)
    {
        for (EClassId cid : {EClassId{3}, EClassId{4}, EClassId{5}, EClassId{6}})
            require(!state.selected_vars[b].count(cid) && !state.start_vars[b].count(cid) &&
                    !state.offset_vars[b].count(cid), "Unreachable or noncanonical class has variables");
    }
    for (VarId var_id = 0; var_id < state.numVars(); ++var_id)
    {
        const VarInfo &info = state.var_infos[var_id];
        require(info.id == var_id, "Variable IDs are not dense");
        if (info.type == VarType::START)
            require(state.domains[var_id].getMax() == static_cast<int32_t>(state.reachable_cids[info.bucket_idx].size()) - 1,
                    "Schedule domain still includes unreachable classes");
    }

    SearchEngine engine(std::move(state));
    engine.addPropagator(std::make_unique<SelectionReachabilityPropagator>());
    engine.addPropagator(std::make_unique<SelectionChildrenPropagator>());
    require(checkPropagation(engine), "Compact state failed initial propagation");
    const size_t marker = engine.state.getTrailMarker();
    const auto initial_domains = engine.state.domains;
    const VarId root_var = engine.state.selected_vars[0].at(EClassId{0});
    const VarId first_var = engine.state.selected_vars[0].at(EClassId{1});
    const VarId second_var = engine.state.selected_vars[0].at(EClassId{2});
    engine.state.setDomain(root_var, Domain::makeFixed(1, true));
    require(checkPropagation(engine) && !engine.state.domains[first_var].contains(0) &&
            engine.state.domains[second_var].fixedValue() == 0, "First alternative lost its child");
    engine.state.backtrackTo(marker);
    require(engine.state.domains == initial_domains, "Compact state did not restore domains");
    engine.state.setDomain(root_var, Domain::makeFixed(2, true));
    require(checkPropagation(engine) && !engine.state.domains[second_var].contains(0) &&
            engine.state.domains[first_var].fixedValue() == 0, "Rollback lost the other alternative");
}

inline void testVariableDomainBoundaries()
{
    SearchState state;
    state.buckets.emplace_back();
    state.bucket_egraphs.emplace_back();
    EGraph &egraph = state.bucket_egraphs[0];
    const MemSpace mem_space{0, HandleType::STORAGE};
    EClassId root_id = egraph.addEClass({1}, {1}, DType::FLOAT32, mem_space);
    for (uint32_t i = 0; i < 31; ++i)
        egraph.addENode(root_id, ENode(KernelId{0}, OpType::INPUT, "boundary_" + std::to_string(i),
                                     {}, {1}, {1}, DType::FLOAT32, mem_space, {}));
    state.bucket_root_ids.push_back(root_id);
    state.addBucketVariables(0);
    require(state.numVars() == 32 && state.offset_vars[0].empty(), "Storage class should not have an offset");
    const Domain &selection = state.domains[state.selected_vars[0].at(root_id)];
    require(selection.size() == 31 && !selection.contains(0) && selection.contains(31),
            "31 enodes must fit in the selection mask");
    for (VarId start_var : state.start_vars[0].at(root_id))
        require(state.domains[start_var].isFixed() && state.domains[start_var].fixedValue() == 0,
                "Single reachable class needs only one schedule position");

    // Relative mask offset tests
    {
        Domain d = Domain::makeMask(0b1101, 10); // {10, 12, 13}
        require(d.size() == 3, "Mask size should be 3");
        require(d.getMin() == 10, "Min should be 10");
        require(d.getMax() == 13, "Max should be 13");
        require(d.contains(10) && !d.contains(11) && d.contains(12) && d.contains(13), "Contains check failed");
        require(!d.contains(9) && !d.contains(14) && !d.contains(50), "Out-of-range contains check failed");

        // remove min
        require(d.remove(10), "Remove 10 should succeed");
        require(d.getMin() == 12 && d.size() == 2, "Min should now be 12");

        // remove until fixed
        require(d.remove(12), "Remove 12 should succeed");
        require(d.isFixed() && d.fixedValue() == 13, "Should be fixed to 13");

        // setMin / setMax with offset
        Domain d2 = Domain::makeMask(0b1111, 20); // {20, 21, 22, 23}
        require(d2.setMin(22), "setMin(22) should change domain");
        require(d2.getMin() == 22 && d2.size() == 2, "setMin result incorrect");

        Domain d3 = Domain::makeMask(0b1111, 20); // {20, 21, 22, 23}
        require(d3.setMax(21), "setMax(21) should change domain");
        require(d3.getMax() == 21 && d3.size() == 2, "setMax result incorrect");

        // makeFixed with arbitrary value
        Domain d_fixed = Domain::makeFixed(45, true);
        require(d_fixed.isFixed() && d_fixed.fixedValue() == 45, "makeFixed mask out of 0..31 failed");

        // Automatic conversion from range to mask on interior removal
        Domain d_range = Domain::makeRange(10, 20);
        require(d_range.canConvertToMask(), "Range [10..20] should be convertible to mask");
        require(d_range.remove(15), "Interior removal should convert to mask and remove 15");
        require(d_range.is_mask, "Should now be mask");
        require(!d_range.contains(15) && d_range.contains(14) && d_range.contains(16), "15 removed but neighbors intact");
        require(d_range.getMin() == 10 && d_range.getMax() == 20 && d_range.size() == 10, "Range-to-mask bounds preserved");

        // Intersecting two masks with different relative offsets
        Domain m1 = Domain::makeMask(0b10101, 10); // {10, 12, 14}
        Domain m2 = Domain::makeMask(0b00101, 12); // {12, 14}
        require(m1.intersectWith(m2), "Intersection should narrow m1");
        require(m1.size() == 2 && m1.contains(12) && m1.contains(14) && !m1.contains(10), "Shifted mask intersection failed");

        // Equality between differently offset masks representing same values
        Domain eq1 = Domain::makeMask(0b0100, 10); // {12}
        Domain eq2 = Domain::makeMask(0b0001, 12); // {12}
        require(eq1 == eq2, "Mask equality with different offsets should hold");
    }
}

inline void testRepairsAndUndo()
{
    SearchState state;
    // A direct path, a longer alternative, parallel edges, and a cycle with a
    // self-loop. All disappear when both root alternatives are removed.
    const GraphSpec spec = {{{1, 1}, {2}, {}}, {{3}, {}}, {{4}, {}}, {{1, 3}, {}}, {{1}, {}}};
    const auto vars = addBucket(state, spec);
    const auto other_vars = addBucket(state, spec);
    SearchEngine engine(std::move(state));
    engine.addPropagator(std::make_unique<SelectionReachabilityPropagator>());
    engine.addPropagator(std::make_unique<SelectionChildrenPropagator>());
    require(checkPropagation(engine), "Initial reachability failed");
    const auto initial_domains = engine.state.domains;
    const size_t marker = engine.state.getTrailMarker();

    Domain root_domain = engine.state.domains[vars[0]];
    root_domain.remove(1);
    engine.state.setDomain(vars[0], root_domain);
    require(checkPropagation(engine), "Longer replacement path was lost");
    require(!engine.state.domains[vars[1]].isFixed(), "Reachable alternative was pruned");
    const size_t nested_marker = engine.state.getTrailMarker();
    const auto nested_domains = engine.state.domains;

    root_domain.remove(2);
    engine.state.setDomain(vars[0], root_domain);
    require(checkPropagation(engine), "Disconnected cycle should be pruned");
    for (size_t i = 1; i < vars.size(); ++i)
    {
        require(engine.state.domains[vars[i]] == Domain::makeFixed(0, true), "Disconnected cycle stayed selectable");
        require(engine.state.domains[other_vars[i]] == initial_domains[other_vars[i]], "Another bucket was changed");
    }
    engine.state.backtrackTo(nested_marker);
    require(engine.state.domains == nested_domains && checkPropagation(engine), "Nested rollback failed");
    engine.state.backtrackTo(marker);
    require(engine.state.domains == initial_domains && checkPropagation(engine), "Rollback failed");

    // A sibling chooses the direct path; a copied state must own independent undo data.
    SearchEngine copied(engine.state);
    copied.addPropagator(std::make_unique<SelectionReachabilityPropagator>());
    copied.addPropagator(std::make_unique<SelectionChildrenPropagator>());
    copied.state.setDomain(vars[0], Domain::makeFixed(1, true));
    require(checkPropagation(copied), "Copied state lost its tree");
    require(engine.state.domains == initial_domains, "Copied reachability modified the original state");

    // Contradiction after repairs, then restore and take a feasible sibling.
    engine.state.setDomain(vars[1], Domain::makeFixed(1, true));
    engine.state.setDomain(vars[0], Domain::makeFixed(3, true));
    require(!checkPropagation(engine), "Required disconnected node did not contradict");
    engine.state.backtrackTo(marker);
    engine.state.setDomain(vars[0], Domain::makeFixed(2, true));
    require(checkPropagation(engine), "Sibling after contradiction failed");
}

inline void testInitializationAndWorklist()
{
    SearchState state;
    const auto vars = addBucket(state, {{{1}, {}}, {{2}}, {{}}});
    SearchEngine engine(std::move(state));
    engine.addPropagator(std::make_unique<SelectionReachabilityPropagator>());
    engine.addPropagator(std::make_unique<SelectionChildrenPropagator>());
    const auto initial_domains = engine.state.domains;
    const size_t marker = engine.state.getTrailMarker();
    engine.state.setDomain(vars[0], Domain::makeFixed(1, true));
    require(checkPropagation(engine), "Worklist did not propagate selected children");
    require(engine.state.domains[vars[2]] == Domain::makeFixed(1, true), "Child propagation did not reach a fixpoint");
    engine.state.backtrackTo(marker);
    require(engine.state.domains == initial_domains, "Backtracking before initialization failed");
    engine.state.setDomain(vars[0], Domain::makeFixed(2, true));
    require(checkPropagation(engine), "Reinitialization after rollback failed");
    require(engine.state.domains[vars[2]] == Domain::makeFixed(0, true), "Unused descendant stayed selectable");
}

inline void testRandomBranches()
{
    std::mt19937 random(71237);
    for (uint32_t trial = 0; trial < 80; ++trial)
    {
        GraphSpec spec(3 + random() % 6);
        for (auto &enodes : spec)
        {
            enodes.resize(2 + random() % 3);
            for (auto &children : enodes)
            {
                children.resize(random() % 4);
                for (auto &child : children)
                    child = random() % spec.size();
            }
        }
        SearchState state;
        addBucket(state, spec);
        SearchEngine engine(std::move(state));
        engine.addPropagator(std::make_unique<SelectionReachabilityPropagator>());
    engine.addPropagator(std::make_unique<SelectionChildrenPropagator>());
        std::function<void(uint32_t)> visit = [&](uint32_t depth) {
            if (!checkPropagation(engine) || depth == 5)
                return;
            std::vector<VarId> candidates;
            for (VarId var_id = 0; var_id < engine.state.numVars(); ++var_id)
                if (!engine.state.domains[var_id].isFixed())
                    candidates.push_back(var_id);
            if (candidates.empty())
                return;
            const VarId var_id = candidates[random() % candidates.size()];
            const Domain original = engine.state.domains[var_id];
            std::vector<int32_t> values;
            for (int32_t value = original.getMin(); value <= original.getMax(); ++value)
                if (original.contains(value))
                    values.push_back(value);
            const int32_t value = values[random() % values.size()];
            Domain remainder = original;
            remainder.remove(value);
            const size_t marker = engine.state.getTrailMarker();
            const auto parent_domains = engine.state.domains;
            for (const Domain &branch : {Domain::makeFixed(value, true), remainder})
            {
                engine.state.setDomain(var_id, branch);
                visit(depth + 1);
                engine.state.backtrackTo(marker);
                require(engine.state.domains == parent_domains, "Random branch rollback changed parent domains");
            }
        };
        visit(0);
    }
}

inline void testSearchNodeRestore()
{
    SearchState state;
    const auto vars = addBucket(state, {{{1}, {2}, {}}, {{}, {2}}, {{1}, {}}});
    SearchEngine engine(std::move(state));
    engine.addPropagator(std::make_unique<SelectionReachabilityPropagator>());
    engine.addPropagator(std::make_unique<SelectionChildrenPropagator>());
    engine.state.setDomain(vars[1], Domain::makeFixed(1, true));
    require(checkPropagation(engine), "Required node should initially be reachable");
    const size_t marker = engine.state.getTrailMarker();
    const auto parent_domains = engine.state.domains;
    auto root = std::make_shared<SearchNode>(0, UINT32_MAX,
        std::make_pair(kInvalidVarId, Domain{}), 0.0f, 0.0f, 0, marker);
    auto failed = std::make_shared<SearchNode>(1, 0,
        std::make_pair(vars[0], Domain::makeFixed(3, true)), 0.0f, 0.0f, 1);
    auto sibling = std::make_shared<SearchNode>(2, 0,
        std::make_pair(vars[0], Domain::makeFixed(1, true)), 0.0f, 0.0f, 1);
    engine.all_nodes = {root, failed, sibling};
    engine.current_node_id = 0;
    require(!engine.restoreNode(failed), "Disconnected required node should fail replay");
    require(engine.current_node_id == 0 && engine.state.getTrailMarker() == marker &&
            engine.state.domains == parent_domains, "Failed replay did not restore its parent");
    require(engine.restoreNode(sibling) && checkPropagation(engine), "Sibling replay failed after conflict");
    require(engine.restoreNode(root) && engine.state.domains == parent_domains, "LCA rollback failed");
    require(engine.restoreNode(sibling) && checkPropagation(engine), "Repeated sibling replay failed");
}

} // namespace selection_reachability_test

inline void runSelectionReachabilityTests()
{
    selection_reachability_test::testStructuralVariables();
    selection_reachability_test::testVariableDomainBoundaries();
    selection_reachability_test::testRepairsAndUndo();
    selection_reachability_test::testInitializationAndWorklist();
    selection_reachability_test::testRandomBranches();
    selection_reachability_test::testSearchNodeRestore();
    std::cout << "selection reachability tests passed" << std::endl;
}
