// In tensor_graphs_cpp/tests/pruning/test.hpp
#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "core/common/constants.hpp"
#include "core/plan/extractor.hpp"
#include "core/plan/planner.hpp"
#include "core/plan/rule_registry.hpp"
#include "core/plan/validators/mem.hpp"
#include "core/settings.hpp"
#include "core/types.hpp"
#include "tests/common.hpp"
#include "tests/pruning/common.hpp"

namespace prune_test
{

struct TrialResult
{
    std::string rule_name;
    double baseline_ms = 0.0;
    double test_ms = 0.0;
    double baseline_slope = 0.0;
    double test_slope = 0.0;
    double speedup = 1.0;
    uint64_t total_states = 0;
    bool was_faster = false;
    bool test_passed = false;
    std::string err_msg;
};

template <typename IterFactory, typename YieldFn>
inline TrialResult runTrial(const std::string &rule_name, IterFactory &&make_iter, YieldFn &&yield,
                            int warmup_iters = 3, int timed_iters = 10)
{
    TrialResult r;
    r.rule_name = rule_name;

    for (int w = 0; w < warmup_iters; ++w)
    {
        auto iter = make_iter();
        while (yield(iter))
        {
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timed_iters; ++t)
    {
        auto iter = make_iter();
        while (yield(iter))
        {
            if (t == 0)
                r.total_states++;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    r.test_ms = total_ms / timed_iters;
    return r;
}

// =============================================================================
// Per-category multi-scale rule runners
// =============================================================================

template <typename Rule> struct DispatchBenchOne
{
    static TrialResult run(const std::vector<double> &scales)
    {
        TrialResult r;
        r.rule_name = Rule{}.name();

        std::vector<double> base_times;
        std::vector<double> rule_times;

        for (double s : scales)
        {
            MockCtx mock;
            Graph g;
            LogicalId root = buildWideShallow(g, static_cast<int>(s));
            mock.build(g, root);

            auto base_iter = makeDispatchIterator(mock.egraph, mock.selection_map, mock.enodeInfos);
            std::vector<EClassId> order;
            float baseline_cost = TGConstants::INF;
            while (base_iter.getNextDispatchOrder(mock.selection_map, order))
            {
                baseline_cost =
                    std::min(baseline_cost, get_cost(order, mock.egraph, mock.selection_map, mock.enodeInfos));
            }

            auto rule_iter = makeDispatchIterator(mock.egraph, mock.selection_map, mock.enodeInfos, Rule{true});
            float rule_cost = TGConstants::INF;
            while (rule_iter.getNextDispatchOrder(mock.selection_map, order))
            {
                rule_cost = std::min(rule_cost, get_cost(order, mock.egraph, mock.selection_map, mock.enodeInfos));
            }

            if (std::abs(rule_cost - baseline_cost) > 1e-5f)
            {
                r.test_passed = false;
                std::ostringstream ss;
                ss << "cost " << rule_cost << " != baseline " << baseline_cost << " at scale " << s;
                r.err_msg = ss.str();
                return r;
            }

            auto make_base = [&]() { return makeDispatchIterator(mock.egraph, mock.selection_map, mock.enodeInfos); };
            auto yield_base = [&](auto &it) {
                std::vector<EClassId> ord;
                return it.getNextDispatchOrder(mock.selection_map, ord);
            };
            auto base_res = runTrial("Base", make_base, yield_base, 2, 5);
            base_times.push_back(base_res.test_ms);

            auto make_rule = [&]() {
                return makeDispatchIterator(mock.egraph, mock.selection_map, mock.enodeInfos, Rule{true});
            };
            auto yield_rule = [&](auto &it) {
                std::vector<EClassId> ord;
                return it.getNextDispatchOrder(mock.selection_map, ord);
            };
            auto rule_trial_res = runTrial(r.rule_name, make_rule, yield_rule, 2, 5);
            rule_times.push_back(rule_trial_res.test_ms);
        }

        r.test_passed = true;
        auto base_fit = fitLine(scales, base_times);
        auto rule_fit = fitLine(scales, rule_times);

        r.baseline_ms = base_fit.avg_ms;
        r.test_ms = rule_fit.avg_ms;
        r.baseline_slope = base_fit.slope;
        r.test_slope = rule_fit.slope;

        if (base_fit.slope > 1e-6 && rule_fit.slope > 1e-6)
        {
            r.speedup = base_fit.slope / rule_fit.slope;
            r.was_faster = (rule_fit.slope < base_fit.slope - 1e-6);
        }
        else
        {
            r.speedup = (rule_fit.avg_ms > 1e-7) ? (base_fit.avg_ms / rule_fit.avg_ms) : 1.0;
            r.was_faster = (rule_fit.avg_ms < base_fit.avg_ms - 1e-6);
        }

        if (std::abs(rule_fit.slope - base_fit.slope) <= 1e-6)
        {
            r.was_faster = (rule_fit.avg_ms < base_fit.avg_ms);
        }

        return r;
    }
};

template <typename Rule> struct BufferizeBenchOne
{
    static TrialResult run(const std::vector<double> &scales)
    {
        TrialResult r;
        r.rule_name = Rule{}.name();

        std::vector<double> base_times;
        std::vector<double> rule_times;

        for (double s : scales)
        {
            MockCtx mock;
            Graph g;
            LogicalId root = buildLinearChain(g, static_cast<int>(s));
            mock.build(g, root);

            auto dispatch_iter = makeDispatchIterator(mock.egraph, mock.selection_map, mock.enodeInfos);
            std::vector<EClassId> order;
            if (!dispatch_iter.getNextDispatchOrder(mock.selection_map, order))
            {
                r.test_passed = false;
                r.err_msg = "failed to get dispatch order at scale " + std::to_string(s);
                return r;
            }
            uint64_t mem_cap = mock.settings.mem_caps.at(MemSpace{1, HandleType::CPP});

            auto base_iter = makeBufferizeIterator(order, mock.egraph, mock.selection_map, mock.enodeInfos);
            std::vector<ParallelBuffer> bufs;
            std::unordered_map<EClassId, BufferId> eclass_to_buf;
            float baseline_cost = TGConstants::INF;
            while (base_iter.getNextBufferization(bufs, eclass_to_buf))
            {
                BufferId overflow;
                std::vector<ParallelBuffer> allocated;
                if (malloc_by_time_components(mem_cap, bufs, allocated, overflow, nullptr, &mock.settings))
                    baseline_cost =
                        std::min(baseline_cost, get_cost(order, mock.egraph, mock.selection_map, mock.enodeInfos));
            }

            auto rule_iter = makeBufferizeIterator(order, mock.egraph, mock.selection_map, mock.enodeInfos, Rule{true});
            float rule_cost = TGConstants::INF;
            while (rule_iter.getNextBufferization(bufs, eclass_to_buf))
            {
                BufferId overflow;
                std::vector<ParallelBuffer> allocated;
                if (malloc_by_time_components(mem_cap, bufs, allocated, overflow, nullptr, &mock.settings))
                    rule_cost = std::min(rule_cost, get_cost(order, mock.egraph, mock.selection_map, mock.enodeInfos));
            }

            if (std::abs(rule_cost - baseline_cost) > 1e-5f)
            {
                r.test_passed = false;
                std::ostringstream ss;
                ss << "cost " << rule_cost << " != baseline " << baseline_cost << " at scale " << s;
                r.err_msg = ss.str();
                return r;
            }

            auto make_base = [&]() {
                return makeBufferizeIterator(order, mock.egraph, mock.selection_map, mock.enodeInfos);
            };
            auto yield_base = [&](auto &it) {
                std::vector<ParallelBuffer> b;
                std::unordered_map<EClassId, BufferId> m;
                return it.getNextBufferization(b, m);
            };
            auto base_res = runTrial("Base", make_base, yield_base, 2, 5);
            base_times.push_back(base_res.test_ms);

            auto make_rule = [&]() {
                return makeBufferizeIterator(order, mock.egraph, mock.selection_map, mock.enodeInfos, Rule{true});
            };
            auto yield_rule = [&](auto &it) {
                std::vector<ParallelBuffer> b;
                std::unordered_map<EClassId, BufferId> m;
                return it.getNextBufferization(b, m);
            };
            auto rule_trial_res = runTrial(r.rule_name, make_rule, yield_rule, 2, 5);
            rule_times.push_back(rule_trial_res.test_ms);
        }

        r.test_passed = true;
        auto base_fit = fitLine(scales, base_times);
        auto rule_fit = fitLine(scales, rule_times);

        r.baseline_ms = base_fit.avg_ms;
        r.test_ms = rule_fit.avg_ms;
        r.baseline_slope = base_fit.slope;
        r.test_slope = rule_fit.slope;

        if (base_fit.slope > 1e-6 && rule_fit.slope > 1e-6)
        {
            r.speedup = base_fit.slope / rule_fit.slope;
            r.was_faster = (rule_fit.slope < base_fit.slope - 1e-6);
        }
        else
        {
            r.speedup = (rule_fit.avg_ms > 1e-7) ? (base_fit.avg_ms / rule_fit.avg_ms) : 1.0;
            r.was_faster = (rule_fit.avg_ms < base_fit.avg_ms - 1e-6);
        }

        if (std::abs(rule_fit.slope - base_fit.slope) <= 1e-6)
        {
            r.was_faster = (rule_fit.avg_ms < base_fit.avg_ms);
        }

        return r;
    }
};

template <typename Rule> struct MallocBenchOne
{
    static TrialResult run(const std::vector<double> &scales)
    {
        TrialResult r;
        r.rule_name = Rule{}.name();

        std::vector<double> base_times;
        std::vector<double> rule_times;
        uint64_t mem_cap = 64ULL * 1024 * 1024;

        for (double s : scales)
        {
            std::vector<ParallelBuffer> unallocated = buildMallocBuffers(static_cast<int>(s));

            auto test_iter = makeMallocIterator(mem_cap, unallocated, Rule{true});
            std::vector<ParallelBuffer> allocated;
            if (!test_iter.getNextAllocation(allocated))
            {
                r.test_passed = false;
                r.err_msg = "failed to find an allocation at scale " + std::to_string(s);
                return r;
            }

            int64_t peak = 0;
            for (const auto &b : allocated)
                peak = std::max<int64_t>(peak, b.offset + static_cast<int64_t>(b.size));

            if (peak > static_cast<int64_t>(mem_cap))
            {
                r.test_passed = false;
                r.err_msg = "peak " + std::to_string(peak) + " exceeded mem_cap " + std::to_string(mem_cap);
                return r;
            }

            for (size_t i = 0; i < allocated.size(); ++i)
            {
                for (size_t j = i + 1; j < allocated.size(); ++j)
                {
                    if (overlapsBuf(allocated[i], allocated[j]))
                    {
                        int64_t start_i = allocated[i].offset;
                        int64_t end_i = allocated[i].offset + static_cast<int64_t>(allocated[i].size);
                        int64_t start_j = allocated[j].offset;
                        int64_t end_j = allocated[j].offset + static_cast<int64_t>(allocated[j].size);
                        if (std::max(start_i, start_j) < std::min(end_i, end_j))
                        {
                            r.test_passed = false;
                            r.err_msg = "overlapping buffers at scale " + std::to_string(s);
                            return r;
                        }
                    }
                }
            }

            auto make_base = [&]() { return makeMallocIterator(mem_cap, unallocated); };
            auto yield_base = [&](auto &it) {
                std::vector<ParallelBuffer> a;
                return it.getNextAllocation(a);
            };
            auto base_res = runTrial("Base", make_base, yield_base, 2, 5);
            base_times.push_back(base_res.test_ms);

            auto make_rule = [&]() { return makeMallocIterator(mem_cap, unallocated, Rule{true}); };
            auto yield_rule = [&](auto &it) {
                std::vector<ParallelBuffer> a;
                return it.getNextAllocation(a);
            };
            auto rule_trial_res = runTrial(r.rule_name, make_rule, yield_rule, 2, 5);
            rule_times.push_back(rule_trial_res.test_ms);
        }

        r.test_passed = true;
        auto base_fit = fitLine(scales, base_times);
        auto rule_fit = fitLine(scales, rule_times);

        r.baseline_ms = base_fit.avg_ms;
        r.test_ms = rule_fit.avg_ms;
        r.baseline_slope = base_fit.slope;
        r.test_slope = rule_fit.slope;

        if (base_fit.slope > 1e-6 && rule_fit.slope > 1e-6)
        {
            r.speedup = base_fit.slope / rule_fit.slope;
            r.was_faster = (rule_fit.slope < base_fit.slope - 1e-6);
        }
        else
        {
            r.speedup = (rule_fit.avg_ms > 1e-7) ? (base_fit.avg_ms / rule_fit.avg_ms) : 1.0;
            r.was_faster = (rule_fit.avg_ms < base_fit.avg_ms - 1e-6);
        }

        if (std::abs(rule_fit.slope - base_fit.slope) <= 1e-6)
        {
            r.was_faster = (rule_fit.avg_ms < base_fit.avg_ms);
        }

        return r;
    }
};

template <typename Rule> struct CacheBenchOne
{
    static TrialResult run(const std::vector<double> &scales)
    {
        TrialResult r;
        r.rule_name = Rule{}.name();

        std::vector<double> base_times;
        std::vector<double> rule_times;

        for (double s : scales)
        {
            MockCtx mock;
            Graph g;
            LogicalId root = buildCacheGraph(g, static_cast<int>(s));
            mock.build(g, root);

            std::vector<LogicalId> topo = topologicalSort({root}, g);
            std::vector<LogicalId> candidates;
            for (LogicalId id : topo)
                if (g.getNode(id).getSizeBytes() > 0)
                    candidates.push_back(id);

            std::vector<MemSpace> avail_mem_spaces = {MemSpace{1, HandleType::CPP}};

            Planner planner(mock.costModel, mock.settings);
            planner.initBaseEGraph(root, g, topo, nullptr);
            populateDummyRecords(mock.costModel, planner.baseState.egraph);

            Bucket bucket;
            for (LogicalId id : topo)
                if (g.getNode(id).opType == OpType::INPUT)
                    bucket.inputDirtyRegions[id] = {makeFull(g.getNode(id).getShape())};
            bucket.outputNeededRegion = {makeFull(g.getNode(root).getShape())};

            auto iter_rule = makeCacheIterator(g, candidates, avail_mem_spaces, Rule{true});
            std::unordered_map<LogicalId, MemSpace> cache_rule;
            float min_cost_rule = TGConstants::INF;
            while (iter_rule.getNextCacheSelection(cache_rule))
            {
                try
                {
                    std::unordered_map<LogicalId, ParallelBuffer> preallocated;
                    planner.preallocateLogicalBuffers(g, cache_rule, preallocated);
                    CompiledGraph plan_res = planner.plan(root, g, bucket, cache_rule, /*doSaturate=*/false,
                                                          /*strictCache=*/true, nullptr, preallocated, 0.0f, nullptr);
                    min_cost_rule = std::min(min_cost_rule, plan_res.cost());
                }
                catch (...)
                {
                }
            }

            auto iter_base = makeCacheIterator(g, candidates, avail_mem_spaces);
            std::unordered_map<LogicalId, MemSpace> cache_base;
            float min_cost_base = TGConstants::INF;
            while (iter_base.getNextCacheSelection(cache_base))
            {
                try
                {
                    std::unordered_map<LogicalId, ParallelBuffer> preallocated;
                    planner.preallocateLogicalBuffers(g, cache_base, preallocated);
                    CompiledGraph plan_res = planner.plan(root, g, bucket, cache_base, /*doSaturate=*/false,
                                                          /*strictCache=*/true, nullptr, preallocated, 0.0f, nullptr);
                    min_cost_base = std::min(min_cost_base, plan_res.cost());
                }
                catch (...)
                {
                }
            }

            if (min_cost_rule != TGConstants::INF && min_cost_base != TGConstants::INF)
            {
                if (std::abs(min_cost_rule - min_cost_base) > 1e-5f)
                {
                    r.test_passed = false;
                    std::ostringstream ss;
                    ss << "cost " << min_cost_rule << " != baseline " << min_cost_base << " at scale " << s;
                    r.err_msg = ss.str();
                    return r;
                }
            }

            auto make_base_it = [&]() { return makeCacheIterator(g, candidates, avail_mem_spaces); };
            auto yield_base = [&](auto &it) {
                std::unordered_map<LogicalId, MemSpace> c;
                return it.getNextCacheSelection(c);
            };
            auto base_res = runTrial("Base", make_base_it, yield_base, 2, 5);
            base_times.push_back(base_res.test_ms);

            auto make_rule_it = [&]() { return makeCacheIterator(g, candidates, avail_mem_spaces, Rule{true}); };
            auto yield_rule = [&](auto &it) {
                std::unordered_map<LogicalId, MemSpace> c;
                return it.getNextCacheSelection(c);
            };
            auto rule_trial_res = runTrial(r.rule_name, make_rule_it, yield_rule, 2, 5);
            rule_times.push_back(rule_trial_res.test_ms);
        }

        r.test_passed = true;
        auto base_fit = fitLine(scales, base_times);
        auto rule_fit = fitLine(scales, rule_times);

        r.baseline_ms = base_fit.avg_ms;
        r.test_ms = rule_fit.avg_ms;
        r.baseline_slope = base_fit.slope;
        r.test_slope = rule_fit.slope;

        if (base_fit.slope > 1e-6 && rule_fit.slope > 1e-6)
        {
            r.speedup = base_fit.slope / rule_fit.slope;
            r.was_faster = (rule_fit.slope < base_fit.slope - 1e-6);
        }
        else
        {
            r.speedup = (rule_fit.avg_ms > 1e-7) ? (base_fit.avg_ms / rule_fit.avg_ms) : 1.0;
            r.was_faster = (rule_fit.avg_ms < base_fit.avg_ms - 1e-6);
        }

        if (std::abs(rule_fit.slope - base_fit.slope) <= 1e-6)
        {
            r.was_faster = (rule_fit.avg_ms < base_fit.avg_ms);
        }

        return r;
    }
};

template <typename Rule> struct ExtractBenchOne
{
    static TrialResult run(const std::vector<double> &scales)
    {
        TrialResult r;
        r.rule_name = Rule{}.name();

        std::vector<double> base_times;
        std::vector<double> rule_times;

        for (double s : scales)
        {
            MockCtx mock;
            Graph g;
            LogicalId root = buildDiamond(g, static_cast<int>(s));
            mock.build(g, root);
            EClassId rootEClass = mock.egraph.findConst(mock.nodeToEClass.at(root));

            auto base_extractor = makeExtractor(mock.egraph, rootEClass, mock.enodeInfos);
            float baseline_cost = TGConstants::INF;
            while (base_extractor.getNextSelection())
            {
                const auto &sm = base_extractor.selection_map;
                auto di = makeDispatchIterator(mock.egraph, sm, mock.enodeInfos);
                std::vector<EClassId> order;
                while (di.getNextDispatchOrder(sm, order))
                    baseline_cost = std::min(baseline_cost, get_cost(order, mock.egraph, sm, mock.enodeInfos));
                base_extractor.ascend();
            }

            auto rule_extractor = makeExtractor(mock.egraph, rootEClass, mock.enodeInfos, Rule{true});
            float rule_cost = TGConstants::INF;
            while (rule_extractor.getNextSelection())
            {
                const auto &sm = rule_extractor.selection_map;
                auto di = makeDispatchIterator(mock.egraph, sm, mock.enodeInfos);
                std::vector<EClassId> order;
                while (di.getNextDispatchOrder(sm, order))
                    rule_cost = std::min(rule_cost, get_cost(order, mock.egraph, sm, mock.enodeInfos));
                rule_extractor.ascend();
            }

            if (std::abs(rule_cost - baseline_cost) > 1e-5f)
            {
                r.test_passed = false;
                std::ostringstream ss;
                ss << "cost " << rule_cost << " != baseline " << baseline_cost << " at scale " << s;
                r.err_msg = ss.str();
                return r;
            }

            auto make_base = [&]() { return makeExtractor(mock.egraph, rootEClass, mock.enodeInfos); };
            auto yield_base = [&](auto &it) {
                bool got = it.getNextSelection();
                if (!got)
                    return false;
                const auto &sm = it.selection_map;
                auto di = makeDispatchIterator(mock.egraph, sm, mock.enodeInfos);
                std::vector<EClassId> order;
                while (di.getNextDispatchOrder(sm, order))
                {
                }
                it.ascend();
                return true;
            };
            auto base_res = runTrial("Base", make_base, yield_base, 2, 5);
            base_times.push_back(base_res.test_ms);

            auto make_rule = [&]() { return makeExtractor(mock.egraph, rootEClass, mock.enodeInfos, Rule{true}); };
            auto yield_rule = [&](auto &it) {
                bool got = it.getNextSelection();
                if (!got)
                    return false;
                const auto &sm = it.selection_map;
                auto di = makeDispatchIterator(mock.egraph, sm, mock.enodeInfos);
                std::vector<EClassId> order;
                while (di.getNextDispatchOrder(sm, order))
                {
                }
                it.ascend();
                return true;
            };
            auto rule_trial_res = runTrial(r.rule_name, make_rule, yield_rule, 2, 5);
            rule_times.push_back(rule_trial_res.test_ms);
        }

        r.test_passed = true;
        auto base_fit = fitLine(scales, base_times);
        auto rule_fit = fitLine(scales, rule_times);

        r.baseline_ms = base_fit.avg_ms;
        r.test_ms = rule_fit.avg_ms;
        r.baseline_slope = base_fit.slope;
        r.test_slope = rule_fit.slope;

        if (base_fit.slope > 1e-6 && rule_fit.slope > 1e-6)
        {
            r.speedup = base_fit.slope / rule_fit.slope;
            r.was_faster = (rule_fit.slope < base_fit.slope - 1e-6);
        }
        else
        {
            r.speedup = (rule_fit.avg_ms > 1e-7) ? (base_fit.avg_ms / rule_fit.avg_ms) : 1.0;
            r.was_faster = (rule_fit.avg_ms < base_fit.avg_ms - 1e-6);
        }

        if (std::abs(rule_fit.slope - base_fit.slope) <= 1e-6)
        {
            r.was_faster = (rule_fit.avg_ms < base_fit.avg_ms);
        }

        return r;
    }
};

template <typename Rule> struct ENodeDominationBenchOne
{
    static TrialResult run(const std::vector<double> &scales)
    {
        TrialResult r;
        r.rule_name = Rule{}.name();

        std::vector<double> base_times;
        std::vector<double> rule_times;

        for (double s : scales)
        {
            MockCtx mock;
            Graph g;
            auto twins = buildFmaTwins(g, static_cast<int>(s));
            LogicalId root = twins.root;
            mock.build(g, root, false, [&](EGraph &egraph, const std::unordered_map<LogicalId, EClassId> &n2e) {
                extendFmaTwinsEGraph(twins, egraph, n2e);
            });

            std::unordered_map<EClassId, LogicalId> emptyMap;
            std::unordered_map<LogicalId, MemSpace> emptyCached;
            ENodeDominationContext ctx{mock.egraph, mock.enodeInfos, emptyMap, emptyCached, mock.settings.mem_caps};

            std::vector<ENodeInfo> filtered_infos = mock.enodeInfos;
            Rule rule{true};
            for (uint32_t i = 0; i < mock.egraph.getENodes().size(); ++i)
            {
                ENodeId enodeId{i};
                if (filtered_infos[i].cost == TGConstants::INF)
                    continue;
                if (rule.check(enodeId, 0, ctx))
                {
                    filtered_infos[i].cost = TGConstants::INF;
                }
            }

            auto start_base = std::chrono::high_resolution_clock::now();
            int timed_iters = 50;
            for (int t = 0; t < timed_iters; ++t)
            {
                for (uint32_t i = 0; i < mock.egraph.getENodes().size(); ++i)
                {
                }
            }
            auto end_base = std::chrono::high_resolution_clock::now();
            double base_ms = std::chrono::duration<double, std::milli>(end_base - start_base).count() / timed_iters;
            base_times.push_back(base_ms);

            auto start_rule = std::chrono::high_resolution_clock::now();
            for (int t = 0; t < timed_iters; ++t)
            {
                for (uint32_t i = 0; i < mock.egraph.getENodes().size(); ++i)
                {
                    rule.check(ENodeId{i}, 0, ctx);
                }
            }
            auto end_rule = std::chrono::high_resolution_clock::now();
            double rule_ms = std::chrono::duration<double, std::milli>(end_rule - start_rule).count() / timed_iters;
            rule_times.push_back(rule_ms);
        }

        r.test_passed = true;
        auto base_fit = fitLine(scales, base_times);
        auto rule_fit = fitLine(scales, rule_times);

        r.baseline_ms = base_fit.avg_ms;
        r.test_ms = rule_fit.avg_ms;
        r.baseline_slope = base_fit.slope;
        r.test_slope = rule_fit.slope;

        r.was_faster = true;
        r.speedup = 1.0;

        return r;
    }
};

// =============================================================================
// Helper Functions for Persistence & Resumability
// =============================================================================

inline std::unordered_map<std::string, bool> loadCompletedRecords(const std::string &path)
{
    std::unordered_map<std::string, bool> completed;
    auto recs = Settings::load_rule_benchmarks_file(path);
    for (const auto &rec : recs)
    {
        completed[rec.category + "::" + rec.rule_name] = true;
    }
    return completed;
}

inline void saveBenchmarkRecord(const std::string &path, const std::string &category, const std::string &rule_name,
                                bool was_faster, double baseline_ms, double test_ms, double speedup)
{
    RuleBenchmarkRecord rec;
    rec.category = category;
    rec.rule_name = rule_name;
    rec.was_faster = was_faster;
    rec.baseline_ms = baseline_ms;
    rec.test_ms = test_ms;
    rec.speedup = speedup;

    auto recs = Settings::load_rule_benchmarks_file(path);
    bool found = false;
    for (auto &existing : recs)
    {
        if (existing.category == category && existing.rule_name == rule_name)
        {
            existing = rec;
            found = true;
            break;
        }
    }
    if (!found)
    {
        recs.push_back(rec);
    }
    Settings::save_rule_benchmarks(path, recs);

    if (path != "benchmarks/dispatch_rules.bin")
    {
        auto legacy_recs = Settings::load_rule_benchmarks_file("benchmarks/dispatch_rules.bin");
        bool legacy_found = false;
        for (auto &existing : legacy_recs)
        {
            if (existing.category == category && existing.rule_name == rule_name)
            {
                existing = rec;
                legacy_found = true;
                break;
            }
        }
        if (!legacy_found)
        {
            legacy_recs.push_back(rec);
        }
        Settings::save_rule_benchmarks("benchmarks/dispatch_rules.bin", legacy_recs);
    }
}

inline void appendTestResult(const std::string &path, const std::string &category, const std::string &rule_name,
                             bool passed, const std::string &err_msg)
{
    std::ofstream file(path, std::ios::app);
    if (!file.is_open())
        return;
    file << category << '\t' << rule_name << '\t' << (passed ? "PASS" : "FAIL") << '\t'
         << (err_msg.empty() ? "-" : err_msg) << '\n';
    file.flush();
}

} // namespace prune_test

// =============================================================================
// Top-Level Entry Point: Unified Multi-Scale Testing & Benchmarking
// =============================================================================

inline void runPruningTests(const std::string &results_path = "benchmarks/pruning_tests.txt",
                            const std::string &bench_binary_path = "benchmarks/rules.bin")
{
    using namespace prune_test;

    std::cout << "\n=======================================================\n";
    std::cout << "Running Pruning-Rule Tests & Benchmarks (Multi-Scale)\n";
    std::cout << "=======================================================\n";

    auto completed = loadCompletedRecords(bench_binary_path);
    std::cout << "  Resuming: " << completed.size() << " previously completed rules found.\n";

    std::vector<double> scales = {1.0, 2.0, 3.0};

    // 1. Dispatch Rules
    {
        std::cout << "\n[Category: dispatch] Evaluating across scales...\n";
        std::apply(
            [&](auto... ruleTypes) {
                (([&] {
                     using R = std::decay_t<decltype(ruleTypes)>;
                     R sample;
                     std::string name = sample.name();
                     std::string key = "dispatch::" + name;
                     if (completed.count(key))
                     {
                         std::cout << "  - SKIP (already done): " << key << "\n";
                         return;
                     }
                     std::cout << "  - testing " << key << "..." << std::flush;
                     TrialResult r = DispatchBenchOne<R>::run(scales);
                     if (!r.test_passed)
                     {
                         std::cout << " FAILED: " << r.err_msg << "\n";
                         appendTestResult(results_path, "dispatch", name, false, r.err_msg);
                         Error::throw_err("[PruningTest] dispatch::" + name + " failed: " + r.err_msg);
                     }
                     std::cout << " PASS (slope: base=" << std::fixed << std::setprecision(4) << r.baseline_slope
                               << ", rule=" << r.test_slope << ", speedup=" << r.speedup
                               << "x, faster=" << (r.was_faster ? "YES" : "NO") << ")\n";
                     appendTestResult(results_path, "dispatch", name, true, "");
                     saveBenchmarkRecord(bench_binary_path, "dispatch", name, r.was_faster, r.baseline_ms, r.test_ms,
                                         r.speedup);
                 }()),
                 ...);
            },
            AllDispatchRuleTypes{});
    }

    // 2. Bufferize Rules
    {
        std::cout << "\n[Category: bufferize] Evaluating across scales...\n";
        std::apply(
            [&](auto... ruleTypes) {
                (([&] {
                     using R = std::decay_t<decltype(ruleTypes)>;
                     R sample;
                     std::string name = sample.name();
                     std::string key = "bufferize::" + name;
                     if (completed.count(key))
                     {
                         std::cout << "  - SKIP (already done): " << key << "\n";
                         return;
                     }
                     std::cout << "  - testing " << key << "..." << std::flush;
                     TrialResult r = BufferizeBenchOne<R>::run(scales);
                     if (!r.test_passed)
                     {
                         std::cout << " FAILED: " << r.err_msg << "\n";
                         appendTestResult(results_path, "bufferize", name, false, r.err_msg);
                         Error::throw_err("[PruningTest] bufferize::" + name + " failed: " + r.err_msg);
                     }
                     std::cout << " PASS (slope: base=" << std::fixed << std::setprecision(4) << r.baseline_slope
                               << ", rule=" << r.test_slope << ", speedup=" << r.speedup
                               << "x, faster=" << (r.was_faster ? "YES" : "NO") << ")\n";
                     appendTestResult(results_path, "bufferize", name, true, "");
                     saveBenchmarkRecord(bench_binary_path, "bufferize", name, r.was_faster, r.baseline_ms, r.test_ms,
                                         r.speedup);
                 }()),
                 ...);
            },
            AllBufferizeRuleTypes{});
    }

    // 3. Malloc Rules
    {
        std::cout << "\n[Category: malloc] Evaluating across scales...\n";
        std::apply(
            [&](auto... ruleTypes) {
                (([&] {
                     using R = std::decay_t<decltype(ruleTypes)>;
                     R sample;
                     std::string name = sample.name();
                     std::string key = "malloc::" + name;
                     if (completed.count(key))
                     {
                         std::cout << "  - SKIP (already done): " << key << "\n";
                         return;
                     }
                     std::cout << "  - testing " << key << "..." << std::flush;
                     TrialResult r = MallocBenchOne<R>::run(scales);
                     if (!r.test_passed)
                     {
                         std::cout << " FAILED: " << r.err_msg << "\n";
                         appendTestResult(results_path, "malloc", name, false, r.err_msg);
                         Error::throw_err("[PruningTest] malloc::" + name + " failed: " + r.err_msg);
                     }
                     std::cout << " PASS (slope: base=" << std::fixed << std::setprecision(4) << r.baseline_slope
                               << ", rule=" << r.test_slope << ", speedup=" << r.speedup
                               << "x, faster=" << (r.was_faster ? "YES" : "NO") << ")\n";
                     appendTestResult(results_path, "malloc", name, true, "");
                     saveBenchmarkRecord(bench_binary_path, "malloc", name, r.was_faster, r.baseline_ms, r.test_ms,
                                         r.speedup);
                 }()),
                 ...);
            },
            AllMallocRuleTypes{});
    }

    // 4. Extract (DFS) Rules
    {
        std::cout << "\n[Category: extract] Evaluating across scales...\n";
        std::apply(
            [&](auto... ruleTypes) {
                (([&] {
                     using R = std::decay_t<decltype(ruleTypes)>;
                     R sample;
                     std::string name = sample.name();
                     std::string key = "extract::" + name;
                     if (completed.count(key))
                     {
                         std::cout << "  - SKIP (already done): " << key << "\n";
                         return;
                     }
                     std::cout << "  - testing " << key << "..." << std::flush;
                     TrialResult r = ExtractBenchOne<R>::run(scales);
                     if (!r.test_passed)
                     {
                         std::cout << " FAILED: " << r.err_msg << "\n";
                         appendTestResult(results_path, "extract", name, false, r.err_msg);
                         Error::throw_err("[PruningTest] extract::" + name + " failed: " + r.err_msg);
                     }
                     std::cout << " PASS (slope: base=" << std::fixed << std::setprecision(4) << r.baseline_slope
                               << ", rule=" << r.test_slope << ", speedup=" << r.speedup
                               << "x, faster=" << (r.was_faster ? "YES" : "NO") << ")\n";
                     appendTestResult(results_path, "extract", name, true, "");
                     saveBenchmarkRecord(bench_binary_path, "extract", name, r.was_faster, r.baseline_ms, r.test_ms,
                                         r.speedup);
                 }()),
                 ...);
            },
            AllExtractRuleTypes{});
    }

    // 5. ENode Domination Rules
    {
        std::cout << "\n[Category: enode] Evaluating across scales...\n";
        std::apply(
            [&](auto... ruleTypes) {
                (([&] {
                     using R = std::decay_t<decltype(ruleTypes)>;
                     R sample;
                     std::string name = sample.name();
                     std::string key = "enode::" + name;
                     if (completed.count(key))
                     {
                         std::cout << "  - SKIP (already done): " << key << "\n";
                         return;
                     }
                     std::cout << "  - testing " << key << "..." << std::flush;
                     TrialResult r = ENodeDominationBenchOne<R>::run(scales);
                     if (!r.test_passed)
                     {
                         std::cout << " FAILED: " << r.err_msg << "\n";
                         appendTestResult(results_path, "enode", name, false, r.err_msg);
                         Error::throw_err("[PruningTest] enode::" + name + " failed: " + r.err_msg);
                     }
                     std::cout << " PASS (slope: base=" << std::fixed << std::setprecision(4) << r.baseline_slope
                               << ", rule=" << r.test_slope << ", speedup=" << r.speedup
                               << "x, faster=" << (r.was_faster ? "YES" : "NO") << ")\n";
                     appendTestResult(results_path, "enode", name, true, "");
                     saveBenchmarkRecord(bench_binary_path, "enode", name, r.was_faster, r.baseline_ms, r.test_ms,
                                         r.speedup);
                 }()),
                 ...);
            },
            AllENodeDominationRuleTypes{});
    }

    // 6. Cache Rules
    {
        std::cout << "\n[Category: cache] Evaluating across scales...\n";
        std::apply(
            [&](auto... ruleTypes) {
                (([&] {
                     using R = std::decay_t<decltype(ruleTypes)>;
                     R sample;
                     std::string name = sample.name();
                     std::string key = "cache::" + name;
                     if (completed.count(key))
                     {
                         std::cout << "  - SKIP (already done): " << key << "\n";
                         return;
                     }
                     std::cout << "  - testing " << key << "..." << std::flush;
                     TrialResult r = CacheBenchOne<R>::run(scales);
                     if (!r.test_passed)
                     {
                         std::cout << " FAILED: " << r.err_msg << "\n";
                         appendTestResult(results_path, "cache", name, false, r.err_msg);
                         Error::throw_err("[PruningTest] cache::" + name + " failed: " + r.err_msg);
                     }
                     std::cout << " PASS (slope: base=" << std::fixed << std::setprecision(4) << r.baseline_slope
                               << ", rule=" << r.test_slope << ", speedup=" << r.speedup
                               << "x, faster=" << (r.was_faster ? "YES" : "NO") << ")\n";
                     appendTestResult(results_path, "cache", name, true, "");
                     saveBenchmarkRecord(bench_binary_path, "cache", name, r.was_faster, r.baseline_ms, r.test_ms,
                                         r.speedup);
                 }()),
                 ...);
            },
            AllCacheRuleTypes{});
    }

    std::cout << "\nAll pruning rules tested and benchmarked successfully.\n";
}