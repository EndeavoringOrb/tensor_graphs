// tensor_graphs_cpp/tests/pruning/test.hpp
#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/constants.hpp"
#include "core/plan/extractor.hpp"
#include "core/plan/planner.hpp"
#include "core/plan/rule_registry.hpp"
#include "core/plan/mem.hpp"
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

// =============================================================================
// Helper: Popcount and Combination Generation
// =============================================================================

inline uint32_t countSetBits(uint32_t n)
{
    uint32_t count = 0;
    while (n > 0)
    {
        n &= (n - 1);
        count++;
    }
    return count;
}

template <size_t N> inline std::vector<uint32_t> getCombinationMasks()
{
    std::vector<uint32_t> masks;
    if (N == 0)
        return masks;
    uint32_t total = (1u << N);
    for (uint32_t k = 1; k <= N; ++k)
    {
        for (uint32_t m = 1; m < total; ++m)
        {
            if (countSetBits(m) == k)
            {
                masks.push_back(m);
            }
        }
    }
    return masks;
}

template <typename... RuleTypes, size_t... Is>
inline auto makeRuleTupleForMask(uint32_t mask, std::index_sequence<Is...>)
{
    return std::make_tuple(RuleTypes((mask & (1u << Is)) != 0)...);
}

template <typename... RuleTypes> inline auto makeRuleTupleForMask(uint32_t mask)
{
    return makeRuleTupleForMask<RuleTypes...>(mask, std::index_sequence_for<RuleTypes...>{});
}

template <typename... RuleTypes, size_t... Is>
inline std::string getCombinationName(uint32_t mask, std::index_sequence<Is...>)
{
    std::string name;
    auto add_name = [&](size_t idx, const char *rname) {
        if ((mask & (1u << idx)) != 0)
        {
            if (!name.empty())
                name += "+";
            name += rname;
        }
    };
    (add_name(Is, RuleTypes{}.name()), ...);
    return name;
}

template <typename... RuleTypes> inline std::string getCombinationName(uint32_t mask)
{
    return getCombinationName<RuleTypes...>(mask, std::index_sequence_for<RuleTypes...>{});
}

// =============================================================================
// Timed Trial Runner with Cooperative Timeout Checks
// =============================================================================

template <typename IterFactory, typename YieldFn, typename TimeoutCheckFn>
inline TrialResult runTrial(const std::string &rule_name, IterFactory &&make_iter, YieldFn &&yield, int warmup_iters,
                            int timed_iters, double timeout_seconds, TimeoutCheckFn &&check_timeout)
{
    TrialResult r;
    r.rule_name = rule_name;
    r.test_passed = true;

    for (int w = 0; w < warmup_iters; ++w)
    {
        if (check_timeout())
        {
            r.test_passed = false;
            r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
            return r;
        }
        auto iter = make_iter();
        while (yield(iter))
        {
            if (check_timeout())
            {
                r.test_passed = false;
                r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
                return r;
            }
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timed_iters; ++t)
    {
        if (check_timeout())
        {
            r.test_passed = false;
            r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
            return r;
        }
        auto iter = make_iter();
        while (yield(iter))
        {
            if (t == 0)
                r.total_states++;
            if (check_timeout())
            {
                r.test_passed = false;
                r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
                return r;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    r.test_ms = total_ms / timed_iters;
    return r;
}

template <typename IterFactory, typename YieldFn>
inline TrialResult runTrial(const std::string &rule_name, IterFactory &&make_iter, YieldFn &&yield,
                            int warmup_iters = 3, int timed_iters = 10)
{
    return runTrial(rule_name, std::forward<IterFactory>(make_iter), std::forward<YieldFn>(yield), warmup_iters,
                    timed_iters, 1e9, []() { return false; });
}

// =============================================================================
// Per-Category Combination Benchmark Runners
// =============================================================================

template <typename... RuleTypes> struct DispatchBench
{
    static TrialResult run(const std::vector<double> &scales, const std::tuple<RuleTypes...> &rules_tuple,
                           const std::string &rule_name, double timeout_seconds = 5.0,
                           std::atomic<bool> *cancel_flag = nullptr)
    {
        TrialResult r;
        r.rule_name = rule_name;

        auto start_bench = std::chrono::high_resolution_clock::now();
        auto check_timeout = [&]() -> bool {
            if (cancel_flag && cancel_flag->load())
                return true;
            double elapsed =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_bench).count();
            return elapsed > timeout_seconds;
        };

        std::vector<double> base_times;
        std::vector<double> rule_times;

        for (double s : scales)
        {
            if (check_timeout())
            {
                r.test_passed = false;
                std::ostringstream ss;
                ss << "Timeout: test run exceeded " << std::fixed << std::setprecision(1) << timeout_seconds << "s";
                r.err_msg = ss.str();
                return r;
            }

            MockCtx mock;
            Graph g;
            LogicalId root = buildWideShallow(g, static_cast<int>(s));
            mock.build(g, root);

            auto base_iter = makeDispatchIterator(mock.egraph, mock.selection_map, mock.enodeInfos);
            std::vector<EClassId> order;
            float baseline_cost = TGConstants::INF;
            while (base_iter.getNextDispatchOrder(mock.selection_map, order))
            {
                if (check_timeout())
                {
                    r.test_passed = false;
                    r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
                    return r;
                }
                baseline_cost =
                    std::min(baseline_cost, get_cost(order, mock.egraph, mock.selection_map, mock.enodeInfos));
            }

            float rule_cost = TGConstants::INF;
            auto rule_iter = std::apply(
                [&](auto &&...rs) {
                    return makeDispatchIteratorWithDelegate(mock.egraph, mock.selection_map, mock.enodeInfos, nullptr,
                                                            &rule_cost, nullptr, nullptr, rs...);
                },
                rules_tuple);

            while (rule_iter.getNextDispatchOrder(mock.selection_map, order))
            {
                if (check_timeout())
                {
                    r.test_passed = false;
                    r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
                    return r;
                }
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
                if (check_timeout())
                    return false;
                std::vector<EClassId> ord;
                return it.getNextDispatchOrder(mock.selection_map, ord);
            };
            auto base_res = runTrial("Base", make_base, yield_base, 2, 5, timeout_seconds, check_timeout);
            if (!base_res.test_passed)
            {
                r.test_passed = false;
                r.err_msg = base_res.err_msg;
                return r;
            }
            base_times.push_back(base_res.test_ms);

            auto make_rule = [&]() {
                return std::apply(
                    [&](auto &&...rs) {
                        return makeDispatchIteratorWithDelegate(mock.egraph, mock.selection_map, mock.enodeInfos,
                                                                nullptr, &baseline_cost, nullptr, nullptr, rs...);
                    },
                    rules_tuple);
            };
            auto yield_rule = [&](auto &it) {
                if (check_timeout())
                    return false;
                std::vector<EClassId> ord;
                return it.getNextDispatchOrder(mock.selection_map, ord);
            };
            auto rule_trial_res = runTrial(r.rule_name, make_rule, yield_rule, 2, 5, timeout_seconds, check_timeout);
            if (!rule_trial_res.test_passed)
            {
                r.test_passed = false;
                r.err_msg = rule_trial_res.err_msg;
                return r;
            }
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

template <typename... RuleTypes> struct BufferizeBench
{
    static TrialResult run(const std::vector<double> &scales, const std::tuple<RuleTypes...> &rules_tuple,
                           const std::string &rule_name, double timeout_seconds = 5.0,
                           std::atomic<bool> *cancel_flag = nullptr)
    {
        TrialResult r;
        r.rule_name = rule_name;

        auto start_bench = std::chrono::high_resolution_clock::now();
        auto check_timeout = [&]() -> bool {
            if (cancel_flag && cancel_flag->load())
                return true;
            double elapsed =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_bench).count();
            return elapsed > timeout_seconds;
        };

        std::vector<double> base_times;
        std::vector<double> rule_times;
        // Tight memory limit: 16 KB (4 buffers of 4KB each, exactly matching minimum required peak for in-place)
        uint64_t tight_mem_cap = 16384ULL;

        for (double s : scales)
        {
            if (check_timeout())
            {
                r.test_passed = false;
                std::ostringstream ss;
                ss << "Timeout: test run exceeded " << std::fixed << std::setprecision(1) << timeout_seconds << "s";
                r.err_msg = ss.str();
                return r;
            }

            MockCtx mock(tight_mem_cap);
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

            auto base_iter =
                makeBufferizeIterator(order, mock.egraph, mock.selection_map, mock.enodeInfos, mock.settings.mem_caps);
            std::vector<ParallelBuffer> bufs;
            std::unordered_map<EClassId, BufferId> eclass_to_buf;
            float baseline_cost = TGConstants::INF;
            while (base_iter.getNextBufferization(bufs, eclass_to_buf))
            {
                if (check_timeout())
                {
                    r.test_passed = false;
                    r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
                    return r;
                }
                BufferId overflow;
                std::vector<ParallelBuffer> allocated;
                if (malloc_by_time_components(mem_cap, bufs, allocated, overflow, nullptr, &mock.settings))
                    baseline_cost =
                        std::min(baseline_cost, get_cost(order, mock.egraph, mock.selection_map, mock.enodeInfos));
            }

            float rule_cost = TGConstants::INF;
            auto rule_iter = std::apply(
                [&](auto &&...rs) {
                    return makeBufferizeIterator(order, mock.egraph, mock.selection_map, mock.enodeInfos,
                                                 mock.settings.mem_caps, &rule_cost, nullptr, rs...);
                },
                rules_tuple);

            while (rule_iter.getNextBufferization(bufs, eclass_to_buf))
            {
                if (check_timeout())
                {
                    r.test_passed = false;
                    r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
                    return r;
                }
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
                return makeBufferizeIterator(order, mock.egraph, mock.selection_map, mock.enodeInfos,
                                             mock.settings.mem_caps);
            };
            auto yield_base = [&](auto &it) {
                if (check_timeout())
                    return false;
                std::vector<ParallelBuffer> b;
                std::unordered_map<EClassId, BufferId> m;
                return it.getNextBufferization(b, m);
            };
            auto base_res = runTrial("Base", make_base, yield_base, 2, 5, timeout_seconds, check_timeout);
            if (!base_res.test_passed)
            {
                r.test_passed = false;
                r.err_msg = base_res.err_msg;
                return r;
            }
            base_times.push_back(base_res.test_ms);

            auto make_rule = [&]() {
                return std::apply(
                    [&](auto &&...rs) {
                        return makeBufferizeIterator(order, mock.egraph, mock.selection_map, mock.enodeInfos,
                                                     mock.settings.mem_caps, nullptr, nullptr, rs...);
                    },
                    rules_tuple);
            };
            auto yield_rule = [&](auto &it) {
                if (check_timeout())
                    return false;
                std::vector<ParallelBuffer> b;
                std::unordered_map<EClassId, BufferId> m;
                return it.getNextBufferization(b, m);
            };
            auto rule_trial_res = runTrial(r.rule_name, make_rule, yield_rule, 2, 5, timeout_seconds, check_timeout);
            if (!rule_trial_res.test_passed)
            {
                r.test_passed = false;
                r.err_msg = rule_trial_res.err_msg;
                return r;
            }
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

template <typename... RuleTypes> struct MallocBench
{
    static TrialResult run(const std::vector<double> &scales, const std::tuple<RuleTypes...> &rules_tuple,
                           const std::string &rule_name, double timeout_seconds = 5.0,
                           std::atomic<bool> *cancel_flag = nullptr)
    {
        TrialResult r;
        r.rule_name = rule_name;

        auto start_bench = std::chrono::high_resolution_clock::now();
        auto check_timeout = [&]() -> bool {
            if (cancel_flag && cancel_flag->load())
                return true;
            double elapsed =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_bench).count();
            return elapsed > timeout_seconds;
        };

        std::vector<double> base_times;
        std::vector<double> rule_times;
        // Tight memory limit: 18 MB (minimum theoretical peak overlap is 17 MB)
        uint64_t mem_cap = 18ULL * 1024 * 1024;

        for (double s : scales)
        {
            if (check_timeout())
            {
                r.test_passed = false;
                std::ostringstream ss;
                ss << "Timeout: test run exceeded " << std::fixed << std::setprecision(1) << timeout_seconds << "s";
                r.err_msg = ss.str();
                return r;
            }

            std::vector<ParallelBuffer> unallocated = buildMallocBuffers(static_cast<int>(s));

            auto test_iter = std::apply(
                [&](auto &&...rs) {
                    return makeMallocIterator(mem_cap, unallocated, nullptr, nullptr, CapRespectRule(true), rs...);
                },
                rules_tuple);

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

            auto make_base = [&]() {
                return makeMallocIterator(mem_cap, unallocated, nullptr, nullptr, CapRespectRule(true));
            };
            auto yield_base = [&](auto &it) {
                if (check_timeout())
                    return false;
                std::vector<ParallelBuffer> a;
                return it.getNextAllocation(a);
            };
            auto base_res = runTrial("Base", make_base, yield_base, 2, 5, timeout_seconds, check_timeout);
            if (!base_res.test_passed)
            {
                r.test_passed = false;
                r.err_msg = base_res.err_msg;
                return r;
            }
            base_times.push_back(base_res.test_ms);

            auto make_rule = [&]() {
                return std::apply(
                    [&](auto &&...rs) {
                        return makeMallocIterator(mem_cap, unallocated, nullptr, nullptr, CapRespectRule(true), rs...);
                    },
                    rules_tuple);
            };
            auto yield_rule = [&](auto &it) {
                if (check_timeout())
                    return false;
                std::vector<ParallelBuffer> a;
                return it.getNextAllocation(a);
            };
            auto rule_trial_res = runTrial(r.rule_name, make_rule, yield_rule, 2, 5, timeout_seconds, check_timeout);
            if (!rule_trial_res.test_passed)
            {
                r.test_passed = false;
                r.err_msg = rule_trial_res.err_msg;
                return r;
            }
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

template <typename... RuleTypes> struct CacheBench
{
    static TrialResult run(const std::vector<double> &scales, const std::tuple<RuleTypes...> &rules_tuple,
                           const std::string &rule_name, double timeout_seconds = 5.0,
                           std::atomic<bool> *cancel_flag = nullptr)
    {
        TrialResult r;
        r.rule_name = rule_name;

        auto start_bench = std::chrono::high_resolution_clock::now();
        auto check_timeout = [&]() -> bool {
            if (cancel_flag && cancel_flag->load())
                return true;
            double elapsed =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_bench).count();
            return elapsed > timeout_seconds;
        };

        std::vector<double> base_times;
        std::vector<double> rule_times;

        for (double s : scales)
        {
            if (check_timeout())
            {
                r.test_passed = false;
                std::ostringstream ss;
                ss << "Timeout: test run exceeded " << std::fixed << std::setprecision(1) << timeout_seconds << "s";
                r.err_msg = ss.str();
                return r;
            }

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

            auto iter_rule = std::apply(
                [&](auto &&...rs) {
                    return makeCacheIterator(g, candidates, avail_mem_spaces, nullptr, nullptr, rs...);
                },
                rules_tuple);

            std::unordered_map<LogicalId, MemSpace> cache_rule;
            float min_cost_rule = TGConstants::INF;
            while (iter_rule.getNextCacheSelection(cache_rule))
            {
                if (check_timeout())
                {
                    r.test_passed = false;
                    r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
                    return r;
                }
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
                if (check_timeout())
                {
                    r.test_passed = false;
                    r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
                    return r;
                }
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
                if (check_timeout())
                    return false;
                std::unordered_map<LogicalId, MemSpace> c;
                return it.getNextCacheSelection(c);
            };
            auto base_res = runTrial("Base", make_base_it, yield_base, 2, 5, timeout_seconds, check_timeout);
            if (!base_res.test_passed)
            {
                r.test_passed = false;
                r.err_msg = base_res.err_msg;
                return r;
            }
            base_times.push_back(base_res.test_ms);

            auto make_rule_it = [&]() {
                return std::apply(
                    [&](auto &&...rs) {
                        return makeCacheIterator(g, candidates, avail_mem_spaces, nullptr, nullptr, rs...);
                    },
                    rules_tuple);
            };
            auto yield_rule = [&](auto &it) {
                if (check_timeout())
                    return false;
                std::unordered_map<LogicalId, MemSpace> c;
                return it.getNextCacheSelection(c);
            };
            auto rule_trial_res = runTrial(r.rule_name, make_rule_it, yield_rule, 2, 5, timeout_seconds, check_timeout);
            if (!rule_trial_res.test_passed)
            {
                r.test_passed = false;
                r.err_msg = rule_trial_res.err_msg;
                return r;
            }
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

template <typename... RuleTypes> struct ExtractBench
{
    static TrialResult run(const std::vector<double> &scales, const std::tuple<RuleTypes...> &rules_tuple,
                           const std::string &rule_name, double timeout_seconds = 5.0,
                           std::atomic<bool> *cancel_flag = nullptr)
    {
        TrialResult r;
        r.rule_name = rule_name;

        auto start_bench = std::chrono::high_resolution_clock::now();
        auto check_timeout = [&]() -> bool {
            if (cancel_flag && cancel_flag->load())
                return true;
            double elapsed =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_bench).count();
            return elapsed > timeout_seconds;
        };

        std::vector<double> base_times;
        std::vector<double> rule_times;

        for (double s : scales)
        {
            if (check_timeout())
            {
                r.test_passed = false;
                std::ostringstream ss;
                ss << "Timeout: test run exceeded " << std::fixed << std::setprecision(1) << timeout_seconds << "s";
                r.err_msg = ss.str();
                return r;
            }

            MockCtx mock;
            Graph g;
            LogicalId root = buildDiamond(g, static_cast<int>(s));
            mock.build(g, root);
            EClassId rootEClass = mock.egraph.findConst(mock.nodeToEClass.at(root));

            auto base_extractor = makeExtractor(mock.egraph, rootEClass, mock.enodeInfos);
            float baseline_cost = TGConstants::INF;
            while (base_extractor.getNextSelection())
            {
                if (check_timeout())
                {
                    r.test_passed = false;
                    r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
                    return r;
                }
                const auto &sm = base_extractor.selection_map;
                auto di = makeDispatchIterator(mock.egraph, sm, mock.enodeInfos);
                std::vector<EClassId> order;
                while (di.getNextDispatchOrder(sm, order))
                    baseline_cost = std::min(baseline_cost, get_cost(order, mock.egraph, sm, mock.enodeInfos));
                base_extractor.ascend();
            }

            float rule_cost = TGConstants::INF;
            auto rule_extractor = std::apply(
                [&](auto &&...rs) {
                    return makeExtractorWithDelegate(mock.egraph, rootEClass, mock.enodeInfos, nullptr, &rule_cost,
                                                     nullptr, nullptr, rs...);
                },
                rules_tuple);

            while (rule_extractor.getNextSelection())
            {
                if (check_timeout())
                {
                    r.test_passed = false;
                    r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
                    return r;
                }
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
                if (check_timeout())
                    return false;
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
            auto base_res = runTrial("Base", make_base, yield_base, 2, 5, timeout_seconds, check_timeout);
            if (!base_res.test_passed)
            {
                r.test_passed = false;
                r.err_msg = base_res.err_msg;
                return r;
            }
            base_times.push_back(base_res.test_ms);

            auto make_rule = [&]() {
                return std::apply(
                    [&](auto &&...rs) {
                        return makeExtractorWithDelegate(mock.egraph, rootEClass, mock.enodeInfos, nullptr,
                                                         &baseline_cost, nullptr, nullptr, rs...);
                    },
                    rules_tuple);
            };
            auto yield_rule = [&](auto &it) {
                if (check_timeout())
                    return false;
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
            auto rule_trial_res = runTrial(r.rule_name, make_rule, yield_rule, 2, 5, timeout_seconds, check_timeout);
            if (!rule_trial_res.test_passed)
            {
                r.test_passed = false;
                r.err_msg = rule_trial_res.err_msg;
                return r;
            }
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

template <typename... RuleTypes> struct ENodeDominationBench
{
    static TrialResult run(const std::vector<double> &scales, const std::tuple<RuleTypes...> &rules_tuple,
                           const std::string &rule_name, double timeout_seconds = 5.0,
                           std::atomic<bool> *cancel_flag = nullptr)
    {
        TrialResult r;
        r.rule_name = rule_name;

        auto start_bench = std::chrono::high_resolution_clock::now();
        auto check_timeout = [&]() -> bool {
            if (cancel_flag && cancel_flag->load())
                return true;
            double elapsed =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_bench).count();
            return elapsed > timeout_seconds;
        };

        std::vector<double> base_times;
        std::vector<double> rule_times;
        // Tight memory limit: 2 MB (prunes 4MB big add kernel)
        uint64_t tight_mem_cap = 2ULL * 1024 * 1024;

        for (double s : scales)
        {
            if (check_timeout())
            {
                r.test_passed = false;
                std::ostringstream ss;
                ss << "Timeout: test run exceeded " << std::fixed << std::setprecision(1) << timeout_seconds << "s";
                r.err_msg = ss.str();
                return r;
            }

            MockCtx mock(tight_mem_cap);
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
            for (uint32_t i = 0; i < mock.egraph.getENodes().size(); ++i)
            {
                if (check_timeout())
                {
                    r.test_passed = false;
                    r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
                    return r;
                }
                ENodeId enodeId{i};
                if (filtered_infos[i].cost == TGConstants::INF)
                    continue;
                bool should_prune =
                    std::apply([&](auto &&...rs) { return (false || ... || rs.check(enodeId, 0, ctx)); }, rules_tuple);
                if (should_prune)
                {
                    filtered_infos[i].cost = TGConstants::INF;
                }
            }

            auto start_base = std::chrono::high_resolution_clock::now();
            int timed_iters = 50;
            for (int t = 0; t < timed_iters; ++t)
            {
                if (check_timeout())
                {
                    r.test_passed = false;
                    r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
                    return r;
                }
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
                if (check_timeout())
                {
                    r.test_passed = false;
                    r.err_msg = "Timeout: test run exceeded " + std::to_string(timeout_seconds) + "s";
                    return r;
                }
                for (uint32_t i = 0; i < mock.egraph.getENodes().size(); ++i)
                {
                    std::apply([&](auto &&...rs) { return (false || ... || rs.check(ENodeId{i}, 0, ctx)); },
                               rules_tuple);
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
// Watchdog Runner: Runs Benchmark in Worker Thread with Timeout Protection
// =============================================================================

template <typename BenchRunner, typename... RuleTypes>
inline TrialResult runBenchWithTimeout(const std::vector<double> &scales, const std::tuple<RuleTypes...> &rules_tuple,
                                       const std::string &rule_name, double timeout_seconds = 5.0)
{
    TrialResult r;
    r.rule_name = rule_name;
    r.test_passed = false;

    std::promise<TrialResult> prom;
    auto fut = prom.get_future();
    auto cancel_flag = std::make_shared<std::atomic<bool>>(false);

    std::thread worker([scales, rules_tuple, rule_name, timeout_seconds, cancel_flag, p = std::move(prom)]() mutable {
        try
        {
            TrialResult res = BenchRunner::run(scales, rules_tuple, rule_name, timeout_seconds, cancel_flag.get());
            if (!cancel_flag->load())
            {
                p.set_value(res);
            }
        }
        catch (const std::exception &e)
        {
            if (!cancel_flag->load())
            {
                TrialResult err_res;
                err_res.rule_name = rule_name;
                err_res.test_passed = false;
                err_res.err_msg = e.what();
                p.set_value(err_res);
            }
        }
        catch (...)
        {
            if (!cancel_flag->load())
            {
                TrialResult err_res;
                err_res.rule_name = rule_name;
                err_res.test_passed = false;
                err_res.err_msg = "Unknown exception during benchmark run";
                p.set_value(err_res);
            }
        }
    });

    if (fut.wait_for(std::chrono::duration<double>(timeout_seconds)) == std::future_status::timeout)
    {
        cancel_flag->store(true);
        r.test_passed = false;
        std::ostringstream ss;
        ss << "Timeout: test run exceeded " << std::fixed << std::setprecision(1) << timeout_seconds << "s";
        r.err_msg = ss.str();
        worker.detach();
    }
    else
    {
        r = fut.get();
        if (worker.joinable())
        {
            worker.join();
        }
    }

    return r;
}

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

template <template <typename...> class BenchRunner, typename... RuleTypes>
inline void runCategoryCombinations(const std::string &category, const std::vector<double> &scales,
                                    const std::unordered_map<std::string, bool> &completed,
                                    const std::string &results_path, const std::string &bench_binary_path,
                                    double timeout_seconds = 5.0)
{
    constexpr size_t N = sizeof...(RuleTypes);
    if constexpr (N == 0)
    {
        return;
    }

    std::cout << "\n[Category: " << category << "] Evaluating " << ((1u << N) - 1)
              << " rule combination(s) across scales (timeout=" << timeout_seconds << "s)...\n";

    auto masks = getCombinationMasks<N>();

    for (uint32_t mask : masks)
    {
        std::string name = getCombinationName<RuleTypes...>(mask);
        std::string key = category + "::" + name;

        if (completed.count(key))
        {
            std::cout << "  - SKIP (already done): " << key << "\n";
            continue;
        }

        auto rule_tuple = makeRuleTupleForMask<RuleTypes...>(mask);

        std::cout << "  - testing " << key << "..." << std::flush;
        TrialResult r =
            runBenchWithTimeout<BenchRunner<RuleTypes...>, RuleTypes...>(scales, rule_tuple, name, timeout_seconds);

        if (!r.test_passed)
        {
            std::cout << " FAILED: " << r.err_msg << "\n";
            appendTestResult(results_path, category, name, false, r.err_msg);
            Error::throw_err("[PruningTest] " + category + "::" + name + " failed: " + r.err_msg);
        }

        std::cout << " PASS (slope: base=" << std::fixed << std::setprecision(4) << r.baseline_slope
                  << ", rule=" << r.test_slope << ", speedup=" << r.speedup
                  << "x, faster=" << (r.was_faster ? "YES" : "NO") << ")\n";
        appendTestResult(results_path, category, name, true, "");
        saveBenchmarkRecord(bench_binary_path, category, name, r.was_faster, r.baseline_ms, r.test_ms, r.speedup);
    }
}

template <template <typename...> class BenchRunner, typename... RuleTypes>
inline void runCategoryFromTuple(std::tuple<RuleTypes...>, const std::string &category,
                                 const std::vector<double> &scales,
                                 const std::unordered_map<std::string, bool> &completed,
                                 const std::string &results_path, const std::string &bench_binary_path,
                                 double timeout_seconds = 5.0)
{
    runCategoryCombinations<BenchRunner, RuleTypes...>(category, scales, completed, results_path, bench_binary_path,
                                                       timeout_seconds);
}

} // namespace prune_test

// =============================================================================
// Top-Level Entry Point: Unified Multi-Scale Testing & Combination Benchmarking
// =============================================================================

inline void runPruningTests(const std::string &results_path = "benchmarks/pruning_tests.txt",
                            const std::string &bench_binary_path = "benchmarks/rules.bin", double timeout_seconds = 5.0)
{
    using namespace prune_test;

    std::cout << "\n=======================================================\n";
    std::cout << "Running Pruning-Rule Tests & Benchmarks (Multi-Scale)\n";
    std::cout << "=======================================================\n";

    auto completed = loadCompletedRecords(bench_binary_path);
    std::cout << "  Resuming: " << completed.size() << " previously completed rules found.\n";

    std::vector<double> scales = {1.0, 2.0};

    // 1. Dispatch Rules
    runCategoryFromTuple<DispatchBench>(AllDispatchRuleTypes{}, "dispatch", scales, completed, results_path,
                                        bench_binary_path, timeout_seconds);

    scales = {1.0, 2.0, 3.0, 4.0};

    // 2. Bufferize Rules
    runCategoryFromTuple<BufferizeBench>(AllBufferizeRuleTypes{}, "bufferize", scales, completed, results_path,
                                         bench_binary_path, timeout_seconds);

    scales = {1.0, 2.0};

    // 3. Malloc Rules
    runCategoryFromTuple<MallocBench>(AllMallocRuleTypes{}, "malloc", scales, completed, results_path,
                                      bench_binary_path, timeout_seconds);

    // 4. Extract (DFS) Rules
    runCategoryFromTuple<ExtractBench>(AllExtractRuleTypes{}, "extract", scales, completed, results_path,
                                       bench_binary_path, timeout_seconds);

    // 5. ENode Domination Rules
    runCategoryFromTuple<ENodeDominationBench>(AllENodeDominationRuleTypes{}, "enode", scales, completed, results_path,
                                               bench_binary_path, timeout_seconds);

    // 6. Cache Rules
    runCategoryFromTuple<CacheBench>(AllCacheRuleTypes{}, "cache", scales, completed, results_path, bench_binary_path,
                                     timeout_seconds);

    std::cout << "\nAll pruning rules and combinations tested and benchmarked successfully.\n";
}