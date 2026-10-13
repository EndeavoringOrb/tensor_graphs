// tensor_graphs_cpp/core/plan/primal_heuristic.hpp
#pragma once

// Root primal heuristic (docs/core/relaxation_bound.md, "Primal heuristic").
// Builds a complete plan directly: a cost-aware selection guided by the
// relaxation bound's per-region estimates, a cache set consistent across
// buckets, and a list-scheduled dispatch order. The plan is only a proposal:
// SearchEngine applies it to the root state, runs every propagator and
// evaluates it with the normal leaf evaluation before accepting it.

#include <algorithm>
#include <chrono>
#include <deque>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/plan/propagators/base.hpp"
#include "core/plan/relaxation_bound.hpp"

namespace plan
{

struct PrimalBucketPlan
{
    std::vector<int32_t> value;  // per local class: 0 inactive, else selection value
    std::vector<uint32_t> order; // dispatch order of the active classes
    double makespan = std::numeric_limits<double>::infinity();
};

struct PrimalPlan
{
    std::vector<PrimalBucketPlan> buckets;
    std::unordered_set<BaseEClassId> cached;
    double cost = std::numeric_limits<double>::infinity();
    std::vector<double> lambda;
    // Per bucket: engine weights and region estimates used to build the plan.
    std::vector<std::vector<double>> bucket_lambda;
    std::vector<std::vector<double>> estimates;
};

// Alternatives that are only legal when their class's base is cached
// (CacheExclusionPropagator removes them otherwise).
inline void markCacheAlternatives(const SearchState &state, RelaxationBucket &rb)
{
    static std::mutex memo_mutex;
    static std::unordered_map<uint64_t, bool> memo;
    const EGraph &egraph = state.bucket_egraphs.at(rb.bucket_idx);
    for (RelaxationClass &rc : rb.classes)
    {
        const EClass &cls = egraph.getEClass(rc.cid);
        for (RelaxationAlternative &alt : rc.alternatives)
        {
            const ENode &enode = egraph.getENode(cls.enodes.at(static_cast<size_t>(alt.value - 1)));
            if (rc.base == BaseEClassId{})
            {
                alt.requires_cache = false;
                continue;
            }
            if (enode.getOpType() == OpType::CACHE || enode.getOpType() == OpType::SCATTER)
            {
                alt.requires_cache = true;
                continue;
            }
            if (enode.getOpType() != OpType::FUSED)
            {
                alt.requires_cache = false;
                continue;
            }
            const uint64_t key = enode.getKernelId().value;
            {
                std::lock_guard<std::mutex> lock(memo_mutex);
                auto it = memo.find(key);
                if (it != memo.end())
                {
                    alt.requires_cache = it->second;
                    continue;
                }
            }
            const bool needs = isOpCacheOrScatter(enode);
            std::lock_guard<std::mutex> lock(memo_mutex);
            memo[key] = needs;
            alt.requires_cache = needs;
        }
    }
}

// Simulated dispatch order: repeatedly dispatch the ready operation that can
// start earliest (ties: longest remaining path), mirroring MakespanSimulator.
inline double listSchedule(const RelaxationBucket &rb, PrimalBucketPlan &plan)
{
    const uint32_t n = static_cast<uint32_t>(rb.classes.size());
    std::vector<uint32_t> remaining(n, 0);
    std::vector<std::vector<uint32_t>> parents(n);
    std::vector<uint32_t> active;
    std::vector<const RelaxationAlternative *> chosen(n, nullptr);
    for (uint32_t u = 0; u < n; ++u)
    {
        if (plan.value[u] <= 0)
            continue;
        active.push_back(u);
        for (const auto &candidate : rb.classes[u].alternatives)
            if (candidate.value == plan.value[u])
                chosen[u] = &candidate;
        if (!chosen[u])
            Error::throw_err("Primal heuristic: selected value is not a feasible alternative");
        for (uint32_t ch : chosen[u]->children)
        {
            if (plan.value[ch] <= 0)
                Error::throw_err("Primal heuristic: selected alternative needs an inactive child");
            ++remaining[u];
            parents[ch].push_back(u);
        }
    }
    auto altOf = [&](uint32_t u) -> const RelaxationAlternative & { return *chosen[u]; };
    // Longest remaining path (by duration) to any sink, for tie-breaking.
    std::vector<double> tail(n, 0.0);
    for (auto it = rb.topo.rbegin(); it != rb.topo.rend(); ++it)
    {
        const uint32_t u = *it;
        if (plan.value[u] <= 0)
            continue;
        double best = 0.0;
        for (uint32_t p : parents[u])
            best = std::max(best, tail[p]);
        tail[u] = best + altOf(u).duration;
    }
    std::vector<double> finish(n, 0.0), ready_at(n, 0.0);
    std::vector<double> engine_free(rb.engines.size(), 0.0);
    struct Item
    {
        double ready;
        double tail;
        uint32_t u;
    };
    // Per engine mask: operations waiting for inputs (by ready time) and
    // operations whose inputs are ready (by tail).
    struct Lane
    {
        uint32_t mask;
        std::vector<Item> waiting; // heap on ready (min)
        std::vector<Item> ready;   // heap on tail (max)
    };
    std::deque<Lane> lanes; // stable references while new lanes are added
    auto laneOf = [&](uint32_t mask) -> Lane & {
        for (Lane &lane : lanes)
            if (lane.mask == mask)
                return lane;
        lanes.push_back(Lane{mask, {}, {}});
        return lanes.back();
    };
    auto by_ready = [](const Item &a, const Item &b) { return a.ready > b.ready; };
    auto by_tail = [](const Item &a, const Item &b) { return a.tail < b.tail; };
    auto laneFree = [&](uint32_t mask) {
        double t = 0.0;
        for (uint32_t e = 0; e < engine_free.size(); ++e)
            if (mask & (1u << e))
                t = std::max(t, engine_free[e]);
        return t;
    };
    std::vector<uint32_t> instant; // views and sources: no engine
    auto push = [&](uint32_t u) {
        const RelaxationAlternative &alt = altOf(u);
        if (alt.is_view || alt.is_source)
        {
            instant.push_back(u);
            return;
        }
        Lane &lane = laneOf(alt.sched_mask);
        lane.waiting.push_back(Item{ready_at[u], tail[u], u});
        std::push_heap(lane.waiting.begin(), lane.waiting.end(), by_ready);
    };
    for (uint32_t u : active)
        if (remaining[u] == 0)
            push(u);
    plan.order.clear();
    double makespan = 0.0;
    auto complete = [&](uint32_t u) {
        plan.order.push_back(u);
        for (uint32_t p : parents[u])
        {
            ready_at[p] = std::max(ready_at[p], finish[u]);
            if (--remaining[p] == 0)
                push(p);
        }
    };
    while (true)
    {
        while (!instant.empty())
        {
            const uint32_t u = instant.back();
            instant.pop_back();
            finish[u] = altOf(u).is_source ? 0.0 : ready_at[u];
            complete(u);
        }
        Lane *best_lane = nullptr;
        double best_start = 0.0, best_tail = 0.0;
        for (Lane &lane : lanes)
        {
            const double free = laneFree(lane.mask);
            while (!lane.waiting.empty() && lane.waiting.front().ready <= free)
            {
                std::pop_heap(lane.waiting.begin(), lane.waiting.end(), by_ready);
                lane.ready.push_back(lane.waiting.back());
                lane.waiting.pop_back();
                std::push_heap(lane.ready.begin(), lane.ready.end(), by_tail);
            }
            double start;
            double tail_value;
            if (!lane.ready.empty())
            {
                start = free;
                tail_value = lane.ready.front().tail;
            }
            else if (!lane.waiting.empty())
            {
                start = lane.waiting.front().ready;
                tail_value = lane.waiting.front().tail;
            }
            else
                continue;
            if (!best_lane || start < best_start || (start == best_start && tail_value > best_tail))
            {
                best_lane = &lane;
                best_start = start;
                best_tail = tail_value;
            }
        }
        if (!best_lane)
            break;
        Item item;
        if (!best_lane->ready.empty())
        {
            std::pop_heap(best_lane->ready.begin(), best_lane->ready.end(), by_tail);
            item = best_lane->ready.back();
            best_lane->ready.pop_back();
        }
        else
        {
            std::pop_heap(best_lane->waiting.begin(), best_lane->waiting.end(), by_ready);
            item = best_lane->waiting.back();
            best_lane->waiting.pop_back();
        }
        const uint32_t u = item.u;
        const double start = std::max(ready_at[u], laneFree(best_lane->mask));
        finish[u] = start + altOf(u).duration;
        for (uint32_t e = 0; e < engine_free.size(); ++e)
            if (best_lane->mask & (1u << e))
                engine_free[e] = finish[u];
        makespan = std::max(makespan, finish[u]);
        complete(u);
    }
    if (plan.order.size() != active.size())
        Error::throw_err("Primal heuristic: list schedule did not dispatch every active class");
    plan.makespan = makespan;
    return makespan;
}

// Top-down selection: parents choose the alternative minimizing its weighted
// duration plus the region estimates of children that are not yet active.
inline void extractSelection(const RelaxationBucket &rb, const std::vector<double> &lambda,
                             const std::vector<double> &estimates, const std::unordered_set<BaseEClassId> &allowed_cache,
                             PrimalBucketPlan &plan)
{
    const uint32_t n = static_cast<uint32_t>(rb.classes.size());
    plan.value.assign(n, 0);
    std::vector<uint8_t> active(n, 0);
    active[rb.root] = 1;
    for (auto it = rb.topo.rbegin(); it != rb.topo.rend(); ++it)
    {
        const uint32_t u = *it;
        if (!active[u])
            continue;
        const RelaxationClass &rc = rb.classes[u];
        double best = std::numeric_limits<double>::infinity();
        const RelaxationAlternative *chosen = nullptr;
        for (const auto &alt : rc.alternatives)
        {
            if (alt.requires_cache && !allowed_cache.count(rc.base))
                continue;
            double scale = 0.0;
            for (uint32_t e = 0; e < lambda.size(); ++e)
                if (alt.engine_mask & (1u << e))
                    scale += lambda[e];
            double value = alt.duration * scale;
            for (uint32_t ch : alt.children)
                if (!active[ch])
                    value += estimates[ch];
            if (value < best - 1e-12)
            {
                best = value;
                chosen = &alt;
            }
        }
        if (!chosen)
            Error::throw_err("Primal heuristic: no usable alternative for an active class");
        plan.value[u] = chosen->value;
        for (uint32_t ch : chosen->children)
            active[ch] = 1;
    }
}

// Selection for a zero-weight bucket: its cost does not count, but every
// class whose base the weighted buckets cache must be selected here too.
// Parents first, each active class picks the alternative whose children can
// still reach the most inactive targets (ties: cheaper estimate).
inline void extractCovering(const RelaxationBucket &rb, const std::vector<double> &lambda,
                            const std::vector<double> &estimates, const std::unordered_set<BaseEClassId> &allowed_cache,
                            const std::unordered_set<BaseEClassId> &targets, PrimalBucketPlan &plan)
{
    const uint32_t n = static_cast<uint32_t>(rb.classes.size());
    std::vector<int32_t> target_index(n, -1);
    uint32_t nt = 0;
    for (uint32_t u = 0; u < n; ++u)
        if (targets.count(rb.classes[u].base))
            target_index[u] = static_cast<int32_t>(nt++);
    const size_t words = (nt + 63) / 64;
    std::vector<uint64_t> reach(static_cast<size_t>(n) * std::max<size_t>(words, 1), 0);
    auto row = [&](uint32_t u) { return &reach[static_cast<size_t>(u) * std::max<size_t>(words, 1)]; };
    for (uint32_t u : rb.topo)
    {
        uint64_t *ru = row(u);
        for (const auto &alt : rb.classes[u].alternatives)
            for (uint32_t ch : alt.children)
            {
                const uint64_t *rc = row(ch);
                for (size_t w = 0; w < words; ++w)
                    ru[w] |= rc[w];
                if (target_index[ch] >= 0)
                    ru[target_index[ch] >> 6] |= 1ull << (target_index[ch] & 63);
            }
    }
    std::vector<uint64_t> covered(std::max<size_t>(words, 1), 0);
    plan.value.assign(n, 0);
    std::vector<uint8_t> active(n, 0);
    active[rb.root] = 1;
    if (target_index[rb.root] >= 0)
        covered[target_index[rb.root] >> 6] |= 1ull << (target_index[rb.root] & 63);
    std::vector<uint64_t> alt_reach(std::max<size_t>(words, 1));
    for (auto it = rb.topo.rbegin(); it != rb.topo.rend(); ++it)
    {
        const uint32_t u = *it;
        if (!active[u])
            continue;
        const RelaxationClass &rc = rb.classes[u];
        const RelaxationAlternative *chosen = nullptr;
        int64_t best_reach = -1;
        double best_cost = std::numeric_limits<double>::infinity();
        for (const auto &alt : rc.alternatives)
        {
            if (alt.requires_cache && !allowed_cache.count(rc.base))
                continue;
            std::fill(alt_reach.begin(), alt_reach.end(), 0);
            for (uint32_t ch : alt.children)
            {
                const uint64_t *r = row(ch);
                for (size_t w = 0; w < words; ++w)
                    alt_reach[w] |= r[w];
                if (target_index[ch] >= 0)
                    alt_reach[target_index[ch] >> 6] |= 1ull << (target_index[ch] & 63);
            }
            int64_t count = 0;
            for (size_t w = 0; w < words; ++w)
                count += __builtin_popcountll(alt_reach[w] & ~covered[w]);
            double scale = 0.0;
            for (uint32_t e = 0; e < lambda.size(); ++e)
                if (alt.engine_mask & (1u << e))
                    scale += lambda[e];
            double cost = alt.duration * scale;
            for (uint32_t ch : alt.children)
                if (!active[ch])
                    cost += estimates[ch];
            if (count > best_reach || (count == best_reach && cost < best_cost - 1e-12))
            {
                best_reach = count;
                best_cost = cost;
                chosen = &alt;
            }
        }
        if (!chosen)
            Error::throw_err("Primal heuristic: no usable alternative for an active class");
        plan.value[u] = chosen->value;
        for (uint32_t ch : chosen->children)
        {
            active[ch] = 1;
            if (target_index[ch] >= 0)
                covered[target_index[ch] >> 6] |= 1ull << (target_index[ch] & 63);
        }
    }
}

struct PrimalBucketContext
{
    RelaxationBucket rb;
    std::unique_ptr<RelaxationBoundComputer> computer;
};

// Builds one plan per engine weighting (those with a consistent cache set),
// sorted by simulated weighted makespan. Throws only on internal
// inconsistencies; an empty result means no consistent plan was produced.
inline std::vector<PrimalPlan> buildRootPrimalPlans(const SearchState &state, std::vector<PrimalBucketContext> &contexts)
{
    const uint32_t nb = static_cast<uint32_t>(state.buckets.size());
    contexts.clear();
    contexts.resize(nb);
    auto parallel = [&](auto &&fn) {
        const unsigned hw = std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
        std::vector<std::thread> threads;
        std::vector<std::exception_ptr> errors(hw);
        for (unsigned t = 0; t < hw; ++t)
            threads.emplace_back([&, t]() {
                try
                {
                    for (uint32_t b = t; b < nb; b += hw)
                        fn(b);
                }
                catch (...)
                {
                    errors[t] = std::current_exception();
                }
            });
        for (auto &thread : threads)
            thread.join();
        for (auto &error : errors)
            if (error)
                std::rethrow_exception(error);
    };
    bool usable = true;
    parallel([&](uint32_t b) {
        contexts[b].rb = buildRelaxationBucket(state, b);
        if (!contexts[b].rb.acyclic || !contexts[b].rb.root_feasible)
            return;
        markCacheAlternatives(state, contexts[b].rb);
        contexts[b].computer = std::make_unique<RelaxationBoundComputer>(contexts[b].rb);
    });
    for (uint32_t b = 0; b < nb; ++b)
        if (!contexts[b].computer || !contexts[b].computer->ready())
            usable = false;
    std::vector<PrimalPlan> plans;
    if (!usable)
        return plans;

    // Engine weightings to try: the union of engines over buckets, mapped
    // per bucket by identity.
    std::vector<Engine> engines;
    for (const auto &ctx : contexts)
        for (const Engine &e : ctx.rb.engines)
            if (std::find(engines.begin(), engines.end(), e) == engines.end())
                engines.push_back(e);
    std::vector<std::vector<double>> weightings;
    if (engines.size() <= 1)
        weightings.push_back(std::vector<double>(engines.size(), 1.0));
    else if (engines.size() == 2)
        for (double t : {0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.9})
        {
            // Put the larger weight on the engine with more work at t = 0.5
            // is unknown here; try both orientations.
            weightings.push_back({t, 1.0 - t});
            if (t != 0.5)
                weightings.push_back({1.0 - t, t});
        }
    else
    {
        weightings.push_back(std::vector<double>(engines.size(), 1.0 / static_cast<double>(engines.size())));
        for (size_t e = 0; e < engines.size(); ++e)
        {
            std::vector<double> w(engines.size(), 0.25 / static_cast<double>(engines.size() - 1));
            w[e] = 0.75;
            weightings.push_back(w);
        }
    }

    std::unordered_set<BaseEClassId> cache_possible;
    std::unordered_map<BaseEClassId, VarId> cache_var;
    for (const auto &[base, var] : state.cached_vars)
    {
        cache_var[base] = var;
        if (state.domains[var].contains(1))
            cache_possible.insert(base);
    }

    for (const auto &global_lambda : weightings)
    {
        std::vector<std::vector<double>> lambda(nb), estimates(nb);
        for (uint32_t b = 0; b < nb; ++b)
        {
            lambda[b].assign(contexts[b].rb.engines.size(), 0.0);
            for (size_t e = 0; e < contexts[b].rb.engines.size(); ++e)
            {
                const size_t g = static_cast<size_t>(
                    std::find(engines.begin(), engines.end(), contexts[b].rb.engines[e]) - engines.begin());
                lambda[b][e] = global_lambda[g];
            }
        }
        parallel([&](uint32_t b) {
            contexts[b].computer->evaluate(lambda[b]);
            contexts[b].computer->regionEstimates(estimates[b]);
        });
        std::unordered_set<BaseEClassId> allowed = cache_possible;
        PrimalPlan plan;
        plan.buckets.resize(nb);
        bool consistent = false;
        size_t used_count = 0;
        for (int round = 0; round < 64 && !consistent; ++round)
        {
            auto weighted = [&](uint32_t b) { return !(b < state.bucket_weights.size()) || state.bucket_weights[b] > 0.0f; };
            parallel([&](uint32_t b) {
                if (weighted(b))
                    extractSelection(contexts[b].rb, lambda[b], estimates[b], allowed, plan.buckets[b]);
            });
            // Cache set: bases whose cache-only alternatives the weighted
            // buckets use, plus bases root propagation forced to be cached.
            std::unordered_set<BaseEClassId> used;
            for (uint32_t b = 0; b < nb; ++b)
            {
                if (!weighted(b))
                    continue;
                const auto &rb = contexts[b].rb;
                for (uint32_t u = 0; u < rb.classes.size(); ++u)
                {
                    const int32_t v = plan.buckets[b].value[u];
                    if (v <= 0)
                        continue;
                    for (const auto &alt : rb.classes[u].alternatives)
                        if (alt.value == v && alt.requires_cache)
                            used.insert(rb.classes[u].base);
                }
            }
            for (const auto &[base, var] : cache_var)
                if (state.domains[var].isFixed() && state.domains[var].fixedValue() == 1)
                    used.insert(base);
            parallel([&](uint32_t b) {
                if (!weighted(b))
                    extractCovering(contexts[b].rb, lambda[b], estimates[b], used, used, plan.buckets[b]);
            });
            // A cached base must be selected in every bucket.
            std::unordered_set<BaseEClassId> violations;
            for (uint32_t b = 0; b < nb; ++b)
            {
                const auto &rb = contexts[b].rb;
                for (uint32_t u = 0; u < rb.classes.size(); ++u)
                    if (plan.buckets[b].value[u] <= 0 && used.count(rb.classes[u].base))
                        violations.insert(rb.classes[u].base);
            }

            // Cache budget per memory space: drop the largest bases first.
            std::unordered_map<MemSpace, uint64_t> bytes;
            std::vector<std::pair<uint64_t, BaseEClassId>> by_size;
            for (const BaseEClassId &base : used)
            {
                if (violations.count(base))
                    continue;
                const VarInfo &info = state.var_infos[cache_var.at(base)];
                bytes[info.mem_space] += info.size_bytes;
                by_size.push_back({info.size_bytes, base});
            }
            std::sort(by_size.rbegin(), by_size.rend());
            for (const auto &[size, base] : by_size)
            {
                const VarInfo &info = state.var_infos[cache_var.at(base)];
                if (bytes[info.mem_space] <= state.getMemoryCap(info.mem_space))
                    continue;
                bytes[info.mem_space] -= size;
                violations.insert(base);
            }
            used_count = used.size();
            if (violations.empty())
            {
                plan.cached = used;
                consistent = true;
                break;
            }
            bool removed = false;
            for (const BaseEClassId &base : violations)
                removed |= allowed.erase(base) > 0;
            if (!removed)
                break; // forced by root propagation: no consistent plan here
        }
        if (!consistent)
        {
            LOG(INFO) << "[PrimalHeuristic] weighting " << global_lambda[0] << ": no consistent cache set";
            continue;
        }
        parallel([&](uint32_t b) { listSchedule(contexts[b].rb, plan.buckets[b]); });
        double total = 0.0;
        for (uint32_t b = 0; b < nb; ++b)
        {
            const double w = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0;
            total += w * plan.buckets[b].makespan;
        }
        plan.cost = total;
        plan.lambda = global_lambda;
        plan.bucket_lambda = lambda;
        plan.estimates = estimates;
        {
            std::ostringstream os;
            for (size_t e = 0; e < global_lambda.size(); ++e)
                os << (e ? "," : "") << global_lambda[e];
            double load0 = 0.0, load1 = 0.0;
            const auto &rb0 = contexts[0].rb;
            for (uint32_t u = 0; u < rb0.classes.size(); ++u)
                for (const auto &alt : rb0.classes[u].alternatives)
                    if (alt.value == plan.buckets[0].value[u])
                    {
                        if (alt.engine_mask & 1u) load0 += alt.duration;
                        if (alt.engine_mask & 2u) load1 += alt.duration;
                    }
            LOG(INFO) << "[PrimalHeuristic] weighting [" << os.str() << "] cost=" << total << " bucket0 makespan="
                      << plan.buckets[0].makespan << " loads=" << load0 << "/" << load1 << " cached=" << used_count
                      << " disallowed=" << (cache_possible.size() - allowed.size());
        }
        // Different weightings often produce the same selection; keep one.
        bool duplicate = false;
        for (const PrimalPlan &other : plans)
        {
            bool same = other.cached == plan.cached;
            for (uint32_t b = 0; same && b < nb; ++b)
                same = other.buckets[b].value == plan.buckets[b].value;
            duplicate |= same;
        }
        if (!duplicate)
            plans.push_back(std::move(plan));
    }
    std::sort(plans.begin(), plans.end(), [](const PrimalPlan &a, const PrimalPlan &b) { return a.cost < b.cost; });
    return plans;
}


// Local search on one bucket's selection. A move switches one active class
// to another alternative; newly needed classes pick alternatives greedily by
// the plan's estimates and classes nobody needs any more become inactive.
// Cached bases must stay selected and cache-only alternatives are limited to
// cached bases, so the cache set stays consistent across buckets. Moves are
// screened by the maximum engine load and accepted only if the list-scheduled
// makespan decreases. Stops when a full pass finds no improvement or the
// deadline passes.
inline bool improveBucketLocalSearch(const RelaxationBucket &rb, const std::vector<double> &lambda,
                                     const std::vector<double> &estimates,
                                     const std::unordered_set<BaseEClassId> &cached, PrimalBucketPlan &plan,
                                     std::chrono::steady_clock::time_point deadline, uint64_t &evaluations)
{
    const uint32_t n = static_cast<uint32_t>(rb.classes.size());
    const size_t ne = rb.engines.size();
    std::vector<int32_t> alt_of(n, -1);
    for (uint32_t u = 0; u < n; ++u)
    {
        if (plan.value[u] <= 0)
            continue;
        const auto &alts = rb.classes[u].alternatives;
        for (uint32_t a = 0; a < alts.size(); ++a)
            if (alts[a].value == plan.value[u])
                alt_of[u] = static_cast<int32_t>(a);
        if (alt_of[u] < 0)
            Error::throw_err("Primal local search: selected value is not a feasible alternative");
    }
    std::vector<uint32_t> refs(n, 0);
    refs[rb.root] = 1;
    for (uint32_t u = 0; u < n; ++u)
        if (alt_of[u] >= 0)
            for (uint32_t ch : rb.classes[u].alternatives[static_cast<size_t>(alt_of[u])].children)
                ++refs[ch];
    std::vector<double> loads(ne, 0.0);
    auto addLoad = [&](uint32_t u, int32_t a, double sign) {
        const auto &alt = rb.classes[u].alternatives[static_cast<size_t>(a)];
        for (size_t e = 0; e < ne; ++e)
            if (alt.engine_mask & (1u << e))
                loads[e] += sign * alt.duration;
    };
    for (uint32_t u = 0; u < n; ++u)
        if (alt_of[u] >= 0)
            addLoad(u, alt_of[u], 1.0);
    auto maxLoad = [&]() { return ne ? *std::max_element(loads.begin(), loads.end()) : 0.0; };
    auto usable = [&](uint32_t u, const RelaxationAlternative &alt) {
        return !alt.requires_cache || cached.count(rb.classes[u].base) > 0;
    };
    auto greedyAlt = [&](uint32_t u) -> int32_t {
        const auto &alts = rb.classes[u].alternatives;
        int32_t best = -1;
        double best_cost = std::numeric_limits<double>::infinity();
        for (uint32_t a = 0; a < alts.size(); ++a)
        {
            if (!usable(u, alts[a]))
                continue;
            double scale = 0.0;
            for (size_t e = 0; e < ne; ++e)
                if (alts[a].engine_mask & (1u << e))
                    scale += lambda[e];
            double cost = alts[a].duration * scale;
            for (uint32_t ch : alts[a].children)
                if (alt_of[ch] < 0)
                    cost += estimates[ch];
            if (cost < best_cost - 1e-12)
            {
                best_cost = cost;
                best = static_cast<int32_t>(a);
            }
        }
        return best;
    };
    // Undo log: (class, previous alternative, previous refs), first touch only.
    std::vector<uint32_t> touched_stamp(n, 0);
    uint32_t stamp = 0;
    struct Touch
    {
        uint32_t u;
        int32_t alt;
        uint32_t refs;
    };
    std::vector<Touch> undo;
    std::vector<double> undo_loads;
    bool violated = false;
    auto touch = [&](uint32_t u) {
        if (touched_stamp[u] != stamp)
        {
            touched_stamp[u] = stamp;
            undo.push_back(Touch{u, alt_of[u], refs[u]});
        }
    };
    std::function<void(uint32_t)> incRef, decRef;
    incRef = [&](uint32_t c) {
        touch(c);
        if (refs[c]++ > 0)
            return;
        const int32_t a = greedyAlt(c);
        if (a < 0)
        {
            violated = true;
            return;
        }
        alt_of[c] = a;
        addLoad(c, a, 1.0);
        for (uint32_t ch : rb.classes[c].alternatives[static_cast<size_t>(a)].children)
            incRef(ch);
    };
    decRef = [&](uint32_t c) {
        touch(c);
        if (refs[c] == 0)
            Error::throw_err("Primal local search: reference count underflow");
        if (--refs[c] > 0)
            return;
        if (cached.count(rb.classes[c].base))
            violated = true; // a cached base must stay selected
        const int32_t a = alt_of[c];
        addLoad(c, a, -1.0);
        alt_of[c] = -1;
        for (uint32_t ch : rb.classes[c].alternatives[static_cast<size_t>(a)].children)
            decRef(ch);
    };
    auto rollback = [&]() {
        for (auto it = undo.rbegin(); it != undo.rend(); ++it)
        {
            alt_of[it->u] = it->alt;
            refs[it->u] = it->refs;
        }
        loads = undo_loads;
    };
    auto writeValues = [&](std::vector<int32_t> &out) {
        out.assign(n, 0);
        for (uint32_t u = 0; u < n; ++u)
            if (alt_of[u] >= 0)
                out[u] = rb.classes[u].alternatives[static_cast<size_t>(alt_of[u])].value;
    };
    PrimalBucketPlan trial;
    double best_makespan = plan.makespan;
    bool improved_any = false;
    bool improved = true;
    while (improved && std::chrono::steady_clock::now() < deadline)
    {
        improved = false;
        for (auto it = rb.topo.rbegin(); it != rb.topo.rend(); ++it)
        {
            const uint32_t u = *it;
            if (alt_of[u] < 0 || rb.classes[u].alternatives.size() < 2)
                continue;
            if (std::chrono::steady_clock::now() >= deadline)
                break;
            const int32_t current = alt_of[u];
            for (uint32_t a = 0; a < rb.classes[u].alternatives.size(); ++a)
            {
                if (static_cast<int32_t>(a) == current || alt_of[u] != current ||
                    !usable(u, rb.classes[u].alternatives[a]))
                    continue;
                ++stamp;
                undo.clear();
                undo_loads = loads;
                violated = false;
                const double before = maxLoad();
                touch(u);
                addLoad(u, current, -1.0);
                alt_of[u] = static_cast<int32_t>(a);
                addLoad(u, static_cast<int32_t>(a), 1.0);
                for (uint32_t ch : rb.classes[u].alternatives[a].children)
                    incRef(ch);
                for (uint32_t ch : rb.classes[u].alternatives[static_cast<size_t>(current)].children)
                    decRef(ch);
                if (violated || maxLoad() > before + 1e-9)
                {
                    rollback();
                    continue;
                }
                writeValues(trial.value);
                ++evaluations;
                const double makespan = listSchedule(rb, trial);
                if (makespan < best_makespan - 1e-9)
                {
                    best_makespan = makespan;
                    plan.value = trial.value;
                    plan.order = trial.order;
                    plan.makespan = makespan;
                    improved = true;
                    improved_any = true;
                    break; // the class has a new current alternative
                }
                rollback();
            }
        }
    }
    return improved_any;
}

// Improves every weighted bucket in parallel until convergence or deadline.
inline bool improvePrimalPlan(const SearchState &state, const std::vector<PrimalBucketContext> &contexts,
                              PrimalPlan &plan, std::chrono::steady_clock::time_point deadline,
                              uint64_t &evaluations)
{
    const uint32_t nb = static_cast<uint32_t>(contexts.size());
    std::vector<uint8_t> improved(nb, 0);
    std::vector<uint64_t> evals(nb, 0);
    const unsigned hw = std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(hw);
    for (unsigned t = 0; t < hw; ++t)
        threads.emplace_back([&, t]() {
            try
            {
                for (uint32_t b = t; b < nb; b += hw)
                {
                    const float w = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0f;
                    if (!(w > 0.0f))
                        continue;
                    improved[b] = improveBucketLocalSearch(contexts[b].rb, plan.bucket_lambda[b], plan.estimates[b],
                                                           plan.cached, plan.buckets[b], deadline, evals[b]);
                }
            }
            catch (...)
            {
                errors[t] = std::current_exception();
            }
        });
    for (auto &thread : threads)
        thread.join();
    for (auto &error : errors)
        if (error)
            std::rethrow_exception(error);
    double total = 0.0;
    bool any = false;
    for (uint32_t b = 0; b < nb; ++b)
    {
        const double w = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0;
        total += w * plan.buckets[b].makespan;
        any |= improved[b] != 0;
        evaluations += evals[b];
    }
    plan.cost = total;
    return any;
}

} // namespace plan
