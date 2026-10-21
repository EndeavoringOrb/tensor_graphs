// tensor_graphs_cpp/core/plan/primal_heuristic.hpp
#pragma once

// Root primal heuristic (docs/core/relaxation_bound.md, "Primal heuristic").
// Builds a complete plan directly: a cost-aware selection guided by the
// relaxation bound's per-region estimates, a cache set consistent across
// buckets, and a list-scheduled dispatch order. The plan is only a proposal:
// SearchEngine applies it to the root state, runs every propagator and
// evaluates it with the normal leaf evaluation before accepting it.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <tuple>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/plan/propagators/base.hpp"
#include "core/common/thread_pool.hpp"
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
            if (alt.is_alias || alt.value <= 0 || static_cast<size_t>(alt.value - 1) >= cls.enodes.size())
            {
                alt.requires_cache = false;
                continue;
            }
            // A class without a base has no cache variable: its cache-only
            // alternatives are marked too and never usable (no base is ever
            // cached), as CacheRequirementPropagator rejects them.
            const ENode &enode = egraph.getENode(cls.enodes.at(static_cast<size_t>(alt.value - 1)));
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
inline double listSchedule(const RelaxationBucket &rb, PrimalBucketPlan &plan,
                           std::vector<double> *finish_out = nullptr)
{
    const uint32_t n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
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
        for (uint32_t ch : childrenOf(*chosen[u]))
        {
            if (ch >= n || plan.value[ch] <= 0)
                Error::throw_err("Primal heuristic: selected alternative needs an inactive child");
            ++remaining[u];
            parents[ch].push_back(u);
        }
    }
    auto altOf = [&](uint32_t u) -> const RelaxationAlternative & { return *chosen[u]; };
    // Longest remaining path (by duration) to any sink, for tie-breaking.
    std::vector<double> tail(n, -1.0);
    std::vector<uint8_t> visited(n, 0);
    bool cycle_detected = false;
    std::function<double(uint32_t)> getTail = [&](uint32_t u) -> double {
        if (visited[u] == 1)
        {
            cycle_detected = true;
            return 0.0;
        }
        if (visited[u] == 2)
            return tail[u];
        visited[u] = 1;
        double best = 0.0;
        for (uint32_t p : parents[u])
        {
            best = std::max(best, getTail(p));
            if (cycle_detected)
                return 0.0;
        }
        visited[u] = 2;
        return tail[u] = best + altOf(u).duration;
    };
    for (uint32_t u : active)
    {
        getTail(u);
        if (cycle_detected)
            return std::numeric_limits<double>::infinity();
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
        return std::numeric_limits<double>::infinity();
    if (finish_out)
        *finish_out = std::move(finish);
    plan.makespan = makespan;
    return makespan;
}

// Maps helper class choices and alias chains back to original classes and zeros helper classes.
inline void resolveAliases(const RelaxationBucket &rb, PrimalBucketPlan &plan)
{
    const uint32_t orig_n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
    if (orig_n == rb.classes.size())
        return;

    auto resolveClassValue = [&](uint32_t start_v) -> int32_t {
        uint32_t curr = start_v;
        while (curr < rb.classes.size() && plan.value[curr] == -1)
        {
            const auto &alts = rb.classes[curr].alternatives;
            uint32_t next_v = UINT32_MAX;
            for (const auto &alt : alts)
            {
                if (alt.is_alias && alt.value == -1)
                {
                    if (!alt.children.empty())
                        next_v = alt.children[0];
                    break;
                }
            }
            if (next_v == UINT32_MAX || next_v == curr)
                break;
            curr = next_v;
        }
        if (curr < rb.classes.size() && plan.value[curr] > 0)
            return plan.value[curr];
        return 0;
    };

    for (uint32_t u = 0; u < orig_n; ++u)
    {
        if (plan.value[u] == -1)
            plan.value[u] = resolveClassValue(u);
    }

    for (uint32_t h = orig_n; h < rb.classes.size(); ++h)
    {
        if (plan.value[h] != 0)
        {
            const uint32_t u = rb.orig_class_of[h];
            if (plan.value[u] <= 0)
            {
                const int32_t val = resolveClassValue(h);
                if (val > 0)
                    plan.value[u] = val;
            }
        }
    }

    for (uint32_t h = orig_n; h < rb.classes.size(); ++h)
        plan.value[h] = 0;
}

// Top-down selection: parents choose the alternative minimizing its weighted
// duration plus the region estimates of children that are not yet active.
inline void extractSelection(const RelaxationBucket &rb, const std::vector<double> &lambda,
                             const std::vector<double> &estimates, const std::unordered_set<BaseEClassId> &allowed_cache,
                             PrimalBucketPlan &plan)
{
    const uint32_t n = static_cast<uint32_t>(rb.classes.size());
    plan.value.assign(n, 0);
    // A class can be produced when one of its alternatives is allowed (cache
    // alternatives need an allowed base) and all its children can be.
    auto allowed = [&](uint32_t u, const RelaxationAlternative &alt) {
        return !alt.requires_cache || allowed_cache.count(rb.classes[u].base) > 0;
    };
    std::vector<uint8_t> producible(n, 0);
    for (uint32_t u : rb.topo)
        for (const auto &alt : rb.classes[u].alternatives)
        {
            bool ok = allowed(u, alt);
            for (uint32_t ch : alt.children)
                ok = ok && producible[ch];
            if (ok)
            {
                producible[u] = 1;
                break;
            }
        }
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
            bool ok = allowed(u, alt);
            for (uint32_t ch : alt.children)
                ok = ok && producible[ch];
            if (!ok)
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
    resolveAliases(rb, plan);
}

// Selection for a zero-weight bucket: its cost does not count, but every
// class whose base the weighted buckets cache must be selected here too.
// Bottom-up, every class keeps the alternative whose children's kept
// selections cover the most target value (ties: the cheaper estimate), so
// a class's coverage is exactly what following the kept alternatives
// selects; then the kept alternatives are followed from the root. Targets
// are worth `target_value` (1 when absent): the cache savings of their
// base, so a route to one expensive cached tensor beats one to several
// cheap ones.
inline void extractCovering(const RelaxationBucket &rb, const std::vector<double> &lambda,
                            const std::vector<double> &estimates, const std::unordered_set<BaseEClassId> &allowed_cache,
                            const std::unordered_set<BaseEClassId> &targets, PrimalBucketPlan &plan,
                            const std::unordered_map<BaseEClassId, double> *target_value = nullptr)
{
    const uint32_t n = static_cast<uint32_t>(rb.classes.size());
    const uint32_t orig_n = rb.orig_class_count ? rb.orig_class_count : n;
    std::vector<int32_t> target_index(n, -1);
    std::vector<double> value_of;
    uint32_t nt = 0;
    for (uint32_t u = 0; u < orig_n; ++u)
        if (targets.count(rb.classes[u].base))
        {
            target_index[u] = static_cast<int32_t>(nt++);
            double v = 1.0;
            if (target_value)
            {
                const auto it = target_value->find(rb.classes[u].base);
                v = it != target_value->end() ? it->second : 0.0;
            }
            value_of.push_back(v);
        }
    const size_t words = std::max<size_t>((nt + 63) / 64, 1);
    std::vector<uint64_t> cover(static_cast<size_t>(n) * words, 0);
    auto row = [&](uint32_t u) { return &cover[static_cast<size_t>(u) * words]; };
    auto valueOf = [&](const std::vector<uint64_t> &bits) {
        double v = 0.0;
        for (size_t w = 0; w < words; ++w)
            for (uint64_t b = bits[w]; b; b &= b - 1)
                v += value_of[w * 64 + static_cast<size_t>(__builtin_ctzll(b))];
        return v;
    };
    std::vector<int32_t> kept(n, -1);
    std::vector<uint64_t> bits(words), best_bits(words);
    for (uint32_t u : rb.topo)
    {
        const RelaxationClass &rc = rb.classes[u];
        double best_value = -1.0, best_cost = std::numeric_limits<double>::infinity();
        for (uint32_t a = 0; a < rc.alternatives.size(); ++a)
        {
            const auto &alt = rc.alternatives[a];
            if (alt.requires_cache && !allowed_cache.count(rc.base))
                continue;
            bool usable = true;
            std::fill(bits.begin(), bits.end(), 0);
            for (uint32_t ch : alt.children)
            {
                usable = usable && kept[ch] >= 0;
                const uint64_t *r = row(ch);
                for (size_t w = 0; w < words; ++w)
                    bits[w] |= r[w];
            }
            if (!usable)
                continue;
            const double v = valueOf(bits);
            double scale = 0.0;
            for (uint32_t e = 0; e < lambda.size(); ++e)
                if (alt.engine_mask & (1u << e))
                    scale += lambda[e];
            double cost = alt.duration * scale;
            for (uint32_t ch : alt.children)
                cost += estimates[ch];
            if (v > best_value + 1e-12 || (std::abs(v - best_value) <= 1e-12 && cost < best_cost - 1e-12))
            {
                best_value = v;
                best_cost = cost;
                kept[u] = static_cast<int32_t>(a);
                best_bits = bits;
            }
        }
        if (kept[u] < 0)
            continue; // not producible
        uint64_t *ru = row(u);
        std::copy(best_bits.begin(), best_bits.end(), ru);
        if (target_index[u] >= 0)
            ru[target_index[u] >> 6] |= 1ull << (target_index[u] & 63);
    }
    if (kept[rb.root] < 0)
        Error::throw_err("Primal heuristic: the covering bucket's root cannot be produced");
    plan.value.assign(n, 0);
    std::vector<uint8_t> active(n, 0);
    active[rb.root] = 1;
    for (auto it = rb.topo.rbegin(); it != rb.topo.rend(); ++it)
    {
        const uint32_t u = *it;
        if (!active[u])
            continue;
        const RelaxationAlternative &alt = rb.classes[u].alternatives[static_cast<size_t>(kept[u])];
        plan.value[u] = alt.value;
        for (uint32_t ch : alt.children)
            active[ch] = 1;
    }
    resolveAliases(rb, plan);
}

inline double selectionWork(const RelaxationBucket &rb, const std::vector<double> &lambda, const PrimalBucketPlan &plan)
{
    double work = 0.0;
    const uint32_t orig_n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
    for (uint32_t u = 0; u < orig_n; ++u)
        for (const auto &alt : rb.classes[u].alternatives)
            if (alt.value == plan.value[u])
            {
                double scale = 0.0;
                for (uint32_t e = 0; e < lambda.size(); ++e)
                    if (alt.engine_mask & (1u << e))
                        scale += lambda[e];
                work += alt.duration * scale;
            }
    return work;
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
        {
            LOG(INFO) << "[PrimalHeuristic] bucket " << b << " relaxation unusable: acyclic=" << contexts[b].rb.acyclic
                      << " root_feasible=" << contexts[b].rb.root_feasible
                      << " computer=" << (contexts[b].computer ? "not ready" : "none");
            usable = false;
        }
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

    // Two covering policies per weighting (see below): a weighted bucket that
    // leaves a cached base unselected takes the covering selection only when
    // it costs at most 3x its own selection, or always. Local search starts
    // from both; the better result is validated.
    // Two kinds of class estimates: region values of the relaxation, and
    // the cheap from-scratch costs (cheapEstimates). Region values treat
    // work below a class's region as paid, which suits a plan that shares
    // it (decode with cached weights) and misleads one that must produce it
    // itself (an incremental step whose cache cannot hold it); cheap
    // estimates price everything. Starts with region estimates stop after
    // kRegionStartSeconds of building (they evaluate the relaxation).
    struct Start
    {
        std::vector<double> global_lambda;
        double cover_limit;
        bool cheap;
    };
    std::vector<Start> starts;
    for (bool cheap : {true, false})
        for (const auto &global_lambda : weightings)
            for (double cover_limit : {3.0, std::numeric_limits<double>::infinity()})
                starts.push_back({global_lambda, cover_limit, cheap});
    // Buckets with the same relaxation share estimates.
    std::vector<uint32_t> same_as(nb);
    {
        std::vector<std::vector<uint64_t>> signatures(nb);
        for (uint32_t b = 0; b < nb; ++b)
        {
            signatures[b] = relaxationSignature(contexts[b].rb);
            same_as[b] = b;
            for (uint32_t c = 0; c < b; ++c)
                if (same_as[c] == c && signatures[c] == signatures[b])
                {
                    same_as[b] = c;
                    break;
                }
        }
    }
    constexpr double kRegionStartSeconds = 3.0;
    const auto build_start = std::chrono::steady_clock::now();
    for (const Start &start_policy : starts)
    {
        if (!start_policy.cheap &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count() > kRegionStartSeconds)
            break;
        const auto &global_lambda = start_policy.global_lambda;
        const double cover_limit = start_policy.cover_limit;
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
            if (same_as[b] != b)
                return;
            if (start_policy.cheap)
                contexts[b].computer->cheapEstimates(lambda[b], estimates[b]);
            else
            {
                contexts[b].computer->evaluate(lambda[b]);
                contexts[b].computer->regionEstimates(estimates[b]);
            }
        });
        for (uint32_t b = 0; b < nb; ++b)
            if (same_as[b] != b)
                estimates[b] = estimates[same_as[b]];
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
                const uint32_t orig_n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
                for (uint32_t u = 0; u < orig_n; ++u)
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
            // A cached base is worth its size: large cached tensors (weight
            // transforms, copies of the embedding) are the expensive ones to
            // produce. (Estimates price many of them at 0: a free view of a
            // class below its region.)
            std::unordered_map<BaseEClassId, double> saving;
            for (const BaseEClassId &base : used)
                saving[base] = static_cast<double>(state.var_infos[cache_var.at(base)].size_bytes);
            // Every bucket must select the cached bases. Unweighted buckets
            // cover them. A weighted bucket whose own selection left one out
            // takes the covering selection unless it costs over 3x as much
            // (caching pays off in every bucket; covering costs this one);
            // otherwise the base leaves the cache set below.
            parallel([&](uint32_t b) {
                const auto &rb = contexts[b].rb;
                const uint32_t orig_n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
                if (!weighted(b))
                {
                    extractCovering(rb, lambda[b], estimates[b], used, used, plan.buckets[b], &saving);
                    return;
                }
                bool missing = false;
                for (uint32_t u = 0; u < orig_n && !missing; ++u)
                    missing = plan.buckets[b].value[u] <= 0 && used.count(rb.classes[u].base) > 0;
                if (!missing)
                    return;
                PrimalBucketPlan covering;
                extractCovering(rb, lambda[b], estimates[b], used, used, covering, &saving);
                if (selectionWork(rb, lambda[b], covering) <= cover_limit * selectionWork(rb, lambda[b], plan.buckets[b]))
                    plan.buckets[b] = std::move(covering);
            });
            // A cached base must be selected in every bucket.
            std::unordered_set<BaseEClassId> violations;
            for (uint32_t b = 0; b < nb; ++b)
            {
                const auto &rb = contexts[b].rb;
                const uint32_t orig_n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
                for (uint32_t u = 0; u < orig_n; ++u)
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
        if (std::getenv("TG_DEBUG_PRIMAL_TOP"))
        {
            // Diagnostics: the most expensive selected alternatives of bucket 0.
            const auto &rb = contexts[0].rb;
            const uint32_t orig_n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
            const EGraph &eg = state.bucket_egraphs[0];
            std::vector<std::pair<double, std::string>> rows;
            for (uint32_t u = 0; u < orig_n; ++u)
                for (const auto &alt : rb.classes[u].alternatives)
                    if (alt.value == plan.buckets[0].value[u] && alt.duration > 0.0)
                    {
                        const EClass &cls = eg.getEClass(rb.classes[u].cid);
                        const ENode &en = eg.getENode(cls.enodes[alt.value - 1]);
                        rows.push_back({alt.duration, toString(en.getOpType()) + ":" + en.getOpName() + " " +
                                                          toString(cls.shape) + " base=" +
                                                          std::to_string(rb.classes[u].base.value) + " allowed_cache=" +
                                                          std::to_string(allowed.count(rb.classes[u].base))});
                    }
            std::sort(rows.rbegin(), rows.rend());
            for (size_t i = 0; i < rows.size() && i < 15; ++i)
                LOG(INFO) << "[debug primal top] " << rows[i].first << " " << rows[i].second;
        }
        if (const char *dbg_shape = std::getenv("TG_DEBUG_PRIMAL_SHAPE"))
        {
            // Diagnostics: alternatives and choice for classes of one shape in bucket 1.
            const uint32_t b = nb > 1 ? 1 : 0;
            const auto &rb = contexts[b].rb;
            const uint32_t orig_n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
            const EGraph &eg = state.bucket_egraphs[b];
            for (uint32_t u = 0; u < orig_n; ++u)
            {
                const EClass &cls = eg.getEClass(rb.classes[u].cid);
                std::string shape;
                for (uint32_t d : cls.shape)
                    shape += (shape.empty() ? "" : ",") + std::to_string(d);
                if (shape != dbg_shape)
                    continue;
                std::ostringstream os;
                os << "[debug primal] bucket=" << b << " cid=" << rb.classes[u].cid.value << " dtype="
                   << toString(cls.dtype) << " chosen=" << plan.buckets[b].value[u] << " allowed_cache="
                   << allowed.count(rb.classes[u].base) << " alts:";
                for (const auto &alt : rb.classes[u].alternatives)
                {
                    if (alt.is_alias)
                        continue;
                    const ENode &en = eg.getENode(cls.enodes[alt.value - 1]);
                    os << " [" << alt.value << " " << toString(en.getOpType()) << ":" << en.getOpName()
                       << " d=" << alt.duration << (alt.requires_cache ? " cache" : "") << "]";
                }
                LOG(INFO) << os.str();
            }
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
            const uint32_t orig_n0 = rb0.orig_class_count ? rb0.orig_class_count : static_cast<uint32_t>(rb0.classes.size());
            for (uint32_t u = 0; u < orig_n0; ++u)
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
                                     std::chrono::steady_clock::time_point deadline, uint64_t &evaluations,
                                     const std::unordered_set<BaseEClassId> *releasable = nullptr,
                                     std::unordered_map<BaseEClassId, uint32_t> *blocked = nullptr)
{
    const uint32_t n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
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
            for (uint32_t ch : childrenOf(rb.classes[u].alternatives[static_cast<size_t>(alt_of[u])]))
                if (ch < n)
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
        if (alt.is_alias)
            return false;
        return !alt.requires_cache || cached.count(rb.classes[u].base) > 0;
    };
    // Classes whose completion is in progress (the moved class and the
    // chain of newly activated classes above the current one): a child among
    // them would close a cycle (e.g. a COPY_TO back to the moved class).
    std::vector<uint8_t> completing(n, 0);
    auto greedyAlt = [&](uint32_t u) -> int32_t {
        const auto &alts = rb.classes[u].alternatives;
        int32_t best = -1;
        double best_cost = std::numeric_limits<double>::infinity();
        for (uint32_t a = 0; a < alts.size(); ++a)
        {
            if (!usable(u, alts[a]))
                continue;
            bool closes_cycle = false;
            for (uint32_t ch : childrenOf(alts[a]))
                closes_cycle |= ch < n && completing[ch];
            if (closes_cycle)
                continue;
            double scale = 0.0;
            for (size_t e = 0; e < ne; ++e)
                if (alts[a].engine_mask & (1u << e))
                    scale += lambda[e];
            double cost = alts[a].duration * scale;
            for (uint32_t ch : childrenOf(alts[a]))
                if (ch < n && alt_of[ch] < 0)
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
    std::vector<BaseEClassId> blocked_now; // cached bases a rejected move would release
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
        completing[c] = 1;
        for (uint32_t ch : childrenOf(rb.classes[c].alternatives[static_cast<size_t>(a)]))
            if (ch < n)
                incRef(ch);
        completing[c] = 0;
    };
    decRef = [&](uint32_t c) {
        touch(c);
        if (refs[c] == 0)
            Error::throw_err("Primal local search: reference count underflow");
        if (--refs[c] > 0)
            return;
        if (cached.count(rb.classes[c].base) && !(releasable && releasable->count(rb.classes[c].base)))
        {
            violated = true; // a cached base must stay selected
            blocked_now.push_back(rb.classes[c].base);
        }
        const int32_t a = alt_of[c];
        addLoad(c, a, -1.0);
        alt_of[c] = -1;
        for (uint32_t ch : childrenOf(rb.classes[c].alternatives[static_cast<size_t>(a)]))
            if (ch < n)
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
        out.assign(rb.classes.size(), 0);
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
            if (u >= n)
                continue;
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
                blocked_now.clear();
                const double before = maxLoad();
                touch(u);
                addLoad(u, current, -1.0);
                alt_of[u] = static_cast<int32_t>(a);
                addLoad(u, static_cast<int32_t>(a), 1.0);
                completing[u] = 1;
                for (uint32_t ch : childrenOf(rb.classes[u].alternatives[a]))
                    if (ch < n)
                        incRef(ch);
                completing[u] = 0;
                for (uint32_t ch : childrenOf(rb.classes[u].alternatives[static_cast<size_t>(current)]))
                    if (ch < n)
                        decRef(ch);
                if (violated || maxLoad() > before + 1e-9)
                {
                    if (blocked && violated && maxLoad() < before - 1e-9)
                        for (BaseEClassId base : blocked_now)
                            ++(*blocked)[base];
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

// Improves every weighted bucket of every plan in parallel (one task per plan
// and bucket) until convergence or deadline. Returns, per plan, whether any
// bucket improved.
inline std::vector<uint8_t> improvePrimalPlans(const SearchState &state, const std::vector<PrimalBucketContext> &contexts,
                                               const std::vector<PrimalPlan *> &plans,
                                               std::chrono::steady_clock::time_point deadline, uint64_t &evaluations,
                                               const std::unordered_set<BaseEClassId> *releasable = nullptr,
                                               std::unordered_map<BaseEClassId, uint32_t> *blocked = nullptr)
{
    const uint32_t nb = static_cast<uint32_t>(contexts.size());
    const size_t np = plans.size();
    const size_t tasks = np * nb;
    std::vector<uint8_t> improved(tasks, 0);
    std::vector<uint64_t> evals(tasks, 0);
    std::vector<std::unordered_map<BaseEClassId, uint32_t>> task_blocked(tasks);
    const unsigned hw = std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
    std::atomic<size_t> next_task{0};
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(hw);
    for (unsigned t = 0; t < hw; ++t)
        threads.emplace_back([&, t]() {
            try
            {
                for (size_t task = next_task++; task < tasks; task = next_task++)
                {
                    PrimalPlan &plan = *plans[task / nb];
                    const uint32_t b = static_cast<uint32_t>(task % nb);
                    const float w = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0f;
                    if (!(w > 0.0f))
                        continue;
                    improved[task] = improveBucketLocalSearch(contexts[b].rb, plan.bucket_lambda[b], plan.estimates[b],
                                                              plan.cached, plan.buckets[b], deadline, evals[task],
                                                              releasable, blocked ? &task_blocked[task] : nullptr);
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
    std::vector<uint8_t> any(np, 0);
    for (size_t i = 0; i < np; ++i)
    {
        PrimalPlan &plan = *plans[i];
        double total = 0.0;
        for (uint32_t b = 0; b < nb; ++b)
        {
            const size_t task = i * nb + b;
            const double w = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0;
            total += w * plan.buckets[b].makespan;
            any[i] |= improved[task];
            evaluations += evals[task];
            if (blocked)
                for (const auto &[base, count] : task_blocked[task])
                    (*blocked)[base] += count;
        }
        plan.cost = total;
    }
    return any;
}

inline bool improvePrimalPlan(const SearchState &state, const std::vector<PrimalBucketContext> &contexts,
                              PrimalPlan &plan, std::chrono::steady_clock::time_point deadline,
                              uint64_t &evaluations, const std::unordered_set<BaseEClassId> *releasable = nullptr,
                              std::unordered_map<BaseEClassId, uint32_t> *blocked = nullptr)
{
    return improvePrimalPlans(state, contexts, {&plan}, deadline, evaluations, releasable, blocked)[0] != 0;
}

// Cache-set moves. A cached base must stay selected in every bucket, which
// blocks local-search moves that would stop using it. For bases that block
// otherwise load-reducing moves, try dropping the base from the cache set:
// re-run local search with the base releasable, require that no weighted
// bucket still uses its cache-only alternatives, and keep the change if the
// weighted makespan decreases. Repeats until no candidate helps.
// Selects class u of a bucket plan (and, recursively, whatever its chosen
// alternatives need that is not selected yet): the usable alternative with
// the lowest weighted duration plus estimates of unselected children.
inline bool activateClass(const RelaxationBucket &rb, const std::vector<double> &lambda,
                          const std::vector<double> &estimates, const std::unordered_set<BaseEClassId> &cached,
                          PrimalBucketPlan &bp, uint32_t u)
{
    const uint32_t n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
    if (u >= n)
        return false;
    if (bp.value[u] > 0)
        return true;
    const auto &alts = rb.classes[u].alternatives;
    int32_t best = -1;
    double best_cost = std::numeric_limits<double>::infinity();
    for (uint32_t a = 0; a < alts.size(); ++a)
    {
        if (alts[a].is_alias)
            continue;
        if (alts[a].requires_cache && !cached.count(rb.classes[u].base))
            continue;
        double scale = 0.0;
        for (size_t e = 0; e < lambda.size(); ++e)
            if (alts[a].engine_mask & (1u << e))
                scale += lambda[e];
        double cost = alts[a].duration * scale;
        for (uint32_t ch : childrenOf(alts[a]))
            if (ch < n && bp.value[ch] <= 0)
                cost += estimates[ch];
        if (cost < best_cost - 1e-12)
        {
            best_cost = cost;
            best = static_cast<int32_t>(a);
        }
    }
    if (best < 0)
        return false;
    bp.value[u] = alts[static_cast<size_t>(best)].value;
    for (uint32_t ch : childrenOf(alts[static_cast<size_t>(best)]))
        if (ch < n && !activateClass(rb, lambda, estimates, cached, bp, ch))
            return false;
    return true;
}

// Applies a window selection (classes of `rb`, alternative indices) to the
// buckets of `group` (weighted buckets with that relaxation) in `trial`,
// adjusting the cache set: classes the selection needs and does not select
// are activated greedily, and may use the cache-only alternative of any base
// that can be cached (the base joins the cache set); classes nothing needs
// any more are dropped, and cached bases no bucket reads from the cache any
// more leave the cache set. Zero-weight buckets then cover the new cache set
// (extractCovering). False when the result is inconsistent (a bucket leaves
// a cached base unselected or reads an uncached base from the cache) or
// exceeds a memory cap. Leaves the makespans of changed buckets to the
// caller (`changed`).
inline bool spliceWindowSelection(const SearchState &state, const std::vector<PrimalBucketContext> &contexts,
                                  const std::vector<uint32_t> &group,
                                  const std::vector<RelaxationBoundComputer::WindowChoice> &selection,
                                  PrimalPlan &trial, std::vector<uint8_t> &changed)
{
    auto cacheable = [&](BaseEClassId base) {
        if (base == BaseEClassId{})
            return false;
        auto var = state.cached_vars.find(base);
        return var != state.cached_vars.end() && state.domains[var->second].contains(1);
    };
    // Cache set the closure may use: current bases plus every cacheable one.
    std::unordered_set<BaseEClassId> usable = trial.cached;
    bool any_change = false;
    for (uint32_t b : group)
    {
        const RelaxationBucket &rb = contexts[b].rb;
        const uint32_t n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
        auto origOf = [&](uint32_t u) { return u < n ? u : rb.orig_class_of[u]; };
        for (uint32_t u = 0; u < n; ++u)
            if (cacheable(rb.classes[u].base))
                usable.insert(rb.classes[u].base);
        PrimalBucketPlan &bp = trial.buckets[b];
        bool bucket_change = false;
        for (const auto &entry : selection)
        {
            const uint32_t u = entry.cls;
            const uint32_t uo = origOf(u);
            // Among equivalent alternatives, the one whose children the
            // bucket computes already (else the cheapest to add).
            uint32_t a = entry.alts.front();
            double least = std::numeric_limits<double>::infinity();
            for (uint32_t candidate : entry.alts)
            {
                double extra = 0.0;
                for (uint32_t ch : childrenOf(rb.classes[u].alternatives[candidate]))
                    if (ch < n && bp.value[ch] <= 0)
                        extra += trial.estimates[b][ch];
                if (extra < least - 1e-12)
                {
                    least = extra;
                    a = candidate;
                }
            }
            const RelaxationAlternative &alt = rb.classes[u].alternatives[a];
            if (alt.is_alias)
                continue;
            if (alt.requires_cache && !usable.count(rb.classes[uo].base))
                return false;
            bucket_change |= bp.value[uo] != alt.value;
            bp.value[uo] = alt.value;
        }
        if (!bucket_change)
            continue;
        any_change = true;
        changed[b] = 1;
        // Close the selection from the root: chosen alternatives' children
        // are activated (greedily when unselected); unreached classes drop.
        std::vector<uint8_t> reached(n, 0);
        std::vector<uint32_t> stack{rb.root};
        reached[rb.root] = 1;
        while (!stack.empty())
        {
            const uint32_t u = stack.back();
            stack.pop_back();
            if (bp.value[u] <= 0 &&
                !activateClass(rb, trial.bucket_lambda[b], trial.estimates[b], usable, bp, u))
                return false;
            const RelaxationAlternative *chosen = nullptr;
            for (const auto &alt : rb.classes[u].alternatives)
                if (alt.value == bp.value[u] && !alt.is_alias)
                    chosen = &alt;
            if (!chosen)
                return false;
            for (uint32_t ch : childrenOf(*chosen))
                if (ch < n && !reached[ch])
                {
                    reached[ch] = 1;
                    stack.push_back(ch);
                }
        }
        for (uint32_t u = 0; u < n; ++u)
            if (!reached[u])
                bp.value[u] = 0;
    }
    if (!any_change)
        return false;
    // The new cache set: bases some bucket reads from the cache.
    std::unordered_set<BaseEClassId> cached;
    for (uint32_t b = 0; b < contexts.size(); ++b)
    {
        const RelaxationBucket &rb = contexts[b].rb;
        const uint32_t n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
        for (uint32_t u = 0; u < n; ++u)
        {
            const int32_t v = trial.buckets[b].value[u];
            if (v <= 0)
                continue;
            for (const auto &alt : rb.classes[u].alternatives)
                if (alt.value == v && !alt.is_alias && alt.requires_cache)
                    cached.insert(rb.classes[u].base);
        }
    }
    // Bases cached before and still read keep their place; bases nothing
    // reads from the cache any more are released.
    const std::unordered_set<BaseEClassId> plan_cached = trial.cached;
    trial.cached = std::move(cached);
    // Memory caps.
    std::unordered_map<MemSpace, uint64_t> bytes;
    for (BaseEClassId base : trial.cached)
    {
        auto var = state.cached_vars.find(base);
        if (var == state.cached_vars.end())
            return false;
        const VarInfo &info = state.var_infos[var->second];
        bytes[info.mem_space] += info.size_bytes;
        if (bytes[info.mem_space] > state.getMemoryCap(info.mem_space))
            return false;
    }
    // Every bucket selects every cached base, reachably from its root.
    // Zero-weight buckets take a new covering selection when the cache set
    // changed; weighted ones must select them already.
    std::unordered_map<BaseEClassId, double> saving;
    for (const BaseEClassId &base : trial.cached)
        saving[base] = static_cast<double>(state.var_infos[state.cached_vars.at(base)].size_bytes);
    for (uint32_t b = 0; b < contexts.size(); ++b)
    {
        const RelaxationBucket &rb = contexts[b].rb;
        const uint32_t n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
        const double weight = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0;
        if (!(weight > 0.0) && trial.cached != plan_cached)
        {
            extractCovering(rb, trial.bucket_lambda[b], trial.estimates[b], trial.cached, trial.cached,
                            trial.buckets[b], &saving);
            changed[b] = 1;
        }
        std::unordered_set<BaseEClassId> selected;
        for (uint32_t u = 0; u < n; ++u)
        {
            const int32_t v = trial.buckets[b].value[u];
            if (v <= 0)
                continue;
            selected.insert(rb.classes[u].base);
            for (const auto &alt : rb.classes[u].alternatives)
                if (alt.value == v && !alt.is_alias && alt.requires_cache && !trial.cached.count(rb.classes[u].base))
                    return false;
        }
        for (uint32_t u = 0; u < n; ++u)
            if (trial.cached.count(rb.classes[u].base) && !selected.count(rb.classes[u].base))
                return false;
    }
    return true;
}

// Window moves: the relaxation's flexible interval partition splits a
// weighted bucket into windows along its chain of necessary tensors. For each
// window, the best selection a bounded search finds for the window's value
// (fastest max(chain, engine loads) inside the window) replaces the plan's
// choices there, in every weighted bucket with that relaxation at once, with
// the cache set adjusted to it (spliceWindowSelection). The move is kept if
// the weighted list-scheduled makespan decreases. This changes many classes
// together (e.g. a layer's norm and copies, or how a matmul is split), which
// single-class switches cannot.
// `partitioned(b)` may supply a computer over the same relaxation whose
// flexible partition is already computed (the bound's); otherwise the
// bucket's own computer computes one.
inline bool improveByWindows(const SearchState &state, std::vector<PrimalBucketContext> &contexts, PrimalPlan &plan,
                             std::chrono::steady_clock::time_point deadline, uint64_t &evaluations,
                             const std::function<RelaxationBoundComputer *(uint32_t)> &partitioned)
{
    // Search budget per window by how far the plan's time across it exceeds
    // the window's bound; windows the plan already meets are skipped.
    constexpr uint64_t kWindowSearchNodes = 200000, kLongSearchNodes = 2000000;
    constexpr double kTightExcess = 1e-6, kLargeExcess = 0.06;
    constexpr int kWindowSearchSeconds = 2; // per window (large windows' nodes are slow)
    // Weighted buckets grouped by the computer holding their partition
    // (buckets sharing a relaxation share it).
    std::vector<std::pair<RelaxationBoundComputer *, std::vector<uint32_t>>> groups;
    for (uint32_t b = 0; b < contexts.size(); ++b)
    {
        const double weight = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0;
        if (!(weight > 0.0) || !contexts[b].computer || !contexts[b].computer->ready())
            continue;
        RelaxationBoundComputer *computer = partitioned ? partitioned(b) : nullptr;
        if (!computer || computer->lastPartition().empty())
        {
            computer = contexts[b].computer.get();
            computer->flexibleIntervalBound(8, deadline);
        }
        auto it = std::find_if(groups.begin(), groups.end(), [&](const auto &g) { return g.first == computer; });
        if (it == groups.end())
            groups.push_back({computer, {b}});
        else
            it->second.push_back(b);
    }
    bool any = false;
    for (const auto &entry : groups)
    {
        RelaxationBoundComputer *computer = entry.first;
        const std::vector<uint32_t> &group = entry.second;
        const auto partition = computer->lastPartition();
        const auto values = computer->lastPartitionValues();
        // The plan's time across each window: from the first finish of a
        // member of the link before it to that of its closing link.
        std::vector<uint64_t> budget(partition.size(), kWindowSearchNodes);
        {
            const uint32_t b = group.front();
            const RelaxationBucket &rb = contexts[b].rb;
            const uint32_t n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
            PrimalBucketPlan scheduled = plan.buckets[b];
            std::vector<double> finish;
            listSchedule(rb, scheduled, &finish);
            const size_t k = computer->intervalChainLength();
            auto reached = [&](size_t link) {
                if (link == k)
                    return scheduled.makespan;
                double t = std::numeric_limits<double>::infinity();
                for (uint32_t m : computer->chainLinkMembers(link))
                {
                    const uint32_t mo = m < n ? m : rb.orig_class_of[m];
                    if (scheduled.value[mo] > 0)
                        t = std::min(t, finish[mo]);
                }
                return t;
            };
            if (values.size() == partition.size() && !finish.empty())
                for (size_t w = 0; w < partition.size(); ++w)
                {
                    const double span = reached(partition[w].second) -
                                        (partition[w].first == 0 ? 0.0 : reached(partition[w].first - 1));
                    const double excess = span - values[w];
                    budget[w] = !(excess > kTightExcess)  ? 0
                                : excess > kLargeExcess ? kLongSearchNodes
                                                        : kWindowSearchNodes;
                }
        }
        std::vector<std::vector<std::vector<RelaxationBoundComputer::WindowChoice>>> per_window(partition.size());
        ThreadPool::get().parallel_for(static_cast<uint32_t>(partition.size()), [&](uint32_t w) {
            if (budget[w] == 0 || std::chrono::steady_clock::now() >= deadline)
                return;
            per_window[w] = computer->windowSelections(
                partition[w].first, partition[w].second, 1, budget[w],
                std::min(deadline, std::chrono::steady_clock::now() + std::chrono::seconds(kWindowSearchSeconds)));
        });
        std::vector<std::vector<RelaxationBoundComputer::WindowChoice>> selections;
        for (auto &found : per_window)
            for (auto &selection : found)
                selections.push_back(std::move(selection));
        for (const auto &selection : selections)
        {
            if (std::chrono::steady_clock::now() >= deadline)
                break;
            if (selection.empty())
                continue;
            PrimalPlan trial = plan;
            std::vector<uint8_t> changed(contexts.size(), 0);
            if (!spliceWindowSelection(state, contexts, group, selection, trial, changed))
                continue;
            double total = 0.0;
            for (uint32_t b = 0; b < contexts.size(); ++b)
            {
                if (changed[b])
                {
                    ++evaluations;
                    listSchedule(contexts[b].rb, trial.buckets[b]);
                }
                const double w = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0;
                if (w > 0.0)
                    total += w * trial.buckets[b].makespan;
            }
            trial.cost = total;
            if (trial.cost < plan.cost - 1e-9)
            {
                plan = std::move(trial);
                any = true;
            }
        }
    }
    return any;
}

// Cache-set additions: a base whose cache-only alternative (an in-place
// update or a cached read) is cheaper than what a weighted bucket selects now.
// Adding it to the cache set makes every bucket keep it selected (classes not
// selected yet are activated), then local search re-optimizes; the change is
// kept if the weighted makespan decreases and the cache fits its memory cap.
inline bool addCacheBases(const SearchState &state, const std::vector<PrimalBucketContext> &contexts,
                          PrimalPlan &plan, std::chrono::steady_clock::time_point deadline, uint64_t &evaluations,
                          uint32_t &added)
{
    std::unordered_map<BaseEClassId, double> saving;
    for (uint32_t b = 0; b < contexts.size(); ++b)
    {
        const double w = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0;
        if (!(w > 0.0))
            continue;
        const RelaxationBucket &rb = contexts[b].rb;
        const uint32_t orig_n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
        for (uint32_t u = 0; u < orig_n; ++u)
        {
            const int32_t v = plan.buckets[b].value[u];
            const BaseEClassId base = rb.classes[u].base;
            if (v <= 0 || base == BaseEClassId{} || plan.cached.count(base))
                continue;
            auto var = state.cached_vars.find(base);
            if (var == state.cached_vars.end() || !state.domains[var->second].contains(1))
                continue;
            double current = 0.0;
            for (const auto &alt : rb.classes[u].alternatives)
                if (alt.value == v)
                    current = alt.duration;
            for (const auto &alt : rb.classes[u].alternatives)
                if (alt.requires_cache && alt.duration < current - 1e-9)
                {
                    saving[base] = std::max(saving[base], w * (current - alt.duration));
                    break;
                }
        }
    }
    std::vector<std::pair<double, BaseEClassId>> candidates;
    for (const auto &[base, s] : saving)
        candidates.push_back({s, base});
    std::sort(candidates.begin(), candidates.end(), [](const auto &x, const auto &y) {
        return x.first > y.first || (x.first == y.first && x.second.value < y.second.value);
    });
    // All candidates at once first, then one at a time (best first).
    std::vector<std::vector<BaseEClassId>> sets;
    if (candidates.size() > 1)
    {
        sets.emplace_back();
        for (const auto &[s, base] : candidates)
            sets.back().push_back(base);
    }
    for (const auto &[s, base] : candidates)
        sets.push_back({base});
    bool any = false;
    for (const auto &set : sets)
    {
        if (std::chrono::steady_clock::now() >= deadline)
            break;
        bool fresh = false;
        for (BaseEClassId base : set)
            fresh |= !plan.cached.count(base);
        if (!fresh)
            continue;
        PrimalPlan trial = plan;
        for (BaseEClassId base : set)
            trial.cached.insert(base);
        // Memory caps.
        std::unordered_map<MemSpace, uint64_t> bytes;
        bool fits = true;
        for (BaseEClassId base : trial.cached)
        {
            auto var = state.cached_vars.find(base);
            if (var == state.cached_vars.end())
                continue;
            const VarInfo &info = state.var_infos[var->second];
            bytes[info.mem_space] += info.size_bytes;
            fits = fits && bytes[info.mem_space] <= state.getMemoryCap(info.mem_space);
        }
        if (!fits)
            continue;
        // Every bucket must select the newly cached bases.
        bool ok = true;
        for (uint32_t b = 0; b < contexts.size() && ok; ++b)
        {
            const RelaxationBucket &rb = contexts[b].rb;
            bool changed = false;
            for (uint32_t u = 0; u < rb.classes.size() && ok; ++u)
                if (trial.buckets[b].value[u] <= 0 && std::find(set.begin(), set.end(), rb.classes[u].base) != set.end())
                {
                    ok = activateClass(rb, trial.bucket_lambda[b], trial.estimates[b], trial.cached, trial.buckets[b], u);
                    changed = true;
                }
            if (ok && changed)
                listSchedule(rb, trial.buckets[b]);
        }
        if (!ok)
            continue;
        improvePrimalPlan(state, contexts, trial, deadline, evaluations, nullptr, nullptr);
        if (trial.cost < plan.cost - 1e-9)
        {
            uint32_t count = 0;
            for (BaseEClassId base : set)
                count += !plan.cached.count(base);
            plan = std::move(trial);
            added += count;
            any = true;
        }
    }
    return any;
}

inline bool improveCacheSet(const SearchState &state, const std::vector<PrimalBucketContext> &contexts,
                            PrimalPlan &plan, std::chrono::steady_clock::time_point deadline, uint64_t &evaluations,
                            uint32_t &dropped)
{
    bool any = false;
    std::unordered_set<BaseEClassId> rejected;
    while (std::chrono::steady_clock::now() < deadline)
    {
        std::unordered_map<BaseEClassId, uint32_t> blocked;
        improvePrimalPlan(state, contexts, plan, deadline, evaluations, nullptr, &blocked);
        std::vector<std::pair<uint32_t, BaseEClassId>> candidates;
        for (const auto &[base, count] : blocked)
            if (plan.cached.count(base) && !rejected.count(base))
                candidates.push_back({count, base});
        std::sort(candidates.begin(), candidates.end(), [](const auto &x, const auto &y) {
            return x.first > y.first || (x.first == y.first && x.second.value < y.second.value);
        });
        // A move may be blocked by several bases at once: try dropping all
        // blocking bases together first, then each one alone.
        std::vector<std::vector<BaseEClassId>> sets;
        if (candidates.size() > 1)
        {
            sets.emplace_back();
            for (const auto &[count, base] : candidates)
                sets.back().push_back(base);
        }
        for (const auto &[count, base] : candidates)
            sets.push_back({base});
        bool improved = false;
        for (const auto &set : sets)
        {
            if (std::chrono::steady_clock::now() >= deadline)
                break;
            PrimalPlan trial = plan;
            for (BaseEClassId base : set)
                trial.cached.erase(base);
            const std::unordered_set<BaseEClassId> releasable(set.begin(), set.end());
            const uint32_t count = static_cast<uint32_t>(set.size());
            const BaseEClassId base = set.front();
            improvePrimalPlan(state, contexts, trial, deadline, evaluations, &releasable, nullptr);
            bool valid = true;
            for (uint32_t b = 0; b < contexts.size() && valid; ++b)
            {
                const float w = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0f;
                if (!(w > 0.0f))
                    continue;
                const RelaxationBucket &rb = contexts[b].rb;
                const uint32_t orig_n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
                for (uint32_t u = 0; u < orig_n && valid; ++u)
                {
                    if (trial.buckets[b].value[u] <= 0)
                        continue;
                    for (const auto &alt : rb.classes[u].alternatives)
                        if (alt.value == trial.buckets[b].value[u] && alt.requires_cache &&
                            !trial.cached.count(rb.classes[u].base))
                            valid = false;
                }
            }
            if (valid && trial.cost < plan.cost - 1e-9)
            {
                plan = std::move(trial);
                dropped += count;
                improved = any = true;
                break;
            }
            if (set.size() == 1)
                rejected.insert(base);
        }
        if (!improved)
            break;
    }
    return any;
}

} // namespace plan
