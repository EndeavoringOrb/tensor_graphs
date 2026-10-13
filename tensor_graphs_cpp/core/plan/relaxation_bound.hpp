// tensor_graphs_cpp/core/plan/relaxation_bound.hpp
#pragma once

// Selection relaxation of one bucket: which alternatives can still be chosen,
// their simulated durations and engines. Used by RelaxationBoundPropagator
// (a root lower bound on each bucket's makespan) and by the root primal
// heuristic. See docs/core/relaxation_bound.md.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/logging.hpp"
#include "core/plan/search_state.hpp"

namespace plan
{

struct RelaxationAlternative
{
    int32_t value = 0;       // selection value (1-based e-node index)
    double duration = 0.0;   // simulated duration (0 for views, INPUT and CACHE)
    uint32_t engine_mask = 0; // bucket-local engines occupied for a positive `duration`
    uint32_t sched_mask = 0;  // engines the simulator waits for (any non-view, non-source op)
    bool is_copy = false;    // COPY_TO: same logical tensor in another memory space
    bool is_view = false;
    bool is_source = false;  // INPUT or CACHE: finishes at time 0
    bool requires_cache = false; // filled by markCacheAlternatives
    std::vector<uint32_t> children; // local class indices, deduplicated
};

struct RelaxationClass
{
    EClassId cid;
    VarId selected_var = kInvalidVarId;
    BaseEClassId base;
    std::vector<RelaxationAlternative> alternatives; // feasible alternatives only
};

struct RelaxationBucket
{
    uint32_t bucket_idx = 0;
    std::vector<RelaxationClass> classes;
    std::vector<uint32_t> topo; // children before parents
    std::vector<uint32_t> topo_pos;
    uint32_t root = UINT32_MAX;
    std::vector<Engine> engines;
    // Classes linked by COPY_TO alternatives hold the same logical tensor.
    std::vector<uint32_t> group;
    std::vector<std::vector<uint32_t>> group_members;
    bool acyclic = true;
    bool root_feasible = false;
};

inline uint32_t relaxationEngineIndex(RelaxationBucket &rb, const Engine &engine)
{
    for (uint32_t i = 0; i < rb.engines.size(); ++i)
        if (rb.engines[i] == engine)
            return i;
    if (rb.engines.size() >= 32)
        Error::throw_err("Relaxation bound supports at most 32 engines per bucket");
    rb.engines.push_back(engine);
    return static_cast<uint32_t>(rb.engines.size() - 1);
}

// Durations and engines mirror SearchEngine::evaluateMakespan and
// MakespanSimulator: INPUT/CACHE finish at 0 without an engine, views take no
// engine time, an infinite cost simulates as 1, and an empty engine list runs
// on CPU engine 0.
inline RelaxationBucket buildRelaxationBucket(const SearchState &state, uint32_t b)
{
    RelaxationBucket rb;
    rb.bucket_idx = b;
    const EGraph &egraph = state.bucket_egraphs.at(b);
    const auto &infos = state.bucket_enode_infos.at(b);
    std::unordered_map<EClassId, uint32_t> local;
    for (EClassId cid : state.reachable_cids.at(b))
    {
        auto it = state.selected_vars.at(b).find(cid);
        if (it == state.selected_vars[b].end())
            continue;
        local.emplace(cid, static_cast<uint32_t>(rb.classes.size()));
        RelaxationClass rc;
        rc.cid = cid;
        rc.selected_var = it->second;
        rc.base = egraph.getEClass(cid).base_eclass_id;
        rb.classes.push_back(std::move(rc));
    }
    const auto root_it = local.find(egraph.findConst(state.bucket_root_ids.at(b)));
    if (root_it == local.end())
        Error::throw_err("Relaxation bound: bucket " + std::to_string(b) + " root has no selection variable");
    rb.root = root_it->second;

    for (RelaxationClass &rc : rb.classes)
    {
        const Domain &domain = state.domains[rc.selected_var];
        const EClass &cls = egraph.getEClass(rc.cid);
        for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
        {
            const int32_t value = static_cast<int32_t>(en_idx + 1);
            if (!domain.contains(value))
                continue;
            const ENodeId en_id = cls.enodes[en_idx];
            const ENode &enode = egraph.getENode(en_id);
            RelaxationAlternative alt;
            alt.value = value;
            alt.is_copy = enode.getOpType() == OpType::COPY_TO;
            const bool is_view = en_id.value < infos.size() && infos[en_id.value].is_view;
            const bool is_source = enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE;
            alt.is_view = is_view;
            alt.is_source = is_source;
            if (!is_view && !is_source)
            {
                if (enode.getEngines().empty())
                    alt.sched_mask |= 1u << relaxationEngineIndex(rb, Engine{0, EngineType::CPU});
                for (const Engine &engine : enode.getEngines())
                    alt.sched_mask |= 1u << relaxationEngineIndex(rb, engine);
                float cost = en_id.value < infos.size() ? infos[en_id.value].cost : 0.0f;
                if (cost == TGConstants::INF || std::isinf(cost))
                    cost = 1.0f;
                if (std::isnan(cost) || cost < 0.0f)
                    Error::throw_err("Relaxation bound: invalid cost for e-node " + std::to_string(en_id.value));
                alt.duration = cost;
                if (cost > 0.0f)
                {
                    if (enode.getEngines().empty())
                        alt.engine_mask |= 1u << relaxationEngineIndex(rb, Engine{0, EngineType::CPU});
                    for (const Engine &engine : enode.getEngines())
                        alt.engine_mask |= 1u << relaxationEngineIndex(rb, engine);
                }
            }
            for (EClassId child : enode.getChildren())
            {
                const auto child_it = local.find(egraph.findConst(child));
                if (child_it == local.end())
                    Error::throw_err("Relaxation bound: child class " + std::to_string(child.value) +
                                     " of a reachable class has no selection variable");
                if (std::find(alt.children.begin(), alt.children.end(), child_it->second) == alt.children.end())
                    alt.children.push_back(child_it->second);
            }
            rc.alternatives.push_back(std::move(alt));
        }
    }

    // Children-first order; a cycle disables the relaxation for this bucket.
    const uint32_t n = static_cast<uint32_t>(rb.classes.size());
    std::vector<uint8_t> color(n, 0);
    std::vector<std::pair<uint32_t, uint32_t>> stack;
    rb.topo.reserve(n);
    for (uint32_t start = 0; start < n && rb.acyclic; ++start)
    {
        if (color[start] != 0)
            continue;
        color[start] = 1;
        stack.push_back({start, 0});
        while (!stack.empty() && rb.acyclic)
        {
            auto &[u, next] = stack.back();
            // Flatten (alternative, child) pairs into one cursor.
            uint32_t cursor = next++;
            const auto &alts = rb.classes[u].alternatives;
            bool pushed = false;
            for (const auto &alt : alts)
            {
                if (cursor < alt.children.size())
                {
                    const uint32_t v = alt.children[cursor];
                    if (color[v] == 1)
                        rb.acyclic = false;
                    else if (color[v] == 0)
                    {
                        color[v] = 1;
                        stack.push_back({v, 0});
                    }
                    pushed = true;
                    break;
                }
                cursor -= static_cast<uint32_t>(alt.children.size());
            }
            if (!pushed)
            {
                color[u] = 2;
                rb.topo.push_back(u);
                stack.pop_back();
            }
        }
    }
    if (!rb.acyclic)
        return rb;
    rb.topo_pos.assign(n, 0);
    for (uint32_t i = 0; i < n; ++i)
        rb.topo_pos[rb.topo[i]] = i;

    // Alternatives needing a class that can never be active are infeasible.
    std::vector<uint8_t> feasible(n, 0);
    for (uint32_t u : rb.topo)
    {
        auto &alts = rb.classes[u].alternatives;
        alts.erase(std::remove_if(alts.begin(), alts.end(), [&](const RelaxationAlternative &alt) {
                       for (uint32_t child : alt.children)
                           if (!feasible[child])
                               return true;
                       return false;
                   }),
                   alts.end());
        feasible[u] = !alts.empty();
    }
    rb.root_feasible = feasible[rb.root] != 0;

    std::vector<uint32_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0u);
    auto find = [&](uint32_t x) {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    for (uint32_t u = 0; u < n; ++u)
        for (const auto &alt : rb.classes[u].alternatives)
            if (alt.is_copy)
                for (uint32_t child : alt.children)
                    parent[find(child)] = find(u);
    std::vector<uint32_t> group_of_rep(n, UINT32_MAX);
    rb.group.assign(n, 0);
    for (uint32_t u = 0; u < n; ++u)
    {
        const uint32_t rep = find(u);
        if (group_of_rep[rep] == UINT32_MAX)
        {
            group_of_rep[rep] = static_cast<uint32_t>(rb.group_members.size());
            rb.group_members.emplace_back();
        }
        rb.group[u] = group_of_rep[rep];
        rb.group_members[rb.group[u]].push_back(u);
    }
    return rb;
}

namespace relaxation_detail
{
inline size_t words(size_t bits) { return (bits + 63) / 64; }
inline void setBit(uint64_t *w, size_t i) { w[i >> 6] |= 1ull << (i & 63); }
inline bool testBit(const uint64_t *w, size_t i) { return (w[i >> 6] >> (i & 63)) & 1ull; }
} // namespace relaxation_detail

// Lower bound on the minimum over plans of sum_e lambda_e * load_e(plan) for
// one bucket, where load_e is the simulated work on engine e. Because each
// engine runs its operations serially, makespan >= max_e load_e >= this value
// whenever lambda is a probability vector.
//
// Recursion (one region at a time, see docs/core/relaxation_bound.md):
//   L(c) = min over alternatives a of [ w(a) + sum of L(anchor) over children
//          with pairwise disjoint supports + sum of mp(G) over necessary
//          groups G of a's children disjoint from those supports ]
// with the invariant L(c) <= weighted cost of the plan's classes in S(c), the
// support of c. Regions follow a containment chain of necessary groups of the
// root, so the shared history below a link is counted only once.
class RelaxationBoundComputer
{
  public:
    explicit RelaxationBoundComputer(const RelaxationBucket &rb) : rb_(rb)
    {
        if (!rb.acyclic || !rb.root_feasible)
            return;
        prepare();
        ready_ = true;
    }

    bool ready() const { return ready_; }
    size_t numRegions() const { return regions_.size(); }

    // After evaluate(): each class's bound within its own region, i.e. its
    // cost excluding the shared history below the region (0 outside regions).
    void regionEstimates(std::vector<double> &estimates) const
    {
        estimates.assign(rb_.classes.size(), 0.0);
        for (const Region &region : regions_)
            if (!region.zero_value)
                for (size_t i = 0; i < region.nodes.size(); ++i)
                    estimates[region.nodes[i]] = region.L[i];
    }

    double evaluate(const std::vector<double> &lambda)
    {
        if (!ready_)
            Error::throw_err("Relaxation bound evaluated on an unprepared bucket");
        if (lambda.size() != rb_.engines.size())
            Error::throw_err("Relaxation bound: lambda has the wrong number of engines");
        alt_weight_.resize(alt_offset_.back());
        for (uint32_t u = 0; u < rb_.classes.size(); ++u)
        {
            const auto &alts = rb_.classes[u].alternatives;
            for (uint32_t a = 0; a < alts.size(); ++a)
            {
                double scale = 0.0;
                for (uint32_t e = 0; e < lambda.size(); ++e)
                    if (alts[a].engine_mask & (1u << e))
                        scale += lambda[e];
                alt_weight_[alt_offset_[u] + a] = alts[a].duration * scale;
            }
        }
        double total = 0.0;
        for (Region &region : regions_)
            total += evaluateRegion(region);
        return total;
    }

  private:
    struct AltInfo
    {
        uint32_t alt_index = 0;
        std::vector<uint32_t> kids;   // region-local indices of positive children
        std::vector<uint32_t> extras; // region-local groups with positive production cost
    };
    struct Region
    {
        std::vector<uint32_t> nodes; // children first
        std::vector<std::vector<AltInfo>> alts;
        std::vector<uint8_t> pos;
        size_t support_words = 0;
        std::vector<uint64_t> support; // nodes.size() * support_words
        std::vector<uint32_t> targets; // region-local indices whose min is the region value
        bool zero_value = false;       // a target lies below the region
        std::vector<std::vector<uint32_t>> group_members; // region-local
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> group_production; // (global class, alt)
        std::vector<double> L;
        std::vector<double> mp;
    };

    const RelaxationBucket &rb_;
    bool ready_ = false;
    std::vector<Region> regions_;
    std::vector<uint32_t> alt_offset_;
    std::vector<double> alt_weight_;

    double weight(uint32_t u, uint32_t a) const { return alt_weight_[alt_offset_[u] + a]; }

    void prepare()
    {
        using namespace relaxation_detail;
        const uint32_t n = static_cast<uint32_t>(rb_.classes.size());
        const uint32_t ng = static_cast<uint32_t>(rb_.group_members.size());
        alt_offset_.assign(n + 1, 0);
        for (uint32_t u = 0; u < n; ++u)
            alt_offset_[u + 1] = alt_offset_[u] + static_cast<uint32_t>(rb_.classes[u].alternatives.size());

        // Descendants (any alternative) and necessary groups, children first.
        const size_t wc = words(n), wg = words(ng);
        std::vector<uint64_t> desc(static_cast<size_t>(n) * wc, 0);
        std::vector<uint64_t> gnd(static_cast<size_t>(n) * wg, 0);
        std::vector<uint64_t> acc(wg), uni(wg);
        for (uint32_t u : rb_.topo)
        {
            uint64_t *du = &desc[static_cast<size_t>(u) * wc];
            setBit(du, u);
            bool first = true;
            std::fill(acc.begin(), acc.end(), 0);
            for (const auto &alt : rb_.classes[u].alternatives)
            {
                std::fill(uni.begin(), uni.end(), 0);
                for (uint32_t ch : alt.children)
                {
                    const uint64_t *dc = &desc[static_cast<size_t>(ch) * wc];
                    for (size_t w = 0; w < wc; ++w)
                        du[w] |= dc[w];
                    const uint64_t *gc = &gnd[static_cast<size_t>(ch) * wg];
                    for (size_t w = 0; w < wg; ++w)
                        uni[w] |= gc[w];
                    setBit(uni.data(), rb_.group[ch]);
                }
                if (first)
                    acc = uni;
                else
                    for (size_t w = 0; w < wg; ++w)
                        acc[w] &= uni[w];
                first = false;
            }
            uint64_t *gu = &gnd[static_cast<size_t>(u) * wg];
            std::copy(acc.begin(), acc.end(), gu);
            gu[rb_.group[u] >> 6] &= ~(1ull << (rb_.group[u] & 63));
        }

        // Containment chain of the root's necessary groups.
        auto groupDesc = [&](uint32_t g) {
            std::vector<uint64_t> bits(wc, 0);
            for (uint32_t m : rb_.group_members[g])
            {
                const uint64_t *dm = &desc[static_cast<size_t>(m) * wc];
                for (size_t w = 0; w < wc; ++w)
                    bits[w] |= dm[w];
            }
            return bits;
        };
        std::vector<std::pair<size_t, uint32_t>> necessary;
        std::vector<std::vector<uint64_t>> necessary_desc(ng);
        const uint64_t *groot = &gnd[static_cast<size_t>(rb_.root) * wg];
        for (uint32_t g = 0; g < ng; ++g)
        {
            if (!testBit(groot, g))
                continue;
            necessary_desc[g] = groupDesc(g);
            size_t count = 0;
            for (uint64_t w : necessary_desc[g])
                count += static_cast<size_t>(__builtin_popcountll(w));
            necessary.push_back({count, g});
        }
        std::sort(necessary.begin(), necessary.end());
        std::vector<uint32_t> chain;
        for (const auto &[count, g] : necessary)
        {
            if (!chain.empty())
            {
                const uint32_t last = chain.back();
                bool contains = true;
                for (uint32_t m : rb_.group_members[last])
                    if (!testBit(necessary_desc[g].data(), m))
                        contains = false;
                if (!contains || necessary_desc[g] == necessary_desc[last])
                    continue;
            }
            chain.push_back(g);
        }

        std::vector<uint64_t> below(wc, 0);
        std::vector<uint64_t> region_bits(wc, 0);
        for (size_t link = 0; link <= chain.size(); ++link)
        {
            const bool final_region = link == chain.size();
            const std::vector<uint64_t> &top =
                final_region ? std::vector<uint64_t>(desc.begin() + static_cast<std::ptrdiff_t>(rb_.root * wc),
                                                     desc.begin() + static_cast<std::ptrdiff_t>((rb_.root + 1) * wc))
                             : necessary_desc[chain[link]];
            for (size_t w = 0; w < wc; ++w)
                region_bits[w] = top[w] & ~below[w];
            std::vector<uint32_t> targets_global =
                final_region ? std::vector<uint32_t>{rb_.root} : rb_.group_members[chain[link]];
            buildRegion(region_bits, below, targets_global);
            below = top;
        }
    }

    void buildRegion(const std::vector<uint64_t> &region_bits, const std::vector<uint64_t> &below,
                     const std::vector<uint32_t> &targets_global)
    {
        using namespace relaxation_detail;
        Region region;
        for (uint32_t u : rb_.topo)
            if (testBit(region_bits.data(), u))
                region.nodes.push_back(u);
        for (uint32_t t : targets_global)
            if (testBit(below.data(), t))
                region.zero_value = true;
        if (region.nodes.empty() || region.zero_value)
        {
            region.zero_value = true;
            regions_.push_back(std::move(region));
            return;
        }
        const uint32_t rn = static_cast<uint32_t>(region.nodes.size());
        std::unordered_map<uint32_t, uint32_t> local;
        local.reserve(rn * 2);
        for (uint32_t i = 0; i < rn; ++i)
            local.emplace(region.nodes[i], i);
        for (uint32_t t : targets_global)
        {
            auto it = local.find(t);
            if (it == local.end())
                Error::throw_err("Relaxation bound: chain target outside its region");
            region.targets.push_back(it->second);
        }

        // Groups whose members all lie in the region; only they may be
        // charged here, so different regions never count the same class.
        std::unordered_map<uint32_t, uint32_t> local_group;
        for (uint32_t i = 0; i < rn; ++i)
        {
            const uint32_t g = rb_.group[region.nodes[i]];
            if (local_group.count(g))
                continue;
            bool inside = true;
            for (uint32_t m : rb_.group_members[g])
                if (!local.count(m))
                    inside = false;
            if (!inside)
                continue;
            const uint32_t lg = static_cast<uint32_t>(region.group_members.size());
            local_group.emplace(g, lg);
            region.group_members.emplace_back();
            region.group_production.emplace_back();
            for (uint32_t m : rb_.group_members[g])
            {
                region.group_members[lg].push_back(local.at(m));
                const auto &alts = rb_.classes[m].alternatives;
                for (uint32_t a = 0; a < alts.size(); ++a)
                    if (!alts[a].is_copy)
                        region.group_production[lg].push_back({m, a});
            }
        }
        const uint32_t lgn = static_cast<uint32_t>(region.group_members.size());
        // A group's cheapest production is positive for every interior lambda
        // exactly when all its production alternatives have positive duration.
        std::vector<uint8_t> group_pos(lgn, 1);
        for (uint32_t lg = 0; lg < lgn; ++lg)
        {
            if (region.group_production[lg].empty())
                group_pos[lg] = 1; // no production: never active, any charge is vacuous
            for (const auto &[m, a] : region.group_production[lg])
                if (!(rb_.classes[m].alternatives[a].duration > 0.0 && rb_.classes[m].alternatives[a].engine_mask))
                    group_pos[lg] = 0;
        }

        const size_t wl = words(lgn);
        std::vector<uint64_t> gloc(static_cast<size_t>(rn) * wl, 0);
        std::vector<uint64_t> acc(wl), uni(wl);
        region.alts.resize(rn);
        region.pos.assign(rn, 0);
        for (uint32_t i = 0; i < rn; ++i)
        {
            const uint32_t u = region.nodes[i];
            const auto &alts = rb_.classes[u].alternatives;
            const auto own_it = local_group.find(rb_.group[u]);
            bool first = true;
            bool pos = true;
            for (uint32_t a = 0; a < alts.size(); ++a)
            {
                std::fill(uni.begin(), uni.end(), 0);
                AltInfo info;
                info.alt_index = a;
                for (uint32_t ch : alts[a].children)
                {
                    auto it = local.find(ch);
                    if (it == local.end())
                        continue; // below the region: counted by an earlier region
                    const uint32_t li = it->second;
                    const uint64_t *gc = &gloc[static_cast<size_t>(li) * wl];
                    for (size_t w = 0; w < wl; ++w)
                        uni[w] |= gc[w];
                    auto g_it = local_group.find(rb_.group[ch]);
                    if (g_it != local_group.end())
                        setBit(uni.data(), g_it->second);
                    if (region.pos[li])
                        info.kids.push_back(li);
                }
                if (own_it != local_group.end())
                    uni[own_it->second >> 6] &= ~(1ull << (own_it->second & 63));
                for (uint32_t lg = 0; lg < lgn; ++lg)
                    if (testBit(uni.data(), lg) && group_pos[lg])
                        info.extras.push_back(lg);
                const bool alt_pos = (alts[a].duration > 0.0 && alts[a].engine_mask) || !info.kids.empty() ||
                                     !info.extras.empty();
                if (!alt_pos)
                    pos = false;
                if (first)
                    acc = uni;
                else
                    for (size_t w = 0; w < wl; ++w)
                        acc[w] &= uni[w];
                first = false;
                region.alts[i].push_back(std::move(info));
            }
            if (alts.empty())
                pos = false;
            std::copy(acc.begin(), acc.end(), &gloc[static_cast<size_t>(i) * wl]);
            region.pos[i] = pos ? 1 : 0;
        }

        // Supports: a positive class, every positive child's support and every
        // charged group's members, over all alternatives.
        region.support_words = words(rn);
        const size_t ws = region.support_words;
        region.support.assign(static_cast<size_t>(rn) * ws, 0);
        for (uint32_t i = 0; i < rn; ++i)
        {
            if (!region.pos[i])
                continue;
            uint64_t *si = &region.support[static_cast<size_t>(i) * ws];
            setBit(si, i);
            for (const AltInfo &info : region.alts[i])
            {
                for (uint32_t k : info.kids)
                {
                    const uint64_t *sk = &region.support[static_cast<size_t>(k) * ws];
                    for (size_t w = 0; w < ws; ++w)
                        si[w] |= sk[w];
                }
                for (uint32_t lg : info.extras)
                    for (uint32_t m : region.group_members[lg])
                        setBit(si, m);
            }
        }
        region.L.assign(rn, 0.0);
        region.mp.assign(lgn, 0.0);
        regions_.push_back(std::move(region));
    }

    double evaluateRegion(Region &region)
    {
        using namespace relaxation_detail;
        if (region.zero_value)
            return 0.0;
        const uint32_t lgn = static_cast<uint32_t>(region.group_members.size());
        for (uint32_t lg = 0; lg < lgn; ++lg)
        {
            double best = std::numeric_limits<double>::infinity();
            for (const auto &[m, a] : region.group_production[lg])
                best = std::min(best, weight(m, a));
            region.mp[lg] = std::isfinite(best) ? best : 0.0;
        }
        const size_t ws = region.support_words;
        std::vector<uint64_t> cover(ws);
        std::vector<uint32_t> order;
        const uint32_t rn = static_cast<uint32_t>(region.nodes.size());
        for (uint32_t i = 0; i < rn; ++i)
        {
            if (!region.pos[i])
            {
                region.L[i] = 0.0;
                continue;
            }
            const uint32_t u = region.nodes[i];
            double best = std::numeric_limits<double>::infinity();
            for (const AltInfo &info : region.alts[i])
            {
                double cand = 0.0;
                if (!info.kids.empty() || !info.extras.empty())
                {
                    order = info.kids;
                    std::sort(order.begin(), order.end(),
                              [&](uint32_t x, uint32_t y) { return region.L[x] > region.L[y]; });
                    const size_t tries = std::max<size_t>(1, order.size());
                    cand = -1.0;
                    for (size_t first = 0; first < tries; ++first)
                    {
                        std::fill(cover.begin(), cover.end(), 0);
                        double sum = 0.0;
                        for (size_t step = 0; step < order.size(); ++step)
                        {
                            const uint32_t k = order[step == 0 ? first : (step <= first ? step - 1 : step)];
                            const uint64_t *sk = &region.support[static_cast<size_t>(k) * ws];
                            bool overlap = false;
                            for (size_t w = 0; w < ws && !overlap; ++w)
                                overlap = (cover[w] & sk[w]) != 0;
                            if (overlap)
                                continue;
                            sum += region.L[k];
                            for (size_t w = 0; w < ws; ++w)
                                cover[w] |= sk[w];
                        }
                        for (uint32_t lg : info.extras)
                        {
                            bool overlap = false;
                            for (uint32_t m : region.group_members[lg])
                                if (testBit(cover.data(), m))
                                {
                                    overlap = true;
                                    break;
                                }
                            if (!overlap)
                                sum += region.mp[lg];
                        }
                        cand = std::max(cand, sum);
                    }
                }
                best = std::min(best, weight(u, info.alt_index) + cand);
            }
            region.L[i] = std::isfinite(best) ? best : 0.0;
        }
        double value = std::numeric_limits<double>::infinity();
        for (uint32_t t : region.targets)
            value = std::min(value, region.L[t]);
        return std::isfinite(value) ? value : 0.0;
    }
};

// Best lower bound over a small set of engine weightings. Any probability
// vector gives a valid bound; the search over lambda only tightens it.
inline double maximizeRelaxationBound(RelaxationBoundComputer &computer, size_t num_engines,
                                      std::vector<double> *best_lambda = nullptr)
{
    if (num_engines == 0)
        return 0.0;
    double best = -1.0;
    std::vector<double> best_l;
    auto consider = [&](const std::vector<double> &lambda) {
        const double v = computer.evaluate(lambda);
        if (v > best)
        {
            best = v;
            best_l = lambda;
        }
        return v;
    };
    if (num_engines == 1)
        consider({1.0});
    else if (num_engines == 2)
    {
        // Golden-section search on lambda_0 in [lo, hi], keeping both weights
        // positive so zero-duration structure stays lambda-independent.
        const double lo = 0.01, hi = 0.99, phi = (std::sqrt(5.0) - 1.0) / 2.0;
        double a = lo, b = hi;
        double c = b - phi * (b - a), d = a + phi * (b - a);
        double fc = consider({c, 1.0 - c}), fd = consider({d, 1.0 - d});
        for (int it = 0; it < 18; ++it)
        {
            if (fc < fd)
            {
                a = c;
                c = d;
                fc = fd;
                d = a + phi * (b - a);
                fd = consider({d, 1.0 - d});
            }
            else
            {
                b = d;
                d = c;
                fd = fc;
                c = b - phi * (b - a);
                fc = consider({c, 1.0 - c});
            }
        }
        consider({0.5, 0.5});
    }
    else
    {
        std::vector<double> uniform(num_engines, 1.0 / static_cast<double>(num_engines));
        consider(uniform);
        for (size_t e = 0; e < num_engines; ++e)
        {
            std::vector<double> lambda(num_engines, 0.5 / static_cast<double>(num_engines - 1));
            lambda[e] = 0.5;
            consider(lambda);
            std::vector<double> heavy(num_engines, 0.1 / static_cast<double>(num_engines - 1));
            heavy[e] = 0.9;
            consider(heavy);
        }
    }
    if (best_lambda)
        *best_lambda = best_l;
    return std::max(best, 0.0);
}

} // namespace plan
