// tensor_graphs_cpp/core/plan/propagators/start_rank_window.hpp
#pragma once

#include <cmath>

#include "core/plan/propagators/base.hpp"

namespace plan
{

// StartRankWindowPropagator: see docs/core/propagators.md.
//
// Root-only. For a selected class c of bucket b with N reachable classes:
//   ND(c) = classes that precede c in every plan selecting c
//         = intersection over remaining alternatives of the union over their
//           children ch of {ch} + ND(ch);
//   NA(c) = classes that follow c in every plan selecting c
//         = intersection over possible parents p of {p} + NA(p).
// Active starts are distinct and producers precede consumers, so
// start(c) >= |ND(c)| in every feasible plan. Cost and every constraint
// depend only on the relative order of active starts, so ranks of any plan
// can be compressed to 0..k-1 (k <= N active classes) without changing
// feasibility or cost; then start(c) <= k-1-|NA(c)| <= N-1-|NA(c)|.
// An unselected class keeps an irrelevant start inside its domain.
class StartRankWindowPropagator : public Propagator
{
    bool applied_ = false;
    std::string summary_;

    static double startBits(const SearchState &state, uint32_t b)
    {
        double bits = 0.0;
        for (const auto &[cid, st_v] : state.start_vars[b])
            bits += std::log2(static_cast<double>(state.domains[st_v].size()));
        return bits;
    }

    // A bucket's windows depend only on its own selection and start
    // domains, and narrowing touches only its own starts, so every bucket's
    // windows are computed in parallel from the entry domains and applied
    // afterwards in bucket order, exactly as a sequential pass would.
    struct BucketWindows
    {
        bool consistent = true;
        std::vector<std::pair<VarId, Domain>> narrowed;
        std::vector<std::string> messages;
    };

    static void computeBucket(const SearchState &state, uint32_t b, BucketWindows &out)
    {
        const EGraph &egraph = state.bucket_egraphs[b];
        const std::vector<EClassId> &cids = state.reachable_cids[b];
        const size_t n = cids.size();
        if (n == 0)
            Error::throw_err("StartRankWindow requires reachable classes in bucket " + std::to_string(b));
        std::unordered_map<uint32_t, uint32_t> index_of;
        index_of.reserve(n * 2);
        for (uint32_t i = 0; i < n; ++i)
            index_of[cids[i].value] = i;
        const uint32_t root_idx = index_of.at(egraph.findConst(state.bucket_root_ids[b]).value);

        // Children of every remaining alternative, by class index.
        std::vector<std::vector<std::vector<uint32_t>>> alt_children(n);
        std::vector<std::vector<uint32_t>> parents(n);
        std::vector<uint32_t> pending_children(n, 0);
        std::vector<std::vector<uint32_t>> child_classes(n);
        for (uint32_t i = 0; i < n; ++i)
        {
            const EClassId cid = cids[i];
            const Domain &sel = state.domains[state.selected_vars[b].at(cid)];
            const EClass &cls = egraph.getEClass(cid);
            for (size_t a = 0; a < cls.enodes.size(); ++a)
            {
                if (!sel.contains(static_cast<int32_t>(a + 1)))
                    continue;
                std::vector<uint32_t> children;
                for (EClassId child : egraph.getENode(cls.enodes[a]).getChildren())
                {
                    const auto it = index_of.find(egraph.findConst(child).value);
                    if (it == index_of.end())
                        Error::throw_err("StartRankWindow child outside reachable classes in bucket " + std::to_string(b));
                    children.push_back(it->second);
                }
                std::sort(children.begin(), children.end());
                children.erase(std::unique(children.begin(), children.end()), children.end());
                alt_children[i].push_back(children);
                child_classes[i].insert(child_classes[i].end(), children.begin(), children.end());
            }
            std::sort(child_classes[i].begin(), child_classes[i].end());
            child_classes[i].erase(std::unique(child_classes[i].begin(), child_classes[i].end()), child_classes[i].end());
            for (uint32_t ch : child_classes[i])
                parents[ch].push_back(i);
            pending_children[i] = static_cast<uint32_t>(child_classes[i].size());
        }

        // Children-first topological order over possible edges.
        std::vector<uint32_t> order;
        order.reserve(n);
        for (uint32_t i = 0; i < n; ++i)
            if (pending_children[i] == 0)
                order.push_back(i);
        for (size_t head = 0; head < order.size(); ++head)
            for (uint32_t p : parents[order[head]])
                if (--pending_children[p] == 0)
                    order.push_back(p);
        if (order.size() != n)
        {
            out.messages.push_back("bucket=" + std::to_string(b) + " skipped: possible-selection graph is cyclic");
            return;
        }

        // Persistent sources: possibly selected classes whose remaining
        // alternatives are childless INPUT/CACHE or non-writing views of
        // sources. They have zero runtime and persistent storage.
        std::vector<char> is_source(n, 0);
        std::vector<uint32_t> source_order;
        for (uint32_t i : order)
        {
            const EClassId cid = cids[i];
            const Domain &sel = state.domains[state.selected_vars[b].at(cid)];
            if (sel.isFixed() && sel.fixedValue() == 0)
                continue;
            const EClass &cls = egraph.getEClass(cid);
            bool source = false;
            for (size_t a = 0; a < cls.enodes.size(); ++a)
            {
                if (!sel.contains(static_cast<int32_t>(a + 1)))
                    continue;
                const ENodeId en_id = cls.enodes[a];
                const ENode &enode = egraph.getENode(en_id);
                const OpType op = enode.getOpType();
                if (op == OpType::INPUT || op == OpType::CACHE)
                    source = enode.getChildren().empty();
                else if (state.bucket_enode_infos[b].at(en_id.value).is_view && op != OpType::FILL &&
                         op != OpType::SCATTER && !enode.getChildren().empty())
                {
                    source = true;
                    for (EClassId child : enode.getChildren())
                        source = source && is_source[index_of.at(egraph.findConst(child).value)];
                }
                else
                    source = false;
                if (!source)
                    break;
            }
            if (!source)
                continue;
            is_source[i] = 1;
            source_order.push_back(i);
        }

        // Disjoint class groups. Any partition is sound: a group that must
        // have an active member before (after) c contributes one distinct
        // class. Classes linked by COPY_TO hold the same tensor in different
        // memory spaces, so necessity survives device-copy bypasses.
        std::vector<uint32_t> uf(n);
        for (uint32_t i = 0; i < n; ++i)
            uf[i] = i;
        std::function<uint32_t(uint32_t)> findRoot = [&](uint32_t x) {
            while (uf[x] != x)
            {
                uf[x] = uf[uf[x]];
                x = uf[x];
            }
            return x;
        };
        for (uint32_t i = 0; i < n; ++i)
        {
            const EClassId cid = cids[i];
            const Domain &sel = state.domains[state.selected_vars[b].at(cid)];
            const EClass &cls = egraph.getEClass(cid);
            for (size_t a = 0; a < cls.enodes.size(); ++a)
            {
                if (!sel.contains(static_cast<int32_t>(a + 1)))
                    continue;
                const ENode &enode = egraph.getENode(cls.enodes[a]);
                if (enode.getOpType() != OpType::COPY_TO || enode.getChildren().size() != 1)
                    continue;
                const uint32_t ch = index_of.at(egraph.findConst(enode.getChildren()[0]).value);
                uf[findRoot(i)] = findRoot(ch);
            }
        }
        std::vector<uint32_t> group(n);
        std::unordered_map<uint32_t, uint32_t> group_of_root;
        for (uint32_t i = 0; i < n; ++i)
        {
            const uint32_t r = findRoot(i);
            const auto it = group_of_root.emplace(r, static_cast<uint32_t>(group_of_root.size())).first;
            group[i] = it->second;
        }
        const size_t group_count = group_of_root.size();
        const size_t words = (group_count + 63) / 64;
        auto popcount = [&](const uint64_t *row, const std::vector<uint64_t> *mask_out) {
            size_t count = 0;
            for (size_t w = 0; w < words; ++w)
            {
                const uint64_t value = mask_out ? (row[w] & ~(*mask_out)[w]) : row[w];
#if defined(_MSC_VER) && !defined(__clang__)
                count += static_cast<size_t>(__popcnt64(value));
#else
                count += static_cast<size_t>(__builtin_popcountll(value));
#endif
            }
            return count;
        };
        std::vector<uint64_t> scratch(words);
        // Groups containing a source: their guaranteed member may sit in the
        // source prefix, so they do not count among non-source predecessors.
        std::vector<uint64_t> source_groups(words, 0);
        for (uint32_t i : source_order)
            source_groups[group[i] / 64] |= 1ull << (group[i] % 64);

        // Necessary descendant groups.
        std::vector<uint64_t> nd(n * words, 0);
        for (uint32_t i : order)
        {
            uint64_t *row = &nd[i * words];
            bool first = true;
            for (const auto &children : alt_children[i])
            {
                std::fill(scratch.begin(), scratch.end(), 0);
                for (uint32_t ch : children)
                {
                    const uint64_t *child_row = &nd[ch * words];
                    for (size_t w = 0; w < words; ++w)
                        scratch[w] |= child_row[w];
                    scratch[group[ch] / 64] |= 1ull << (group[ch] % 64);
                }
                for (size_t w = 0; w < words; ++w)
                    row[w] = first ? scratch[w] : (row[w] & scratch[w]);
                first = false;
            }
        }
        std::vector<uint32_t> lower(n), lower_non_source(n);
        for (uint32_t i = 0; i < n; ++i)
        {
            lower[i] = static_cast<uint32_t>(popcount(&nd[i * words], nullptr));
            lower_non_source[i] = static_cast<uint32_t>(popcount(&nd[i * words], &source_groups));
        }
        std::vector<uint64_t>().swap(nd);

        // Distinct active starts are all >= the smallest start value, so the
        // lower bound holds for any domains. Rank compression onto 0..k-1
        // stays inside the domains only if each is downward closed from 0.
        // Root propagation may already have raised some minima soundly; the
        // compression argument concerns the domains at solve entry. Without
        // an entry record (direct propagation outside solve) only the lower
        // bound is applied.
        int64_t min_start = std::numeric_limits<int64_t>::max();
        for (const auto &[cid, st_v] : state.start_vars[b])
        {
            const Domain &dom = state.domains[st_v];
            if (dom.isEmpty())
            {
                out.consistent = false;
                return;
            }
            min_start = std::min<int64_t>(min_start, dom.getMin());
        }
        const bool downward_closed = b < state.start_domains_downward_closed_at_entry.size() &&
                                     state.start_domains_downward_closed_at_entry[b];
        if (downward_closed)
            min_start = 0;
        if (!downward_closed)
            out.messages.push_back("bucket=" + std::to_string(b) +
                                   " start domains are not downward closed from 0; upper rank windows skipped");
        // Canonical form: sources at fixed ranks 0..S-1 in children-first
        // order (unselected sources leave holes), then the other active
        // classes densely in their original relative order.
        const bool source_prefix = downward_closed && !source_order.empty();
        if (source_prefix)
        {
            for (size_t rank = 0; rank < source_order.size(); ++rank)
            {
                const uint32_t i = source_order[rank];
                if (!state.domains[state.start_vars[b].at(cids[i])].contains(static_cast<int32_t>(rank)))
                    Error::throw_err("StartRankWindow source rank outside a downward closed start domain");
                if (lower[i] > rank)
                    Error::throw_err("StartRankWindow source rank precedes a necessary descendant");
            }
        }
        const int64_t prefix = source_prefix ? static_cast<int64_t>(source_order.size()) : 0;

        // Necessary ancestors (parents first).
        std::vector<uint64_t> na(n * words, 0);
        std::vector<char> has_parent_support(n, 0);
        has_parent_support[root_idx] = 1;
        for (auto it = order.rbegin(); it != order.rend(); ++it)
        {
            const uint32_t i = *it;
            if (i == root_idx)
                continue;
            uint64_t *row = &na[i * words];
            bool first = true;
            for (uint32_t p : parents[i])
            {
                const uint64_t *parent_row = &na[p * words];
                std::copy(parent_row, parent_row + words, scratch.begin());
                scratch[group[p] / 64] |= 1ull << (group[p] % 64);
                for (size_t w = 0; w < words; ++w)
                    row[w] = first ? scratch[w] : (row[w] & scratch[w]);
                first = false;
            }
            has_parent_support[i] = !first;
        }

        for (uint32_t i = 0; i < n; ++i)
        {
            const EClassId cid = cids[i];
            const VarId sel_v = state.selected_vars[b].at(cid);
            const Domain &sel = state.domains[sel_v];
            if (sel.isFixed() && sel.fixedValue() == 0)
                continue;
            // A class without parent support cannot be selected; leave its
            // removal to the reachability propagators.
            if (!has_parent_support[i])
                continue;
            const VarId source_st_v = state.start_vars[b].at(cid);
            if (source_prefix && is_source[i])
            {
                const int32_t rank = static_cast<int32_t>(
                    std::find(source_order.begin(), source_order.end(), i) - source_order.begin());
                if (!state.domains[source_st_v].isFixed())
                    out.narrowed.emplace_back(source_st_v, Domain::makeFixed(rank));
                continue;
            }
            const int64_t lo = min_start + (source_prefix ? prefix + lower_non_source[i] : lower[i]);
            const int64_t hi = downward_closed
                                   ? static_cast<int64_t>(n) - 1 - static_cast<int64_t>(popcount(&na[i * words], nullptr))
                                   : static_cast<int64_t>(std::numeric_limits<int32_t>::max());
            if (lo > std::numeric_limits<int32_t>::max())
                Error::throw_err("StartRankWindow lower bound exceeds the start value range");
            const VarId st_v = state.start_vars[b].at(cid);
            Domain next = state.domains[st_v];
            next.setRange(static_cast<int32_t>(lo), static_cast<int32_t>(hi));
            if (next.isEmpty())
            {
                if (sel.contains(0))
                    continue; // cannot be selected; still a valid unselected class
                out.messages.push_back("conflict bucket=" + std::to_string(b) + " cid=" + std::to_string(cid.value) +
                                       " window=[" + std::to_string(lo) + "," + std::to_string(hi) +
                                       "] domain=" + state.domains[st_v].toString());
                out.consistent = false;
                return;
            }
            if (next.size() == state.domains[st_v].size())
                continue;
            out.narrowed.emplace_back(st_v, next);
        }
    }

  public:
    // Root propagation visits every dirty variable by type; the first start
    // visit applies the windows once, before any branching.
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::START); }

    std::string name() const override
    {
        return "StartRankWindowPropagator";
    }

    std::string diagnosticSummary() const override { return summary_; }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (applied_)
            return true;
        applied_ = true;
        std::vector<BucketWindows> windows(state.buckets.size());
        ThreadPool::get().parallel_for(static_cast<uint32_t>(state.buckets.size()),
                                       [&](uint32_t b) { computeBucket(state, b, windows[b]); });
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            const double before = startBits(state, b);
            for (const std::string &message : windows[b].messages)
                LOG(INFO) << "[StartRankWindow] " << message;
            if (!windows[b].consistent)
                return false;
            for (const auto &[st_v, next] : windows[b].narrowed)
            {
                state.setDomain(st_v, next);
                worklist.push_back(st_v);
            }
            const size_t narrowed = windows[b].narrowed.size();
            const double after = startBits(state, b);
            LOG(INFO) << "[StartRankWindow] bucket=" << b << " narrowed=" << narrowed << " start-bits " << before
                      << " -> " << after;
            summary_ += std::to_string(b) + ":" + std::to_string(narrowed) + " ";
        }
        return true;
    }
};

} // namespace plan
