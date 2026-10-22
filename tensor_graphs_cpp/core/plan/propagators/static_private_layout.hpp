// tensor_graphs_cpp/core/plan/propagators/static_private_layout.hpp
#pragma once

#include <map>
#include <optional>
#include <tuple>

#include "core/common/thread_pool.hpp"
#include "core/plan/propagators/base.hpp"

namespace plan
{

// StaticPrivateLayoutPropagator: see docs/core/propagators.md.
//
// Offsets never enter the objective; they only have to be feasible. When every
// allocation of a memory space can own a private page range at once, any
// feasible plan stays feasible (with identical selections, starts and cost)
// after moving each allocation to that range:
//   [0, prealloc)            preallocated buffers (unchanged)
//   [prealloc, cache_end)    one slot per cacheable base, shared by all buckets
//   [cache_end, ...)         one slot per other non-view e-class of a bucket
// Distinct allocations never overlap, so memory reuse and write-after-read
// constraints become vacuous, cached copies share one address in every bucket,
// and a view keeps its base's address. Allocation pairs are only compared
// within one bucket, so buckets reuse the private range above cache_end.
// The root offsets are fixed to this layout only if every slot fits the space
// capacity and lies inside the variable's current domain; otherwise the space
// is left untouched.
class StaticPrivateLayoutPropagator : public Propagator
{
    bool applied_ = false;
    std::string summary_;

    static bool isViewENode(const SearchState &state, uint32_t b, ENodeId en_id)
    {
        return en_id.value < state.bucket_enode_infos[b].size() && state.bucket_enode_infos[b][en_id.value].is_view;
    }

    static uint32_t enodePages(const SearchState &state, uint32_t b, ENodeId en_id, const MemSpace &space)
    {
        const auto &enode = state.bucket_egraphs[b].getENode(en_id);
        const uint64_t bytes = getStorageSpanBytes(enode.getShape(), enode.getStrides(), enode.getDType());
        const uint64_t alignment = state.getPageAlignment(space);
        const uint64_t pages = bytes / alignment + (bytes % alignment != 0);
        if (pages > std::numeric_limits<uint32_t>::max())
            Error::throw_err("StaticPrivateLayout alternative footprint exceeds page count range");
        return std::max<uint32_t>(1, static_cast<uint32_t>(pages));
    }

    static uint32_t allocationPages(const SearchState &state, VarId off_v)
    {
        const auto &info = state.var_infos[off_v];
        const auto &cls = state.bucket_egraphs[info.bucket_idx].getEClass(info.eclass_id);
        uint32_t pages = std::max<uint32_t>(1, info.size_pages);
        for (auto en_id : cls.enodes)
            pages = std::max(pages, enodePages(state, info.bucket_idx, en_id, cls.mem_space));
        return pages;
    }

    static uint32_t basePages(const SearchState &state, BaseEClassId base_id,
                              const std::vector<VarId> &offset_vars)
    {
        uint32_t pages = 1;
        const auto cache_it = state.cached_vars.find(base_id);
        if (cache_it != state.cached_vars.end())
        {
            const VarInfo &info = state.var_infos[cache_it->second];
            pages = std::max(pages, state.bytesToPages(info.size_bytes, info.mem_space));
        }
        for (VarId off_v : offset_vars)
            pages = std::max(pages, allocationPages(state, off_v));
        return pages;
    }

    struct SpaceLayout
    {
        // Slot of every non-view allocation and of every resolvable view.
        std::vector<std::pair<VarId, int32_t>> fixes;
        // Address hull of each view that may alias several bases.
        std::vector<std::pair<VarId, std::pair<int32_t, int32_t>>> ranges;
        std::string skip_reason;
    };

    // A nested view can have several ultimate bases. Validate their entire
    // address union, even when no single address can be fixed for the view.
    // Unknown/cyclic aliases make this dominance certificate unavailable.
    static bool collectViewAddresses(const SearchState &state, uint32_t b, EClassId cid,
                                     const std::unordered_map<uint32_t, int32_t> &slot_by_cid,
                                     std::unordered_map<uint32_t, std::vector<int32_t>> &memo,
                                     std::unordered_set<uint32_t> &visiting)
    {
        if (memo.count(cid.value))
            return true;
        const auto slot_it = slot_by_cid.find(cid.value);
        if (slot_it != slot_by_cid.end())
        {
            memo[cid.value] = {slot_it->second};
            return true;
        }
        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        if (state.preallocated_buffers.count(cls.base_eclass_id))
        {
            const auto off_it = state.offset_vars[b].find(cid);
            if (off_it == state.offset_vars[b].end() || !state.domains[off_it->second].isFixed())
            {
                LOG(INFO) << "[StaticPrivateLayout uncertified] bucket=" << b << " cid=" << cid.value
                          << " preallocated base=" << cls.base_eclass_id.value << " missing fixed offset";
                return false;
            }
            memo[cid.value] = {state.domains[off_it->second].fixedValue()};
            return true;
        }
        if (!visiting.insert(cid.value).second)
            return false;
        std::vector<int32_t> addresses;
        bool valid = !cls.enodes.empty();
        for (ENodeId en_id : cls.enodes)
        {
            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
            // Production view offsets alias child zero; additional children
            // carry data/metadata dependencies (for example FILL).
            if (!isViewENode(state, b, en_id) || enode.getChildren().empty())
            {
                LOG(INFO) << "[StaticPrivateLayout uncertified] bucket=" << b << " cid=" << cid.value
                          << " base=" << cls.base_eclass_id.value << " space=" << toString(cls.mem_space)
                          << " op=" << enode.getOpType() << " view=" << isViewENode(state, b, en_id)
                          << " children=" << enode.getChildren().size();
                valid = false;
                break;
            }
            const EClassId child = state.bucket_egraphs[b].findConst(enode.getChildren()[0]);
            if (!collectViewAddresses(state, b, child, slot_by_cid, memo, visiting))
            {
                valid = false;
                break;
            }
            const auto &child_addresses = memo.at(child.value);
            addresses.insert(addresses.end(), child_addresses.begin(), child_addresses.end());
        }
        visiting.erase(cid.value);
        if (!valid)
            return false;
        std::sort(addresses.begin(), addresses.end());
        addresses.erase(std::unique(addresses.begin(), addresses.end()), addresses.end());
        memo[cid.value] = std::move(addresses);
        return true;
    }

    // Resolves the address a view-only e-class must share. Returns false when
    // its alternatives can alias different addresses.
    static bool resolveViewAddress(const SearchState &state, uint32_t b, EClassId cid,
                                   const std::unordered_map<uint32_t, int32_t> &slot_by_cid,
                                   std::unordered_map<uint32_t, int32_t> &memo,
                                   std::unordered_set<uint32_t> &visiting, int32_t &out)
    {
        const auto memo_it = memo.find(cid.value);
        if (memo_it != memo.end())
        {
            out = memo_it->second;
            return out >= 0;
        }
        const auto slot_it = slot_by_cid.find(cid.value);
        if (slot_it != slot_by_cid.end())
        {
            out = slot_it->second;
            return true;
        }
        if (!visiting.insert(cid.value).second)
            return false;
        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        int32_t address = -1;
        bool ok = !cls.enodes.empty();
        for (ENodeId en_id : cls.enodes)
        {
            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
            if (!ok || !isViewENode(state, b, en_id) || enode.getChildren().empty())
            {
                ok = false;
                break;
            }
            const EClassId child = state.bucket_egraphs[b].findConst(enode.getChildren()[0]);
            int32_t child_address = -1;
            if (!resolveViewAddress(state, b, child, slot_by_cid, memo, visiting, child_address) ||
                (address >= 0 && address != child_address))
            {
                ok = false;
                break;
            }
            address = child_address;
        }
        visiting.erase(cid.value);
        memo[cid.value] = ok ? address : -1;
        out = ok ? address : -1;
        return ok;
    }

    SpaceLayout planSpace(const SearchState &state, const MemSpace &space) const
    {
        SpaceLayout layout;
        const uint64_t cap_bytes = state.getMemoryCap(space);
        const uint32_t align = state.getPageAlignment(space);
        const auto prealloc_it = state.preallocated_pages.find(space);
        const int64_t prealloc_end = prealloc_it == state.preallocated_pages.end() ? 0 : prealloc_it->second;

        // Cacheable bases: one shared slot each, in base-id order.
        std::map<uint32_t, std::vector<VarId>> cache_offsets;
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (const auto &[cid, off_v] : state.offset_vars[b])
            {
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (!(cls.mem_space == space) || state.preallocated_buffers.count(cls.base_eclass_id))
                    continue;
                if (!state.cached_vars.count(cls.base_eclass_id))
                    continue;
                for (ENodeId en_id : cls.enodes)
                {
                    if (isViewENode(state, b, en_id))
                    {
                        layout.skip_reason = "cacheable e-class with a view alternative";
                        return layout;
                    }
                }
                cache_offsets[cls.base_eclass_id.value].push_back(off_v);
            }
        }
        int64_t cursor = prealloc_end;
        std::unordered_map<uint32_t, int32_t> cache_slot;
        std::unordered_map<uint32_t, uint32_t> cache_slot_pages;
        for (const auto &[base_value, offset_vars] : cache_offsets)
        {
            const uint32_t pages = basePages(state, BaseEClassId{base_value}, offset_vars);
            cache_slot[base_value] = static_cast<int32_t>(cursor);
            cache_slot_pages[base_value] = pages;
            cursor += pages;
        }
        const int64_t cache_end = cursor;
        int64_t max_end = cache_end;

        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            std::unordered_map<uint32_t, int32_t> slot_by_cid;
            std::unordered_map<int32_t, uint32_t> private_pages;
            std::vector<std::pair<EClassId, VarId>> view_only;
            int64_t bucket_cursor = cache_end;
            std::vector<std::pair<EClassId, VarId>> ordered(state.offset_vars[b].begin(), state.offset_vars[b].end());
            std::sort(ordered.begin(), ordered.end(),
                      [](const auto &x, const auto &y) { return x.first.value < y.first.value; });
            for (const auto &[cid, off_v] : ordered)
            {
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (!(cls.mem_space == space) || state.preallocated_buffers.count(cls.base_eclass_id))
                    continue;
                const auto cache_it = cache_slot.find(cls.base_eclass_id.value);
                if (cache_it != cache_slot.end())
                {
                    slot_by_cid[cid.value] = cache_it->second;
                    private_pages[cache_it->second] = cache_slot_pages.at(cls.base_eclass_id.value);
                    continue;
                }
                bool any_view = false, any_non_view = false;
                for (ENodeId en_id : cls.enodes)
                    (isViewENode(state, b, en_id) ? any_view : any_non_view) = true;
                if (!any_non_view)
                {
                    view_only.push_back({cid, off_v});
                    continue;
                }
                if (any_view)
                {
                    layout.skip_reason = "e-class with both view and non-view alternatives";
                    return layout;
                }
                slot_by_cid[cid.value] = static_cast<int32_t>(bucket_cursor);
                const uint32_t pages = allocationPages(state, off_v);
                private_pages[static_cast<int32_t>(bucket_cursor)] = pages;
                bucket_cursor += pages;
            }
            max_end = std::max(max_end, bucket_cursor);
            if (static_cast<uint64_t>(max_end) * align > cap_bytes || max_end > std::numeric_limits<int32_t>::max())
            {
                layout.skip_reason = "private layout exceeds memory capacity";
                return layout;
            }

            for (const auto &[cid, off_v] : ordered)
            {
                const auto slot_it = slot_by_cid.find(cid.value);
                if (slot_it == slot_by_cid.end())
                    continue;
                const VarId sel_v = state.selected_vars[b].at(cid);
                if (state.domains[sel_v].isFixed() && state.domains[sel_v].fixedValue() == 0)
                    continue; // unselected: its address is irrelevant
                if (!state.domains[off_v].contains(slot_it->second))
                {
                    layout.skip_reason = state.var_infos[off_v].name + " domain excludes its private slot";
                    return layout;
                }
                layout.fixes.push_back({off_v, slot_it->second});
            }

            // A selected view must take its base's address, so every
            // candidate base address must lie in the view's domain.
            std::unordered_map<uint32_t, int32_t> memo;
            std::unordered_map<uint32_t, std::vector<int32_t>> address_memo;
            for (const auto &[cid, off_v] : view_only)
            {
                const VarId sel_v = state.selected_vars[b].at(cid);
                if (state.domains[sel_v].isFixed() && state.domains[sel_v].fixedValue() == 0)
                    continue; // unselected: its address is irrelevant
                std::unordered_set<uint32_t> address_visiting;
                if (!collectViewAddresses(state, b, cid, slot_by_cid, address_memo, address_visiting))
                {
                    layout.skip_reason = state.var_infos[off_v].name + " has an uncertified view address";
                    return layout;
                }
                uint32_t view_pages = 1;
                for (auto en_id : state.bucket_egraphs[b].getEClass(cid).enodes)
                    view_pages = std::max(view_pages, enodePages(state, b, en_id, space));
                for (int32_t address : address_memo.at(cid.value))
                {
                    const auto range_it = private_pages.find(address);
                    const bool fits_private = range_it != private_pages.end() && view_pages <= range_it->second;
                    const bool fits_prealloc = address >= 0 && static_cast<int64_t>(address) + view_pages <= prealloc_end;
                    if (!fits_private && !fits_prealloc)
                    {
                        layout.skip_reason = state.var_infos[off_v].name + " view footprint exceeds its reserved base range";
                        return layout;
                    }
                    if (!state.domains[off_v].contains(address))
                    {
                        layout.skip_reason = state.var_infos[off_v].name + " domain excludes a candidate base address";
                        return layout;
                    }
                }
                std::unordered_set<uint32_t> visiting;
                int32_t address = -1;
                if (resolveViewAddress(state, b, cid, slot_by_cid, memo, visiting, address))
                    layout.fixes.push_back({off_v, address});
                else
                {
                    // A selected view shares one of its candidate base
                    // addresses, all checked to lie in its domain above.
                    const auto &addresses = address_memo.at(cid.value);
                    layout.ranges.push_back({off_v, {addresses.front(), addresses.back()}});
                }
            }
        }
        return layout;
    }

    // Necessary selection conditions of every reachable class of a bucket:
    // sorted (class index, allowed alternative mask) pairs that hold whenever
    // the class is active. Root: itself. Other classes: intersection over
    // their supports (parent P, alternative a) of cond(P) + {P: a}, masks of
    // common classes unioned, plus the class itself. Classes without support
    // cannot be active (empty optional). Returns false for cyclic graphs.
    using Conditions = std::vector<std::pair<uint32_t, uint64_t>>;
    struct BucketConditions
    {
        std::unordered_map<uint32_t, uint32_t> index_of;
        std::vector<std::optional<Conditions>> conditions;
    };

    static bool buildConditions(const SearchState &state, uint32_t b, BucketConditions &out)
    {
        const EGraph &egraph = state.bucket_egraphs[b];
        const auto &cids = state.reachable_cids[b];
        const uint32_t n = static_cast<uint32_t>(cids.size());
        out.index_of.clear();
        for (uint32_t i = 0; i < n; ++i)
            out.index_of[cids[i].value] = i;
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> supports(n);
        std::vector<std::vector<uint32_t>> children(n);
        std::vector<uint64_t> nonzero(n, 0);
        for (uint32_t i = 0; i < n; ++i)
        {
            const Domain &sel = state.domains[state.selected_vars[b].at(cids[i])];
            const EClass &cls = egraph.getEClass(cids[i]);
            if (cls.enodes.size() > 63)
                Error::throw_err("StaticPrivateLayout condition masks support at most 63 alternatives");
            for (size_t a = 0; a < cls.enodes.size(); ++a)
            {
                const uint32_t value = static_cast<uint32_t>(a + 1);
                if (!sel.contains(static_cast<int32_t>(value)))
                    continue;
                nonzero[i] |= uint64_t{1} << value;
                for (EClassId child : egraph.getENode(cls.enodes[a]).getChildren())
                {
                    const uint32_t ch = out.index_of.at(egraph.findConst(child).value);
                    supports[ch].push_back({i, value});
                    children[i].push_back(ch);
                }
            }
        }
        std::vector<uint32_t> indegree(n, 0);
        for (uint32_t i = 0; i < n; ++i)
        {
            std::sort(children[i].begin(), children[i].end());
            children[i].erase(std::unique(children[i].begin(), children[i].end()), children[i].end());
            for (uint32_t ch : children[i])
                ++indegree[ch];
        }
        std::vector<uint32_t> order;
        for (uint32_t i = 0; i < n; ++i)
            if (indegree[i] == 0)
                order.push_back(i);
        for (size_t head = 0; head < order.size(); ++head)
            for (uint32_t ch : children[order[head]])
                if (--indegree[ch] == 0)
                    order.push_back(ch);
        if (order.size() != n)
            return false;
        const uint32_t root = out.index_of.at(egraph.findConst(state.bucket_root_ids[b]).value);
        out.conditions.assign(n, std::nullopt);
        for (uint32_t x : order)
        {
            if (nonzero[x] == 0)
                continue; // cannot be active
            std::optional<Conditions> acc;
            if (x == root)
                acc = Conditions{};
            for (const auto &[parent, value] : supports[x])
            {
                if (x == root || !out.conditions[parent])
                    continue;
                Conditions with = *out.conditions[parent];
                auto it = std::lower_bound(with.begin(), with.end(), std::make_pair(parent, uint64_t{0}));
                if (it != with.end() && it->first == parent)
                    it->second &= uint64_t{1} << value;
                else
                    with.insert(it, {parent, uint64_t{1} << value});
                if (!acc)
                {
                    acc = std::move(with);
                    continue;
                }
                Conditions merged;
                size_t p = 0, q = 0;
                while (p < acc->size() && q < with.size())
                {
                    if ((*acc)[p].first < with[q].first)
                        ++p;
                    else if (with[q].first < (*acc)[p].first)
                        ++q;
                    else
                    {
                        merged.push_back({(*acc)[p].first, (*acc)[p].second | with[q].second});
                        ++p;
                        ++q;
                    }
                }
                acc = std::move(merged);
            }
            if (!acc)
                continue; // no active support: cannot be active
            auto it = std::lower_bound(acc->begin(), acc->end(), std::make_pair(x, uint64_t{0}));
            if (it != acc->end() && it->first == x)
                it->second &= nonzero[x];
            else
                acc->insert(it, {x, nonzero[x]});
            out.conditions[x] = std::move(acc);
        }
        return true;
    }

    // True when the two classes can never be active together in this bucket.
    static bool exclusive(const BucketConditions &conditions, uint32_t x, uint32_t y)
    {
        const auto &cx = conditions.conditions[x];
        const auto &cy = conditions.conditions[y];
        if (!cx || !cy)
            return true;
        size_t p = 0, q = 0;
        while (p < cx->size() && q < cy->size())
        {
            if ((*cx)[p].first < (*cy)[q].first)
                ++p;
            else if ((*cy)[q].first < (*cx)[p].first)
                ++q;
            else
            {
                if (((*cx)[p].second & (*cy)[q].second) == 0)
                    return true;
                ++p;
                ++q;
            }
        }
        return false;
    }

    // First free offset >= start for `pages`, avoiding the given intervals.
    static int64_t firstFit(std::vector<std::pair<int64_t, int64_t>> &blockers, int64_t start, int64_t pages)
    {
        std::sort(blockers.begin(), blockers.end());
        int64_t offset = start;
        for (const auto &[lo, hi] : blockers)
        {
            if (offset + pages <= lo)
                break;
            offset = std::max(offset, hi);
        }
        return offset;
    }

    // Shared layout: every pair of allocations that may be active together in
    // a bucket gets disjoint pages; pairs certified exclusive may share. Used
    // when private slots exceed the capacity. Cacheable bases keep one slot in
    // every bucket and may share only if exclusive in every common bucket (so
    // both can never be cached: a cached base is selected in every bucket).
    SpaceLayout planSharedSpace(const SearchState &state, const MemSpace &space) const
    {
        SpaceLayout layout;
        const uint64_t cap_bytes = state.getMemoryCap(space);
        const uint32_t align = state.getPageAlignment(space);
        const auto prealloc_it = state.preallocated_pages.find(space);
        const int64_t prealloc_end = prealloc_it == state.preallocated_pages.end() ? 0 : prealloc_it->second;
        const int64_t cap_pages = static_cast<int64_t>(cap_bytes / align);

        std::vector<BucketConditions> conditions(state.buckets.size());
        // Per base: (bucket, class index, offset var).
        std::map<uint32_t, std::vector<std::tuple<uint32_t, uint32_t, VarId>>> bases;
        std::vector<std::vector<std::pair<uint32_t, VarId>>> plain(state.buckets.size());
        // Buckets are independent; build their conditions in parallel.
        std::vector<uint8_t> acyclic(state.buckets.size(), 0);
        ThreadPool::get().parallel_for(static_cast<uint32_t>(state.buckets.size()), [&](uint32_t b) {
            acyclic[b] = buildConditions(state, b, conditions[b]) ? 1 : 0;
        });
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            if (!acyclic[b])
            {
                layout.skip_reason = "shared layout needs an acyclic bucket graph";
                return layout;
            }
            std::vector<std::pair<EClassId, VarId>> ordered(state.offset_vars[b].begin(), state.offset_vars[b].end());
            std::sort(ordered.begin(), ordered.end(),
                      [](const auto &x, const auto &y) { return x.first.value < y.first.value; });
            for (const auto &[cid, off_v] : ordered)
            {
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (!(cls.mem_space == space) || state.preallocated_buffers.count(cls.base_eclass_id))
                    continue;
                for (ENodeId en_id : cls.enodes)
                    if (isViewENode(state, b, en_id))
                    {
                        layout.skip_reason = "shared layout does not handle view alternatives";
                        return layout;
                    }
                const uint32_t idx = conditions[b].index_of.at(cid.value);
                if (state.cached_vars.count(cls.base_eclass_id))
                {
                    auto &entries = bases[cls.base_eclass_id.value];
                    for (const auto &entry : entries)
                        if (std::get<0>(entry) == b)
                        {
                            layout.skip_reason = "multiple classes of one cache base in a bucket";
                            return layout;
                        }
                    entries.emplace_back(b, idx, off_v);
                }
                else
                    plain[b].push_back({idx, off_v});
            }
        }

        // Cacheable bases: one slot in every bucket, largest first.
        struct Placed
        {
            int64_t lo, hi;
        };
        std::vector<std::pair<uint32_t, uint32_t>> base_order; // (pages, base)
        std::map<uint32_t, uint32_t> base_pages;
        for (const auto &[base, entries] : bases)
        {
            std::vector<VarId> offset_vars;
            for (const auto &entry : entries)
                offset_vars.push_back(std::get<2>(entry));
            base_pages[base] = basePages(state, BaseEClassId{base}, offset_vars);
            base_order.push_back({base_pages[base], base});
        }
        std::sort(base_order.begin(), base_order.end(),
                  [](const auto &x, const auto &y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
        std::map<uint32_t, Placed> base_slot;
        auto basesConflict = [&](uint32_t x, uint32_t y) {
            bool common = false;
            for (const auto &[bx, ix, vx] : bases.at(x))
                for (const auto &[by, iy, vy] : bases.at(y))
                    if (bx == by)
                    {
                        common = true;
                        if (!exclusive(conditions[bx], ix, iy))
                            return true;
                    }
            return !common;
        };
        // Pairwise conflicts in placement order, computed in parallel; row i
        // holds the conflicts of base_order[i] with every earlier base.
        const uint32_t nbases = static_cast<uint32_t>(base_order.size());
        std::vector<std::vector<uint8_t>> conflicts(nbases);
        ThreadPool::get().parallel_for(nbases, [&](uint32_t i) {
            conflicts[i].assign(i, 0);
            for (uint32_t j = 0; j < i; ++j)
                conflicts[i][j] = basesConflict(base_order[i].second, base_order[j].second) ? 1 : 0;
        });
        std::vector<Placed> order_slot(nbases);
        int64_t max_end = prealloc_end;
        for (uint32_t i = 0; i < nbases; ++i)
        {
            const auto &[pages, base] = base_order[i];
            std::vector<std::pair<int64_t, int64_t>> blockers;
            for (uint32_t j = 0; j < i; ++j)
                if (conflicts[i][j])
                    blockers.push_back({order_slot[j].lo, order_slot[j].hi});
            const int64_t offset = firstFit(blockers, prealloc_end, pages);
            order_slot[i] = {offset, offset + pages};
            base_slot[base] = order_slot[i];
            max_end = std::max(max_end, offset + static_cast<int64_t>(pages));
        }

        // Per bucket: plain allocations, largest first, avoiding every
        // non-exclusive base (absent bases conflict conservatively) and
        // every non-exclusive placed allocation of the same bucket.
        // Buckets are independent, so they are placed in parallel.
        std::vector<std::pair<VarId, int32_t>> fixes;
        for (const auto &[base, slot] : base_slot)
            for (const auto &[b, idx, off_v] : bases.at(base))
                fixes.push_back({off_v, static_cast<int32_t>(slot.lo)});
        std::vector<std::vector<std::pair<VarId, int32_t>>> bucket_fixes(state.buckets.size());
        std::vector<int64_t> bucket_end(state.buckets.size(), prealloc_end);
        ThreadPool::get().parallel_for(static_cast<uint32_t>(state.buckets.size()), [&](uint32_t b) {
            auto &fixes = bucket_fixes[b];
            int64_t &max_end = bucket_end[b];
            std::vector<std::tuple<uint32_t, uint32_t, VarId>> items; // (pages, idx, var)
            for (const auto &[idx, off_v] : plain[b])
                items.emplace_back(allocationPages(state, off_v), idx, off_v);
            std::sort(items.begin(), items.end(), [](const auto &x, const auto &y) {
                return std::get<0>(x) > std::get<0>(y) || (std::get<0>(x) == std::get<0>(y) && std::get<1>(x) < std::get<1>(y));
            });
            std::vector<std::tuple<int64_t, int64_t, uint32_t>> placed; // (lo, hi, idx)
            for (const auto &[pages, idx, off_v] : items)
            {
                std::vector<std::pair<int64_t, int64_t>> blockers;
                for (const auto &[base, slot] : base_slot)
                {
                    bool conflict = true;
                    for (const auto &[bb, bidx, bvar] : bases.at(base))
                        if (bb == b)
                            conflict = !exclusive(conditions[b], idx, bidx);
                    if (conflict)
                        blockers.push_back({slot.lo, slot.hi});
                }
                for (const auto &[lo, hi, other] : placed)
                    if (!exclusive(conditions[b], idx, other))
                        blockers.push_back({lo, hi});
                const int64_t offset = firstFit(blockers, prealloc_end, pages);
                placed.emplace_back(offset, offset + pages, idx);
                fixes.push_back({off_v, static_cast<int32_t>(offset)});
                max_end = std::max(max_end, offset + static_cast<int64_t>(pages));
            }
        });
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            fixes.insert(fixes.end(), bucket_fixes[b].begin(), bucket_fixes[b].end());
            max_end = std::max(max_end, bucket_end[b]);
        }
        if (max_end > cap_pages || max_end > std::numeric_limits<int32_t>::max())
        {
            layout.skip_reason = "shared layout exceeds memory capacity (" + std::to_string(max_end) + " pages)";
            return layout;
        }
        for (const auto &[off_v, slot] : fixes)
        {
            const VarInfo &info = state.var_infos[off_v];
            const Domain &sel = state.domains[state.selected_vars[info.bucket_idx].at(info.eclass_id)];
            if (sel.isFixed() && sel.fixedValue() == 0)
                continue; // unselected: its address is irrelevant
            if (!state.domains[off_v].contains(slot))
            {
                layout.skip_reason = info.name + " domain excludes its shared slot";
                return layout;
            }
            layout.fixes.push_back({off_v, slot});
        }
        LOG(INFO) << "[StaticPrivateLayout shared] space=" << toString(space) << " bases=" << bases.size()
                  << " end-pages=" << max_end << " cap-pages=" << cap_pages;
        return layout;
    }

  public:
    // Root propagation visits every dirty variable by type; the first offset
    // visit applies the layout once, before any branching.
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::OFFSET); }

    std::string name() const override
    {
        return "StaticPrivateLayoutPropagator";
    }

    std::string diagnosticSummary() const override { return summary_; }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (applied_)
            return true;
        applied_ = true;
        // Spaces are named once per distinct space, not once per offset.
        std::unordered_set<MemSpace> distinct_spaces;
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
            for (const auto &[cid, off_v] : state.offset_vars[b])
                distinct_spaces.insert(state.bucket_egraphs[b].getEClass(cid).mem_space);
        std::map<std::string, MemSpace> spaces;
        for (const MemSpace &space : distinct_spaces)
            spaces.emplace(toString(space), space);
        for (const auto &[space_name, space] : spaces)
        {
            SpaceLayout layout = planSpace(state, space);
            if (layout.skip_reason == "private layout exceeds memory capacity")
            {
                LOG(INFO) << "[StaticPrivateLayout] space=" << space_name
                          << " private layout exceeds memory capacity; trying exclusivity-shared layout";
                layout = planSharedSpace(state, space);
            }
            if (!layout.skip_reason.empty())
            {
                LOG(INFO) << "[StaticPrivateLayout] space=" << space_name << " skipped: " << layout.skip_reason;
                summary_ += space_name + "=skipped ";
                continue;
            }
            size_t fixed = 0;
            for (const auto &[off_v, slot] : layout.fixes)
            {
                if (state.domains[off_v].isFixed() && state.domains[off_v].fixedValue() == slot)
                    continue;
                if (!state.domains[off_v].contains(slot))
                    Error::throw_err("StaticPrivateLayout slot left the domain of " + state.var_infos[off_v].name);
                state.setDomain(off_v, Domain::makeFixed(slot));
                worklist.push_back(off_v);
                ++fixed;
            }
            for (const auto &[off_v, hull] : layout.ranges)
            {
                Domain next = state.domains[off_v];
                if (!next.setRange(hull.first, hull.second))
                    continue;
                if (next.isEmpty())
                    Error::throw_err("StaticPrivateLayout view hull left the domain of " + state.var_infos[off_v].name);
                state.setDomain(off_v, next);
                worklist.push_back(off_v);
                ++fixed;
            }
            state.private_layout_spaces.insert(space);
            LOG(INFO) << "[StaticPrivateLayout] space=" << space_name << " fixed " << fixed << " of "
                      << layout.fixes.size() << " planned offsets";
            summary_ += space_name + "=" + std::to_string(fixed) + " ";
        }
        return true;
    }
};

} // namespace plan
