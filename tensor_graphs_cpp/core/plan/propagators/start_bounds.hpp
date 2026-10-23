#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{
// Bounds consistency of mandatory start ranks. Two mirrored Hall sweeps
// use sorted endpoints and O(n) scratch space; reservations skip occupied intervals.
class StartBoundsPropagator : public Propagator
{
    struct Entry { VarId var; int64_t lo; int64_t hi; };
    struct HallTree
    {
        size_t size;
        std::vector<int64_t> minimum, lazy, reservation;
        explicit HallTree(const std::vector<int64_t> &coordinates)
            : size(coordinates.size()), minimum(4 * size), lazy(4 * size),
              reservation(4 * size, INT64_MIN)
        { build(1, 0, size - 1, coordinates); }
        void build(size_t node, size_t left, size_t right, const std::vector<int64_t> &coordinates)
        {
            if (left == right) { minimum[node] = -coordinates[left]; return; }
            const size_t mid = (left + right) / 2;
            build(node * 2, left, mid, coordinates);
            build(node * 2 + 1, mid + 1, right, coordinates);
            minimum[node] = std::min(minimum[node * 2], minimum[node * 2 + 1]);
        }
        void add(size_t node, size_t left, size_t right, size_t end)
        {
            if (right <= end) { --minimum[node]; --lazy[node]; return; }
            const size_t mid = (left + right) / 2;
            add(node * 2, left, mid, end);
            if (end > mid) add(node * 2 + 1, mid + 1, right, end);
            minimum[node] = lazy[node] + std::min(minimum[node * 2], minimum[node * 2 + 1]);
        }
        // Find the earliest overloaded or saturated lower endpoint <= end.
        size_t find(size_t node, size_t left, size_t right, size_t end, int64_t carry, int64_t limit) const
        {
            if (left > end || minimum[node] + carry > limit) return size;
            if (left == right) return left;
            carry += lazy[node];
            const size_t mid = (left + right) / 2;
            const size_t first = find(node * 2, left, mid, end, carry, limit);
            return first != size ? first : find(node * 2 + 1, mid + 1, right, end, carry, limit);
        }
        void reserve(size_t node, size_t left, size_t right, size_t begin, size_t end, int64_t bound)
        {
            if (right < begin || left > end) return;
            if (begin <= left && right <= end) { reservation[node] = std::max(reservation[node], bound); return; }
            const size_t mid = (left + right) / 2;
            reserve(node * 2, left, mid, begin, end, bound);
            reserve(node * 2 + 1, mid + 1, right, begin, end, bound);
        }
        int64_t reserved(size_t node, size_t left, size_t right, size_t index, int64_t bound = INT64_MIN) const
        {
            bound = std::max(bound, reservation[node]);
            if (left == right) return bound;
            const size_t mid = (left + right) / 2;
            return index <= mid ? reserved(node * 2, left, mid, index, bound)
                                : reserved(node * 2 + 1, mid + 1, right, index, bound);
        }
    };
    std::string conflict_reason_;
    uint64_t hall_intervals_ = 0;

    bool sweep(SearchState &state, uint32_t bucket, std::vector<Entry> entries, bool mirrored,
               const std::vector<int64_t> &occupied_ranks)
    {
        if (entries.empty()) return true;
        if (mirrored)
            for (auto &entry : entries) { const int64_t lo = entry.lo; entry.lo = -entry.hi; entry.hi = -lo; }
        std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) { return a.hi < b.hi; });
        std::vector<int64_t> coordinates;
        for (const auto &entry : entries)
        { coordinates.push_back(entry.lo); coordinates.push_back(entry.hi); coordinates.push_back(entry.hi + 1); }
        std::sort(coordinates.begin(), coordinates.end());
        coordinates.erase(std::unique(coordinates.begin(), coordinates.end()), coordinates.end());
        HallTree tree(coordinates);
        for (size_t begin = 0; begin < entries.size();)
        {
            size_t end = begin;
            const int64_t upper = entries[begin].hi;
            while (end < entries.size() && entries[end].hi == upper) ++end;
            // Only earlier upper-endpoint groups can reserve a variable's lower
            // bound: members of a Hall interval must not prune themselves.
            for (size_t i = begin; i < end; ++i)
            {
                const auto &entry = entries[i];
                const size_t index = std::lower_bound(coordinates.begin(), coordinates.end(), entry.lo) - coordinates.begin();
                int64_t bound = entry.lo;
                for (;;)
                {
                    const size_t bound_index = std::lower_bound(coordinates.begin(), coordinates.end(), bound) - coordinates.begin();
                    if (bound_index == tree.size) break;
                    const int64_t next = tree.reserved(1, 0, tree.size - 1, bound_index);
                    if (next <= bound) break;
                    bound = next;
                }
                if (bound > entry.lo) tree.reserve(1, 0, tree.size - 1, index, index, bound);
                if (bound > entry.lo)
                {
                    if (bound > entry.hi)
                    {
                        conflict_reason_ = "bucket=" + std::to_string(bucket) + " Hall reservations cover " +
                            state.var_infos[entry.var].name() + state.domains[entry.var].toString() +
                            " required-" + (mirrored ? std::string("max=") : std::string("min=")) +
                            std::to_string(mirrored ? -bound : bound);
                        return false;
                    }
                    Domain domain = state.domains[entry.var];
                    const int64_t available_rank = mirrored ? -bound : bound;
                    const int64_t raw_bound = available_rank +
                        (std::upper_bound(occupied_ranks.begin(), occupied_ranks.end(), available_rank) - occupied_ranks.begin());
                    if (mirrored) domain.setMax(static_cast<int32_t>(raw_bound));
                    else domain.setMin(static_cast<int32_t>(raw_bound));
                    if (domain.isEmpty()) Error::throw_err("Hall bound adjustment unexpectedly emptied a domain");
                    state.setDomain(entry.var, domain);
                }
                tree.add(1, 0, tree.size - 1, index);
            }
            const size_t last = std::upper_bound(coordinates.begin(), coordinates.end(), upper) - coordinates.begin() - 1;
            const size_t overloaded = tree.find(1, 0, tree.size - 1, last, 0, -upper - 2);
            if (overloaded != tree.size)
            {
                const int64_t lower = coordinates[overloaded];
                size_t count = 0;
                std::string witnesses;
                for (const auto &entry : entries)
                    if (entry.lo >= lower && entry.hi <= upper)
                    {
                        ++count;
                        if (count <= 8) witnesses += " " + state.var_infos[entry.var].name() + state.domains[entry.var].toString();
                    }
                conflict_reason_ = "bucket=" + std::to_string(bucket) + " available-rank-interval=[" +
                    std::to_string(mirrored ? -upper : lower) + ".." + std::to_string(mirrored ? -lower : upper) +
                    "] available-slots=true occupied-fixed=" + std::to_string(occupied_ranks.size()) + " capacity=" + std::to_string(upper - lower + 1) + " mandatory=" + std::to_string(count) +
                    " witnesses=" + witnesses;
                return false;
            }
            const size_t saturated = tree.find(1, 0, tree.size - 1, last, 0, -upper - 1);
            if (saturated != tree.size)
            {
                ++hall_intervals_;
                tree.reserve(1, 0, tree.size - 1, saturated, last, upper + 1);
            }
            begin = end;
        }
        return true;
    }
    bool propagateBucket(SearchState &state, uint32_t bucket)
    {
        std::vector<Entry> entries;
        std::vector<std::pair<int32_t, VarId>> occupied;
        for (const auto &pair : state.selected_vars.at(bucket))
        {
            if (state.domains[pair.second].contains(0)) continue;
            const VarId var = state.start_vars[bucket].at(pair.first);
            const Domain &domain = state.domains[var];
            if (domain.isEmpty()) { conflict_reason_ = "empty mandatory start: " + state.var_infos[var].name(); return false; }
            if (domain.isFixed()) occupied.emplace_back(domain.fixedValue(), var);
            else entries.push_back({var, domain.getMin(), domain.getMax()});
        }
        std::sort(occupied.begin(), occupied.end());
        std::vector<int32_t> fixed_values;
        std::vector<int64_t> occupied_ranks;
        for (size_t i = 0; i < occupied.size(); ++i)
        {
            if (i && occupied[i].first == occupied[i - 1].first)
            {
                conflict_reason_ = "bucket=" + std::to_string(bucket) + " fixed collision at " +
                    std::to_string(occupied[i].first) + " witnesses=" + state.var_infos[occupied[i - 1].second].name() +
                    "/" + state.var_infos[occupied[i].second].name();
                return false;
            }
            fixed_values.push_back(occupied[i].first);
            occupied_ranks.push_back(static_cast<int64_t>(occupied[i].first) - static_cast<int64_t>(i));
        }
        // Remove fixed slots from the rank space. In this coordinate system
        // interval length already subtracts every occupied slot inside it.
        // This avoids inserting thousands of singleton Hall intervals again
        // on each branch while retaining exactly their capacity restrictions.
        auto lower_rank = [&](int32_t value) {
            return static_cast<int64_t>(value) - (std::lower_bound(fixed_values.begin(), fixed_values.end(), value) - fixed_values.begin());
        };
        auto upper_rank = [&](int32_t value) {
            return static_cast<int64_t>(value) - (std::upper_bound(fixed_values.begin(), fixed_values.end(), value) - fixed_values.begin());
        };
        auto raw_rank = [&](int64_t rank) {
            return rank + (std::upper_bound(occupied_ranks.begin(), occupied_ranks.end(), rank) - occupied_ranks.begin());
        };
        for (auto &entry : entries)
        {
            Domain domain = state.domains[entry.var];
            for (;;)
            {
                entry.lo = lower_rank(domain.getMin());
                entry.hi = upper_rank(domain.getMax());
                if (entry.lo > entry.hi)
                {
                    conflict_reason_ = "bucket=" + std::to_string(bucket) + " fixed starts cover " +
                        state.var_infos[entry.var].name() + domain.toString();
                    return false;
                }
                const bool changed_lower = domain.setMin(static_cast<int32_t>(raw_rank(entry.lo)));
                const bool changed_upper = domain.setMax(static_cast<int32_t>(raw_rank(entry.hi)));
                if (domain.isEmpty())
                {
                    conflict_reason_ = "bucket=" + std::to_string(bucket) + " fixed starts cover masked domain " +
                        state.var_infos[entry.var].name() + state.domains[entry.var].toString();
                    return false;
                }
                if (!changed_lower && !changed_upper) break;
            }
            state.setDomain(entry.var, domain);
        }
        // With n common available ranks, any bound has interval support:
        // assign the other n-1 variables inside that common intersection.
        // This is an exact no-pruning certificate for interval bounds, not a
        // size heuristic. Masks deliberately use their interval hulls here.
        int64_t common_lower = INT64_MIN, common_upper = INT64_MAX;
        for (const auto &entry : entries)
        {
            common_lower = std::max(common_lower, entry.lo);
            common_upper = std::min(common_upper, entry.hi);
        }
        if (entries.empty() || (common_upper >= common_lower &&
            common_upper - common_lower + 1 >= static_cast<int64_t>(entries.size())))
            return true;
        if (!sweep(state, bucket, entries, false, occupied_ranks)) return false;
        for (auto &entry : entries)
        {
            entry.lo = lower_rank(state.domains[entry.var].getMin());
            entry.hi = upper_rank(state.domains[entry.var].getMax());
        }
        return sweep(state, bucket, entries, true, occupied_ranks);
    }
  public:
    std::string name() const override { return "StartBoundsPropagator"; }
    std::string conflictReason() const override { return conflict_reason_; }
    std::string diagnosticSummary() const override { return "hall-intervals=" + std::to_string(hall_intervals_); }
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::START) | varTypeMask(VarType::SELECTED); }
    StartSelectionGuard startSelectionGuard() const override { return StartSelectionGuard::NON_OPTIONAL; }
    PropagationBatch propagationBatch(const SearchState &, VarId changed) const override
    { return changed == kInvalidVarId ? PropagationBatch::GLOBAL : PropagationBatch::BUCKET; }
    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &) override
    {
        conflict_reason_.clear();
        if (changed != kInvalidVarId) return propagateBucket(state, state.var_infos.at(changed).bucket_idx);
        for (uint32_t bucket = 0; bucket < state.buckets.size(); ++bucket)
            if (!propagateBucket(state, bucket)) return false;
        return true;
    }
};
} // namespace plan
