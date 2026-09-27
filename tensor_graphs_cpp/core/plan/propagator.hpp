// tensor_graphs_cpp/core/plan/propagator.hpp
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/constants.hpp"
#include "core/plan/domain.hpp"
#include "core/plan/search_state.hpp"

namespace plan
{

class Propagator
{
  public:
    virtual ~Propagator() = default;
    virtual std::string name() const = 0;

    // Shrinks variable domains in state. Returns false on contradiction.
    virtual bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) = 0;

    // Returns a lower bound on total makespan / cost.
    virtual float computeLowerBound(const SearchState &state)
    {
        return 0.0f;
    }
};

inline const std::vector<EClassId> &getReachableCids(const SearchState &state, uint32_t b)
{
    static const std::vector<EClassId> empty;
    return (b < state.reachable_cids.size() && !state.reachable_cids[b].empty())
               ? state.reachable_cids[b]
               : empty;
}

class SelectionPropagator : public Propagator
{
  public:
    std::string name() const override
    {
        return "SelectionPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        VarInfo info = state.var_infos[changed];
        if (info.type != VarType::SELECTED)
            return true;
        EClassId cid = info.eclass_id;
        VarId v = state.selected_vars[info.bucket_idx].at(cid);
        const Domain &dom = state.domains[v];
        if (dom.isEmpty())
            return false;

        // For each eclass that is definitely selected and fixed to enode e,
        // all children of enode e cannot be 0.
        if (dom.isFixed() && dom.fixedValue() > 0)
        {
            uint32_t en_idx = static_cast<uint32_t>(dom.fixedValue() - 1);
            const EClass &cls = state.bucket_egraphs[info.bucket_idx].getEClass(cid);
            if (en_idx < cls.enodes.size())
            {
                ENodeId en_id = cls.enodes[en_idx];
                const ENode &enode = state.bucket_egraphs[info.bucket_idx].getENode(en_id);
                for (EClassId child : enode.getChildren())
                {
                    EClassId canon_child = state.bucket_egraphs[info.bucket_idx].findConst(child);
                    auto ch_it = state.selected_vars[info.bucket_idx].find(canon_child);
                    if (ch_it != state.selected_vars[info.bucket_idx].end())
                    {
                        VarId ch_v = ch_it->second;
                        Domain ch_dom = state.domains[ch_v];
                        if (ch_dom.contains(0))
                        {
                            ch_dom.remove(0);
                            if (ch_dom.isEmpty())
                                return false;
                            state.setDomain(ch_v, ch_dom);
                        }
                    }
                }
            }
        }

        std::vector<VarId> unreachable;
        state.updateSelectionReachability(changed, unreachable);
        for (VarId sel_v : unreachable)
        {
            const Domain &sel_dom = state.domains[sel_v];
            if (!sel_dom.contains(0))
                return false;
            if (!sel_dom.isFixed())
                state.setDomain(sel_v, Domain::makeFixed(0, sel_dom.is_mask));
        }
        return true;
    }
};

class CachePropagator : public Propagator
{
    bool propagateSelection(SearchState &state, VarId sel_var)
    {
        const auto &data = state.propagation;
        const VarId cache_var = data.cache_vars[sel_var];
        if (cache_var == kInvalidVarId)
            return true;
        Domain selection = state.domains[sel_var];
        const auto &alternatives = data.alternatives[sel_var];
        for (uint32_t index = 0; index < alternatives.size(); ++index)
        {
            const int32_t value = static_cast<int32_t>(index + 1);
            const OpType op_type = alternatives[index].op_type;
            if (!selection.contains(value) || (op_type != OpType::CACHE && op_type != OpType::SCATTER))
                continue;
            const Domain cache_domain = state.domains[cache_var];
            if (cache_domain.isFixed() && cache_domain.fixedValue() == 0)
            {
                selection.remove(value);
                if (selection.isEmpty())
                    return false;
                state.setDomain(sel_var, selection);
            }
            else if (op_type == OpType::CACHE && selection.isFixed())
            {
                Domain required = cache_domain;
                required.remove(0);
                if (required.isEmpty())
                    return false;
                state.setDomain(cache_var, required);
            }
        }
        return true;
    }

  public:
    std::string name() const override { return "CachePropagator"; }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        const VarType type = state.var_infos[changed].type;
        if (type != VarType::CACHED && type != VarType::SELECTED)
            return true;
        state.ensurePropagationState();
        auto &data = state.propagation;
        if (type == VarType::SELECTED)
        {
            if (!propagateSelection(state, changed))
                return false;
        }
        else
            for (VarId user : data.cache_users[changed])
                if (!propagateSelection(state, user))
                    return false;

        // A cache fixed to zero changes its users, but cannot consume budget.
        // Revisit a memory pool only when its total changed (or on initialization).
        auto spaces = std::move(data.cache_budget_dirty);
        data.cache_budget_dirty.clear();
        auto fail = [&]() {
            data.cache_budget_dirty.insert(spaces.begin(), spaces.end());
            return false;
        };
        for (const MemSpace &space : spaces)
        {
            const uint64_t used = data.fixed_cache_bytes[space];
            const uint64_t cap = state.getMemoryCap(space);
            if (used > cap)
                return fail();
            for (uint32_t index : data.candidates_by_space.at(space))
            {
                const auto &candidate = state.candidates[index];
                const VarId cache_var = state.cached_vars.at(candidate.base_eclass_id);
                Domain domain = state.domains[cache_var];
                if (!domain.isFixed() && domain.contains(1) && candidate.size_bytes > cap - used)
                {
                    domain.remove(1);
                    if (domain.isEmpty())
                        return fail();
                    state.setDomain(cache_var, domain);
                }
            }
        }
        return true;
    }
};

class TopologicalOrderPropagator : public Propagator
{
    bool hasFixedPath(const SearchState &state, VarId from, VarId target) const
    {
        std::vector<VarId> frontier{from};
        std::unordered_set<VarId> visited;
        for (size_t head = 0; head < frontier.size(); ++head)
        {
            const VarId current = frontier[head];
            if (current == target)
                return true;
            if (current == kInvalidVarId || !visited.insert(current).second)
                continue;
            const int32_t index = state.selectedAlternative(current);
            if (index >= 0)
            {
                const auto &children = state.propagation.alternatives[current][index].children;
                frontier.insert(frontier.end(), children.begin(), children.end());
            }
        }
        return false;
    }

    bool pruneSelection(SearchState &state, VarId sel_var)
    {
        Domain selection = state.domains[sel_var];
        const bool was_fixed = selection.isFixed();
        const auto &alternatives = state.propagation.alternatives[sel_var];
        for (uint32_t index = 0; index < alternatives.size(); ++index)
        {
            const int32_t value = static_cast<int32_t>(index + 1);
            if (!selection.contains(value))
                continue;
            for (VarId child : alternatives[index].children)
            {
                const bool cycle = hasFixedPath(state, child, sel_var);
                if (was_fixed)
                {
                    if (cycle)
                        return false;
                }
                else if (cycle || child == kInvalidVarId || state.domains[child].getMax() <= 0)
                {
                    selection.remove(value);
                    break;
                }
            }
        }
        if (selection.isEmpty())
            return false;
        state.setDomain(sel_var, selection);
        return true;
    }

    bool propagateStart(SearchState &state, VarId sel_var)
    {
        const int32_t index = state.selectedAlternative(sel_var);
        if (index < 0)
            return true;
        const auto &alternative = state.propagation.alternatives[sel_var][index];
        if (alternative.start_var == kInvalidVarId)
            return true;
        Domain start = state.domains[alternative.start_var];
        for (VarId child : alternative.children)
        {
            const int32_t child_index = state.selectedAlternative(child);
            if (child_index < 0)
                continue;
            const VarId child_start = state.propagation.alternatives[child][child_index].start_var;
            if (child_start != kInvalidVarId)
                start.setMin(state.domains[child_start].getMin() + 1);
        }
        if (start.isEmpty())
            return false;
        state.setDomain(alternative.start_var, start);
        return true;
    }

  public:
    std::string name() const override { return "TopologicalOrderPropagator"; }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        const VarType type = state.var_infos[changed].type;
        if (type != VarType::SELECTED && type != VarType::START)
            return true;
        state.ensurePropagationState();
        const auto &data = state.propagation;
        const VarId owner = data.owners[changed];
        if (owner == kInvalidVarId)
            return true;
        if (type == VarType::START)
        {
            const int32_t index = state.selectedAlternative(owner);
            if (index < 0 || data.alternatives[owner][index].start_var != changed)
                return true;
        }
        else
        {
            // A new fixed edge can close a cycle only in an alternative whose
            // child reaches this node through fixed edges. Walk those ancestors
            // and inspect their possible parents, including unfixed alternatives.
            std::vector<VarId> frontier{owner};
            std::unordered_set<VarId> visited{owner};
            std::unordered_set<VarId> affected{owner};
            for (size_t head = 0; head < frontier.size(); ++head)
                for (VarId parent : data.parents[frontier[head]])
                {
                    if (!state.domains[parent].isFixed())
                        affected.insert(parent);
                    if (state.isSelectedParent(parent, frontier[head]) && visited.insert(parent).second)
                        frontier.push_back(parent);
                }
            for (VarId sel_var : affected)
                if (!pruneSelection(state, sel_var))
                    return false;
        }
        if (!propagateStart(state, owner))
            return false;
        for (VarId parent : data.parents[owner])
            if (state.isSelectedParent(parent, owner) && !propagateStart(state, parent))
                return false;
        return true;
    }
};

class EngineSchedulePropagator : public Propagator
{
    bool propagateStart(SearchState &state, VarId start_var)
    {
        const auto &data = state.propagation;
        const VarId owner = data.owners[start_var];
        const int32_t index = state.selectedAlternative(owner);
        const auto &alternative = data.alternatives[owner][index];
        Domain start = state.domains[start_var];
        if (start.isEmpty())
            return false;
        if (start.isFixed())
        {
            for (const auto &[engine_idx, slot] : alternative.engine_slots)
                if (data.engines[engine_idx].fixed_starts.at(start.fixedValue()) > 1)
                    return false;
            return true;
        }
        int64_t candidate = start.getMin();
        while (candidate <= start.getMax())
        {
            bool blocked = false;
            for (const auto &[engine_idx, slot] : alternative.engine_slots)
                blocked |= data.engines[engine_idx].fixed_starts.count(static_cast<int32_t>(candidate)) != 0;
            if (!blocked)
                break;
            ++candidate;
        }
        if (candidate > start.getMax())
            return false;
        start.setMin(static_cast<int32_t>(candidate));
        state.setDomain(start_var, start);
        return true;
    }

  public:
    std::string name() const override { return "EngineSchedulePropagator"; }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        const VarType type = state.var_infos[changed].type;
        if (type != VarType::SELECTED && type != VarType::START)
            return true;
        state.ensurePropagationState();
        const auto &data = state.propagation;
        const VarId owner = data.owners[changed];
        const int32_t index = state.selectedAlternative(owner);
        if (index < 0)
            return true;
        const auto &alternative = data.alternatives[owner][index];
        const VarId start_var = alternative.start_var;
        if (start_var == kInvalidVarId || (type == VarType::START && start_var != changed))
            return true;
        if (!propagateStart(state, start_var))
            return false;
        if (state.domains[start_var].isFixed())
        {
            std::unordered_set<VarId> peers;
            for (const auto &[engine_idx, slot] : alternative.engine_slots)
            {
                const auto &starts = data.engines[engine_idx].active_starts;
                peers.insert(starts.begin(), starts.end());
            }
            for (VarId peer : peers)
                if (!propagateStart(state, peer))
                    return false;
        }
        return true;
    }
};

class MemoryNonOverlapPropagator : public Propagator
{
    using Allocation = PropagationState::Allocation;

    bool propagateView(SearchState &state, VarId owner)
    {
        const int32_t index = state.selectedAlternative(owner);
        if (index < 0)
            return true;
        const auto &alternative = state.propagation.alternatives[owner][index];
        if (!alternative.is_view || alternative.children.empty() || alternative.children[0] == kInvalidVarId)
            return true;
        const auto &info = state.var_infos[owner];
        const auto &offsets = state.offset_vars[info.bucket_idx];
        const auto view_it = offsets.find(info.eclass_id);
        const auto child_it = offsets.find(state.var_infos[alternative.children[0]].eclass_id);
        if (view_it == offsets.end() || child_it == offsets.end() || !state.domains[child_it->second].isFixed())
            return true;
        Domain offset = state.domains[view_it->second];
        const int32_t value = state.domains[child_it->second].fixedValue();
        offset.setRange(value, value);
        if (offset.isEmpty())
            return false;
        state.setDomain(view_it->second, offset);
        return true;
    }

    bool updateAllocation(SearchState &state, VarId owner)
    {
        auto &data = state.propagation;
        const VarInfo &info = state.var_infos[owner];
        const uint32_t b = info.bucket_idx;
        auto &bucket = data.buckets[b];
        auto &allocation = data.allocations[owner];
        if (allocation.active)
            bucket.allocations[allocation.mem_space].erase(owner);
        bucket.open_lifetimes.erase(owner);
        allocation = Allocation{};
        const int32_t index = state.selectedAlternative(owner);
        if (index < 0)
            return true;
        const auto &alternative = data.alternatives[owner][index];
        const auto off_it = state.offset_vars[b].find(info.eclass_id);
        if (alternative.is_view || off_it == state.offset_vars[b].end())
            return true;
        const Domain &offset = state.domains[off_it->second];
        if (offset.isEmpty())
            return false;
        if (alternative.start_var == kInvalidVarId || !state.domains[alternative.start_var].isFixed())
            return true;
        const EClass &cls = state.bucket_egraphs[b].getEClass(info.eclass_id);
        allocation.active = true;
        allocation.offset_var = off_it->second;
        allocation.mem_space = cls.mem_space;
        allocation.size = std::max(1u, state.bytesToPages(getSizeBytes(cls.shape, cls.dtype), cls.mem_space));
        allocation.offset_fixed = offset.isFixed();
        if (offset.isFixed())
            allocation.offset = static_cast<uint32_t>(offset.fixedValue());
        const auto pre_it = state.preallocated_buffers.find(cls.base_eclass_id);
        if (cls.base_eclass_id != BaseEClassId{} && pre_it != state.preallocated_buffers.end() && pre_it->second.offset >= 0)
        {
            allocation.offset = static_cast<uint32_t>(pre_it->second.offset / state.getPageAlignment(cls.mem_space));
            allocation.size = state.bytesToPages(pre_it->second.size, cls.mem_space);
            allocation.offset_fixed = true;
        }
        allocation.start = state.domains[alternative.start_var].fixedValue();
        allocation.end = allocation.start + 1;

        // Follow consumers through selected views. Only this physical value's
        // lifetime is recomputed; other allocations retain their cached entries.
        std::vector<VarId> frontier{owner};
        std::unordered_set<VarId> visited;
        bool has_consumer = false;
        bool open_lifetime = false;
        for (size_t head = 0; head < frontier.size(); ++head)
            for (VarId parent : data.parents[frontier[head]])
            {
                if (!state.isSelectedParent(parent, frontier[head]) || !visited.insert(parent).second)
                    continue;
                const auto &consumer = data.alternatives[parent][state.selectedAlternative(parent)];
                if (consumer.is_view)
                    frontier.push_back(parent);
                else
                {
                    has_consumer = true;
                    if (consumer.start_var != kInvalidVarId && state.domains[consumer.start_var].isFixed())
                        allocation.end = std::max(allocation.end, state.domains[consumer.start_var].fixedValue() + 1);
                    else
                        open_lifetime = true;
                }
            }
        if (!has_consumer || open_lifetime)
        {
            bucket.open_lifetimes.insert(owner);
            if (!bucket.finish_times.empty())
                allocation.end = std::max(allocation.end, *bucket.finish_times.rbegin());
        }
        bucket.allocations[cls.mem_space].insert(owner);
        return true;
    }

    bool restrictOffset(SearchState &state, const Allocation &fixed, const Allocation &candidate)
    {
        Domain offset = state.domains[candidate.offset_var];
        const int64_t before_max = static_cast<int64_t>(fixed.offset) - candidate.size;
        const int64_t after_min = static_cast<int64_t>(fixed.offset) + fixed.size;
        if (offset.getMin() > before_max)
        {
            if (after_min > INT32_MAX)
                return false;
            offset.setMin(static_cast<int32_t>(after_min));
        }
        else if (offset.getMax() < after_min)
        {
            if (before_max < INT32_MIN)
                return false;
            offset.setMax(static_cast<int32_t>(before_max));
        }
        if (offset.isEmpty())
            return false;
        state.setDomain(candidate.offset_var, offset);
        return true;
    }

  public:
    std::string name() const override { return "MemoryNonOverlapPropagator"; }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        state.ensurePropagationState();
        auto &data = state.propagation;
        std::unordered_set<VarId> affected;
        auto fail = [&]() {
            // A contradiction can interrupt initialization or a batch of
            // repairs. Preserve all pending work for rollback and replay.
            data.memory_dirty.insert(affected.begin(), affected.end());
            return false;
        };
        // Offset equalities can dirty another view in the same alias chain.
        while (!data.memory_dirty.empty())
        {
            auto dirty = std::move(data.memory_dirty);
            data.memory_dirty.clear();
            affected.insert(dirty.begin(), dirty.end());
            for (VarId owner : dirty)
            {
                if (!propagateView(state, owner))
                    return fail();
            }
        }
        for (VarId owner : affected)
            if (!updateAllocation(state, owner))
                return fail();
        for (VarId owner : affected)
        {
            const Allocation &allocation = data.allocations[owner];
            if (!allocation.active)
                continue;
            const auto &bucket = data.buckets[state.var_infos[owner].bucket_idx];
            for (VarId peer : bucket.allocations.at(allocation.mem_space))
            {
                if (peer == owner)
                    continue;
                const Allocation &other = data.allocations[peer];
                if (allocation.end <= other.start || other.end <= allocation.start)
                    continue;
                if (allocation.offset_fixed && other.offset_fixed)
                {
                    if (static_cast<uint64_t>(allocation.offset) + allocation.size > other.offset &&
                        static_cast<uint64_t>(other.offset) + other.size > allocation.offset)
                        return fail();
                }
                else if (allocation.offset_fixed)
                {
                    if (!restrictOffset(state, allocation, other))
                        return fail();
                }
                else if (other.offset_fixed && !restrictOffset(state, other, allocation))
                    return fail();
            }
        }
        return true;
    }
};

class CostLowerBoundPropagator : public Propagator
{
  public:
    std::string name() const override { return "CostLowerBoundPropagator"; }

    float computeLowerBound(const SearchState &state) override
    {
        return state.costLowerBound();
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        return computeLowerBound(state) < state.best_cost;
    }
};

} // namespace plan
