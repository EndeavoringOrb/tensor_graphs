#pragma once

// Included after SearchState's definition.
namespace plan
{

inline int32_t SearchState::selectedAlternative(VarId sel_var) const
{
    if (sel_var == kInvalidVarId)
        return -1;
    const Domain &domain = domains[sel_var];
    if (!domain.isFixed() || domain.fixedValue() <= 0)
        return -1;
    const int32_t index = domain.fixedValue() - 1;
    return static_cast<size_t>(index) < propagation.alternatives[sel_var].size() ? index : -1;
}

inline bool SearchState::isSelectedParent(VarId parent, VarId child) const
{
    const int32_t index = selectedAlternative(parent);
    if (index < 0)
        return false;
    const auto &children = propagation.alternatives[parent][index].children;
    return std::find(children.begin(), children.end(), child) != children.end();
}

inline void SearchState::ensurePropagationState() const
{
    auto &data = propagation;
    if (data.initialized)
        return;
    data.owners.assign(numVars(), kInvalidVarId);
    data.alternatives.resize(numVars());
    data.parents.resize(numVars());
    data.view_neighbors.resize(numVars());
    data.cache_vars.assign(numVars(), kInvalidVarId);
    data.cache_users.resize(numVars());
    data.cache_candidates.resize(numVars());
    data.allocations.resize(numVars());
    data.buckets.resize(buckets.size());
    data.start_precedence.resize(buckets.size());
    data.write_after_read.resize(buckets.size());
    data.cost.initialize(buckets.size());

    data.topo_path_visited_stamp.assign(numVars(), 0);
    data.topo_path_stamp = 0;
    data.topo_ancestor_visited_stamp.assign(numVars(), 0);
    data.topo_ancestor_stamp = 0;
    data.topo_affected_stamp.assign(numVars(), 0);
    data.topo_aff_stamp = 0;
    data.mem_alloc_visited_stamp.assign(numVars(), 0);
    data.mem_alloc_stamp = 0;

    for (uint32_t b = 0; b < buckets.size(); ++b)
    {
        std::unordered_map<Engine, uint32_t> engine_indices;
        auto &precedence = data.start_precedence[b];
        precedence.visited_stamp.assign(bucket_egraphs[b].classes.size(), 0);
        precedence.frontier.reserve(bucket_egraphs[b].classes.size());
        for (EClassId cid : reachable_cids[b])
        {
            const VarId sel_var = selected_vars[b].at(cid);
            data.owners[sel_var] = sel_var;
            const auto offset_it = offset_vars[b].find(cid);
            if (offset_it != offset_vars[b].end())
                data.owners[offset_it->second] = sel_var;
            const EClass &cls = bucket_egraphs[b].getEClass(cid);
            const auto cache_it = cached_vars.find(cls.base_eclass_id);
            if (cache_it != cached_vars.end())
            {
                data.cache_vars[sel_var] = cache_it->second;
                data.cache_users[cache_it->second].push_back(sel_var);
            }
            for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
            {
                PropagationState::Alternative alternative;
                const ENodeId en_id = cls.enodes[en_idx];
                const ENode &enode = bucket_egraphs[b].getENode(en_id);
                alternative.op_type = enode.getOpType();
                if (b < bucket_enode_infos.size() && en_id.value < bucket_enode_infos[b].size())
                {
                    const ENodeInfo &info = bucket_enode_infos[b][en_id.value];
                    alternative.is_view = info.is_view;
                    if (info.cost > 0.0f && info.cost < TGConstants::INF)
                        alternative.cost = info.cost;
                }
                const auto start_it = start_vars[b].find(cid);
                if (start_it != start_vars[b].end() && en_idx < start_it->second.size())
                {
                    alternative.start_var = start_it->second[en_idx];
                    data.owners[alternative.start_var] = sel_var;
                }
                for (EClassId child : enode.getChildren())
                {
                    const EClassId canonical_child = bucket_egraphs[b].findConst(child);
                    const auto child_it = selected_vars[b].find(canonical_child);
                    const VarId child_var = child_it == selected_vars[b].end() ? kInvalidVarId : child_it->second;
                    alternative.children.push_back(child_var);
                    if (child_var == kInvalidVarId)
                        continue;
                    data.parents[child_var].push_back(sel_var);
                    if (alternative.is_view)
                    {
                        data.view_neighbors[child_var].push_back(sel_var);
                        data.view_neighbors[sel_var].push_back(child_var);
                    }
                }
                for (const Engine &engine : enode.getEngines())
                {
                    auto result = engine_indices.emplace(engine, static_cast<uint32_t>(data.engines.size()));
                    if (result.second)
                    {
                        data.engines.emplace_back();
                        data.engines.back().bucket_idx = b;
                    }
                    auto &usage = data.engines[result.first->second];
                    alternative.engine_slots.emplace_back(result.first->second, usage.slot_count++);
                }
                data.alternatives[sel_var].push_back(std::move(alternative));
            }
        }
        // Preserve the selected_vars iteration order used by the original
        // StartPrecedencePropagator parent cache.
        for (const auto &pair : selected_vars[b])
        {
            const EClassId parent_cid = pair.first;
            const VarId selection_var = pair.second;
            const EClass &cls = bucket_egraphs[b].getEClass(parent_cid);
            const auto start_it = start_vars[b].find(parent_cid);
            for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
            {
                const ENodeId en_id = cls.enodes[en_idx];
                const ENode &enode = bucket_egraphs[b].getENode(en_id);
                const VarId start_var = start_it != start_vars[b].end() && en_idx < start_it->second.size()
                                            ? start_it->second[en_idx]
                                            : kInvalidVarId;
                const bool is_view = en_id.value < bucket_enode_infos[b].size() &&
                                     bucket_enode_infos[b][en_id.value].is_view;
                const PropagationState::StartPrecedenceParent parent_info{
                    parent_cid, en_idx, selection_var, start_var, is_view};
                for (EClassId child : enode.getChildren())
                {
                    const EClassId canonical_child = bucket_egraphs[b].findConst(child);
                    precedence.parents[canonical_child].push_back(parent_info);
                }
            }
        }
    }
    for (auto *lists : {&data.parents, &data.view_neighbors})
        for (auto &list : *lists)
        {
            std::sort(list.begin(), list.end());
            list.erase(std::unique(list.begin(), list.end()), list.end());
        }
    for (uint32_t i = 0; i < candidates.size(); ++i)
    {
        const auto &candidate = candidates[i];
        const auto it = cached_vars.find(candidate.base_eclass_id);
        if (it == cached_vars.end())
            continue;
        data.cache_candidates[it->second].push_back(i);
        data.candidates_by_space[candidate.mem_space].push_back(i);
        data.cache_budget_dirty.insert(candidate.mem_space);
    }
    for (auto &usage : data.engines)
    {
        usage.work.initialize(usage.slot_count);
        data.buckets[usage.bucket_idx].engine_bounds.insert(0.0);
    }
    data.initialized = true;
    for (VarId var_id = 0; var_id < numVars(); ++var_id)
        if (var_infos[var_id].type == VarType::SELECTED || var_infos[var_id].type == VarType::CACHED)
            updatePropagationContribution(var_id, true);
}

inline void SearchState::updatePropagationContribution(VarId var_id, bool add) const
{
    auto &data = propagation;
    if (!data.initialized)
        return;
    const VarInfo &info = var_infos[var_id];
    if (info.type == VarType::CACHED)
    {
        if (domains[var_id].isFixed() && domains[var_id].fixedValue() == 1)
            for (uint32_t index : data.cache_candidates[var_id])
            {
                const auto &candidate = candidates[index];
                auto &bytes = data.fixed_cache_bytes[candidate.mem_space];
                data.cache_budget_dirty.insert(candidate.mem_space);
                if (add)
                    bytes += candidate.size_bytes;
                else
                    bytes -= candidate.size_bytes;
            }
        return;
    }
    if (info.type != VarType::SELECTED && info.type != VarType::START)
        return;
    const VarId owner = data.owners[var_id];
    const int32_t index = selectedAlternative(owner);
    if (index < 0 || (info.type == VarType::START && info.enode_idx != static_cast<uint32_t>(index)))
        return;
    const auto &alternative = data.alternatives[owner][index];
    auto &bucket = data.buckets[info.bucket_idx];
    for (const auto &[engine_idx, slot] : alternative.engine_slots)
    {
        auto &usage = data.engines[engine_idx];
        if (info.type == VarType::SELECTED)
        {
            bucket.engine_bounds.erase(bucket.engine_bounds.find(usage.work.total()));
            usage.work.set(slot, add ? alternative.cost : 0.0);
            bucket.engine_bounds.insert(usage.work.total());
            if (alternative.start_var != kInvalidVarId)
            {
                if (add)
                    usage.active_starts.insert(alternative.start_var);
                else
                    usage.active_starts.erase(alternative.start_var);
            }
        }
        if (alternative.start_var != kInvalidVarId && domains[alternative.start_var].isFixed())
        {
            const int32_t start = domains[alternative.start_var].fixedValue();
            if (add)
                ++usage.fixed_starts[start];
            else if (--usage.fixed_starts.at(start) == 0)
                usage.fixed_starts.erase(start);
        }
    }
    if (info.type == VarType::SELECTED)
    {
        const float weight = info.bucket_idx < bucket_weights.size() ? bucket_weights[info.bucket_idx] : 1.0f;
        const double bound = bucket.engine_bounds.empty() ? 0.0 : *bucket.engine_bounds.rbegin();
        data.cost.set(info.bucket_idx, weight > 0.0f ? weight * bound : 0.0);
    }
}

inline float SearchState::costLowerBound() const
{
    ensurePropagationState();
    return static_cast<float>(propagation.cost.total());
}

} // namespace plan
