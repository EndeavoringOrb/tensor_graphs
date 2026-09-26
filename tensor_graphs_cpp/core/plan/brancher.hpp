#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/logging.hpp"
#include "core/plan/domain.hpp"
#include "core/plan/enode_info.hpp"
#include "core/plan/search_state.hpp"

namespace plan
{

struct BranchDecision
{
    std::vector<std::pair<VarId, Domain>> left_delta;
    std::vector<std::pair<VarId, Domain>> right_delta;
};

class Brancher
{
  public:
    virtual ~Brancher() = default;

    virtual bool chooseBranch(const SearchState &state, BranchDecision &out_decision) = 0;
};

class HeuristicBrancher : public Brancher
{
  private:
    static float getENodeScore(const SearchState &state, uint32_t bucket_idx, EClassId cid,
                               uint32_t enode_idx)
    {
        const EClass &cls = state.bucket_egraphs[bucket_idx].getEClass(cid);
        if (enode_idx >= cls.enodes.size())
            return TGConstants::INF;

        ENodeId enode_id = cls.enodes[enode_idx];
        if (enode_id.value >= state.bucket_enode_infos[bucket_idx].size())
            return 1.0f;

        const ENodeInfo &info = state.bucket_enode_infos[bucket_idx][enode_id.value];
        // Feasibility-first ordering, inspired by the old delegate: prefer
        // implementations with lower estimated memory pressure, then the
        // optimistic parallel DAG makespan.  Keep dp_cost as a fallback for
        // callers that construct ENodeInfo values without the new pass.
        float dag_cost = info.optimistic_dag_cost;
        if (dag_cost == TGConstants::INF)
            dag_cost = info.dp_cost;
        const float heuristic_cost = dag_cost < TGConstants::INF ? dag_cost : info.cost;
        if (info.dp_mem < TGConstants::INF)
            return info.dp_mem * 1.0e-6f + heuristic_cost * 1.0e-3f + info.cost;
        if (dag_cost < TGConstants::INF)
            return dag_cost;
        return info.cost;
    }

    static int32_t preferredENode(const SearchState &state, uint32_t bucket_idx, EClassId cid,
                                  const Domain &domain)
    {
        const EClass &cls = state.bucket_egraphs[bucket_idx].getEClass(cid);
        int32_t best_value = domain.getMin();
        float best_score = TGConstants::INF;
        bool found_compatible = false;

        for (uint32_t enode_idx = 0; enode_idx < cls.enodes.size(); ++enode_idx)
        {
            int32_t value = static_cast<int32_t>(enode_idx + 1);
            if (!domain.contains(value))
                continue;

            if (!candidateIsTopologicallyCompatible(state, bucket_idx, cid, enode_idx))
                continue;

            float score = getENodeScore(state, bucket_idx, cid, enode_idx);
            if (!found_compatible || score < best_score)
            {
                best_score = score;
                best_value = value;
                found_compatible = true;
            }
        }

        // Keep an actual branch value if all alternatives are incompatible;
        // propagation will report the contradiction and preserve completeness.
        return best_value;
    }

    static Domain makeFixedLike(const Domain &domain, int32_t value)
    {
        return Domain::makeFixed(value, domain.is_mask);
    }

    static Domain without(const Domain &domain, int32_t value)
    {
        Domain result = domain;
        result.remove(value);
        return result;
    }

    // Selected edges point from a child e-class to its parent. Following
    // fixed selections therefore tells us whether adding a candidate edge
    // would close a precedence cycle.
    static bool hasSelectedPath(const SearchState &state, uint32_t bucket_idx,
                                EClassId from, EClassId target)
    {
        std::vector<EClassId> frontier = {from};
        std::unordered_set<EClassId> visited;
        visited.insert(from);

        for (size_t i = 0; i < frontier.size(); ++i)
        {
            EClassId cid = frontier[i];
            if (cid == target)
                return true;

            auto sel_it = state.selected_vars[bucket_idx].find(cid);
            if (sel_it == state.selected_vars[bucket_idx].end())
                continue;

            const Domain &selection = state.domains[sel_it->second];
            if (!selection.isFixed() || selection.fixedValue() <= 0)
                continue;

            uint32_t enode_idx = static_cast<uint32_t>(selection.fixedValue() - 1);
            const EClass &cls = state.bucket_egraphs[bucket_idx].getEClass(cid);
            if (enode_idx >= cls.enodes.size())
                continue;

            const ENode &enode = state.bucket_egraphs[bucket_idx].getENode(cls.enodes[enode_idx]);
            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = state.bucket_egraphs[bucket_idx].findConst(child);
                if (visited.insert(canon_child).second)
                    frontier.push_back(canon_child);
            }
        }
        return false;
    }

    static bool candidateIsTopologicallyCompatible(const SearchState &state, uint32_t bucket_idx,
                                                   EClassId cid, uint32_t enode_idx)
    {
        const EClass &cls = state.bucket_egraphs[bucket_idx].getEClass(cid);
        if (enode_idx >= cls.enodes.size())
            return false;

        const ENode &enode = state.bucket_egraphs[bucket_idx].getENode(cls.enodes[enode_idx]);
        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = state.bucket_egraphs[bucket_idx].findConst(child);
            if (canon_child == cid || hasSelectedPath(state, bucket_idx, canon_child, cid))
                return false;

            auto child_it = state.selected_vars[bucket_idx].find(canon_child);
            if (child_it == state.selected_vars[bucket_idx].end())
                return false;

            // This candidate will force every child to be selected.
            if (state.domains[child_it->second].getMax() <= 0)
                return false;
        }
        return true;
    }

    static bool setBinaryDecision(const Domain &domain, int32_t preferred,
                                  BranchDecision &out_decision, VarId var_id)
    {
        if (domain.isFixed() || domain.isEmpty() || !domain.contains(preferred))
            return false;

        Domain right = without(domain, preferred);
        if (right.isEmpty())
            return false;

        out_decision.left_delta.push_back({var_id, makeFixedLike(domain, preferred)});
        out_decision.right_delta.push_back({var_id, right});
        return true;
    }

    bool chooseSelection(const SearchState &state, BranchDecision &out_decision,
                         bool required_only) const
    {
        VarId best_var = kInvalidVarId;
        uint32_t best_bucket = 0;
        EClassId best_cid;
        int32_t best_size = std::numeric_limits<int32_t>::max();

        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            const auto &cids = (b < state.reachable_cids.size()) ? state.reachable_cids[b]
                                                                  : std::vector<EClassId>{};
            for (EClassId cid : cids)
            {
                auto sel_it = state.selected_vars[b].find(cid);
                if (sel_it == state.selected_vars[b].end())
                    continue;

                VarId var_id = sel_it->second;
                const Domain &domain = state.domains[var_id];
                if (domain.isFixed() || domain.isEmpty())
                    continue;

                const bool required = !domain.contains(0);
                if (required != required_only)
                    continue;

                if (domain.size() < best_size)
                {
                    best_var = var_id;
                    best_bucket = b;
                    best_cid = cid;
                    best_size = domain.size();
                }
            }
        }

        if (best_var == kInvalidVarId)
            return false;

        const Domain &domain = state.domains[best_var];
        int32_t preferred = required_only ? preferredENode(state, best_bucket, best_cid, domain) : 0;
        return setBinaryDecision(domain, preferred, out_decision, best_var);
    }

    bool chooseCache(const SearchState &state, BranchDecision &out_decision) const
    {
        for (const CacheCandidate &candidate : state.candidates)
        {
            auto it = state.cached_vars.find(candidate.base_eclass_id);
            if (it == state.cached_vars.end())
                continue;

            VarId var_id = it->second;
            const Domain &domain = state.domains[var_id];
            if (domain.isFixed() || domain.isEmpty())
                continue;

            return setBinaryDecision(domain, 0, out_decision, var_id);
        }
        return false;
    }

    bool chooseSchedule(const SearchState &state, BranchDecision &out_decision) const
    {
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            std::unordered_set<EClassId> active;
            std::unordered_map<EClassId, int> in_degree;
            std::unordered_map<EClassId, std::vector<EClassId>> parents;

            const auto &cids = (b < state.reachable_cids.size()) ? state.reachable_cids[b]
                                                                  : std::vector<EClassId>{};
            for (EClassId cid : cids)
            {
                auto sel_it = state.selected_vars[b].find(cid);
                if (sel_it == state.selected_vars[b].end())
                    continue;

                const Domain &selection = state.domains[sel_it->second];
                if (!selection.isFixed() || selection.fixedValue() <= 0)
                    continue;

                active.insert(cid);
                in_degree[cid] = 0;
            }

            for (EClassId cid : active)
            {
                const Domain &selection = state.domains[state.selected_vars[b].at(cid)];
                uint32_t enode_idx = static_cast<uint32_t>(selection.fixedValue() - 1);
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (enode_idx >= cls.enodes.size())
                    continue;

                std::unordered_set<EClassId> unique_children;
                const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[enode_idx]);
                for (EClassId child : enode.getChildren())
                {
                    EClassId canon_child = state.bucket_egraphs[b].findConst(child);
                    if (active.count(canon_child) != 0 && unique_children.insert(canon_child).second)
                    {
                        ++in_degree[cid];
                        parents[canon_child].push_back(cid);
                    }
                }
            }

            std::vector<EClassId> queue;
            for (const auto &entry : in_degree)
            {
                if (entry.second == 0)
                    queue.push_back(entry.first);
            }

            std::vector<EClassId> topo_order;
            size_t queue_head = 0;
            while (queue_head < queue.size())
            {
                EClassId child = queue[queue_head++];
                topo_order.push_back(child);
                for (EClassId parent : parents[child])
                {
                    if (--in_degree[parent] == 0)
                        queue.push_back(parent);
                }
            }

            if (topo_order.size() != active.size())
                continue;

            // Assign starts in dependency order: children first, then their
            // consumers. This makes the preferred branch a valid topological
            // scheduling attempt instead of a split chosen by domain size.
            for (EClassId cid : topo_order)
            {
                const Domain &selection = state.domains[state.selected_vars[b].at(cid)];
                uint32_t enode_idx = static_cast<uint32_t>(selection.fixedValue() - 1);
                auto start_it = state.start_vars[b].find(cid);
                if (start_it == state.start_vars[b].end() || enode_idx >= start_it->second.size())
                    continue;

                VarId start_var = start_it->second[enode_idx];
                const Domain &start_domain = state.domains[start_var];
                if (!start_domain.isFixed())
                    return setBinaryDecision(start_domain, start_domain.getMin(), out_decision, start_var);
            }

            for (EClassId cid : topo_order)
            {
                auto offset_it = state.offset_vars[b].find(cid);
                if (offset_it != state.offset_vars[b].end())
                {
                    VarId offset_var = offset_it->second;
                    const Domain &offset_domain = state.domains[offset_var];
                    if (!offset_domain.isFixed())
                        return setBinaryDecision(offset_domain, offset_domain.getMin(), out_decision, offset_var);
                }
            }
        }
        return false;
    }

  public:
    bool chooseBranch(const SearchState &state, BranchDecision &out_decision) override
    {
        out_decision.left_delta.clear();
        out_decision.right_delta.clear();

        // First satisfy selections forced by the currently selected DAG.
        if (chooseSelection(state, out_decision, true))
            return true;

        if (chooseCache(state, out_decision))
            return true;

        // Schedule and allocate one active operation at a time. This keeps
        // start/offset variables in the same hyperbox search instead of
        // manufacturing a complete heuristic assignment in one delta.
        if (chooseSchedule(state, out_decision))
            return true;

        // Only after the currently selected DAG has a cache/schedule attempt
        // do we branch optional support. The branch remains complete, but a
        // promising selected DAG can now reach a full feasible leaf without
        // first fixing every potentially reachable e-class in the e-graph.
        if (chooseSelection(state, out_decision, false))
            return true;

        LOG(DEBUG) << "[HeuristicBrancher] No unfixed relevant variable remains; leaf reached.";
        return false;
    }
};

} // namespace plan
