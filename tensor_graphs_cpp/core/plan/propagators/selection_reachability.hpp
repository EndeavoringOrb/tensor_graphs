// tensor_graphs_cpp/core/plan/propagators/selection_reachability.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// SelectionReachabilityPropagator: see docs/core/propagators.md.
class SelectionReachabilityPropagator : public Propagator
{
    bool is_dag_ = false;
    std::string conflict_reason_;
    uint64_t conflict_count_ = 0;
    std::unordered_map<VarId, uint64_t> conflicts_by_victim_;

    bool reportConflict(const SearchState &state, VarId changed, VarId victim,
                        const std::string &kind)
    {
        ++conflict_count_;
        const uint64_t victim_count = ++conflicts_by_victim_[victim];
        const auto describe = [&](VarId var_id) {
            return var_id == kInvalidVarId ? std::string("initial") :
                state.var_infos.at(var_id).name + "=" + state.domains.at(var_id).toString();
        };
        conflict_reason_ = kind + ": trigger=" + describe(changed) +
                           ", victim=" + describe(victim);
        // Sample exponentially after the first few failures. Diagnostics must
        // remain affordable even when millions of similar branches fail.
        if (conflict_count_ > 8 && (conflict_count_ & (conflict_count_ - 1)) != 0)
            return false;

        const auto &reachability = state.selectionReachability(state.var_infos.at(victim).bucket_idx);
        const auto &node = reachability.nodes.at(reachability.node_indices.at(victim));
        // Independently audit reachability using current domains rather than
        // incremental edge flags, so stale support bookkeeping is visible.
        std::vector<bool> reachable(reachability.nodes.size(), false);
        std::vector<uint32_t> queue{reachability.root};
        reachable.at(reachability.root) = true;
        for (size_t head = 0; head < queue.size(); ++head)
        {
            const auto &parent = reachability.nodes.at(queue[head]);
            const auto &domain = state.domains.at(parent.var_id);
            for (size_t en_idx = 0; en_idx < parent.enode_edges.size(); ++en_idx)
            {
                if (!domain.contains(static_cast<int32_t>(en_idx + 1)))
                    continue;
                for (uint32_t edge_id : parent.enode_edges[en_idx])
                {
                    const uint32_t to = reachability.edges.at(edge_id).to;
                    if (!reachable.at(to))
                    {
                        reachable[to] = true;
                        queue.push_back(to);
                    }
                }
            }
        }
        LOG(INFO) << "[Reachability diagnostic] occurrence=" << conflict_count_
                  << " victim-occurrences=" << victim_count
                  << " distinct-victims=" << conflicts_by_victim_.size()
                  << " " << conflict_reason_
                  << " mode=" << (reachability.is_dag ? "dag" : "general")
                  << " audit-reachable=" << reachable.at(reachability.node_indices.at(victim))
                  << " incoming=" << node.incoming.size()
                  << " parent-support-candidates=" << state.pending_parent_selections.size();
        size_t printed = 0;
        for (uint32_t edge_id : node.incoming)
        {
            if (printed++ >= 24)
                break;
            const auto &edge = reachability.edges.at(edge_id);
            const auto &parent = reachability.nodes.at(edge.from);
            const int32_t value = static_cast<int32_t>(edge.en_idx + 1);
            LOG(INFO) << "[Reachability support] parent=" << describe(parent.var_id)
                      << " value=" << value << " edge-active=" << edge.active
                      << " domain-support=" << state.domains.at(parent.var_id).contains(value)
                      << " audit-parent-reachable=" << reachable.at(edge.from)
                      << " processed-domain=" << parent.selection.toString()
                      << " level=" << parent.level;
        }
        std::vector<std::pair<VarId, uint64_t>> ranked(conflicts_by_victim_.begin(), conflicts_by_victim_.end());
        std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        });
        for (size_t idx = 0; idx < std::min<size_t>(5, ranked.size()); ++idx)
            LOG(INFO) << "[Reachability hotspot] victim=" << state.var_infos.at(ranked[idx].first).name
                      << " conflicts=" << ranked[idx].second;
        return false;
    }

    // While the root heuristic validates a complete plan (every selection is
    // fixed before propagation), one search from the root over the selected
    // alternatives per selection revision decides reachability: a selected
    // class it misses is a conflict, nothing is left to prune. The
    // incremental structure is not touched, so it stays at the state the
    // validation restores. Returns false (no shortcut) when a selection of
    // the bucket is still open.
    std::vector<uint64_t> verified_revision_;
    bool completePlanCheck(SearchState &state, uint32_t b, bool &consistent)
    {
        if (!state.completing_plan)
            return false;
        if (verified_revision_.size() < state.buckets.size())
            verified_revision_.resize(state.buckets.size(), UINT64_MAX);
        if (verified_revision_[b] == state.getSelectionRevision())
        {
            consistent = true;
            return true;
        }
        for (const auto &pair : state.selected_vars[b])
            if (!state.domains[pair.second].isFixed())
                return false;
        const EGraph &egraph = state.bucket_egraphs[b];
        std::unordered_set<EClassId> reached;
        std::vector<EClassId> frontier;
        const EClassId root = egraph.findConst(state.bucket_root_ids[b]);
        reached.insert(root);
        frontier.push_back(root);
        for (size_t head = 0; head < frontier.size(); ++head)
        {
            const auto it = state.selected_vars[b].find(frontier[head]);
            if (it == state.selected_vars[b].end())
                continue;
            const int32_t value = state.domains[it->second].fixedValue();
            if (value <= 0)
                continue;
            const EClass &cls = egraph.getEClass(frontier[head]);
            for (EClassId ch : egraph.getENode(cls.enodes[static_cast<size_t>(value - 1)]).getChildren())
            {
                const EClassId canon = egraph.findConst(ch);
                if (reached.insert(canon).second)
                    frontier.push_back(canon);
            }
        }
        for (const auto &pair : state.selected_vars[b])
            if (state.domains[pair.second].fixedValue() > 0 && !reached.count(pair.first))
            {
                conflict_reason_ = "complete plan selects a class unreachable from the root: " +
                                   state.var_infos[pair.second].name;
                consistent = false;
                return true;
            }
        consistent = true;
        verified_revision_[b] = state.getSelectionRevision();
        return true;
    }

  public:
    explicit SelectionReachabilityPropagator(bool is_dag = false) : is_dag_(is_dag) {}

    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED); }

    std::string name() const override
    {
        return "SelectionReachabilityPropagator";
    }

    std::string conflictReason() const override { return conflict_reason_; }

    bool isDag() const
    {
        return is_dag_;
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &) override
    {
        conflict_reason_.clear();
        state.pending_parent_selections.clear();
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;
            bool consistent = true;
            if (completePlanCheck(state, state.var_infos[changed].bucket_idx, consistent))
                return consistent;
            std::vector<VarId> unreachable;
            auto &forced_parents = state.pending_parent_selections;
            state.updateSelectionReachability(changed, unreachable, is_dag_, &forced_parents);
            for (VarId sel_v : unreachable)
            {
                const Domain &sel_dom = state.domains[sel_v];
                if (!sel_dom.contains(0))
                    return reportConflict(state, changed, sel_v, "required class became unreachable");
                if (!sel_dom.isFixed())
                    state.setDomain(sel_v, Domain::makeFixed(0, sel_dom.is_mask));
            }
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                if (b >= state.bucket_root_ids.size())
                    continue;
                EClassId root_cid = state.bucket_root_ids[b];
                auto it = state.selected_vars[b].find(root_cid);
                if (it == state.selected_vars[b].end())
                    continue;
                VarId root_var = it->second;
                std::vector<VarId> unreachable;
                auto &forced_parents = state.pending_parent_selections;
                state.updateSelectionReachability(root_var, unreachable, is_dag_, &forced_parents);
                for (VarId sel_v : unreachable)
                {
                    const Domain &sel_dom = state.domains[sel_v];
                    if (!sel_dom.contains(0))
                        return reportConflict(state, root_var, sel_v, "initial required class unreachable");
                    if (!sel_dom.isFixed())
                        state.setDomain(sel_v, Domain::makeFixed(0, sel_dom.is_mask));
                }
            }
        }
        return true;
    }
};

} // namespace plan
