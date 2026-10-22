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
            const uint32_t b = state.var_infos[changed].bucket_idx;
            bool consistent = true;
            if (completePlanCheck(state, b, consistent))
                return consistent;
            // The first event of a bucket builds its tree from the current
            // domains, like updateSelectionReachability would; building every
            // unbuilt bucket at that point is equally sound (domains only
            // shrink, so a class unreachable now stays unreachable, and later
            // changes still arrive as their own events) and runs in parallel.
            if (!state.selectionReachabilityInitialized(b))
            {
                const bool built = initializeBuckets(state, /*update_initialized=*/false);
                if (!state.selectionReachabilityInitialized(b))
                    Error::throw_err("[SelectionReachability] bucket " + std::to_string(b) +
                                     " has a selection event but no root selection variable");
                return built;
            }
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
            return initializeBuckets(state, /*update_initialized=*/true);
        }
        return true;
    }

  private:
    // Builds every unbuilt bucket's tree in parallel, then removes the
    // unreachable selections bucket by bucket. With update_initialized, an
    // already built bucket is refreshed from its root variable instead.
    bool initializeBuckets(SearchState &state, bool update_initialized)
    {
        std::vector<VarId> root_vars(state.buckets.size(), kInvalidVarId);
        std::vector<uint8_t> was_initialized(state.buckets.size(), 0);
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            if (b >= state.bucket_root_ids.size())
                continue;
            auto it = state.selected_vars[b].find(state.bucket_root_ids[b]);
            if (it == state.selected_vars[b].end())
                continue;
            root_vars[b] = it->second;
            was_initialized[b] = state.selectionReachabilityInitialized(b);
        }
        std::vector<std::vector<VarId>> built_unreachable;
        state.initializeSelectionReachability(root_vars, is_dag_, built_unreachable);
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            const VarId root_var = root_vars[b];
            if (root_var == kInvalidVarId || (was_initialized[b] && !update_initialized))
                continue;
            std::vector<VarId> unreachable;
            if (was_initialized[b])
            {
                auto &forced_parents = state.pending_parent_selections;
                state.updateSelectionReachability(root_var, unreachable, is_dag_, &forced_parents);
            }
            else
            {
                unreachable = std::move(built_unreachable[b]);
            }
            for (VarId sel_v : unreachable)
            {
                const Domain &sel_dom = state.domains[sel_v];
                if (!sel_dom.contains(0))
                    return reportConflict(state, root_var, sel_v, "initial required class unreachable");
                if (!sel_dom.isFixed())
                    state.setDomain(sel_v, Domain::makeFixed(0, sel_dom.is_mask));
            }
        }
        return true;
    }
};

} // namespace plan
