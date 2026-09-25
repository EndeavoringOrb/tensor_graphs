// tensor_graphs_cpp/core/plan/search_engine.hpp
#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/constants.hpp"
#include "core/logging.hpp"
#include "core/plan/brancher.hpp"
#include "core/plan/domain.hpp"
#include "core/plan/propagator.hpp"
#include "core/plan/search_node.hpp"
#include "core/plan/search_state.hpp"
#include "core/plan/selector.hpp"
#include "core/timer.hpp"

namespace plan
{

struct ExtractionResult
{
    std::unordered_map<EClassId, uint32_t> selection_map;
    std::vector<EClassId> order;
    std::vector<ParallelBuffer> buffers;
    std::unordered_map<EClassId, BufferId> eclass_to_buf;
    float cost = TGConstants::INF;
    std::unordered_map<EClassId, float> eclass_to_cost;
};

class SearchEngine
{
  public:
    SearchState state;
    std::shared_ptr<Selector> selector;
    std::shared_ptr<Brancher> brancher;
    std::vector<std::unique_ptr<Propagator>> propagators;

    std::vector<std::shared_ptr<SearchNode>> all_nodes;
    uint32_t current_node_id = UINT32_MAX;

    float incumbent_best_cost = TGConstants::INF;
    std::vector<ExtractionResult> incumbent_extractions;
    std::unordered_set<BaseEClassId> incumbent_cached_nodes;

    SearchEngine(SearchState state, std::shared_ptr<Selector> selector = nullptr,
                 std::shared_ptr<Brancher> brancher = nullptr)
        : state(std::move(state)), selector(selector ? selector : std::make_shared<PriorityQueueSelector>()),
          brancher(brancher ? brancher : std::make_shared<HeuristicBrancher>())
    {
    }

    void addPropagator(std::unique_ptr<Propagator> prop)
    {
        propagators.push_back(std::move(prop));
    }

    bool restoreNode(const std::shared_ptr<SearchNode> &target_node)
    {
        if (current_node_id == target_node->id)
            return true;

        // Path from current_node up to root
        std::vector<uint32_t> current_path;
        uint32_t curr = current_node_id;
        while (curr != UINT32_MAX)
        {
            current_path.push_back(curr);
            curr = all_nodes[curr]->parent_id;
        }

        // Path from target_node up to root
        std::vector<uint32_t> target_path;
        uint32_t tgt = target_node->id;
        while (tgt != UINT32_MAX)
        {
            target_path.push_back(tgt);
            tgt = all_nodes[tgt]->parent_id;
        }

        // Find Lowest Common Ancestor (LCA)
        int i = static_cast<int>(current_path.size()) - 1;
        int j = static_cast<int>(target_path.size()) - 1;
        uint32_t lca = UINT32_MAX;
        while (i >= 0 && j >= 0 && current_path[i] == target_path[j])
        {
            lca = current_path[i];
            i--;
            j--;
        }

        // Backtrack to LCA's trail marker
        size_t lca_marker = (lca != UINT32_MAX) ? all_nodes[lca]->trail_marker : 0;
        state.backtrackTo(lca_marker);

        // Play decisions and propagate from child of LCA down to target_node
        for (int k = j; k >= 0; --k)
        {
            uint32_t nid = target_path[k];
            for (const auto &p : all_nodes[nid]->delta)
            {
                state.setDomain(p.first, p.second);
            }
            float lb = 0.0f;
            if (!runPropagators(lb))
            {
                current_node_id = (k < static_cast<int>(target_path.size()) - 1) ? target_path[k + 1] : lca;
                return false;
            }
            all_nodes[nid]->lower_bound = lb;
            all_nodes[nid]->trail_marker = state.getTrailMarker();
        }

        current_node_id = target_node->id;
        return true;
    }

    bool runPropagators(float &out_lower_bound)
    {
        bool changed = true;
        int max_iters = 3;
        int iters = 0;
        while (changed && iters++ < max_iters)
        {
            changed = false;
            size_t marker_before = state.getTrailMarker();
            for (auto &prop : propagators)
            {
                if (!prop->propagate(state))
                    return false;
            }
            if (state.getTrailMarker() > marker_before)
                changed = true;
        }

        out_lower_bound = 0.0f;
        for (auto &prop : propagators)
        {
            out_lower_bound = std::max(out_lower_bound, prop->computeLowerBound(state));
        }
        return true;
    }

    float evaluateMakespan(const SearchState &st, uint32_t b) const
    {
        // Compute makespan for bucket b
        std::unordered_map<Engine, float> engine_finish;
        std::vector<std::pair<int32_t, EClassId>> sorted_ops;

        const auto &cids = (b < st.reachable_cids.size() && !st.reachable_cids[b].empty())
                               ? st.reachable_cids[b]
                               : std::vector<EClassId>{};
        for (EClassId cid : cids)
        {
            auto it = st.selected_vars[b].find(cid);
            if (it == st.selected_vars[b].end())
                continue;
            VarId sel_v = it->second;
            if (st.domains[sel_v].isFixed() && st.domains[sel_v].fixedValue() > 0)
            {
                uint32_t en_idx = static_cast<uint32_t>(st.domains[sel_v].fixedValue() - 1);
                VarId st_v = st.start_vars[b].at(cid)[en_idx];
                int32_t st_val = st.domains[st_v].fixedValue();
                sorted_ops.push_back({st_val, cid});
            }
        }
        std::sort(sorted_ops.begin(), sorted_ops.end());

        for (const auto &item : sorted_ops)
        {
            EClassId cid = item.second;
            uint32_t en_idx = static_cast<uint32_t>(st.domains[st.selected_vars[b].at(cid)].fixedValue() - 1);
            const EClass &cls = st.bucket_egraphs[b].getEClass(cid);
            ENodeId en_id = cls.enodes[en_idx];
            const ENode &enode = st.bucket_egraphs[b].getENode(en_id);
            float cost = (en_id.value < st.bucket_enode_infos[b].size()) ? st.bucket_enode_infos[b][en_id.value].cost : 0.0f;
            if (cost == TGConstants::INF)
                cost = 1.0f;

            for (const Engine &eng : enode.getEngines())
            {
                engine_finish[eng] += cost;
            }
        }

        float max_finish = 0.0f;
        for (const auto &pair : engine_finish)
            max_finish = std::max(max_finish, pair.second);
        return max_finish;
    }

    std::vector<ExtractionResult> extractSolution(const SearchState &st) const
    {
        std::vector<ExtractionResult> results(st.buckets.size());
        uint32_t buf_counter = 1;

        for (uint32_t b = 0; b < st.buckets.size(); ++b)
        {
            ExtractionResult &res = results[b];
            std::vector<std::pair<int32_t, EClassId>> sorted_ops;

            const auto &cids = (b < st.reachable_cids.size() && !st.reachable_cids[b].empty())
                                   ? st.reachable_cids[b]
                                   : std::vector<EClassId>{};
            for (EClassId cid : cids)
            {
                auto it = st.selected_vars[b].find(cid);
                if (it == st.selected_vars[b].end())
                    continue;
                VarId sel_v = it->second;
                if (st.domains[sel_v].isFixed() && st.domains[sel_v].fixedValue() > 0)
                {
                    uint32_t en_idx = static_cast<uint32_t>(st.domains[sel_v].fixedValue() - 1);
                    res.selection_map[cid] = en_idx;
                    VarId st_v = st.start_vars[b].at(cid)[en_idx];
                    int32_t st_val = st.domains[st_v].fixedValue();
                    sorted_ops.push_back({st_val, cid});
                }
            }
            std::sort(sorted_ops.begin(), sorted_ops.end());
            for (const auto &item : sorted_ops)
                res.order.push_back(item.second);

            // 1. Add preallocated buffers
            std::unordered_map<BaseEClassId, BufferId> base_to_prealloc;
            for (const auto &pair : st.preallocated_buffers)
            {
                res.buffers.push_back(pair.second);
                base_to_prealloc[pair.first] = pair.second.id;
                buf_counter = std::max(buf_counter, pair.second.id.value + 1);
            }

            // Helper to get or allocate buffer for a root non-view eclass
            auto get_or_allocate_buffer = [&](EClassId target_cid) -> BufferId {
                auto it = res.eclass_to_buf.find(target_cid);
                if (it != res.eclass_to_buf.end())
                    return it->second;

                const EClass &cls = st.bucket_egraphs[b].getEClass(target_cid);
                if (cls.base_eclass_id != BaseEClassId{} && base_to_prealloc.count(cls.base_eclass_id))
                {
                    BufferId bid = base_to_prealloc.at(cls.base_eclass_id);
                    res.eclass_to_buf[target_cid] = bid;
                    return bid;
                }

                if (cls.mem_space.type == HandleType::STORAGE)
                {
                    ParallelBuffer pb;
                    pb.id = BufferId{buf_counter++};
                    pb.mem_space = cls.mem_space;
                    pb.size = getSizeBytes(cls.shape, cls.dtype);
                    pb.offset = -1;
                    res.buffers.push_back(pb);
                    res.eclass_to_buf[target_cid] = pb.id;
                    return pb.id;
                }

                int64_t byte_offset = 0;
                auto off_it = st.offset_vars[b].find(target_cid);
                if (off_it != st.offset_vars[b].end())
                {
                    VarId off_v = off_it->second;
                    int32_t page_off = st.domains[off_v].fixedValue();
                    byte_offset = static_cast<int64_t>(page_off) * st.getPageAlignment(cls.mem_space);
                }

                ParallelBuffer pb;
                pb.id = BufferId{buf_counter++};
                pb.mem_space = cls.mem_space;
                pb.size = getSizeBytes(cls.shape, cls.dtype);
                pb.offset = byte_offset;
                res.buffers.push_back(pb);
                res.eclass_to_buf[target_cid] = pb.id;
                return pb.id;
            };

            // 2. Map buffers for all ordered eclasses
            for (EClassId cid : res.order)
            {
                EClassId base_cid = resolve_view_alias(cid, st.bucket_egraphs[b], res.selection_map, st.bucket_enode_infos[b]);
                BufferId bid = get_or_allocate_buffer(base_cid);
                res.eclass_to_buf[cid] = bid;
            }

            for (const auto &pair : res.selection_map)
            {
                EClassId cid = pair.first;
                uint32_t en_idx = pair.second;
                ENodeId en_id = st.bucket_egraphs[b].getEClass(cid).enodes[en_idx];
                float en_cost = (en_id.value < st.bucket_enode_infos[b].size())
                                    ? st.bucket_enode_infos[b][en_id.value].cost
                                    : 0.0f;
                res.eclass_to_cost[cid] = en_cost;
            }

            res.cost = evaluateMakespan(st, b);
        }
        return results;
    }

    bool solve(float timeout_seconds = 0.0f)
    {
        TimeoutChecker timer(timeout_seconds);

        LOG(DEBUG) << "[SearchEngine] Starting solve: num_vars=" << state.numVars()
                   << ", candidates=" << state.candidates.size()
                   << ", buckets=" << state.buckets.size()
                   << ", timeout=" << timeout_seconds << "s";

        // 0. Initial structural reachability from root across all buckets
        state.reachable_cids.clear();
        state.reachable_cids.resize(state.buckets.size());
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            EClassId root_id = state.bucket_root_ids[b];
            std::unordered_set<EClassId> reachable;
            std::vector<EClassId> frontier = {root_id};
            reachable.insert(root_id);

            while (!frontier.empty())
            {
                EClassId curr_cid = frontier.back();
                frontier.pop_back();

                const EClass &cls = state.bucket_egraphs[b].getEClass(curr_cid);
                for (ENodeId en_id : cls.enodes)
                {
                    const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                    for (EClassId ch : enode.getChildren())
                    {
                        EClassId canon_ch = state.bucket_egraphs[b].findConst(ch);
                        if (reachable.insert(canon_ch).second)
                        {
                            frontier.push_back(canon_ch);
                        }
                    }
                }
            }

            for (EClassId cid : reachable)
            {
                state.reachable_cids[b].push_back(cid);
            }

            for (const auto &pair : state.selected_vars[b])
            {
                EClassId cid = pair.first;
                if (reachable.find(cid) == reachable.end())
                {
                    VarId v = pair.second;
                    state.domains[v] = Domain::makeFixed(0);

                    auto off_it = state.offset_vars[b].find(cid);
                    if (off_it != state.offset_vars[b].end())
                    {
                        state.domains[off_it->second] = Domain::makeFixed(0);
                    }
                    auto st_it = state.start_vars[b].find(cid);
                    if (st_it != state.start_vars[b].end())
                    {
                        for (VarId st_v : st_it->second)
                        {
                            state.domains[st_v] = Domain::makeFixed(0);
                        }
                    }
                }
            }
        }

        // 1. Root node
        auto root_node = std::make_shared<SearchNode>(0, UINT32_MAX, std::vector<std::pair<VarId, Domain>>{}, 0.0f,
                                                      0.0f, 0);
        all_nodes.push_back(root_node);
        current_node_id = 0;

        float root_lb = 0.0f;
        if (!runPropagators(root_lb))
        {
            LOG(WARNING) << "[SearchEngine] Root state contradicted during initial propagation.";
            return false;
        }

        LOG(DEBUG) << "[SearchEngine] Initial root propagation succeeded: root LB=" << root_lb;

        root_node->lower_bound = root_lb;
        root_node->priority = root_lb;
        root_node->trail_marker = state.getTrailMarker();
        selector->push(root_node);

        uint32_t iterations = 0;
        while (!selector->empty())
        {
            if (timer.is_expired() && incumbent_best_cost < TGConstants::INF)
            {
                LOG(INFO) << "[SearchEngine] Timeout reached with feasible incumbent best cost: "
                          << incumbent_best_cost << " at iteration " << iterations;
                break;
            }

            if (timeout_seconds <= 0.0f && incumbent_best_cost < TGConstants::INF)
            {
                LOG(INFO) << "[SearchEngine] Greedy solution found with cost: "
                          << incumbent_best_cost << " at iteration " << iterations
                          << " (min compile time = 0); completing search.";
                break;
            }

            auto node = selector->pop();
            if (!node || node->lower_bound >= incumbent_best_cost)
                continue;

            iterations++;
            if (iterations == 1 || iterations % 50 == 0 || iterations >= 1345)
            {
                LOG(DEBUG) << "[SearchEngine] Iter " << iterations
                           << " | Queue: " << selector->size()
                           << " | Depth: " << node->depth
                           << " | Node LB: " << node->lower_bound
                           << " | Best: " << (incumbent_best_cost < TGConstants::INF ? std::to_string(incumbent_best_cost) : "inf");
            }

            if (!restoreNode(node))
            {
                if (iterations <= 20 || iterations % 50 == 0)
                {
                    LOG(DEBUG) << "[SearchEngine] Iter " << iterations << ": backtrack due to propagation conflict";
                }
                continue;
            }

            if (node->lower_bound >= incumbent_best_cost)
                continue;

            // Check if branching is needed or all variables are fixed
            BranchDecision decision;
            bool can_branch = brancher->chooseBranch(state, decision);

            if (!can_branch)
            {
                // Leaf reached: evaluate complete plan
                float total_cost = 0.0f;
                for (uint32_t b = 0; b < state.buckets.size(); ++b)
                {
                    float w = (b < state.bucket_weights.size()) ? state.bucket_weights[b] : 1.0f;
                    total_cost += w * evaluateMakespan(state, b);
                }

                LOG(DEBUG) << "[SearchEngine] Iter " << iterations << ": Leaf reached at depth " << node->depth
                           << ", evaluated total cost=" << total_cost;

                if (total_cost < incumbent_best_cost)
                {
                    incumbent_best_cost = total_cost;
                    incumbent_extractions = extractSolution(state);

                    incumbent_cached_nodes.clear();
                    for (const auto &pair : state.cached_vars)
                    {
                        if (state.domains[pair.second].isFixed() && state.domains[pair.second].fixedValue() == 1)
                        {
                            incumbent_cached_nodes.insert(pair.first);
                        }
                    }

                    LOG(INFO) << "[SearchEngine] New best cost: " << incumbent_best_cost
                              << " at iteration " << iterations;
                    for (auto &prop : propagators)
                    {
                        auto *clb = dynamic_cast<CostLowerBoundPropagator *>(prop.get());
                        if (clb)
                            clb->setBestCost(incumbent_best_cost);
                    }
                }
                continue;
            }

            if (iterations <= 25 || iterations % 50 == 0 || iterations >= 1345)
            {
                if (decision.left_delta.size() == 1)
                {
                    VarId branch_var = decision.left_delta[0].first;
                    LOG(DEBUG) << "[SearchEngine] Iter " << iterations << ": branching on var " << branch_var
                               << " (" << state.var_infos[branch_var].name << ") [dom: " << state.domains[branch_var].toString()
                               << "] -> Left: " << decision.left_delta[0].second.toString()
                               << ", Right: " << (decision.right_delta.empty() ? "{}" : decision.right_delta[0].second.toString());
                }
                else
                {
                    LOG(DEBUG) << "[SearchEngine] Iter " << iterations << ": bulk assigning "
                               << decision.left_delta.size() << " variables (start times & memory offsets)";
                }
            }

            // 1. Create Right Child (lazy alternative, pushed to queue without upfront propagation)
            if (!decision.right_delta.empty())
            {
                uint32_t right_id = static_cast<uint32_t>(all_nodes.size());
                float right_lb = node->lower_bound;
                float right_prio = right_lb + 1.0f;
                auto right_node = std::make_shared<SearchNode>(
                    right_id, node->id, decision.right_delta, right_lb,
                    right_prio, node->depth + 1);
                all_nodes.push_back(right_node);
                selector->push(right_node);
            }

            // 2. Create Left Child (dive immediately in place!)
            for (const auto &p : decision.left_delta)
            {
                state.setDomain(p.first, p.second);
            }
            float left_lb = 0.0f;
            bool left_ok = runPropagators(left_lb);
            if (left_ok && left_lb < incumbent_best_cost)
            {
                uint32_t left_id = static_cast<uint32_t>(all_nodes.size());
                float left_prio = left_lb - (node->depth + 1) * 0.01f; // Dive bias
                auto left_node = std::make_shared<SearchNode>(
                    left_id, node->id, decision.left_delta, left_lb,
                    left_prio, node->depth + 1, state.getTrailMarker());
                all_nodes.push_back(left_node);
                current_node_id = left_id; // Current state stays at left_node!
                selector->push(left_node);
            }
            else
            {
                // Left branch failed, backtrack in place to parent node
                state.backtrackTo(node->trail_marker);
                current_node_id = node->id;
            }
        }

        LOG(INFO) << "[SearchEngine] Search finished after " << iterations << " iterations ("
                  << all_nodes.size() << " total nodes generated). Best cost: "
                  << (incumbent_best_cost < TGConstants::INF ? std::to_string(incumbent_best_cost) : "none");

        return incumbent_best_cost < TGConstants::INF;
    }
};

} // namespace plan
