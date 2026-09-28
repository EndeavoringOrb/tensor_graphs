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
    std::pair<VarId, Domain> left_delta;
    std::pair<VarId, Domain> right_delta;
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

        out_decision.left_delta = {var_id, makeFixedLike(domain, preferred)};
        out_decision.right_delta = {var_id, right};
        return true;
    }

    static EClassId resolveBaseClass(const SearchState &state, uint32_t b, EClassId cid)
    {
        EClassId curr = cid;
        std::unordered_set<EClassId> visited;
        while (visited.insert(curr).second)
        {
            auto sel_it = state.selected_vars[b].find(curr);
            if (sel_it == state.selected_vars[b].end())
                break;
            const Domain &dom = state.domains[sel_it->second];
            if (!dom.isFixed() || dom.fixedValue() <= 0)
                break;
            uint32_t en_idx = static_cast<uint32_t>(dom.fixedValue() - 1);
            const EClass &cls = state.bucket_egraphs[b].getEClass(curr);
            if (en_idx >= cls.enodes.size())
                break;
            ENodeId en_id = cls.enodes[en_idx];
            bool is_view = en_id.value < state.bucket_enode_infos[b].size() &&
                           state.bucket_enode_infos[b][en_id.value].is_view;
            if (!is_view)
                break;
            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
            if (enode.getChildren().empty())
                break;
            curr = state.bucket_egraphs[b].findConst(enode.getChildren()[0]);
        }
        return curr;
    }

    static bool isViewOf(const SearchState &state, uint32_t b, EClassId base, EClassId view_cand)
    {
        return resolveBaseClass(state, b, view_cand) == state.bucket_egraphs[b].findConst(base);
    }

    static std::vector<EClassId> getEClassReaders(const SearchState &state, uint32_t b, EClassId target)
    {
        EClassId canon_target = state.bucket_egraphs[b].findConst(target);
        std::vector<EClassId> readers;
        for (const auto &pair : state.selected_vars[b])
        {
            EClassId cand = pair.first;
            if (cand == target)
                continue;
            const Domain &dom = state.domains[pair.second];
            if (dom.isFixed() && dom.fixedValue() == 0)
                continue;
            uint32_t en_idx = (dom.isFixed() && dom.fixedValue() > 0) ? static_cast<uint32_t>(dom.fixedValue() - 1) : 0;
            const EClass &cls = state.bucket_egraphs[b].getEClass(cand);
            if (en_idx >= cls.enodes.size())
                continue;
            ENodeId en_id = cls.enodes[en_idx];
            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
            for (EClassId ch : enode.getChildren())
            {
                EClassId base_ch = resolveBaseClass(state, b, ch);
                if (state.bucket_egraphs[b].findConst(base_ch) == canon_target)
                {
                    readers.push_back(cand);
                    break;
                }
            }
        }
        return readers;
    }

    static int32_t preferredOffset(const SearchState &state, uint32_t b, EClassId cid,
                                   const Domain &offset_domain)
    {
        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        uint32_t psize = state.bytesToPages(getSizeBytes(cls.shape, cls.dtype), cls.mem_space);
        uint32_t curr_size = (psize == 0) ? 1 : psize;

        EClassId base_cid = resolveBaseClass(state, b, cid);
        if (base_cid != cid)
        {
            auto base_off_it = state.offset_vars[b].find(base_cid);
            if (base_off_it != state.offset_vars[b].end())
            {
                const Domain &base_dom = state.domains[base_off_it->second];
                if (base_dom.isFixed() && offset_domain.contains(base_dom.fixedValue()))
                {
                    return base_dom.fixedValue();
                }
            }
        }

        int32_t curr_start = 0;
        int32_t curr_end = std::numeric_limits<int32_t>::max();
        bool is_curr_root = (b < state.bucket_root_ids.size() &&
                             state.bucket_egraphs[b].findConst(cid) == state.bucket_egraphs[b].findConst(state.bucket_root_ids[b]));

        auto sel_it = state.selected_vars[b].find(cid);
        uint32_t en_idx = 0;
        bool is_curr_input_or_cache = (cls.base_eclass_id != BaseEClassId{} && state.preallocated_buffers.count(cls.base_eclass_id));
        if (sel_it != state.selected_vars[b].end())
        {
            const Domain &sel_dom = state.domains[sel_it->second];
            if (sel_dom.isFixed() && sel_dom.fixedValue() > 0)
                en_idx = static_cast<uint32_t>(sel_dom.fixedValue() - 1);
            if (en_idx < cls.enodes.size())
            {
                const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
                    is_curr_input_or_cache = true;
            }
        }

        if (!is_curr_input_or_cache && !is_curr_root)
        {
            auto st_it = state.start_vars[b].find(cid);
            if (st_it != state.start_vars[b].end() && en_idx < st_it->second.size())
            {
                const Domain &st_dom = state.domains[st_it->second[en_idx]];
                if (st_dom.isFixed())
                    curr_start = st_dom.fixedValue();
            }
            curr_end = curr_start + 1;
            auto readers = getEClassReaders(state, b, cid);
            for (EClassId r_cid : readers)
            {
                auto r_sel_it = state.selected_vars[b].find(r_cid);
                if (r_sel_it == state.selected_vars[b].end())
                    continue;
                const Domain &r_sel_dom = state.domains[r_sel_it->second];
                if (!r_sel_dom.isFixed() || r_sel_dom.fixedValue() <= 0)
                    continue;
                uint32_t r_en = static_cast<uint32_t>(r_sel_dom.fixedValue() - 1);
                auto r_st_it = state.start_vars[b].find(r_cid);
                if (r_st_it != state.start_vars[b].end() && r_en < r_st_it->second.size())
                {
                    const Domain &r_st_dom = state.domains[r_st_it->second[r_en]];
                    if (r_st_dom.isFixed())
                        curr_end = std::max(curr_end, r_st_dom.fixedValue() + 1);
                    else
                        curr_end = std::max(curr_end, r_st_dom.getMax() + 1);
                }
            }
        }

        struct Obstacle
        {
            uint32_t start;
            uint32_t end;
        };
        std::vector<Obstacle> obstacles;

        for (const auto &pair : state.preallocated_buffers)
        {
            if (pair.second.mem_space == cls.mem_space)
            {
                uint32_t align = state.getPageAlignment(cls.mem_space);
                uint32_t start_p = static_cast<uint32_t>(pair.second.offset / align);
                uint32_t end_p = static_cast<uint32_t>((pair.second.offset + pair.second.size + align - 1) / align);
                obstacles.push_back({start_p, end_p});
            }
        }

        for (const auto &pair : state.selected_vars[b])
        {
            EClassId other_cid = pair.first;
            if (other_cid == cid)
                continue;

            auto off_it = state.offset_vars[b].find(other_cid);
            if (off_it == state.offset_vars[b].end())
                continue;
            const Domain &off_dom = state.domains[off_it->second];
            if (!off_dom.isFixed())
                continue;

            const Domain &other_sel_dom = state.domains[pair.second];
            if (other_sel_dom.isFixed() && other_sel_dom.fixedValue() == 0)
                continue;

            const EClass &other_cls = state.bucket_egraphs[b].getEClass(other_cid);
            if (other_cls.mem_space != cls.mem_space)
                continue;

            if (isViewOf(state, b, cid, other_cid) || isViewOf(state, b, other_cid, cid) ||
                (base_cid != cid && base_cid == resolveBaseClass(state, b, other_cid)))
                continue;

            uint32_t o_en = (other_sel_dom.isFixed() && other_sel_dom.fixedValue() > 0)
                                ? static_cast<uint32_t>(other_sel_dom.fixedValue() - 1)
                                : 0;
            bool is_other_input_or_cache = (other_cls.base_eclass_id != BaseEClassId{} && state.preallocated_buffers.count(other_cls.base_eclass_id));
            if (o_en < other_cls.enodes.size())
            {
                const ENode &o_enode = state.bucket_egraphs[b].getENode(other_cls.enodes[o_en]);
                if (o_enode.getOpType() == OpType::INPUT || o_enode.getOpType() == OpType::CACHE)
                    is_other_input_or_cache = true;
            }
            bool is_other_root = (b < state.bucket_root_ids.size() &&
                                  state.bucket_egraphs[b].findConst(other_cid) == state.bucket_egraphs[b].findConst(state.bucket_root_ids[b]));

            int32_t other_start = 0;
            int32_t other_end = std::numeric_limits<int32_t>::max();
            if (!is_other_input_or_cache && !is_other_root)
            {
                auto o_st_it = state.start_vars[b].find(other_cid);
                if (o_st_it != state.start_vars[b].end() && o_en < o_st_it->second.size())
                {
                    const Domain &o_st_dom = state.domains[o_st_it->second[o_en]];
                    if (o_st_dom.isFixed())
                        other_start = o_st_dom.fixedValue();
                }
                other_end = other_start + 1;
                auto other_readers = getEClassReaders(state, b, other_cid);
                for (EClassId r_cid : other_readers)
                {
                    auto r_sel_it = state.selected_vars[b].find(r_cid);
                    if (r_sel_it == state.selected_vars[b].end())
                        continue;
                    const Domain &r_sel_dom = state.domains[r_sel_it->second];
                    if (!r_sel_dom.isFixed() || r_sel_dom.fixedValue() <= 0)
                        continue;
                    uint32_t r_en_idx = static_cast<uint32_t>(r_sel_dom.fixedValue() - 1);
                    auto r_st_it = state.start_vars[b].find(r_cid);
                    if (r_st_it != state.start_vars[b].end() && r_en_idx < r_st_it->second.size())
                    {
                        const Domain &r_st_dom = state.domains[r_st_it->second[r_en_idx]];
                        if (r_st_dom.isFixed())
                            other_end = std::max(other_end, r_st_dom.fixedValue() + 1);
                        else
                            other_end = std::max(other_end, r_st_dom.getMax() + 1);
                    }
                }
            }

            bool overlap = (is_curr_input_or_cache || is_other_input_or_cache ||
                            std::max(curr_start, other_start) < std::min(curr_end, other_end));
            if (!overlap)
                continue;

            uint32_t o_psize = state.bytesToPages(getSizeBytes(other_cls.shape, other_cls.dtype), other_cls.mem_space);
            uint32_t other_size = (o_psize == 0) ? 1 : o_psize;
            uint32_t o_offset = static_cast<uint32_t>(off_dom.fixedValue());
            obstacles.push_back({o_offset, o_offset + other_size});
        }

        std::sort(obstacles.begin(), obstacles.end(), [](const Obstacle &a, const Obstacle &b) {
            return a.start < b.start;
        });

        uint32_t p = static_cast<uint32_t>(offset_domain.getMin());
        bool pushed = true;
        while (pushed)
        {
            pushed = false;
            for (const auto &obs : obstacles)
            {
                if (p < obs.end && p + curr_size > obs.start)
                {
                    p = obs.end;
                    pushed = true;
                }
            }
        }

        if (p <= static_cast<uint32_t>(offset_domain.getMax()))
            return static_cast<int32_t>(p);

        return offset_domain.getMin();
    }

    static bool setOffsetDecision(const Domain &domain, int32_t preferred,
                                  BranchDecision &out_decision, VarId var_id)
    {
        if (domain.isFixed() || domain.isEmpty() || !domain.contains(preferred))
            return false;

        out_decision.left_delta = {var_id, Domain::makeFixed(preferred, false)};
        if (preferred + 1 <= domain.getMax())
        {
            out_decision.right_delta = {var_id, Domain::makeRange(preferred + 1, domain.getMax())};
        }
        else
        {
            Domain right = without(domain, preferred);
            out_decision.right_delta = {var_id, right};
        }
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
                    {
                        int32_t preferred = preferredOffset(state, b, cid, offset_domain);
                        return setOffsetDecision(offset_domain, preferred, out_decision, offset_var);
                    }
                }
            }
        }
        return false;
    }

  public:
    bool chooseBranch(const SearchState &state, BranchDecision &out_decision) override
    {
        out_decision.left_delta = {kInvalidVarId, Domain{}};
        out_decision.right_delta = {kInvalidVarId, Domain{}};

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
