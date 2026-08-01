#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/constants.hpp"
#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/misc.hpp"
#include "core/plan/validators/validator.hpp"
#include "core/rewrite.hpp"
#include "core/shapes.hpp"
#include "core/types.hpp"

struct ENodeInfo
{
    float cost;
    bool is_view;
};

struct DispatchIterator
{
  public:
    DispatchIterator(const EGraph &_egraph, const std::unordered_map<EClassId, uint32_t> &selection_map)
        : egraph(_egraph)
    {
        initOrderState(selection_map);
    }

    bool getNextDispatchOrder(const std::unordered_map<EClassId, uint32_t> &selection_map,
                              std::vector<EClassId> &out_order)
    {
        if (is_done)
            return false;

        if (!first_yield) // The first time we are at the root we don't want to
                          // exit
        {
            if (!ascend()) // If we are back at the root, we have gone through all
                           // dispatch orders
            {
                is_done = true;
                return false;
            }
        }
        first_yield = false;

        while (true)
        {
            while (true)
            {
                std::vector<EClassId> ready = get_ready(selection_map);

                // Safety check for dependency cycles (though CycleValidator handles
                // most of this)
                if (ready.empty() && !remaining.empty())
                {
                    is_done = true;
                    return false;
                }

                uint32_t pos = static_cast<uint32_t>(ordered.size());
                uint32_t choice = 0;
                auto it = selection_at_pos.find(pos);
                if (it != selection_at_pos.end())
                {
                    choice = it->second + 1;
                }

                if (choice < ready.size())
                {
                    selection_at_pos[pos] = choice;
                    EClassId node = ready[choice];
                    ordered.push_back(node);
                    remaining.erase(node);
                }
                else
                {
                    if (ordered.empty())
                    {
                        is_done = true;
                        return false;
                    }
                    if (!ascend())
                    {
                        is_done = true;
                        return false;
                    }
                }

                if (remaining.empty())
                {
                    break;
                }
            }

            out_order = ordered;
            iter++;
            return true;
        }
    }

    uint32_t getIter()
    {
        return iter;
    }

  private:
    const EGraph &egraph;
    std::unordered_set<EClassId> remaining;
    std::vector<EClassId> ordered;
    std::unordered_map<uint32_t, uint32_t> selection_at_pos;
    bool is_done = false;
    bool first_yield = true;
    uint32_t iter = 0;

    void initOrderState(const std::unordered_map<EClassId, uint32_t> &selection_map)
    {
        remaining.clear();
        for (const auto &kv : selection_map)
        {
            remaining.insert(kv.first);
        }
        ordered.clear();
        selection_at_pos.clear();
        is_done = false;
        first_yield = true;
    }

    std::vector<EClassId> get_ready(const std::unordered_map<EClassId, uint32_t> &selection_map)
    {
        std::vector<EClassId> ready;
        for (EClassId node : remaining)
        {
            uint32_t sel = selection_map.at(node);
            ENodeId enode_id = egraph.getEClass(node).enodes[sel];
            const ENode &enode = egraph.getENode(enode_id);

            bool node_ready = true;
            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = egraph.findConst(child);
                if (remaining.find(canon_child) != remaining.end())
                {
                    node_ready = false;
                    break;
                }
            }
            if (node_ready)
            {
                ready.push_back(node);
            }
        }
        return ready;
    }

    bool ascend()
    {
        selection_at_pos.erase(static_cast<uint32_t>(ordered.size()));
        if (ordered.empty())
            return false;
        EClassId last = ordered.back();
        ordered.pop_back();
        remaining.insert(last);
        return true;
    }
};

struct Extractor
{
  private:
    std::vector<std::unique_ptr<ISelectionValidator>> validators;

  public:
    std::unordered_map<EClassId, uint32_t> selection_map; // EClass -> ENode (idx into EClass.enodes)
    const EGraph &egraph;
    std::vector<EClassId> path;                      // List of EClasses in selection_map, in order root -> leaves
    std::vector<EClassId> to_process;                // EClass ids to process
    std::vector<EClassId> to_process_enode;          // what does this do??? is it just used to know when we
                                                     // have extracted all graphs???
    std::unordered_map<EClassId, uint32_t> next_sel; // EClass -> ENode idx, what enode should we move to next time
                                                     // we encounter this eclass
    EClassId target_backtrack_eclass;
    uint64_t numClasses;

    bool updated_buffers = false;
    bool updated_cost = false;

    Extractor(const EGraph &_egraph, EClassId root_eclass_id)
        : egraph(_egraph), numClasses(_egraph.classes.size()), to_process({root_eclass_id})
    {
    }

    void registerValidator(std::unique_ptr<ISelectionValidator> validator)
    {
        validators.push_back(std::move(validator));
    }

    bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<EClassId> &order,
                  std::vector<ParallelBuffer> &buffers, std::unordered_map<EClassId, BufferId> &eclass_to_buf,
                  BufferId &overflow, float &cost, std::string &reason)
    {
        updated_buffers = false;
        updated_cost = false;
        for (const auto &validator : validators)
        {
            if (!validator->validate(selection_map, order, buffers, eclass_to_buf, overflow, cost, reason,
                                     updated_buffers, updated_cost))
            {
                return false;
            }
        }
        if (!updated_buffers)
        {
            Error::throw_err("buffers not updated during validate");
        }
        if (!updated_cost)
        {
            Error::throw_err("cost not updated during validate");
        }
        return true;
    }

    // Returns the next graph contained in the egraph
    const std::unordered_map<EClassId, uint32_t> &getNextSelection()
    {
        while (!to_process.empty())
        {
            EClassId current = to_process.front();
            to_process.erase(to_process.begin());

            if (selection_map.find(current) != selection_map.end())
            {
                continue;
            }

            path.push_back(current);

            uint32_t sel = 0;
            auto nextIt = next_sel.find(current);
            if (nextIt != next_sel.end())
            {
                sel = nextIt->second;
                next_sel.erase(nextIt);
            }

            const auto &enodes = egraph.getEClass(current).enodes;
            if (sel >= enodes.size())
            {
                Error::throw_err("Invalid selection index in EGraph");
            }

            ENodeId enode_id = enodes[sel];
            const ENode &node = egraph.getENode(enode_id);

            selection_map[current] = sel;

            if (enodes.size() > sel + 1)
            {
                if (std::find(to_process_enode.begin(), to_process_enode.end(), current) == to_process_enode.end())
                {
                    to_process_enode.push_back(current);
                }
            }

            std::vector<EClassId> new_to_process;
            new_to_process.reserve(node.getChildren().size());
            for (EClassId child : node.getChildren())
            {
                EClassId childEClass = egraph.findConst(child);
                if (selection_map.find(childEClass) == selection_map.end())
                {
                    new_to_process.push_back(childEClass);
                }
            }
            to_process.insert(to_process.begin(), new_to_process.begin(), new_to_process.end());
        }
        return selection_map;
    }

    void backtrack(const std::string reason, const std::unordered_map<EClassId, BufferId> &eclass_to_buf,
                   const std::vector<ParallelBuffer> &buffers, const BufferId overflow)
    {
        target_backtrack_eclass = EClassId{UINT32_MAX};

        if (reason == "cycle")
        {
            int best_backtrack_idx = std::numeric_limits<int>::max();
            // Compute SCCs of the graph using Tarjan's algorithm.
            // For each SCC of size > 1 (or size == 1 with a self-loop,
            // i.e. a true cycle) choose a representative (max path idx)
            // Backtrack target is min of all representatives

            std::vector<int> path_idx(numClasses, -1);
            for (int i = 0; i < (int)path.size(); ++i)
            {
                path_idx[egraph.findConst(path[i]).value] = i;
            }

            std::vector<int> disc(numClasses, -1);
            std::vector<int> low(numClasses, -1);
            std::vector<bool> onStack(numClasses, false);
            std::vector<EClassId> st;
            int time_counter = 0;

            std::function<void(EClassId)> tarjan = [&](EClassId u) {
                disc[u.value] = low[u.value] = time_counter++;
                st.push_back(u);
                onStack[u.value] = true;

                auto it = selection_map.find(u);
                if (it != selection_map.end())
                {
                    uint32_t sel = it->second;
                    ENodeId enode_id = egraph.getEClass(u).enodes[sel];
                    for (EClassId v : egraph.getENode(enode_id).getChildren())
                    {
                        v = egraph.findConst(v);
                        if (selection_map.find(v) != selection_map.end())
                        {
                            if (disc[v.value] == -1)
                            {
                                tarjan(v);
                                low[u.value] = std::min(low[u.value], low[v.value]);
                            }
                            else if (onStack[v.value])
                            {
                                low[u.value] = std::min(low[u.value], disc[v.value]);
                            }
                        }
                    }
                }

                if (low[u.value] == disc[u.value])
                {
                    std::vector<EClassId> scc;
                    while (true)
                    {
                        EClassId v = st.back();
                        st.pop_back();
                        onStack[v.value] = false;
                        scc.push_back(v);
                        if (u == v)
                            break;
                    }

                    bool is_cycle = false;
                    if (scc.size() > 1)
                    {
                        is_cycle = true;
                    }
                    else if (scc.size() == 1)
                    {
                        EClassId v = scc[0];
                        auto itv = selection_map.find(v);
                        if (itv != selection_map.end())
                        {
                            uint32_t sel = itv->second;
                            ENodeId enode_id = egraph.getEClass(v).enodes[sel];
                            for (EClassId child : egraph.getENode(enode_id).getChildren())
                            {
                                child = egraph.findConst(child);
                                if (child == v)
                                {
                                    is_cycle = true;
                                    break;
                                }
                            }
                        }
                    }

                    if (is_cycle)
                    {
                        int32_t scc_highest = -1;
                        for (EClassId v : scc)
                        {
                            if (path_idx[v.value] != -1)
                            {
                                auto choiceIt = selection_map.find(v);
                                if (choiceIt != selection_map.end())
                                {
                                    uint32_t sel = choiceIt->second;
                                    const auto &enodes = egraph.getEClass(v).enodes;
                                    if (sel + 1 < enodes.size())
                                    {
                                        if (path_idx[v.value] > scc_highest)
                                        {
                                            scc_highest = path_idx[v.value];
                                        }
                                    }
                                }
                            }
                        }
                        if (scc_highest != -1)
                        {
                            best_backtrack_idx = std::min(scc_highest, best_backtrack_idx);
                        }
                    }
                }
            };

            for (const auto &kv : selection_map)
            {
                if (disc[kv.first.value] == -1)
                {
                    tarjan(kv.first);
                }
            }

            if (best_backtrack_idx != std::numeric_limits<int>::max())
            {
                target_backtrack_eclass = path[best_backtrack_idx];
                std::cout << "[Planner.extractBest] cycle: backtracking to eclass " << toString(target_backtrack_eclass)
                          << " (path index " << best_backtrack_idx << " of " << path.size() << ")" << std::endl;
            }
        }
        else if (reason.rfind("OOM", 0) == 0)
        {
            int best_backtrack_idx = -1;

            // Find all buffers overlapping with the overflow buffer
            int buf_idx = -1;
            for (int i = 0; i < buffers.size(); i++)
            {
                if (buffers[i].id == overflow)
                {
                    buf_idx = i;
                }
            }
            std::unordered_set<BufferId> overflows;
            if (buf_idx != -1)
            {
                for (int i = 0; i < buffers.size(); i++)
                {
                    if (overlapsBuf(buffers[buf_idx], buffers[i]))
                    {
                        overflows.insert(buffers[i].id);
                    }
                }
            }
            std::cout << "got " << std::to_string(overflows.size()) << " buffers at overflow" << std::endl;

            // Search path from deepest to highest for an eclass that was in the overflow buffer
            if (overflows.size() > 0)
            {
                for (int i = path.size() - 1; i >= 0; --i)
                {
                    EClassId ec = path[i];
                    auto selIt = selection_map.find(ec);
                    if (selIt == selection_map.end())
                        continue;

                    auto bufIt = eclass_to_buf.find(ec);
                    if (bufIt == eclass_to_buf.end())
                        Error::throw_err("eclass has no buffer");

                    if (!overflows.count(bufIt->second))
                        continue;

                    best_backtrack_idx = i;
                    break;
                }
            }

            if (best_backtrack_idx != -1)
            {
                target_backtrack_eclass = path[best_backtrack_idx];
                std::cout << "[Planner.extractBest] OOM: backtracking to eclass " << toString(target_backtrack_eclass)
                          << " (path index " << best_backtrack_idx << " of " << path.size() << ")" << std::endl;
            }
        }
    }

    void ascend()
    {
        bool skip_increment = (target_backtrack_eclass != EClassId{UINT32_MAX});

        while (!path.empty())
        {
            EClassId current = path.back();
            path.pop_back();

            if (selection_map.find(current) == selection_map.end())
                continue;

            if (skip_increment && current == target_backtrack_eclass)
            {
                std::cout << "skipped back to path size " << std::to_string(path.size()) << std::endl;
                skip_increment = false;
            }

            uint32_t sel = selection_map[current];
            const auto &enodes = egraph.getEClass(current).enodes;
            ENodeId enode_id = enodes[sel];
            const ENode &node = egraph.getENode(enode_id);

            if (!skip_increment && sel + 1 < enodes.size())
            {
                next_sel[current] = sel + 1;

                std::vector<EClassId> keys_to_delete;
                keys_to_delete.reserve(selection_map.size());
                for (const auto &kv : selection_map)
                {
                    if (std::find(path.begin(), path.end(), kv.first) == path.end() && kv.first != current)
                    {
                        keys_to_delete.push_back(kv.first);
                    }
                }
                for (EClassId k : keys_to_delete)
                    selection_map.erase(k);

                selection_map.erase(current);

                auto it = std::remove(to_process_enode.begin(), to_process_enode.end(), current);
                if (it != to_process_enode.end())
                    to_process_enode.erase(it, to_process_enode.end());

                if (enodes.size() > sel + 2)
                {
                    to_process_enode.push_back(current);
                }

                to_process.clear();
                for (EClassId eclass : path)
                {
                    ENodeId n_id = egraph.getEClass(eclass).enodes[selection_map[eclass]];
                    const ENode &n = egraph.getENode(n_id);
                    std::vector<EClassId> new_to_process;
                    new_to_process.reserve(n.getChildren().size());
                    for (EClassId child : n.getChildren())
                    {
                        EClassId childEClass = egraph.findConst(child);
                        if (selection_map.find(childEClass) == selection_map.end())
                        {
                            new_to_process.push_back(childEClass);
                        }
                    }
                    to_process.insert(to_process.begin(), new_to_process.begin(), new_to_process.end());
                }
                to_process.insert(to_process.begin(), current);
                break;
            }
            else
            {
                selection_map.erase(current);
                auto it = std::remove(to_process_enode.begin(), to_process_enode.end(), current);
                if (it != to_process_enode.end())
                    to_process_enode.erase(it, to_process_enode.end());
            }
        }
    }
};