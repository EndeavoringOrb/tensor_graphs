#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <functional>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/cost_model.hpp"
#include "core/kernels.hpp"
#include "core/rewrite.hpp"
#include "core/shapes.hpp"
#include "core/misc.hpp"
#include "core/egraph.hpp"
#include "core/common/constants.hpp"
#include "core/plan/validators/validator.hpp"


struct ENodeInfo
{
    float cost;
    std::unordered_map<uint32_t, uint64_t> mem_sizes;
    bool inplace;
    int32_t inplace_idx;
    bool is_scatter;
    bool is_view;
};

struct Extractor
{
private:
    std::vector<std::unique_ptr<ISelectionValidator>> validators;

public:
    std::unordered_map<EClassId, uint32_t> selection_map; // EClass -> ENode (idx into EClass.enodes)
    const EGraph egraph;
    std::vector<EClassId> path;                      // List of EClasses in selection_map, in order root -> leaves
    std::vector<EClassId> to_process;                // EClass ids to process
    std::vector<EClassId> to_process_enode;          // what does this do??? is it just used to know when we have extracted all graphs???
    std::unordered_map<EClassId, uint32_t> next_sel; // EClass -> ENode idx, what enode should we move to next time we encounter this eclass
    EClassId target_backtrack_eclass;
    uint64_t numClasses;

    Extractor(uint64_t _numClasses) : numClasses(_numClasses) {}

    void registerValidator(std::unique_ptr<ISelectionValidator> validator)
    {
        validators.push_back(std::move(validator));
    }

    bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map,
                  std::string &reason)
    {
        for (const auto &validator : validators)
        {
            if (!validator->validate(selection_map, reason))
            {
                return false;
            }
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
    }

    void backtrack(std::string reason)
    {
        target_backtrack_eclass = EClassId{UINT32_MAX};
        if (reason == "cycle")
        {
            // Compute SCCs of the selection-induced subgraph using Tarjan's algorithm.
            // For each SCC of size > 1 (or size == 1 with a self-loop, i.e. a true cycle):
            //   collect members that are in `path` and have >= 1 alternative.
            // Among all such members across all non-trivial SCCs, pick the one
            // with the smallest path index. That's the backtrack target.

            int best_backtrack_idx = std::numeric_limits<int>::max();
            std::vector<int> path_idx(numClasses, -1);
            for (int i = 0; i < (int)path.size(); ++i)
            {
                path_idx[egraph.findConst(path[i]).value] = i;
            }

            std::vector<int> disc(numClasses, -1);
            std::vector<int> low(numClasses, -1);
            std::vector<bool> onStack(numClasses, false);
            std::vector<uint32_t> st;
            int time_counter = 0;

            std::function<void(uint32_t)> tarjan = [&](uint32_t u)
            {
                disc[u] = low[u] = time_counter++;
                st.push_back(u);
                onStack[u] = true;

                auto it = selection_map.find(u);
                if (it != selection_map.end())
                {
                    uint32_t sel = it->second;
                    uint32_t enode_id = egraph.getEClass(u).enodes[sel];
                    for (uint32_t v : precomp.enode_canon_children[enode_id])
                    {
                        if (selection_map.find(v) != selection_map.end())
                        {
                            if (disc[v] == -1)
                            {
                                tarjan(v);
                                low[u] = std::min(low[u], low[v]);
                            }
                            else if (onStack[v])
                            {
                                low[u] = std::min(low[u], disc[v]);
                            }
                        }
                    }
                }

                if (low[u] == disc[u])
                {
                    std::vector<uint32_t> scc;
                    while (true)
                    {
                        uint32_t v = st.back();
                        st.pop_back();
                        onStack[v] = false;
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
                        uint32_t v = scc[0];
                        auto itv = selection_map.find(v);
                        if (itv != selection_map.end())
                        {
                            uint32_t sel = itv->second;
                            uint32_t enode_id = egraph.getEClass(v).enodes[sel];
                            for (uint32_t child : precomp.enode_canon_children[enode_id])
                            {
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
                        for (uint32_t v : scc)
                        {
                            if (path_idx[v] != -1)
                            {
                                auto choiceIt = selection_map.find(v);
                                if (choiceIt != selection_map.end())
                                {
                                    uint32_t sel = choiceIt->second;
                                    const auto &enodes = egraph.getEClass(v).enodes;
                                    if (sel + 1 < enodes.size())
                                    {
                                        if (path_idx[v] < best_backtrack_idx)
                                        {
                                            best_backtrack_idx = path_idx[v];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            };

            for (const auto &kv : selection_map)
            {
                if (disc[kv.first] == -1)
                {
                    tarjan(kv.first);
                }
            }

            if (best_backtrack_idx != std::numeric_limits<int>::max())
            {
                target_backtrack_eclass = path[best_backtrack_idx];
                std::cout << "[Planner.extractBest] cycle: backtracking to eclass "
                          << std::to_string(target_backtrack_eclass)
                          << " (path index " << best_backtrack_idx << " of " << path.size() << ")"
                          << std::endl;
            }
        }
    }

    void ascend(const std::vector<ENodeInfo> &enodeInfos)
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
            const ENodeInfo &info = enodeInfos[enode_id.value];

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