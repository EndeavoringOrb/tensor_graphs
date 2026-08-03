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
        ProgressTimer t(0, "getNextDispatchOrder", false, true);
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
    std::vector<EClassId> path; // List of EClasses in selection_map, in order root -> leaves
    std::vector<bool> in_path;
    std::vector<int> path_pos;
    std::vector<EClassId> to_process; // EClass ids to process
    std::vector<bool> has_options;
    uint32_t active_options = 0;
    std::unordered_map<EClassId, uint32_t> next_sel; // EClass -> ENode idx, what enode should we move to next time
    EClassId target_backtrack_eclass;
    uint64_t numClasses;

    Extractor(const EGraph &_egraph, EClassId root_eclass_id)
        : egraph(_egraph), numClasses(_egraph.classes.size()), to_process({root_eclass_id}),
          in_path(_egraph.classes.size(), false), path_pos(_egraph.classes.size(), -1), has_options(_egraph.classes.size(), false)
    {
    }

    void registerValidator(std::unique_ptr<ISelectionValidator> validator)
    {
        validators.push_back(std::move(validator));
    }

    bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<EClassId> &order,
                  std::vector<ParallelBuffer> &buffers, std::unordered_map<EClassId, BufferId> &eclass_to_buf,
                  float &cost, std::vector<EClassId> &conflict_nodes)
    {
        for (const auto &validator : validators)
        {
            if (!validator->validate(selection_map, order, buffers, eclass_to_buf, cost, conflict_nodes))
            {
                return false;
            }
        }
        return true;
    }

    // Returns true if successfully formed a selection map, false if it needs to backtrack
    bool getNextSelection()
    {
        ProgressTimer t(0, "getNextSelection", false, true);
        while (!to_process.empty())
        {
            EClassId current = to_process.back();
            to_process.pop_back();

            if (selection_map.find(current) != selection_map.end())
            {
                continue;
            }

            path.push_back(current);
            in_path[current.value] = true;
            path_pos[current.value] = path.size() - 1;

            uint32_t sel = 0;
            auto nextIt = next_sel.find(current);
            if (nextIt != next_sel.end())
            {
                sel = nextIt->second;
                next_sel.erase(nextIt);
            }

            const auto &enodes = egraph.getEClass(current).enodes;
            
            bool found_valid = false;
            int max_conflict_path_pos = -1;

            for (; sel < enodes.size(); ++sel)
            {
                ENodeId enode_id = enodes[sel];
                selection_map[current] = sel;
                
                bool step_valid = true;
                std::vector<EClassId> conflict_nodes;
                
                for (const auto &validator : validators)
                {
                    if (!validator->validateStep(current, enode_id, selection_map, conflict_nodes))
                    {
                        step_valid = false;
                        break;
                    }
                }
                
                if (step_valid)
                {
                    found_valid = true;
                    break;
                }
                else
                {
                    for (EClassId c_node : conflict_nodes)
                    {
                        if (c_node != current && path_pos[c_node.value] != -1)
                        {
                            if (path_pos[c_node.value] > max_conflict_path_pos)
                            {
                                max_conflict_path_pos = path_pos[c_node.value];
                            }
                        }
                    }
                    selection_map.erase(current);
                }
            }

            if (!found_valid)
            {
                if (max_conflict_path_pos != -1)
                {
                    target_backtrack_eclass = path[max_conflict_path_pos];
                }
                return false;
            }

            if (enodes.size() > sel + 1)
            {
                if (!has_options[current.value])
                {
                    has_options[current.value] = true;
                    active_options++;
                }
            }

            ENodeId enode_id = enodes[sel];
            const ENode &node = egraph.getENode(enode_id);
            const auto &children = node.getChildren();
            for (auto it = children.rbegin(); it != children.rend(); ++it)
            {
                EClassId childEClass = egraph.findConst(*it);
                if (selection_map.find(childEClass) == selection_map.end())
                {
                    to_process.push_back(childEClass);
                }
            }
        }
        return true;
    }

    void ascend()
    {
        ProgressTimer t(0, "ascend", false, true);
        bool skip_increment = (target_backtrack_eclass != EClassId{UINT32_MAX});

        while (!path.empty())
        {
            EClassId current = path.back();
            path.pop_back();
            in_path[current.value] = false;

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

                selection_map.erase(current);

                if (enodes.size() <= sel + 2)
                {
                    if (has_options[current.value])
                    {
                        has_options[current.value] = false;
                        active_options--;
                    }
                }
                else
                {
                    if (!has_options[current.value])
                    {
                        has_options[current.value] = true;
                        active_options++;
                    }
                }

                to_process.clear();
                for (EClassId eclass : path)
                {
                    ENodeId n_id = egraph.getEClass(eclass).enodes[selection_map[eclass]];
                    const ENode &n = egraph.getENode(n_id);
                    const auto &children = n.getChildren();
                    for (auto c_it = children.rbegin(); c_it != children.rend(); ++c_it)
                    {
                        EClassId childEClass = egraph.findConst(*c_it);
                        if (selection_map.find(childEClass) == selection_map.end())
                        {
                            to_process.push_back(childEClass);
                        }
                    }
                }
                to_process.push_back(current);
                break;
            }
            else
            {
                selection_map.erase(current);
                if (has_options[current.value])
                {
                    has_options[current.value] = false;
                    active_options--;
                }
            }
        }
    }
};