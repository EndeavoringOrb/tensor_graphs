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

        if (!first_yield)
        {
            if (!ascend())
            {
                is_done = true;
                return false;
            }
        }
        first_yield = false;

        uint32_t total_nodes = static_cast<uint32_t>(num_nodes_in_selection);
        if (total_nodes == 0)
        {
            is_done = true;
            return false;
        }

        while (true)
        {
            uint32_t pos = static_cast<uint32_t>(ordered.size());

            if (pos == total_nodes)
            {
                out_order = ordered;
                iter++;
                return true;
            }

            const auto &ready = ready_stack[pos];
            uint32_t choice = selection_at_pos[pos];

            if (choice < ready.size())
            {
                selection_at_pos[pos] = choice + 1;
                EClassId node = ready[choice];
                ordered.push_back(node);

                if (pos + 1 >= ready_stack.size())
                {
                    ready_stack.resize(pos + 2);
                }
                ready_stack[pos + 1].clear();

                for (uint32_t i = 0; i < ready.size(); ++i)
                {
                    if (i != choice)
                    {
                        ready_stack[pos + 1].push_back(ready[i]);
                    }
                }

                for (EClassId dep : dependents[node.value])
                {
                    current_in_degree[dep.value]--;
                    if (current_in_degree[dep.value] == 0)
                    {
                        ready_stack[pos + 1].push_back(dep);
                    }
                }
            }
            else
            {
                selection_at_pos[pos] = 0;
                if (!ascend())
                {
                    is_done = true;
                    return false;
                }
            }
        }
    }

    uint32_t getIter() const
    {
        return iter;
    }

private:
    const EGraph &egraph;
    size_t num_nodes_in_selection = 0;
    std::vector<EClassId> ordered;
    std::vector<int32_t> current_in_degree;
    std::vector<std::vector<EClassId>> dependents;
    std::vector<std::vector<EClassId>> ready_stack;
    std::vector<uint32_t> selection_at_pos;
    bool is_done = false;
    bool first_yield = true;
    uint32_t iter = 0;

    void initOrderState(const std::unordered_map<EClassId, uint32_t> &selection_map)
    {
        num_nodes_in_selection = selection_map.size();
        ordered.clear();
        ordered.reserve(num_nodes_in_selection);
        is_done = false;
        first_yield = true;
        iter = 0;

        uint32_t max_class_id = static_cast<uint32_t>(egraph.getClasses().size());

        current_in_degree.assign(max_class_id, 0);
        dependents.clear();
        dependents.resize(max_class_id);

        ready_stack.clear();
        ready_stack.resize(num_nodes_in_selection + 1);

        selection_at_pos.assign(num_nodes_in_selection + 1, 0);

        std::vector<EClassId> initial_ready;
        initial_ready.reserve(1024);

        std::vector<uint8_t> in_selection(max_class_id, 0);
        for (const auto &kv : selection_map)
        {
            EClassId canon_key = egraph.findConst(kv.first);
            if (canon_key.value < max_class_id)
            {
                in_selection[canon_key.value] = 1;
            }
        }

        std::vector<EClassId> unique_children;
        for (const auto &kv : selection_map)
        {
            EClassId node = egraph.findConst(kv.first);
            if (node.value >= max_class_id)
                continue;

            uint32_t sel = kv.second;
            ENodeId enode_id = egraph.getEClass(node).enodes[sel];
            const ENode &enode = egraph.getENode(enode_id);

            unique_children.clear();
            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = egraph.findConst(child);
                if (canon_child.value < max_class_id && in_selection[canon_child.value] && canon_child != node)
                {
                    if (std::find(unique_children.begin(), unique_children.end(), canon_child) == unique_children.end())
                    {
                        unique_children.push_back(canon_child);
                    }
                }
            }

            current_in_degree[node.value] = static_cast<int32_t>(unique_children.size());
            if (current_in_degree[node.value] == 0)
            {
                initial_ready.push_back(node);
            }

            for (EClassId canon_child : unique_children)
            {
                dependents[canon_child.value].push_back(node);
            }
        }

        ready_stack[0] = std::move(initial_ready);
    }

    bool ascend()
    {
        uint32_t pos = static_cast<uint32_t>(ordered.size());
        selection_at_pos[pos] = 0;
        if (ordered.empty())
            return false;

        EClassId last = ordered.back();
        ordered.pop_back();

        for (EClassId dep : dependents[last.value])
        {
            current_in_degree[dep.value]++;
        }

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
            if (!validator->validate(selection_map, order, path, buffers, eclass_to_buf, cost, conflict_nodes))
            {
                return false;
            }
        }
        return true;
    }

    // Returns true if successfully formed a selection map, false if it needs to backtrack
    bool getNextSelection()
    {
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
                LOG(L_DEBUG) << "skipped back to path size " << std::to_string(path.size()) << std::endl;
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