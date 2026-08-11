#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
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
#include "core/plan/search_delegate.hpp"
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
    DispatchIterator(const EGraph &_egraph, const std::unordered_map<EClassId, uint32_t> &selection_map,
                     const std::vector<ENodeInfo> &enode_infos,
                     std::shared_ptr<SearchDelegate> _delegate = nullptr)
        : egraph(_egraph), delegate(_delegate)
    {
        LOG(L_DEBUG) << "initializing dispatch iterator";
        initOrderState(selection_map, enode_infos);
    }

    bool getNextDispatchOrder(const std::unordered_map<EClassId, uint32_t> &selection_map,
                              std::vector<EClassId> &out_order)
    {
        LOG(L_DEBUG) << "getNextDispatchOrder";
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

        auto compare_nodes = [&](EClassId a, EClassId b)
        {
            float h_a = heights[a.value];
            float h_b = heights[b.value];
            if (std::abs(h_a - h_b) > 1e-5f)
            {
                return h_a > h_b;
            }
            return a.value < b.value;
        };

        while (true)
        {
            uint32_t pos = static_cast<uint32_t>(ordered.size());

            if (pos == total_nodes)
            {
                out_order = ordered;
                iter++;
                return true;
            }

            if (selection_at_pos[pos] == 0)
            {
                if (delegate)
                {
                    delegate->push_state();
                }
            }

            uint32_t choice_idx = selection_at_pos[pos];

            if (choice_idx < current_ready.size())
            {
                selection_at_pos[pos] = choice_idx + 1;
                uint32_t choice = choice_idx;
                if (delegate)
                {
                    std::vector<ActionFeature> features;
                    features.reserve(current_ready.size());
                    for (auto id : current_ready)
                    {
                        ActionFeature f;
                        f.id = id.value;
                        f.cost = (id.value < heights.size()) ? heights[id.value] : 0.0f;
                        f.size = 0.0f;
                        f.op_type = 0;
                        features.push_back(f);
                    }
                    std::vector<uint32_t> custom_order = delegate->order_dispatch(features);
                    if (choice_idx < custom_order.size()) {
                        choice = custom_order[choice_idx];
                    }
                }

                EClassId node = current_ready[choice];
                ordered.push_back(node);
                chosen_at_pos[pos] = node;
                choice_at_pos[pos] = choice;

                current_ready.erase(current_ready.begin() + choice);

                added_nodes_at_pos[pos].clear();
                for (EClassId dep : dependents[node.value])
                {
                    current_in_degree[dep.value]--;
                    if (current_in_degree[dep.value] == 0)
                    {
                        auto it = std::lower_bound(current_ready.begin(), current_ready.end(), dep, compare_nodes);
                        current_ready.insert(it, dep);
                        added_nodes_at_pos[pos].push_back(dep);
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
    std::shared_ptr<SearchDelegate> delegate;
    size_t num_nodes_in_selection = 0;
    std::vector<EClassId> ordered;
    std::vector<int32_t> current_in_degree;
    std::vector<std::vector<EClassId>> dependents;

    std::vector<EClassId> current_ready;
    std::vector<std::vector<EClassId>> added_nodes_at_pos;
    std::vector<EClassId> chosen_at_pos;
    std::vector<uint32_t> choice_at_pos;
    std::vector<uint32_t> selection_at_pos;

    bool is_done = false;
    bool first_yield = true;
    uint32_t iter = 0;
    std::vector<float> heights;

    void initOrderState(const std::unordered_map<EClassId, uint32_t> &selection_map,
                        const std::vector<ENodeInfo> &enode_infos)
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

        current_ready.clear();
        current_ready.reserve(num_nodes_in_selection);

        added_nodes_at_pos.clear();
        added_nodes_at_pos.resize(num_nodes_in_selection + 1);

        chosen_at_pos.assign(num_nodes_in_selection + 1, EClassId{UINT32_MAX});
        choice_at_pos.assign(num_nodes_in_selection + 1, 0);
        selection_at_pos.assign(num_nodes_in_selection + 1, 0);

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
                current_ready.push_back(node);
            }

            for (EClassId canon_child : unique_children)
            {
                dependents[canon_child.value].push_back(node);
            }
        }

        heights.assign(max_class_id, 0.0f);
        std::vector<bool> height_computed(max_class_id, false);

        std::function<float(EClassId)> get_height = [&](EClassId u) -> float
        {
            if (height_computed[u.value])
                return heights[u.value];

            uint32_t sel = selection_map.at(u);
            ENodeId enode_id = egraph.getEClass(u).enodes[sel];
            float self_cost = enode_infos[enode_id.value].cost;

            float max_dep_height = 0.0f;
            for (EClassId v : dependents[u.value])
            {
                max_dep_height = std::max(max_dep_height, get_height(v));
            }

            float h = self_cost + max_dep_height;
            heights[u.value] = h;
            height_computed[u.value] = true;
            return h;
        };

        for (const auto &kv : selection_map)
        {
            EClassId u = egraph.findConst(kv.first);
            if (u.value < max_class_id)
            {
                get_height(u);
            }
        }

        auto compare_nodes = [&](EClassId a, EClassId b)
        {
            float h_a = heights[a.value];
            float h_b = heights[b.value];
            if (std::abs(h_a - h_b) > 1e-5f)
            {
                return h_a > h_b;
            }
            return a.value < b.value;
        };

        std::sort(current_ready.begin(), current_ready.end(), compare_nodes);
    }

    bool ascend()
    {
        if (delegate)
        {
            delegate->pop_state();
        }
        uint32_t pos = static_cast<uint32_t>(ordered.size());
        selection_at_pos[pos] = 0;
        if (ordered.empty())
            return false;

        EClassId last = ordered.back();
        ordered.pop_back();

        uint32_t parent_pos = pos - 1;

        auto compare_nodes = [&](EClassId a, EClassId b)
        {
            float h_a = heights[a.value];
            float h_b = heights[b.value];
            if (std::abs(h_a - h_b) > 1e-5f)
            {
                return h_a > h_b;
            }
            return a.value < b.value;
        };

        for (auto it_dep = added_nodes_at_pos[parent_pos].rbegin(); it_dep != added_nodes_at_pos[parent_pos].rend();
             ++it_dep)
        {
            EClassId dep = *it_dep;
            auto it = std::lower_bound(current_ready.begin(), current_ready.end(), dep, compare_nodes);
            if (it != current_ready.end() && *it == dep)
            {
                current_ready.erase(it);
            }
            current_in_degree[dep.value]++;
        }
        added_nodes_at_pos[parent_pos].clear();

        EClassId restored_node = chosen_at_pos[parent_pos];
        uint32_t original_choice = choice_at_pos[parent_pos];
        current_ready.insert(current_ready.begin() + original_choice, restored_node);

        return true;
    }
};

struct Extractor
{
private:
    std::vector<std::unique_ptr<ISelectionValidator>> validators;

public:
    std::unordered_map<EClassId, uint32_t> selection_map;
    const EGraph &egraph;
    const std::vector<ENodeInfo> &enodeInfos;
    std::shared_ptr<SearchDelegate> delegate;
    std::vector<EClassId> path;
    std::vector<bool> in_path;
    std::vector<int> path_pos;
    std::vector<EClassId> to_process;
    std::vector<bool> has_options;
    uint32_t active_options = 0;
    std::unordered_map<EClassId, uint32_t> next_sel;
    EClassId target_backtrack_eclass;
    uint64_t numClasses;

    Extractor(const EGraph &_egraph, EClassId root_eclass_id, const std::vector<ENodeInfo> &_enodeInfos, std::shared_ptr<SearchDelegate> _delegate = nullptr)
        : egraph(_egraph), enodeInfos(_enodeInfos), delegate(_delegate), numClasses(_egraph.classes.size()), to_process({root_eclass_id}),
          in_path(_egraph.classes.size(), false), path_pos(_egraph.classes.size(), -1),
          has_options(_egraph.classes.size(), false)
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

    bool getNextSelection()
    {
        LOG(L_DEBUG) << "getNextSelection";
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

            if (sel == 0 && delegate)
            {
                delegate->push_state();
            }

            const auto &enodes = egraph.getEClass(current).enodes;

            bool found_valid = false;
            int max_conflict_path_pos = -1;
            std::vector<EClassId> aggregate_conflicts;

            for (; sel < enodes.size(); ++sel)
            {
                uint32_t chosen_sel = sel;
                if (delegate)
                {
                    std::vector<ActionFeature> features;
                    features.reserve(enodes.size());
                    for (ENodeId enodeId : enodes)
                    {
                        const ENode &enode = egraph.getENode(enodeId);
                        ActionFeature f;
                        f.id = enodeId.value;
                        f.cost = enodeInfos[enodeId.value].cost;
                        f.size = (float)countElements(enode.getShape()) * getDTypeSize(enode.getDType());
                        f.op_type = static_cast<uint32_t>(enode.getOpType());
                        features.push_back(f);
                    }
                    std::vector<uint32_t> custom_order = delegate->order_enodes(current.value, features);
                    if (sel < custom_order.size()) {
                        chosen_sel = custom_order[sel];
                    }
                }

                ENodeId enode_id = enodes[chosen_sel];
                selection_map[current] = chosen_sel;

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
                    aggregate_conflicts.insert(aggregate_conflicts.end(), conflict_nodes.begin(), conflict_nodes.end());
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
                if (delegate) delegate->pop_state();

                int best_conflict_pos = -1;
                for (EClassId c_node : aggregate_conflicts)
                {
                    int pos = path_pos[c_node.value];
                    if (pos != -1 && c_node != current)
                    {
                        auto sel_it = selection_map.find(c_node);
                        if (sel_it != selection_map.end())
                        {
                            uint32_t sel_idx = sel_it->second;
                            if (sel_idx + 1 < egraph.getEClass(c_node).enodes.size())
                            {
                                if (pos > best_conflict_pos)
                                {
                                    best_conflict_pos = pos;
                                }
                            }
                        }
                    }
                }

                if (best_conflict_pos != -1)
                {
                    target_backtrack_eclass = path[best_conflict_pos];
                }
                else
                {
                    if (max_conflict_path_pos != -1)
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

            uint32_t chosen_sel = selection_map[current];
            ENodeId enode_id = enodes[chosen_sel];
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
        LOG(L_DEBUG) << "ascend";
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

            uint32_t chosen_sel = selection_map[current];
            const auto &enodes = egraph.getEClass(current).enodes;

            uint32_t iteration_index = chosen_sel;
            if (delegate)
            {
                std::vector<ActionFeature> features;
                features.reserve(enodes.size());
                for (ENodeId enodeId : enodes)
                {
                    const ENode &enode = egraph.getENode(enodeId);
                    ActionFeature f;
                    f.id = enodeId.value;
                    f.cost = enodeInfos[enodeId.value].cost;
                    f.size = (float)countElements(enode.getShape()) * getDTypeSize(enode.getDType());
                    f.op_type = static_cast<uint32_t>(enode.getOpType());
                    features.push_back(f);
                }
                std::vector<uint32_t> custom_order = delegate->order_enodes(current.value, features);
                auto it = std::find(custom_order.begin(), custom_order.end(), chosen_sel);
                if (it != custom_order.end()) {
                    iteration_index = static_cast<uint32_t>(std::distance(custom_order.begin(), it));
                }
            }

            ENodeId enode_id = enodes[chosen_sel];
            const ENode &node = egraph.getENode(enode_id);

            if (!skip_increment && iteration_index + 1 < enodes.size())
            {
                next_sel[current] = iteration_index + 1;

                selection_map.erase(current);

                if (enodes.size() <= iteration_index + 2)
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
                if (delegate) delegate->pop_state();
                if (has_options[current.value])
                {
                    has_options[current.value] = false;
                    active_options--;
                }
            }
        }
    }
};