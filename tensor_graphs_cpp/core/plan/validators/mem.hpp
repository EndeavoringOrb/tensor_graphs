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
#include "core/plan/extractor.hpp"
#include "core/plan/validators/validator.hpp"
#include "core/common/constants.hpp"

// Interval overlap test
bool overlapsBuf(const ParallelBuffer &a, const ParallelBuffer &b)
{
    ParallelBuffer x = a, y = b;
    if (y.start < x.start)
        std::swap(x, y);
    return y.start < x.end;
}

void get_births(
    const std::vector<EClassId> &ordered,
    const EGraph &egraph,
    const std::unordered_map<EClassId, uint32_t> &selection_map,
    const std::vector<ENodeInfo> &enodeInfos,
    std::unordered_map<EClassId, float> &birth_times,
    std::unordered_map<uint32_t, float> &engine_finish)
{
    for (EClassId eclass : ordered)
    {
        uint32_t sel = selection_map.at(eclass);
        ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
        const ENode &node = egraph.getENode(enode_id);
        float cost = enodeInfos[enode_id.value].cost;

        // 1. Calculate max finish time among all child engines
        float children_finish = 0.0f;
        for (EClassId child : node.getChildren())
        {
            child = egraph.findConst(child);
            uint32_t child_sel = selection_map.at(child);
            ENodeId child_enode_id = egraph.getEClass(child).enodes[child_sel];
            const ENode &child_node = egraph.getENode(child_enode_id);

            for (const auto &engine : child_node.getEngines())
            {
                auto it = engine_finish.find(engine.idx);
                float child_finish = (it != engine_finish.end()) ? it->second : 0.0f;
                children_finish = std::max(children_finish, child_finish);
            }
        }

        // 2. Calculate when all engines required by the current node become free
        float engine_free = 0.0f;
        for (const auto &engine : node.getEngines())
        {
            auto it = engine_finish.find(engine.idx);
            if (it != engine_finish.end())
            {
                engine_free = std::max(engine_free, it->second);
            }
        }

        // 3. Compute birth time and update finish times for all node engines
        float birth = std::max(children_finish, engine_free);
        birth_times[eclass] = birth;

        for (const auto &engine : node.getEngines())
        {
            engine_finish[engine.idx] = birth + cost;
        }
    }
}

static void get_deaths(
    const std::vector<EClassId> &ordered,
    const EGraph &egraph,
    const std::unordered_map<EClassId, uint32_t> &selection_map,
    const std::vector<ENodeInfo> &enodeInfos,
    const std::unordered_map<EClassId, float> &birth_times,
    std::unordered_map<EClassId, float> &death_times)
{
    for (uint64_t i = 0; i < ordered.size(); ++i)
    {
        EClassId node_eclass = ordered[i];
        uint32_t sel = selection_map.at(node_eclass);
        ENodeId enode_id = egraph.getEClass(node_eclass).enodes[sel];
        float cost = enodeInfos[enode_id.value].cost;

        float death = birth_times.at(node_eclass) + cost;
        for (uint64_t j = i + 1; j < ordered.size(); ++j)
        {
            EClassId other_eclass = ordered[j];
            uint32_t other_sel = selection_map.at(other_eclass);
            ENodeId other_enode_id = egraph.getEClass(other_eclass).enodes[other_sel];
            const ENode &other_node = egraph.getENode(other_enode_id);

            bool is_consumed = false;
            for (EClassId child : other_node.getChildren())
            {
                if (egraph.findConst(child) == node_eclass)
                {
                    is_consumed = true;
                    break;
                }
            }
            if (is_consumed)
            {
                float other_cost = enodeInfos[other_enode_id.value].cost;
                death = std::max(death, birth_times.at(other_eclass) + other_cost);
            }
        }
        death_times[node_eclass] = death;
    }
}

// Search for bufferize() function and update its loop
static std::vector<ParallelBuffer> bufferize(
    const std::vector<EClassId> &ordered,
    const EGraph &egraph,
    const std::unordered_map<EClassId, uint32_t> &selection_map,
    const std::vector<ENodeInfo> &enodeInfos,
    std::unordered_map<uint32_t, float> &engine_finish_out)
{
    std::unordered_map<EClassId, float> birth_times;
    std::unordered_map<EClassId, float> death_times;

    get_births(ordered, egraph, selection_map, enodeInfos, birth_times, engine_finish_out);
    get_deaths(ordered, egraph, selection_map, enodeInfos, birth_times, death_times);

    std::vector<ParallelBuffer> buffers;
    buffers.reserve(ordered.size());
    for (EClassId eclass : ordered)
    {
        uint32_t sel = selection_map.at(eclass);
        ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
        const ENode &node = egraph.getENode(enode_id);

        if (node.getMemSpace().type == HandleType::STORAGE)
            continue;

        ParallelBuffer buf;
        buf.idx = static_cast<uint32_t>(buffers.size());
        buf.eclass_val = eclass.value;
        buf.mem_space = node.getMemSpace();
        buf.size = getSizeBytes(node.getShape(), node.getDType());
        buf.start = birth_times.at(eclass);
        buf.end = death_times.at(eclass);
        buf.offset = -1;
        buffers.push_back(std::move(buf));
    }
    return buffers;
}

static bool malloc_recursive(
    uint64_t mem_cap,
    std::vector<ParallelBuffer> &unallocated,
    std::vector<ParallelBuffer> &allocated)
{
    if (unallocated.empty())
        return true;

    auto get_min_height = [&]() -> int64_t
    {
        int64_t min_height = std::numeric_limits<int64_t>::max();
        for (uint64_t i = 0; i < unallocated.size(); ++i)
        {
            int64_t offset_max = 0;
            for (uint64_t j = 0; j < allocated.size(); ++j)
            {
                if (overlapsBuf(unallocated[i], allocated[j]))
                    offset_max = std::max(offset_max,
                                          allocated[j].offset + static_cast<int64_t>(allocated[j].size));
            }
            int64_t height = offset_max + static_cast<int64_t>(unallocated[i].size);
            min_height = std::min(min_height, height);
        }
        return min_height;
    };

    for (uint64_t i = 0; i < unallocated.size(); ++i)
    {
        int64_t offset_i = 0;
        int64_t offset_max = 0;
        for (uint64_t j = 0; j < allocated.size(); ++j)
        {
            if (overlapsBuf(unallocated[i], allocated[j]))
                offset_i = std::max(offset_i,
                                    allocated[j].offset + static_cast<int64_t>(allocated[j].size));
            offset_max = std::max(offset_max, allocated[j].offset);
        }
        if (offset_i < offset_max)
            continue; // offset non-monotonic

        uint32_t idx_max = 0;
        for (uint64_t j = 0; j < allocated.size(); ++j)
        {
            if (allocated[j].offset == offset_i)
                idx_max = std::max(idx_max, allocated[j].idx);
        }
        if (unallocated[i].idx < idx_max)
            continue; // index non-monotonic

        int64_t h_min = get_min_height();
        if (offset_i >= h_min)
            continue; // dominated

        if (mem_cap != std::numeric_limits<uint64_t>::max() &&
            static_cast<uint64_t>(offset_i) + unallocated[i].size > mem_cap)
            continue; // exceeds mem cap

        ParallelBuffer buf = unallocated[i];
        buf.offset = offset_i;

        std::vector<ParallelBuffer> new_unallocated;
        new_unallocated.reserve(unallocated.size() - 1);
        for (uint64_t j = 0; j < unallocated.size(); ++j)
            if (j != i)
                new_unallocated.push_back(unallocated[j]);

        std::vector<ParallelBuffer> new_allocated = allocated;
        new_allocated.push_back(buf);

        if (malloc_recursive(mem_cap, new_unallocated, new_allocated))
            return true;
    }
    return false;
}

struct MemValidator : public ISelectionValidator
{
    const EGraph &egraph;
    const std::vector<ENodeInfo> &enodeInfos;
    const std::unordered_map<uint32_t, uint64_t> &mem_caps;
    bool stopOnFirstValid;

    // Iteration State
    std::unordered_set<EClassId> remaining;
    std::vector<EClassId> ordered;
    std::unordered_map<uint32_t, uint32_t> selection_at_pos;
    bool is_done = false;
    bool first_yield = true;

    MemValidator(const EGraph &_egraph,
                 const std::vector<ENodeInfo> &_enodeInfos,
                 const std::unordered_map<uint32_t, uint64_t> &_mem_caps,
                 bool _stopOnFirstValid = true)
        : egraph(_egraph), enodeInfos(_enodeInfos), mem_caps(_mem_caps), stopOnFirstValid(_stopOnFirstValid) {}

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

    bool getNextDispatchOrder(const std::unordered_map<EClassId, uint32_t> &selection_map, std::vector<EClassId> &out_order)
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

        while (true)
        {
            while (true)
            {
                std::vector<EClassId> ready = get_ready(selection_map);

                // Safety check for dependency cycles (though CycleValidator handles most of this)
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
            return true;
        }
    }

    bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map, std::string &reason) override
    {
        initOrderState(selection_map);

        bool found_valid = false;
        float best_cost = std::numeric_limits<float>::infinity();

        std::vector<EClassId> current_order;
        while (getNextDispatchOrder(selection_map, current_order))
        {
            // --- bufferize: parallel schedule + per-node lifetimes
            std::unordered_map<uint32_t, float> engine_finish;
            std::vector<ParallelBuffer> buffers = bufferize(current_order, egraph, selection_map, enodeInfos, engine_finish);

            // --- parallel critical-path cost = max(engine_finish.values())
            float current_cost = 0.0f;
            for (const auto &kv : engine_finish)
            {
                current_cost = std::max(current_cost, kv.second);
            }

            // prune: worse than best known
            if (current_cost >= best_cost)
            {
                continue; // try next order
            }

            // --- group buffers by mem_space.idx
            std::unordered_map<uint32_t, std::vector<ParallelBuffer>> buf_by_mem_idx;
            for (auto &buf : buffers)
            {
                buf.idx = static_cast<uint32_t>(buf_by_mem_idx[buf.mem_space.idx].size());
                buf_by_mem_idx[buf.mem_space.idx].push_back(buf);
            }

            // --- malloc: try to assign offsets within mem_cap for each mem_idx
            bool alloc_ok = true;
            for (auto &kv : buf_by_mem_idx)
            {
                uint32_t mem_idx = kv.first;
                auto &bufs = kv.second;
                uint64_t cap = mem_caps.count(mem_idx)
                                   ? mem_caps.at(mem_idx)
                                   : std::numeric_limits<uint64_t>::max();
                std::vector<ParallelBuffer> allocated;
                if (!malloc_recursive(cap, bufs, allocated))
                {
                    alloc_ok = false;
                    break;
                }
            }

            if (alloc_ok)
            {
                best_cost = current_cost;
                found_valid = true;

                if (stopOnFirstValid)
                {
                    return true; // stop iterating orders, early exit
                }
            }
        }

        if (!found_valid)
        {
            reason = "OOM"; // no valid (order, allocation) pair found for this selection_map
            return false;
        }

        return true;
    }
};