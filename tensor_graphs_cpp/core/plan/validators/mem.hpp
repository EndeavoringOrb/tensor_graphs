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
    std::unordered_map<uint32_t, float> &engine_finish_out,
    std::unordered_map<EClassId, BufferId> &eclass_to_buf)
{
    ProgressTimer t = ProgressTimer(0, "bufferize ", false, true);
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

        BufferId buf_id = BufferId{(uint32_t)buffers.size()};
        eclass_to_buf[eclass] = buf_id;
        ParallelBuffer buf = {buf_id, node.getMemSpace(), getSizeBytes(node.getShape(), node.getDType()), birth_times.at(eclass), death_times.at(eclass), -1};
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

        BufferId id_max = BufferId{0};
        for (uint64_t j = 0; j < allocated.size(); ++j)
        {
            if (allocated[j].offset == offset_i)
                id_max = std::max(id_max, allocated[j].id);
        }
        if (unallocated[i].id < id_max)
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

    MemValidator(const EGraph &_egraph,
                 const std::vector<ENodeInfo> &_enodeInfos,
                 const std::unordered_map<uint32_t, uint64_t> &_mem_caps)
        : egraph(_egraph), enodeInfos(_enodeInfos), mem_caps(_mem_caps) {}

    bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<EClassId> &order,
                  std::vector<ParallelBuffer> &buffers,
                  std::unordered_map<EClassId, BufferId> &eclass_to_buf,
                  float &cost, std::string &reason, bool &updated_buffers, bool &updated_cost) override
    {
        ProgressTimer t = ProgressTimer(0, "validate ", false, true);
        // bufferize: parallel schedule + per-node lifetimes
        std::unordered_map<uint32_t, float> engine_finish;
        std::vector<ParallelBuffer> unallocated_buffers = bufferize(order, egraph, selection_map, enodeInfos, engine_finish, eclass_to_buf);

        // parallel critical-path cost = max(engine_finish.values())
        float current_cost = 0.0f;
        for (const auto &kv : engine_finish)
        {
            current_cost = std::max(current_cost, kv.second);
        }
        cost = current_cost;
        updated_cost = true;

        // group buffers by mem_space.idx
        std::unordered_map<uint32_t, std::vector<ParallelBuffer>> buf_by_mem_idx;
        for (auto &buf : buffers)
        {
            if (buf.mem_space.type == HandleType::STORAGE)
                continue;
            buf_by_mem_idx[buf.mem_space.idx].push_back(buf);
        }

        // malloc: try to assign offsets within mem_cap for each mem_idx
        buffers.clear();
        buffers.reserve(unallocated_buffers.size());
        bool alloc_ok = true;
        for (auto &kv : buf_by_mem_idx)
        {
            uint32_t mem_idx = kv.first;
            auto &bufs = kv.second;
            uint64_t cap = mem_caps.count(mem_idx)
                               ? mem_caps.at(mem_idx)
                               : std::numeric_limits<uint64_t>::max();
            std::vector<ParallelBuffer> allocated;
            ProgressTimer t = ProgressTimer(0, "malloc_recursive mem_idx=" + std::to_string(mem_idx) + ", n_bufs=" + std::to_string(bufs.size()) + " ", false, true);
            if (!malloc_recursive(cap, bufs, allocated))
            {
                alloc_ok = false;
                break;
            }
            buffers.insert(buffers.end(),
                           std::make_move_iterator(allocated.begin()),
                           std::make_move_iterator(allocated.end()));
        }
        updated_buffers = true;

        if (alloc_ok)
        {
            return true;
        }
        reason = "OOM";
        return false;
    }
};