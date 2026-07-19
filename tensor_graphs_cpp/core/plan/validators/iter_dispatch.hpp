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

struct ParallelBuffer
{
    uint32_t idx = 0;    // index within its mem_space group
    MemSpace mem_space;  // which physical memory this buffer lives in
    uint64_t size = 0;   // bytes
    float start = 0.0f;  // birth time (parallel schedule)
    float end = 0.0f;    // death time (parallel schedule)
    int64_t offset = -1; // assigned byte offset, -1 = unallocated
};

// Interval overlap test
bool overlapsBuf(const ParallelBuffer &a, const ParallelBuffer &b)
{
    ParallelBuffer x = a, y = b;
    if (y.start < x.start)
        std::swap(x, y);
    return y.start < x.end;
}

void get_births(
    const std::vector<uint32_t> &ordered,
    const EGraph &egraph,
    const std::unordered_map<uint32_t, uint32_t> &selection_map,
    const std::vector<ENodeInfo> &enodeInfos,
    std::unordered_map<uint32_t, float> &birth_times,
    std::unordered_map<uint32_t, float> &engine_finish)
{
    for (uint32_t eclass : ordered)
    {
        uint32_t sel = selection_map.at(eclass);
        ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
        const ENode &node = egraph.getENode(enode_id);
        float cost = enodeInfos[enode_id.value].cost;

        float children_finish = 0.0f;
        for (EClassId child : egraph.getENode(enode_id).getChildren())
        {
            child = egraph.findConst(child);
            uint32_t child_sel = selection_map.at(child);
            uint32_t child_enode_id = egraph.getEClass(child).enodes[child_sel];
            const ENode &child_node = egraph.getENodes()[child_enode_id];
            uint32_t child_engine = child_node.getEngine().idx;

            auto it = engine_finish.find(child_engine);
            float child_finish = (it != engine_finish.end()) ? it->second : 0.0f;
            children_finish = std::max(children_finish, child_finish);
        }

        float engine_free = 0.0f;
        auto it = engine_finish.find(node.getEngine().idx);
        if (it != engine_finish.end())
            engine_free = it->second;

        float birth = std::max(children_finish, engine_free);
        birth_times[eclass] = birth;
        engine_finish[node.getEngine().idx] = birth + cost;
    }
}

// get_deaths — ported from bufferize.get_deaths().
//   node.death = max(node.birth + node.cost,
//                    max(consumer.birth + consumer.cost for consumer in ordered[j>i] if node in consumer.children))
static void get_deaths(
    const std::vector<uint32_t> &ordered,
    const EGraph &egraph,
    const std::unordered_map<uint32_t, uint32_t> &selection_map,
    const std::vector<ENodeInfo> &enodeInfos,
    const PrecompData &precomp,
    const std::unordered_map<uint32_t, float> &birth_times,
    std::unordered_map<uint32_t, float> &death_times)
{
    for (size_t i = 0; i < ordered.size(); ++i)
    {
        uint32_t node_eclass = ordered[i];
        uint32_t sel = selection_map.at(node_eclass);
        uint32_t enode_id = egraph.getEClass(node_eclass).enodes[sel];
        float cost = enodeInfos[enode_id].cost;

        float death = birth_times.at(node_eclass) + cost;
        for (size_t j = i + 1; j < ordered.size(); ++j)
        {
            uint32_t other_eclass = ordered[j];
            uint32_t other_sel = selection_map.at(other_eclass);
            uint32_t other_enode_id = egraph.getEClass(other_eclass).enodes[other_sel];

            bool is_consumed = false;
            for (uint32_t child : precomp.enode_canon_children[other_enode_id])
            {
                if (child == node_eclass)
                {
                    is_consumed = true;
                    break;
                }
            }
            if (is_consumed)
            {
                float other_cost = enodeInfos[other_enode_id].cost;
                death = std::max(death, birth_times.at(other_eclass) + other_cost);
            }
        }
        death_times[node_eclass] = death;
    }
}

// bufferize — ported from bufferize.bufferize().
//   Skips nodes whose mem_space.handle_type == STORAGE (we don't allocate storage).
//   Returns the list of buffers; also returns engine_finish via out-param so the
//   caller can compute the critical-path cost.
static std::vector<ParallelBuffer> bufferize(
    const std::vector<uint32_t> &ordered,
    const EGraph &egraph,
    const std::unordered_map<uint32_t, uint32_t> &selection_map,
    const std::vector<ENodeInfo> &enodeInfos,
    const PrecompData &precomp,
    std::unordered_map<uint32_t, float> &engine_finish_out)
{
    std::unordered_map<uint32_t, float> birth_times;
    std::unordered_map<uint32_t, float> death_times;

    get_births(ordered, egraph, selection_map, enodeInfos, precomp, birth_times, engine_finish_out);
    get_deaths(ordered, egraph, selection_map, enodeInfos, precomp, birth_times, death_times);

    std::vector<ParallelBuffer> buffers;
    buffers.reserve(ordered.size());
    for (uint32_t eclass : ordered)
    {
        uint32_t sel = selection_map.at(eclass);
        uint32_t enode_id = egraph.getEClass(eclass).enodes[sel];
        const ENode &node = egraph.getENodes()[enode_id];

        if (node.getMemSpace().type == HandleType::STORAGE)
            continue; // we don't control storage

        ParallelBuffer buf;
        buf.idx = static_cast<uint32_t>(buffers.size());
        buf.mem_space = node.getMemSpace();
        buf.size = getSizeBytes(node.getShape(), node.getDType());
        buf.start = birth_times.at(eclass);
        buf.end = death_times.at(eclass);
        buf.offset = -1;
        buffers.push_back(std::move(buf));
    }
    return buffers;
}

// malloc_recursive — ported from malloc.malloc().
//   Tries each unallocated buffer at the lowest valid offset, with four pruning rules:
//     1. offset non-monotonic  (offset_i < offset_max)
//     2. index non-monotonic   (unallocated[i].idx < idx_max at offset_i)
//     3. dominated             (offset_i >= h_min)
//     4. exceeds mem cap       (offset_i + size > mem_cap)
//   Returns true iff a valid full allocation was found.
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
        for (size_t i = 0; i < unallocated.size(); ++i)
        {
            int64_t offset_max = 0;
            for (size_t j = 0; j < allocated.size(); ++j)
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

    for (size_t i = 0; i < unallocated.size(); ++i)
    {
        int64_t offset_i = 0;
        int64_t offset_max = 0;
        for (size_t j = 0; j < allocated.size(); ++j)
        {
            if (overlapsBuf(unallocated[i], allocated[j]))
                offset_i = std::max(offset_i,
                                    allocated[j].offset + static_cast<int64_t>(allocated[j].size));
            offset_max = std::max(offset_max, allocated[j].offset);
        }
        if (offset_i < offset_max)
            continue; // offset non-monotonic

        uint32_t idx_max = 0;
        for (size_t j = 0; j < allocated.size(); ++j)
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
        for (size_t j = 0; j < unallocated.size(); ++j)
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
    // iter_dispatch_orders — ported from iter_dispatch.iter_dispatch_orders().
    //   Enumerates every valid topological order of the selection-induced subgraph
    //   and invokes `callback(order)` for each. If `callback` returns true, the
    //   generator stops early.
    //
    //   Algorithm: depth-first descent with backtracking. At each position we
    //   compute the ready set (nodes whose children are all already in `ordered`),
    //   and try them in order. When a position is exhausted we ascend (pop the
    //   last node, forget the choice at the now-vacated position) and try the
    //   next choice at the previous position.
    static void iter_dispatch_orders(
        const EGraph &egraph,
        const std::unordered_map<uint32_t, uint32_t> &selection_map,
        const PrecompData &precomp,
        std::function<bool(const std::vector<uint32_t> &)> callback)
    {
        auto get_ready = [&](const std::unordered_set<uint32_t> &remaining) -> std::vector<uint32_t>
        {
            std::vector<uint32_t> ready;
            for (uint32_t node : remaining)
            {
                uint32_t sel = selection_map.at(node);
                ENodeId enode_id = egraph.getEClass(node).enodes[sel];
                bool node_ready = true;
                for (uint32_t child : precomp.enode_canon_children[enode_id])
                {
                    if (remaining.count(child))
                    {
                        node_ready = false;
                        break;
                    }
                }
                if (node_ready)
                    ready.push_back(node);
            }
            return ready;
        };

        std::unordered_set<uint32_t> remaining;
        remaining.reserve(selection_map.size());
        for (const auto &kv : selection_map)
            remaining.insert(kv.first);

        std::vector<uint32_t> ordered;
        ordered.reserve(selection_map.size());
        std::unordered_map<uint32_t, uint32_t> selection_at_pos; // pos -> choice idx

        auto ascend = [&]() -> bool
        {
            // forget the choice at the position we're about to vacate (i.e. position len(ordered))
            selection_at_pos.erase(static_cast<uint32_t>(ordered.size()));
            if (ordered.empty())
                return false;
            uint32_t last = ordered.back();
            ordered.pop_back();
            remaining.insert(last);
            return true;
        };

        while (true)
        {
            while (true)
            {
                auto ready = get_ready(remaining);
                if (ready.empty())
                    return; // shouldn't happen if no cycles; defensive

                uint32_t pos = static_cast<uint32_t>(ordered.size());
                uint32_t choice = 0;
                auto it = selection_at_pos.find(pos);
                if (it != selection_at_pos.end())
                    choice = it->second + 1;

                if (choice < ready.size())
                {
                    selection_at_pos[pos] = choice;
                    uint32_t node = ready[choice];
                    ordered.push_back(node);
                    remaining.erase(node);
                }
                else
                {
                    if (ordered.empty())
                        return;
                    ascend();
                }
                if (remaining.empty())
                    break;
            }

            if (callback(ordered))
                return;

            if (!ascend())
                return;
        }
    }

    bool validate()
    {
        bool found_valid = false;

        iter_dispatch_orders(egraph, selection_map, precomp,
                             [&](const std::vector<uint32_t> &ordered) -> bool
                             {
                                 // --- bufferize: parallel schedule + per-node lifetimes
                                 std::unordered_map<uint32_t, float> engine_finish;
                                 std::vector<ParallelBuffer> buffers =
                                     bufferize(ordered, egraph, selection_map, enodeInfos, precomp, engine_finish);

                                 // --- parallel critical-path cost = max(engine_finish.values())
                                 float critical_path = 0.0f;
                                 for (const auto &kv : engine_finish)
                                     critical_path = std::max(critical_path, kv.second);
                                 float current_cost = critical_path;

                                 // prune: worse than best known
                                 if (best_cost != TGConstants::INF && current_cost >= best_cost)
                                     return false; // try next order

                                 // --- group buffers by mem_space.idx (per iter_dispatch/algos/malloc.py)
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
                                                        ? mem_caps[mem_idx]
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
                                     if (current_cost < best_cost)
                                     {
                                         best_cost = current_cost;
                                         best_selection_map = selection_map;
                                         std::cout << "[Planner.extractBest] new best (parallel) cost: "
                                                   << std::to_string(best_cost)
                                                   << ", #nodes=" << ordered.size() << std::endl;
                                     }
                                     found_valid = true;
                                     if (stopOnFirstValid)
                                         return true; // stop iterating orders
                                 }
                                 return false; // continue to next order
                             });

        if (found_valid && stopOnFirstValid)
            break;
        if (!found_valid)
        {
            valid = false;
            reason = "OOM"; // no valid (order, allocation) pair found for this selection_map
        }
    }
};