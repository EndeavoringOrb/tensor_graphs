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
#include "core/plan/extractor.hpp"
#include "core/plan/validators/validator.hpp"
#include "core/rewrite.hpp"
#include "core/shapes.hpp"
#include "core/types.hpp"

// Interval overlap test
bool overlapsBuf(const ParallelBuffer &a, const ParallelBuffer &b)
{
    ParallelBuffer x = a, y = b;
    if (y.start < x.start)
        std::swap(x, y);
    return y.start <= x.end;
}

float get_cost(const std::vector<EClassId> &ordered, const EGraph &egraph,
               const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<ENodeInfo> &enodeInfos)
{
    std::unordered_map<EClassId, float> birth_times;
    std::unordered_map<uint32_t, float> engine_finish;
    for (EClassId eclass : ordered)
    {
        uint32_t sel = selection_map.at(eclass);
        ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
        const ENode &node = egraph.getENode(enode_id);

        if (node.getOpType() == OpType::INPUT || node.getOpType() == OpType::CACHE)
        {
            birth_times[eclass] = 0.0f;
            continue;
        }

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

    float cost = 0.0f;
    for (const auto &kv : engine_finish)
    {
        cost = std::max(cost, kv.second);
    }
    return cost;
}

inline EClassId resolve_view_alias(EClassId id, const EGraph &egraph,
                                   const std::unordered_map<EClassId, uint32_t> &selection_map,
                                   const std::vector<ENodeInfo> &enodeInfos)
{
    EClassId curr = egraph.findConst(id);
    while (true)
    {
        auto sel_it = selection_map.find(curr);
        if (sel_it == selection_map.end())
            break;
        uint32_t sel = sel_it->second;
        ENodeId enode_id = egraph.getEClass(curr).enodes[sel];
        if (enodeInfos[enode_id.value].is_view)
        {
            const ENode &node = egraph.getENode(enode_id);
            if (!node.getChildren().empty())
            {
                curr = egraph.findConst(node.getChildren()[0]);
            }
            else
            {
                break;
            }
        }
        else
        {
            break;
        }
    }
    return curr;
}

// 2. Refactor `get_deaths`
static void get_deaths(const std::vector<EClassId> &ordered, const EGraph &egraph,
                       const std::unordered_map<EClassId, uint32_t> &selection_map,
                       const std::vector<ENodeInfo> &enodeInfos, const std::unordered_map<EClassId, float> &birth_times,
                       std::unordered_map<EClassId, float> &death_times)
{
    for (uint64_t i = 0; i < ordered.size(); ++i)
    {
        EClassId node_eclass = ordered[i];
        uint32_t sel = selection_map.at(node_eclass);
        ENodeId enode_id = egraph.getEClass(node_eclass).enodes[sel];
        const ENode &node = egraph.getENode(enode_id);
        float cost = enodeInfos[enode_id.value].cost;

        if (node.getOpType() == OpType::INPUT || node.getOpType() == OpType::CACHE)
        {
            death_times[node_eclass] = std::numeric_limits<float>::infinity();
            continue;
        }

        float death = birth_times.at(node_eclass) + std::max(0.1f, cost); // TODO: this 0.1f is very hacky, do something better. maybe integer timing based on order?
        for (uint64_t j = i + 1; j < ordered.size(); ++j)
        {
            EClassId other_eclass = ordered[j];
            uint32_t other_sel = selection_map.at(other_eclass);
            ENodeId other_enode_id = egraph.getEClass(other_eclass).enodes[other_sel];
            const ENode &other_node = egraph.getENode(other_enode_id);

            bool is_consumed = false;
            for (EClassId child : other_node.getChildren())
            {
                if (resolve_view_alias(child, egraph, selection_map, enodeInfos) ==
                    resolve_view_alias(node_eclass, egraph, selection_map, enodeInfos))
                {
                    is_consumed = true;
                    break;
                }
            }
            if (is_consumed)
            {
                float other_cost = enodeInfos[other_enode_id.value].cost;
                death = std::max(death, birth_times.at(other_eclass) + std::max(0.1f, other_cost)); // TODO: use std::nextafter
            }
        }
        death_times[node_eclass] = death;
    }
}

// 3. Refactor `bufferize`
static std::vector<ParallelBuffer> bufferize(const std::vector<EClassId> &ordered, const EGraph &egraph,
                                             const std::unordered_map<EClassId, uint32_t> &selection_map,
                                             const std::vector<ENodeInfo> &enodeInfos,
                                             std::unordered_map<EClassId, BufferId> &eclass_to_buf)
{
    ProgressTimer t = ProgressTimer(0, "bufferize ", false, true);

    std::unordered_map<EClassId, uint32_t> birth_times;
    std::unordered_map<EClassId, uint32_t> death_times;
    for (uint32_t i = 0; i < ordered.size(); ++i)
    {
        EClassId eclass = ordered[i];
        birth_times[eclass] = i;
        death_times[eclass] = i + 1;
        uint32_t sel = selection_map.at(eclass);
        ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
        const ENode &node = egraph.getENode(enode_id);
        for (EClassId child : node.getChildren())
        {
            EClassId child_base = resolve_view_alias(child, egraph, selection_map, enodeInfos);
            death_times[child_base] = std::max(death_times[child_base], i);
        }
    }

    std::unordered_map<EClassId, EClassId> inplace_alias;
    auto get_inplace_alias = [&](EClassId id)
    {
        while (inplace_alias.count(id))
            id = inplace_alias.at(id);
        return id;
    };

    for (uint32_t i = 0; i < ordered.size(); ++i)
    {
        EClassId eclass = ordered[i];
        uint32_t sel = selection_map.at(eclass);
        ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
        const ENode &node = egraph.getENode(enode_id);

        if (node.getKernelId().value != 0)
        {
            const KernelEntry &kernel = KernelRegistry::get().getKernel(node.getKernelId());
            if (!kernel.safe_inplace_idxs.empty())
            {
                for (uint32_t idx : kernel.safe_inplace_idxs)
                {
                    if (idx < node.getChildren().size())
                    {
                        EClassId child = egraph.findConst(node.getChildren()[idx]);
                        EClassId child_base = resolve_view_alias(child, egraph, selection_map, enodeInfos);

                        if (death_times[child_base] == i)
                        {
                            uint32_t c_sel = selection_map.at(child_base);
                            const ENode &c_node = egraph.getENode(egraph.getEClass(child_base).enodes[c_sel]);
                            if (c_node.getOpType() != OpType::INPUT && c_node.getOpType() != OpType::CACHE)
                            {
                                uint64_t out_size = getSizeBytes(node.getShape(), node.getDType());
                                uint64_t in_size = getSizeBytes(c_node.getShape(), c_node.getDType());
                                if (out_size <= in_size)
                                {
                                    inplace_alias[eclass] = get_inplace_alias(child_base);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<ParallelBuffer> buffers;

    // Pass 1: resolve aliases and lifetimes
    for (uint32_t i = 0; i < ordered.size(); ++i)
    {
        EClassId eclass = ordered[i];
        uint32_t sel = selection_map.at(eclass);
        ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
        const ENodeInfo &info = enodeInfos[enode_id.value];

        EClassId base = eclass;
        if (info.is_view)
        {
            base = resolve_view_alias(eclass, egraph, selection_map, enodeInfos);
        }

        EClassId target_base = get_inplace_alias(base);

        if (target_base != eclass)
        {
            if (birth_times.count(target_base))
                birth_times[target_base] = std::min(birth_times[target_base], birth_times[base]);
            if (death_times.count(target_base))
                death_times[target_base] = std::max(death_times[target_base], death_times[base]);
        }
    }

    // Pass 2: assign BufferIds
    std::unordered_map<EClassId, BufferId> base_to_buf;
    for (EClassId eclass : ordered)
    {
        EClassId base = eclass;
        uint32_t sel = selection_map.at(eclass);
        ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
        if (enodeInfos[enode_id.value].is_view)
        {
            base = resolve_view_alias(eclass, egraph, selection_map, enodeInfos);
        }

        EClassId target_base = get_inplace_alias(base);

        if (base_to_buf.find(target_base) == base_to_buf.end())
        {
            BufferId buf_id = BufferId{(uint32_t)buffers.size()};
            base_to_buf[target_base] = buf_id;

            uint32_t base_sel = selection_map.at(target_base);
            ENodeId base_enode_id = egraph.getEClass(target_base).enodes[base_sel];
            const ENode &base_node = egraph.getENode(base_enode_id);

            uint64_t size_bytes = getSizeBytes(base_node.getShape(), base_node.getDType());
            if (size_bytes == 0)
            {
                Error::throw_err("empty node");
            }
            // Align to 4096 bytes to prevent OpenCL CL_MISALIGNED_SUB_BUFFER_OFFSET (-13)
            size_bytes = (size_bytes + 4095) & ~4095ULL;

            ParallelBuffer buf = {
                buf_id, base_node.getMemSpace(), size_bytes, birth_times.at(target_base), death_times.at(target_base), -1};
            buffers.push_back(std::move(buf));
        }

        eclass_to_buf[eclass] = base_to_buf[target_base];
    }
    return buffers;
}

// Just a simpler reference for how malloc works
static bool malloc_recursive(uint64_t mem_cap, std::vector<ParallelBuffer> &unallocated,
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
                    offset_max = std::max(offset_max, allocated[j].offset + static_cast<int64_t>(allocated[j].size));
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
                offset_i = std::max(offset_i, allocated[j].offset + static_cast<int64_t>(allocated[j].size));
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

static bool malloc(uint64_t mem_cap, const std::vector<ParallelBuffer> &unallocated,
                   std::vector<ParallelBuffer> &allocated)
{
    if (unallocated.empty())
        return true;
    int N = static_cast<int>(unallocated.size());

    std::vector<int64_t> unallocated_sizes(N);
    for (int i = 0; i < N; ++i)
        unallocated_sizes[i] = unallocated[i].size;

    // PRE-COMPUTATION: Adjacency List.
    // Instead of calling overlapsBuf() in O(N^2) inner loops during the search,
    // we precompute which buffers overlap in time.
    std::vector<std::vector<int>> adj(N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (i != j && overlapsBuf(unallocated[i], unallocated[j]))
            {
                adj[i].push_back(j);
            }
        }
    }

    // IN-PLACE PARTITIONING: `avail` array.
    // Replaces creating a new std::vector<ParallelBuffer> at every recursion
    // depth. Elements from 0 to k-1 are "allocated". Elements from k to N-1 are
    // "unallocated".
    std::vector<int> avail(N);
    std::iota(avail.begin(), avail.end(), 0);

    // FLATTENED RECURSION STACK:
    // `k` is the current recursion depth (number of placed buffers).
    // `state[k]` remembers where we were in the unallocated loop for depth `k`.
    std::vector<int> state(N, 0);
    state[0] = 0;

    std::vector<int> order(N, 0);                     // Which buffer index was chosen at depth k
    std::vector<int64_t> chosen_offset(N, 0);         // The memory offset assigned at depth k
    std::vector<int64_t> global_offset_max(N + 1, 0); // Max offset used so far (monotonicity constraint)

    // INCREMENTAL OFFSETS ("Push" calculation):
    // Instead of looking back at all allocated buffers to calculate our offset,
    // when a buffer is assigned, it "pushes" its end_boundary to all overlapping
    // neighbors.
    std::vector<int64_t> current_offsets(N, 0);

    // UNDO LOG (Write-Ahead Log):
    // When we push boundaries to `current_offsets`, we save the old values here.
    // When we backtrack, we pop from this log to restore the state instantly.
    struct Backup
    {
        int j;
        int64_t old_val;
    };
    std::vector<Backup> trail;
    trail.reserve(N * 50);
    std::vector<size_t> trail_starts(N + 1,
                                     0); // Where the log started for depth k

    int k = 0;
    while (k >= 0)
    {
        // Base Case: All buffers successfully placed!
        if (k == N)
        {
            for (int d = 0; d < N; ++d)
            {
                ParallelBuffer buf = unallocated[order[d]];
                buf.offset = chosen_offset[d];
                allocated.push_back(buf);
            }
            return true;
        }

        bool advanced = false;

        // DOMINATION CHECK (h_min):
        // Calculate the minimum upper bound for currently available buffers.
        int64_t h_min = std::numeric_limits<int64_t>::max();
        for (int idx = k; idx < N; ++idx)
        {
            int i = avail[idx];
            int64_t h = current_offsets[i] + unallocated_sizes[i];
            if (h < h_min)
                h_min = h;
        }

        // The inner loop simulating: for (uint64_t i = 0; i < unallocated.size();
        // ++i)
        while (state[k] < N)
        {
            int idx = state[k]; // Index in the `avail` array
            state[k]++;
            int i = avail[idx]; // The actual buffer index

            int64_t offset_i = current_offsets[i];

            // CONSTRAINT: Monotonic offset growth
            // (Equivalent to `if (offset_i < offset_max) continue;`)
            if (offset_i < global_offset_max[k])
                continue;

            // SYMMETRY BREAKING (id_max):
            // If multiple buffers start at the exact same memory offset, we force
            // them to be picked in ascending order of their IDs to prevent duplicate
            // search paths. Because offsets strictly grow (enforced above), we only
            // need to look at the immediately preceding allocations in O(1) time
            // rather than O(N).
            BufferId id_max = BufferId{0};
            for (int d = k - 1; d >= 0; --d)
            {
                if (chosen_offset[d] == offset_i)
                {
                    id_max = std::max(id_max, unallocated[order[d]].id);
                }
                else
                {
                    break; // Offsets dropped, we can stop looking immediately!
                }
            }
            if (unallocated[i].id < id_max)
                continue;

            // CONSTRAINT: Domination
            if (offset_i >= h_min)
                continue;

            // CONSTRAINT: Memory limits
            if (mem_cap != std::numeric_limits<uint64_t>::max() &&
                static_cast<uint64_t>(offset_i) + unallocated_sizes[i] > mem_cap)
                continue;

            // --- ALL CHECKS PASSED. APPLY STATE ---

            // 1. Swap the chosen buffer into the "allocated" portion of the `avail`
            // array (O(1) remove)
            std::swap(avail[k], avail[idx]);
            order[k] = i;
            chosen_offset[k] = offset_i;
            global_offset_max[k + 1] = std::max(global_offset_max[k], offset_i);

            // 2. Push new memory requirements to overlapping, unallocated buffers
            trail_starts[k] = trail.size();
            int64_t new_end = offset_i + unallocated_sizes[i];

            for (int j : adj[i])
            {
                if (current_offsets[j] < new_end)
                {
                    trail.push_back({j, current_offsets[j]}); // Save for undo
                    current_offsets[j] = new_end;             // Apply new constraint
                }
            }

            advanced = true;
            break;
        }

        if (advanced)
        {
            // Dive deeper (Recursive Call)
            k++;
            if (k < N)
                state[k] = k; // Start loop after the allocated partition
        }
        else
        {
            // Backtrack (Return False)
            k--;
            if (k >= 0)
            {
                // Undo the O(1) remove
                int idx = state[k] - 1;
                std::swap(avail[k], avail[idx]);

                // Rollback memory constraints using the Undo Log
                size_t ts = trail_starts[k];
                while (trail.size() > ts)
                {
                    current_offsets[trail.back().j] = trail.back().old_val;
                    trail.pop_back();
                }
            }
        }
    }
    return false;
}

// 1. O(N log N) Sweep-line to check if it's mathematically impossible to fit in
// mem_cap
static bool check_peak_memory(const std::vector<ParallelBuffer> &bufs, uint64_t mem_cap,
                              BufferId &overflow // if failed due to mem, which buffer pushed mem over the edge
)
{
    if (mem_cap == std::numeric_limits<uint64_t>::max())
        return true;

    struct Event
    {
        uint32_t time;
        int type; // 0 for end, 1 for start
        int64_t size;
        BufferId buffer_id;
    };

    std::vector<Event> events;
    events.reserve(bufs.size() * 2);
    for (const auto &b : bufs)
    {
        events.push_back({b.start, 1, static_cast<int64_t>(b.size), b.id});
        events.push_back({b.end, 0, static_cast<int64_t>(b.size), b.id});
    }

    // Process Ends before Starts to correctly simulate memory release
    std::sort(events.begin(), events.end(), [](const Event &a, const Event &b)
              {
        if (a.time != b.time)
            return a.time < b.time;
        return a.type < b.type; });

    int64_t current_mem = 0;
    for (const auto &ev : events)
    {
        if (ev.type == 1)
        { // start
            current_mem += ev.size;
            if (current_mem > static_cast<int64_t>(mem_cap))
            {
                overflow = ev.buffer_id;
                return false;
            }
        }
        else
        { // end
            current_mem -= ev.size;
        }
    }
    return true;
}

// 2. O(N^2) Fast Heuristic Allocator: First-Fit Decreasing
static bool greedy_alloc(uint64_t mem_cap, const std::vector<ParallelBuffer> &unallocated,
                         std::vector<ParallelBuffer> &allocated,
                         BufferId &overflow // if failed due to mem, which
                                            // buffer pushed mem over the edge
)
{
    std::vector<ParallelBuffer> bufs = unallocated;

    // Heuristic: Place largest buffers first to minimize fragmentation
    std::sort(bufs.begin(), bufs.end(), [](const ParallelBuffer &a, const ParallelBuffer &b)
              {
                  if (a.size != b.size)
                      return a.size > b.size;
                  return a.id < b.id; // Deterministic tie-breaker
              });

    allocated.clear();
    allocated.reserve(bufs.size());

    for (auto &buf : bufs)
    {
        int64_t best_offset = 0;

        // Find all ALREADY PLACED buffers that overlap in TIME
        std::vector<const ParallelBuffer *> time_overlaps;
        for (const auto &alloc : allocated)
        {
            if (overlapsBuf(buf, alloc))
            {
                time_overlaps.push_back(&alloc);
            }
        }

        // Sort overlapping buffers by their memory offset ascending
        std::sort(time_overlaps.begin(), time_overlaps.end(),
                  [](const ParallelBuffer *a, const ParallelBuffer *b)
                  { return a->offset < b->offset; });

        // First-fit algorithm: push best_offset upwards if there's a memory
        // collision
        for (const auto *alloc : time_overlaps)
        {
            if (best_offset < alloc->offset + static_cast<int64_t>(alloc->size) &&
                best_offset + static_cast<int64_t>(buf.size) > alloc->offset)
            {
                best_offset = alloc->offset + alloc->size; // Move above the collision
            }
        }

        if (mem_cap != std::numeric_limits<uint64_t>::max() &&
            best_offset + static_cast<int64_t>(buf.size) > static_cast<int64_t>(mem_cap))
        {
            overflow = buf.id;
            return false; // Greedy failed to fit within mem_cap
        }

        buf.offset = best_offset;
        allocated.push_back(buf);
    }
    return true;
}

// 3. Rewritten function to bridge the optimizations
static bool malloc_by_time_components(uint64_t mem_cap, const std::vector<ParallelBuffer> &unallocated,
                                      std::vector<ParallelBuffer> &allocated, BufferId &overflow)
{
    if (unallocated.empty())
        return true;

    // 1. Sort buffers by start time (and then end time to be deterministic)
    std::vector<ParallelBuffer> sorted_bufs = unallocated;
    std::sort(sorted_bufs.begin(), sorted_bufs.end(), [](const ParallelBuffer &a, const ParallelBuffer &b)
              {
        if (a.start != b.start)
            return a.start < b.start;
        return a.end < b.end; });

    if (sorted_bufs.empty())
        return true;

    // OPTIMIZATION 1: Absolute strict lower bound check
    if (!check_peak_memory(sorted_bufs, mem_cap, overflow))
    {
        return false; // Mathematically impossible to fit; abort instantly
    }

    // OPTIMIZATION 2: Fast greedy allocator path (solves immediately most of
    // the time)
    std::vector<ParallelBuffer> comp_allocated;
    if (greedy_alloc(mem_cap, sorted_bufs, comp_allocated, overflow))
    {
        allocated.insert(allocated.end(), comp_allocated.begin(), comp_allocated.end());
        return true;
    }

    // OPTIMIZATION 3: Exact solver fallback with aggressive pruning ordering
    // Sorting by size descending forces the tree to hit conflict limits much
    // faster.
    std::sort(sorted_bufs.begin(), sorted_bufs.end(), [](const ParallelBuffer &a, const ParallelBuffer &b)
              {
            if (a.size != b.size)
                return a.size > b.size;
            return a.id < b.id; });

    if (!malloc(mem_cap, sorted_bufs, comp_allocated))
    {
        return false;
    }

    allocated.insert(allocated.end(), comp_allocated.begin(), comp_allocated.end());
    return true;
}

struct MemValidator : public ISelectionValidator
{
    const EGraph &egraph;
    const std::vector<ENodeInfo> &enodeInfos;
    const std::unordered_map<MemSpace, uint64_t> &mem_caps;
    // Map from canonical eclass id -> logical id (used to detect INPUT/CACHE
    // enodes that correspond to a logical node whose buffer was pre-allocated
    // outside of the Planner).
    const std::unordered_map<EClassId, LogicalId> &eclassToLogical;
    // Pre-allocated stable ParallelBuffers for INPUT/CACHE logical nodes,
    // keyed by LogicalId. These buffers are placed in a contiguous region
    // starting at offset 0 in their MemSpace, and the malloc solver is given
    // a reduced mem_cap so it places all other buffers *above* this region.
    const std::unordered_map<LogicalId, ParallelBuffer> &preallocatedBuffers;

    MemValidator(const EGraph &_egraph, const std::vector<ENodeInfo> &_enodeInfos,
                 const std::unordered_map<MemSpace, uint64_t> &_mem_caps,
                 const std::unordered_map<EClassId, LogicalId> &_eclassToLogical,
                 const std::unordered_map<LogicalId, ParallelBuffer> &_preallocatedBuffers)
        : egraph(_egraph), enodeInfos(_enodeInfos), mem_caps(_mem_caps), eclassToLogical(_eclassToLogical),
          preallocatedBuffers(_preallocatedBuffers)
    {
    }

    bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<EClassId> &order,
                  std::vector<ParallelBuffer> &buffers, std::unordered_map<EClassId, BufferId> &eclass_to_buf,
                  BufferId &overflow, float &cost, std::string &reason, bool &updated_buffers,
                  bool &updated_cost) override
    {
        ProgressTimer t = ProgressTimer(0, "validate ", false, true);
        cost = get_cost(order, egraph, selection_map, enodeInfos);
        updated_cost = true;
        // bufferize: parallel schedule + per-node lifetimes
        std::vector<ParallelBuffer> unallocated_buffers =
            bufferize(order, egraph, selection_map, enodeInfos, eclass_to_buf);

        // ------------------------------------------------------------------
        // Identify pre-allocated buffers (INPUT/CACHE enodes whose eclass has
        // a logical mapping AND for which a pre-allocated ParallelBuffer was
        // supplied by the Session). These buffers must NOT be re-allocated by
        // the malloc solver; their offset is fixed by the Session so that the
        // same logical node lands at the same byte offset across runs.
        // ------------------------------------------------------------------
        std::unordered_set<BufferId> preallocated_buf_ids;
        std::unordered_map<BufferId, ParallelBuffer> preallocated_overrides;
        // Per-MemSpace "high water mark" of the pre-allocated region.
        // Buffers are placed contiguously from offset 0 upward by the Session,
        // so max(offset+size) == total bytes reserved in that MemSpace.
        std::unordered_map<MemSpace, uint64_t> reserved_per_ms;
        for (EClassId eclass : order)
        {
            auto logicalIt = eclassToLogical.find(eclass);
            if (logicalIt == eclassToLogical.end())
                continue;

            uint32_t sel = selection_map.at(eclass);
            ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
            const ENode &node = egraph.getENode(enode_id);
            if (node.getOpType() != OpType::INPUT && node.getOpType() != OpType::CACHE)
                continue;

            auto preIt = preallocatedBuffers.find(logicalIt->second);
            if (preIt == preallocatedBuffers.end())
                continue;

            // The eclass' buffer id was assigned by bufferize() above. We use
            // that same id in the final `buffers` vector, but override the
            // offset with the pre-allocated value.
            BufferId buf_id = eclass_to_buf.at(eclass);
            preallocated_buf_ids.insert(buf_id);
            preallocated_overrides[buf_id] = preIt->second;

            const ParallelBuffer &pre = preIt->second;
            uint64_t extent = static_cast<uint64_t>(pre.offset) + pre.size;
            uint64_t &cur = reserved_per_ms[pre.mem_space];
            cur = std::max(cur, extent);
        }

        // group buffers by mem_space instead of mem_space.idx.
        // Pre-allocated buffers are excluded from the malloc input (the user's
        // "ignore any INPUT or CACHE where eclassToLogical.count(eclass_id) != 0"
        // rule). They will be added back to the final `buffers` vector below
        // with their pre-allocated offsets.
        std::unordered_map<MemSpace, std::vector<ParallelBuffer>> buf_by_mem_space;
        for (auto &buf : unallocated_buffers)
        {
            if (buf.mem_space.type == HandleType::STORAGE)
                continue;
            if (preallocated_buf_ids.count(buf.id))
                continue;
            buf_by_mem_space[buf.mem_space].push_back(buf);
        }

        // malloc: try to assign offsets within (reduced) mem_cap for each
        // mem_space. The malloc solver places buffers starting from offset 0,
        // so after it returns we shift every malloc'd buffer UP by
        // `reserved_per_ms[ms]` to leave the low-offset region for the
        // pre-allocated INPUT/CACHE buffers.
        buffers.clear();
        buffers.reserve(unallocated_buffers.size());

        // 1. STORAGE buffers: dynamic offset resolved at runtime, no malloc.
        for (auto &buf : unallocated_buffers)
        {
            if (buf.mem_space.type == HandleType::STORAGE)
            {
                buf.offset = 0; // Actual file offset is resolved dynamically in
                                // StorageBuffer::setupInput
                buffers.push_back(buf);
            }
        }

        // 2. Pre-allocated INPUT/CACHE buffers: use the Session-assigned offset.
        for (auto &buf : unallocated_buffers)
        {
            if (preallocated_buf_ids.count(buf.id))
            {
                ParallelBuffer pre = preallocated_overrides.at(buf.id);
                // Preserve bufferize()'s assigned id / lifetime / size; only
                // the offset comes from the pre-allocated buffer.
                buf.offset = pre.offset;
                buffers.push_back(buf);
            }
        }

        bool alloc_ok = true;
        std::string oom_reason = "OOM";

        for (auto &kv : buf_by_mem_space)
        {
            MemSpace ms = kv.first;
            auto &bufs = kv.second;
            uint64_t cap = mem_caps.count(ms) ? mem_caps.at(ms) : std::numeric_limits<uint64_t>::max();
            // Reduce mem_caps by the max(offset+size) of the pre-allocated
            // logical-node buffers in this MemSpace.
            uint64_t reserved = reserved_per_ms.count(ms) ? reserved_per_ms.at(ms) : 0;
            uint64_t reduced_cap =
                (cap == std::numeric_limits<uint64_t>::max()) ? cap : (cap > reserved ? cap - reserved : 0);

            std::vector<ParallelBuffer> allocated;
            ProgressTimer t2 = ProgressTimer(
                0,
                "malloc mem_space=(" + std::to_string(ms.idx) + "," + std::to_string((int)ms.type) +
                    "), n_bufs=" + std::to_string(bufs.size()) + ", reserved=" + std::to_string(reserved) + " ",
                false, true);
            if (!malloc_by_time_components(reduced_cap, bufs, allocated, overflow))
            {
                alloc_ok = false;
                oom_reason = "OOM:" + std::to_string(ms.idx) + ":" + std::to_string(static_cast<int>(ms.type));
                break;
            }

            // Shift malloc'd buffers above the pre-allocated region so they
            // do not collide with the INPUT/CACHE buffers placed at low offsets.
            for (auto &buf : allocated)
            {
                buf.offset += static_cast<int64_t>(reserved);
            }

            buffers.insert(buffers.end(), std::make_move_iterator(allocated.begin()),
                           std::make_move_iterator(allocated.end()));
        }
        updated_buffers = true;

        if (alloc_ok)
        {
            std::unordered_map<BufferId, ParallelBuffer> id_to_buf;
            for (auto &buf : buffers)
            {
                id_to_buf[buf.id] = buf;
            }
            for (uint32_t i = 0; i < order.size(); i++)
            {
                EClassId eclass_id = order[i];
                if (!eclass_to_buf.count(eclass_id))
                    Error::throw_err("eclass_id " + toString(eclass_id) + " not present in eclass_to_buf");
                BufferId buffer_id = eclass_to_buf.at(eclass_id);
                if (!id_to_buf.count(buffer_id))
                    Error::throw_err("buffer_id " + toString(buffer_id) + " not present in id_to_buf");
                ParallelBuffer &buf = id_to_buf.at(buffer_id);
                uint32_t sel = selection_map.at(eclass_id);
                ENodeId enode_id = egraph.getEClass(eclass_id).enodes[sel];
                const ENode &enode = egraph.getENode(enode_id);
                for (EClassId child : enode.getChildren())
                {
                    if (!eclass_to_buf.count(child))
                        continue;
                    ParallelBuffer &child_buf = id_to_buf.at(eclass_to_buf.at(child));
                    if (buf.offset == child_buf.offset && !(enode.getOpType() == OpType::PERMUTE || enode.getOpType() == OpType::REPEAT || enode.getOpType() == OpType::RESHAPE || enode.getOpType() == OpType::SLICE))
                    {
                        std::cout << "inplace op at " << toString(enode) << " " << (eclassToLogical.count(eclass_id) ? toString(eclassToLogical.at(eclass_id)) : "no logical id") << std::endl;
                    }
                }
            }

            return true;
        }
        reason = oom_reason;
        return false;
    }
};
