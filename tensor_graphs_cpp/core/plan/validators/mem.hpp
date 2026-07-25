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

        BufferId buf_id = BufferId{(uint32_t)buffers.size()};
        eclass_to_buf[eclass] = buf_id;

        uint64_t size_bytes = getSizeBytes(node.getShape(), node.getDType());
        if (size_bytes == 0)
        {
            size_bytes = 1;
        }
        // Align to 4096 bytes to prevent OpenCL CL_MISALIGNED_SUB_BUFFER_OFFSET (-13)
        size_bytes = (size_bytes + 4095) & ~4095ULL;

        ParallelBuffer buf = {buf_id, node.getMemSpace(), size_bytes, birth_times.at(eclass), death_times.at(eclass), -1};
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

static bool malloc(
    uint64_t mem_cap,
    const std::vector<ParallelBuffer> &unallocated,
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
    // Replaces creating a new std::vector<ParallelBuffer> at every recursion depth.
    // Elements from 0 to k-1 are "allocated". Elements from k to N-1 are "unallocated".
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
    // when a buffer is assigned, it "pushes" its end_boundary to all overlapping neighbors.
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
    std::vector<size_t> trail_starts(N + 1, 0); // Where the log started for depth k

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

        // The inner loop simulating: for (uint64_t i = 0; i < unallocated.size(); ++i)
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
            // If multiple buffers start at the exact same memory offset, we force them
            // to be picked in ascending order of their IDs to prevent duplicate search paths.
            // Because offsets strictly grow (enforced above), we only need to look at
            // the immediately preceding allocations in O(1) time rather than O(N).
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

            // 1. Swap the chosen buffer into the "allocated" portion of the `avail` array (O(1) remove)
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

// 1. O(N log N) Sweep-line to check if it's mathematically impossible to fit in mem_cap
static bool check_peak_memory(const std::vector<ParallelBuffer> &bufs, uint64_t mem_cap,
                              BufferId &overflow // if failed due to mem, which buffer pushed mem over the edge
)
{
    if (mem_cap == std::numeric_limits<uint64_t>::max())
        return true;

    struct Event
    {
        float time;
        int type; // 0 for end, 1 for start
        int64_t size;
        bool is_zero_duration;
        BufferId buffer_id;
    };

    std::vector<Event> events;
    events.reserve(bufs.size() * 2);
    for (const auto &b : bufs)
    {
        bool is_zero = (b.start == b.end);
        if (is_zero)
        {
            events.push_back({b.start, 1, static_cast<int64_t>(b.size), true, b.id});
        }
        else
        {
            events.push_back({b.start, 1, static_cast<int64_t>(b.size), false, b.id});
            events.push_back({b.end, 0, static_cast<int64_t>(b.size), false, b.id});
        }
    }

    // Process Ends before Starts to correctly simulate memory release
    std::sort(events.begin(), events.end(), [](const Event &a, const Event &b)
              {
        if (a.time != b.time) return a.time < b.time;
        return a.type < b.type; });

    int64_t current_mem = 0;
    for (const auto &ev : events)
    {
        if (ev.type == 1)
        { // start
            if (ev.is_zero_duration)
            {
                // Zero-duration buffers only overlap with strictly active intervals
                if (current_mem + ev.size > static_cast<int64_t>(mem_cap))
                {
                    overflow = ev.buffer_id;
                    return false;
                }
            }
            else
            {
                current_mem += ev.size;
                if (current_mem > static_cast<int64_t>(mem_cap))
                {
                    overflow = ev.buffer_id;
                    return false;
                }
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
static bool greedy_alloc(uint64_t mem_cap, const std::vector<ParallelBuffer> &unallocated, std::vector<ParallelBuffer> &allocated,
                         BufferId &overflow // if failed due to mem, which buffer pushed mem over the edge
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
        std::sort(time_overlaps.begin(), time_overlaps.end(), [](const ParallelBuffer *a, const ParallelBuffer *b)
                  { return a->offset < b->offset; });

        // First-fit algorithm: push best_offset upwards if there's a memory collision
        for (const auto *alloc : time_overlaps)
        {
            if (best_offset < alloc->offset + static_cast<int64_t>(alloc->size) &&
                best_offset + static_cast<int64_t>(buf.size) > alloc->offset)
            {
                best_offset = alloc->offset + alloc->size; // Move above the collision
            }
        }

        if (mem_cap != std::numeric_limits<uint64_t>::max() && best_offset + static_cast<int64_t>(buf.size) > static_cast<int64_t>(mem_cap))
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
static bool malloc_by_time_components(
    uint64_t mem_cap,
    const std::vector<ParallelBuffer> &unallocated,
    std::vector<ParallelBuffer> &allocated,
    BufferId &overflow)
{
    if (unallocated.empty())
        return true;

    // 1. Sort buffers by start time (and then end time to be deterministic)
    std::vector<ParallelBuffer> sorted_bufs = unallocated;
    std::sort(sorted_bufs.begin(), sorted_bufs.end(), [](const ParallelBuffer &a, const ParallelBuffer &b)
              {
        if (a.start != b.start) return a.start < b.start;
        return a.end < b.end; });

    // Helper lambda to process an independent connected component of buffers
    auto process_component = [&](std::vector<ParallelBuffer> &current_comp) -> bool
    {
        if (current_comp.empty())
            return true;

        // OPTIMIZATION 1: Absolute strict lower bound check
        if (!check_peak_memory(current_comp, mem_cap, overflow))
        {
            return false; // Mathematically impossible to fit; abort instantly
        }

        // OPTIMIZATION 2: Fast greedy allocator path (solves immediately most of the time)
        std::vector<ParallelBuffer> comp_allocated;
        if (greedy_alloc(mem_cap, current_comp, comp_allocated, overflow))
        {
            allocated.insert(allocated.end(), comp_allocated.begin(), comp_allocated.end());
            return true;
        }

        // OPTIMIZATION 3: Exact solver fallback with aggressive pruning ordering
        // Sorting by size descending forces the tree to hit conflict limits much faster.
        std::sort(current_comp.begin(), current_comp.end(), [](const ParallelBuffer &a, const ParallelBuffer &b)
                  {
            if (a.size != b.size) return a.size > b.size;
            return a.id < b.id; });

        if (!malloc(mem_cap, current_comp, comp_allocated))
        {
            return false;
        }

        allocated.insert(allocated.end(), comp_allocated.begin(), comp_allocated.end());
        return true;
    };

    std::vector<ParallelBuffer> current_comp;
    float max_end = sorted_bufs[0].end;
    current_comp.push_back(sorted_bufs[0]);

    // 2. Iterate and split components at time gaps
    for (size_t i = 1; i < sorted_bufs.size(); ++i)
    {
        if (sorted_bufs[i].start >= max_end)
        {
            if (!process_component(current_comp))
                return false;

            // Reset for the next component
            current_comp.clear();
            max_end = sorted_bufs[i].end;
        }
        else
        {
            max_end = std::max(max_end, sorted_bufs[i].end);
        }
        current_comp.push_back(sorted_bufs[i]);
    }

    // 3. Solve the final component
    if (!process_component(current_comp))
    {
        return false;
    }

    return true;
}

struct MemValidator : public ISelectionValidator
{
    const EGraph &egraph;
    const std::vector<ENodeInfo> &enodeInfos;
    const std::unordered_map<MemSpace, uint64_t> &mem_caps;

    MemValidator(const EGraph &_egraph,
                 const std::vector<ENodeInfo> &_enodeInfos,
                 const std::unordered_map<MemSpace, uint64_t> &_mem_caps)
        : egraph(_egraph), enodeInfos(_enodeInfos), mem_caps(_mem_caps) {}

    bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<EClassId> &order,
                  std::vector<ParallelBuffer> &buffers,
                  std::unordered_map<EClassId, BufferId> &eclass_to_buf,
                  BufferId &overflow,
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

        // group buffers by mem_space instead of mem_space.idx
        std::unordered_map<MemSpace, std::vector<ParallelBuffer>> buf_by_mem_space;
        for (auto &buf : unallocated_buffers)
        {
            if (buf.mem_space.type == HandleType::STORAGE)
                continue;
            buf_by_mem_space[buf.mem_space].push_back(buf);
        }

        // malloc: try to assign offsets within mem_cap for each mem_space
        buffers.clear();
        buffers.reserve(unallocated_buffers.size());

        // Directly copy STORAGE buffers to the final list since they don't need allocation
        for (auto &buf : unallocated_buffers)
        {
            if (buf.mem_space.type == HandleType::STORAGE)
            {
                buf.offset = 0; // Actual file offset is resolved dynamically in StorageBuffer::setupInput
                buffers.push_back(buf);
            }
        }

        bool alloc_ok = true;
        std::string oom_reason = "OOM";

        for (auto &kv : buf_by_mem_space)
        {
            MemSpace ms = kv.first;
            auto &bufs = kv.second;
            uint64_t cap = mem_caps.count(ms)
                               ? mem_caps.at(ms)
                               : std::numeric_limits<uint64_t>::max();
            std::vector<ParallelBuffer> allocated;
            ProgressTimer t2 = ProgressTimer(0, "malloc mem_space=(" + std::to_string(ms.idx) + "," + std::to_string((int)ms.type) + "), n_bufs=" + std::to_string(bufs.size()) + " ", false, true);
            if (!malloc_by_time_components(cap, bufs, allocated, overflow))
            {
                alloc_ok = false;
                oom_reason = "OOM:" + std::to_string(ms.idx) + ":" + std::to_string(static_cast<int>(ms.type));
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
        reason = oom_reason;
        return false;
    }
};