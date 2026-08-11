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
#include "core/plan/extractor.hpp"
#include "core/plan/search_delegate.hpp"
#include "core/plan/validators/validator.hpp"
#include "core/rewrite.hpp"
#include "core/shapes.hpp"
#include "core/types.hpp"

float get_cost(const std::vector<EClassId> &ordered, const EGraph &egraph,
               const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<ENodeInfo> &enodeInfos,
               bool print_utilization = false)
{
    std::unordered_map<EClassId, float> birth_times;
    std::unordered_map<Engine, float> engine_finish;
    std::unordered_map<Engine, float> engine_active_time;

    for (EClassId eclass : ordered)
    {
        auto sel_it = selection_map.find(eclass);
        if (sel_it == selection_map.end())
            continue;
        uint32_t sel = sel_it->second;
        ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
        const ENode &node = egraph.getENode(enode_id);

        if (node.getOpType() == OpType::INPUT || node.getOpType() == OpType::CACHE)
        {
            birth_times[eclass] = 0.0f;
            continue;
        }

        float cost = enodeInfos[enode_id.value].cost;

        float children_finish = 0.0f;
        for (EClassId child : node.getChildren())
        {
            child = egraph.findConst(child);
            auto c_sel_it = selection_map.find(child);
            if (c_sel_it == selection_map.end())
                continue;
            uint32_t child_sel = c_sel_it->second;
            ENodeId child_enode_id = egraph.getEClass(child).enodes[child_sel];
            const ENode &child_node = egraph.getENode(child_enode_id);

            for (const auto &engine : child_node.getEngines())
            {
                auto it = engine_finish.find(engine);
                float child_finish = (it != engine_finish.end()) ? it->second : 0.0f;
                children_finish = std::max(children_finish, child_finish);
            }
        }

        float engine_free = 0.0f;
        for (const auto &engine : node.getEngines())
        {
            auto it = engine_finish.find(engine);
            if (it != engine_finish.end())
            {
                engine_free = std::max(engine_free, it->second);
            }
        }

        float birth = std::max(children_finish, engine_free);
        birth_times[eclass] = birth;

        for (const auto &engine : node.getEngines())
        {
            engine_finish[engine] = birth + cost;
            engine_active_time[engine] += cost;
        }
    }

    float total_cost = 0.0f;
    for (const auto &kv : engine_finish)
    {
        total_cost = std::max(total_cost, kv.second);
    }

    if (print_utilization && total_cost > 0.0f)
    {
        std::cout << "Total Makespan (Cost): " << total_cost << " ms\n";
        for (const auto &kv : engine_active_time)
        {
            Engine eng = kv.first;
            float active_duration = kv.second;
            float percentage = (active_duration / total_cost) * 100.0f;

            std::cout << "  - Engine " << eng.idx << " (" << toString(eng.type) << "): "
                      << std::fixed << std::setprecision(2) << percentage << "% "
                      << "(" << active_duration << " ms active)\n";
        }
    }

    return total_cost;
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

static std::vector<ParallelBuffer> bufferize(const std::vector<EClassId> &ordered, const EGraph &egraph,
                                             const std::unordered_map<EClassId, uint32_t> &selection_map,
                                             const std::vector<ENodeInfo> &enodeInfos,
                                             std::unordered_map<EClassId, BufferId> &eclass_to_buf)
{
#ifdef DEBUG
    ProgressTimer t = ProgressTimer(0, "bufferize", false, true);
#endif

    std::unordered_map<EClassId, uint32_t> birth_times;
    std::unordered_map<EClassId, uint32_t> death_times;
    for (uint32_t i = 0; i < ordered.size(); ++i)
    {
        EClassId eclass = ordered[i];
        birth_times[eclass] = i;
        death_times[eclass] = i + 1;
        auto sel_it = selection_map.find(eclass);
        if (sel_it == selection_map.end())
            Error::throw_err("eclass from ordered not in selection_map");
        uint32_t sel = sel_it->second;
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
        auto sel_it = selection_map.find(eclass);
        if (sel_it == selection_map.end())
            Error::throw_err("eclass from ordered not in selection_map");
        uint32_t sel = sel_it->second;
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
                            auto c_sel_it = selection_map.find(child_base);
                            if (c_sel_it == selection_map.end())
                                continue;
                            uint32_t c_sel = c_sel_it->second;
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

    for (uint32_t i = 0; i < ordered.size(); ++i)
    {
        EClassId eclass = ordered[i];
        auto sel_it = selection_map.find(eclass);
        if (sel_it == selection_map.end())
            continue;
        uint32_t sel = sel_it->second;
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

    std::unordered_map<EClassId, BufferId> base_to_buf;
    for (EClassId eclass : ordered)
    {
        EClassId base = eclass;
        auto sel_it = selection_map.find(eclass);
        if (sel_it == selection_map.end())
            continue;
        uint32_t sel = sel_it->second;
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

            auto base_sel_it = selection_map.find(target_base);
            if (base_sel_it == selection_map.end())
                continue;
            uint32_t base_sel = base_sel_it->second;
            ENodeId base_enode_id = egraph.getEClass(target_base).enodes[base_sel];
            const ENode &base_node = egraph.getENode(base_enode_id);

            uint64_t size_bytes = getSizeBytes(base_node.getShape(), base_node.getDType());
            if (size_bytes == 0)
            {
                Error::throw_err("empty node");
            }
            size_bytes = (size_bytes + 4095) & ~4095ULL;

            auto b_it = birth_times.find(target_base);
            auto d_it = death_times.find(target_base);
            uint32_t b_time = (b_it != birth_times.end()) ? b_it->second : 0;
            uint32_t d_time = (d_it != death_times.end()) ? d_it->second : 0;

            ParallelBuffer buf = {
                buf_id, base_node.getMemSpace(), size_bytes, b_time, d_time,
                -1};
            buffers.push_back(std::move(buf));
        }

        eclass_to_buf[eclass] = base_to_buf[target_base];
    }
    return buffers;
}

static bool malloc(uint64_t mem_cap, const std::vector<ParallelBuffer> &unallocated,
                   std::vector<ParallelBuffer> &allocated,
                   std::shared_ptr<SearchDelegate> delegate = nullptr)
{
    LOG(L_INFO) << "malloc " + std::to_string(unallocated.size());
    ProgressTimer t(0, "malloc", false, true);
    if (unallocated.empty())
        return true;
    int N = static_cast<int>(unallocated.size());

    std::vector<int64_t> unallocated_sizes(N);
    for (int i = 0; i < N; ++i)
        unallocated_sizes[i] = unallocated[i].size;

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

    std::vector<int> avail(N);
    std::iota(avail.begin(), avail.end(), 0);

    std::vector<int> state(N, 0);
    state[0] = 0;

    std::vector<int> order(N, 0);
    std::vector<int64_t> chosen_offset(N, 0);
    std::vector<int64_t> global_offset_max(N + 1, 0);

    std::vector<int64_t> current_offsets(N, 0);

    struct Backup
    {
        int j;
        int64_t old_val;
    };
    std::vector<Backup> trail;
    trail.reserve(N * 50);
    std::vector<size_t> trail_starts(N + 1, 0);

    int k = 0;
    while (k >= 0)
    {
        if (k % 100 == 0)
        {
            LOG(L_INFO) << "malloc k=" << std::to_string(k) << "/" << std::to_string(N);
        }

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

        if (state[k] == (k == 0 ? 0 : k))
        {
            if (delegate)
            {
                delegate->push_state();
            }
        }

        bool advanced = false;

        int64_t h_min = std::numeric_limits<int64_t>::max();
        for (int idx = k; idx < N; ++idx)
        {
            int i = avail[idx];
            int64_t h = current_offsets[i] + unallocated_sizes[i];
            if (h < h_min)
                h_min = h;
        }

        while (state[k] < N)
        {
            uint32_t sel_idx = state[k] - k;
            int mapped_idx = k + sel_idx;

            if (delegate)
            {
                std::vector<ActionFeature> features;
                for (int idx = k; idx < N; ++idx)
                {
                    ActionFeature f;
                    f.id = unallocated[avail[idx]].id.value;
                    f.cost = 0.0f;
                    f.size = (float)unallocated[avail[idx]].size;
                    f.op_type = 0;
                    features.push_back(f);
                }
                std::vector<uint32_t> custom_order = delegate->order_malloc(features);
                if (sel_idx < custom_order.size())
                {
                    mapped_idx = k + custom_order[sel_idx];
                }
            }

            int idx = mapped_idx;
            state[k]++;
            int i = avail[idx];

            int64_t offset_i = current_offsets[i];

            if (offset_i < global_offset_max[k])
                continue;

            BufferId id_max = BufferId{0};
            for (int d = k - 1; d >= 0; --d)
            {
                if (chosen_offset[d] == offset_i)
                {
                    id_max = std::max(id_max, unallocated[order[d]].id);
                }
                else
                {
                    break;
                }
            }
            if (unallocated[i].id < id_max)
                continue;

            if (offset_i >= h_min)
                continue;

            if (mem_cap != std::numeric_limits<uint64_t>::max() &&
                static_cast<uint64_t>(offset_i) + unallocated_sizes[i] > mem_cap)
                continue;

            std::swap(avail[k], avail[idx]);
            order[k] = i;
            chosen_offset[k] = offset_i;
            global_offset_max[k + 1] = std::max(global_offset_max[k], offset_i);

            trail_starts[k] = trail.size();
            int64_t new_end = offset_i + unallocated_sizes[i];

            for (int j : adj[i])
            {
                if (current_offsets[j] < new_end)
                {
                    trail.push_back({j, current_offsets[j]});
                    current_offsets[j] = new_end;
                }
            }

            advanced = true;
            break;
        }

        if (advanced)
        {
            k++;
            if (k < N)
                state[k] = k;
        }
        else
        {
            if (delegate)
                delegate->pop_state();
            k--;
            if (k >= 0)
            {
                int idx = state[k] - 1;
                std::swap(avail[k], avail[idx]);

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

static bool check_peak_memory(const std::vector<ParallelBuffer> &bufs, uint64_t mem_cap,
                              BufferId &overflow)
{
    if (mem_cap == std::numeric_limits<uint64_t>::max())
        return true;

    struct Event
    {
        uint32_t time;
        int type;
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

    std::sort(events.begin(), events.end(), [](const Event &a, const Event &b)
              {
        if (a.time != b.time)
            return a.time < b.time;
        return a.type < b.type; });

    int64_t current_mem = 0;
    for (const auto &ev : events)
    {
        if (ev.type == 1)
        {
            current_mem += ev.size;
            if (current_mem > static_cast<int64_t>(mem_cap))
            {
                overflow = ev.buffer_id;
                LOG(L_INFO) << "[check_peak_memory] OOM error at idx=" << ev.time << std::endl;
                return false;
            }
        }
        else
        {
            current_mem -= ev.size;
        }
    }
    return true;
}

static bool greedy_alloc(uint64_t mem_cap, const std::vector<ParallelBuffer> &unallocated,
                         std::vector<ParallelBuffer> &allocated, BufferId &overflow)
{
    std::vector<ParallelBuffer> bufs = unallocated;

    std::sort(bufs.begin(), bufs.end(), [](const ParallelBuffer &a, const ParallelBuffer &b)
              {
        if (a.size != b.size)
            return a.size > b.size;
        return a.id < b.id; });

    allocated.clear();
    allocated.reserve(bufs.size());

    for (auto &buf : bufs)
    {
        int64_t best_offset = 0;

        std::vector<const ParallelBuffer *> time_overlaps;
        for (const auto &alloc : allocated)
        {
            if (overlapsBuf(buf, alloc))
            {
                time_overlaps.push_back(&alloc);
            }
        }

        std::sort(time_overlaps.begin(), time_overlaps.end(),
                  [](const ParallelBuffer *a, const ParallelBuffer *b)
                  { return a->offset < b->offset; });

        for (const auto *alloc : time_overlaps)
        {
            if (best_offset < alloc->offset + static_cast<int64_t>(alloc->size) &&
                best_offset + static_cast<int64_t>(buf.size) > alloc->offset)
            {
                best_offset = alloc->offset + alloc->size;
            }
        }

        if (mem_cap != std::numeric_limits<uint64_t>::max() &&
            best_offset + static_cast<int64_t>(buf.size) > static_cast<int64_t>(mem_cap))
        {
            overflow = buf.id;
            return false;
        }

        buf.offset = best_offset;
        allocated.push_back(buf);
    }
    return true;
}

static bool malloc_by_time_components(uint64_t mem_cap, const std::vector<ParallelBuffer> &unallocated,
                                      std::vector<ParallelBuffer> &allocated, BufferId &overflow,
                                      std::shared_ptr<SearchDelegate> delegate = nullptr)
{
    if (unallocated.empty())
        return true;

    std::vector<ParallelBuffer> sorted_bufs = unallocated;
    std::sort(sorted_bufs.begin(), sorted_bufs.end(), [](const ParallelBuffer &a, const ParallelBuffer &b)
              {
        if (a.start != b.start)
            return a.start < b.start;
        return a.end < b.end; });

    if (sorted_bufs.empty())
        return true;

    if (!check_peak_memory(sorted_bufs, mem_cap, overflow))
    {
        return false;
    }

    std::vector<ParallelBuffer> comp_allocated;
    if (greedy_alloc(mem_cap, sorted_bufs, comp_allocated, overflow))
    {
        allocated.insert(allocated.end(), comp_allocated.begin(), comp_allocated.end());
        return true;
    }

    std::sort(sorted_bufs.begin(), sorted_bufs.end(), [](const ParallelBuffer &a, const ParallelBuffer &b)
              {
        if (a.size != b.size)
            return a.size > b.size;
        return a.id < b.id; });

    if (!malloc(mem_cap, sorted_bufs, comp_allocated, delegate))
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
    const std::unordered_map<EClassId, LogicalId> &eclassToLogical;
    const std::unordered_map<LogicalId, ParallelBuffer> &preallocatedBuffers;
    std::shared_ptr<SearchDelegate> delegate;

    MemValidator(const EGraph &_egraph, const std::vector<ENodeInfo> &_enodeInfos,
                 const std::unordered_map<MemSpace, uint64_t> &_mem_caps,
                 const std::unordered_map<EClassId, LogicalId> &_eclassToLogical,
                 const std::unordered_map<LogicalId, ParallelBuffer> &_preallocatedBuffers,
                 std::shared_ptr<SearchDelegate> _delegate = nullptr)
        : egraph(_egraph), enodeInfos(_enodeInfos), mem_caps(_mem_caps), eclassToLogical(_eclassToLogical),
          preallocatedBuffers(_preallocatedBuffers), delegate(_delegate)
    {
    }

    bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<EClassId> &order,
                  const std::vector<EClassId> &path, std::vector<ParallelBuffer> &buffers,
                  std::unordered_map<EClassId, BufferId> &eclass_to_buf, float &cost,
                  std::vector<EClassId> &conflict_nodes) override
    {
        cost = get_cost(order, egraph, selection_map, enodeInfos);

        std::vector<ParallelBuffer> unallocated_buffers =
            bufferize(order, egraph, selection_map, enodeInfos, eclass_to_buf);

        std::unordered_set<BufferId> preallocated_buf_ids;
        std::unordered_map<BufferId, ParallelBuffer> preallocated_overrides;
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

            BufferId buf_id = eclass_to_buf.at(eclass);
            preallocated_buf_ids.insert(buf_id);
            preallocated_overrides[buf_id] = preIt->second;

            const ParallelBuffer &pre = preIt->second;
            uint64_t extent = static_cast<uint64_t>(pre.offset) + pre.size;
            uint64_t &cur = reserved_per_ms[pre.mem_space];
            cur = std::max(cur, extent);
        }

        std::unordered_map<MemSpace, std::vector<ParallelBuffer>> buf_by_mem_space;
        for (auto &buf : unallocated_buffers)
        {
            if (buf.mem_space.type == HandleType::STORAGE)
                continue;
            if (preallocated_buf_ids.count(buf.id))
                continue;
            buf_by_mem_space[buf.mem_space].push_back(buf);
        }

        buffers.clear();
        buffers.reserve(unallocated_buffers.size());

        for (auto &buf : unallocated_buffers)
        {
            if (buf.mem_space.type == HandleType::STORAGE)
            {
                buf.offset = 0;
                buffers.push_back(buf);
            }
        }

        for (auto &buf : unallocated_buffers)
        {
            if (preallocated_buf_ids.count(buf.id))
            {
                ParallelBuffer pre = preallocated_overrides.at(buf.id);
                buf.offset = pre.offset;
                buffers.push_back(buf);
            }
        }

        bool alloc_ok = true;
        BufferId overflow;
        MemSpace failed_ms;
        uint64_t failed_reduced_cap = std::numeric_limits<uint64_t>::max();

        for (auto &kv : buf_by_mem_space)
        {
            MemSpace ms = kv.first;
            auto &bufs = kv.second;
            uint64_t cap = mem_caps.count(ms) ? mem_caps.at(ms) : std::numeric_limits<uint64_t>::max();
            uint64_t reserved = reserved_per_ms.count(ms) ? reserved_per_ms.at(ms) : 0;
            uint64_t reduced_cap =
                (cap == std::numeric_limits<uint64_t>::max()) ? cap : (cap > reserved ? cap - reserved : 0);

            std::vector<ParallelBuffer> allocated;
            if (!malloc_by_time_components(reduced_cap, bufs, allocated, overflow, delegate))
            {
                alloc_ok = false;
                failed_ms = ms;
                failed_reduced_cap = reduced_cap;
                LOG(L_INFO) << "[MemValidator] OOM error in mem_space (" << ms.idx << ", " << (int)ms.type << ")"
                            << std::endl;
                break;
            }

            for (auto &buf : allocated)
            {
                buf.offset += static_cast<int64_t>(reserved);
            }

            buffers.insert(buffers.end(), std::make_move_iterator(allocated.begin()),
                           std::make_move_iterator(allocated.end()));
        }

        if (alloc_ok)
        {
            return true;
        }

        const ParallelBuffer *overflow_buf_ptr = nullptr;
        for (const auto &b : unallocated_buffers)
        {
            if (b.id == overflow)
            {
                overflow_buf_ptr = &b;
                break;
            }
        }

        std::unordered_map<BufferId, uint64_t> overlapping_buf_sizes;
        std::unordered_set<BufferId> overflows;

        if (overflow_buf_ptr)
        {
            for (const auto &b : unallocated_buffers)
            {
                if (b.mem_space == failed_ms && overlapsBuf(*overflow_buf_ptr, b))
                {
                    overflows.insert(b.id);
                    overlapping_buf_sizes[b.id] = b.size;
                }
            }
        }

        std::unordered_set<BufferId> seen_bufs;
        uint64_t running_sum = 0;

        for (EClassId node_in_path : path)
        {
            auto it = eclass_to_buf.find(node_in_path);
            if (it == eclass_to_buf.end())
                continue;

            bool added = false;
            BufferId buf_id = it->second;
            auto size_it = overlapping_buf_sizes.find(buf_id);
            if (size_it != overlapping_buf_sizes.end())
            {
                if (seen_bufs.insert(buf_id).second)
                {
                    running_sum += size_it->second;
                    conflict_nodes.push_back(node_in_path);
                    added = true;
                }
            }

            if (!added)
            {
                uint32_t sel = selection_map.at(node_in_path);
                ENodeId enode_id = egraph.getEClass(node_in_path).enodes[sel];
                const ENode &node = egraph.getENode(enode_id);
                for (EClassId child : node.getChildren())
                {
                    EClassId canon_child = egraph.findConst(child);
                    auto child_buf_it = eclass_to_buf.find(canon_child);
                    if (child_buf_it != eclass_to_buf.end() && overflows.count(child_buf_it->second))
                    {
                        conflict_nodes.push_back(node_in_path);
                        break;
                    }
                }
            }

            if (failed_reduced_cap != std::numeric_limits<uint64_t>::max() && running_sum > failed_reduced_cap)
            {
                break;
            }
        }

        buffers.insert(buffers.end(), unallocated_buffers.begin(), unallocated_buffers.end());
        return false;
    }
};