#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/cost_model.hpp"
#include "core/kernels.hpp"
#include "core/rewrite.hpp"
#include "core/shapes.hpp"
#include "core/misc.hpp"
#include "core/egraph.hpp"
#include "core/extractor.hpp"
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

void propagateDirtyRegionsAtomic(
    const std::vector<uint32_t> &topo,
    const Graph &graph,
    std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
    std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &dirtyInputRegions)
{
    ShapePropagator propagator;

    for (uint32_t nodeId : topo)
    {
        if (!graph.hasNode(nodeId))
            continue;
        const TensorNode &node = graph.getNode(nodeId);
        if (node.opType == OpType::INPUT)
            continue;

        std::vector<std::vector<Region>> parentRegions;
        bool anyParentDirty = false;
        for (uint32_t pid : node.parentIds)
        {
            auto it = dirtyOutputRegions.find(pid);
            if (it != dirtyOutputRegions.end() && !it->second.empty())
            {
                parentRegions.push_back(it->second);
                anyParentDirty = true;
            }
            else
            {
                parentRegions.push_back({});
            }
        }

        std::vector<Region> propagated;
        if (anyParentDirty)
            propagated = propagator.forward(node, graph, parentRegions);

        auto existingIt = dirtyOutputRegions.find(nodeId);
        if (existingIt != dirtyOutputRegions.end() && !existingIt->second.empty())
        {
            if (!propagated.empty())
            {
                dirtyOutputRegions[nodeId] = mergeRegions(propagated, existingIt->second);
            }
        }
        else
        {
            dirtyOutputRegions[nodeId] = propagated;
        }

        dirtyInputRegions[nodeId] = propagator.backward(node, graph, dirtyOutputRegions[nodeId]);
    }
}

struct Planner
{
    uint32_t egraph_dump_counter_ = 0;
    CostModel &costModel;
    std::unordered_map<uint32_t, uint64_t> mem_caps;

    void dumpEGraphBinary(const EGraph &egraph, uint32_t rootEClassId)
    {
        const std::string dir = "egraph_viewer/egraphs";
        std::filesystem::create_directories(dir);

        std::string path;
        while (true)
        {
            path = dir + "/" + std::to_string(egraph_dump_counter_) + ".bin";
            egraph_dump_counter_++;
            if (!std::filesystem::exists(path))
                break;
        }

        std::ofstream out(path, std::ios::binary);
        if (!out)
        {
            std::cerr << "[Planner.dumpEGraphBinary] Failed to open " << path << " for writing." << std::endl;
            return;
        }

        const auto &classes = egraph.getClasses();
        const auto &enodes = egraph.getENodes();

        uint32_t num_classes = static_cast<uint32_t>(classes.size());
        uint32_t num_enodes = static_cast<uint32_t>(enodes.size());

        out.write(reinterpret_cast<const char *>(&num_classes), 4);
        out.write(reinterpret_cast<const char *>(&num_enodes), 4);
        out.write(reinterpret_cast<const char *>(&rootEClassId), 4);

        for (const auto &cls : classes)
        {
            out.write(reinterpret_cast<const char *>(&cls.id), 4);
            uint32_t s_size = static_cast<uint32_t>(cls.shape.size());
            out.write(reinterpret_cast<const char *>(&s_size), 4);
            if (s_size > 0)
                out.write(reinterpret_cast<const char *>(cls.shape.data()), s_size * 4);

            uint32_t st_size = static_cast<uint32_t>(cls.strides.size());
            out.write(reinterpret_cast<const char *>(&st_size), 4);
            if (st_size > 0)
                out.write(reinterpret_cast<const char *>(cls.strides.data()), st_size * 8);

            out.write(reinterpret_cast<const char *>(&cls.viewOffset), 8);
            out.write(reinterpret_cast<const char *>(&cls.dtype), 4);
            out.write(reinterpret_cast<const char *>(&cls.backend), 4);

            uint32_t e_size = static_cast<uint32_t>(cls.enodes.size());
            out.write(reinterpret_cast<const char *>(&e_size), 4);
            if (e_size > 0)
                out.write(reinterpret_cast<const char *>(cls.enodes.data()), e_size * 4);
        }

        for (const auto &enode : enodes)
        {
            out.write(reinterpret_cast<const char *>(&enode.kernelUid), 8);
            uint32_t op_type = static_cast<uint32_t>(enode.opType);
            out.write(reinterpret_cast<const char *>(&op_type), 4);
            uint32_t n_len = static_cast<uint32_t>(enode.opName.length());
            out.write(reinterpret_cast<const char *>(&n_len), 4);
            if (n_len > 0)
                out.write(enode.opName.c_str(), n_len);

            uint32_t c_size = static_cast<uint32_t>(enode.children.size());
            out.write(reinterpret_cast<const char *>(&c_size), 4);
            if (c_size > 0)
                out.write(reinterpret_cast<const char *>(enode.children.data()), c_size * 4);

            out.write(reinterpret_cast<const char *>(&enode.leafId), 4);

            uint32_t s_size = static_cast<uint32_t>(enode.shape.size());
            out.write(reinterpret_cast<const char *>(&s_size), 4);
            if (s_size > 0)
                out.write(reinterpret_cast<const char *>(enode.shape.data()), s_size * 4);

            uint32_t st_size = static_cast<uint32_t>(enode.strides.size());
            out.write(reinterpret_cast<const char *>(&st_size), 4);
            if (st_size > 0)
                out.write(reinterpret_cast<const char *>(enode.strides.data()), st_size * 8);

            out.write(reinterpret_cast<const char *>(&enode.viewOffset), 8);
            out.write(reinterpret_cast<const char *>(&enode.dtype), 4);
            out.write(reinterpret_cast<const char *>(&enode.backend), 4);
            out.write(reinterpret_cast<const char *>(&enode.sig), 8);
        }

        uint32_t num_constants = static_cast<uint32_t>(egraph.constantStaging.size());
        out.write(reinterpret_cast<const char *>(&num_constants), 4);
        for (const auto &[e_class_id, data_ptr] : egraph.constantStaging)
        {
            uint32_t canonId = e_class_id;
            out.write(reinterpret_cast<const char *>(&canonId), 4);
            const auto &data = *data_ptr;
            uint64_t data_size = static_cast<uint64_t>(data.size());
            out.write(reinterpret_cast<const char *>(&data_size), 8);
            out.write(reinterpret_cast<const char *>(data.data()), data_size);
        }

        out.close();
        std::cout << "[Planner.dumpEGraphBinary] Dumped EGraph to " << path << std::endl;
    }

    struct ExtractChoice
    {
        uint32_t enodeId = 0;
        float cost = std::numeric_limits<float>::infinity();
        bool valid = false;
    };

    struct ExtractionResult
    {
        std::unordered_map<uint32_t, ExtractChoice> choiceByEClass;
        float totalCost = std::numeric_limits<float>::infinity();
    };

    uint64_t getMemoryLimit(MemorySpace space) const
    {
        auto it = maxMemoryBySpace.find(space);
        if (it != maxMemoryBySpace.end())
            return it->second;
        return std::numeric_limits<uint64_t>::max();
    }

    void inferShapes(const std::vector<LogicalId> &topo, Graph &graph)
    {
        ShapePropagator propagator;
        for (LogicalId nodeId : topo)
        {
            propagator.inferShape(nodeId, graph);
        }
    }

    void saturate(EGraph &egraph, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical, bool injected, bool allowPushDownOnProtected = false, Repo *repo = nullptr)
    {
        RuleCtx ctx{egraph, protectedEClasses, eclassToLogical, repo};
        std::vector<std::unique_ptr<Rule>> rules;
        rules.emplace_back(std::make_unique<FusionRule>());
        rules.emplace_back(std::make_unique<FlattenBatchDot>());
        rules.emplace_back(std::make_unique<FlattenElementwise>());
        if (injected)
        {
            rules.emplace_back(std::make_unique<InfinityDomination>());
            rules.emplace_back(std::make_unique<SlicePushDownElementwise>(allowPushDownOnProtected));
            rules.emplace_back(std::make_unique<SlicePushDownDot>(allowPushDownOnProtected));
        }

        std::map<std::string, uint32_t> ruleMatchCounts;
        size_t iterations = 0;
        bool changed = true;
        uint32_t nMatches = 0;
#ifdef DEBUG
        ProgressTimer timer(0, "saturating ");
#endif
        while (changed)
        {
            if (InterruptManager::isInterrupted())
            {
                std::cerr << "\n[Planner.saturate] Interrupt detected, aborting execution..." << std::endl;
                InterruptManager::cleanup();
                std::exit(SIGINT);
            }
            iterations++;
            uint32_t numENodes = egraph.getENodes().size();
            ProgressTimer timer2(0, "saturation round " + std::to_string(iterations - 1) + " ");
            for (uint32_t eNodeIdx = 0; eNodeIdx < egraph.getENodes().size(); eNodeIdx++)
            {
                for (const auto &rule : rules)
                {
                    if (!rule->match(eNodeIdx, ctx))
                        continue;

                    rule->apply(eNodeIdx, ctx);
                    changed = true;
                    ruleMatchCounts[rule->name()]++;
                    nMatches++;
                }
                timer2.tick();
            }
            egraph.rebuild();
            changed = egraph.getENodes().size() != numENodes;
            std::cout << "\n--- Saturation Summary (" << iterations << " iterations) ---" << std::endl;
            for (auto const &[name, count] : ruleMatchCounts)
            {
                std::cout << "  " << name << ": " << count << " matches" << std::endl;
            }
            std::cout << "Total Matches: " << nMatches << std::endl;
#ifdef DEBUG
            timer.tick();
            std::cout << "# New enodes: " << egraph.getENodes().size() - numENodes << std::endl;
#endif
        }
        std::cout << "Finished saturation in " << iterations << " iterations with " << nMatches << " matches\n"
                  << std::flush;
    }

    std::unordered_map<uint32_t, uint32_t> build_ref_counts(const EGraph &egraph, const std::unordered_map<uint32_t, uint32_t> &selection_map, uint32_t root) const
    {
        std::unordered_map<uint32_t, uint32_t> ref;
        for (const auto &kv : selection_map)
        {
            uint32_t eclass = kv.first;
            uint32_t sel = kv.second;
            const ENode &node = egraph.getENodes()[egraph.getEClass(eclass).enodes[sel]];
            for (uint32_t c : node.children)
            {
                ref[c]++;
            }
        }
        ref[root]++;
        return ref;
    }

    struct PrecompData
    {
        std::vector<bool> is_eclass_persistent;
        std::vector<bool> is_eclass_cached;
        std::vector<std::vector<uint32_t>> enode_canon_children;
        uint64_t static_baseline_arr[static_cast<size_t>(Backend::_COUNT)];
        uint64_t mem_limits_arr[static_cast<size_t>(Backend::_COUNT)];
    };

private:
    struct SimNode
    {
        uint32_t e_class_id;
        uint32_t enodeId;
        float cost;
        Backend engine;
        std::vector<uint32_t> deps;
        std::vector<uint32_t> consumers;
        float upward_rank = 0.0f;
        float start_time = 0.0f;
        float end_time = 0.0f;
        bool is_view = false;
        bool is_inplace = false;
        int32_t inplace_idx = -1;
        uint64_t size_bytes = 0;
        uint32_t memory_space = 0;
        bool is_persistent = false;
        OpType op_type;
        uint64_t view_offset = 0;
    };

    struct AllocResult
    {
        bool fits = true;
        uint32_t oomEClassId = UINT32_MAX;
        std::vector<bool> oomLiveAllocated;
        std::unordered_map<uint32_t, uint64_t> offsets;
        std::unordered_map<uint32_t, uint64_t> peak_mem;
    };

    struct MemBlockSim
    {
        uint64_t size = 0;
        float start_time = 0.0f;
        float end_time = 0.0f;
        uint32_t space = 0;
        bool is_persistent = false;
        std::unordered_map<uint32_t, uint64_t> node_offsets;
        uint64_t base_offset = 0;
        std::vector<uint32_t> member_eclasses;
    };

    float simulateExecutionTime(
        const EGraph &egraph,
        const std::unordered_map<uint32_t, uint32_t> &selection_map,
        const std::vector<ENodeInfo> &enodeInfos,
        const PrecompData &precomp,
        const std::vector<uint32_t> &topo_order,
        std::unordered_map<uint32_t, SimNode> &simNodesOut) const
    {
        for (uint32_t e_class_id : topo_order)
        {
            uint32_t sel = selection_map.at(e_class_id);
            uint32_t enodeId = egraph.getEClass(e_class_id).enodes[sel];
            const ENode &enode = egraph.getENodes()[enodeId];
            const ENodeInfo &info = enodeInfos[enodeId];

            SimNode sn;
            sn.e_class_id = e_class_id;
            sn.enodeId = enodeId;
            sn.cost = info.cost;
            sn.engine = enode.backend;
            sn.deps = precomp.enode_canon_children[enodeId];
            sn.is_view = info.isView;
            sn.is_inplace = info.inplace;
            sn.inplace_idx = info.inplace_idx;
            sn.size_bytes = info.memSizes.count(enode.backend) ? info.memSizes.at(enode.backend) : 0;
            sn.memory_space = getMemorySpace(enode.backend);
            sn.is_persistent = precomp.is_eclass_persistent[e_class_id];
            sn.op_type = enode.opType;
            sn.view_offset = enode.viewOffset * getDTypeSize(enode.dtype);

            simNodesOut[e_class_id] = sn;
        }

        for (auto &kv : simNodesOut)
        {
            for (uint32_t dep : kv.second.deps)
            {
                if (simNodesOut.count(dep))
                {
                    simNodesOut[dep].consumers.push_back(kv.first);
                }
            }
        }

        for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it)
        {
            SimNode &sn = simNodesOut[*it];
            float max_child_rank = 0.0f;
            for (uint32_t c : sn.consumers)
            {
                max_child_rank = std::max(max_child_rank, simNodesOut[c].upward_rank);
            }
            sn.upward_rank = sn.cost + max_child_rank;
        }

        std::unordered_map<uint32_t, int> in_degree;
        for (uint32_t e_class_id : topo_order)
        {
            in_degree[e_class_id] = simNodesOut[e_class_id].deps.size();
        }

        auto cmp = [&](uint32_t a, uint32_t b)
        {
            return simNodesOut[a].upward_rank < simNodesOut[b].upward_rank;
        };
        std::priority_queue<uint32_t, std::vector<uint32_t>, decltype(cmp)> ready_queue(cmp);

        for (uint32_t e_class_id : topo_order)
        {
            if (in_degree[e_class_id] == 0)
            {
                ready_queue.push(e_class_id);
            }
        }

        std::unordered_map<Backend, float> engine_ready_time;
        std::unordered_map<uint32_t, float> node_finish_time;
        float total_makespan = 0.0f;

        while (!ready_queue.empty())
        {
            uint32_t curr = ready_queue.top();
            ready_queue.pop();
            SimNode &sn = simNodesOut[curr];

            float earliest_start = 0.0f;
            for (uint32_t dep : sn.deps)
            {
                earliest_start = std::max(earliest_start, node_finish_time[dep]);
            }

            float actual_start = std::max(earliest_start, engine_ready_time[sn.engine]);
            sn.start_time = actual_start;
            sn.end_time = actual_start + sn.cost;

            engine_ready_time[sn.engine] = sn.end_time;
            node_finish_time[curr] = sn.end_time;
            total_makespan = std::max(total_makespan, sn.end_time);

            for (uint32_t c : sn.consumers)
            {
                in_degree[c]--;
                if (in_degree[c] == 0)
                {
                    ready_queue.push(c);
                }
            }
        }

        return total_makespan;
    }

    AllocResult allocateStaticMemory(
        std::unordered_map<uint32_t, SimNode> &simNodes,
        const std::vector<uint32_t> &topo_order,
        const std::unordered_map<Backend, uint64_t> &maxMemoryByBackend,
        size_t numClasses,
        uint64_t alignment = 256) const
    {
        AllocResult result;
        result.oomLiveAllocated.assign(numClasses, false);

        std::unordered_map<uint32_t, uint32_t> node_to_block;
        std::unordered_map<uint32_t, MemBlockSim> blocks;

        // Phase 1: Alias Resolution & Lifetime Analysis
        for (uint32_t e_class_id : topo_order)
        {
            SimNode &node = simNodes[e_class_id];

            float end_time = node.end_time;
            for (uint32_t c : node.consumers)
            {
                end_time = std::max(end_time, simNodes[c].end_time);
            }

            if (node.is_inplace || node.is_view)
            {
                uint32_t parent = node.deps.empty() ? UINT32_MAX : node.deps[node.is_inplace ? node.inplace_idx : 0];
                if (parent != UINT32_MAX && node_to_block.count(parent))
                {
                    uint32_t block_id = node_to_block[parent];
                    blocks[block_id].end_time = std::max(blocks[block_id].end_time, end_time);
                    uint64_t required_size = node.view_offset + node.size_bytes;
                    blocks[block_id].size = std::max(blocks[block_id].size, required_size);
                    node_to_block[e_class_id] = block_id;
                    blocks[block_id].node_offsets[e_class_id] = node.view_offset;
                    blocks[block_id].member_eclasses.push_back(e_class_id);
                    continue;
                }
            }
            else if ((node.op_type == OpType::COPY_TO || node.op_type == OpType::TRANSFER) && !node.deps.empty())
            {
                uint32_t parent = node.deps[0];
                if (simNodes.count(parent) && simNodes[parent].memory_space == node.memory_space)
                {
                    uint32_t block_id = node_to_block[parent];
                    blocks[block_id].end_time = std::max(blocks[block_id].end_time, end_time);
                    node_to_block[e_class_id] = block_id;
                    blocks[block_id].node_offsets[e_class_id] = 0;
                    blocks[block_id].member_eclasses.push_back(e_class_id);
                    continue;
                }
            }

            uint32_t block_id = e_class_id;
            MemBlockSim b;
            b.size = node.size_bytes;
            b.start_time = node.start_time;
            b.end_time = end_time;
            b.space = node.memory_space;
            b.is_persistent = node.is_persistent;
            b.node_offsets[e_class_id] = 0;
            b.member_eclasses.push_back(e_class_id);
            blocks[block_id] = b;
            node_to_block[e_class_id] = block_id;
        }

        struct FreeBlock
        {
            uint64_t offset;
            uint64_t size;
        };
        struct ActiveAlloc
        {
            float end_time;
            uint64_t offset;
            uint64_t size;
            std::vector<uint32_t> members;
        };

        std::unordered_map<uint32_t, std::vector<uint32_t>> blocks_by_space;
        for (auto &kv : blocks)
        {
            blocks_by_space[kv.second.space].push_back(kv.first);
        }

        // Phase 2: Interval Graph Allocation
        for (auto &kv : blocks_by_space)
        {
            uint32_t space = kv.first;
            std::vector<uint32_t> &space_block_ids = kv.second;

            uint64_t persistent_offset = 0;
            std::vector<uint32_t> transient_blocks;

            for (uint32_t b_id : space_block_ids)
            {
                if (blocks[b_id].is_persistent)
                {
                    blocks[b_id].base_offset = persistent_offset;
                    persistent_offset += (blocks[b_id].size + alignment - 1) & ~(alignment - 1);
                }
                else
                {
                    transient_blocks.push_back(b_id);
                }
            }

            std::sort(transient_blocks.begin(), transient_blocks.end(), [&](uint32_t a, uint32_t b)
                      { return blocks[a].start_time < blocks[b].start_time; });

            std::vector<FreeBlock> free_list;
            std::vector<ActiveAlloc> active_allocs;
            uint64_t peak_mem = persistent_offset;

            auto insert_free = [&](uint64_t off, uint64_t sz)
            {
                free_list.push_back({off, sz});
                std::sort(free_list.begin(), free_list.end(), [](const FreeBlock &a, const FreeBlock &b)
                          { return a.offset < b.offset; });
                std::vector<FreeBlock> merged;
                for (auto &fb : free_list)
                {
                    if (merged.empty())
                        merged.push_back(fb);
                    else if (merged.back().offset + merged.back().size == fb.offset)
                    {
                        merged.back().size += fb.size;
                    }
                    else
                    {
                        merged.push_back(fb);
                    }
                }
                free_list = std::move(merged);
            };

            for (uint32_t b_id : transient_blocks)
            {
                MemBlockSim &b = blocks[b_id];
                float current_time = b.start_time;

                std::vector<ActiveAlloc> still_active;
                for (auto &alloc : active_allocs)
                {
                    if (alloc.end_time <= current_time)
                    {
                        insert_free(alloc.offset, alloc.size);
                    }
                    else
                    {
                        still_active.push_back(alloc);
                    }
                }
                active_allocs = std::move(still_active);

                uint64_t required_size = (b.size + alignment - 1) & ~(alignment - 1);
                if (required_size == 0)
                    required_size = alignment;

                int best_idx = -1;
                uint64_t best_size = UINT64_MAX;

                for (size_t i = 0; i < free_list.size(); ++i)
                {
                    if (free_list[i].size >= required_size && free_list[i].size < best_size)
                    {
                        best_idx = (int)i;
                        best_size = free_list[i].size;
                    }
                }

                if (best_idx != -1)
                {
                    b.base_offset = free_list[best_idx].offset;
                    free_list[best_idx].offset += required_size;
                    free_list[best_idx].size -= required_size;
                    if (free_list[best_idx].size == 0)
                    {
                        free_list.erase(free_list.begin() + best_idx);
                    }
                }
                else
                {
                    b.base_offset = peak_mem;
                    peak_mem += required_size;

                    uint64_t limit = UINT64_MAX;
                    for (const auto &limit_kv : maxMemoryByBackend)
                    {
                        if (getMemorySpace(limit_kv.first) == space)
                        {
                            limit = std::min(limit, limit_kv.second);
                        }
                    }
                    if (peak_mem > limit)
                    {
                        result.fits = false;
                        result.oomEClassId = b_id;
                        for (auto &alloc : active_allocs)
                        {
                            for (uint32_t mbr : alloc.members)
                                result.oomLiveAllocated[mbr] = true;
                        }
                        for (uint32_t mbr : b.member_eclasses)
                            result.oomLiveAllocated[mbr] = true;
                        return result;
                    }
                }

                active_allocs.push_back({b.end_time, b.base_offset, required_size, b.member_eclasses});
            }
            result.peak_mem[space] = peak_mem;
        }

        for (uint32_t e_class_id : topo_order)
        {
            uint32_t b_id = node_to_block[e_class_id];
            result.offsets[e_class_id] = blocks[b_id].base_offset + blocks[b_id].node_offsets[e_class_id];
        }

        return result;
    }

    void visitTopoFast(
        uint32_t e_class_id,
        const EGraph &egraph,
        const std::unordered_map<uint32_t, uint32_t> &selection_map,
        const PrecompData &precomp,
        std::vector<bool> &visited_classes,
        std::vector<uint32_t> &topo_order) const
    {
        e_class_id = egraph.findConst(e_class_id);
        if (visited_classes[e_class_id])
            return;
        visited_classes[e_class_id] = true;

        auto choiceIt = selection_map.find(e_class_id);
        if (choiceIt == selection_map.end())
            return;

        uint32_t enode_id = egraph.getEClass(e_class_id).enodes[choiceIt->second];
        for (uint32_t child : precomp.enode_canon_children[enode_id])
            visitTopoFast(child, egraph, selection_map, precomp, visited_classes, topo_order);
        topo_order.push_back(e_class_id);
    }

    bool validateInplaceSchedulesFast(
        const EGraph &egraph,
        const std::unordered_map<uint32_t, uint32_t> &selection_map,
        const std::vector<ENodeInfo> &enodeInfos,
        uint32_t rootEClassId,
        const std::unordered_map<uint32_t, uint32_t> &eclassToLogical,
        const PrecompData &precomp,
        std::vector<uint32_t> &topo_order,
        std::vector<bool> &val_visited_classes,
        std::vector<uint32_t> &val_overwritten,
        std::vector<uint32_t> &val_mem_root,
        std::string &outReason,
        uint32_t &conflictEClass1,
        uint32_t &conflictEClass2) const
    {
        topo_order.clear();
        std::fill(val_visited_classes.begin(), val_visited_classes.end(), false);
        visitTopoFast(rootEClassId, egraph, selection_map, precomp, val_visited_classes, topo_order);

        std::fill(val_overwritten.begin(), val_overwritten.end(), UINT32_MAX);
        std::unordered_map<uint32_t, uint32_t> overwritten_logical;
        for (size_t i = 0; i < egraph.getClasses().size(); ++i)
            val_mem_root[i] = static_cast<uint32_t>(i);

        for (uint32_t eclass : topo_order)
        {
            uint32_t sel = selection_map.at(eclass);
            uint32_t enodeId = egraph.getEClass(eclass).enodes[sel];
            const ENodeInfo &info = enodeInfos[enodeId];

            if (info.isView && !precomp.enode_canon_children[enodeId].empty())
            {
                uint32_t child_eclass = precomp.enode_canon_children[enodeId][0];
                val_mem_root[eclass] = val_mem_root[child_eclass];
            }

            for (uint32_t child_eclass : precomp.enode_canon_children[enodeId])
            {
                uint32_t child_root = val_mem_root[child_eclass];

                if (val_overwritten[child_root] != UINT32_MAX)
                {
                    outReason = "inplace " + std::to_string(child_root);
                    conflictEClass1 = val_overwritten[child_root];
                    conflictEClass2 = eclass;
                    return false;
                }

                uint32_t logicalId = eclassToLogical.count(child_root) ? eclassToLogical.at(child_root) : UINT32_MAX;
                if (logicalId != UINT32_MAX && overwritten_logical.count(logicalId))
                {
                    if (overwritten_logical[logicalId] != child_root)
                    {
                        outReason = "inplace_logical " + std::to_string(logicalId);
                        conflictEClass1 = overwritten_logical[logicalId];
                        conflictEClass2 = eclass;
                        return false;
                    }
                }
            }

            if (info.inplace)
            {
                uint32_t inplace_child = precomp.enode_canon_children[enodeId][info.inplace_idx];
                uint32_t child_root = val_mem_root[inplace_child];
                val_overwritten[child_root] = eclass;

                uint32_t logicalId = eclassToLogical.count(child_root) ? eclassToLogical.at(child_root) : UINT32_MAX;
                if (logicalId != UINT32_MAX)
                {
                    overwritten_logical[logicalId] = eclass;
                }
            }
        }
        return true;
    }

    ExtractionResult extractBest(const uint32_t rootId, const Graph &graph, EGraph &egraph,
                                 const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                                 const std::unordered_map<Backend, uint64_t> &maxMemoryByBackend,
                                 const std::unordered_map<uint32_t, Backend> &cachedNodes,
                                 const std::unordered_map<uint32_t, uint32_t> &eclassToLogical,
                                 const std::unordered_set<uint32_t> &immutable_eclasses,
                                 bool stopOnFirstValid = true,
                                 bool strictCache = false)
    {
        constexpr float EPS = 1e-6f;

        auto isConstantNeeded = [](OpType op, size_t inputIdx, size_t numInputs) -> bool
        {
            if (op == OpType::REPEAT && (inputIdx == 1 || inputIdx == 2))
                return true;
            if (op == OpType::RESHAPE && inputIdx == 1)
                return true;
            if (op == OpType::PERMUTE && inputIdx == 1)
                return true;
            if (op == OpType::SLICE && (inputIdx == 1 || inputIdx == 2 || inputIdx == 3))
                return true;
            if (op == OpType::SCATTER && (inputIdx == 2 || inputIdx == 3 || inputIdx == 4))
                return true;
            if ((op == OpType::SUM || op == OpType::MAX) && inputIdx == 1)
                return true;
            if (op == OpType::CONCAT && inputIdx == numInputs - 1)
                return true;
            if (op == OpType::TRIU && inputIdx == 1)
                return true;
            if (op == OpType::FILL && inputIdx == 1)
                return true;
            if (op == OpType::IM2COL && (inputIdx == 1 || inputIdx == 2 || inputIdx == 3))
                return true;
            if (op == OpType::ARANGE && (inputIdx == 0 || inputIdx == 1 || inputIdx == 2))
                return true;
            if (op == OpType::ARGMAX && (inputIdx == 1 || inputIdx == 2))
                return true;
            return false;
        };

        ProgressTimer timer3(egraph.getENodes().size(), "calculating enode info ");
        std::vector<ENodeInfo> enodeInfos(egraph.getENodes().size());
        for (size_t i = 0; i < egraph.getENodes().size(); ++i)
        {
            const ENode &enode = egraph.getENodes()[i];
            ENodeInfo info;
            info.memSizes[enode.backend] = getSizeBytes(enode.shape, enode.dtype);
            info.inplace = false;
            info.inplace_idx = -1;
            info.isScatter = false;
            info.isView = false;

            if (enode.kernelUid != 0)
            {
                const auto &kernel = KernelRegistry::get().getKernel(enode.kernelUid);
                info.inplace = kernel.inplace;
                info.isView = kernel.isView;
                if (info.inplace && kernel.numInputs > 0)
                {
                    info.inplace_idx = 0; // TODO: populate this somehow
                }

                if (enode.opType == OpType::SCATTER)
                {
                    info.isScatter = true;
                }
                else if (enode.opType == OpType::FUSED)
                {
                    const auto *refEntry = ReferenceGraphRegistry::get().getFactory(kernel.opName);
                    if (refEntry)
                    {
                        Graph pGraph;
                        std::vector<uint32_t> pInputs;
                        for (size_t k = 0; k < kernel.numInputs; ++k)
                        {
                            pInputs.push_back(pGraph.input(kernel.dummyShapes[k], kernel.dtypes[k]));
                        }
                        uint32_t pRoot = refEntry->factory(pInputs, pGraph);
                        if (pGraph.getNode(pRoot).opType == OpType::SCATTER)
                        {
                            info.isScatter = true;
                        }
                    }
                }
            }

            if (enode.opType == OpType::INPUT || enode.opType == OpType::CACHE)
            {
                info.cost = 0.0f;
                if (strictCache && (enode.leafId & 0x80000000))
                {
                    uint32_t e_class_id = egraph.getENodeEClass(i);
                    uint32_t canonId = egraph.findConst(e_class_id);
                    uint32_t logicalId = eclassToLogical.count(canonId) ? eclassToLogical.at(canonId) : UINT32_MAX;
                    if (logicalId == UINT32_MAX || cachedNodes.find(logicalId) == cachedNodes.end())
                    {
                        info.cost = TGConstants::INF;
                    }
                    else if (enode.backend != cachedNodes.at(logicalId))
                    {
                        info.cost = TGConstants::INF;
                    }
                }
            }
            else if (enode.kernelUid != 0)
            {

                std::vector<std::vector<uint32_t>> inShapes;
                std::vector<std::vector<uint64_t>> inStrides;
                std::vector<DType> inDTypes;
                std::vector<std::vector<uint8_t>> inConstants;

                inShapes.reserve(enode.children.size());
                inStrides.reserve(enode.children.size());
                inDTypes.reserve(enode.children.size());
                inConstants.reserve(enode.children.size());

                const ReferenceGraphEntry *refEntry = nullptr;
                std::unique_ptr<Graph> pGraph;
                std::vector<uint32_t> pInputs;

                const auto &kernel = KernelRegistry::get().getKernel(enode.kernelUid);
                if (enode.opType == OpType::FUSED)
                {
                    refEntry = ReferenceGraphRegistry::get().getFactory(kernel.opName);
                    if (refEntry)
                    {
                        pGraph = std::make_unique<Graph>();
                        for (size_t k = 0; k < kernel.numInputs; ++k)
                        {
                            pInputs.push_back(pGraph->input(kernel.dummyShapes[k], kernel.dtypes[k]));
                        }
                        refEntry->factory(pInputs, *pGraph);
                    }
                }

                for (size_t j = 0; j < enode.children.size(); j++)
                {
                    uint32_t childEClassId = enode.children[j];
                    const EClass &childCls = egraph.getEClass(egraph.find(childEClassId));
                    inShapes.push_back(childCls.shape);

                    std::vector<uint64_t> strides_cast;
                    strides_cast.reserve(childCls.strides.size());
                    for (uint64_t s : childCls.strides)
                        strides_cast.push_back(s);
                    inStrides.push_back(std::move(strides_cast));

                    inDTypes.push_back(childCls.dtype);

                    uint32_t canonChild = egraph.find(childEClassId);
                    bool needed = false;

                    if (enode.opType == OpType::FUSED)
                    {
                        if (refEntry && pGraph)
                        {
                            auto traceToInputIdx = [&](uint32_t pid) -> int
                            {
                                uint32_t curr = pid;
                                while (pGraph->hasNode(curr) &&
                                       (pGraph->getNode(curr).opType == OpType::CONTIGUOUS ||
                                        pGraph->getNode(curr).opType == OpType::CAST ||
                                        pGraph->getNode(curr).opType == OpType::COPY_TO ||
                                        pGraph->getNode(curr).opType == OpType::RESHAPE ||
                                        pGraph->getNode(curr).opType == OpType::PERMUTE))
                                {
                                    if (pGraph->getNode(curr).parentIds.empty())
                                        break;
                                    curr = pGraph->getNode(curr).parentIds[0];
                                }
                                for (size_t k = 0; k < pInputs.size(); ++k)
                                {
                                    if (pInputs[k] == curr)
                                        return (int)k;
                                }
                                return -1;
                            };

                            for (const auto &pair : pGraph->nodes)
                            {
                                const TensorNode &n = pair.second;
                                for (size_t p_idx = 0; p_idx < n.parentIds.size(); ++p_idx)
                                {
                                    if (isConstantNeeded(n.opType, p_idx, n.parentIds.size()))
                                    {
                                        int inputIdx = traceToInputIdx(n.parentIds[p_idx]);
                                        if (kernel.isVariadic)
                                        {
                                            if (inputIdx == (int)kernel.numInputs - 1 && j == enode.children.size() - 1)
                                            {
                                                needed = true;
                                                break;
                                            }
                                            else if (inputIdx >= 0 && inputIdx < (int)kernel.numInputs - 1 && j < enode.children.size() - 1)
                                            {
                                                needed = true;
                                                break;
                                            }
                                        }
                                        else if (inputIdx == (int)j)
                                        {
                                            needed = true;
                                            break;
                                        }
                                    }
                                }
                                if (needed)
                                    break;
                            }
                        }
                    }
                    else
                    {
                        needed = isConstantNeeded(enode.opType, j, enode.children.size());
                    }

                    if (!needed)
                    {
                        inConstants.push_back({});
                    }
                    else if (egraph.constantStaging.count(canonChild))
                    {
                        inConstants.push_back(*egraph.constantStaging.at(canonChild));
                    }
                    else
                    {
                        inConstants.push_back({});
                    }
                }

                info.cost = costModel.estimateCost(
                    enode.kernelUid,
                    enode.shape,
                    enode.strides,
                    enode.dtype,
                    inShapes, inStrides, inDTypes, inConstants);

                if (info.inplace && info.inplace_idx >= 0)
                {
                    uint32_t mutated_eclass = egraph.find(enode.children[info.inplace_idx]);
                    if (immutable_eclasses.count(mutated_eclass) && !info.isScatter)
                    {
                        info.cost = TGConstants::INF;
                    }
                }
            }
            else
            {
                Error::throw_err("[Planner.extractBest] enode.kernelUid != 0, but isn't OpType::INPUT or OpType::CACHE. this shouldn't happen");
            }

            enodeInfos[i] = std::move(info);
            timer3.tick();
        }

        bool droppedInf = false;
        size_t totalPruned = 0;
        for (size_t i = 0; i < egraph.getClasses().size(); ++i)
        {
            uint32_t e_class_id = egraph.find(static_cast<uint32_t>(i));
            if (e_class_id != i)
                continue;

            EClass &cls = egraph.getEClass(e_class_id);
            std::vector<uint32_t> validEnodes;
            validEnodes.reserve(cls.enodes.size());

            // 1. Remove infinite-cost nodes
            for (uint32_t enodeId : cls.enodes)
            {
                if (enodeInfos[enodeId].cost == TGConstants::INF)
                {
                    droppedInf = true;
                }
                else
                {
                    validEnodes.push_back(enodeId);
                }
            }

            // 2. Prune duplicated nodes to minimize search space bloat
            std::vector<uint32_t> deduped;
            deduped.reserve(validEnodes.size());
            for (size_t idxA = 0; idxA < validEnodes.size(); ++idxA)
            {
                uint32_t idA = validEnodes[idxA];
                const ENode &a = egraph.getENodes()[idA];
                const ENodeInfo &ia = enodeInfos[idA];

                bool dominated = false;
                for (size_t idxB = 0; idxB < validEnodes.size(); ++idxB)
                {
                    if (idxA == idxB)
                        continue;
                    uint32_t idB = validEnodes[idxB];
                    const ENode &b = egraph.getENodes()[idB];
                    const ENodeInfo &ib = enodeInfos[idB];

                    // Require FULL structural equality (including kernelUid, opType, leafId, opName,
                    // shape, dtype) — not just the relaxed signature from the old check.
                    const bool sameStruct =
                        a.kernelUid == b.kernelUid &&
                        a.opType == b.opType &&
                        a.leafId == b.leafId &&
                        a.opName == b.opName &&
                        a.children == b.children &&
                        a.backend == b.backend &&
                        a.shape == b.shape &&
                        a.dtype == b.dtype &&
                        a.strides == b.strides &&
                        a.viewOffset == b.viewOffset &&
                        ia.inplace == ib.inplace &&
                        ia.inplace_idx == ib.inplace_idx &&
                        ia.isView == ib.isView &&
                        ia.isScatter == ib.isScatter;

                    if (!sameStruct)
                        continue;

                    // B dominates A only if strictly cheaper, OR equal-cost with smaller ID.
                    if (ib.cost < ia.cost - 1e-9f)
                    {
                        dominated = true;
                        break;
                    }
                    if (std::abs(ib.cost - ia.cost) <= 1e-9f && idB < idA)
                    {
                        dominated = true;
                        break;
                    }
                }
                if (!dominated)
                    deduped.push_back(idA);
            }
            totalPruned += (validEnodes.size() - deduped.size());
            cls.enodes = std::move(deduped);
        }

        if (totalPruned > 0)
        {
            std::cout << "[Planner.extractBest] Pruned " << totalPruned << " dominated enodes from the search space." << std::endl;
        }

        if (droppedInf)
        {
            std::cout << "[Planner.extractBest] Warning: Filtered out nodes with infinite cost. "
                      << "You may need to run 'bench' to gather missing kernel performance data." << std::endl;
        }

        auto rootIt = nodeToEClass.find(rootId);
        if (rootIt == nodeToEClass.end())
        {
            Error::throw_err("[Planner.extractBest] Root node missing from nodeToEClass.");
        }
        uint32_t rootEClassId = egraph.find(rootIt->second);

        const size_t numClasses = egraph.getClasses().size();

        std::vector<uint32_t> canonicalClasses;
        std::vector<uint32_t> classToBitIdx(numClasses, UINT32_MAX);
        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t e_class_id = egraph.find(static_cast<uint32_t>(i));
            if (e_class_id == i)
            {
                classToBitIdx[i] = static_cast<uint32_t>(canonicalClasses.size());
                canonicalClasses.push_back(i);
            }
        }

        const size_t numCanonical = canonicalClasses.size();
        const size_t bitWords = numCanonical == 0 ? 0 : (numCanonical + 63) >> 6;

        auto bitTest = [&](const std::vector<uint64_t> &bits, uint32_t e_class_id) -> bool
        {
            uint32_t idx = classToBitIdx[e_class_id];
            if (idx == UINT32_MAX || bits.empty())
                return false;
            return (bits[idx >> 6] >> (idx & 63)) & 1ULL;
        };

        auto bitSet = [&](std::vector<uint64_t> &bits, uint32_t e_class_id)
        {
            uint32_t idx = classToBitIdx[e_class_id];
            if (idx != UINT32_MAX && !bits.empty())
            {
                bits[idx >> 6] |= (1ULL << (idx & 63));
            }
        };

        struct OptSummary
        {
            float cost = TGConstants::INF;
            float intrinsic = TGConstants::INF;
            uint32_t chosenEnode = UINT32_MAX;
            std::vector<uint64_t> coveredBits;
            bool valid = false;
        };

        std::vector<OptSummary> opt(numClasses);
        for (uint32_t canonId : canonicalClasses)
        {
            opt[canonId].coveredBits.assign(bitWords, 0);
        }

        std::vector<std::vector<uint32_t>> parentMap(numClasses);
        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t e_class_id = egraph.find(static_cast<uint32_t>(i));
            if (e_class_id != i)
                continue;

            const EClass &cls = egraph.getEClass(e_class_id);
            for (uint32_t enodeId : cls.enodes)
            {
                const ENode &enode = egraph.getENodes()[enodeId];
                for (uint32_t child : enode.children)
                {
                    uint32_t childEClass = egraph.find(child);
                    parentMap[childEClass].push_back(e_class_id);
                }
            }
        }

        for (auto &parents : parentMap)
        {
            std::sort(parents.begin(), parents.end());
            parents.erase(std::unique(parents.begin(), parents.end()), parents.end());
        }

        std::vector<uint32_t> worklist;
        std::vector<uint32_t> next_worklist;
        std::vector<bool> inQueue(numClasses, false);

        worklist.reserve(numClasses);
        next_worklist.reserve(numClasses);

        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t e_class_id = egraph.find(static_cast<uint32_t>(i));
            if (e_class_id == i)
            {
                worklist.push_back(e_class_id);
                inQueue[e_class_id] = true;
            }
        }

        std::vector<uint64_t> candidateBits(bitWords, 0);
        std::vector<float> optimisticEnodeDagCost(egraph.getENodes().size(), TGConstants::INF);

        ProgressTimer optTimer(0, "calculating optimistic cost");
        while (!worklist.empty())
        {
            for (uint32_t e_class_id : worklist)
            {
                inQueue[e_class_id] = false;

                const EClass &cls = egraph.getEClass(e_class_id);
                OptSummary best;
                best.coveredBits.assign(bitWords, 0);

                for (uint32_t enodeId : cls.enodes)
                {
                    const ENodeInfo &info = enodeInfos[enodeId];
                    if (info.cost == TGConstants::INF)
                        continue;

                    std::fill(candidateBits.begin(), candidateBits.end(), 0);
                    float candidateCost = info.cost;
                    bool candidateValid = true;

                    bitSet(candidateBits, e_class_id);

                    const ENode &enode = egraph.getENodes()[enodeId];

                    std::vector<uint32_t> childEClasses;
                    childEClasses.reserve(enode.children.size());
                    for (uint32_t child : enode.children)
                    {
                        childEClasses.push_back(egraph.find(child));
                    }
                    std::sort(childEClasses.begin(), childEClasses.end());
                    childEClasses.erase(std::unique(childEClasses.begin(), childEClasses.end()), childEClasses.end());

                    for (uint32_t childEClass : childEClasses)
                    {
                        if (childEClass == e_class_id)
                        {
                            candidateValid = false;
                            break;
                        }

                        const OptSummary &childOpt = opt[childEClass];
                        if (!childOpt.valid)
                        {
                            candidateValid = false;
                            break;
                        }

                        if (bitTest(childOpt.coveredBits, e_class_id))
                        {
                            candidateValid = false;
                            break;
                        }

                        for (size_t w = 0; w < bitWords; ++w)
                        {
                            uint64_t newBits = childOpt.coveredBits[w] & ~candidateBits[w];
                            if (!newBits)
                                continue;

                            candidateBits[w] |= newBits;

                            while (newBits)
                            {
#if defined(__GNUG__) || defined(__clang__)
                                unsigned bit = static_cast<unsigned>(__builtin_ctzll(newBits));
#else
                                unsigned bit = 0;
                                uint64_t tmp = newBits;
                                while ((tmp & 1ULL) == 0)
                                {
                                    tmp >>= 1;
                                    ++bit;
                                }
#endif
                                uint32_t k_idx = static_cast<uint32_t>((w << 6) + bit);
                                if (k_idx < numCanonical)
                                {
                                    uint32_t k_eclass = canonicalClasses[k_idx];
                                    candidateCost += opt[k_eclass].intrinsic;
                                }
                                newBits &= (newBits - 1);
                            }
                        }
                    }

                    if (!candidateValid)
                        continue;

                    optimisticEnodeDagCost[enodeId] = candidateCost;

                    if (!best.valid ||
                        candidateCost < best.cost - EPS ||
                        (std::abs(candidateCost - best.cost) <= EPS && enodeId < best.chosenEnode))
                    {
                        best.valid = true;
                        best.cost = candidateCost;
                        best.intrinsic = info.cost;
                        best.chosenEnode = enodeId;
                        best.coveredBits = candidateBits;
                    }
                }

                if (!best.valid)
                    continue;

                if (!opt[e_class_id].valid ||
                    best.cost < opt[e_class_id].cost - EPS ||
                    (std::abs(best.cost - opt[e_class_id].cost) <= EPS && best.chosenEnode < opt[e_class_id].chosenEnode))
                {
                    opt[e_class_id] = std::move(best);

                    for (uint32_t parentId : parentMap[e_class_id])
                    {
                        if (!inQueue[parentId])
                        {
                            inQueue[parentId] = true;
                            next_worklist.push_back(parentId);
                        }
                    }
                }
            }

            worklist.clear();
            std::swap(worklist, next_worklist);
            optTimer.tick();
        }

        std::vector<float> eclassMinCost(numClasses, TGConstants::INF);
        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t e_class_id = egraph.find(static_cast<uint32_t>(i));
            if (e_class_id == i && opt[e_class_id].valid)
            {
                eclassMinCost[e_class_id] = opt[e_class_id].cost;
            }
        }

        std::vector<uint64_t> tempBits(bitWords, 0);
        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t e_class_id = egraph.find(static_cast<uint32_t>(i));
            if (e_class_id != i)
                continue;

            const EClass &cls = egraph.getEClass(e_class_id);
            for (uint32_t enodeId : cls.enodes)
            {
                const ENodeInfo &info = enodeInfos[enodeId];
                if (info.cost == TGConstants::INF)
                {
                    optimisticEnodeDagCost[enodeId] = TGConstants::INF;
                    continue;
                }

                std::fill(tempBits.begin(), tempBits.end(), 0);
                bitSet(tempBits, e_class_id);

                float total = info.cost;
                bool valid = true;

                const ENode &enode = egraph.getENodes()[enodeId];

                std::vector<uint32_t> childEClasses;
                childEClasses.reserve(enode.children.size());
                for (uint32_t child : enode.children)
                {
                    childEClasses.push_back(egraph.find(child));
                }
                std::sort(childEClasses.begin(), childEClasses.end());
                childEClasses.erase(std::unique(childEClasses.begin(), childEClasses.end()), childEClasses.end());

                for (uint32_t childEClass : childEClasses)
                {
                    if (childEClass == e_class_id)
                    {
                        valid = false;
                        break;
                    }
                    if (!opt[childEClass].valid)
                    {
                        valid = false;
                        break;
                    }
                    if (bitTest(opt[childEClass].coveredBits, e_class_id))
                    {
                        valid = false;
                        break;
                    }

                    for (size_t w = 0; w < bitWords; ++w)
                    {
                        uint64_t newBits = opt[childEClass].coveredBits[w] & ~tempBits[w];
                        if (!newBits)
                            continue;

                        tempBits[w] |= newBits;

                        while (newBits)
                        {
#if defined(__GNUG__) || defined(__clang__)
                            unsigned bit = static_cast<unsigned>(__builtin_ctzll(newBits));
#else
                            unsigned bit = 0;
                            uint64_t tmp = newBits;
                            while ((tmp & 1ULL) == 0)
                            {
                                tmp >>= 1;
                                ++bit;
                            }
#endif
                            uint32_t k_idx = static_cast<uint32_t>((w << 6) + bit);
                            if (k_idx < numCanonical)
                            {
                                uint32_t k_eclass = canonicalClasses[k_idx];
                                total += opt[k_eclass].intrinsic;
                            }
                            newBits &= (newBits - 1);
                        }
                    }
                }

                optimisticEnodeDagCost[enodeId] = valid ? total : TGConstants::INF;
            }
        }

        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t e_class_id = egraph.find(static_cast<uint32_t>(i));
            if (e_class_id != i)
                continue;

            EClass &cls = egraph.getEClass(e_class_id);
            std::sort(cls.enodes.begin(), cls.enodes.end(),
                      [&](uint32_t a, uint32_t b)
                      {
                          float costA = optimisticEnodeDagCost[a];
                          float costB = optimisticEnodeDagCost[b];

                          if (costA < costB)
                              return true;
                          if (costA > costB)
                              return false;
                          return a < b;
                      });
        }

        if (egraph.getEClass(rootEClassId).enodes.size() == 0)
        {
            Error::throw_err("[Planner.extractBest] no valid extractions");
        }
        std::cout << "[Planner.extractBest] Optimistic root cost: "
                  << std::to_string(optimisticEnodeDagCost[egraph.getEClass(rootEClassId).enodes[0]]) << std::endl;
        if (std::isinf(optimisticEnodeDagCost[egraph.getEClass(rootEClassId).enodes[0]]))
        {
            Error::throw_err("[Planner.extractBest] cannot have inf root cost");
        }

        PrecompData precomp;
        precomp.is_eclass_persistent.assign(numClasses, false);
        precomp.is_eclass_cached.assign(numClasses, false);
        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t e_class_id = egraph.find(static_cast<uint32_t>(i));
            if (e_class_id == i)
            {
                uint32_t logicalId = eclassToLogical.count(e_class_id) ? eclassToLogical.at(e_class_id) : UINT32_MAX;
                if (logicalId != UINT32_MAX && graph.hasNode(logicalId) && graph.getNode(logicalId).storageType == StorageType::PERSISTENT)
                    precomp.is_eclass_persistent[e_class_id] = true;
                if (egraph.constantStaging.count(e_class_id))
                    precomp.is_eclass_persistent[e_class_id] = true;

                if (logicalId != UINT32_MAX && cachedNodes.count(logicalId))
                    precomp.is_eclass_cached[e_class_id] = true;
            }
        }

        auto baseline_map = computeStaticBaseline(graph, cachedNodes);
        for (int i = 0; i < (uint32_t)Backend::_COUNT; ++i)
        {
            precomp.static_baseline_arr[i] = 0;
            precomp.mem_limits_arr[i] = std::numeric_limits<uint64_t>::max();
        }
        for (const auto &kv : baseline_map)
            precomp.static_baseline_arr[(uint32_t)kv.first] = kv.second;
        for (const auto &kv : maxMemoryByBackend)
            precomp.mem_limits_arr[(uint32_t)kv.first] = kv.second;

        precomp.enode_canon_children.resize(egraph.getENodes().size());
        for (size_t i = 0; i < egraph.getENodes().size(); ++i)
        {
            for (uint32_t c : egraph.getENodes()[i].children)
            {
                precomp.enode_canon_children[i].push_back(egraph.findConst(c));
            }
        }

        std::vector<uint32_t> ref_counts(numClasses, 0);
        std::vector<bool> processed_mem(numClasses, false);
        std::vector<uint32_t> sim_aliasMap(numClasses, UINT32_MAX);
        std::vector<bool> visit_visited(numClasses, false);

        std::vector<uint32_t> topo_order;
        topo_order.reserve(numClasses);
        std::vector<bool> val_visited_classes(numClasses, false);
        std::vector<uint32_t> val_overwritten(numClasses, UINT32_MAX);
        std::vector<uint32_t> val_mem_root(numClasses);

        std::vector<uint32_t> indegree(numClasses, 0);
        std::vector<uint32_t> zero_indegree;
        zero_indegree.reserve(numClasses);

        Extractor extractor = Extractor(enodeInfos);

        float best_cost = TGConstants::INF;
        std::unordered_map<uint32_t, uint32_t> best_selection_map;
        std::unordered_map<Backend, uint64_t> minPeakMemSeen;

        int max_iters = 100000;
        int remaining_iters = max_iters;
        ProgressTimer timer(max_iters, "extracting graphs ");
        ProgressTimer loopTimer(0, "", true);
        while (remaining_iters-- > 0)
        {
            std::cout << "loop " << std::to_string(loopTimer.getElapsed() * 1000) << "ms";
            if (max_iters - (remaining_iters + 1) > 0)
            {
                std::cout << ", avg loop " << std::to_string(timer.getElapsed() * 1000 / (max_iters - (remaining_iters + 1))) << "ms";
            }
            std::cout << std::endl;
            loopTimer.reset();
            timer.tick();

            const std::unordered_map<uint32_t, uint32_t> &selection_map = extractor.getNextSelection();

            if (valid)
            {
                std::fill(indegree.begin(), indegree.end(), 0);
                for (const auto &kv : selection_map)
                {
                    uint32_t sel = kv.second;
                    uint32_t enode_id = egraph.getEClass(kv.first).enodes[sel];
                    for (uint32_t child : precomp.enode_canon_children[enode_id])
                    {
                        indegree[child]++;
                    }
                }

                zero_indegree.clear();
                for (const auto &kv : selection_map)
                {
                    if (indegree[kv.first] == 0)
                    {
                        zero_indegree.push_back(kv.first);
                    }
                }

                uint32_t processed = 0;
                while (!zero_indegree.empty())
                {
                    uint32_t curr = zero_indegree.back();
                    zero_indegree.pop_back();
                    processed++;

                    uint32_t sel = selection_map[curr];
                    uint32_t enode_id = egraph.getEClass(curr).enodes[sel];
                    for (uint32_t canonChild : precomp.enode_canon_children[enode_id])
                    {
                        indegree[canonChild]--;
                        if (indegree[canonChild] == 0)
                        {
                            zero_indegree.push_back(canonChild);
                        }
                    }
                }

                if (processed < selection_map.size())
                {
                    valid = false;
                    reason = "cycle";
                }
            }

            uint32_t conflictEClass1 = UINT32_MAX;
            uint32_t conflictEClass2 = UINT32_MAX;

            if (valid)
            {
                valid = validateInplaceSchedulesFast(egraph, selection_map, enodeInfos, rootEClassId, eclassToLogical, precomp, topo_order, val_visited_classes, val_overwritten, val_mem_root, reason, conflictEClass1, conflictEClass2);
            }

            float current_cost = TGConstants::INF;
            std::unordered_map<Backend, uint64_t> peak;
            uint32_t oomEClassId = UINT32_MAX;
            std::vector<bool> oomLiveAllocated(egraph.getClasses().size(), false);

            if (valid)
            {
                std::unordered_map<uint32_t, SimNode> simNodes;
                current_cost = simulateExecutionTime(egraph, selection_map, enodeInfos, precomp, topo_order, simNodes);

                AllocResult alloc_res = allocateStaticMemory(simNodes, topo_order, maxMemoryByBackend, egraph.getClasses().size());

                if (!alloc_res.fits)
                {
                    valid = false;
                    reason = "OOM";
                    oomEClassId = alloc_res.oomEClassId;
                    oomLiveAllocated = alloc_res.oomLiveAllocated;
                }
                else
                {
                    for (const auto &kv : alloc_res.peak_mem)
                    {
                        Backend rep_b = Backend::CPU;
                        if (kv.first == 0)
                            rep_b = Backend::STORAGE;
                        else if (kv.first == 2)
                            rep_b = Backend::CUDA;
                        else if (kv.first == 3)
                            rep_b = Backend::OPENCL;
                        peak[rep_b] = kv.second;
                    }

                    bool newMinSeen = false;
                    for (const auto &kv : peak)
                    {
                        auto mit = minPeakMemSeen.find(kv.first);
                        if (mit == minPeakMemSeen.end() || kv.second < mit->second)
                        {
                            minPeakMemSeen[kv.first] = kv.second;
                            newMinSeen = true;
                        }
                    }
                    if (newMinSeen)
                    {
                        std::cout << "[Planner.extractBest] New lowest peak mem candidate seen (iter " << (max_iters - remaining_iters) << "): ";
                        for (auto it = minPeakMemSeen.begin(); it != minPeakMemSeen.end(); ++it)
                        {
                            if (it != minPeakMemSeen.begin())
                                std::cout << ", ";
                            std::cout << toString(it->first) << ": " << it->second << " bytes";
                        }
                        std::cout << std::endl;
                    }
                }
            }

            if (valid)
            {
                if (current_cost < best_cost)
                {
                    best_cost = current_cost;
                    best_selection_map = selection_map;
                    std::cout << "new best cost: " << std::to_string(best_cost) << std::endl;
                    std::cout << "peak mem: " << toString(peak) << std::endl;
                }

                if (stopOnFirstValid)
                {
                    break;
                }
            }

            if (!valid)
            {
                std::cout << "[Planner.extractBest] [iter " << std::to_string(max_iters - remaining_iters) << "] invalid reason: " << reason << std::endl;
            }

            if (to_process_enode.empty())
                break;

            uint32_t target_backtrack_eclass = UINT32_MAX;
            if (!valid && conflictEClass1 != UINT32_MAX && conflictEClass2 != UINT32_MAX)
            {
                int idx1 = -1;
                int idx2 = -1;
                for (int i = 0; i < (int)path.size(); ++i)
                {
                    if (path[i] == conflictEClass1)
                        idx1 = i;
                    if (path[i] == conflictEClass2)
                        idx2 = i;
                }
                if (idx1 >= 0 && idx2 >= 0)
                {
                    target_backtrack_eclass = path[std::max(idx1, idx2)];
                }
            }
            else if (!valid && reason == "OOM" && oomEClassId != UINT32_MAX)
            {
                int backtrack_idx = -1;
                // Scan path to find first node that is actually live in memory and has alternative kernel paths to try
                for (int i = 0; i < path.size(); i++)
                {
                    uint32_t e_class_id = path[i];
                    uint32_t canon = egraph.findConst(e_class_id);

                    if (canon < oomLiveAllocated.size() && oomLiveAllocated[canon])
                    {
                        auto choiceIt = selection_map.find(canon);
                        if (choiceIt != selection_map.end())
                        {
                            uint32_t sel = choiceIt->second;
                            const auto &enodes = egraph.getEClass(canon).enodes;
                            if (sel + 1 < enodes.size())
                            {
                                backtrack_idx = i;
                                break;
                            }
                        }
                    }
                }

                if (backtrack_idx >= 0)
                {
                    target_backtrack_eclass = path[backtrack_idx];
                }
            }
            else if (!valid && reason == "cycle")
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
                    path_idx[egraph.findConst(path[i])] = i;
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

            bool skip_increment = (target_backtrack_eclass != UINT32_MAX);
            std::cout << "path size " << std::to_string(path.size()) << std::endl;

            while (!path.empty())
            {
                uint32_t current = path.back();
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
                uint32_t enode_id = enodes[sel];
                const ENode &node = egraph.getENodes()[enode_id];
                const ENodeInfo &info = enodeInfos[enode_id];

                if (!skip_increment && sel + 1 < enodes.size())
                {
                    next_sel[current] = sel + 1;

                    std::vector<uint32_t> keys_to_delete;
                    keys_to_delete.reserve(selection_map.size());
                    for (const auto &kv : selection_map)
                    {
                        if (std::find(path.begin(), path.end(), kv.first) == path.end() && kv.first != current)
                        {
                            keys_to_delete.push_back(kv.first);
                        }
                    }
                    for (uint32_t k : keys_to_delete)
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
                    for (uint32_t eclass : path)
                    {
                        uint32_t n_id = egraph.getEClass(eclass).enodes[selection_map[eclass]];
                        const ENode &n = egraph.getENodes()[n_id];
                        std::vector<uint32_t> new_to_process;
                        new_to_process.reserve(n.children.size());
                        for (uint32_t child : n.children)
                        {
                            uint32_t childEClass = egraph.find(child);
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

        if (best_cost == TGConstants::INF)
        {
            Error::throw_err("[Planner.extractBest] no valid extraction found under given constraints. try running bench");
        }

        ExtractionResult result;
        result.totalCost = best_cost;

        for (auto const &kv : best_selection_map)
        {
            ExtractChoice c;
            c.enodeId = egraph.getEClass(kv.first).enodes[kv.second];
            c.cost = enodeInfos[c.enodeId].cost;
            c.valid = true;
            result.choiceByEClass[kv.first] = c;
        }

        return result;
    }

    CompiledGraph buildCompiledGraph(
        uint32_t rootId,
        const Graph &graph,
        EGraph &egraph,
        const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
        const ExtractionResult &extraction,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        const std::unordered_map<uint32_t, uint32_t> &eclassToLogical)
    {
        CompiledGraph compiled;

        std::vector<uint32_t> topo;
        std::unordered_set<uint32_t> visited_classes;
        std::function<void(uint32_t)> visit = [&](uint32_t e_class_id)
        {
            e_class_id = egraph.find(e_class_id);
            if (visited_classes.count(e_class_id))
                return;
            visited_classes.insert(e_class_id);

            auto choiceIt = extraction.choiceByEClass.find(e_class_id);
            if (choiceIt == extraction.choiceByEClass.end() || !choiceIt->second.valid)
                return;

            const ENode &enode = egraph.getENodes()[choiceIt->second.enodeId];
            for (uint32_t child : enode.children)
                visit(child);
            topo.push_back(e_class_id);
        };

        uint32_t rootEClassId = egraph.find(nodeToEClass.at(rootId));
        visit(rootEClassId);

        std::unordered_map<uint32_t, uint32_t> eclassToPhys;
        for (uint32_t e_class_id : topo)
        {
            eclassToPhys[e_class_id] = GlobalNextPhysId++;
        }

        std::unordered_map<uint32_t, uint32_t> lastPhysIdForLogical;
        for (uint32_t e_class_id : topo)
        {
            uint32_t logicalId = eclassToLogical.count(e_class_id) ? eclassToLogical.at(e_class_id) : UINT32_MAX;
            if (logicalId != UINT32_MAX)
            {
                lastPhysIdForLogical[logicalId] = eclassToPhys[e_class_id];
            }
        }

        for (uint32_t e_class_id : topo)
        {
            const ExtractChoice &choice = extraction.choiceByEClass.at(e_class_id);
            const ENode &enode = egraph.getENodes()[choice.enodeId];
            uint32_t logicalId = eclassToLogical.count(e_class_id) ? eclassToLogical.at(e_class_id) : UINT32_MAX;

            uint32_t physId = eclassToPhys[e_class_id];

            TensorNode tNode;
            tNode.id = physId;
            tNode.opType = enode.opType;
            tNode.opName = enode.opName;
            tNode.dtype = enode.dtype;
            tNode.setShape(enode.shape);
            tNode.strides = enode.strides;
            tNode.viewOffset = enode.viewOffset;
            tNode.backend = enode.backend;
            tNode.parentIds.reserve(enode.children.size());
            for (uint32_t c : enode.children)
                tNode.parentIds.push_back(eclassToPhys[egraph.find(c)]);

            OpInstruction inst;
            inst.nodeId = physId;
            inst.logicalNodeId = logicalId;
            inst.inputNodeIds = tNode.parentIds;
            inst.mem_space = enode.mem_space;
            inst.engine = enode.engine;
            inst.fullKernelId = enode.kernelUid;

            if (enode.kernelUid != 0)
            {
                const KernelEntry &kEntry = KernelRegistry::get().getKernel(enode.kernelUid);
                inst.inplaceInputIndex = kEntry.inplace ? 0 : -1;
                inst.viewInputIndex = kEntry.isView ? 0 : -1;
            }

            if (logicalId != UINT32_MAX)
            {
                if (graph.hasNode(logicalId))
                {
                    tNode.debugOrigin = graph.getNode(logicalId).debugOrigin;
                }
            }

            if (egraph.constantStaging.count(e_class_id))
            {
                compiled.constantStaging[physId] = egraph.constantStaging.at(e_class_id);
            }

            if (enode.opType != OpType::INPUT && enode.opType != OpType::CACHE)
            {
                compiled.instructions.push_back(inst);
            }

            compiled.nodesMap[physId] = tNode;
            compiled.nodeCosts[physId] = choice.cost;
            compiled.physicalToLogicalNodeMap[physId] = logicalId;
        }

        std::unordered_map<uint32_t, uint32_t> compiledRefCounts;
        for (const auto &inst : compiled.instructions)
        {
            for (uint32_t pid : inst.inputNodeIds)
            {
                compiledRefCounts[pid]++;
            }
        }
        uint32_t rootPhysId = eclassToPhys[rootEClassId];
        compiledRefCounts[rootPhysId] = std::max<uint32_t>(1, compiledRefCounts[rootPhysId]);
        compiled.refCounts = compiledRefCounts;

        for (const auto &pair : compiled.nodesMap)
        {
            const TensorNode &node = pair.second;
            if ((node.opType == OpType::INPUT || node.opType == OpType::CACHE) && node.storageType == StorageType::TRANSIENT)
            {
                uint32_t logicalId = compiled.getLogicalId(node.id);
                if (logicalId == UINT32_MAX)
                {
                    Error::throw_err("[buildCompiledGraph] Orphan cache INPUT/CACHE node " + std::to_string(node.id) + " has no logicalId mapping and is TRANSIENT. This will crash at runtime.");
                }
            }
        }

        return compiled;
    }

    struct BaseEGraphState
    {
        EGraph egraph;
        std::unordered_map<LogicalId, EClassId> nodeToEClass;
        std::unordered_map<EClassId, LogicalId> eclassToLogical;
    };

    BaseEGraphState baseState;
    bool baseStateInitialized = false;

    // Initialize baseState.egraph from graph
    void initBaseEGraph(LogicalId rootId, Graph &graph, bool doSaturate, Repo *repo = nullptr)
    {
        if (baseStateInitialized)
            return;

        std::vector<LogicalId> topo = topologicalSort({rootId}, graph);

        inferShapes(topo, graph);

        baseState.nodeToEClass.reserve(graph.nodes.size());

        MemSpace storage = MemSpace(0, HandleType::STORAGE);
        MemSpace ram = MemSpace(1, HandleType::CPP);
        Engine cpu = Engine(0, EngineType::CPU);

        for (LogicalId nodeId : topo)
        {
            TensorNode &node = graph.getNode(nodeId);
            MemSpace mem_space = ram;
            if (node.opType == OpType::INPUT && graph.getInputDataType(nodeId) == InputDataType::STORAGE)
            {
                mem_space = storage;
            }
            EClassId e_class_id = baseState.egraph.addEClass(node.getShape(), node.strides, node.dtype, mem_space);
            baseState.nodeToEClass[nodeId] = e_class_id;
            if (graph.constantStaging.count(nodeId))
            {
                baseState.egraph.constantStaging[e_class_id] = graph.constantStaging.at(nodeId);
            }
        }

        for (LogicalId nodeId : topo)
        {
            const TensorNode &node = graph.getNode(nodeId);
            EClassId e_class_id = baseState.nodeToEClass[nodeId];

            if (node.opType == OpType::INPUT)
            {
                ENode enode;
                enode.kernelUid = 0;
                enode.opType = node.opType;
                enode.opName = node.opName;
                for (LogicalId pid : node.parentIds)
                    enode.children.push_back(baseState.nodeToEClass[pid]);
                enode.shape = node.getShape();
                enode.strides = node.strides;
                enode.dtype = node.dtype;
                enode.mem_space = graph.getInputDataType(nodeId) == InputDataType::STORAGE ? storage : ram;
                enode.engine = cpu;
                enode.leafId = nodeId;
                baseState.egraph.addENode(e_class_id, enode);
                continue;
            }

            std::vector<TensorNode> inputs;
            std::vector<MemSpace> input_mem_spaces;
            for (LogicalId pid : node.parentIds)
            {
                inputs.push_back(graph.getNode(pid));
                input_mem_spaces.push_back(graph.getInputDataType(pid) == InputDataType::STORAGE ? storage : ram);
            }

            std::vector<KernelId> refs = KernelRegistry::get().findMatchingKernels(
                node.opType, node.opName, true, inputs, node, ram, {cpu}, );

            if (refs.size() == 0)
            {
                Error::throw_err("[Planner.initBaseEGraph] couldn't find any kernels to init EClass " + toString(e_class_id) + " " + toString(baseState.egraph.getEClass(e_class_id)) + "\nNode " + toString(node, graph));
            }
            for (KernelId uid : refs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                ENode enode;
                enode.kernelUid = uid;
                enode.opType = node.opType;
                enode.opName = node.opName;
                for (uint32_t pid : node.parentIds)
                    enode.children.push_back(baseState.nodeToEClass[pid]);
                enode.shape = node.getShape();
                enode.dtype = node.dtype;
                enode.backend = node.backend;

                if (kernel.isView)
                {
                    enode.strides = node.strides;
                    enode.viewOffset = node.viewOffset;
                }
                else
                {
                    enode.strides = calcContiguousStrides(node.getShape());
                    enode.viewOffset = 0;
                }

                baseState.egraph.addENode(e_class_id, enode);
            }
        }

        for (const auto &kv : baseState.nodeToEClass)
        {
            uint32_t physId = kv.first;
            uint32_t ecl = baseState.egraph.find(kv.second);
            baseState.eclassToLogical[ecl] = physId;
        }

        baseStateInitialized = true;
    }

    bool injectPartialPath(
        EGraph &egraph,
        const Graph &graph,
        uint32_t logicalId,
        const std::vector<Region> &regions,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
        std::unordered_map<uint32_t, uint32_t> &eclassToLogical,
        bool strictCache = false)
    {
        bool injected = false;
        uint32_t E_L = egraph.find(nodeToEClass.at(logicalId));
        const TensorNode &sourceNode = graph.getNode(logicalId);

        if (sourceNode.opType == OpType::INPUT)
        {
            return injected;
        }

        bool isFullRegion = false;
        if (regions.size() == 1)
        {
            const Region &reg = regions[0];
            const auto &shape = sourceNode.getShape();
            if (reg.region.size() == shape.size())
            {
                isFullRegion = true;
                for (size_t d = 0; d < shape.size(); ++d)
                {
                    if (reg.region[d].start != 0 || reg.region[d].stop != shape[d])
                    {
                        isFullRegion = false;
                        break;
                    }
                }
            }
        }

        if (isFullRegion)
        {
            return injected;
        }

        Backend targetBackend = sourceNode.backend;
        auto it = cachedNodes.find(logicalId);
        if (it != cachedNodes.end())
        {
            targetBackend = it->second;
        }
        else if (strictCache)
        {
            return injected;
        }

        const EClass lClass = egraph.getEClass(E_L);

        uint32_t E_Cache = egraph.addEClass(lClass.shape, lClass.strides, lClass.viewOffset, lClass.dtype, targetBackend);
        ENode cacheNode;
        cacheNode.opType = OpType::CACHE;
        cacheNode.dtype = lClass.dtype;
        cacheNode.shape = lClass.shape;
        cacheNode.strides = lClass.strides;
        cacheNode.viewOffset = lClass.viewOffset;
        cacheNode.backend = targetBackend;
        cacheNode.leafId = logicalId | 0x80000000;
        egraph.addENode(E_Cache, cacheNode);

        eclassToLogical[E_Cache] = logicalId;
        uint32_t current_E = E_Cache;

        auto addConst = [&](const std::vector<int32_t> &vals)
        {
            return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, Backend::CPU, vals);
        };

        for (size_t r = 0; r < regions.size(); ++r)
        {
            const Region &recomputeRegion = regions[r];

            std::vector<uint32_t> partialShape;
            for (const Dim &d : recomputeRegion.region)
                partialShape.push_back(d.stop - d.start);

            ShapePropagator prop;
            std::vector<std::vector<Region>> dirtyInputRegions = prop.backward(sourceNode, graph, {recomputeRegion});

            std::vector<int32_t> starts, ends, steps;
            for (const Dim &d : recomputeRegion.region)
            {
                starts.push_back(d.start);
                ends.push_back(d.stop);
                steps.push_back(1);
            }

            uint32_t startsId = addConst(starts);
            uint32_t endsId = addConst(ends);
            uint32_t stepsId = addConst(steps);

            uint32_t slicedEClass = UINT32_MAX;

            if (sourceNode.opType == OpType::INPUT)
            {
                const EClass &lClass = egraph.getEClass(E_L);
                std::vector<uint64_t> sliceStrides = lClass.strides;
                uint64_t sliceViewOffset = lClass.viewOffset;

                for (size_t d = 0; d < starts.size(); ++d)
                {
                    int32_t start = starts[d];
                    if (start < 0)
                        start += lClass.shape[d];
                    sliceViewOffset += start * sliceStrides[d];
                    sliceStrides[d] *= steps[d];
                }

                slicedEClass = egraph.addEClass(partialShape, sliceStrides, sliceViewOffset, lClass.dtype, lClass.backend);
                ENode sliceNode;
                sliceNode.opType = OpType::SLICE;
                sliceNode.children = {E_L, startsId, endsId, stepsId};
                sliceNode.shape = partialShape;
                sliceNode.strides = sliceStrides;
                sliceNode.viewOffset = sliceViewOffset;
                sliceNode.dtype = lClass.dtype;
                sliceNode.backend = lClass.backend;

                TensorNode dOut;
                dOut.setShape(partialShape);
                dOut.dtype = sliceNode.dtype;
                dOut.backend = sliceNode.backend;
                std::vector<TensorNode> dIns(4);
                dIns[0].setShape(lClass.shape);
                dIns[0].dtype = sliceNode.dtype;
                dIns[0].backend = sliceNode.backend;
                dIns[1].setShape({(uint32_t)starts.size()});
                dIns[1].dtype = DType::INT32;
                dIns[1].backend = Backend::CPU;
                dIns[2].setShape({(uint32_t)ends.size()});
                dIns[2].dtype = DType::INT32;
                dIns[2].backend = Backend::CPU;
                dIns[3].setShape({(uint32_t)steps.size()});
                dIns[3].dtype = DType::INT32;
                dIns[3].backend = Backend::CPU;

                auto sliceRefs = KernelRegistry::get().findMatchingKernels(OpType::SLICE, "", sliceNode.backend, dIns, dOut, true);
                for (uint64_t uid : sliceRefs)
                {
                    const auto &kernel = KernelRegistry::get().getKernel(uid);
                    ENode sn = sliceNode;
                    sn.kernelUid = uid;
                    if (kernel.isView)
                    {
                        sn.strides = sliceStrides;
                        sn.viewOffset = sliceViewOffset;
                    }
                    else
                    {
                        sn.strides = calcContiguousStrides(partialShape);
                        sn.viewOffset = 0;
                    }
                    egraph.addENode(slicedEClass, sn);
                }
            }
            else
            {
                std::vector<uint32_t> slicedInputs;
                std::vector<TensorNode> dummyInputNodes;

                for (size_t p_idx = 0; p_idx < sourceNode.parentIds.size(); ++p_idx)
                {
                    uint32_t parentLogicalId = sourceNode.parentIds[p_idx];
                    uint32_t E_parent = egraph.find(nodeToEClass.at(parentLogicalId));
                    const EClass &pClass = egraph.getEClass(E_parent);

                    std::vector<Region> inputSliceRegions = dirtyInputRegions[p_idx];
                    if (inputSliceRegions.size() != 1)
                    {
                        Error::throw_err("[Planner.injectPartialPath] expected exactly 1 input slice region for parent " + std::to_string(p_idx) + " but got " + std::to_string(inputSliceRegions.size()));
                    }
                    Region inputSliceRegion = inputSliceRegions[0];

                    std::vector<uint32_t> pPartialShape;
                    for (const Dim &d : inputSliceRegion.region)
                        pPartialShape.push_back(d.stop - d.start);

                    std::vector<int32_t> pStarts, pEnds, pSteps;
                    for (const Dim &d : inputSliceRegion.region)
                    {
                        pStarts.push_back(d.start);
                        pEnds.push_back(d.stop);
                        pSteps.push_back(1);
                    }

                    uint32_t pStartsId = addConst(pStarts);
                    uint32_t pEndsId = addConst(pEnds);
                    uint32_t pStepsId = addConst(pSteps);

                    std::vector<uint64_t> pSliceStrides = pClass.strides;
                    uint64_t pSliceViewOffset = pClass.viewOffset;

                    for (size_t d = 0; d < pStarts.size(); ++d)
                    {
                        int32_t start = pStarts[d];
                        if (start < 0)
                            start += pClass.shape[d];
                        pSliceViewOffset += start * pSliceStrides[d];
                        pSliceStrides[d] *= pSteps[d];
                    }

                    uint32_t pSliceEClass = egraph.addEClass(pPartialShape, pSliceStrides, pSliceViewOffset, pClass.dtype, pClass.backend);
                    ENode pSliceNode;
                    pSliceNode.opType = OpType::SLICE;
                    pSliceNode.children = {E_parent, pStartsId, pEndsId, pStepsId};
                    pSliceNode.shape = pPartialShape;
                    pSliceNode.strides = pSliceStrides;
                    pSliceNode.viewOffset = pSliceViewOffset;
                    pSliceNode.dtype = pClass.dtype;
                    pSliceNode.backend = pClass.backend;

                    TensorNode pOut;
                    pOut.setShape(pPartialShape);
                    pOut.dtype = pSliceNode.dtype;
                    pOut.backend = pSliceNode.backend;
                    std::vector<TensorNode> pIns(4);
                    pIns[0].setShape(pClass.shape);
                    pIns[0].dtype = pSliceNode.dtype;
                    pIns[0].backend = pSliceNode.backend;
                    pIns[1].setShape({(uint32_t)pStarts.size()});
                    pIns[1].dtype = DType::INT32;
                    pIns[1].backend = Backend::CPU;
                    pIns[2].setShape({(uint32_t)pEnds.size()});
                    pIns[2].dtype = DType::INT32;
                    pIns[2].backend = Backend::CPU;
                    pIns[3].setShape({(uint32_t)pSteps.size()});
                    pIns[3].dtype = DType::INT32;
                    pIns[3].backend = Backend::CPU;

                    auto pSliceRefs = KernelRegistry::get().findMatchingKernels(OpType::SLICE, "", pSliceNode.backend, pIns, pOut, true);
                    for (uint64_t uid : pSliceRefs)
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(uid);
                        ENode sn = pSliceNode;
                        sn.kernelUid = uid;
                        if (kernel.isView)
                        {
                            sn.strides = pSliceStrides;
                            sn.viewOffset = pSliceViewOffset;
                        }
                        else
                        {
                            sn.strides = calcContiguousStrides(pPartialShape);
                            sn.viewOffset = 0;
                        }
                        egraph.addENode(pSliceEClass, sn);
                    }

                    uint32_t pContigEClass = egraph.addEClass(pPartialShape, calcContiguousStrides(pPartialShape), 0, pSliceNode.dtype, pSliceNode.backend);
                    ENode pContigNode;
                    pContigNode.opType = OpType::CONTIGUOUS;
                    pContigNode.children = {pSliceEClass};
                    pContigNode.shape = pPartialShape;
                    pContigNode.strides = calcContiguousStrides(pPartialShape);
                    pContigNode.dtype = pSliceNode.dtype;
                    pContigNode.backend = pSliceNode.backend;

                    TensorNode cOut;
                    cOut.setShape(pPartialShape);
                    cOut.dtype = pSliceNode.dtype;
                    cOut.backend = pSliceNode.backend;
                    cOut.strides = pContigNode.strides;
                    TensorNode cIn;
                    cIn.setShape(pPartialShape);
                    cIn.dtype = pSliceNode.dtype;
                    cIn.backend = pSliceNode.backend;
                    cIn.strides = pSliceStrides;

                    auto contigRefs = KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", pContigNode.backend, {cIn}, cOut, true);
                    for (uint64_t uid : contigRefs)
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(uid);
                        ENode cn = pContigNode;
                        cn.kernelUid = uid;
                        if (kernel.isView)
                        {
                            cn.strides = pSliceStrides;
                            cn.viewOffset = pSliceViewOffset;
                        }
                        else
                        {
                            cn.strides = calcContiguousStrides(pPartialShape);
                            cn.viewOffset = 0;
                        }
                        egraph.addENode(pContigEClass, cn);
                    }

                    slicedInputs.push_back(pContigEClass);

                    TensorNode dummyIn;
                    dummyIn.opType = OpType::INPUT;
                    dummyIn.setShape(pPartialShape);
                    dummyIn.dtype = pSliceNode.dtype;
                    dummyIn.backend = pSliceNode.backend;
                    dummyIn.strides = pContigNode.strides;
                    dummyIn.viewOffset = 0;
                    dummyInputNodes.push_back(dummyIn);
                }

                ENode opSlicedNode;
                opSlicedNode.opType = sourceNode.opType;
                opSlicedNode.opName = sourceNode.opName;
                opSlicedNode.children = slicedInputs;
                opSlicedNode.shape = partialShape;
                opSlicedNode.strides = calcContiguousStrides(partialShape);
                opSlicedNode.viewOffset = 0;
                opSlicedNode.dtype = sourceNode.dtype;
                opSlicedNode.backend = sourceNode.backend;

                TensorNode dummyOut;
                dummyOut.opType = sourceNode.opType;
                dummyOut.opName = sourceNode.opName;
                dummyOut.setShape(partialShape);
                dummyOut.dtype = sourceNode.dtype;
                dummyOut.backend = sourceNode.backend;
                dummyOut.strides = opSlicedNode.strides;
                dummyOut.viewOffset = 0;

                auto opRefs = KernelRegistry::get().findMatchingKernels(sourceNode.opType, sourceNode.opName, sourceNode.backend, dummyInputNodes, dummyOut, true);
                if (opRefs.size() == 0)
                {
                    Error::throw_err("[Planner.injectPartialPath] couldn't find any slice kernels");
                }
                slicedEClass = egraph.addEClass(partialShape, calcContiguousStrides(partialShape), 0, sourceNode.dtype, sourceNode.backend);
                for (uint64_t uid : opRefs)
                {
                    ENode sn = opSlicedNode;
                    sn.kernelUid = uid;
                    egraph.addENode(slicedEClass, sn);
                }
            }

            uint32_t contigEClass = egraph.addEClass(partialShape, calcContiguousStrides(partialShape), 0, sourceNode.dtype, targetBackend);
            ENode contigNode;
            contigNode.opType = OpType::CONTIGUOUS;
            contigNode.children = {slicedEClass};
            contigNode.shape = partialShape;
            contigNode.strides = calcContiguousStrides(partialShape);
            contigNode.dtype = sourceNode.dtype;
            contigNode.backend = targetBackend;

            TensorNode cOut;
            cOut.setShape(partialShape);
            cOut.dtype = sourceNode.dtype;
            cOut.backend = targetBackend;
            cOut.strides = contigNode.strides;
            TensorNode cIn;
            cIn.setShape(partialShape);
            cIn.dtype = sourceNode.dtype;
            cIn.backend = sourceNode.backend;
            cIn.strides = calcContiguousStrides(partialShape);

            auto contigRefs = KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", contigNode.backend, {cIn}, cOut, true);
            for (uint64_t uid : contigRefs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                ENode cn = contigNode;
                cn.kernelUid = uid;
                if (kernel.isView)
                {
                    cn.strides = cIn.strides;
                    cn.viewOffset = 0;
                }
                else
                {
                    cn.strides = calcContiguousStrides(partialShape);
                    cn.viewOffset = 0;
                }
                egraph.addENode(contigEClass, cn);
            }

            uint32_t scatterEClass = egraph.addEClass(lClass.shape, lClass.strides, lClass.viewOffset, lClass.dtype, targetBackend);
            ENode scatterNode;
            scatterNode.opType = OpType::SCATTER;
            scatterNode.children = {current_E, contigEClass, startsId, endsId, stepsId};
            scatterNode.shape = lClass.shape;
            scatterNode.strides = lClass.strides;
            scatterNode.viewOffset = lClass.viewOffset;
            scatterNode.dtype = lClass.dtype;
            scatterNode.backend = targetBackend;

            TensorNode sOut;
            sOut.setShape(scatterNode.shape);
            sOut.dtype = scatterNode.dtype;
            sOut.backend = scatterNode.backend;
            std::vector<TensorNode> sIns(5);
            sIns[0].setShape(egraph.getEClass(current_E).shape);
            sIns[0].dtype = scatterNode.dtype;
            sIns[0].backend = scatterNode.backend;
            sIns[1].setShape(partialShape);
            sIns[1].dtype = scatterNode.dtype;
            sIns[1].backend = scatterNode.backend;
            sIns[2].setShape({(uint32_t)starts.size()});
            sIns[2].dtype = DType::INT32;
            sIns[2].backend = Backend::CPU;
            sIns[3].setShape({(uint32_t)ends.size()});
            sIns[3].dtype = DType::INT32;
            sIns[3].backend = Backend::CPU;
            sIns[4].setShape({(uint32_t)steps.size()});
            sIns[4].dtype = DType::INT32;
            sIns[4].backend = Backend::CPU;

            auto scatterRefs = KernelRegistry::get().findMatchingKernels(OpType::SCATTER, "", scatterNode.backend, sIns, sOut, true);
            for (uint64_t uid : scatterRefs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                ENode sn = scatterNode;
                sn.kernelUid = uid;
                if (kernel.isView || kernel.inplace)
                {
                    sn.strides = lClass.strides;
                    sn.viewOffset = lClass.viewOffset;
                }
                else
                {
                    sn.strides = calcContiguousStrides(scatterNode.shape);
                    sn.viewOffset = 0;
                }
                egraph.addENode(scatterEClass, sn);
            }

            current_E = scatterEClass;
        }

        egraph.merge(E_L, current_E);
        eclassToLogical[egraph.find(E_L)] = logicalId;
        return true;
    }

    bool injectInputPartialPaths(
        EGraph &egraph,
        const Graph &graph,
        const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
        std::unordered_map<uint32_t, uint32_t> &eclassToLogical)
    {
        bool injected = false;
        for (const auto &kv : dirtyOutputRegions)
        {
            uint32_t nodeId = kv.first;
            if (!graph.hasNode(nodeId))
                continue;

            const TensorNode &node = graph.getNode(nodeId);
            if (node.opType == OpType::INPUT && graph.constantStaging.count(nodeId) == 0)
            {
                if (!kv.second.empty())
                {
                    injected = injected || injectPartialPath(egraph, graph, nodeId, kv.second, cachedNodes, nodeToEClass, eclassToLogical);
                }
            }
        }
        if (injected)
        {
            egraph.rebuild();
        }
        return injected;
    }

    bool injectOutputPartialPaths(
        EGraph &egraph,
        const Graph &graph,
        uint32_t rootId,
        const std::vector<Region> &outputNeeded,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
        std::unordered_map<uint32_t, uint32_t> &eclassToLogical)
    {
        bool injected = false;
        if (!outputNeeded.empty())
        {
            injected = injectPartialPath(egraph, graph, rootId, outputNeeded, cachedNodes, nodeToEClass, eclassToLogical);
        }
        if (injected)
        {
            egraph.rebuild();
        }
        return injected;
    }

    Planner(CostModel &costModel, std::unordered_map<uint32_t, uint64_t> &mem_caps)
        : costModel(costModel), mem_caps(mem_caps) {}

    CompiledGraph plan(
        LogicalId rootId,
        const Graph &graph,
        const Bucket &bucket,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        bool doSaturate = true,
        bool strictCache = false,
        Repo *repo = nullptr)
    {
        Graph tempGraph = graph;
        initBaseEGraph(rootId, tempGraph, doSaturate, repo);

        EGraph egraph = baseState.egraph;
        auto eclassToLogical = baseState.eclassToLogical;

        std::unordered_map<uint32_t, uint32_t> canonToLogical;
        canonToLogical.reserve(eclassToLogical.size());
        for (const auto &kv : eclassToLogical)
        {
            canonToLogical[egraph.find(kv.first)] = kv.second;
        }
        eclassToLogical = canonToLogical;

        std::unordered_map<uint32_t, bool> logicalDirty;
        std::vector<uint32_t> topo = topologicalSort({rootId}, graph);
        for (uint32_t nodeId : topo)
        {
            if (bucket.inputDirtyRegions.count(nodeId) && !bucket.inputDirtyRegions.at(nodeId).empty())
            {
                logicalDirty[nodeId] = true;
            }
            else if (graph.getNode(nodeId).opType == OpType::INPUT)
            {
                logicalDirty[nodeId] = false;
            }
            else
            {
                bool isDirty = false;
                for (uint32_t pid : graph.getNode(nodeId).parentIds)
                {
                    if (logicalDirty[pid])
                    {
                        isDirty = true;
                        break;
                    }
                }
                logicalDirty[nodeId] = isDirty;
            }
        }

        for (const auto &cls : egraph.getClasses())
        {
            uint32_t canonId = egraph.find(cls.id);
            if (canonId != cls.id)
                continue;
            if (strictCache)
            {
                if (eclassToLogical.count(canonId) == 0)
                    continue;
                if (cachedNodes.count(eclassToLogical.at(canonId)) == 0)
                    continue;
            }
            for (int i = 0; i < cls.enodes.size(); i++)
            {
                if (egraph.getENodes()[cls.enodes[i]].opType == OpType::CACHE)
                {
                    continue;
                }
            }

            uint32_t logicalId = UINT32_MAX;
            auto it = eclassToLogical.find(canonId);
            if (it != eclassToLogical.end())
            {
                logicalId = it->second;
            }

            if (logicalId != UINT32_MAX && !logicalDirty[logicalId])
            {
                ENode cacheNode;
                cacheNode.kernelUid = 0;
                cacheNode.opType = OpType::CACHE;
                cacheNode.shape = cls.shape;
                cacheNode.strides = cls.strides;
                cacheNode.viewOffset = cls.viewOffset;
                cacheNode.dtype = cls.dtype;
                cacheNode.backend = cls.backend;
                cacheNode.leafId = logicalId | 0x40000000;
                egraph.addENode(canonId, cacheNode);
            }
        }

        std::unordered_set<uint32_t> protectedEClasses;
        for (const auto &kv : cachedNodes)
        {
            uint32_t logicalId = kv.first;
            protectedEClasses.insert(egraph.find(baseState.nodeToEClass.at(logicalId)));
        }

        bool dirtyInjected = injectInputPartialPaths(egraph, graph, bucket.inputDirtyRegions, cachedNodes, baseState.nodeToEClass, eclassToLogical);

        bool neededInjected = injectOutputPartialPaths(egraph, graph, rootId, bucket.outputNeededRegion, cachedNodes, baseState.nodeToEClass, eclassToLogical);

        if (doSaturate)
        {
            saturate(egraph, protectedEClasses, eclassToLogical, true, false, repo);
        }

        bool injected = dirtyInjected || neededInjected;
        std::cout << "Injected: " << injected << std::endl;

#ifdef DEBUG
        auto rootIt = baseState.nodeToEClass.find(rootId);
        if (rootIt == baseState.nodeToEClass.end())
        {
            Error::throw_err("[Planner.plan] Root node missing from baseState.nodeToEClass.");
        }
        uint32_t rootEClassId = egraph.find(rootIt->second);
        dumpEGraphBinary(egraph, rootEClassId);
#endif

        std::unordered_map<uint32_t, uint32_t> updatedEClassToLogical;
        for (const auto &kv : eclassToLogical)
        {
            updatedEClassToLogical[egraph.find(kv.first)] = kv.second;
        }
        eclassToLogical = std::move(updatedEClassToLogical);

        std::cout << "[Planner.plan] initializing immutable classes" << std::endl;
        std::unordered_set<uint32_t> immutable_eclasses;
        for (const auto &kv : eclassToLogical)
        {
            uint32_t logicalId = kv.second;
            uint32_t ecl = egraph.find(kv.first);
            const TensorNode &node = graph.getNode(logicalId);
            if (node.storageType != StorageType::TRANSIENT || cachedNodes.count(logicalId))
            {
                immutable_eclasses.insert(ecl);
            }
        }

        bool changed = true;
        while (changed)
        {
            changed = false;
            for (const auto &cls : egraph.getClasses())
            {
                uint32_t e_class_id = egraph.find(cls.id);
                if (immutable_eclasses.count(e_class_id))
                    continue;

                for (uint32_t enodeId : cls.enodes)
                {
                    const ENode &enode = egraph.getENodes()[enodeId];
                    if (enode.kernelUid != 0)
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(enode.kernelUid);
                        if (kernel.isView && !enode.children.empty())
                        {
                            uint32_t parentEclass = egraph.find(enode.children[0]);
                            if (immutable_eclasses.count(parentEclass))
                            {
                                immutable_eclasses.insert(e_class_id);
                                changed = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        auto extraction = extractBest(rootId, graph, egraph, baseState.nodeToEClass, maxMemoryByBackend, cachedNodes, eclassToLogical, immutable_eclasses, true, strictCache);
        return buildCompiledGraph(
            rootId, graph, egraph, baseState.nodeToEClass, extraction, cachedNodes, eclassToLogical);
    }
};