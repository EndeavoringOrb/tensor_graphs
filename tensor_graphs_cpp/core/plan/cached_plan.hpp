#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/plan/planner.hpp"
#include "core/plan/search_delegate.hpp"
#include "core/types.hpp"
#include "models/run_models.hpp"

struct SaturatedEGraphContext
{
    Graph graph;
    std::unique_ptr<MemoryManager> mem;
    LogicalId rootId;
    LogicalId inputIdsId;
    uint32_t vocab_size = 0;
    uint32_t max_seq_len = 0;
    std::vector<Bucket> buckets;

    // Base state
    EGraph baseEGraph;
    std::unordered_map<LogicalId, EClassId> baseNodeToEClass;
    std::unordered_map<EClassId, LogicalId> baseEclassToLogical;

    CostModel costModel;

    SaturatedEGraphContext() : costModel(false)
    {
    }

    std::unordered_map<LogicalId, ParallelBuffer> preallocateLogicalBuffers(
        const std::unordered_map<LogicalId, MemSpace> &cachedNodes) const
    {
        std::unordered_map<LogicalId, ParallelBuffer> out;
        Planner planner(const_cast<CostModel &>(costModel), mem->getMemCaps());
        planner.preallocateLogicalBuffers(graph, cachedNodes, out);
        return out;
    }
};

inline std::shared_ptr<SaturatedEGraphContext> build_and_saturate_egraph_from_graph(const Graph &input_graph,
                                                                                    LogicalId rootId,
                                                                                    const std::vector<Bucket> &buckets,
                                                                                    bool log_cost_calls = false)
{
    auto ctx = std::make_shared<SaturatedEGraphContext>();
    ctx->costModel.setLogging(log_cost_calls);

    std::unordered_map<MemSpace, uint64_t> bufferSizes = {{MemSpace{1, HandleType::CPP}, 16ULL * 1024 * 1024 * 1024}};
#ifdef TG_USE_CUDA
    bufferSizes[MemSpace{2, HandleType::CUDA}] = 90ULL * 1024 * 1024 * 1024;
#endif
    if (HardwareCaps::get().has_opencl)
    {
        bufferSizes[MemSpace{1, HandleType::OPENCL}] = 1ULL * 1024 * 1024 * 1024;
    }

    ctx->mem = std::make_unique<MemoryManager>(bufferSizes);
    ctx->graph = input_graph;
    ctx->rootId = rootId;
    ctx->buckets = buckets;

    if (ctx->buckets.empty())
    {
        Bucket fullB;
        for (const auto &pair : ctx->graph.nodes)
        {
            if (pair.second.opType == OpType::INPUT)
            {
                fullB.inputDirtyRegions[pair.first] = {makeFull(pair.second.getShape())};
            }
        }
        fullB.outputNeededRegion = {makeFull(ctx->graph.getNode(rootId).getShape())};
        ctx->buckets.push_back(fullB);
    }

    std::vector<LogicalId> topo = topologicalSort({rootId}, ctx->graph);

    Planner planner(ctx->costModel, ctx->mem->getMemCaps());
    planner.initBaseEGraph(ctx->rootId, ctx->graph, topo, nullptr);

    ctx->baseEGraph = planner.baseState.egraph;
    ctx->baseNodeToEClass = planner.baseState.nodeToEClass;
    ctx->baseEclassToLogical = planner.baseState.eclassToLogical;

    ctx->costModel.load("benchmarks/records.bin");

    return ctx;
}

inline std::shared_ptr<SaturatedEGraphContext> build_and_saturate_egraph(const std::string &model_name,
                                                                         const std::string &model_path,
                                                                         bool log_cost_calls = false,
                                                                         bool compile_decode_buckets = true)
{
    auto ctx = std::make_shared<SaturatedEGraphContext>();
    ctx->costModel.setLogging(log_cost_calls);

    std::unordered_map<MemSpace, uint64_t> bufferSizes = {{MemSpace{1, HandleType::CPP}, 16ULL * 1024 * 1024 * 1024}};
#ifdef TG_USE_CUDA
    bufferSizes[MemSpace{2, HandleType::CUDA}] = 90ULL * 1024 * 1024 * 1024;
#endif
    if (HardwareCaps::get().has_opencl)
    {
        bufferSizes[MemSpace{1, HandleType::OPENCL}] = 1ULL * 1024 * 1024 * 1024;
    }

    ctx->mem = std::make_unique<MemoryManager>(bufferSizes);

    uint32_t max_seq_len = 8;
    ModelGraphRoots roots;
    if (model_name == "gemma-3-270m")
    {
        roots = build_gemma_graph(ctx->graph, *ctx->mem, model_path, max_seq_len);
        ctx->vocab_size = Gemma3ModelConfig().vocab_size;
    }
    else if (model_name == "qwen-3.6-35b-a3b")
    {
        roots = build_qwen_graph(ctx->graph, *ctx->mem, model_path, max_seq_len);
        ctx->vocab_size = Qwen3_6_35B_A3B_Config().vocab_size;
    }
    else
    {
        Error::throw_err("Unsupported model for E-Graph caching: " + model_name);
    }

    ctx->rootId = roots.roots[0];
    ctx->inputIdsId = roots.inputs[0];
    ctx->max_seq_len = max_seq_len;

    if (compile_decode_buckets)
    {
        for (uint32_t i = 0; i < max_seq_len; ++i)
        {
            Bucket b;
            Region inR;
            inR.region = {{0, 1}, {i, i + 1}};
            b.inputDirtyRegions[ctx->inputIdsId] = {inR};

            Region outR;
            outR.region = {{0, 1}, {i, i + 1}, {0, ctx->vocab_size}};
            b.outputNeededRegion = {outR};
            ctx->buckets.push_back(b);
        }
    }
    Bucket fullB;
    Region fullIn;
    fullIn.region = {{0, 1}, {0, max_seq_len}};
    fullB.inputDirtyRegions[ctx->inputIdsId] = {fullIn};
    Region fullOut;
    fullOut.region = {{0, 1}, {0, max_seq_len}, {0, ctx->vocab_size}};
    fullB.outputNeededRegion = {fullOut};
    ctx->buckets.push_back(fullB);

    std::vector<LogicalId> topo = topologicalSort(roots.roots, ctx->graph);

    Planner planner(ctx->costModel, ctx->mem->getMemCaps());
    planner.initBaseEGraph(ctx->rootId, ctx->graph, topo, nullptr);

    ctx->baseEGraph = planner.baseState.egraph;
    ctx->baseNodeToEClass = planner.baseState.nodeToEClass;
    ctx->baseEclassToLogical = planner.baseState.eclassToLogical;

    ctx->costModel.load("benchmarks/records.bin");

    return ctx;
}

inline std::vector<float> run_hierarchical_simulations(std::shared_ptr<SaturatedEGraphContext> ctx, int bucket_idx,
                                                       std::shared_ptr<SearchDelegate> delegate,
                                                       const std::vector<uint32_t> &level_simulations,
                                                       bool log_cost_calls = false)
{
    ctx->costModel.setLogging(log_cost_calls);

    uint32_t num_cache = 1;
    uint32_t num_extract = 1;
    uint32_t num_dispatch = 1;
    uint32_t num_bufferize = 1;
    uint32_t num_malloc = 1;

    if (level_simulations.size() == 4)
    {
        num_cache = 1;
        num_extract = level_simulations[0];
        num_dispatch = level_simulations[1];
        num_bufferize = level_simulations[2];
        num_malloc = level_simulations[3];
    }
    else if (level_simulations.size() >= 5)
    {
        num_cache = level_simulations[0];
        num_extract = level_simulations[1];
        num_dispatch = level_simulations[2];
        num_bufferize = level_simulations[3];
        num_malloc = level_simulations[4];
    }
    else if (level_simulations.size() == 1)
    {
        num_extract = level_simulations[0];
    }

    if (bucket_idx < 0 || bucket_idx >= static_cast<int>(ctx->buckets.size()))
    {
        bucket_idx = static_cast<int>(ctx->buckets.size()) - 1;
    }
    const Bucket &bucket = ctx->buckets[bucket_idx];

    std::vector<LogicalId> topo = topologicalSort({ctx->rootId}, ctx->graph);

    std::unordered_map<LogicalId, bool> logicalDirty;
    for (LogicalId nodeId : topo)
    {
        if (bucket.inputDirtyRegions.count(nodeId) && !bucket.inputDirtyRegions.at(nodeId).empty())
        {
            logicalDirty[nodeId] = true;
        }
        else
        {
            bool isDirty = false;
            for (LogicalId pid : ctx->graph.getNode(nodeId).child_ids)
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

    std::vector<LogicalId> candidates;
    for (LogicalId nodeId : topo)
    {
        if (!logicalDirty[nodeId] && ctx->graph.getNode(nodeId).getSizeBytes() > 0)
        {
            candidates.push_back(nodeId);
        }
    }

    std::vector<MemSpace> avail_mem_spaces;
    for (const auto &kv : ctx->mem->getMemCaps())
    {
        if (kv.first.type != HandleType::STORAGE)
        {
            avail_mem_spaces.push_back(kv.first);
        }
    }
    std::sort(avail_mem_spaces.begin(), avail_mem_spaces.end(), [](const MemSpace &a, const MemSpace &b) {
        if (a.type != b.type)
            return a.type < b.type;
        return a.idx < b.idx;
    });

    CacheIterator cache_iter(ctx->graph, candidates, avail_mem_spaces, delegate);
    std::unordered_map<LogicalId, MemSpace> cachedNodes;

    std::vector<float> all_costs;
    uint32_t cache_eval_count = 0;
    Planner planner(ctx->costModel, ctx->mem->getMemCaps());

    while (cache_iter.getNextCacheSelection(cachedNodes))
    {
        cache_eval_count++;

        auto preallocatedBuffers = ctx->preallocateLogicalBuffers(cachedNodes);

        EGraph egraph = ctx->baseEGraph;
        auto eclassToLogical = ctx->baseEclassToLogical;

        Engine cpu = Engine{0, EngineType::CPU};
        for (const auto &cls : egraph.getClasses())
        {
            EClassId canonId = egraph.find(cls.id);
            if (canonId != cls.id)
                continue;
            if (eclassToLogical.count(canonId) == 0)
                continue;
            LogicalId logicalId = eclassToLogical.at(canonId);
            if (cachedNodes.count(logicalId) == 0)
                continue;

            bool hasCache = false;
            for (int i = 0; i < cls.enodes.size(); i++)
            {
                if (egraph.getENode(cls.enodes[i]).getOpType() == OpType::CACHE)
                {
                    hasCache = true;
                    break;
                }
            }
            if (!hasCache)
            {
                ENode cacheNode = ENode(KernelId{0}, OpType::CACHE, "", {}, cls.shape, cls.strides, cls.dtype,
                                        cachedNodes.at(logicalId), {cpu}, toString(logicalId));
                egraph.addENode(canonId, cacheNode);
            }
        }

        std::unordered_set<EClassId> protectedEClasses;
        for (const auto &kv : cachedNodes)
        {
            LogicalId logicalId = kv.first;
            if (ctx->baseNodeToEClass.count(logicalId))
            {
                protectedEClasses.insert(egraph.findConst(ctx->baseNodeToEClass.at(logicalId)));
            }
        }

        planner.injectInputPartialPaths(egraph, ctx->graph, bucket.inputDirtyRegions, cachedNodes,
                                        ctx->baseNodeToEClass, eclassToLogical);
        planner.injectOutputPartialPaths(egraph, ctx->graph, ctx->rootId, bucket.outputNeededRegion, cachedNodes,
                                         ctx->baseNodeToEClass, eclassToLogical);

        planner.saturate(egraph, protectedEClasses, eclassToLogical, true, false, nullptr);

        std::unordered_map<EClassId, LogicalId> updatedEClassToLogical;
        for (const auto &kv : eclassToLogical)
        {
            updatedEClassToLogical[egraph.findConst(kv.first)] = kv.second;
        }
        eclassToLogical = std::move(updatedEClassToLogical);

        auto enodeInfos = planner.computeENodeInfos(egraph, eclassToLogical, cachedNodes, /*strictCache=*/true);
        planner.pruneEGraph(egraph, enodeInfos);

        EClassId rootEClassId = egraph.findConst(ctx->baseNodeToEClass.at(ctx->rootId));
        if (egraph.getEClass(rootEClassId).enodes.empty())
        {
            if (delegate)
                delegate->on_leaf_evaluated(TGConstants::INF);
            if (cache_eval_count >= num_cache)
                break;
            continue;
        }

        // Extractor (Level 1)
        Extractor extractor(egraph, rootEClassId, enodeInfos, delegate);
        extractor.registerValidator(std::make_unique<CycleValidator>(egraph));
        extractor.registerValidator(std::make_unique<MemValidator>(egraph, enodeInfos, ctx->mem->getMemCaps(),
                                                                   eclassToLogical, preallocatedBuffers, delegate));

        uint32_t extract_count = 0;
        while (extractor.getNextSelection())
        {
            extract_count++;
            const auto &selection_map = extractor.selection_map;

            // Dispatch (Level 2)
            DispatchIterator dispatch_iterator(egraph, selection_map, enodeInfos, delegate);
            uint32_t dispatch_count = 0;
            std::vector<EClassId> order;

            while (dispatch_iterator.getNextDispatchOrder(selection_map, order))
            {
                dispatch_count++;

                // Bufferize (Level 3)
                BufferizeIterator buf_iter(order, egraph, selection_map, enodeInfos, delegate);
                uint32_t buf_count = 0;
                std::vector<ParallelBuffer> unallocated_buffers;
                std::unordered_map<EClassId, BufferId> eclass_to_buf_local;

                while (buf_iter.getNextBufferization(unallocated_buffers, eclass_to_buf_local))
                {
                    buf_count++;

                    // Malloc (Level 4)
                    std::unordered_set<BufferId> preallocated_buf_ids;
                    std::unordered_map<BufferId, ParallelBuffer> preallocated_overrides;
                    std::unordered_map<MemSpace, uint64_t> reserved_per_ms;

                    for (EClassId eclass : order)
                    {
                        auto logicalIt = eclassToLogical.find(eclass);
                        if (logicalIt == eclassToLogical.end())
                            continue;
                        auto sel_it = selection_map.find(eclass);
                        if (sel_it == selection_map.end())
                            continue;
                        uint32_t sel = sel_it->second;
                        ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
                        const ENode &node = egraph.getENode(enode_id);
                        if (node.getOpType() != OpType::INPUT && node.getOpType() != OpType::CACHE)
                            continue;

                        auto preIt = preallocatedBuffers.find(logicalIt->second);
                        if (preIt == preallocatedBuffers.end())
                            continue;

                        BufferId buf_id = eclass_to_buf_local.at(eclass);
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

                    bool alloc_ok = true;
                    BufferId overflow;
                    for (auto &kv : buf_by_mem_space)
                    {
                        MemSpace ms = kv.first;
                        auto &bufs = kv.second;
                        uint64_t cap = ctx->mem->getMemCaps().count(ms) ? ctx->mem->getMemCaps().at(ms)
                                                                        : std::numeric_limits<uint64_t>::max();
                        uint64_t reserved = reserved_per_ms.count(ms) ? reserved_per_ms.at(ms) : 0;
                        uint64_t reduced_cap =
                            (cap == std::numeric_limits<uint64_t>::max()) ? cap : (cap > reserved ? cap - reserved : 0);

                        std::vector<ParallelBuffer> allocated;
                        if (!malloc_by_time_components(reduced_cap, bufs, allocated, overflow, delegate))
                        {
                            alloc_ok = false;
                            break;
                        }
                    }

                    float cost = TGConstants::INF;
                    if (alloc_ok)
                    {
                        cost = get_cost(order, egraph, selection_map, enodeInfos);
                        all_costs.push_back(cost);
                    }

                    if (delegate)
                    {
                        delegate->on_leaf_evaluated(cost);
                    }

                    if (buf_count >= num_bufferize)
                        break;
                }

                if (dispatch_count >= num_dispatch)
                    break;
            }

            extractor.ascend();
            if (extract_count >= num_extract)
                break;
        }

        if (cache_eval_count >= num_cache)
            break;
    }

    return all_costs;
}

inline float extract_best_from_egraph(std::shared_ptr<SaturatedEGraphContext> ctx,
                                      std::shared_ptr<SearchDelegate> delegate, bool log_cost_calls)
{
    auto costs = run_hierarchical_simulations(ctx, -1, delegate, {1, 1, 1, 1}, log_cost_calls);
    if (costs.empty())
        return TGConstants::INF;
    return *std::min_element(costs.begin(), costs.end());
}