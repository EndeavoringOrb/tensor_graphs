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

struct PreparedEGraph
{
    EGraph egraph;
    std::unordered_map<EClassId, LogicalId> eclassToLogical;
    std::vector<ENodeInfo> enodeInfos;
    std::unordered_map<LogicalId, MemSpace> cachedNodes;
    std::unordered_map<LogicalId, ParallelBuffer> preallocatedBuffers;
    bool is_strict_cache = false;
};

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

    PreparedEGraph prepared;

    CostModel costModel;

    SaturatedEGraphContext() : costModel(false)
    {
    }

    std::unordered_map<LogicalId, ParallelBuffer> preallocateLogicalBuffers(
        const std::unordered_map<LogicalId, MemSpace> &cachedNodes) const
    {
        std::unordered_map<LogicalId, ParallelBuffer> out;

        struct PreAllocEntry
        {
            LogicalId logicalId;
            MemSpace memSpace;
            std::vector<uint32_t> shape;
            DType dtype;
        };
        std::vector<PreAllocEntry> entries;

        MemSpace storage = MemSpace{0, HandleType::STORAGE};
        MemSpace ram = MemSpace{1, HandleType::CPP};

        for (const auto &pair : graph.nodes)
        {
            const TensorNode &node = pair.second;
            if (node.opType != OpType::INPUT)
                continue;

            auto idtIt = graph.input_data_types.find(node.id);
            if (idtIt != graph.input_data_types.end() && idtIt->second == InputDataType::STORAGE)
                continue;

            entries.push_back({node.id, ram, node.getShape(), node.dtype});
        }

        for (const auto &kv : cachedNodes)
        {
            LogicalId logicalId = kv.first;
            MemSpace ms = kv.second;
            if (!graph.hasNode(logicalId))
                continue;
            const TensorNode &node = graph.getNode(logicalId);
            bool alreadyAdded = false;
            for (const auto &e : entries)
            {
                if (e.logicalId == logicalId)
                {
                    alreadyAdded = true;
                    break;
                }
            }
            if (alreadyAdded)
                continue;
            entries.push_back({logicalId, ms, node.getShape(), node.dtype});
        }

        std::sort(entries.begin(), entries.end(),
                  [](const PreAllocEntry &a, const PreAllocEntry &b) { return a.logicalId < b.logicalId; });

        std::unordered_map<MemSpace, uint64_t> cursor;
        BufferId nextId{0};
        for (const auto &e : entries)
        {
            if (e.memSpace == storage)
                continue;

            uint64_t size_bytes = getSizeBytes(e.shape, e.dtype);
            if (size_bytes == 0)
                continue;
            size_bytes = (size_bytes + 4095) & ~4095ULL;

            uint64_t offset = cursor[e.memSpace];
            cursor[e.memSpace] = offset + size_bytes;

            ParallelBuffer buf;
            buf.id = nextId++;
            buf.mem_space = e.memSpace;
            buf.size = size_bytes;
            buf.start = 0;
            buf.end = std::numeric_limits<uint32_t>::max();
            buf.offset = static_cast<int64_t>(offset);
            out[e.logicalId] = std::move(buf);
        }

        return out;
    }

    void setup_for_bucket(int bucket_idx, const std::unordered_map<LogicalId, MemSpace> &cachedNodes, bool strictCache)
    {
        auto preallocatedBuffers = preallocateLogicalBuffers(cachedNodes);
        Planner planner(costModel, mem->getMemCaps());
        planner.baseState.egraph = baseEGraph;
        planner.baseState.nodeToEClass = baseNodeToEClass;
        planner.baseState.eclassToLogical = baseEclassToLogical;
        planner.baseStateInitialized = true;

        prepared.cachedNodes = cachedNodes;
        prepared.preallocatedBuffers = preallocatedBuffers;
        prepared.is_strict_cache = strictCache;

        const Bucket &bucket = buckets[bucket_idx];

        EGraph egraph = baseEGraph;
        auto eclassToLogical = baseEclassToLogical;

        std::unordered_map<LogicalId, bool> logicalDirty;
        std::vector<LogicalId> topo = topologicalSort({rootId}, graph);
        for (LogicalId nodeId : topo)
        {
            if (bucket.inputDirtyRegions.count(nodeId) && !bucket.inputDirtyRegions.at(nodeId).empty())
            {
                logicalDirty[nodeId] = true;
            }
            else
            {
                bool isDirty = false;
                for (LogicalId pid : graph.getNode(nodeId).child_ids)
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

        Engine cpu = Engine{0, EngineType::CPU};
        for (const auto &cls : egraph.getClasses())
        {
            EClassId canonId = egraph.find(cls.id);
            if (canonId != cls.id)
                continue;
            if (strictCache)
            {
                if (eclassToLogical.count(canonId) == 0)
                    continue;
                if (cachedNodes.count(eclassToLogical.at(canonId)) == 0)
                    continue;
            }
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
                LogicalId logicalId;
                auto it = eclassToLogical.find(canonId);
                if (it != eclassToLogical.end())
                {
                    logicalId = it->second;
                }

                if (logicalId != LogicalId{UINT32_MAX} && !logicalDirty[logicalId])
                {
                    ENode cacheNode = ENode(KernelId{0}, OpType::CACHE, "", {}, cls.shape, cls.strides, cls.dtype,
                                            cls.mem_space, {cpu}, toString(logicalId));
                    egraph.addENode(canonId, cacheNode);
                }
            }
        }

        std::unordered_set<EClassId> protectedEClasses;
        for (const auto &kv : cachedNodes)
        {
            LogicalId logicalId = kv.first;
            protectedEClasses.insert(egraph.findConst(baseNodeToEClass.at(logicalId)));
        }

        planner.injectInputPartialPaths(egraph, graph, bucket.inputDirtyRegions, cachedNodes, baseNodeToEClass,
                                        eclassToLogical);
        planner.injectOutputPartialPaths(egraph, graph, rootId, bucket.outputNeededRegion, cachedNodes,
                                         baseNodeToEClass, eclassToLogical);

        planner.saturate(egraph, protectedEClasses, eclassToLogical, true, false, nullptr);

        std::unordered_map<EClassId, LogicalId> updatedEClassToLogical;
        for (const auto &kv : eclassToLogical)
        {
            updatedEClassToLogical[egraph.findConst(kv.first)] = kv.second;
        }

        prepared.egraph = egraph;
        prepared.eclassToLogical = std::move(updatedEClassToLogical);

        prepared.enodeInfos =
            planner.computeENodeInfos(prepared.egraph, prepared.eclassToLogical, cachedNodes, strictCache);
        planner.pruneEGraph(prepared.egraph, prepared.enodeInfos);
    }
};

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

inline float extract_best_from_egraph(std::shared_ptr<SaturatedEGraphContext> ctx,
                                      std::shared_ptr<SearchDelegate> delegate, bool log_cost_calls)
{
    ctx->costModel.setLogging(log_cost_calls);

    Planner planner(ctx->costModel, ctx->mem->getMemCaps());

    auto extraction =
        planner.extractBest(ctx->rootId, ctx->graph, ctx->prepared.egraph, ctx->baseNodeToEClass,
                            ctx->prepared.cachedNodes, ctx->prepared.eclassToLogical, ctx->prepared.preallocatedBuffers,
                            /*stopOnFirstValid=*/true, ctx->prepared.is_strict_cache, /*minCompileSeconds=*/0.0f,
                            delegate, ctx->prepared.enodeInfos);

    return extraction.cost;
}

inline std::unordered_map<LogicalId, MemSpace> get_cached_nodes(std::shared_ptr<SaturatedEGraphContext> ctx,
                                                                std::shared_ptr<SearchDelegate> delegate,
                                                                bool log_cost_calls)
{
    ctx->costModel.setLogging(log_cost_calls);
    Planner planner(ctx->costModel, ctx->mem->getMemCaps());

    auto extraction =
        planner.extractBest(ctx->rootId, ctx->graph, ctx->prepared.egraph, ctx->baseNodeToEClass,
                            ctx->prepared.cachedNodes, ctx->prepared.eclassToLogical, ctx->prepared.preallocatedBuffers,
                            /*stopOnFirstValid=*/true, ctx->prepared.is_strict_cache, /*minCompileSeconds=*/0.0f,
                            delegate, ctx->prepared.enodeInfos);

    CompiledGraph plan =
        planner.buildCompiledGraph(ctx->rootId, ctx->graph, ctx->prepared.egraph, ctx->baseNodeToEClass, extraction,
                                   ctx->prepared.cachedNodes, ctx->prepared.eclassToLogical);

    std::unordered_map<LogicalId, MemSpace> protectedCachedNodes;
    for (const auto &inst : plan.instructions)
    {
        if (plan.has_logical_id(inst.eclass_id))
        {
            LogicalId logical_id = plan.get_logical_id(inst.eclass_id);
            OpType op_type = ctx->graph.getNode(logical_id).opType;
            if (op_type == OpType::CACHE)
            {
                protectedCachedNodes[logical_id] = inst.outBuffer.mem_space;
            }
        }
    }
    return protectedCachedNodes;
}