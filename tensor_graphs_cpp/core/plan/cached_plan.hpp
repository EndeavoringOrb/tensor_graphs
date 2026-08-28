#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
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

struct SaturatedState
{
    EGraph egraph;
    std::unordered_map<EClassId, LogicalId> eclassToLogical;
    std::vector<ENodeInfo> enodeInfos;
    std::unordered_map<LogicalId, ParallelBuffer> preallocatedBuffers;
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
    Settings settings;

    // Base state
    EGraph baseEGraph;
    std::unordered_map<LogicalId, EClassId> baseNodeToEClass;
    std::unordered_map<EClassId, LogicalId> baseEclassToLogical;

    CostModel costModel;

    std::unordered_map<std::string, std::shared_ptr<SaturatedState>> saturationCache;

    SaturatedEGraphContext() : costModel(false), settings(Settings::get_default())
    {
    }

    std::unordered_map<LogicalId, ParallelBuffer> preallocateLogicalBuffers(
        const std::unordered_map<LogicalId, MemSpace> &cachedNodes) const
    {
        std::unordered_map<LogicalId, ParallelBuffer> out;
        Planner planner(const_cast<CostModel &>(costModel), settings);
        planner.preallocateLogicalBuffers(graph, cachedNodes, out);
        return out;
    }
};

inline std::shared_ptr<SaturatedEGraphContext> build_and_saturate_egraph_from_graph(const Graph &input_graph,
                                                                                    LogicalId rootId,
                                                                                    const std::vector<Bucket> &buckets,
                                                                                    bool log_cost_calls = false,
                                                                                    uint64_t mem_cap_override = 0)
{
    auto ctx = std::make_shared<SaturatedEGraphContext>();
    ctx->costModel.setLogging(log_cost_calls);

    if (mem_cap_override > 0)
    {
        ctx->settings.mem_caps[MemSpace{1, HandleType::CPP}] = mem_cap_override;
#ifdef TG_USE_CUDA
        for (uint32_t dev = 0; dev < System::get().getNumCudaDevices(); ++dev)
        {
            ctx->settings.mem_caps[MemSpace{dev, HandleType::CUDA}] = mem_cap_override;
        }
#endif
    }

    ctx->mem = std::make_unique<MemoryManager>(ctx->settings.mem_caps);
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

    Planner planner(ctx->costModel, ctx->settings);
    planner.initBaseEGraph(ctx->rootId, ctx->graph, topo, nullptr);

    ctx->baseEGraph = planner.baseState.egraph;
    ctx->baseNodeToEClass = planner.baseState.nodeToEClass;
    ctx->baseEclassToLogical = planner.baseState.eclassToLogical;

    return ctx;
}

inline std::shared_ptr<SaturatedEGraphContext> build_and_saturate_egraph(const std::string &model_name,
                                                                         const std::string &model_path,
                                                                         bool log_cost_calls = false,
                                                                         bool compile_decode_buckets = true,
                                                                         uint32_t max_seq_len = 8)
{
    auto ctx = std::make_shared<SaturatedEGraphContext>();
    ctx->costModel.setLogging(log_cost_calls);
    ctx->mem = std::make_unique<MemoryManager>(ctx->settings.mem_caps);

    ModelGraphRoots roots;
    if (model_name == "gemma-3-270m" || model_name == "gemma")
    {
        roots = build_gemma_graph(ctx->graph, *ctx->mem, model_path, max_seq_len);
        ctx->vocab_size = Gemma3ModelConfig().vocab_size;
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
    }
    else if (model_name == "qwen-3.6-35b-a3b" || model_name == "qwen")
    {
        roots = build_qwen_graph(ctx->graph, *ctx->mem, model_path, max_seq_len);
        ctx->vocab_size = Qwen3_6_35B_A3B_Config().vocab_size;
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
    }
    else if (model_name == "krea" || model_name == "krea-2-turbo" || model_name == "krea2-turbo" ||
             model_name == "krea2")
    {
        std::string actual_dit = model_path;
        if (std::filesystem::is_directory(model_path))
        {
            if (std::filesystem::exists(model_path + "/krea.safetensors"))
                actual_dit = model_path + "/krea.safetensors";
            else if (std::filesystem::exists(model_path + "/turbo.safetensors"))
                actual_dit = model_path + "/turbo.safetensors";
            else if (std::filesystem::exists(model_path + "/krea2_turbo_fp8_scaled.safetensors"))
                actual_dit = model_path + "/krea2_turbo_fp8_scaled.safetensors";
            else if (std::filesystem::exists(model_path + "/transformer"))
                actual_dit = model_path + "/transformer";
        }
        std::string actual_te = model_path;
        if (std::filesystem::is_directory(model_path))
        {
            if (std::filesystem::exists(model_path + "/qwen3vl_4b_bf16.safetensors"))
                actual_te = model_path + "/qwen3vl_4b_bf16.safetensors";
            else if (std::filesystem::exists(model_path + "/text_encoder"))
                actual_te = model_path + "/text_encoder";
        }
        std::string actual_vae = model_path;
        if (std::filesystem::is_directory(model_path))
        {
            if (std::filesystem::exists(model_path + "/qwen_image_vae.safetensors"))
                actual_vae = model_path + "/qwen_image_vae.safetensors";
            else if (std::filesystem::exists(model_path + "/vae"))
                actual_vae = model_path + "/vae";
        }
        roots = build_krea2_pipeline_graph(ctx->graph, *ctx->mem, actual_dit, actual_te, actual_vae, 512, 512, 128, 8,
                                           1.15f);
        ctx->rootId = roots.roots[0];
        ctx->inputIdsId = roots.inputs[0];
        ctx->max_seq_len = 128;

        LogicalId latentId = roots.inputs[0];
        LogicalId timestepId = roots.inputs[1];
        LogicalId textId = roots.inputs[2];
        LogicalId velocityOut = roots.roots[0];

        if (compile_decode_buckets)
        {
            Bucket stepB;
            stepB.inputDirtyRegions[latentId] = {makeFull(ctx->graph.getNode(latentId).getShape())};
            stepB.inputDirtyRegions[timestepId] = {makeFull(ctx->graph.getNode(timestepId).getShape())};
            stepB.outputNeededRegion = {makeFull(ctx->graph.getNode(velocityOut).getShape())};
            ctx->buckets.push_back(stepB);
        }

        Bucket fullB;
        fullB.inputDirtyRegions[latentId] = {makeFull(ctx->graph.getNode(latentId).getShape())};
        fullB.inputDirtyRegions[timestepId] = {makeFull(ctx->graph.getNode(timestepId).getShape())};
        fullB.inputDirtyRegions[textId] = {makeFull(ctx->graph.getNode(textId).getShape())};
        fullB.outputNeededRegion = {makeFull(ctx->graph.getNode(velocityOut).getShape())};
        ctx->buckets.push_back(fullB);
    }
    else if (model_name == "vae" || model_name == "krea-2-turbo-vae" || model_name == "krea-vae" ||
             model_name == "krea2-vae" || model_name == "qwen-image-vae")
    {
        std::string actual_path = model_path;
        if (std::filesystem::is_directory(model_path))
        {
            if (std::filesystem::exists(model_path + "/vae"))
                actual_path = model_path + "/vae";
            else if (std::filesystem::exists(model_path + "/qwen_image_vae.safetensors"))
                actual_path = model_path + "/qwen_image_vae.safetensors";
        }
        roots = build_krea2_vae_graph(ctx->graph, *ctx->mem, actual_path, 512, 512);
        ctx->rootId = roots.roots[0];
        ctx->inputIdsId = roots.inputs[0];
        ctx->max_seq_len = 0;

        LogicalId latentId = roots.inputs[0];
        LogicalId imageOut = roots.roots[0];

        Bucket fullB;
        fullB.inputDirtyRegions[latentId] = {makeFull(ctx->graph.getNode(latentId).getShape())};
        fullB.outputNeededRegion = {makeFull(ctx->graph.getNode(imageOut).getShape())};
        ctx->buckets.push_back(fullB);
    }
    else if (model_name == "qwen3-vl" || model_name == "qwen3-vl-bf16" || model_name == "qwen3vl" ||
             model_name == "qwen3vl-bf16" || model_name == "qwen3vl_4b_bf16" || model_name == "qwen3-vl-4b-bf16")
    {
        std::string actual_path = model_path;
        if (std::filesystem::is_directory(model_path))
        {
            if (std::filesystem::exists(model_path + "/text_encoder"))
                actual_path = model_path + "/text_encoder";
            else if (std::filesystem::exists(model_path + "/text_encoders"))
                actual_path = model_path + "/text_encoders";
            else if (std::filesystem::exists(model_path + "/qwen3vl_4b_bf16.safetensors"))
                actual_path = model_path + "/qwen3vl_4b_bf16.safetensors";
            else if (std::filesystem::exists(model_path + "/qwen3vl_4b.safetensors"))
                actual_path = model_path + "/qwen3vl_4b.safetensors";
            else if (std::filesystem::exists(model_path + "/qwen3vl_4b_fp8_scaled.safetensors"))
                actual_path = model_path + "/qwen3vl_4b_fp8_scaled.safetensors";
        }
        roots = build_qwen3_vl_graph(ctx->graph, *ctx->mem, actual_path, 128);
        ctx->rootId = roots.roots[0];
        ctx->inputIdsId = roots.inputs[0];
        ctx->max_seq_len = 128;

        LogicalId inputIds = roots.inputs[0];
        LogicalId textEmbOut = roots.roots[0];

        Bucket fullB;
        fullB.inputDirtyRegions[inputIds] = {makeFull(ctx->graph.getNode(inputIds).getShape())};
        fullB.outputNeededRegion = {makeFull(ctx->graph.getNode(textEmbOut).getShape())};
        ctx->buckets.push_back(fullB);
    }
    else
    {
        Error::throw_err("Unsupported model for E-Graph caching: " + model_name);
    }

    std::vector<LogicalId> topo = topologicalSort(roots.roots, ctx->graph);

    Planner planner(ctx->costModel, ctx->settings);
    planner.initBaseEGraph(ctx->rootId, ctx->graph, topo, nullptr);

    ctx->baseEGraph = planner.baseState.egraph;
    ctx->baseNodeToEClass = planner.baseState.nodeToEClass;
    ctx->baseEclassToLogical = planner.baseState.eclassToLogical;

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

    auto cache_iter = makeConfiguredCacheIterator(ctx->graph, candidates, avail_mem_spaces, delegate, ctx->settings);
    std::unordered_map<LogicalId, MemSpace> cachedNodes;

    std::vector<float> all_costs;
    uint32_t cache_eval_count = 0;
    Planner planner(ctx->costModel, ctx->settings);

    while (cache_iter.getNextCacheSelection(cachedNodes))
    {
        cache_eval_count++;

        // 1. Generate deterministic string key for cached node configuration
        std::vector<LogicalId> keys;
        keys.reserve(cachedNodes.size());
        for (const auto &kv : cachedNodes)
            keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());

        std::stringstream ss;
        for (auto k : keys)
        {
            ss << k.value << ":" << cachedNodes.at(k).idx << ":" << static_cast<int>(cachedNodes.at(k).type) << ";";
        }
        std::string state_key = ss.str();

        std::shared_ptr<SaturatedState> state;

        // 2. Query saturation cache
        if (ctx->saturationCache.count(state_key))
        {
            state = ctx->saturationCache[state_key];
        }
        else
        {
            // Cache Miss: Perform injection, saturation, cost evaluation, and pruning
            state = std::make_shared<SaturatedState>();
            state->preallocatedBuffers = ctx->preallocateLogicalBuffers(cachedNodes);

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
                for (size_t i = 0; i < cls.enodes.size(); i++)
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

            state->egraph = std::move(egraph);
            state->eclassToLogical = std::move(eclassToLogical);
            state->enodeInfos = std::move(enodeInfos);

            ctx->saturationCache[state_key] = state;
        }

        EClassId rootEClassId = state->egraph.findConst(ctx->baseNodeToEClass.at(ctx->rootId));
        if (state->egraph.getEClass(rootEClassId).enodes.empty())
        {
            if (delegate)
                delegate->on_leaf_evaluated(-1.0f);
            if (cache_eval_count >= num_cache)
                break;
            continue;
        }

        if (delegate)
        {
            std::vector<float> node_features;
            std::vector<uint32_t> edge_src;
            std::vector<uint32_t> edge_dst;

            uint32_t num_classes = static_cast<uint32_t>(state->egraph.getClasses().size());
            uint32_t num_enodes = static_cast<uint32_t>(state->egraph.getENodes().size());

            for (uint32_t i = 0; i < num_classes; ++i)
            {
                const EClass &cls = state->egraph.getClasses()[i];
                node_features.push_back(1.0f); // is_eclass
                node_features.push_back(0.0f); // is_enode
                node_features.push_back(static_cast<float>(countElements(cls.shape) * getDTypeSize(cls.dtype)));
                node_features.push_back(static_cast<float>(cls.dtype));
                node_features.push_back(0.0f); // dp_cost pad

                for (ENodeId enode_id : cls.enodes)
                {
                    edge_src.push_back(i);
                    edge_dst.push_back(num_classes + enode_id.value);
                }
            }
            for (uint32_t i = 0; i < num_enodes; ++i)
            {
                const ENode &enode = state->egraph.getENodes()[i];
                node_features.push_back(0.0f); // is_eclass
                node_features.push_back(1.0f); // is_enode
                node_features.push_back(state->enodeInfos[i].cost);
                node_features.push_back(static_cast<float>(enode.getOpType()));
                node_features.push_back(state->enodeInfos[i].dp_cost);

                for (EClassId child : enode.getChildren())
                {
                    edge_src.push_back(num_classes + i);
                    edge_dst.push_back(state->egraph.findConst(child).value);
                }
            }
            delegate->init_egraph(node_features, edge_src, edge_dst);
        }

        std::unordered_map<MemSpace, uint64_t> reduced_caps;
        std::unordered_map<MemSpace, uint64_t> reserved_per_ms;
        for (const auto &kv : ctx->mem->getMemCaps())
        {
            reduced_caps[kv.first] = kv.second;
        }
        for (const auto &kv : state->preallocatedBuffers)
        {
            uint64_t extent = static_cast<uint64_t>(kv.second.offset) + kv.second.size;
            reserved_per_ms[kv.second.mem_space] = std::max(reserved_per_ms[kv.second.mem_space], extent);
        }
        for (const auto &kv : reserved_per_ms)
        {
            if (reduced_caps.count(kv.first))
            {
                if (kv.second >= reduced_caps[kv.first])
                {
                    reduced_caps[kv.first] = 0;
                }
                else
                {
                    reduced_caps[kv.first] -= kv.second;
                }
            }
        }

        auto extractor = makeConfiguredExtractor(state->egraph, rootEClassId, state->enodeInfos, delegate,
                                                 ctx->settings, nullptr, &reduced_caps);
        extractor.registerValidator(std::make_unique<CycleValidator>(state->egraph));

        uint32_t extract_count = 0;
        while (extractor.getNextSelection())
        {
            extract_count++;
            const auto &selection_map = extractor.selection_map;

            // Dispatch (Level 2)
            auto dispatch_iterator = makeConfiguredDispatchIterator(state->egraph, selection_map, state->enodeInfos,
                                                                    delegate, ctx->settings, nullptr, &reduced_caps);
            uint32_t dispatch_count = 0;
            std::vector<EClassId> order;

            while (dispatch_iterator.getNextDispatchOrder(selection_map, order))
            {
                dispatch_count++;

                // Bufferize (Level 3)
                auto buf_iter = makeConfiguredBufferizeIterator(order, state->egraph, selection_map, state->enodeInfos,
                                                                reduced_caps, delegate, ctx->settings);

                uint32_t buf_count = 0;
                std::vector<ParallelBuffer> unallocated_buffers;
                std::unordered_map<EClassId, BufferId> eclass_to_buf_local;

                while (buf_iter.getNextBufferization(unallocated_buffers, eclass_to_buf_local))
                {
                    buf_count++;

                    // Malloc (Level 4)
                    std::unordered_set<BufferId> preallocated_buf_ids;
                    std::unordered_map<BufferId, ParallelBuffer> preallocated_overrides;

                    for (EClassId eclass : order)
                    {
                        auto logicalIt = state->eclassToLogical.find(eclass);
                        if (logicalIt == state->eclassToLogical.end())
                            continue;
                        auto sel_it = selection_map.find(eclass);
                        if (sel_it == selection_map.end())
                            continue;
                        uint32_t sel = sel_it->second;
                        ENodeId enode_id = state->egraph.getEClass(eclass).enodes[sel];
                        const ENode &node = state->egraph.getENode(enode_id);
                        if (node.getOpType() != OpType::INPUT && node.getOpType() != OpType::CACHE)
                            continue;

                        auto preIt = state->preallocatedBuffers.find(logicalIt->second);
                        if (preIt == state->preallocatedBuffers.end())
                            continue;

                        BufferId buf_id = eclass_to_buf_local.at(eclass);
                        preallocated_buf_ids.insert(buf_id);
                        preallocated_overrides[buf_id] = preIt->second;
                    }

                    std::unordered_map<MemSpace, std::vector<ParallelBuffer>> buf_by_mem_space;
                    for (auto &buf : unallocated_buffers)
                    {
                        if (buf.mem_space.type == HandleType::STORAGE || preallocated_buf_ids.count(buf.id))
                            continue;
                        buf_by_mem_space[buf.mem_space].push_back(buf);
                    }

                    bool alloc_ok = true;
                    BufferId overflow;
                    size_t total_alloc_count = 0;
                    size_t total_unalloc_count = 0;

                    for (auto &kv : buf_by_mem_space)
                    {
                        total_unalloc_count += kv.second.size();
                        MemSpace ms = kv.first;
                        uint64_t cap =
                            reduced_caps.count(ms) ? reduced_caps.at(ms) : std::numeric_limits<uint64_t>::max();
                        std::vector<ParallelBuffer> allocated;
                        if (!malloc_by_time_components(cap, kv.second, allocated, overflow, delegate, &ctx->settings))
                        {
                            alloc_ok = false;
                            break;
                        }
                        total_alloc_count += allocated.size();
                    }

                    if (alloc_ok)
                    {
                        float cost = get_cost(order, state->egraph, selection_map, state->enodeInfos);
                        all_costs.push_back(cost);
                        if (delegate)
                            delegate->on_leaf_evaluated(cost);
                    }
                    else
                    {
                        // Malloc / OOM failure in range [-0.25, 0.0)
                        float prog = static_cast<float>(total_alloc_count) /
                                     std::max(1.0f, static_cast<float>(total_unalloc_count));
                        float shaped_reward = -0.25f + 0.25f * std::clamp(prog, 0.0f, 0.999f);
                        if (delegate)
                            delegate->on_leaf_evaluated(shaped_reward);
                    }

                    if (buf_count >= num_bufferize)
                        break;
                }

                // If bufferize generated 0 valid configurations, emit bufferize failure reward in [-0.50, -0.25)
                if (buf_count == 0)
                {
                    float prog = static_cast<float>(buf_iter.k) /
                                 std::max(1.0f, static_cast<float>(order.size()));
                    float shaped_reward = -0.50f + 0.25f * std::clamp(prog, 0.0f, 0.999f);
                    if (delegate)
                        delegate->on_leaf_evaluated(shaped_reward);
                }

                if (dispatch_count >= num_dispatch)
                    break;
            }

            // If dispatch generated 0 valid orders, emit dispatch failure reward in [-0.75, -0.50)
            if (dispatch_count == 0)
            {
                float prog = static_cast<float>(order.size()) /
                             std::max(1.0f, static_cast<float>(selection_map.size()));
                float shaped_reward = -0.75f + 0.25f * std::clamp(prog, 0.0f, 0.999f);
                if (delegate)
                    delegate->on_leaf_evaluated(shaped_reward);
            }

            extractor.ascend();
            if (extract_count >= num_extract)
                break;
        }

        // If extractor generated 0 valid selections, emit extractor failure reward in [-1.00, -0.75)
        if (extract_count == 0)
        {
            float prog = static_cast<float>(extractor.path.size()) /
                         std::max(1.0f, static_cast<float>(state->egraph.getClasses().size()));
            float shaped_reward = -1.00f + 0.25f * std::clamp(prog, 0.0f, 0.999f);
            if (delegate)
                delegate->on_leaf_evaluated(shaped_reward);
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