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
    EGraph saturatedEGraph;
    std::unordered_map<LogicalId, EClassId> nodeToEClass;
    std::unordered_map<EClassId, LogicalId> eclassToLogical;
    std::unordered_map<LogicalId, MemSpace> cachedNodes;
    std::unordered_map<LogicalId, ParallelBuffer> preallocatedBuffers;
    CostModel costModel;
    std::vector<ENodeInfo> enodeInfos;

    SaturatedEGraphContext() : costModel(false)
    {
    }
};

inline std::shared_ptr<SaturatedEGraphContext> build_and_saturate_egraph(const std::string &model_name,
                                                                         const std::string &model_path)
{
    auto ctx = std::make_shared<SaturatedEGraphContext>();

    // TODO: Load memory sizes dynamically from hardware probing configuration
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
    }
    else if (model_name == "qwen-3.6-35b-a3b")
    {
        roots = build_qwen_graph(ctx->graph, *ctx->mem, model_path, max_seq_len);
    }
    else
    {
        Error::throw_err("Unsupported model for E-Graph caching: " + model_name);
    }

    ctx->rootId = roots.roots[0];
    std::vector<LogicalId> topo = topologicalSort(roots.roots, ctx->graph);

    Planner planner(ctx->costModel, ctx->mem->getMemCaps());
    planner.initBaseEGraph(ctx->rootId, ctx->graph, topo, nullptr);

    ctx->saturatedEGraph = planner.baseState.egraph;
    ctx->nodeToEClass = planner.baseState.nodeToEClass;
    ctx->eclassToLogical = planner.baseState.eclassToLogical;

    // Run rewrite rules and saturation once across the graph
    std::unordered_set<EClassId> protectedEClasses;
    planner.saturate(ctx->saturatedEGraph, protectedEClasses, ctx->eclassToLogical, true, false, nullptr);

    // Normalize canonical mappings after saturation
    std::unordered_map<EClassId, LogicalId> updatedEClassToLogical;
    for (const auto &kv : ctx->eclassToLogical)
    {
        updatedEClassToLogical[ctx->saturatedEGraph.find(kv.first)] = kv.second;
    }
    ctx->eclassToLogical = std::move(updatedEClassToLogical);

    // Load cost model once during initialization
    ctx->costModel.load("benchmarks/records.bin");

    // Calculate cost and view information for E-nodes once during context creation
    ctx->enodeInfos.resize(ctx->saturatedEGraph.getENodes().size());
    for (uint32_t i = 0; i < ctx->saturatedEGraph.getENodes().size(); ++i)
    {
        const ENode &enode = ctx->saturatedEGraph.getENodes()[i];
        ENodeInfo info;
        info.is_view = false;

        if (enode.getKernelId() != KernelId{0})
        {
            const auto &kernel = KernelRegistry::get().getKernel(enode.getKernelId());
            info.is_view = kernel.is_view;
        }

        if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
        {
            info.cost = 0.0f;
        }
        else if (enode.getKernelId() != KernelId{0})
        {
            std::vector<std::vector<uint32_t>> inShapes;
            std::vector<std::vector<uint64_t>> inStrides;
            std::vector<DType> inDTypes;
            std::vector<std::vector<uint8_t>> inConstants;

            for (EClassId childEClassId : enode.getChildren())
            {
                const EClass &childCls = ctx->saturatedEGraph.getEClass(ctx->saturatedEGraph.find(childEClassId));
                inShapes.push_back(childCls.shape);
                inStrides.push_back(childCls.strides);
                inDTypes.push_back(childCls.dtype);
                inConstants.push_back({});
            }

            info.cost = ctx->costModel.estimateCost(enode.getKernelId(), enode.getShape(), enode.getStrides(),
                                                    enode.getDType(), inShapes, inStrides, inDTypes, inConstants);
        }

        ctx->enodeInfos[i] = info;
    }

    return ctx;
}

inline float extract_best_from_egraph(std::shared_ptr<SaturatedEGraphContext> ctx,
                                      std::shared_ptr<SearchDelegate> delegate, bool log_cost_calls)
{
    ctx->costModel.setLogging(log_cost_calls);

    // Copy the pre-saturated EGraph state for this extraction pass
    EGraph egraph_working_copy = ctx->saturatedEGraph;
    auto eclassToLogical_copy = ctx->eclassToLogical;

    Planner planner(ctx->costModel, ctx->mem->getMemCaps());

    // Run extraction with the custom Python SearchDelegate, passing precalculated enodeInfos
    auto extraction = planner.extractBest(ctx->rootId, ctx->graph, egraph_working_copy, ctx->nodeToEClass,
                                          ctx->cachedNodes, eclassToLogical_copy, ctx->preallocatedBuffers,
                                          /*stopOnFirstValid=*/true, /*strictCache=*/false, /*minCompileSeconds=*/0.0f,
                                          delegate, ctx->enodeInfos);

    return extraction.cost;
}