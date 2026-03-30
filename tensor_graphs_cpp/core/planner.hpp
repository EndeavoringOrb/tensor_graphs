#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/cost_model.hpp"
#include "core/kernels.hpp"
#include "core/rewrite.hpp"
#include "core/hashing.hpp"
#include "core/shapes.hpp"
#include "core/misc.hpp"
#include "core/egraph.hpp"
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

inline bool updateRegionListIfChanged(std::vector<Region> &dst, const std::vector<Region> &src)
{
    std::vector<Region> merged = mergeRegions(dst, src);
    if (encodeRegionList(merged) == encodeRegionList(dst))
        return false;
    dst = std::move(merged);
    return true;
}

struct PlanningRegionState
{
    std::unordered_map<uint32_t, std::vector<Region>> needed;
    std::unordered_map<uint32_t, std::vector<Region>> recompute;
};

// Worklist backward pass to compute needed regions from the root.
static void updateNeeded(
    uint32_t rootId,
    const Graph &graph,
    ShapePropagator &prop,
    std::unordered_map<uint32_t, std::vector<Region>> &needed)
{
    if (!graph.hasNode(rootId))
        return;

    std::vector<uint32_t> worklist = {rootId};
    std::unordered_set<uint32_t> queued = {rootId};

    while (!worklist.empty())
    {
        uint32_t nodeId = worklist.back();
        worklist.pop_back();
        queued.erase(nodeId);

        if (!graph.hasNode(nodeId))
            continue;

        const TensorNode &node = graph.getNode(nodeId);
        if (node.opType == OpType::INPUT)
            continue;

        auto neededIt = needed.find(nodeId);
        if (neededIt == needed.end() || neededIt->second.empty())
            continue;

        std::vector<std::vector<Region>> parentNeeded = prop.backward(node, graph, neededIt->second);
        for (size_t i = 0; i < node.parentIds.size(); ++i)
        {
            if (i >= parentNeeded.size() || parentNeeded[i].empty())
                continue;

            uint32_t parentId = node.parentIds[i];
            if (!graph.hasNode(parentId))
                continue;

            const TensorNode &parentNode = graph.getNode(parentId);
            if (parentNode.opType == OpType::INPUT)
                continue;

            if (updateRegionListIfChanged(needed[parentId], parentNeeded[i]) && !queued.count(parentId))
            {
                worklist.push_back(parentId);
                queued.insert(parentId);
            }
        }
    }
}

static PlanningRegionState derivePlanningRegions(
    uint32_t rootId,
    const Graph &graph,
    const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
    const std::unordered_set<uint32_t> &cachedNodes)
{
    PlanningRegionState state;
    ShapePropagator prop;

    auto rootIt = dirtyOutputRegions.find(rootId);
    if (rootIt != dirtyOutputRegions.end())
        state.needed[rootId] = mergeRegions(rootIt->second);
    else
        state.needed[rootId] = {};

    updateNeeded(rootId, graph, prop, state.needed);

    for (const auto &pair : state.needed)
    {
        uint32_t nodeId = pair.first;
        const std::vector<Region> &neededRegions = pair.second;
        if (neededRegions.empty())
            continue;

        auto dirtyIt = dirtyOutputRegions.find(nodeId);
        if (cachedNodes.count(nodeId))
        {
            if (dirtyIt != dirtyOutputRegions.end() && !dirtyIt->second.empty())
                state.recompute[nodeId] = mergeRegions(dirtyIt->second);
        }
        else
        {
            state.recompute[nodeId] = mergeRegions(neededRegions);
        }
    }

    return state;
}

struct CacheAwarePlanningGraph
{
    Graph graph;
    uint32_t physicalRootId = 0;
    std::unordered_map<uint32_t, uint32_t> logicalToPhysicalNodeMap;
    std::unordered_map<uint32_t, uint32_t> physicalToLogicalNodeMap;
    std::unordered_map<uint32_t, std::vector<Region>> physicalOutputRegions;
    std::unordered_map<uint32_t, std::vector<std::vector<Region>>> physicalInputSlices;
    std::unordered_map<uint32_t, std::vector<uint32_t>> logicalPartialNodeMap;
    std::unordered_map<uint32_t, std::vector<uint32_t>> logicalScatterNodeMap;
};

inline std::vector<uint32_t> getRegionShape(const Region &region)
{
    std::vector<uint32_t> shape;
    shape.reserve(region.region.size());
    for (const Dim &dim : region.region)
        shape.push_back(dim.stop - dim.start);
    return shape;
}

class CacheAwarePlanningGraphBuilder
{
public:
    CacheAwarePlanningGraphBuilder(
        const Graph &sourceGraph,
        const PlanningRegionState &regionState,
        const std::unordered_set<uint32_t> &cachedNodes)
        : sourceGraph(sourceGraph), regionState(regionState), cachedNodes(cachedNodes)
    {
        result.graph = sourceGraph;
    }

    CacheAwarePlanningGraph build(uint32_t rootId)
    {
        result.physicalRootId = buildLogicalNode(rootId);
        return result;
    }

private:
    struct PartialCloneResult
    {
        uint32_t nodeId = 0;
        std::vector<std::vector<Region>> parentRegions;
    };

    const Graph &sourceGraph;
    const PlanningRegionState &regionState;
    const std::unordered_set<uint32_t> &cachedNodes;
    ShapePropagator prop;
    CacheAwarePlanningGraph result;
    std::unordered_map<uint32_t, uint32_t> memoizedPhysicalIds;

    static std::vector<int32_t> toInt32Shape(const std::vector<uint32_t> &shape)
    {
        std::vector<int32_t> out;
        out.reserve(shape.size());
        for (uint32_t dim : shape)
            out.push_back(static_cast<int32_t>(dim));
        return out;
    }

    uint32_t addInt32Constant(const std::vector<int32_t> &values)
    {
        return result.graph.constant({static_cast<uint32_t>(values.size())}, values.data(), DType::INT32);
    }

    uint32_t addRegionStarts(const Region &region)
    {
        std::vector<int32_t> starts;
        starts.reserve(region.region.size());
        for (const Dim &dim : region.region)
            starts.push_back(static_cast<int32_t>(dim.start));
        return addInt32Constant(starts);
    }

    uint32_t addRegionEnds(const Region &region)
    {
        std::vector<int32_t> ends;
        ends.reserve(region.region.size());
        for (const Dim &dim : region.region)
            ends.push_back(static_cast<int32_t>(dim.stop));
        return addInt32Constant(ends);
    }

    uint32_t addRegionSteps(const Region &region)
    {
        std::vector<int32_t> steps(region.region.size(), 1);
        return addInt32Constant(steps);
    }

    bool shouldSliceParentInput(const TensorNode &node, size_t parentIdx) const
    {
        switch (node.opType)
        {
        case OpType::SUM:
        case OpType::MAX:
        case OpType::PERMUTE:
        case OpType::TRIU:
        case OpType::RESHAPE:
        case OpType::FILL:
            return parentIdx == 0;
        case OpType::REPEAT:
            return parentIdx == 0;
        case OpType::CONCAT:
            return parentIdx + 1 < node.parentIds.size();
        case OpType::SLICE:
            return parentIdx == 0;
        case OpType::SCATTER:
            return parentIdx < 2;
        case OpType::GATHER:
            return parentIdx == 1;
        default:
            return true;
        }
    }

    void convertLogicalNodeToPlaceholder(uint32_t logicalId)
    {
        TensorNode &placeholder = result.graph.getNode(logicalId);
        placeholder.opType = OpType::INPUT;
        placeholder.parentIds.clear();
        placeholder.view.shape = placeholder.shape;
        placeholder.view.strides = TensorView::calcContiguousStrides(placeholder.shape);
        placeholder.view.baseOffset = 0;
        placeholder.view.dtype = placeholder.dtype;
        result.physicalToLogicalNodeMap[logicalId] = logicalId;
    }

    uint32_t cloneNode(const TensorNode &sourceNode, const std::vector<uint32_t> &parentIds)
    {
        TensorNode &cloned = result.graph.allocateNode();
        uint32_t clonedId = cloned.id;
        cloned = sourceNode;
        cloned.id = clonedId;
        cloned.parentIds = parentIds;
        return clonedId;
    }

    void retargetPartialShapeInputs(uint32_t newNodeId, const TensorNode &sourceNode, const Region &outRegion)
    {
        TensorNode &newNode = result.graph.getNode(newNodeId);
        const std::vector<uint32_t> outShape = getRegionShape(outRegion);

        if (sourceNode.opType == OpType::RESHAPE || sourceNode.opType == OpType::FILL)
        {
            if (newNode.parentIds.size() > 1)
                newNode.parentIds[1] = addInt32Constant(toInt32Shape(outShape));
            return;
        }

        if (sourceNode.opType == OpType::SLICE)
        {
            Region localRegion;
            for (uint32_t dim : outShape)
                localRegion.region.push_back({0, dim});
            if (newNode.parentIds.size() > 3)
            {
                newNode.parentIds[1] = addRegionStarts(localRegion);
                newNode.parentIds[2] = addRegionEnds(localRegion);
                newNode.parentIds[3] = addRegionSteps(localRegion);
            }
            return;
        }

        if (sourceNode.opType == OpType::SCATTER)
        {
            Region localRegion;
            for (uint32_t dim : outShape)
                localRegion.region.push_back({0, dim});
            if (newNode.parentIds.size() > 4)
            {
                newNode.parentIds[2] = addRegionStarts(localRegion);
                newNode.parentIds[3] = addRegionEnds(localRegion);
                newNode.parentIds[4] = addRegionSteps(localRegion);
            }
            return;
        }

        if (sourceNode.opType == OpType::ARANGE)
        {
            std::vector<int32_t> startVals = getConstantInt32(sourceNode.parentIds[0], sourceGraph);
            std::vector<int32_t> stopVals = getConstantInt32(sourceNode.parentIds[1], sourceGraph);
            std::vector<int32_t> stepVals = getConstantInt32(sourceNode.parentIds[2], sourceGraph);
            if (!startVals.empty() && !stopVals.empty() && !stepVals.empty())
            {
                const int32_t baseStart = startVals[0];
                const int32_t step = stepVals[0];
                const int32_t localStart = baseStart + static_cast<int32_t>(outRegion.region[0].start) * step;
                const int32_t localStop = baseStart + static_cast<int32_t>(outRegion.region[0].stop) * step;
                newNode.parentIds[0] = addInt32Constant({localStart});
                newNode.parentIds[1] = addInt32Constant({localStop});
                newNode.parentIds[2] = addInt32Constant({step});
            }
        }
    }

    uint32_t buildParentInputSlice(
        const TensorNode &sourceNode,
        size_t parentIdx,
        uint32_t physicalParentId,
        const std::vector<Region> &parentRegions)
    {
        if (!shouldSliceParentInput(sourceNode, parentIdx))
            return physicalParentId;
        if (parentRegions.empty())
            return physicalParentId;
        if (parentRegions.size() != 1)
            return physicalParentId;

        const Region &inputRegion = parentRegions.front();
        if (inputRegion.empty())
            return physicalParentId;

        const TensorNode &sourceParent = sourceGraph.getNode(sourceNode.parentIds[parentIdx]);
        uint32_t safeParentId = result.graph.contiguous(physicalParentId);

        Region producerRegion;
        auto it = result.physicalOutputRegions.find(physicalParentId);
        if (it != result.physicalOutputRegions.end())
        {
            producerRegion = it->second.front();
        }
        else
        {
            auto fullRegions = makeFull(sourceParent.shape);
            if (!fullRegions.empty())
            {
                producerRegion = fullRegions.front();
            }
        }

        Region localSlice = inputRegion;
        for (size_t d = 0; d < localSlice.region.size(); ++d)
        {
            if (d < producerRegion.region.size())
            {
                localSlice.region[d].start -= producerRegion.region[d].start;
                localSlice.region[d].stop -= producerRegion.region[d].start;
            }
        }

        uint32_t startsId = addRegionStarts(localSlice);
        uint32_t endsId = addRegionEnds(localSlice);
        uint32_t stepsId = addRegionSteps(localSlice);
        uint32_t sliceId = result.graph.slice(safeParentId, startsId, endsId, stepsId);
        return result.graph.contiguous(sliceId);
    }

    PartialCloneResult buildPartialClone(uint32_t logicalId, const Region &outRegion)
    {
        const TensorNode &sourceNode = sourceGraph.getNode(logicalId);
        std::vector<std::vector<Region>> parentRegions = prop.backward(sourceNode, sourceGraph, {outRegion});

        std::vector<uint32_t> partialParents;
        partialParents.reserve(sourceNode.parentIds.size());
        for (size_t parentIdx = 0; parentIdx < sourceNode.parentIds.size(); ++parentIdx)
        {
            uint32_t physicalParentId = buildLogicalNode(sourceNode.parentIds[parentIdx]);
            const std::vector<Region> emptyRegions;
            const std::vector<Region> &regionsForParent = parentIdx < parentRegions.size() ? parentRegions[parentIdx] : emptyRegions;
            partialParents.push_back(buildParentInputSlice(sourceNode, parentIdx, physicalParentId, regionsForParent));
        }

        uint32_t partialId = cloneNode(sourceNode, partialParents);
        TensorNode &partialNode = result.graph.getNode(partialId);
        partialNode.shape = getRegionShape(outRegion);
        partialNode.view.shape.clear();
        partialNode.view.strides.clear();
        partialNode.view.baseOffset = 0;
        partialNode.view.dtype = partialNode.dtype;
        retargetPartialShapeInputs(partialId, sourceNode, outRegion);
        result.physicalToLogicalNodeMap[partialId] = logicalId;
        result.physicalOutputRegions[partialId] = {outRegion};
        result.physicalInputSlices[partialId] = parentRegions;
        return {partialId, parentRegions};
    }

    uint32_t buildLogicalNode(uint32_t logicalId)
    {
        auto memoIt = memoizedPhysicalIds.find(logicalId);
        if (memoIt != memoizedPhysicalIds.end())
            return memoIt->second;

        if (!sourceGraph.hasNode(logicalId))
            return logicalId;

        const TensorNode &sourceNode = sourceGraph.getNode(logicalId);
        if (sourceNode.opType == OpType::INPUT)
        {
            result.logicalToPhysicalNodeMap[logicalId] = logicalId;
            result.physicalToLogicalNodeMap[logicalId] = logicalId;
            memoizedPhysicalIds[logicalId] = logicalId;
            return logicalId;
        }

        auto recomputeIt = regionState.recompute.find(logicalId);
        if (recomputeIt == regionState.recompute.end() || recomputeIt->second.empty())
        {
            convertLogicalNodeToPlaceholder(logicalId);
            result.logicalToPhysicalNodeMap[logicalId] = logicalId;
            result.physicalToLogicalNodeMap[logicalId] = logicalId;
            memoizedPhysicalIds[logicalId] = logicalId;
            return logicalId;
        }

        const std::vector<Region> recomputeRegions = normalizeRegions(recomputeIt->second);
        const bool singleFullRegion = recomputeRegions.size() == 1 && isFullRegion(recomputeRegions.front(), sourceNode.shape);
        if (singleFullRegion)
        {
            std::vector<uint32_t> rebuiltParents;
            rebuiltParents.reserve(sourceNode.parentIds.size());
            for (uint32_t parentId : sourceNode.parentIds)
                rebuiltParents.push_back(buildLogicalNode(parentId));

            uint32_t rebuiltId = cloneNode(sourceNode, rebuiltParents);
            result.logicalToPhysicalNodeMap[logicalId] = rebuiltId;
            result.physicalToLogicalNodeMap[rebuiltId] = logicalId;
            result.physicalOutputRegions[rebuiltId] = recomputeRegions;
            memoizedPhysicalIds[logicalId] = rebuiltId;
            return rebuiltId;
        }

        if (cachedNodes.count(logicalId))
        {
            convertLogicalNodeToPlaceholder(logicalId);
            uint32_t currentId = logicalId;

            for (const Region &recomputeRegion : recomputeRegions)
            {
                PartialCloneResult partial = buildPartialClone(logicalId, recomputeRegion);
                result.logicalPartialNodeMap[logicalId].push_back(partial.nodeId);

                uint32_t startsId = addRegionStarts(recomputeRegion);
                uint32_t endsId = addRegionEnds(recomputeRegion);
                uint32_t stepsId = addRegionSteps(recomputeRegion);
                uint32_t scatterId = result.graph.scatter(currentId, partial.nodeId, startsId, endsId, stepsId);

                result.logicalScatterNodeMap[logicalId].push_back(scatterId);
                result.physicalToLogicalNodeMap[scatterId] = logicalId;
                result.physicalOutputRegions[scatterId] = {sourceNode.fullRegion()};

                currentId = scatterId;
            }

            result.logicalToPhysicalNodeMap[logicalId] = currentId;
            memoizedPhysicalIds[logicalId] = currentId;
            return currentId;
        }
        else
        {
            if (recomputeRegions.size() > 1)
            {
                Error::throw_err("Uncached nodes with multiple recompute regions are not supported yet!");
            }
            PartialCloneResult partial = buildPartialClone(logicalId, recomputeRegions.front());
            result.logicalToPhysicalNodeMap[logicalId] = partial.nodeId;
            result.physicalToLogicalNodeMap[partial.nodeId] = logicalId;
            result.physicalOutputRegions[partial.nodeId] = recomputeRegions;
            memoizedPhysicalIds[logicalId] = partial.nodeId;
            return partial.nodeId;
        }
    }
};

static CacheAwarePlanningGraph buildCacheAwarePlanningGraph(
    uint32_t rootId,
    const Graph &graph,
    const PlanningRegionState &regionState,
    const std::unordered_set<uint32_t> &cachedNodes)
{
    CacheAwarePlanningGraphBuilder builder(graph, regionState, cachedNodes);
    return builder.build(rootId);
}

class Planner
{
public:
    Planner(CostModel &costModel, std::unordered_map<Backend, uint64_t> maxMemoryByBackend = {})
        : costModel(costModel), maxMemoryByBackend(std::move(maxMemoryByBackend)) {}

    CompiledGraph plan(
        uint32_t rootId,
        Graph &graph,
        const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
        const std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &dirtyInputRegions,
        const std::unordered_set<uint32_t> &cachedNodes)
    {
        // Planning pipeline:
        // 1. seed the root dirty region
        // 2. derive needed regions backward from the root
        // 3. derive cache-aware recompute regions
        // 4. saturate / extract / emit a compiled graph for this bucket
        if (InterruptManager::isInterrupted())
        {
            std::cerr << "\n[Executor] Interrupt detected, aborting execution..." << std::endl;
            InterruptManager::cleanup();
            std::exit(SIGINT);
        }
        PlanningRegionState regionState = derivePlanningRegions(rootId, graph, dirtyOutputRegions, cachedNodes);
        CacheAwarePlanningGraph planningGraph = buildCacheAwarePlanningGraph(rootId, graph, regionState, cachedNodes);

        std::vector<uint32_t> topo = topologicalSort(planningGraph.physicalRootId, planningGraph.graph);
        inferShapes(topo, planningGraph.graph);

        auto refCounts = computeRefCounts(topo, planningGraph.physicalRootId, planningGraph.graph);

        EGraph egraph;
        std::unordered_map<uint32_t, uint32_t> nodeToEClass;
        nodeToEClass.reserve(planningGraph.graph.nodes.size());

        for (uint32_t nodeId : topo)
        {
            TensorNode &node = planningGraph.graph.getNode(nodeId);
            ensureContiguousView(node);
            uint32_t refCount = 0;
            auto rcIt = refCounts.find(nodeId);
            if (rcIt != refCounts.end())
                refCount = rcIt->second;
            uint32_t eclassId = egraph.addEClass(node.shape, node.dtype, refCount, node.view.isContiguous());
            egraph.getEClass(eclassId).backends.insert(node.backend);
            nodeToEClass[nodeId] = eclassId;
        }

        // Seed with reference kernels only
        for (uint32_t nodeId : topo)
        {
            const TensorNode &node = planningGraph.graph.getNode(nodeId);
            uint32_t eclassId = nodeToEClass[nodeId];
            if (node.opType == OpType::INPUT || node.opType == OpType::CONTIGUOUS || node.opType == OpType::SLICE)
            {
                if (node.opType != OpType::INPUT)
                {
                    std::vector<TensorNode> inputs;
                    for (uint32_t pid : node.parentIds)
                        inputs.push_back(planningGraph.graph.getNode(pid));
                    std::vector<uint64_t> refs = KernelRegistry::get().findMatchingKernels(node.opType, node.opName, node.backend, inputs, node, refCounts, true);
                    if (refs.empty())
                    {
                        Error::throw_err("No reference kernel found for SLICE/CONTIGUOUS node.\n" + toString(node, planningGraph.graph));
                    }
                    for (uint64_t uid : refs)
                    {
                        ENode enode;
                        enode.nodeId = nodeId;
                        enode.kernelUid = uid;
                        enode.opType = node.opType;
                        enode.backend = node.backend;
                        for (uint32_t pid : node.parentIds)
                            enode.children.push_back(nodeToEClass[pid]);
                        egraph.addENode(eclassId, enode);
                    }
                }
                else
                {
                    ENode enode;
                    enode.nodeId = nodeId;
                    enode.kernelUid = 0;
                    enode.opType = node.opType;
                    enode.backend = node.backend;
                    for (uint32_t pid : node.parentIds)
                        enode.children.push_back(nodeToEClass[pid]);
                    egraph.addENode(eclassId, enode);
                }
                continue;
            }

            std::vector<TensorNode> inputs;
            for (uint32_t pid : node.parentIds)
                inputs.push_back(planningGraph.graph.getNode(pid));

            std::vector<uint64_t> refs = KernelRegistry::get().findMatchingKernels(
                node.opType, node.opName, node.backend, inputs, node, refCounts, true);
            if (refs.empty())
            {
                Error::throw_err("No reference kernel found for node " + toString(node, planningGraph.graph, ""));
            }

            for (uint64_t uid : refs)
            {
                ENode enode;
                enode.nodeId = nodeId;
                enode.kernelUid = uid;
                enode.opType = node.opType;
                enode.opName = node.opName;
                enode.backend = node.backend;
                for (uint32_t pid : node.parentIds)
                    enode.children.push_back(nodeToEClass[pid]);

                egraph.addENode(eclassId, enode);
            }
        }

        saturate(topo, planningGraph.graph, egraph, nodeToEClass, refCounts);

        auto extraction = extractBest(planningGraph.physicalRootId, planningGraph.graph, egraph, nodeToEClass, refCounts, maxMemoryByBackend, cachedNodes);

        return buildCompiledGraph(
            planningGraph.physicalRootId,
            planningGraph.graph,
            egraph,
            nodeToEClass,
            refCounts,
            extraction,
            dirtyOutputRegions,
            dirtyInputRegions,
            planningGraph.logicalToPhysicalNodeMap,
            planningGraph.physicalToLogicalNodeMap,
            planningGraph.physicalOutputRegions,
            planningGraph.physicalInputSlices,
            cachedNodes);
    }

private:
    struct ExtractChoice
    {
        uint32_t enodeId = 0;
        float cost = std::numeric_limits<float>::infinity();
        bool valid = false;
        TensorView view;
        uint64_t memSize = 0; // Memory size for this node's output
    };

    struct ExtractionResult
    {
        std::unordered_map<uint32_t, ExtractChoice> choiceByEClass;
        std::unordered_map<uint32_t, uint32_t> eclassToNodeId;
    };

    CostModel &costModel;
    std::unordered_map<Backend, uint64_t> maxMemoryByBackend;

    uint64_t getMemoryLimit(Backend backend) const
    {
        auto it = maxMemoryByBackend.find(backend);
        if (it != maxMemoryByBackend.end())
            return it->second;
        return std::numeric_limits<uint64_t>::max();
    }

    static void ensureContiguousView(TensorNode &node)
    {
        if (node.view.shape.empty() && !node.shape.empty())
        {
            node.view.shape = node.shape;
            node.view.strides = TensorView::calcContiguousStrides(node.shape);
            node.view.dtype = node.dtype;
            node.view.baseOffset = 0;
        }
    }

    void inferShapes(const std::vector<uint32_t> &topo, Graph &graph)
    {
        ShapePropagator propagator;
        for (uint32_t nodeId : topo)
        {
            propagator.inferShape(nodeId, graph);
            if (graph.getNode(nodeId).view.shape.empty() && !graph.getNode(nodeId).shape.empty())
            {
                graph.getNode(nodeId).view.shape = graph.getNode(nodeId).shape;
                graph.getNode(nodeId).view.strides = TensorView::calcContiguousStrides(graph.getNode(nodeId).shape);
                graph.getNode(nodeId).view.dtype = graph.getNode(nodeId).dtype;
            }
        }
    }

    std::unordered_map<uint32_t, uint32_t> computeRefCounts(const std::vector<uint32_t> &topo, uint32_t rootId, const Graph &graph) const
    {
        std::unordered_map<uint32_t, uint32_t> refCounts;
        for (uint32_t nodeId : topo)
        {
            for (uint32_t pid : graph.getNode(nodeId).parentIds)
            {
                refCounts[pid]++;
            }
        }
        refCounts[rootId] = std::max<uint32_t>(1, refCounts[rootId]);
        return refCounts;
    }

    std::vector<uint32_t> topologicalSort(const uint32_t rootId, const Graph &graph) const
    {
        std::vector<uint32_t> order;
        std::unordered_set<uint32_t> visited;
        auto visit = [&](auto &self, uint32_t node) -> void
        {
            if (visited.count(node))
                return;
            visited.insert(node);
            for (uint32_t pid : graph.getNode(node).parentIds)
            {
                self(self, pid);
            }
            order.push_back(node);
        };
        visit(visit, rootId);
        return order;
    }

    struct RuleMatch
    {
        uint32_t nodeId = 0;
    };

    struct Rule
    {
        virtual ~Rule() = default;
        virtual bool match(uint32_t nodeId, const Graph &graph, RuleMatch &out) const = 0;
        virtual bool predicate(const RuleMatch &m, const Graph &graph, const EGraph &egraph,
                               const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                               const std::unordered_map<uint32_t, uint32_t> &refCounts) const
        {
            (void)m;
            (void)graph;
            (void)egraph;
            (void)nodeToEClass;
            (void)refCounts;
            return true;
        }
        virtual std::vector<uint32_t> apply(const RuleMatch &m, Graph &graph) const = 0;
    };

    struct GraphRewriteRuleAdapter : public Rule
    {
        const Rewrite::RewriteRule *rule;
        explicit GraphRewriteRuleAdapter(const Rewrite::RewriteRule *r) : rule(r) {}

        bool match(uint32_t nodeId, const Graph &graph, RuleMatch &out) const override
        {
            (void)graph;
            out.nodeId = nodeId;
            return true;
        }

        std::vector<uint32_t> apply(const RuleMatch &m, Graph &graph) const override
        {
            return rule->apply(m.nodeId, graph);
        }
    };

    struct CopyElimRule : public Rule
    {
        bool match(uint32_t nodeId, const Graph &graph, RuleMatch &out) const override
        {
            if (!graph.hasNode(nodeId))
                return false;
            const auto &node = graph.getNode(nodeId);
            if (node.opType != OpType::COPY_TO || node.parentIds.empty())
                return false;
            uint32_t parentId = node.parentIds[0];
            if (!graph.hasNode(parentId))
                return false;
            const auto &parent = graph.getNode(parentId);
            if (parent.opType != OpType::COPY_TO || parent.parentIds.empty())
                return false;
            out.nodeId = nodeId;
            return true;
        }

        std::vector<uint32_t> apply(const RuleMatch &m, Graph &graph) const override
        {
            const TensorNode &node = graph.getNode(m.nodeId);
            const TensorNode &parent = graph.getNode(node.parentIds[0]);
            uint32_t grandparentId = parent.parentIds[0];

            std::vector<uint32_t> results;
            // E.g. copyto(copyto(X, GPU), CPU) => X
            if (node.backend == graph.getNode(grandparentId).backend)
            {
                results.push_back(grandparentId);
            }
            // E.g. copyto(copyto(X, GPU), GPU) => copyto(X, GPU)
            if (node.backend == parent.backend)
            {
                results.push_back(node.parentIds[0]);
            }
            return results;
        }
    };

    struct FusionRule : public Rule
    {
        struct Pattern
        {
            std::string opName;
            uint32_t rootId;
            std::vector<uint32_t> variables;
            std::vector<DType> dtypes;
            std::vector<std::vector<uint32_t>> dummyShapes;
            Graph graph;
        };

        std::vector<Pattern> patterns;

        FusionRule()
        {
            const auto &refGraphs = ReferenceGraphRegistry::get().getAll();
            for (const auto &pair : refGraphs)
            {
                Pattern pattern;
                pattern.opName = pair.first;
                const auto &entry = pair.second;

                for (size_t i = 0; i < entry.numInputs; ++i)
                {
                    TensorNode &node = pattern.graph.allocateNode();
                    uint32_t inId = node.id;
                    TensorView view;
                    view.shape = entry.dummyShapes[i];
                    view.strides = TensorView::calcContiguousStrides(view.shape);
                    view.baseOffset = 0;
                    view.dtype = entry.dtypes[i];
                    pattern.graph.inputWithId(inId, view.shape, view.dtype, view);
                    pattern.variables.push_back(inId);
                }
                pattern.rootId = entry.factory(pattern.variables, pattern.graph);
                pattern.dtypes = entry.dtypes;
                pattern.dummyShapes = entry.dummyShapes;
                patterns.push_back(std::move(pattern));
            }
        }

        bool match(uint32_t nodeId, const Graph &graph, RuleMatch &out) const override
        {
            if (!graph.hasNode(nodeId))
                return false;
            if (graph.getNode(nodeId).opType == OpType::INPUT)
                return false;
            out.nodeId = nodeId;
            return true;
        }

        std::vector<uint32_t> apply(const RuleMatch &m, Graph &graph) const override
        {
            std::vector<uint32_t> results;
            const TensorNode &refNode = graph.getNode(m.nodeId);

            for (const auto &pattern : patterns)
            {
                std::unordered_map<uint32_t, uint32_t> binding;
                if (matchPattern(m.nodeId, graph, pattern.rootId, pattern.graph, pattern.variables, binding, pattern.dtypes))
                {
                    std::vector<uint32_t> inputs;
                    inputs.reserve(pattern.variables.size());
                    for (uint32_t var : pattern.variables)
                        inputs.push_back(binding[var]);

                    for (const auto &kernel : KernelRegistry::get().getAllKernels())
                    {
                        if (kernel.opType == OpType::FUSED && kernel.opName == pattern.opName)
                        {
                            for (Backend targetBackend : kernel.backends)
                            {
                                uint32_t newNode = addFusedNode(graph, kernel, targetBackend, inputs, refNode);
                                if (newNode != UINT32_MAX)
                                    results.push_back(newNode);
                            }
                        }
                    }
                }
            }
            return results;
        }

        uint32_t addFusedNode(Graph &graph, const KernelEntry &kernel, Backend targetBackend, const std::vector<uint32_t> &parentIds, const TensorNode &refNode) const
        {
            std::vector<uint32_t> adaptedParents;
            for (size_t i = 0; i < parentIds.size(); ++i)
            {
                uint32_t pid = parentIds[i];
                const TensorNode &parent = graph.getNode(pid);

                Backend expectedBackend = targetBackend;
                if (i < kernel.inputBackends.size())
                {
                    expectedBackend = kernel.inputBackends[i];
                }

                bool needCopy = (parent.backend != expectedBackend);
                bool needContig = false;
                if (i < kernel.requiresContiguous.size())
                {
                    needContig = kernel.requiresContiguous[i] && !parent.view.isContiguous();
                }

                if (!needCopy && !needContig)
                {
                    adaptedParents.push_back(pid);
                    continue;
                }

                uint32_t currentId = pid;
                if (needCopy && needContig)
                {
                    TensorNode dummyCopyOut = parent;
                    dummyCopyOut.backend = expectedBackend;
                    bool copyWorks = !KernelRegistry::get().findMatchingKernels(OpType::COPY_TO, "", expectedBackend, {parent}, dummyCopyOut, {}, false).empty();

                    TensorNode dummyContigOut = dummyCopyOut;
                    dummyContigOut.view.strides = TensorView::calcContiguousStrides(dummyContigOut.shape);
                    bool contigWorksAfterCopy = !KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", expectedBackend, {dummyCopyOut}, dummyContigOut, {}, false).empty();

                    if (copyWorks && contigWorksAfterCopy)
                    {
                        currentId = graph.copyto(currentId, expectedBackend);
                        currentId = graph.contiguous(currentId);
                    }
                    else
                    {
                        TensorNode dummyContigOut2 = parent;
                        dummyContigOut2.view.strides = TensorView::calcContiguousStrides(dummyContigOut2.shape);
                        bool contigWorks = !KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", parent.backend, {parent}, dummyContigOut2, {}, false).empty();

                        TensorNode dummyCopyOut2 = dummyContigOut2;
                        dummyCopyOut2.backend = expectedBackend;
                        bool copyWorksAfterContig = !KernelRegistry::get().findMatchingKernels(OpType::COPY_TO, "", expectedBackend, {dummyContigOut2}, dummyCopyOut2, {}, false).empty();

                        if (contigWorks && copyWorksAfterContig)
                        {
                            currentId = graph.contiguous(currentId);
                            currentId = graph.copyto(currentId, expectedBackend);
                        }
                        else
                        {
                            Error::throw_err("No valid adapter chain for copyto (" + toString(parent.backend) + " -> " + toString(expectedBackend) + ") and contiguous");
                        }
                    }
                }
                else if (needCopy)
                {
                    currentId = graph.copyto(currentId, expectedBackend);
                }
                else if (needContig)
                {
                    currentId = graph.contiguous(currentId);
                }
                adaptedParents.push_back(currentId);
            }

            TensorNode &node = graph.allocateNode();
            uint32_t id = node.id;
            node.opType = kernel.opType;
            node.opName = kernel.opName;
            node.dtype = refNode.dtype;
            node.shape = refNode.shape;
            node.parentIds = adaptedParents;
            node.backend = targetBackend;

            if (node.backend != refNode.backend)
            {
                id = graph.copyto(id, refNode.backend);
            }

            return id;
        }

        static bool matchPattern(uint32_t concreteId, const Graph &mainGraph,
                                 uint32_t patternId, const Graph &patternGraph,
                                 const std::vector<uint32_t> &patternVariables,
                                 std::unordered_map<uint32_t, uint32_t> &binding,
                                 const std::vector<DType> &patternDtypes)
        {
            auto itVar = std::find(patternVariables.begin(), patternVariables.end(), patternId);
            if (itVar != patternVariables.end())
            {
                size_t varIdx = static_cast<size_t>(std::distance(patternVariables.begin(), itVar));
                const TensorNode &cNode = mainGraph.getNode(concreteId);
                if (varIdx < patternDtypes.size() && cNode.dtype != patternDtypes[varIdx])
                    return false;

                if (binding.count(patternId))
                {
                    return binding[patternId] == concreteId;
                }
                binding[patternId] = concreteId;
                return true;
            }

            const auto &cNode = mainGraph.getNode(concreteId);
            const auto &pNode = patternGraph.getNode(patternId);

            if (cNode.opType != pNode.opType)
                return false;
            if (cNode.opType == OpType::FUSED && cNode.opName != pNode.opName)
                return false;
            if (cNode.parentIds.size() != pNode.parentIds.size())
                return false;

            for (size_t i = 0; i < cNode.parentIds.size(); ++i)
            {
                if (!matchPattern(cNode.parentIds[i], mainGraph, pNode.parentIds[i], patternGraph, patternVariables, binding, patternDtypes))
                {
                    return false;
                }
            }

            return true;
        }
    };

    uint32_t ensureNodeInEGraph(uint32_t nodeId, Graph &graph, EGraph &egraph,
                                std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                                const std::unordered_map<uint32_t, uint32_t> &refCounts)
    {
        if (nodeToEClass.count(nodeId))
        {
            return egraph.find(nodeToEClass[nodeId]);
        }

        const TensorNode &node = graph.getNode(nodeId);

        // First, ensure all parents are in the EGraph (Recursive call)
        for (uint32_t pid : node.parentIds)
        {
            ensureNodeInEGraph(pid, graph, egraph, nodeToEClass, refCounts);
        }

        // Register this node's EClass
        uint32_t refCount = 0;
        auto rcIt = refCounts.find(nodeId);
        if (rcIt != refCounts.end())
            refCount = rcIt->second;

        ensureContiguousView(graph.getNode(nodeId));
        uint32_t eclassId = egraph.addEClass(node.shape, node.dtype, refCount, node.view.isContiguous());
        egraph.getEClass(eclassId).backends.insert(node.backend);
        nodeToEClass[nodeId] = eclassId;

        // Add the Enode for this specific physical implementation
        addBasicEnode(graph, egraph, nodeToEClass, refCounts, nodeId);

        return eclassId;
    }

    void saturate(const std::vector<uint32_t> &topo, Graph &graph, EGraph &egraph,
                  std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                  const std::unordered_map<uint32_t, uint32_t> &refCounts)
    {
        Rewrite::CommutativeRule cr;
        Rewrite::DistributiveRule dr;
        Rewrite::FactoringRule fr;
        Rewrite::AssociativeRule ar;
        Rewrite::DoubleNegationRule dnr;
        Rewrite::NegateAddRule nar;
        Rewrite::DivMulRule dmr;
        Rewrite::DivAddRule dar;
        Rewrite::CopyToContiguousReorderRule ccr;
        Rewrite::CopyToScatterReorderRule csr;

        // TODO: make some sort of rule registry so I don't have to update this every time I add/remove a rule
        std::vector<std::unique_ptr<Rule>> rules;
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&cr));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&dr));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&fr));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&ar));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&dnr));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&nar));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&dmr));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&dar));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&ccr));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&csr));
        rules.emplace_back(std::make_unique<FusionRule>());
        rules.emplace_back(std::make_unique<CopyElimRule>());

        std::vector<uint32_t> worklist = topo;

        size_t iterations = 0;
        const size_t maxIterations = 3;
        while (!worklist.empty() && iterations < maxIterations)
        {
            iterations++;
            std::vector<uint32_t> nextWorklist;
            nextWorklist.reserve(worklist.size());
            for (uint32_t nodeId : worklist)
            {
                for (const auto &rule : rules)
                {
                    RuleMatch match;
                    if (!rule->match(nodeId, graph, match))
                        continue;
                    if (!rule->predicate(match, graph, egraph, nodeToEClass, refCounts))
                        continue;

                    size_t oldSize = graph.nodes.size();
                    std::vector<uint32_t> newNodes = rule->apply(match, graph);
                    if (newNodes.empty())
                    {
                        continue;
                    }
                    // We only want to process new nodes. But nodes are now an unordered_map, so looping by size is broken.
                    // However, we already have `newNodes` list from `rule->apply`
                    for (uint32_t newId : newNodes)
                    {
                        ShapePropagator prop;
                        prop.inferShapeRecursive(newId, graph);
                        ensureContiguousView(graph.getNode(newId));

                        ensureNodeInEGraph(newId, graph, egraph, nodeToEClass, refCounts);

                        addBasicEnode(graph, egraph, nodeToEClass, refCounts, newId);
                        nextWorklist.push_back(newId);
                    }
                    for (uint32_t newId : newNodes)
                    {
                        if (nodeToEClass.count(nodeId) && nodeToEClass.count(newId))
                        {
                            egraph.merge(nodeToEClass[nodeId], nodeToEClass[newId]);
                            nodeToEClass[newId] = egraph.find(nodeToEClass[nodeId]);
                        }
                    }
                }
            }
            egraph.rebuild();
            worklist = std::move(nextWorklist);
        }
    }

    bool addBasicEnode(const Graph &graph, EGraph &egraph,
                       const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                       const std::unordered_map<uint32_t, uint32_t> &refCounts,
                       uint32_t nodeId)
    {
        const TensorNode &node = graph.getNode(nodeId);
        if (node.opType == OpType::INPUT)
            return false;

        std::vector<TensorNode> inputs;
        for (uint32_t pid : node.parentIds)
            inputs.push_back(graph.getNode(pid));

        std::vector<uint64_t> refs = KernelRegistry::get().findMatchingKernels(
            node.opType, node.opName, node.backend, inputs, node, refCounts, false);
        if (refs.empty())
            return false;

        for (uint64_t uid : refs)
        {
            ENode enode;
            enode.nodeId = nodeId;
            enode.kernelUid = uid;
            enode.opType = node.opType;
            enode.opName = node.opName;
            enode.backend = node.backend;
            for (uint32_t pid : node.parentIds)
                enode.children.push_back(nodeToEClass.at(pid));
            egraph.addENode(nodeToEClass.at(nodeId), enode);
        }
        return true;
    }

    // Helper: Calculate peak memory usage with liveness tracking
    // Used for memory-aware extraction and bounds calculation
    uint64_t calculatePeakMemoryWithLiveness(
        const std::vector<uint32_t> &topo,
        const Graph &graph,
        const std::unordered_map<uint32_t, uint64_t> &nodeMemorySizes,
        const std::unordered_map<uint32_t, uint32_t> &refCounts,
        const std::unordered_set<uint32_t> &cachedNodes) const
    {
        uint64_t currentMem = 0;
        uint64_t peakMem = 0;

        // Track remaining uses for each node
        std::unordered_map<uint32_t, uint32_t> uses = refCounts;

        for (uint32_t nodeId : topo)
        {
            // Allocate output (if not cached, cached nodes already in memory)
            if (cachedNodes.find(nodeId) == cachedNodes.end())
            {
                auto it = nodeMemorySizes.find(nodeId);
                if (it != nodeMemorySizes.end())
                {
                    currentMem += it->second;
                }
            }

            // Parents might die after this op
            const TensorNode &node = graph.getNode(nodeId);
            for (uint32_t parentId : node.parentIds)
            {
                auto useIt = uses.find(parentId);
                if (useIt != uses.end())
                {
                    useIt->second--;
                    if (useIt->second == 0 && cachedNodes.find(parentId) == cachedNodes.end())
                    {
                        auto sizeIt = nodeMemorySizes.find(parentId);
                        if (sizeIt != nodeMemorySizes.end())
                        {
                            currentMem -= sizeIt->second;
                        }
                    }
                }
            }

            peakMem = std::max(peakMem, currentMem);
        }

        return peakMem;
    }

    ExtractionResult extractBest(const uint32_t rootId, const Graph &graph, EGraph &egraph,
                                 const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                                 const std::unordered_map<uint32_t, uint32_t> &refCounts,
                                 const std::unordered_map<Backend, uint64_t> &maxMemoryByBackend,
                                 const std::unordered_set<uint32_t> &cachedNodes)
    {
        ExtractionResult result;

        std::unordered_map<uint32_t, ExtractChoice> &choice = result.choiceByEClass;

        std::function<ExtractChoice(uint32_t)> solve = [&](uint32_t eclassId) -> ExtractChoice
        {
            eclassId = egraph.find(eclassId);
            auto it = choice.find(eclassId);
            if (it != choice.end())
                return it->second;

            const EClass &cls = egraph.getEClass(eclassId);
            ExtractChoice best;

            for (uint32_t enodeId : cls.enodes)
            {
                const ENode &enode = egraph.getENodes()[enodeId]; // TODO: make EGraph::getENode(uint32_t enodeId) function
                if (enode.opType == OpType::INPUT)
                {
                    ExtractChoice c;
                    c.enodeId = enodeId;
                    c.cost = 0.0f;
                    c.valid = true;
                    TensorNode inNode = graph.getNode(enode.nodeId);
                    ensureContiguousView(inNode);
                    c.view = inNode.view;
                    c.memSize = getSizeBytes(inNode.shape, inNode.dtype);
                    if (c.cost < best.cost)
                        best = c;
                    continue;
                }

                float childrenCost = 0.0f;
                std::vector<TensorNode> inputs;
                inputs.reserve(enode.children.size());
                bool childValid = true;
                for (uint32_t childEClass : enode.children)
                {
                    ExtractChoice childChoice = solve(childEClass);
                    if (!childChoice.valid)
                    {
                        childValid = false;
                        break;
                    }
                    childrenCost += childChoice.cost;
                    const ENode &childEnode = egraph.getENodes()[childChoice.enodeId];
                    TensorNode inNode = graph.getNode(childEnode.nodeId);
                    inNode.backend = childEnode.backend;
                    if (childChoice.view.shape.size() > 0)
                    {
                        inNode.view = childChoice.view;
                    }
                    else
                    {
                        ensureContiguousView(inNode);
                    }
                    inputs.push_back(inNode);
                }
                if (!childValid)
                {
                    continue;
                }

                if (enode.opType != OpType::COPY_TO)
                {
                    bool backendMatch = true;
                    const KernelEntry *entry = nullptr;
                    if (enode.kernelUid != 0)
                    {
                        entry = &KernelRegistry::get().getKernel(enode.kernelUid);
                    }

                    for (size_t i = 0; i < inputs.size(); ++i)
                    {
                        Backend expectedBack = enode.backend;
                        if (entry && i < entry->inputBackends.size())
                        {
                            expectedBack = entry->inputBackends[i];
                        }

                        if (inputs[i].backend != expectedBack)
                        {
                            backendMatch = false;
                            break;
                        }
                    }
                    if (!backendMatch)
                    {
                        continue;
                    }
                }

                TensorNode outNode = graph.getNode(enode.nodeId);
                outNode.backend = enode.backend;
                ensureContiguousView(outNode);

                if (enode.kernelUid == 0)
                    continue;

                const KernelEntry &entry = KernelRegistry::get().getKernel(enode.kernelUid);

                if (entry.inplace)
                {
                    if (entry.inferView)
                    {
                        outNode.view = entry.inferView(outNode, inputs);
                    }
                    else if (!inputs.empty())
                    {
                        outNode.view = inputs[0].view;
                    }
                }

                if (!entry.match(inputs, outNode, refCounts))
                {
                    continue;
                }

                float kernelCost = costModel.estimateCost(outNode, inputs, graph, enode.kernelUid);

                ExtractChoice c;
                c.enodeId = enodeId;
                c.cost = childrenCost + kernelCost;
                c.valid = true;
                c.view = outNode.view;
                c.memSize = getSizeBytes(outNode.shape, outNode.dtype);
                if (!best.valid || c.cost < best.cost)
                    best = c;
            }

            if (!best.valid)
            {
                best.cost = std::numeric_limits<float>::infinity();
            }
            choice[eclassId] = best;
            return best;
        };

        auto it = nodeToEClass.find(rootId);
        if (it == nodeToEClass.end())
        {
            Error::throw_err("Root node missing from egraph.");
        }
        solve(it->second);

        for (const auto &kv : choice)
        {
            const ExtractChoice &c = kv.second;
            if (c.valid && c.enodeId < egraph.getENodes().size())
            {
                result.eclassToNodeId[kv.first] = egraph.getENodes()[c.enodeId].nodeId;
            }
        }

        // Memory validation: calculate peak memory with liveness
        if (!cachedNodes.empty() || !maxMemoryByBackend.empty())
        {
            // Build node memory sizes map from extraction choices
            std::unordered_map<uint32_t, uint64_t> nodeMemorySizes;
            std::unordered_map<Backend, uint64_t> peakMemByBackend;
            std::unordered_map<Backend, uint64_t> currentMemByBackend;
            std::unordered_map<Backend, uint64_t> cachedMemByBackend;
            for (const auto &kv : choice)
            {
                const ExtractChoice &c = kv.second;
                if (c.valid && c.enodeId < egraph.getENodes().size())
                {
                    uint32_t nodeId = egraph.getENodes()[c.enodeId].nodeId;
                    nodeMemorySizes[nodeId] = c.memSize;
                }
            }

            // Get topological order
            std::vector<uint32_t> topo;
            std::unordered_set<uint32_t> visited;
            std::function<void(uint32_t)> visit = [&](uint32_t nid)
            {
                if (visited.count(nid))
                    return;
                visited.insert(nid);
                const TensorNode &node = graph.getNode(nid);
                for (uint32_t pid : node.parentIds)
                {
                    visit(pid);
                }
                topo.push_back(nid);
            };
            visit(rootId);

            // Calculate backend-local peak memory with liveness.
            std::unordered_map<uint32_t, uint32_t> uses = refCounts;
            for (uint32_t nodeId : topo)
            {
                const TensorNode &node = graph.getNode(nodeId);
                Backend backend = node.backend;
                if (cachedNodes.find(nodeId) == cachedNodes.end())
                {
                    auto sizeIt = nodeMemorySizes.find(nodeId);
                    if (sizeIt != nodeMemorySizes.end())
                    {
                        currentMemByBackend[backend] += sizeIt->second;
                    }
                }

                for (uint32_t parentId : node.parentIds)
                {
                    auto useIt = uses.find(parentId);
                    if (useIt != uses.end())
                    {
                        useIt->second--;
                        if (useIt->second == 0 && cachedNodes.find(parentId) == cachedNodes.end())
                        {
                            auto sizeIt = nodeMemorySizes.find(parentId);
                            if (sizeIt != nodeMemorySizes.end())
                            {
                                peakMemByBackend[graph.getNode(parentId).backend] = std::max(
                                    peakMemByBackend[graph.getNode(parentId).backend],
                                    currentMemByBackend[graph.getNode(parentId).backend]);
                                currentMemByBackend[graph.getNode(parentId).backend] -= sizeIt->second;
                            }
                        }
                    }
                }

                peakMemByBackend[backend] = std::max(peakMemByBackend[backend], currentMemByBackend[backend]);
            }

            for (uint32_t cachedNodeId : cachedNodes)
            {
                auto it = nodeMemorySizes.find(cachedNodeId);
                if (it != nodeMemorySizes.end())
                {
                    cachedMemByBackend[graph.getNode(cachedNodeId).backend] += it->second;
                }
            }

            std::unordered_set<Backend> backends;
            for (const auto &kv : peakMemByBackend)
                backends.insert(kv.first);
            for (const auto &kv : cachedMemByBackend)
                backends.insert(kv.first);
            for (const auto &kv : maxMemoryByBackend)
                backends.insert(kv.first);

            for (Backend backend : backends)
            {
                uint64_t limit = getMemoryLimit(backend);
                uint64_t required = peakMemByBackend[backend] + cachedMemByBackend[backend];
                if (required > limit)
                {
                    throw MemoryExhaustedError(required, limit);
                }
            }
        }

        return result;
    }

    CompiledGraph buildCompiledGraph(
        uint32_t rootId,
        Graph &graph,
        EGraph &egraph,
        const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
        const std::unordered_map<uint32_t, uint32_t> &refCounts,
        const ExtractionResult &extraction,
        const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
        const std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &dirtyInputRegions,
        const std::unordered_map<uint32_t, uint32_t> &logicalToPhysicalNodeMap,
        const std::unordered_map<uint32_t, uint32_t> &physicalToLogicalNodeMap,
        const std::unordered_map<uint32_t, std::vector<Region>> &physicalOutputRegions,
        const std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &physicalInputSlices,
        const std::unordered_set<uint32_t> &cachedNodes)
    {
        CompiledGraph compiled;
        auto mapToSelected = [&](uint32_t nodeId)
        {
            auto it = nodeToEClass.find(nodeId);
            if (it == nodeToEClass.end())
            {
                Error::throw_err("[Planner.buildCompiledGraph] nodeToEClass does not have mapping for nodeId " + std::to_string(nodeId));
            }

            uint32_t eClassId = egraph.find(it->second);
            auto sit = extraction.eclassToNodeId.find(eClassId);
            if (sit != extraction.eclassToNodeId.end())
                return sit->second;
            Error::throw_err("[Planner.buildCompiledGraph] extraction.eclassToNodeId does not have mapping for eClassId " + std::to_string(eClassId));
        };

        std::vector<uint32_t> topo;
        std::unordered_set<uint32_t> visited;
        std::function<void(uint32_t)> visit = [&](uint32_t nodeId)
        {
            if (visited.count(nodeId))
                return;
            visited.insert(nodeId);
            for (uint32_t pid : graph.getNode(nodeId).parentIds)
            {
                visit(mapToSelected(pid));
            }
            topo.push_back(nodeId);
        };
        visit(rootId);

        std::unordered_map<uint32_t, uint32_t> compiledRefCounts;
        for (uint32_t nodeId : topo)
        {
            for (uint32_t pid : graph.getNode(nodeId).parentIds)
            {
                compiledRefCounts[mapToSelected(pid)]++;
            }
        }
        compiledRefCounts[rootId] = std::max<uint32_t>(1, compiledRefCounts[rootId]);

        compiled.refCounts = compiledRefCounts;

        for (uint32_t nodeId : topo)
        {
            compiled.nodesMap[nodeId] = graph.getNode(nodeId);
            if (graph.constantStaging.count(nodeId))
                compiled.constantStaging[nodeId] = graph.constantStaging.at(nodeId);
        }

        for (uint32_t nodeId : topo)
        {
            const TensorNode &node = graph.getNode(nodeId);
            if (node.opType == OpType::INPUT)
                continue;

            uint32_t logicalNodeId = nodeId;
            auto logicalIt = physicalToLogicalNodeMap.find(nodeId);
            if (logicalIt != physicalToLogicalNodeMap.end())
            {
                logicalNodeId = logicalIt->second;
            }

            uint32_t eclassId = nodeToEClass.at(nodeId);
            auto choiceIt = extraction.choiceByEClass.find(egraph.find(eclassId));
            if (choiceIt == extraction.choiceByEClass.end() || !choiceIt->second.valid)
            {
                Error::throw_err("Missing or invalid extraction choice for node " + std::to_string(nodeId));
            }

            const ExtractChoice &choice = choiceIt->second;
            const ENode &enode = egraph.getENodes()[choice.enodeId];

            OpInstruction inst;
            inst.nodeId = nodeId;
            inst.logicalNodeId = logicalNodeId;
            inst.inputNodeIds.reserve(node.parentIds.size());
            for (uint32_t pid : node.parentIds)
                inst.inputNodeIds.push_back(mapToSelected(pid));
            inst.backend = enode.backend;

            std::vector<TensorNode> checkInputs;
            checkInputs.reserve(inst.inputNodeIds.size());
            for (uint32_t pid : inst.inputNodeIds)
                checkInputs.push_back(compiled.nodesMap.at(pid));

            TensorNode checkOutNode = compiled.nodesMap.at(nodeId);
            checkOutNode.backend = enode.backend;
            ensureContiguousView(checkOutNode);

            const KernelEntry *kEntry = &KernelRegistry::get().getKernel(enode.kernelUid);
            if (kEntry->inplace && !kEntry->match(checkInputs, checkOutNode, compiledRefCounts))
            {
                auto altRefs = KernelRegistry::get().findMatchingKernels(
                    node.opType, node.opName, enode.backend, checkInputs, checkOutNode, compiledRefCounts, false);
                if (altRefs.empty())
                {
                    Error::throw_err("[Planner.buildCompiledGraph] No fallback kernel found for rejected inplace kernel");
                }
                kEntry = &KernelRegistry::get().getKernel(altRefs.front());
            }

            inst.fullKernelId = kEntry->uid;
            inst.inplaceInputIndex = kEntry->inplace ? 0 : -1;
            const bool nodeIsFinalLogicalOutput = logicalToPhysicalNodeMap.count(logicalNodeId) && logicalToPhysicalNodeMap.at(logicalNodeId) == nodeId;
            inst.outputStorageType = (cachedNodes.count(logicalNodeId) && nodeIsFinalLogicalOutput) ? StorageType::PINNED : node.storageType;

            compiled.nodesMap[nodeId].backend = enode.backend;
            if (kEntry->inplace)
            {
                std::vector<TensorNode> inplaceInputs;
                inplaceInputs.reserve(inst.inputNodeIds.size());
                for (uint32_t pid : inst.inputNodeIds)
                    inplaceInputs.push_back(compiled.nodesMap.at(pid));

                if (kEntry->inferView)
                {
                    compiled.nodesMap[nodeId].view = kEntry->inferView(compiled.nodesMap[nodeId], inplaceInputs);
                }
                else if (!inplaceInputs.empty())
                {
                    compiled.nodesMap[nodeId].view = inplaceInputs[static_cast<size_t>(inst.inplaceInputIndex)].view;
                }
            }

            compiled.instructions.push_back(inst);

            std::vector<TensorNode> costInputs;
            costInputs.reserve(inst.inputNodeIds.size());
            for (uint32_t pid : inst.inputNodeIds)
            {
                costInputs.push_back(compiled.nodesMap.at(pid));
            }
            TensorNode costOut = compiled.nodesMap.at(nodeId);
            compiled.nodeCosts[nodeId] = costModel.estimateCost(costOut, costInputs, graph, enode.kernelUid);

            compiled.physicalToLogicalNodeMap[nodeId] = logicalNodeId;
        }

        return compiled;
    }
};
