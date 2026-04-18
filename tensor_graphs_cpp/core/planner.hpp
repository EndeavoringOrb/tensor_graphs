#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/cost_model.hpp"
#include "core/kernels.hpp"
#include "core/rewrite.hpp"
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
    bool initialized = false;
    std::unordered_map<uint32_t, std::vector<Region>> recompute;
    std::unordered_map<uint32_t, std::vector<Region>> recomputeCached;
    std::unordered_map<uint32_t, std::vector<Region>> needed;

    const std::vector<Region> *getRecompute(
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        uint32_t nodeId) const
    {
        if (cachedNodes.count(nodeId) != 0)
        {
            auto it = recomputeCached.find(nodeId);
            if (it != recomputeCached.end())
            {
                return &it->second;
            }
        }

        auto it = recompute.find(nodeId);
        if (it != recompute.end())
        {
            return &it->second;
        }

        return nullptr;
    }
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

        // These operations rely on global coordinates or full context to compute correct values, so they cannot be partially evaluated.
        bool forceFull = (node.opType == OpType::ARANGE ||
                          node.opType == OpType::TRIU ||
                          node.opType == OpType::IM2COL ||
                          node.opType == OpType::RESHAPE ||
                          node.opType == OpType::PERMUTE ||
                          node.opType == OpType::REPEAT ||
                          node.opType == OpType::FILL);

        std::vector<Region> currentNeeded = neededIt->second;
        if (forceFull)
        {
            currentNeeded = {node.fullRegion()};
            needed[nodeId] = currentNeeded;
        }

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
    const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions)
{
    PlanningRegionState state;
    ShapePropagator prop;

    auto rootIt = dirtyOutputRegions.find(rootId);
    if (rootIt == dirtyOutputRegions.end())
    {
        Error::throw_err("[derivePlanningRegions] Cannot find dirty region for output");
    }
    state.needed[rootId] = mergeRegions(rootIt->second);

    updateNeeded(rootId, graph, prop, state.needed);

    for (const auto &pair : state.needed)
    {
        uint32_t nodeId = pair.first;
        const std::vector<Region> &neededRegions = pair.second;
        if (neededRegions.empty())
            continue;
        const TensorNode &node = graph.nodes.at(nodeId);
        bool forceFull = (node.opType == OpType::ARANGE ||
                          node.opType == OpType::TRIU ||
                          node.opType == OpType::IM2COL ||
                          node.opType == OpType::RESHAPE ||
                          node.opType == OpType::PERMUTE ||
                          node.opType == OpType::REPEAT ||
                          node.opType == OpType::FILL);

        auto dirtyIt = dirtyOutputRegions.find(nodeId);
        state.recompute[nodeId] = normalizeRegions(mergeRegions(neededRegions));
        if (dirtyIt != dirtyOutputRegions.end() && !dirtyIt->second.empty() && !forceFull)
            state.recomputeCached[nodeId] = normalizeRegions(intersectRegionLists(dirtyIt->second, neededRegions));
    }

    state.initialized = true;
    return state;
}

struct CacheAwarePlanningGraph
{
    Graph graph;
    uint32_t physicalRootId = 0;
    std::unordered_map<uint32_t, uint32_t> logicalToPhysicalNodeMap;
    std::unordered_map<uint32_t, uint32_t> physicalToLogicalNodeMap;
    std::unordered_map<uint32_t, std::vector<std::vector<Region>>> physicalInputSlices;
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
        const std::unordered_map<uint32_t, Backend> &cachedNodes)
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
    const std::unordered_map<uint32_t, Backend> &cachedNodes;
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
        placeholder.storageType = StorageType::PINNED;
        if (cachedNodes.count(logicalId))
        {
            placeholder.backend = cachedNodes.at(logicalId);
        }
        placeholder.parentIds.clear();
        result.physicalToLogicalNodeMap[logicalId] = logicalId;
    }

    uint32_t cloneNode(const TensorNode &sourceNode, const std::vector<uint32_t> &parentIds)
    {
        TensorNode &cloned = result.graph.allocateNode(sourceNode.opType, sourceNode.opName, sourceNode.dtype, parentIds, sourceNode.getShape(), sourceNode.strides, sourceNode.backend, sourceNode.storageType, sourceNode.contentHash);
        return cloned.id;
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
        if (parentRegions.size() != 1) // TODO: handle multiple parent regions, or change std::vector<Region> -> Region if none of the backwards give multiple regions
            Error::throw_err("[CacheAwarePlanningGraphBuilder.buildParentInputSlice] expected parentRegions.size() == 1 but got " + std::to_string(parentRegions.size()));

        const Region &inputRegion = parentRegions.front();
        if (inputRegion.empty())
            Error::throw_err("[CacheAwarePlanningGraphBuilder.buildParentInputSlice] parent input region is empty");

        const TensorNode &sourceParent = sourceGraph.getNode(sourceNode.parentIds[parentIdx]);
        uint32_t safeParentId = result.graph.contiguous(physicalParentId);

        uint32_t startsId = addRegionStarts(inputRegion);
        uint32_t endsId = addRegionEnds(inputRegion);
        uint32_t stepsId = addRegionSteps(inputRegion);
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
            const std::vector<Region> &regionsForParent = parentRegions[parentIdx];
            partialParents.push_back(buildParentInputSlice(sourceNode, parentIdx, physicalParentId, regionsForParent));
        }

        uint32_t partialId = cloneNode(sourceNode, partialParents);
        TensorNode &partialNode = result.graph.getNode(partialId);
        partialNode.setShape(getRegionShape(outRegion));
        result.physicalToLogicalNodeMap[partialId] = logicalId;
        result.physicalInputSlices[partialId] = parentRegions;
        return {partialId, parentRegions};
    }

    uint32_t buildLogicalNode(uint32_t logicalId)
    {
        auto memoIt = memoizedPhysicalIds.find(logicalId);
        if (memoIt != memoizedPhysicalIds.end())
            return memoIt->second;

        if (!sourceGraph.hasNode(logicalId))
        {
            Error::throw_err("[CacheAwarePlanningGraphBuilder.buildLogicalNode] sourceGraph doesn't have node for logicalId");
        }

        const TensorNode &sourceNode = sourceGraph.getNode(logicalId);
        if (sourceNode.opType == OpType::INPUT)
        {
            TensorNode &placeholder = result.graph.getNode(logicalId);
            if (cachedNodes.count(logicalId))
            {
                placeholder.backend = cachedNodes.at(logicalId);
            }
            result.logicalToPhysicalNodeMap[logicalId] = logicalId;
            result.physicalToLogicalNodeMap[logicalId] = logicalId;
            memoizedPhysicalIds[logicalId] = logicalId;
            return logicalId;
        }

        const std::vector<Region> *recomputePtr = regionState.getRecompute(cachedNodes, logicalId);
        if (!recomputePtr || recomputePtr->empty())
        {
            convertLogicalNodeToPlaceholder(logicalId);
            result.logicalToPhysicalNodeMap[logicalId] = logicalId;
            result.physicalToLogicalNodeMap[logicalId] = logicalId;
            memoizedPhysicalIds[logicalId] = logicalId;
            return logicalId;
        }
        const std::vector<Region> &recomputeRegions = *recomputePtr;

        if (cachedNodes.count(logicalId))
        {
            convertLogicalNodeToPlaceholder(logicalId);
            uint32_t currentId = logicalId;

            for (const Region &recomputeRegion : recomputeRegions)
            {
                PartialCloneResult partial = buildPartialClone(logicalId, recomputeRegion);

                uint32_t startsId = addRegionStarts(recomputeRegion);
                uint32_t endsId = addRegionEnds(recomputeRegion);
                uint32_t stepsId = addRegionSteps(recomputeRegion);
                uint32_t scatterId = result.graph.scatter(currentId, partial.nodeId, startsId, endsId, stepsId);

                result.physicalToLogicalNodeMap[scatterId] = logicalId;

                currentId = scatterId;
            }

            for (auto &pair : result.graph.nodes)
            {
                if (pair.second.opType == OpType::SCATTER)
                {
                    continue;
                }
                for (uint32_t &pid : pair.second.parentIds)
                {
                    if (pid == logicalId)
                    {
                        pid = currentId;
                    }
                }
            }

            result.logicalToPhysicalNodeMap[logicalId] = currentId;
            memoizedPhysicalIds[logicalId] = currentId;
            return currentId;
        }
        else
        {
            result.logicalToPhysicalNodeMap[logicalId] = logicalId;
            result.physicalToLogicalNodeMap[logicalId] = logicalId;
            memoizedPhysicalIds[logicalId] = logicalId;
            return logicalId;
        }
    }
};

static CacheAwarePlanningGraph buildCacheAwarePlanningGraph(
    uint32_t rootId,
    const Graph &graph,
    const PlanningRegionState &regionState,
    const std::unordered_map<uint32_t, Backend> &cachedNodes)
{
    CacheAwarePlanningGraphBuilder builder(graph, regionState, cachedNodes);
    return builder.build(rootId);
}

class Planner
{
private:
    struct ENodeInfo
    {
        float cost;
        std::unordered_map<Backend, uint64_t> memSizes;
        bool inplace;
        int32_t inplace_idx;
    };

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

    struct EGraphSetupResult
    {
        CacheAwarePlanningGraph planningGraph;
        EGraph egraph;
        std::unordered_map<uint32_t, uint32_t> nodeToEClass;
        std::unordered_map<uint32_t, uint32_t> eclassToLogical;
        std::unordered_set<uint32_t> immutable_eclasses;
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

    void inferShapes(const std::vector<uint32_t> &topo, Graph &graph)
    {
        ShapePropagator propagator;
        for (uint32_t nodeId : topo)
        {
            propagator.inferShape(nodeId, graph);
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

    void saturate(EGraph &egraph, const std::unordered_set<uint32_t> &protectedEClasses)
    {
        std::vector<std::unique_ptr<Rule>> rules;
        rules.emplace_back(std::make_unique<FusionRule>());
        rules.emplace_back(std::make_unique<CopyToOfContiguous>());
        rules.emplace_back(std::make_unique<ContiguousOfCopyTo>());
        // rules.emplace_back(std::make_unique<DistributiveProperty>());

        size_t iterations = 0;
        bool changed = true;
        uint32_t nMatches = 0;
#ifdef DEBUG
        ProgressTimer timer(0, "saturating ");
#endif
        while (changed)
        {
            iterations++;
            uint32_t numENodes = egraph.getENodes().size();
            for (uint32_t eNodeIdx = 0; eNodeIdx < numENodes; eNodeIdx++)
            {
                for (const auto &rule : rules)
                {
                    if (!rule->match(egraph, eNodeIdx, protectedEClasses))
                        continue;

                    rule->apply(egraph, eNodeIdx, protectedEClasses);
                    changed = true;
                    nMatches++;
                }
            }
            egraph.rebuild();
            changed = egraph.getENodes().size() != numENodes;
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

    std::unordered_map<Backend, uint64_t> computePeakMemory(
        const EGraph &egraph,
        const std::unordered_map<uint32_t, uint32_t> &selection_map,
        const std::vector<ENodeInfo> &enodeInfos,
        uint32_t root,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        const std::unordered_map<uint32_t, uint32_t> &eclassToLogical) const
    {
        auto ref = build_ref_counts(egraph, selection_map, root);
        std::unordered_map<Backend, uint64_t> live_mem;
        std::unordered_map<Backend, uint64_t> peak_mem;
        std::unordered_set<uint32_t> visited;

        std::function<void(uint32_t)> visit = [&](uint32_t eclass)
        {
            if (visited.count(eclass))
                return;
            visited.insert(eclass);

            uint32_t sel = selection_map.at(eclass);
            uint32_t enode_id = egraph.getEClass(eclass).enodes[sel];
            const ENode &node = egraph.getENodes()[enode_id];
            const ENodeInfo &info = enodeInfos[enode_id];

            for (uint32_t c : node.children)
            {
                visit(c);
            }

            uint32_t logicalId = eclassToLogical.count(eclass) ? eclassToLogical.at(eclass) : UINT32_MAX;
            bool isCached = (logicalId != UINT32_MAX && cachedNodes.count(logicalId));

            if (!info.inplace && !isCached)
            {
                for (const auto &kv : info.memSizes)
                {
                    live_mem[kv.first] += kv.second;
                    peak_mem[kv.first] = std::max(peak_mem[kv.first], live_mem[kv.first]);
                }
            }

            for (size_t i = 0; i < node.children.size(); ++i)
            {
                uint32_t c = node.children[i];
                ref[c]--;
                if (ref[c] == 0)
                {
                    uint32_t child_sel = selection_map.at(c);
                    uint32_t child_enode_id = egraph.getEClass(c).enodes[child_sel];
                    const ENodeInfo &child_info = enodeInfos[child_enode_id];
                    uint32_t child_logicalId = eclassToLogical.count(c) ? eclassToLogical.at(c) : UINT32_MAX;
                    bool childIsCached = (child_logicalId != UINT32_MAX && cachedNodes.count(child_logicalId));

                    if (info.inplace && (int)i == info.inplace_idx)
                    {
                        continue;
                    }

                    if (!child_info.inplace && !childIsCached)
                    {
                        for (const auto &kv : child_info.memSizes)
                        {
                            live_mem[kv.first] -= kv.second;
                        }
                    }
                }
            }
        };

        visit(root);
        return peak_mem;
    }

    EGraphSetupResult setupEGraph(
        uint32_t rootId,
        const Graph &graph,
        const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        bool doSaturate,
        PlanningRegionState &regionState)
    {
        if (InterruptManager::isInterrupted())
        {
            std::cerr << "\n[Executor] Interrupt detected, aborting execution..." << std::endl;
            InterruptManager::cleanup();
            std::exit(SIGINT);
        }
        if (!regionState.initialized)
        {
            regionState = derivePlanningRegions(rootId, graph, dirtyOutputRegions);
        }
        EGraphSetupResult result;
        result.planningGraph = buildCacheAwarePlanningGraph(rootId, graph, regionState, cachedNodes);

        std::vector<uint32_t> topo = topologicalSort(result.planningGraph.physicalRootId, result.planningGraph.graph);
        inferShapes(topo, result.planningGraph.graph);

        auto refCounts = computeRefCounts(topo, result.planningGraph.physicalRootId, result.planningGraph.graph);

        result.nodeToEClass.reserve(result.planningGraph.graph.nodes.size());

        for (uint32_t nodeId : topo)
        {
            TensorNode &node = result.planningGraph.graph.getNode(nodeId);
            uint32_t refCount = 0;
            auto rcIt = refCounts.find(nodeId);
            if (rcIt == refCounts.end())
            {
                Error::throw_err("Node " + std::to_string(nodeId) + " not found in refCounts");
            }
            refCount = rcIt->second;
            uint32_t eclassId = result.egraph.addEClass(node.getShape(), node.strides, node.viewOffset, node.dtype, node.backend);
            result.nodeToEClass[nodeId] = eclassId;

            // Map constants directly to EGraph EClass
            if (result.planningGraph.graph.constantStaging.count(nodeId))
            {
                result.egraph.constantStaging[eclassId] = result.planningGraph.graph.constantStaging.at(nodeId);
            }
        }

        // Seed with reference kernels only
        for (uint32_t nodeId : topo)
        {
            const TensorNode &node = result.planningGraph.graph.getNode(nodeId);
            uint32_t eclassId = result.nodeToEClass[nodeId];
            uint32_t refCount = 0;
            auto rcIt = refCounts.find(nodeId);
            if (rcIt == refCounts.end())
            {
                Error::throw_err("Node " + std::to_string(nodeId) + " not found in refCounts");
            }
            refCount = rcIt->second;

            if (node.opType == OpType::INPUT)
            {
                ENode enode;
                enode.kernelUid = 0;
                enode.opType = node.opType;
                enode.opName = node.opName;
                for (uint32_t pid : node.parentIds)
                    enode.children.push_back(result.nodeToEClass[pid]);
                enode.shape = node.getShape();
                enode.strides = node.strides;
                enode.viewOffset = node.viewOffset;
                enode.dtype = node.dtype;
                enode.backend = node.backend;
                result.egraph.addENode(eclassId, enode);

                continue;
            }

            std::vector<TensorNode> inputs;
            for (uint32_t pid : node.parentIds)
                inputs.push_back(result.planningGraph.graph.getNode(pid));

            std::vector<uint64_t> refs = KernelRegistry::get().findMatchingKernels(
                node.opType, node.opName, node.backend, inputs, node, true);
            if (refs.empty())
            {
                Error::throw_err("No reference kernel found for node " + toString(node, result.planningGraph.graph, ""));
            }

            for (uint64_t uid : refs)
            {
                ENode enode;
                enode.kernelUid = uid;
                enode.opType = node.opType;
                enode.opName = node.opName;
                for (uint32_t pid : node.parentIds)
                    enode.children.push_back(result.nodeToEClass[pid]);
                enode.shape = node.getShape();
                enode.strides = node.strides;
                enode.viewOffset = node.viewOffset;
                enode.dtype = node.dtype;
                enode.backend = node.backend;
                result.egraph.addENode(eclassId, enode);
            }
        }

        std::unordered_set<uint32_t> protectedEClasses;
        for (const auto &kv : cachedNodes)
        {
            uint32_t logicalId = kv.first;
            auto it = result.planningGraph.logicalToPhysicalNodeMap.find(logicalId);
            if (it != result.planningGraph.logicalToPhysicalNodeMap.end())
            {
                uint32_t physId = it->second;
                auto it2 = result.nodeToEClass.find(physId);
                if (it2 != result.nodeToEClass.end())
                {
                    protectedEClasses.insert(result.egraph.find(it2->second));
                }
            }
        }

        if (doSaturate)
        {
            saturate(result.egraph, protectedEClasses);
        }

        for (const auto &kv : result.nodeToEClass)
        {
            uint32_t physId = kv.first;
            uint32_t ecl = result.egraph.find(kv.second);
            if (result.planningGraph.physicalToLogicalNodeMap.count(physId))
            {
                result.eclassToLogical[ecl] = result.planningGraph.physicalToLogicalNodeMap.at(physId);
            }
            else
            {
                result.eclassToLogical[ecl] = physId;
            }

            const TensorNode &node = result.planningGraph.graph.getNode(physId);
            if (node.storageType != StorageType::TRANSIENT)
            {
                result.immutable_eclasses.insert(ecl);
            }
            auto logIt = result.planningGraph.physicalToLogicalNodeMap.find(physId);
            if (logIt != result.planningGraph.physicalToLogicalNodeMap.end())
            {
                uint32_t logicalId = logIt->second;
                if (cachedNodes.count(logicalId))
                {
                    result.immutable_eclasses.insert(ecl);
                }
            }
        }

        bool changed = true;
        while (changed)
        {
            changed = false;
            for (size_t i = 0; i < result.egraph.getClasses().size(); ++i)
            {
                uint32_t ecl = result.egraph.find(static_cast<uint32_t>(i));
                if (ecl != i)
                    continue;
                if (result.immutable_eclasses.count(ecl))
                    continue;

                const EClass &cls = result.egraph.getEClass(ecl);
                for (uint32_t enodeId : cls.enodes)
                {
                    const ENode &enode = result.egraph.getENodes()[enodeId];
                    if (enode.kernelUid != 0)
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(enode.kernelUid);
                        if (kernel.isView && enode.children.size() > 0)
                        {
                            uint32_t childEcl = result.egraph.find(enode.children[0]);
                            if (result.immutable_eclasses.count(childEcl))
                            {
                                result.immutable_eclasses.insert(ecl);
                                changed = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    ExtractionResult extractBest(const uint32_t rootId, const Graph &graph, EGraph &egraph,
                                 const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                                 const std::unordered_map<Backend, uint64_t> &maxMemoryByBackend,
                                 const std::unordered_map<uint32_t, Backend> &cachedNodes,
                                 const std::unordered_map<uint32_t, uint32_t> &eclassToLogical,
                                 const std::unordered_set<uint32_t> &immutable_eclasses,
                                 bool stopOnFirstValid = false,
                                 bool cheapInputCopy = false)
    {
        std::vector<ENodeInfo> enodeInfos(egraph.getENodes().size());
        for (size_t i = 0; i < egraph.getENodes().size(); ++i)
        {
            const ENode &enode = egraph.getENodes()[i];
            ENodeInfo info;
            info.memSizes[enode.backend] = getSizeBytes(enode.shape, enode.dtype);
            info.inplace = false;
            info.inplace_idx = -1;

            if (enode.kernelUid != 0)
            {
                const auto &kernel = KernelRegistry::get().getKernel(enode.kernelUid);
                info.inplace = kernel.inplace;
                if (info.inplace && kernel.numInputs > 0)
                {
                    info.inplace_idx = 0;
                }
            }

            if (enode.opType == OpType::INPUT)
            {
                info.cost = 0.0f;
            }
            else if (enode.kernelUid != 0)
            {
                bool isCheapCopy = false;
                if (cheapInputCopy && enode.opType == OpType::COPY_TO && enode.children.size() == 1)
                {
                    uint32_t childEClassId = egraph.find(enode.children[0]);
                    uint32_t logicalId = eclassToLogical.count(childEClassId) ? eclassToLogical.at(childEClassId) : UINT32_MAX;
                    if (logicalId != UINT32_MAX && graph.hasNode(logicalId))
                    {
                        const TensorNode &childLogicalNode = graph.getNode(logicalId);
                        if (childLogicalNode.opType == OpType::INPUT && childLogicalNode.storageType == StorageType::PERSISTENT)
                        {
                            isCheapCopy = true;
                        }
                    }
                }

                if (isCheapCopy)
                {
                    info.cost = 0.0f;
                }
                else
                {
                    std::vector<std::vector<uint32_t>> inShapes;
                    std::vector<std::vector<uint64_t>> inStrides;
                    std::vector<DType> inDTypes;
                    std::vector<std::vector<uint8_t>> inConstants;

                    for (uint32_t childEClassId : enode.children)
                    {
                        const EClass &childCls = egraph.getEClass(childEClassId);
                        inShapes.push_back(childCls.shape);

                        std::vector<uint64_t> strides_cast;
                        for (uint64_t s : childCls.strides)
                            strides_cast.push_back(s);
                        inStrides.push_back(strides_cast);

                        inDTypes.push_back(childCls.dtype);

                        if (egraph.constantStaging.count(childEClassId))
                        {
                            inConstants.push_back(egraph.constantStaging.at(childEClassId));
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
                }

                if (info.inplace && info.inplace_idx >= 0)
                {
                    uint32_t mutated_eclass = egraph.find(enode.children[info.inplace_idx]);
                    if (immutable_eclasses.count(mutated_eclass) && enode.opType != OpType::SCATTER)
                    {
                        info.cost = std::numeric_limits<float>::infinity();
                    }
                }
            }
            else
            {
                info.cost = std::numeric_limits<float>::infinity();
            }
            enodeInfos[i] = info;
        }

        // 1. Filter out infinite cost nodes early
        bool droppedInf = false;
        for (size_t i = 0; i < egraph.getClasses().size(); ++i)
        {
            uint32_t eclassId = egraph.find(static_cast<uint32_t>(i));
            if (eclassId != i)
                continue;

            EClass &cls = egraph.getEClass(eclassId);
            std::vector<uint32_t> validEnodes;
            for (uint32_t enodeId : cls.enodes)
            {
                if (enodeInfos[enodeId].cost == std::numeric_limits<float>::infinity())
                {
                    droppedInf = true;
                }
                else
                {
                    validEnodes.push_back(enodeId);
                }
            }
            cls.enodes = validEnodes;
        }

        if (droppedInf)
        {
            std::cout << "[Planner.extractBest] Warning: Filtered out nodes with infinite cost. "
                      << "You may need to run 'bench' to gather missing kernel performance data." << std::endl;
        }

        // 2. Compute Optimistic Subtree Costs via Dynamic Programming (Worklist Algorithm)
        size_t numClasses = egraph.getClasses().size();
        std::vector<float> eclassMinCost(numClasses, std::numeric_limits<float>::infinity());

        std::vector<std::vector<uint32_t>> dependent_classes(numClasses);
        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t eclassId = egraph.find(static_cast<uint32_t>(i));
            if (eclassId != i)
                continue;

            for (uint32_t enodeId : egraph.getEClass(eclassId).enodes)
            {
                const ENode &enode = egraph.getENodes()[enodeId];
                for (uint32_t child : enode.children)
                {
                    uint32_t childEClass = egraph.find(child);
                    dependent_classes[childEClass].push_back(eclassId);
                }
            }
        }

        for (auto &deps : dependent_classes)
        {
            std::sort(deps.begin(), deps.end());
            deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
        }

        std::vector<uint32_t> worklist;
        std::vector<bool> in_worklist(numClasses, false);
        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t eclassId = egraph.find(static_cast<uint32_t>(i));
            if (eclassId == i)
            {
                worklist.push_back(eclassId);
                in_worklist[eclassId] = true;
            }
        }

        while (!worklist.empty())
        {
            uint32_t c = worklist.back();
            worklist.pop_back();
            in_worklist[c] = false;

            float old_cost = eclassMinCost[c];
            float new_cost = old_cost;

            const EClass &cls = egraph.getEClass(c);
            for (uint32_t enodeId : cls.enodes)
            {
                float cost = enodeInfos[enodeId].cost;
                if (cost == std::numeric_limits<float>::infinity())
                    continue;

                const ENode &enode = egraph.getENodes()[enodeId];
                for (uint32_t child : enode.children)
                {
                    cost += eclassMinCost[egraph.find(child)];
                }

                if (cost < new_cost)
                {
                    new_cost = cost;
                }
            }

            if (new_cost < old_cost && (old_cost == std::numeric_limits<float>::infinity() || (old_cost - new_cost) > 1e-6f))
            {
                eclassMinCost[c] = new_cost;
                for (uint32_t dep : dependent_classes[c])
                {
                    if (!in_worklist[dep])
                    {
                        worklist.push_back(dep);
                        in_worklist[dep] = true;
                    }
                }
            }
        }

        // 3. Sort enodes in each eclass by optimistic subtree cost
        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t eclassId = egraph.find(static_cast<uint32_t>(i));
            if (eclassId != i)
                continue;

            EClass &cls = egraph.getEClass(eclassId);
            std::sort(cls.enodes.begin(), cls.enodes.end(),
                      [&](uint32_t a, uint32_t b)
                      {
                          const ENode &enodeA = egraph.getENodes()[a];
                          float costA = enodeInfos[a].cost;
                          for (uint32_t c : enodeA.children)
                              costA += eclassMinCost[egraph.find(c)];

                          const ENode &enodeB = egraph.getENodes()[b];
                          float costB = enodeInfos[b].cost;
                          for (uint32_t c : enodeB.children)
                              costB += eclassMinCost[egraph.find(c)];

                          if (costA != costB)
                              return costA < costB;
                          return a < b; // stable tie-break
                      });
        }

        auto rootIt = nodeToEClass.find(rootId);
        if (rootIt == nodeToEClass.end())
        {
            Error::throw_err("[Planner.extractBest] Root node missing from nodeToEClass.");
        }
        uint32_t rootEClassId = egraph.find(rootIt->second);

        std::unordered_map<uint32_t, uint32_t> selection_map;
        std::vector<uint32_t> path;
        std::vector<uint32_t> to_process = {rootEClassId};
        std::vector<uint32_t> to_process_enode;
        std::unordered_map<uint32_t, uint32_t> local_ref_counts;
        std::unordered_map<uint32_t, uint32_t> need_single_ref;

        float best_cost = std::numeric_limits<float>::infinity();
        std::unordered_map<uint32_t, uint32_t> best_selection_map;

        int max_iters = 100;
        ProgressTimer timer(max_iters, "extracting graphs ", stopOnFirstValid);
        while (max_iters-- > 0)
        {
            timer.tick();
            bool valid = true;
            std::string reason = "";
            float current_cost = 0.0f;

            for (auto const &kv : selection_map)
            {
                current_cost += enodeInfos[egraph.getEClass(kv.first).enodes[kv.second]].cost;
            }

            while (!to_process.empty())
            {
                uint32_t current = to_process.front();
                to_process.erase(to_process.begin());
                path.push_back(current);

                uint32_t sel = 0;
                auto it = selection_map.find(current);
                if (it != selection_map.end())
                    sel = it->second;

                const auto &enodes = egraph.getEClass(current).enodes;
                if (sel >= enodes.size())
                {
                    Error::throw_err("Invalid selection index in EGraph");
                }

                uint32_t enode_id = enodes[sel];
                const ENode &node = egraph.getENodes()[enode_id];
                const ENodeInfo &info = enodeInfos[enode_id];

                selection_map[current] = sel;
                current_cost += info.cost;

                if (enodes.size() > sel + 1)
                {
                    if (std::find(to_process_enode.begin(), to_process_enode.end(), current) == to_process_enode.end())
                    {
                        to_process_enode.push_back(current);
                    }
                }

                if (info.inplace)
                    need_single_ref[node.children[info.inplace_idx]]++;

                for (uint32_t child : node.children)
                    local_ref_counts[child]++;

                if (info.cost == std::numeric_limits<float>::infinity())
                {
                    valid = false;
                    reason = "cost=inf";
                    break;
                }

                if (info.inplace)
                {
                    uint32_t inplace_child = node.children[info.inplace_idx];
                    if (local_ref_counts[inplace_child] > 1 || (immutable_eclasses.count(inplace_child) && node.opType != OpType::SCATTER))
                    {
                        valid = false;
                        reason = "inplace";
                        break;
                    }
                }

                for (uint32_t child : node.children)
                {
                    if (need_single_ref.count(child) && need_single_ref[child] > 0 && local_ref_counts[child] > 1)
                    {
                        valid = false;
                        reason = "inplace_ref";
                        break;
                    }
                }
                if (!valid)
                    break;

                if (best_cost != std::numeric_limits<float>::infinity() && current_cost >= best_cost)
                {
                    valid = false;
                    reason = "cost=" + std::to_string(current_cost);
                    break;
                }

                std::vector<uint32_t> new_to_process;
                for (uint32_t child : node.children)
                {
                    if (selection_map.find(child) == selection_map.end())
                    {
                        new_to_process.push_back(child);
                    }
                }
                to_process.insert(to_process.begin(), new_to_process.begin(), new_to_process.end());
            }

            if (valid)
            {
                std::unordered_map<Backend, uint64_t> peak = computePeakMemory(egraph, selection_map, enodeInfos, rootEClassId, cachedNodes, eclassToLogical);
                for (const auto &kv : maxMemoryByBackend)
                {
                    if (peak[kv.first] > kv.second)
                    {
                        valid = false;
                        reason = "OOM";
                        break;
                    }
                }
            }

            if (valid)
            {
                if (current_cost < best_cost)
                {
                    best_cost = current_cost;
                    best_selection_map = selection_map;
                    if (!stopOnFirstValid)
                    {
                        std::cout << "new best cost: " << std::to_string(best_cost) << std::endl;
                    }
                }
                if (stopOnFirstValid)
                {
                    break;
                }
            }
            else
            {
#ifdef DEBUG
                std::cout << "invalid graph (" << reason << ")" << std::endl;
#endif
            }

            if (to_process_enode.empty())
                break;

            // backtrack
            while (!path.empty())
            {
                uint32_t current = path.back();
                path.pop_back();

                if (selection_map.find(current) == selection_map.end())
                    continue;
                uint32_t sel = selection_map[current];
                const auto &enodes = egraph.getEClass(current).enodes;
                uint32_t enode_id = enodes[sel];
                const ENode &node = egraph.getENodes()[enode_id];
                const ENodeInfo &info = enodeInfos[enode_id];

                for (uint32_t child : node.children)
                {
                    local_ref_counts[child]--;
                    if (local_ref_counts[child] == 0)
                        local_ref_counts.erase(child);
                }

                if (info.inplace)
                {
                    need_single_ref[node.children[info.inplace_idx]]--;
                    if (need_single_ref[node.children[info.inplace_idx]] == 0)
                        need_single_ref.erase(node.children[info.inplace_idx]);
                }

                if (sel + 1 < enodes.size())
                {
                    selection_map[current] = sel + 1;

                    std::vector<uint32_t> keys_to_delete;
                    for (const auto &kv : selection_map)
                    {
                        if (std::find(path.begin(), path.end(), kv.first) == path.end() && kv.first != current)
                        {
                            keys_to_delete.push_back(kv.first);
                        }
                    }
                    for (uint32_t k : keys_to_delete)
                        selection_map.erase(k);

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
                        for (uint32_t child : n.children)
                        {
                            if (selection_map.find(child) == selection_map.end())
                            {
                                new_to_process.push_back(child);
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

        if (best_cost == std::numeric_limits<float>::infinity())
        {
            Error::throw_err("[Planner.extractBest] no valid extraction found under given constraints. try running bench");
        }

        ExtractionResult result;
        result.totalCost = best_cost;
        if (stopOnFirstValid)
        {
            return result; // return early to speed up
        }
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
        Graph &graph,
        EGraph &egraph,
        const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
        const ExtractionResult &extraction,
        const std::unordered_map<uint32_t, uint32_t> &logicalToPhysicalNodeMap,
        const std::unordered_map<uint32_t, uint32_t> &physicalToLogicalNodeMap,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        const std::unordered_map<uint32_t, uint32_t> &eclassToLogical)
    {
        CompiledGraph compiled;

        std::vector<uint32_t> topo;
        std::unordered_set<uint32_t> visited_classes;
        std::function<void(uint32_t)> visit = [&](uint32_t eclassId)
        {
            eclassId = egraph.find(eclassId);
            if (visited_classes.count(eclassId))
                return;
            visited_classes.insert(eclassId);

            auto choiceIt = extraction.choiceByEClass.find(eclassId);
            if (choiceIt == extraction.choiceByEClass.end() || !choiceIt->second.valid)
                return;

            const ENode &enode = egraph.getENodes()[choiceIt->second.enodeId];
            for (uint32_t child : enode.children)
                visit(child);
            topo.push_back(eclassId);
        };

        uint32_t rootEClassId = egraph.find(nodeToEClass.at(rootId));
        visit(rootEClassId);

        for (uint32_t eclassId : topo)
        {
            const ExtractChoice &choice = extraction.choiceByEClass.at(eclassId);
            const ENode &enode = egraph.getENodes()[choice.enodeId];

            uint32_t logicalId = eclassToLogical.count(eclassId) ? eclassToLogical.at(eclassId) : UINT32_MAX;

            // Offset physical IDs so they never collide with logical IDs from the original Graph
            uint32_t physId = eclassId | 0x80000000;

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
                tNode.parentIds.push_back(egraph.find(c) | 0x80000000);

            tNode.storageType = StorageType::TRANSIENT;
            if (logicalId != UINT32_MAX && graph.hasNode(logicalId))
            {
                tNode.storageType = graph.getNode(logicalId).storageType;
            }

            compiled.nodesMap[physId] = tNode;
            if (egraph.constantStaging.count(eclassId))
            {
                compiled.constantStaging[physId] = egraph.constantStaging.at(eclassId);
            }

            OpInstruction inst;
            inst.nodeId = physId;
            inst.logicalNodeId = logicalId;
            inst.inputNodeIds = tNode.parentIds;
            inst.backend = enode.backend;
            inst.fullKernelId = enode.kernelUid;

            const KernelEntry *kEntry = nullptr;
            if (enode.kernelUid != 0)
            {
                kEntry = &KernelRegistry::get().getKernel(enode.kernelUid);
                inst.inplaceInputIndex = kEntry->inplace ? 0 : -1;
                inst.viewInputIndex = kEntry->isView ? 0 : -1;
            }
            else
            {
                inst.inplaceInputIndex = -1;
                inst.viewInputIndex = -1;
            }

            const bool nodeIsFinalLogicalOutput = logicalToPhysicalNodeMap.count(logicalId) && logicalToPhysicalNodeMap.at(logicalId) == rootId;
            inst.outputStorageType = (cachedNodes.count(logicalId) && (logicalId != UINT32_MAX) && nodeIsFinalLogicalOutput) ? StorageType::PINNED : tNode.storageType;

            if (enode.opType != OpType::INPUT)
            {
                compiled.instructions.push_back(inst);
            }

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
        compiledRefCounts[rootEClassId | 0x80000000] = std::max<uint32_t>(1, compiledRefCounts[rootEClassId | 0x80000000]);
        compiled.refCounts = compiledRefCounts;

        return compiled;
    }

public:
    Planner(CostModel &costModel, std::unordered_map<Backend, uint64_t> maxMemoryByBackend = {})
        : costModel(costModel), maxMemoryByBackend(std::move(maxMemoryByBackend)) {}

    float extractFirstValidCost(const uint32_t rootId, const Graph &graph, EGraph &egraph,
                                const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                                const std::unordered_map<Backend, uint64_t> &maxMemoryByBackend,
                                const std::unordered_map<uint32_t, Backend> &cachedNodes,
                                const std::unordered_map<uint32_t, uint32_t> &eclassToLogical,
                                const std::unordered_set<uint32_t> &immutable_eclasses,
                                bool cheapInputCopy = false)
    {
        return extractBest(rootId, graph, egraph, nodeToEClass, maxMemoryByBackend, cachedNodes, eclassToLogical, immutable_eclasses, true, cheapInputCopy).totalCost;
    }

    float estimateCostForCacheSet(
        uint32_t rootId,
        const Graph &graph,
        const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
        const std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &dirtyInputRegions,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        PlanningRegionState &regionState,
        bool doSaturate = true,
        bool cheapInputCopy = false)
    {
        auto setup = setupEGraph(rootId, graph, dirtyOutputRegions, cachedNodes, doSaturate, regionState);
        return extractFirstValidCost(setup.planningGraph.physicalRootId, graph, setup.egraph, setup.nodeToEClass, maxMemoryByBackend, cachedNodes, setup.eclassToLogical, setup.immutable_eclasses, cheapInputCopy);
    }

    CompiledGraph plan(
        uint32_t rootId,
        Graph &graph,
        const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
        const std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &dirtyInputRegions,
        const std::unordered_map<uint32_t, Backend> &cachedNodes, PlanningRegionState &regionState, bool doSaturate = true, bool cheapInputCopy = false)
    {
        auto setup = setupEGraph(rootId, graph, dirtyOutputRegions, cachedNodes, doSaturate, regionState);
        auto extraction = extractBest(setup.planningGraph.physicalRootId, graph, setup.egraph, setup.nodeToEClass, maxMemoryByBackend, cachedNodes, setup.eclassToLogical, setup.immutable_eclasses, false, cheapInputCopy);

        return buildCompiledGraph(
            setup.planningGraph.physicalRootId,
            setup.planningGraph.graph,
            setup.egraph,
            setup.nodeToEClass,
            extraction,
            setup.planningGraph.logicalToPhysicalNodeMap,
            setup.planningGraph.physicalToLogicalNodeMap,
            cachedNodes,
            setup.eclassToLogical);
    }
};