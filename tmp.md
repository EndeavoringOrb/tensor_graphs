```cpp
    std::vector<std::vector<Region>> backwardDot(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        const auto &sB = graph.nodes[node.parentIds[1]].shape;
====
    std::vector<std::vector<Region>> backwardDot(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        const auto &sA = graph.getNode(node.parentIds[0]).shape;
        const auto &sB = graph.getNode(node.parentIds[1]).shape;
```
```cpp
    std::vector<Region> forwardReshape(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        const auto &outShape = node.shape;
====
    std::vector<Region> forwardReshape(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &sA = graph.getNode(node.parentIds[0]).shape;
        const auto &outShape = node.shape;
```
```cpp
    std::vector<std::vector<Region>> backwardReshape(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};
        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        const auto &sShape = graph.nodes[node.parentIds[1]].shape;
====
    std::vector<std::vector<Region>> backwardReshape(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};
        const auto &sA = graph.getNode(node.parentIds[0]).shape;
        const auto &sShape = graph.getNode(node.parentIds[1]).shape;
```
```cpp
    std::vector<Region> forwardReduce(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;
====
    std::vector<Region> forwardReduce(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &sA = graph.getNode(node.parentIds[0]).shape;
```
```cpp
    std::vector<std::vector<Region>> backwardReduce(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        int32_t axis = getConstantInt32(node.parentIds[1], graph)[0];
====
    std::vector<std::vector<Region>> backwardReduce(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        const auto &sA = graph.getNode(node.parentIds[0]).shape;
        int32_t axis = getConstantInt32(node.parentIds[1], graph)[0];
```
```cpp
        }
        return {mergeRegions(inBoxes), makeFull(graph.nodes[node.parentIds[1]].shape)};
    }
====
        }
        return {mergeRegions(inBoxes), makeFull(graph.getNode(node.parentIds[1]).shape)};
    }
```
```cpp
        for (const auto &outReg : outputRegions)
        {
            Region inBox;
            for (int32_t d : invDims)
            {
                inBox.region.push_back(outReg.region[d]);
            }
            inBoxes.push_back(inBox);
        }
        return {mergeRegions(inBoxes), makeFull(graph.nodes[node.parentIds[1]].shape)};
    }
====
        for (const auto &outReg : outputRegions)
        {
            Region inBox;
            for (int32_t d : invDims)
            {
                inBox.region.push_back(outReg.region[d]);
            }
            inBoxes.push_back(inBox);
        }
        return {mergeRegions(inBoxes), makeFull(graph.getNode(node.parentIds[1]).shape)};
    }
```
```cpp
    std::vector<Region> forwardGather(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &dataReg = parentRegions[0];
        const auto &idxReg = parentRegions[1];
        if (dataReg.empty() && idxReg.empty())
            return {};

        const auto &dataShape = graph.nodes[node.parentIds[0]].shape;
        const auto &idxShape = graph.nodes[node.parentIds[1]].shape;
====
    std::vector<Region> forwardGather(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &dataReg = parentRegions[0];
        const auto &idxReg = parentRegions[1];
        if (dataReg.empty() && idxReg.empty())
            return {};

        const auto &dataShape = graph.getNode(node.parentIds[0]).shape;
        const auto &idxShape = graph.getNode(node.parentIds[1]).shape;
```
```cpp
    std::vector<std::vector<Region>> backwardGather(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        const auto &dataShape = graph.nodes[node.parentIds[0]].shape;
        const auto &idxShape = graph.nodes[node.parentIds[1]].shape;
====
    std::vector<std::vector<Region>> backwardGather(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        const auto &dataShape = graph.getNode(node.parentIds[0]).shape;
        const auto &idxShape = graph.getNode(node.parentIds[1]).shape;
```
```cpp
        uint32_t current_offset = 0;
        for (size_t i = 0; i < node.parentIds.size() - 1; ++i)
        {
            const auto &pShape = graph.nodes[node.parentIds[i]].shape;
            const auto &pReg = parentRegions[i];
====
        uint32_t current_offset = 0;
        for (size_t i = 0; i < node.parentIds.size() - 1; ++i)
        {
            const auto &pShape = graph.getNode(node.parentIds[i]).shape;
            const auto &pReg = parentRegions[i];
```
```cpp
        uint32_t current_offset = 0;
        for (size_t i = 0; i < node.parentIds.size() - 1; ++i)
        {
            const auto &pShape = graph.nodes[node.parentIds[i]].shape;
            uint32_t in_dim = pShape[axis];
====
        uint32_t current_offset = 0;
        for (size_t i = 0; i < node.parentIds.size() - 1; ++i)
        {
            const auto &pShape = graph.getNode(node.parentIds[i]).shape;
            uint32_t in_dim = pShape[axis];
```
```cpp
            res[i] = mergeRegions(inBoxes);
            current_offset = in_end;
        }
        res.back() = makeFull(graph.nodes[node.parentIds.back()].shape);
        return res;
    }
====
            res[i] = mergeRegions(inBoxes);
            current_offset = in_end;
        }
        res.back() = makeFull(graph.getNode(node.parentIds.back()).shape);
        return res;
    }
```
```cpp
    std::vector<Region> forwardRepeat(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;

        int32_t repeats = getConstantInt32(node.parentIds[1], graph)[0];
====
    std::vector<Region> forwardRepeat(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &sA = graph.getNode(node.parentIds[0]).shape;

        int32_t repeats = getConstantInt32(node.parentIds[1], graph)[0];
```
```cpp
    std::vector<std::vector<Region>> backwardRepeat(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        int32_t repeats = getConstantInt32(node.parentIds[1], graph)[0];
====
    std::vector<std::vector<Region>> backwardRepeat(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}};

        const auto &sA = graph.getNode(node.parentIds[0]).shape;
        int32_t repeats = getConstantInt32(node.parentIds[1], graph)[0];
```
```cpp
            inBoxes.push_back(inBox);
        }

        return {mergeRegions(inBoxes), makeFull(graph.nodes[node.parentIds[1]].shape), makeFull(graph.nodes[node.parentIds[2]].shape)};
    }
====
            inBoxes.push_back(inBox);
        }

        return {mergeRegions(inBoxes), makeFull(graph.getNode(node.parentIds[1]).shape), makeFull(graph.getNode(node.parentIds[2]).shape)};
    }
```
```cpp
    std::vector<Region> forwardSlice(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &shape = graph.nodes[node.parentIds[0]].shape;
        auto starts = getConstantInt32(node.parentIds[1], graph);
====
    std::vector<Region> forwardSlice(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &shape = graph.getNode(node.parentIds[0]).shape;
        auto starts = getConstantInt32(node.parentIds[1], graph);
```
```cpp
    std::vector<std::vector<Region>> backwardSlice(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}, {}};

        const auto &shape = graph.nodes[node.parentIds[0]].shape;
        auto starts = getConstantInt32(node.parentIds[1], graph);
====
    std::vector<std::vector<Region>> backwardSlice(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}, {}};

        const auto &shape = graph.getNode(node.parentIds[0]).shape;
        auto starts = getConstantInt32(node.parentIds[1], graph);
```
```cpp
        for (const auto &region : outputRegions)
            inBoxes.push_back(mapSliceRegionBackward(region, shape, starts, ends, steps));

        return {mergeRegions(inBoxes), makeFull(graph.nodes[node.parentIds[1]].shape), makeFull(graph.nodes[node.parentIds[2]].shape), makeFull(graph.nodes[node.parentIds[3]].shape)};
    }
====
        for (const auto &region : outputRegions)
            inBoxes.push_back(mapSliceRegionBackward(region, shape, starts, ends, steps));

        return {mergeRegions(inBoxes), makeFull(graph.getNode(node.parentIds[1]).shape), makeFull(graph.getNode(node.parentIds[2]).shape), makeFull(graph.getNode(node.parentIds[3]).shape)};
    }
```
```cpp
    std::vector<Region> forwardScatter(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &targetRegions = parentRegions[0];
        const auto &updateRegions = parentRegions[1];
        if (targetRegions.empty() && updateRegions.empty())
            return {};

        const auto &targetShape = graph.nodes[node.parentIds[0]].shape;
====
    std::vector<Region> forwardScatter(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &targetRegions = parentRegions[0];
        const auto &updateRegions = parentRegions[1];
        if (targetRegions.empty() && updateRegions.empty())
            return {};

        const auto &targetShape = graph.getNode(node.parentIds[0]).shape;
```
```cpp
    std::vector<std::vector<Region>> backwardScatter(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}, {}, {}};

        const auto &targetShape = graph.nodes[node.parentIds[0]].shape;
====
    std::vector<std::vector<Region>> backwardScatter(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}, {}, {}};

        const auto &targetShape = graph.getNode(node.parentIds[0]).shape;
```
```cpp
        }

        return {mergeRegions(targetBoxes), mergeRegions(updateBoxes), makeFull(graph.nodes[node.parentIds[2]].shape), makeFull(graph.nodes[node.parentIds[3]].shape), makeFull(graph.nodes[node.parentIds[4]].shape)};
    }
====
        }

        return {mergeRegions(targetBoxes), mergeRegions(updateBoxes), makeFull(graph.getNode(node.parentIds[2]).shape), makeFull(graph.getNode(node.parentIds[3]).shape), makeFull(graph.getNode(node.parentIds[4]).shape)};
    }
```
```cpp
    std::vector<Region> forward(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        for (uint32_t pid : node.parentIds)
        {
            if (pid >= graph.nodes.size())
            {
                std::stringstream ss;
                ss << "[ShapePropagator.forward] Invalid parent ID " << pid << " for OpType " << node.opType;
                Error::throw_err(ss.str());
            }
        }
====
    std::vector<Region> forward(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        for (uint32_t pid : node.parentIds)
        {
            if (!graph.hasNode(pid))
            {
                std::stringstream ss;
                ss << "[ShapePropagator.forward] Invalid parent ID " << pid << " for OpType " << node.opType;
                Error::throw_err(ss.str());
            }
        }
```

Update `tensor_graphs_cpp/core/planner.hpp`:
```cpp
void propagateDirtyRegionsAtomic(
    const std::vector<uint32_t> &topo,
    const Graph &graph,
    std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
    std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &dirtyInputRegions)
{
    ShapePropagator propagator;

    for (uint32_t nodeId : topo)
    {
        if (nodeId >= graph.nodes.size())
            continue;
        const TensorNode &node = graph.nodes[nodeId];
====
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
```
```cpp
static void updateNeeded(
    uint32_t rootId,
    const Graph &graph,
    ShapePropagator &prop,
    std::unordered_map<uint32_t, std::vector<Region>> &needed)
{
    if (rootId >= graph.nodes.size())
        return;

    std::vector<uint32_t> worklist = {rootId};
    std::unordered_set<uint32_t> queued = {rootId};

    while (!worklist.empty())
    {
        uint32_t nodeId = worklist.back();
        worklist.pop_back();
        queued.erase(nodeId);

        if (nodeId >= graph.nodes.size())
            continue;

        const TensorNode &node = graph.nodes[nodeId];
        if (node.opType == OpType::INPUT)
            continue;
====
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
```
```cpp
        std::vector<std::vector<Region>> parentNeeded = prop.backward(node, graph, neededIt->second);
        for (size_t i = 0; i < node.parentIds.size(); ++i)
        {
            if (i >= parentNeeded.size() || parentNeeded[i].empty())
                continue;

            uint32_t parentId = node.parentIds[i];
            if (parentId >= graph.nodes.size())
                continue;

            const TensorNode &parentNode = graph.nodes[parentId];
            if (parentNode.opType == OpType::INPUT)
                continue;
====
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
```
```cpp
    void convertLogicalNodeToPlaceholder(uint32_t logicalId)
    {
        TensorNode &placeholder = result.graph.nodes[logicalId];
====
    void convertLogicalNodeToPlaceholder(uint32_t logicalId)
    {
        TensorNode &placeholder = result.graph.getNode(logicalId);
```
```cpp
    void retargetPartialShapeInputs(uint32_t newNodeId, const TensorNode &sourceNode, const Region &outRegion)
    {
        TensorNode &newNode = result.graph.nodes[newNodeId];
====
    void retargetPartialShapeInputs(uint32_t newNodeId, const TensorNode &sourceNode, const Region &outRegion)
    {
        TensorNode &newNode = result.graph.getNode(newNodeId);
```
```cpp
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

        const TensorNode &sourceParent = sourceGraph.nodes[sourceNode.parentIds[parentIdx]];
====
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
```
```cpp
    PartialCloneResult buildPartialClone(uint32_t logicalId, const Region &outRegion)
    {
        const TensorNode &sourceNode = sourceGraph.nodes[logicalId];
        std::vector<std::vector<Region>> parentRegions = prop.backward(sourceNode, sourceGraph, {outRegion});
====
    PartialCloneResult buildPartialClone(uint32_t logicalId, const Region &outRegion)
    {
        const TensorNode &sourceNode = sourceGraph.getNode(logicalId);
        std::vector<std::vector<Region>> parentRegions = prop.backward(sourceNode, sourceGraph, {outRegion});
```
```cpp
        uint32_t partialId = cloneNode(sourceNode, partialParents);
        TensorNode &partialNode = result.graph.nodes[partialId];
        partialNode.shape = getRegionShape(outRegion);
====
        uint32_t partialId = cloneNode(sourceNode, partialParents);
        TensorNode &partialNode = result.graph.getNode(partialId);
        partialNode.shape = getRegionShape(outRegion);
```
```cpp
    uint32_t buildLogicalNode(uint32_t logicalId)
    {
        auto memoIt = memoizedPhysicalIds.find(logicalId);
        if (memoIt != memoizedPhysicalIds.end())
            return memoIt->second;

        if (logicalId >= sourceGraph.nodes.size())
            return logicalId;

        const TensorNode &sourceNode = sourceGraph.nodes[logicalId];
====
    uint32_t buildLogicalNode(uint32_t logicalId)
    {
        auto memoIt = memoizedPhysicalIds.find(logicalId);
        if (memoIt != memoizedPhysicalIds.end())
            return memoIt->second;

        if (!sourceGraph.hasNode(logicalId))
            return logicalId;

        const TensorNode &sourceNode = sourceGraph.getNode(logicalId);
```
```cpp
        for (uint32_t nodeId : topo)
        {
            TensorNode &node = planningGraph.graph.nodes[nodeId];
            ensureContiguousView(node);
====
        for (uint32_t nodeId : topo)
        {
            TensorNode &node = planningGraph.graph.getNode(nodeId);
            ensureContiguousView(node);
```
```cpp
        for (uint32_t nodeId : topo)
        {
            const TensorNode &node = planningGraph.graph.nodes[nodeId];
            uint32_t eclassId = nodeToEClass[nodeId];
            if (node.opType == OpType::INPUT || node.opType == OpType::CONTIGUOUS || node.opType == OpType::SLICE)
            {
                if (node.opType != OpType::INPUT)
                {
                    std::vector<TensorNode> inputs;
                    for (uint32_t pid : node.parentIds)
                        inputs.push_back(planningGraph.graph.nodes[pid]);
====
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
```
```cpp
            std::vector<TensorNode> inputs;
            for (uint32_t pid : node.parentIds)
                inputs.push_back(planningGraph.graph.nodes[pid]);
====
            std::vector<TensorNode> inputs;
            for (uint32_t pid : node.parentIds)
                inputs.push_back(planningGraph.graph.getNode(pid));
```
```cpp
        bool match(uint32_t nodeId, const Graph &graph, RuleMatch &out) const override
        {
            if (nodeId >= graph.nodes.size())
                return false;
            const auto &node = graph.nodes[nodeId];
            if (node.opType != OpType::COPY_TO || node.parentIds.empty())
                return false;
            uint32_t parentId = node.parentIds[0];
            if (parentId >= graph.nodes.size())
                return false;
            const auto &parent = graph.nodes[parentId];
            if (parent.opType != OpType::COPY_TO || parent.parentIds.empty())
                return false;
            out.nodeId = nodeId;
            return true;
        }

        std::vector<uint32_t> apply(const RuleMatch &m, Graph &graph) const override
        {
            const TensorNode &node = graph.nodes[m.nodeId];
            const TensorNode &parent = graph.nodes[node.parentIds[0]];
            uint32_t grandparentId = parent.parentIds[0];

            std::vector<uint32_t> results;
            // E.g. copyto(copyto(X, GPU), CPU) => X
            if (node.backend == graph.nodes[grandparentId].backend)
====
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
```
```cpp
        bool match(uint32_t nodeId, const Graph &graph, RuleMatch &out) const override
        {
            if (nodeId >= graph.nodes.size())
                return false;
            if (graph.nodes[nodeId].opType == OpType::INPUT)
                return false;
            out.nodeId = nodeId;
            return true;
        }

        std::vector<uint32_t> apply(const RuleMatch &m, Graph &graph) const override
        {
            std::vector<uint32_t> results;
            const TensorNode &refNode = graph.nodes[m.nodeId];
====
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
```
```cpp
        uint32_t addFusedNode(Graph &graph, const KernelEntry &kernel, Backend targetBackend, const std::vector<uint32_t> &parentIds, const TensorNode &refNode) const
        {
            std::vector<uint32_t> adaptedParents;
            for (size_t i = 0; i < parentIds.size(); ++i)
            {
                uint32_t pid = parentIds[i];
                const TensorNode &parent = graph.nodes[pid];
====
        uint32_t addFusedNode(Graph &graph, const KernelEntry &kernel, Backend targetBackend, const std::vector<uint32_t> &parentIds, const TensorNode &refNode) const
        {
            std::vector<uint32_t> adaptedParents;
            for (size_t i = 0; i < parentIds.size(); ++i)
            {
                uint32_t pid = parentIds[i];
                const TensorNode &parent = graph.getNode(pid);
```
```cpp
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
                const TensorNode &cNode = mainGraph.nodes[concreteId];
====
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
```
```cpp
                binding[patternId] = concreteId;
                return true;
            }

            const auto &cNode = mainGraph.nodes[concreteId];
            const auto &pNode = patternGraph.nodes[patternId];
====
                binding[patternId] = concreteId;
                return true;
            }

            const auto &cNode = mainGraph.getNode(concreteId);
            const auto &pNode = patternGraph.getNode(patternId);
```
```cpp
                    for (size_t i = oldSize; i < graph.nodes.size(); ++i)
                    {
                        uint32_t newId = static_cast<uint32_t>(i);
                        ShapePropagator prop;
                        prop.inferShapeRecursive(newId, graph);
                        ensureContiguousView(graph.nodes[newId]);

                        if (!nodeToEClass.count(newId))
                        {
                            uint32_t refCount = 0;
                            auto rcIt = refCounts.find(newId);
                            if (rcIt != refCounts.end())
                                refCount = rcIt->second;
                            uint32_t eclassId = egraph.addEClass(graph.nodes[newId].shape, graph.nodes[newId].dtype, refCount, graph.nodes[newId].view.isContiguous());
====
                    // We only want to process new nodes. But nodes are now an unordered_map, so looping by size is broken.
                    // However, we already have `newNodes` list from `rule->apply`
                    for (uint32_t newId : newNodes)
                    {
                        ShapePropagator prop;
                        prop.inferShapeRecursive(newId, graph);
                        ensureContiguousView(graph.getNode(newId));

                        if (!nodeToEClass.count(newId))
                        {
                            uint32_t refCount = 0;
                            auto rcIt = refCounts.find(newId);
                            if (rcIt != refCounts.end())
                                refCount = rcIt->second;
                            uint32_t eclassId = egraph.addEClass(graph.getNode(newId).shape, graph.getNode(newId).dtype, refCount, graph.getNode(newId).view.isContiguous());
```
```cpp
    bool addBasicEnode(const Graph &graph, EGraph &egraph,
                       const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                       const std::unordered_map<uint32_t, uint32_t> &refCounts,
                       uint32_t nodeId)
    {
        const TensorNode &node = graph.nodes[nodeId];
        if (node.opType == OpType::INPUT)
            return false;

        std::vector<TensorNode> inputs;
        for (uint32_t pid : node.parentIds)
            inputs.push_back(graph.nodes[pid]);
====
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
```
```cpp
            // Parents might die after this op
            const TensorNode &node = graph.nodes[nodeId];
            for (uint32_t parentId : node.parentIds)
====
            // Parents might die after this op
            const TensorNode &node = graph.getNode(nodeId);
            for (uint32_t parentId : node.parentIds)
```
```cpp
                        if (useIt->second == 0 && cachedNodes.find(parentId) == cachedNodes.end())
                        {
                            auto sizeIt = nodeMemorySizes.find(parentId);
                            if (sizeIt != nodeMemorySizes.end())
                            {
                                peakMemByBackend[graph.nodes[parentId].backend] = std::max(
                                    peakMemByBackend[graph.nodes[parentId].backend],
                                    currentMemByBackend[graph.nodes[parentId].backend]);
                                currentMemByBackend[graph.nodes[parentId].backend] -= sizeIt->second;
                            }
                        }
====
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
```
```cpp
            for (uint32_t cachedNodeId : cachedNodes)
            {
                auto it = nodeMemorySizes.find(cachedNodeId);
                if (it != nodeMemorySizes.end())
                {
                    cachedMemByBackend[graph.nodes[cachedNodeId].backend] += it->second;
                }
            }
====
            for (uint32_t cachedNodeId : cachedNodes)
            {
                auto it = nodeMemorySizes.find(cachedNodeId);
                if (it != nodeMemorySizes.end())
                {
                    cachedMemByBackend[graph.getNode(cachedNodeId).backend] += it->second;
                }
            }
```
```cpp
                    ExtractChoice c;
                    c.enodeId = enodeId;
                    c.cost = 0.0f;
                    c.valid = true;
                    TensorNode inNode = graph.nodes[enode.nodeId];
====
                    ExtractChoice c;
                    c.enodeId = enodeId;
                    c.cost = 0.0f;
                    c.valid = true;
                    TensorNode inNode = graph.getNode(enode.nodeId);
```
```cpp
                    childrenCost += childChoice.cost;
                    const ENode &childEnode = egraph.getENodes()[childChoice.enodeId];
                    TensorNode inNode = graph.nodes[childEnode.nodeId];
====
                    childrenCost += childChoice.cost;
                    const ENode &childEnode = egraph.getENodes()[childChoice.enodeId];
                    TensorNode inNode = graph.getNode(childEnode.nodeId);
```
```cpp
                }

                TensorNode outNode = graph.nodes[enode.nodeId];
                outNode.backend = enode.backend;
====
                }

                TensorNode outNode = graph.getNode(enode.nodeId);
                outNode.backend = enode.backend;
```
```cpp
            std::function<void(uint32_t)> visit = [&](uint32_t nid)
            {
                if (visited.count(nid))
                    return;
                visited.insert(nid);
                const TensorNode &node = graph.nodes[nid];
                for (uint32_t pid : node.parentIds)
                {
                    visit(pid);
                }
====
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
```
```cpp
            std::unordered_map<uint32_t, uint32_t> uses = refCounts;
            for (uint32_t nodeId : topo)
            {
                const TensorNode &node = graph.nodes[nodeId];
====
            std::unordered_map<uint32_t, uint32_t> uses = refCounts;
            for (uint32_t nodeId : topo)
            {
                const TensorNode &node = graph.getNode(nodeId);
```
```cpp
        std::unordered_set<uint32_t> visited;
        std::function<void(uint32_t)> visit = [&](uint32_t nodeId)
        {
            if (visited.count(nodeId))
                return;
            visited.insert(nodeId);
            for (uint32_t pid : graph.nodes[nodeId].parentIds)
            {
                visit(mapToSelected(pid));
            }
            topo.push_back(nodeId);
        };
        visit(rootId);

        std::unordered_map<uint32_t, uint32_t> compiledRefCounts;
        for (uint32_t nodeId : topo)
        {
            for (uint32_t pid : graph.nodes[nodeId].parentIds)
            {
                compiledRefCounts[mapToSelected(pid)]++;
            }
        }
====
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
```
```cpp
        for (uint32_t nodeId : topo)
        {
            compiled.nodesMap[nodeId] = graph.nodes[nodeId];
            if (graph.constantStaging.count(nodeId))
                compiled.constantStaging[nodeId] = graph.constantStaging.at(nodeId);
        }

        for (uint32_t nodeId : topo)
        {
            const TensorNode &node = graph.nodes[nodeId];
            if (node.opType == OpType::INPUT)
                continue;
====
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
```
```cpp
        std::vector<uint32_t> order;
        std::unordered_set<uint32_t> visited;
        auto visit = [&](auto &self, uint32_t node) -> void
        {
            if (visited.count(node))
                return;
            visited.insert(node);
            for (uint32_t pid : graph.nodes[node].parentIds)
            {
                self(self, pid);
            }
            order.push_back(node);
        };
        visit(visit, rootId);
====
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
```

Update `tensor_graphs_cpp/core/session.hpp`:
```cpp
    std::vector<uint32_t> collectInputNodeIds() const
    {
        std::vector<uint32_t> inputNodeIds;
        for (const TensorNode &node : graph.nodes)
        {
            if (node.opType != OpType::INPUT)
                continue;
====
    std::vector<uint32_t> collectInputNodeIds() const
    {
        std::vector<uint32_t> inputNodeIds;
        for (const auto &pair : graph.nodes)
        {
            const TensorNode &node = pair.second;
            if (node.opType != OpType::INPUT)
                continue;
```
```cpp
        auto visit = [&](auto &self, uint32_t nodeId) -> void
        {
            if (visited.count(nodeId))
                return;
            visited.insert(nodeId);
            if (nodeId < graph.nodes.size())
            {
                for (uint32_t parentId : graph.nodes[nodeId].parentIds)
                    self(self, parentId);
            }
            atomicTopo.push_back(nodeId);
        };
        visit(visit, rootId);

        ShapePropagator prop;
        for (uint32_t nodeId : atomicTopo)
        {
            if (nodeId >= graph.nodes.size())
                continue;

            prop.inferShape(nodeId, graph);
            if (graph.nodes[nodeId].view.shape.empty() && !graph.nodes[nodeId].shape.empty())
            {
                graph.nodes[nodeId].view.shape = graph.nodes[nodeId].shape;
                graph.nodes[nodeId].view.strides = TensorView::calcContiguousStrides(graph.nodes[nodeId].shape);
                graph.nodes[nodeId].view.dtype = graph.nodes[nodeId].dtype;
            }
        }
====
        auto visit = [&](auto &self, uint32_t nodeId) -> void
        {
            if (visited.count(nodeId))
                return;
            visited.insert(nodeId);
            if (graph.hasNode(nodeId))
            {
                for (uint32_t parentId : graph.getNode(nodeId).parentIds)
                    self(self, parentId);
            }
            atomicTopo.push_back(nodeId);
        };
        visit(visit, rootId);

        ShapePropagator prop;
        for (uint32_t nodeId : atomicTopo)
        {
            if (!graph.hasNode(nodeId))
                continue;

            prop.inferShape(nodeId, graph);
            if (graph.getNode(nodeId).view.shape.empty() && !graph.getNode(nodeId).shape.empty())
            {
                graph.getNode(nodeId).view.shape = graph.getNode(nodeId).shape;
                graph.getNode(nodeId).view.strides = TensorView::calcContiguousStrides(graph.getNode(nodeId).shape);
                graph.getNode(nodeId).view.dtype = graph.getNode(nodeId).dtype;
            }
        }
```
```cpp
        for (uint32_t nodeId : inputNodeIds)
        {
            if (graph.nodes[nodeId].opType != OpType::INPUT)
                continue;

            InputOption option;
            option.nodeId = nodeId;
            for (uint32_t dimLen : graph.nodes[nodeId].shape)
            {
                option.dimSlices.push_back(generateSlicesForDim(dimLen, nBucketSizes));
            }
            inputOptions.push_back(std::move(option));
        }

        if (inputOptions.empty())
        {
            DirtyBucket bucket;
            bucket.regions[rootId] = makeFull(graph.nodes[rootId].shape);
            return {{"", bucket}};
        }
====
        for (uint32_t nodeId : inputNodeIds)
        {
            if (graph.getNode(nodeId).opType != OpType::INPUT)
                continue;

            InputOption option;
            option.nodeId = nodeId;
            for (uint32_t dimLen : graph.getNode(nodeId).shape)
            {
                option.dimSlices.push_back(generateSlicesForDim(dimLen, nBucketSizes));
            }
            inputOptions.push_back(std::move(option));
        }

        if (inputOptions.empty())
        {
            DirtyBucket bucket;
            bucket.regions[rootId] = makeFull(graph.getNode(rootId).shape);
            return {{"", bucket}};
        }
```
```cpp
            if (graph.constantStaging.count(nodeId))
                neededConstants.insert(nodeId);
            else if (pair.second.constantStaging.count(nodeId))
                neededConstants.insert(nodeId);
====
            if (graph.constantStaging.count(nodePair.first))
                neededConstants.insert(nodePair.first);
            else if (pair.second.constantStaging.count(nodePair.first))
                neededConstants.insert(nodePair.first);
```
```cpp
            std::unordered_map<uint32_t, std::vector<Region>> fullInputRegions;
            for (uint32_t inId : inputNodeIds)
            {
                fullInputRegions[inId] = makeFull(graph.nodes[inId].shape);
            }
====
            std::unordered_map<uint32_t, std::vector<Region>> fullInputRegions;
            for (uint32_t inId : inputNodeIds)
            {
                fullInputRegions[inId] = makeFull(graph.getNode(inId).shape);
            }
```
```cpp
            for (size_t d = 0; d < box.region.size(); ++d)
            {
                uint32_t dimLen = graph.nodes[nodeId].shape[d];
                uint32_t start = box.region[d].start;
====
            for (size_t d = 0; d < box.region.size(); ++d)
            {
                uint32_t dimLen = graph.getNode(nodeId).shape[d];
                uint32_t start = box.region[d].start;
```
```cpp
        for (const auto &pair : inputs)
        {
            uint32_t nodeId = pair.first;
            const void *newData = pair.second;

            if (graph.nodes[nodeId].opType != OpType::INPUT)
                continue;
====
        for (const auto &pair : inputs)
        {
            uint32_t nodeId = pair.first;
            const void *newData = pair.second;

            if (graph.getNode(nodeId).opType != OpType::INPUT)
                continue;
```
```cpp
            auto diff = computeInputDiff(oldData, newData, graph.nodes[nodeId].shape, graph.nodes[nodeId].dtype);
            std::cout << "Input Diffs " << pair.first << ": " << std::endl;
====
            auto diff = computeInputDiff(oldData, newData, graph.getNode(nodeId).shape, graph.getNode(nodeId).dtype);
            std::cout << "Input Diffs " << pair.first << ": " << std::endl;
```
```cpp
            if (!diff.empty())
            {
                inputDiffs[nodeId] = diff;
            }

            uint64_t sizeBytes = getSizeBytes(graph.nodes[nodeId].shape, graph.nodes[nodeId].dtype);
            auto &stored = previousInputData[nodeId];
            stored.resize(sizeBytes);
            std::memcpy(stored.data(), newData, sizeBytes);

            memManager.write(graph.nodes[nodeId].backend, nodeId, newData, sizeBytes);
        }

        auto canonicalDiffs = canonicalizeInputDiffs(inputDiffs);
        if (inputDiffs.empty())
        {
            const TensorNode &rootNode = graph.nodes.at(rootId);
====
            if (!diff.empty())
            {
                inputDiffs[nodeId] = diff;
            }

            uint64_t sizeBytes = getSizeBytes(graph.getNode(nodeId).shape, graph.getNode(nodeId).dtype);
            auto &stored = previousInputData[nodeId];
            stored.resize(sizeBytes);
            std::memcpy(stored.data(), newData, sizeBytes);

            memManager.write(graph.getNode(nodeId).backend, nodeId, newData, sizeBytes);
        }

        auto canonicalDiffs = canonicalizeInputDiffs(inputDiffs);
        if (inputDiffs.empty())
        {
            const TensorNode &rootNode = graph.getNode(rootId);
```
```cpp
        for (const TensorNode &node : graph.nodes)
        {
            if (node.opType == OpType::INPUT && graph.weightSources.count(node.id) == 0 && graph.constantStaging.count(node.id) == 0)
            {
====
        for (const auto &pair : graph.nodes)
        {
            const TensorNode &node = pair.second;
            if (node.opType == OpType::INPUT && graph.weightSources.count(node.id) == 0 && graph.constantStaging.count(node.id) == 0)
            {
```

Update `tensor_graphs_cpp/core/cache_optimizer.hpp`:
```cpp
static std::unordered_map<Backend, uint64_t> calculateCachedResidentMemoryByBackend(
    const Graph &graph,
    const std::unordered_set<uint32_t> &cachedNodes)
{
    std::unordered_map<Backend, uint64_t> residentByBackend;
    for (uint32_t nodeId : cachedNodes)
    {
        if (nodeId >= graph.nodes.size())
            continue;
        const TensorNode &node = graph.nodes[nodeId];
====
static std::unordered_map<Backend, uint64_t> calculateCachedResidentMemoryByBackend(
    const Graph &graph,
    const std::unordered_set<uint32_t> &cachedNodes)
{
    std::unordered_map<Backend, uint64_t> residentByBackend;
    for (uint32_t nodeId : cachedNodes)
    {
        if (!graph.hasNode(nodeId))
            continue;
        const TensorNode &node = graph.getNode(nodeId);
```
```cpp
static std::vector<uint32_t> collectCacheableNodes(const Graph &graph)
{
    std::vector<uint32_t> cacheableNodes;
    cacheableNodes.reserve(graph.nodes.size());

    for (const TensorNode &node : graph.nodes)
    {
        if (node.opType == OpType::INPUT || node.shape.empty())
            continue;
        cacheableNodes.push_back(node.id);
    }
====
static std::vector<uint32_t> collectCacheableNodes(const Graph &graph)
{
    std::vector<uint32_t> cacheableNodes;
    cacheableNodes.reserve(graph.nodes.size());

    for (const auto &pair : graph.nodes)
    {
        const TensorNode &node = pair.second;
        if (node.opType == OpType::INPUT || node.shape.empty())
            continue;
        cacheableNodes.push_back(node.id);
    }
```
```cpp
static std::unordered_map<uint32_t, uint64_t> buildLogicalNodeMemorySizes(
    const Graph &graph,
    const std::vector<uint32_t> &nodeIds)
{
    std::unordered_map<uint32_t, uint64_t> nodeMemorySizes;
    for (uint32_t nodeId : nodeIds)
    {
        if (nodeId >= graph.nodes.size())
            continue;
        const TensorNode &node = graph.nodes[nodeId];
====
static std::unordered_map<uint32_t, uint64_t> buildLogicalNodeMemorySizes(
    const Graph &graph,
    const std::vector<uint32_t> &nodeIds)
{
    std::unordered_map<uint32_t, uint64_t> nodeMemorySizes;
    for (uint32_t nodeId : nodeIds)
    {
        if (!graph.hasNode(nodeId))
            continue;
        const TensorNode &node = graph.getNode(nodeId);
```
```cpp
        std::unordered_map<Backend, std::vector<uint64_t>> sizesByBackend;
        for (uint32_t nodeId : cacheableNodes)
        {
            auto sizeIt = nodeMemorySizes.find(nodeId);
            if (sizeIt == nodeMemorySizes.end())
                continue;
            sizesByBackend[graph.nodes[nodeId].backend].push_back(sizeIt->second);
        }
====
        std::unordered_map<Backend, std::vector<uint64_t>> sizesByBackend;
        for (uint32_t nodeId : cacheableNodes)
        {
            auto sizeIt = nodeMemorySizes.find(nodeId);
            if (sizeIt == nodeMemorySizes.end())
                continue;
            sizesByBackend[graph.getNode(nodeId).backend].push_back(sizeIt->second);
        }
```

Update `tensor_graphs_cpp/core/misc.hpp`:
```cpp
        for (size_t i = 0; i < node.parentIds.size(); ++i)
        {
            uint32_t pid = node.parentIds[i];
            if (pid < graph.nodes.size())
            {
                const auto &parent = graph.nodes[pid];
                ss << "\n"
                   << prefix << "    [" << i << "] Parent ID " << pid
                   << "\n"
                   << toString(parent, (std::string) "    ");
            }
            else
            {
                ss << "\n"
                   << prefix << "    [" << i << "] Parent ID " << pid << " [OUT OF BOUNDS]";
            }
        }
====
        for (size_t i = 0; i < node.parentIds.size(); ++i)
        {
            uint32_t pid = node.parentIds[i];
            if (graph.hasNode(pid))
            {
                const auto &parent = graph.getNode(pid);
                ss << "\n"
                   << prefix << "    [" << i << "] Parent ID " << pid
                   << "\n"
                   << toString(parent, (std::string) "    ");
            }
            else
            {
                ss << "\n"
                   << prefix << "[" << i << "] Parent ID " << pid << " [OUT OF BOUNDS/NOT FOUND]";
            }
        }
```

Update `tensor_graphs_cpp/test.cpp`:
```cpp
    auto findEquivalent =[](const std::vector<uint32_t> &equivalents, const Graph &graph, OpType opType) -> bool
    {
        for (uint32_t id : equivalents)
        {
            if (id < graph.nodes.size() && graph.nodes[id].opType == opType)
                return true;
        }
        return false;
    };
====
    auto findEquivalent =[](const std::vector<uint32_t> &equivalents, const Graph &graph, OpType opType) -> bool
    {
        for (uint32_t id : equivalents)
        {
            if (graph.hasNode(id) && graph.getNode(id).opType == opType)
                return true;
        }
        return false;
    };
```
```cpp
        size_t logicalMulConsumers = 0;
        for (const TensorNode &node : rebuilt.graph.nodes)
        {
            for (uint32_t parentId : node.parentIds)
            {
====
        size_t logicalMulConsumers = 0;
        for (const auto &pair : rebuilt.graph.nodes)
        {
            const TensorNode &node = pair.second;
            for (uint32_t parentId : node.parentIds)
            {
```
```cpp
        const TensorNode &firstScatter = rebuilt.graph.nodes[scatterIt->second.front()];
        const TensorNode &finalScatter = rebuilt.graph.nodes[scatterIt->second.back()];
====
        const TensorNode &firstScatter = rebuilt.graph.getNode(scatterIt->second.front());
        const TensorNode &finalScatter = rebuilt.graph.getNode(scatterIt->second.back());
```
```cpp
        for (uint32_t partialId : partialIt->second)
        {
            const TensorNode &partialNode = rebuilt.graph.nodes[partialId];
            if (partialNode.opType != OpType::MUL)
                Error::throw_err("[PlannerTest] cached rebuild produced a non-MUL partial node");

            bool sawSliceChain = false;
            for (uint32_t parentId : partialNode.parentIds)
            {
                const TensorNode &parentNode = rebuilt.graph.nodes[parentId];
                if (parentNode.opType == OpType::CONTIGUOUS &&
                    !parentNode.parentIds.empty() &&
                    rebuilt.graph.nodes[parentNode.parentIds[0]].opType == OpType::SLICE)
====
        for (uint32_t partialId : partialIt->second)
        {
            const TensorNode &partialNode = rebuilt.graph.getNode(partialId);
            if (partialNode.opType != OpType::MUL)
                Error::throw_err("[PlannerTest] cached rebuild produced a non-MUL partial node");

            bool sawSliceChain = false;
            for (uint32_t parentId : partialNode.parentIds)
            {
                const TensorNode &parentNode = rebuilt.graph.getNode(parentId);
                if (parentNode.opType == OpType::CONTIGUOUS &&
                    !parentNode.parentIds.empty() &&
                    rebuilt.graph.getNode(parentNode.parentIds[0]).opType == OpType::SLICE)
```
```cpp
    auto visit = [&](auto &self, uint32_t node) -> void
    {
        if (visited.count(node))
            return;
        visited.insert(node);
        if (node < graph.nodes.size())
        {
            for (uint32_t pid : graph.nodes[node].parentIds)
            {
                self(self, pid);
            }
        }
        order.push_back(node);
    };
====
    auto visit = [&](auto &self, uint32_t node) -> void
    {
        if (visited.count(node))
            return;
        visited.insert(node);
        if (graph.hasNode(node))
        {
            for (uint32_t pid : graph.getNode(node).parentIds)
            {
                self(self, pid);
            }
        }
        order.push_back(node);
    };
```
```cpp
    for (uint32_t nodeId : topo)
    {
        const TensorNode &node = graph.nodes[nodeId];
        uint64_t elemSize = getDTypeSize(node.dtype);
====
    for (uint32_t nodeId : topo)
    {
        const TensorNode &node = graph.getNode(nodeId);
        uint64_t elemSize = getDTypeSize(node.dtype);
```
```cpp
        for (uint32_t pid : node.parentIds)
        {
            auto resultIt = results.find(pid);
            if (resultIt == results.end())
            {
                Error::throw_err("Parent node " + std::to_string(pid) + " not found in results");
            }
            inputPtrs.push_back(resultIt->second.data());
            inputViews.push_back(views[pid]);
            TensorNode inNode = graph.nodes[pid];
            inNode.view = views[pid];
            inputNodes.push_back(inNode);
        }
====
        for (uint32_t pid : node.parentIds)
        {
            auto resultIt = results.find(pid);
            if (resultIt == results.end())
            {
                Error::throw_err("Parent node " + std::to_string(pid) + " not found in results");
            }
            inputPtrs.push_back(resultIt->second.data());
            inputViews.push_back(views[pid]);
            TensorNode inNode = graph.getNode(pid);
            inNode.view = views[pid];
            inputNodes.push_back(inNode);
        }
```
```cpp
    uint64_t numRootElems = countElements(graph.nodes[rootId].shape);
    std::vector<float> finalOut(numRootElems, 0.0f);
    TensorView rootView = views[rootId];

    for (size_t i = 0; i < numRootElems; ++i)
    {
        uint64_t idx = getStridedIndex(i, rootView.shape, rootView.strides);
        if (graph.nodes[rootId].dtype == DType::FLOAT32)
        {
            std::memcpy(&finalOut[i], results[rootId].data() + idx * 4, 4);
        }
        else if (graph.nodes[rootId].dtype == DType::INT32)
        {
            int32_t val;
            std::memcpy(&val, results[rootId].data() + idx * 4, 4);
            finalOut[i] = static_cast<float>(val);
        }
        else if (graph.nodes[rootId].dtype == DType::BF16)
        {
            uint16_t val;
            std::memcpy(&val, results[rootId].data() + idx * 2, 2);
            uint32_t f32_bits = static_cast<uint32_t>(val) << 16;
            std::memcpy(&finalOut[i], &f32_bits, 4);
        }
        else if (graph.nodes[rootId].dtype == DType::BOOL)
        {
            uint8_t val;
            std::memcpy(&val, results[rootId].data() + idx, 1);
            finalOut[i] = static_cast<float>(val);
        }
====
    uint64_t numRootElems = countElements(graph.getNode(rootId).shape);
    std::vector<float> finalOut(numRootElems, 0.0f);
    TensorView rootView = views[rootId];

    for (size_t i = 0; i < numRootElems; ++i)
    {
        uint64_t idx = getStridedIndex(i, rootView.shape, rootView.strides);
        if (graph.getNode(rootId).dtype == DType::FLOAT32)
        {
            std::memcpy(&finalOut[i], results[rootId].data() + idx * 4, 4);
        }
        else if (graph.getNode(rootId).dtype == DType::INT32)
        {
            int32_t val;
            std::memcpy(&val, results[rootId].data() + idx * 4, 4);
            finalOut[i] = static_cast<float>(val);
        }
        else if (graph.getNode(rootId).dtype == DType::BF16)
        {
            uint16_t val;
            std::memcpy(&val, results[rootId].data() + idx * 2, 2);
            uint32_t f32_bits = static_cast<uint32_t>(val) << 16;
            std::memcpy(&finalOut[i], &f32_bits, 4);
        }
        else if (graph.getNode(rootId).dtype == DType::BOOL)
        {
            uint8_t val;
            std::memcpy(&val, results[rootId].data() + idx, 1);
            finalOut[i] = static_cast<float>(val);
        }
```
```cpp
            // ========== FUSED KERNEL EXECUTION ==========
            // Execute fused kernel directly
            std::vector<float> fusedOutput = executeFusedKernel(kernel, refInputs.rawData, refOutput.size(), refGraph.nodes[rootId].shape);
====
            // ========== FUSED KERNEL EXECUTION ==========
            // Execute fused kernel directly
            std::vector<float> fusedOutput = executeFusedKernel(kernel, refInputs.rawData, refOutput.size(), refGraph.getNode(rootId).shape);
```

Update `tensor_graphs_cpp/kernels/cpu/general/add/FP32_3D_1D.hpp`:
```cpp
inline uint32_t refFactoryAdd3D_1D(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Add 3D+1D requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t id1D = inputs[1];

    auto shape3D = graph.nodes[id3D].shape;
    auto shape1D = graph.nodes[id1D].shape;
====
inline uint32_t refFactoryAdd3D_1D(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Add 3D+1D requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t id1D = inputs[1];

    auto shape3D = graph.getNode(id3D).shape;
    auto shape1D = graph.getNode(id1D).shape;
```

Update `tensor_graphs_cpp/kernels/cpu/general/add/FP32_3D_scalar.hpp`:
```cpp
inline uint32_t refFactoryAdd3D_Scalar(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Add 3D+Scalar requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t idScalar = inputs[1];

    auto shape3D = graph.nodes[id3D].shape;
====
inline uint32_t refFactoryAdd3D_Scalar(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Add 3D+Scalar requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t idScalar = inputs[1];

    auto shape3D = graph.getNode(id3D).shape;
```

Update `tensor_graphs_cpp/kernels/cpu/general/add/inplace_FP32_3D_1D.hpp`:
```cpp
inline uint32_t refFactoryAdd3D_1D_Inplace(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Add 3D+1D requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t id1D = inputs[1];

    auto shape3D = graph.nodes[id3D].shape;
    auto shape1D = graph.nodes[id1D].shape;
====
inline uint32_t refFactoryAdd3D_1D_Inplace(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Add 3D+1D requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t id1D = inputs[1];

    auto shape3D = graph.getNode(id3D).shape;
    auto shape1D = graph.getNode(id1D).shape;
```

Update `tensor_graphs_cpp/kernels/cpu/general/add/inplace_FP32_3D_scalar.hpp`:
```cpp
inline uint32_t refFactoryAdd3D_Scalar_Inplace(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Add 3D+Scalar requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t idScalar = inputs[1];

    auto shape3D = graph.nodes[id3D].shape;
====
inline uint32_t refFactoryAdd3D_Scalar_Inplace(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Add 3D+Scalar requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t idScalar = inputs[1];

    auto shape3D = graph.getNode(id3D).shape;
```

Update `tensor_graphs_cpp/kernels/cpu/general/tanh/F32_1D.hpp`:
```cpp
uint32_t refFactoryTanh(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 1)
        Error::throw_err("Tanh requires 1 input");
    uint32_t x = inputs[0];
    uint32_t n_elements = graph.nodes[x].shape[0];
====
uint32_t refFactoryTanh(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 1)
        Error::throw_err("Tanh requires 1 input");
    uint32_t x = inputs[0];
    uint32_t n_elements = graph.getNode(x).shape[0];
```