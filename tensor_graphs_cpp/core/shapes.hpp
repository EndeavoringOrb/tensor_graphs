#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/planner.hpp"

bool shapesMatch(const std::vector<uint32_t> &shape1, const std::vector<uint32_t> &shape2)
{
    if (shape1.size() != shape2.size())
        return false;
    for (size_t i = 0; i < shape1.size(); ++i)
    {
        if (shape1[i] != shape2[i])
            return false;
    }
    return true;
}

struct ShapePropagator
{
    std::vector<Region> forward(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        for (uint32_t pid : node.parentIds)
        {
            if (pid >= graph.nodes.size())
            {
                std::stringstream ss;
                ss << "[ShapePropagator.forward] Invalid parent ID " << pid << " for OpType " << node.opType;
                throw std::runtime_error(ss.str());
            }
        }
        switch (node.opType)
        {
        case OpType::ADD:
            return forwardAdd(node, graph, parentRegions);
        default:
            std::stringstream ss;
            ss << "[ShapePropagator.forward] Unsupported OpType for ShapePropagator.forward: " << node.opType;
            throw std::runtime_error(ss.str());
        }
    }

    // Output regions are the unique set of all parent regions
    std::vector<Region> forwardAdd(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        if (node.parentIds.size() != 2)
        {
            std::stringstream ss;
            ss << "[ShapePropagator.forwardAdd] ADD requires exactly 2 parents, got "
               << node.parentIds.size();
            throw std::runtime_error(ss.str());
        }

        uint32_t pid0 = node.parentIds[0];
        uint32_t pid1 = node.parentIds[1];
        const auto &parent0 = graph.nodes[pid0];
        const auto &parent1 = graph.nodes[pid1];

        if (!shapesMatch(parent0.shape, parent1.shape))
        {
            std::stringstream ss;
            ss << "[ShapePropagator.forwardAdd] Shape mismatch in ADD node: " << toString(parent0.shape) << ", " << toString(parent1.shape);
            throw std::runtime_error(ss.str());
        }

        std::vector<Region> outputRegions;

        auto regionExists = [&](const Region &r)
        {
            for (const auto &existing : outputRegions)
            {
                if (regionsMatch(existing, r))
                    return true;
            }
            return false;
        };

        for (const auto &region : parentRegions[0])
        {
            if (!regionExists(region))
            {
                outputRegions.push_back(region);
            }
        }
        for (const auto &region : parentRegions[1])
        {
            if (!regionExists(region))
            {
                outputRegions.push_back(region);
            }
        }

        return outputRegions;
    }

    // Dispatch to op-specific backward function
    std::vector<std::vector<Region>> backward(const TensorNode &node, const std::vector<TensorNode> &allNodes, const std::vector<Region> &outputRegions)
    {
        switch (node.opType)
        {
        case OpType::ADD:
            return backwardAdd(node, allNodes, outputRegions);
        default:
            std::stringstream ss;
            ss << "[ShapePropagator.backward] Unsupported OpType for ShapePropagator.backward: " << node.opType;
            throw std::runtime_error(ss.str());
        }
    }

    // For every dirty region in the output, BOTH corresponding inputs are also dirty
    std::vector<std::vector<Region>> backwardAdd(const TensorNode &node, const std::vector<TensorNode> &allNodes, const std::vector<Region> &outputRegions)
    {
        std::vector<std::vector<Region>> inputRegions(2);
        for (size_t i = 0; i < 2; ++i)
        {
            inputRegions[i] = outputRegions;
        }

        return inputRegions;
    }
};

// ---------------------------------------------------------------------------
// Free functions for dirty region propagation (no DirtyPropagator class)
// ---------------------------------------------------------------------------

// Forward-propagate dirty regions through the compiled graph.
// inputDirtyRegions maps input node IDs to their dirty regions.
// Returns a map of every node ID to its propagated output regions.
inline std::unordered_map<uint32_t, std::vector<Region>> propagateDirtyRegions(
    const CompiledGraph &compiled,
    const Graph &graph,
    const std::unordered_map<uint32_t, std::vector<Region>> &inputDirtyRegions)
{
    ShapePropagator propagator;
    std::unordered_map<uint32_t, std::vector<Region>> allRegions(inputDirtyRegions);

    for (const OpInstruction &inst : compiled.instructions)
    {
        const TensorNode &node = graph.nodes[inst.nodeId];

        // Gather parent regions
        std::vector<std::vector<Region>> parentRegions;
        bool anyParentDirty = false;
        for (uint32_t pid : node.parentIds)
        {
            auto it = allRegions.find(pid);
            if (it != allRegions.end() && !it->second.empty())
            {
                parentRegions.push_back(it->second);
                anyParentDirty = true;
            }
            else
            {
                parentRegions.push_back({});
            }
        }

        if (anyParentDirty)
        {
            allRegions[inst.nodeId] = propagator.forward(node, graph, parentRegions);
        }
        else
        {
            allRegions[inst.nodeId] = {};
        }
    }

    return allRegions;
}

// Backward-propagate: given a node and its output dirty regions,
// compute the required input regions per parent.
// Returns one vector<Region> per parent (same order as node.parentIds).
inline std::vector<std::vector<Region>> getInputSlices(
    const Graph &graph,
    uint32_t nodeId,
    const std::vector<Region> &outputRegions)
{
    const TensorNode &node = graph.nodes[nodeId];

    if (outputRegions.empty())
    {
        return std::vector<std::vector<Region>>(node.parentIds.size());
    }

    ShapePropagator propagator;
    return propagator.backward(node, graph.nodes, outputRegions);
}