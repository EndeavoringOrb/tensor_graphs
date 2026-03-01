#pragma once
#include "core/types.hpp"

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
    std::vector<Region> forward(const TensorNode &node, const std::vector<TensorNode> &allNodes, const std::vector<std::vector<Region>> &parentRegions)
    {
        for (uint32_t pid : node.parentIds)
        {
            if (pid >= allNodes.size())
            {
                std::stringstream ss;
                ss << "[ShapePropagator.forward] Invalid parent ID " << pid << " for OpType " << node.opType;
                throw std::runtime_error(ss.str());
            }
        }
        switch (node.opType)
        {
        case OpType::ADD:
            return forwardAdd(node, allNodes, parentRegions);
        default:
            std::stringstream ss;
            ss << "[ShapePropagator.forward] Unsupported OpType for ShapePropagator.forward: " << node.opType;
            throw std::runtime_error(ss.str());
        }
    }

    // Output regions are the unique set of all parent regions
    std::vector<Region> forwardAdd(const TensorNode &node, const std::vector<TensorNode> &allNodes, const std::vector<std::vector<Region>> &parentRegions)
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
        const auto &parent0 = allNodes[pid0];
        const auto &parent1 = allNodes[pid1];

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