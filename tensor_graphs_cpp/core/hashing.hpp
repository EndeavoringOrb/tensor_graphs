#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include "core/types.hpp"
#include "core/graph.hpp"

namespace Hashing
{
    inline std::string structuralHash(uint32_t nodeId, const Graph &graph, std::unordered_map<uint32_t, std::string> &memo)
    {
        if (memo.count(nodeId))
        {
            return memo[nodeId];
        }
        if (nodeId >= graph.nodes.size())
        {
            throw TensorGraphError("Invalid node ID encountered during hashing.");
        }

        const TensorNode &node = graph.nodes[nodeId];
        SHA256 sha; // Now available from types.hpp

        uint32_t opVal = static_cast<uint32_t>(node.opType);
        sha.update(reinterpret_cast<const uint8_t *>(&opVal), sizeof(opVal));

        if (node.opType == OpType::FUSED)
        {
            sha.update(node.opName);
            sha.update("|");
        }

        uint32_t dtypeVal = static_cast<uint32_t>(node.dtype);
        sha.update(reinterpret_cast<const uint8_t *>(&dtypeVal), sizeof(dtypeVal));

        uint32_t shapeRank = static_cast<uint32_t>(node.shape.size());
        sha.update(reinterpret_cast<const uint8_t *>(&shapeRank), sizeof(shapeRank));
        for (uint32_t dim : node.shape)
        {
            sha.update(reinterpret_cast<const uint8_t *>(&dim), sizeof(dim));
        }

        if (node.opType == OpType::INPUT)
        {
            if (node.storageType == StorageType::PERSISTENT)
            {
                sha.update(node.contentHash);
            }
            else
            {
                uint32_t idVal = node.id;
                sha.update(reinterpret_cast<const uint8_t *>(&idVal), sizeof(idVal));
            }
        }
        else
        {
            uint32_t numParents = static_cast<uint32_t>(node.parentIds.size());
            sha.update(reinterpret_cast<const uint8_t *>(&numParents), sizeof(numParents));

            const std::string delim = "|";
            for (uint32_t pid : node.parentIds)
            {
                std::string ph = structuralHash(pid, graph, memo);
                sha.update(ph);
                sha.update(delim);
            }
        }

        std::string result = sha.digest();
        memo[nodeId] = result;
        return result;
    }

    inline std::string patternHash(uint32_t nodeId, const Graph &graph, std::unordered_map<uint32_t, std::string> &memo)
    {
        if (memo.count(nodeId))
        {
            return memo[nodeId];
        }
        if (nodeId >= graph.nodes.size())
        {
            throw TensorGraphError("Invalid node ID encountered during hashing.");
        }

        const TensorNode &node = graph.nodes[nodeId];
        SHA256 sha;

        uint32_t opVal = static_cast<uint32_t>(node.opType);
        sha.update(reinterpret_cast<const uint8_t *>(&opVal), sizeof(opVal));

        if (node.opType == OpType::FUSED)
        {
            sha.update(node.opName);
            sha.update("|");
        }

        if (node.opType == OpType::INPUT)
        {
            sha.update("*|");
        }
        else
        {
            uint32_t numParents = static_cast<uint32_t>(node.parentIds.size());
            sha.update(reinterpret_cast<const uint8_t *>(&numParents), sizeof(numParents));

            const std::string delim = "|";
            for (uint32_t pid : node.parentIds)
            {
                std::string ph = patternHash(pid, graph, memo);
                sha.update(ph);
                sha.update(delim);
            }
        }

        std::string result = sha.digest();
        memo[nodeId] = result;
        return result;
    }
} // namespace Hashing
