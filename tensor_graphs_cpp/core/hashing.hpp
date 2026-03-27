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
    inline std::string patternHash(uint32_t nodeId, const Graph &graph, std::unordered_map<uint32_t, std::string> &memo)
    {
        if (memo.count(nodeId))
        {
            return memo[nodeId];
        }
        if (!graph.hasNode(nodeId))
        {
            throw TensorGraphError("Invalid node ID encountered during hashing.");
        }

        const TensorNode &node = graph.getNode(nodeId);
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
