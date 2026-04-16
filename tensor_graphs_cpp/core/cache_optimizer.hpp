#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/cost_model.hpp"
#include "core/planner.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
#include <sstream>
#include <future>
#include <thread>
#include <mutex>

enum class CacheHeuristic
{
    SIZE,
    ROI,
    FUSION
};

struct BucketPlanRequest
{
    std::string key;
    DirtyBucket bucket;
};

struct CacheOptimizationResult
{
    std::unordered_map<uint32_t, Backend> bestCachedNodes;
    std::unordered_map<std::string, CompiledGraph> bucketPlans;
    float runtimeScore;
    bool foundValidCombination;

    CacheOptimizationResult() : runtimeScore(std::numeric_limits<float>::infinity()), foundValidCombination(false) {}
};

static std::vector<uint32_t> collectCacheableNodes(const Graph &graph)
{
    std::vector<uint32_t> cacheableNodes;
    cacheableNodes.reserve(graph.nodes.size());

    for (const auto &pair : graph.nodes)
    {
        const TensorNode &node = pair.second;
        if (node.opType == OpType::INPUT || node.opType == OpType::PERMUTE || node.opType == OpType::REPEAT || node.opType == OpType::RESHAPE || node.opType == OpType::SLICE || node.getShape().empty())
            continue;
        cacheableNodes.push_back(node.id);
    }

    std::sort(cacheableNodes.begin(), cacheableNodes.end());
    return cacheableNodes;
}

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
        if (node.getShape().empty())
            continue;
        nodeMemorySizes[nodeId] = getSizeBytes(node.getShape(), node.dtype);
    }
    return nodeMemorySizes;
}

static float getPlanRuntime(const CompiledGraph &plan)
{
    float totalCost = 0.0f;
    for (const auto &kv : plan.nodeCosts)
    {
        totalCost += kv.second;
    }
    return totalCost;
}

CacheOptimizationResult optimizeCacheByFusion(
    uint32_t rootId,
    Graph &graph,
    const std::vector<BucketPlanRequest> &buckets,
    const std::string &fullKey,
    const std::unordered_map<std::string, uint64_t> &bucketCallCounts,
    uint64_t maxCacheMemory,
    Planner &planner,
    bool doSaturate = true)
{
    CacheOptimizationResult result;
    if (buckets.empty())
        return result;

    const std::vector<uint32_t> cacheableNodes = collectCacheableNodes(graph);
    const std::unordered_map<uint32_t, uint64_t> nodeMemorySizes = buildLogicalNodeMemorySizes(graph, cacheableNodes);

    std::cout << "[CacheOptimizer] Planning full bucket to find unfused nodes..." << std::endl;

    std::unordered_map<std::string, PlanningRegionState> regionStates;
    std::unordered_map<uint32_t, Backend> survivingLogicalNodes;

    // First, plan all buckets to get regionStates, but explicitly extract survivors from the full bucket
    for (const BucketPlanRequest &bucket : buckets)
    {
        regionStates[bucket.key] = derivePlanningRegions(rootId, graph, bucket.bucket.regions);

        if (bucket.key == fullKey)
        {
            CompiledGraph plan = planner.plan(rootId, graph, bucket.bucket.regions, bucket.bucket.inputSlices, {}, regionStates[bucket.key], doSaturate, true);
            for (const auto &inst : plan.instructions)
            {
                uint32_t logicalId = plan.getLogicalId(inst.nodeId);
                if (logicalId != UINT32_MAX)
                {
                    survivingLogicalNodes[logicalId] = inst.backend;
                }
            }
        }
    }

    std::cout << "[CacheOptimizer] Sorting surviving nodes by size..." << std::endl;
    std::vector<uint32_t> candidateNodes;
    for (uint32_t id : cacheableNodes)
    {
        if (survivingLogicalNodes.count(id))
        {
            candidateNodes.push_back(id);
        }
    }

    std::sort(candidateNodes.begin(), candidateNodes.end(), [&](uint32_t a, uint32_t b)
              {
                  uint64_t sizeA = nodeMemorySizes.at(a);
                  uint64_t sizeB = nodeMemorySizes.at(b);
                  if (sizeA != sizeB)
                      return sizeA > sizeB;
                  return a < b; // stable tie-break
              });

    std::unordered_map<uint32_t, Backend> selectedCachedNodes;
    uint64_t currentCacheMem = 0;

    for (uint32_t nodeId : candidateNodes)
    {
        uint64_t nodeSize = nodeMemorySizes.at(nodeId);
        if (currentCacheMem + nodeSize <= maxCacheMemory)
        {
            selectedCachedNodes[nodeId] = survivingLogicalNodes.at(nodeId);
            currentCacheMem += nodeSize;
            std::cout << "[CacheOptimizer] Selected Node " << nodeId
                      << " | Size: " << nodeSize << " bytes" << std::endl;
        }
    }

    // Add persistent inputs to ensure their backend is forced correctly across normal buckets
    for (const auto &pair : graph.nodes)
    {
        if (pair.second.opType == OpType::INPUT && pair.second.storageType == StorageType::PERSISTENT)
        {
            if (survivingLogicalNodes.count(pair.first))
            {
                selectedCachedNodes[pair.first] = survivingLogicalNodes.at(pair.first);
            }
        }
    }

    std::cout << "[CacheOptimizer] Final fusion-based cache set: " << selectedCachedNodes.size()
              << " nodes, using " << currentCacheMem << " bytes." << std::endl;

    std::unordered_map<std::string, CompiledGraph> plans;
    float finalScore = 0.0f;

    std::cout << "[CacheOptimizer] Generating plans for " << buckets.size() << " buckets..." << std::endl;
    for (const BucketPlanRequest &bucket : buckets)
    {
        bool isFullKey = (bucket.key == fullKey);
        CompiledGraph plan = planner.plan(rootId, graph, bucket.bucket.regions, bucket.bucket.inputSlices, selectedCachedNodes, regionStates[bucket.key], doSaturate, isFullKey);

        float cost = getPlanRuntime(plan);
        auto countIt = bucketCallCounts.find(bucket.key);
        uint64_t callCount = (countIt != bucketCallCounts.end()) ? countIt->second : 1;
        finalScore += cost * callCount;
        plans[bucket.key] = std::move(plan);
    }

    result.bestCachedNodes = std::move(selectedCachedNodes);
    result.bucketPlans = std::move(plans);
    result.runtimeScore = finalScore;
    result.foundValidCombination = true;

    return result;
}