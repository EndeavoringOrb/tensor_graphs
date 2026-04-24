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

    // 1. Identify a partial bucket to evaluate cache profitability organically
    const BucketPlanRequest *partialBucket = nullptr;
    for (const auto &b : buckets)
    {
        if (b.key != fullKey)
        {
            partialBucket = &b;
            break;
        }
    }
    if (!partialBucket)
        partialBucket = &buckets[0];

    std::cout << "[CacheOptimizer] Evaluating cache profitability organically on bucket: " << partialBucket->key << std::endl;

    // Provide ALL nodes as potential caches
    std::unordered_map<uint32_t, Backend> allCachedNodes;
    for (uint32_t id : cacheableNodes)
    {
        allCachedNodes[id] = graph.getNode(id).backend;
    }

    std::unordered_map<std::string, PlanningRegionState> regionStates;
    regionStates[partialBucket->key] = derivePlanningRegions(rootId, graph, partialBucket->bucket.regions, partialBucket->bucket.outputNeeded);

    // Plan with ALL nodes cached, but DO NOT protect them! This lets extractBest pick the fastest organic path.
    CompiledGraph exploratoryPlan = planner.plan(
        rootId, graph, partialBucket->bucket.regions, partialBucket->bucket.inputSlices,
        allCachedNodes, regionStates[partialBucket->key], partialBucket->bucket.outputNeeded,
        doSaturate, false, true); // <--- Note: protectCachedNodes = false

    std::unordered_map<uint32_t, Backend> survivingLogicalNodes;
    for (const auto &inst : exploratoryPlan.instructions)
    {
        uint32_t logicalId = exploratoryPlan.getLogicalId(inst.nodeId);
        if (logicalId != UINT32_MAX && allCachedNodes.count(logicalId))
        {
            // If the logical ID survived in the graph (e.g. as a SCATTER update), it was deemed profitable!
            survivingLogicalNodes[logicalId] = inst.backend;
        }
    }

    std::vector<uint32_t> candidateNodes;
    for (uint32_t id : cacheableNodes)
    {
        if (survivingLogicalNodes.count(id))
            candidateNodes.push_back(id);
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
        }
    }

    // Add persistent inputs to ensure their backend is forced correctly across normal buckets
    for (const auto &pair : graph.nodes)
    {
        if (pair.second.opType == OpType::INPUT && pair.second.storageType == StorageType::PERSISTENT)
        {
            selectedCachedNodes[pair.first] = pair.second.backend;
        }
    }

    std::cout << "[CacheOptimizer] Selected " << selectedCachedNodes.size()
              << " nodes for caching, using " << currentCacheMem << " bytes." << std::endl;

    std::unordered_map<std::string, CompiledGraph> plans;
    float finalScore = 0.0f;

    std::cout << "[CacheOptimizer] Generating final constrained plans for " << buckets.size() << " buckets..." << std::endl;
    for (const BucketPlanRequest &bucket : buckets)
    {
        bool isFullKey = (bucket.key == fullKey);
        if (regionStates.find(bucket.key) == regionStates.end())
        {
            regionStates[bucket.key] = derivePlanningRegions(rootId, graph, bucket.bucket.regions, bucket.bucket.outputNeeded);
        }

        // Final plans DO protect the selected caches
        CompiledGraph plan = planner.plan(
            rootId, graph, bucket.bucket.regions, bucket.bucket.inputSlices,
            selectedCachedNodes, regionStates[bucket.key], bucket.bucket.outputNeeded,
            doSaturate, true, isFullKey);

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