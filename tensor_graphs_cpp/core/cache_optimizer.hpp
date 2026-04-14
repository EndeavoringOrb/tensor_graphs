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

struct BucketPlanRequest
{
    std::string key;
    DirtyBucket bucket;
};

struct CacheOptimizationResult
{
    std::unordered_set<uint32_t> bestCachedNodes;
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

CacheOptimizationResult optimizeCacheCombination(
    uint32_t rootId,
    Graph &graph,
    const std::vector<BucketPlanRequest> &buckets,
    const std::unordered_map<std::string, uint64_t> &bucketCallCounts,
    uint64_t maxCacheMemory,
    Planner &planner)
{
    CacheOptimizationResult result;
    if (buckets.empty())
        return result;

    const std::vector<uint32_t> cacheableNodes = collectCacheableNodes(graph);
    const std::unordered_map<uint32_t, uint64_t> nodeMemorySizes = buildLogicalNodeMemorySizes(graph, cacheableNodes);

    std::cout << "[CacheOptimizer] Evaluating baseline cost..." << std::endl;
    float baselineScore = 0.0f;
    for (const BucketPlanRequest &bucket : buckets)
    {
        float cost = planner.estimateCostForCacheSet(rootId, graph, bucket.bucket.regions, bucket.bucket.inputSlices, {});
        auto countIt = bucketCallCounts.find(bucket.key);
        uint64_t callCount = (countIt != bucketCallCounts.end()) ? countIt->second : 1;
        baselineScore += cost * callCount;
    }
    if (baselineScore == std::numeric_limits<float>::infinity())
    {
        std::cout << "[CacheOptimizer] WARNING: Baseline Score is inf, terminating cache optimization" << std::endl;
        return result;
    }

    std::cout << "[CacheOptimizer] Baseline Score: " << baselineScore << std::endl;

    struct NodeROI
    {
        uint32_t nodeId;
        float costSaved;
        uint64_t sizeBytes;
        float roi;
    };

    std::vector<NodeROI> nodeROIs;
    ProgressTimer timer(cacheableNodes.size(), "Evaluating nodes for caching: ");

    for (uint32_t nodeId : cacheableNodes)
    {
        timer.tick();
        float testScore = 0.0f;
        for (const BucketPlanRequest &bucket : buckets)
        {
            float cost = planner.estimateCostForCacheSet(rootId, graph, bucket.bucket.regions, bucket.bucket.inputSlices, {nodeId});
            auto countIt = bucketCallCounts.find(bucket.key);
            uint64_t callCount = (countIt != bucketCallCounts.end()) ? countIt->second : 1;
            testScore += cost * callCount;
        }

        float costSaved = baselineScore - testScore;
        uint64_t sizeBytes = nodeMemorySizes.count(nodeId) ? nodeMemorySizes.at(nodeId) : 0;
        float roi = sizeBytes > 0 ? (costSaved / static_cast<float>(sizeBytes)) : 0.0f;

        nodeROIs.push_back({nodeId, costSaved, sizeBytes, roi});
    }

    std::sort(nodeROIs.begin(), nodeROIs.end(), [](const NodeROI &a, const NodeROI &b)
              { return a.roi > b.roi; });

    std::unordered_set<uint32_t> selectedCachedNodes;
    uint64_t currentCacheMem = 0;

    for (const auto &nr : nodeROIs)
    {
        if (nr.roi > 0.0f && currentCacheMem + nr.sizeBytes <= maxCacheMemory)
        {
            selectedCachedNodes.insert(nr.nodeId);
            currentCacheMem += nr.sizeBytes;
            std::cout << "[CacheOptimizer] Selected Node " << nr.nodeId
                      << " | ROI: " << nr.roi
                      << " | Saved: " << nr.costSaved
                      << " | Mem: " << nr.sizeBytes << " bytes" << std::endl;
        }
    }

    std::cout << "[CacheOptimizer] Final selected cache set size: " << selectedCachedNodes.size()
              << " nodes, using " << currentCacheMem << " bytes." << std::endl;

    std::unordered_map<std::string, CompiledGraph> plans;
    float finalScore = 0.0f;

    std::cout << "[CacheOptimizer] Generating final plans..." << std::endl;
    for (const BucketPlanRequest &bucket : buckets)
    {
        CompiledGraph plan = planner.plan(rootId, graph, bucket.bucket.regions, bucket.bucket.inputSlices, selectedCachedNodes);
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