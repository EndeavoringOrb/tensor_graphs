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
    Planner &planner,
    bool doSaturate = true)
{
    CacheOptimizationResult result;
    if (buckets.empty())
        return result;

    const std::vector<uint32_t> cacheableNodes = collectCacheableNodes(graph);
    const std::unordered_map<uint32_t, uint64_t> nodeMemorySizes = buildLogicalNodeMemorySizes(graph, cacheableNodes);

    std::cout << "[CacheOptimizer] Evaluating baseline cost..." << std::endl;
    std::unordered_map<std::string, PlanningRegionState> regionStates;
    float baselineScore = 0.0f;
    for (const BucketPlanRequest &bucket : buckets)
    {
        regionStates[bucket.key] = derivePlanningRegions(rootId, graph, bucket.bucket.regions);
        float cost = planner.estimateCostForCacheSet(rootId, graph, bucket.bucket.regions, bucket.bucket.inputSlices, {}, regionStates[bucket.key], doSaturate);
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
    nodeROIs.resize(cacheableNodes.size());

    ProgressTimer timer(cacheableNodes.size(), "Evaluating nodes for caching (Parallel): ");
    std::mutex timerMtx;

    // Determine number of threads
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0)
        numThreads = 4;

    auto worker = [&](size_t startIdx, size_t endIdx)
    {
        for (size_t i = startIdx; i < endIdx; ++i)
        {
            uint32_t nodeId = cacheableNodes[i];
            float testScore = 0.0f;

            for (const BucketPlanRequest &bucket : buckets)
            {
                // The planner and graph are read-only here
                float cost = planner.estimateCostForCacheSet(
                    rootId, graph, bucket.bucket.regions,
                    bucket.bucket.inputSlices, {nodeId},
                    regionStates.at(bucket.key), doSaturate);

                auto countIt = bucketCallCounts.find(bucket.key);
                uint64_t callCount = (countIt != bucketCallCounts.end()) ? countIt->second : 1;
                testScore += cost * callCount;
            }

            float costSaved = baselineScore - testScore;
            uint64_t sizeBytes = nodeMemorySizes.at(nodeId);
            float roi = sizeBytes > 0 ? (costSaved / static_cast<float>(sizeBytes)) : 0.0f;

            nodeROIs[i] = {nodeId, costSaved, sizeBytes, roi};

            // Update progress timer safely
            {
                std::lock_guard<std::mutex> lock(timerMtx);
                std::cout << nodeId << ": " << std::to_string(testScore) << std::endl;
                timer.tick();
            }
        }
    };

    // Dispatch chunks
    std::vector<std::future<void>> futures;
    size_t chunkSize = (cacheableNodes.size() + numThreads - 1) / numThreads;

    for (unsigned int t = 0; t < numThreads; ++t)
    {
        size_t start = t * chunkSize;
        size_t end = std::min(start + chunkSize, cacheableNodes.size());
        if (start < end)
        {
            futures.push_back(std::async(std::launch::async, worker, start, end));
        }
    }

    // Wait for all workers to finish
    for (auto &f : futures)
        f.get();

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
        CompiledGraph plan = planner.plan(rootId, graph, bucket.bucket.regions, bucket.bucket.inputSlices, selectedCachedNodes, regionStates[bucket.key]);
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
    std::unordered_set<uint32_t> survivingLogicalNodes;

    // First, plan all buckets to get regionStates, but explicitly extract survivors from the full bucket
    for (const BucketPlanRequest &bucket : buckets)
    {
        regionStates[bucket.key] = derivePlanningRegions(rootId, graph, bucket.bucket.regions);

        if (bucket.key == fullKey)
        {
            CompiledGraph plan = planner.plan(rootId, graph, bucket.bucket.regions, bucket.bucket.inputSlices, {}, regionStates[bucket.key], doSaturate);
            for (const auto &kv : plan.physicalToLogicalNodeMap)
            {
                if (kv.second != UINT32_MAX)
                {
                    survivingLogicalNodes.insert(kv.second);
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

    std::unordered_set<uint32_t> selectedCachedNodes;
    uint64_t currentCacheMem = 0;

    for (uint32_t nodeId : candidateNodes)
    {
        uint64_t nodeSize = nodeMemorySizes.at(nodeId);
        if (currentCacheMem + nodeSize <= maxCacheMemory)
        {
            selectedCachedNodes.insert(nodeId);
            currentCacheMem += nodeSize;
            std::cout << "[CacheOptimizer] Selected Node " << nodeId
                      << " | Size: " << nodeSize << " bytes" << std::endl;
        }
    }

    std::cout << "[CacheOptimizer] Final fusion-based cache set: " << selectedCachedNodes.size()
              << " nodes, using " << currentCacheMem << " bytes." << std::endl;

    std::unordered_map<std::string, CompiledGraph> plans;
    float finalScore = 0.0f;

    std::cout << "[CacheOptimizer] Generating plans for " << buckets.size() << " buckets..." << std::endl;
    for (const BucketPlanRequest &bucket : buckets)
    {
        CompiledGraph plan = planner.plan(rootId, graph, bucket.bucket.regions, bucket.bucket.inputSlices, selectedCachedNodes, regionStates[bucket.key], doSaturate);

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

/**
 * Alternative Heuristic: Selects the largest nodes (by bytes) first.
 * Useful for fast initialization when simulation is too slow.
 */
CacheOptimizationResult optimizeCacheBySize(
    uint32_t rootId,
    Graph &graph,
    const std::vector<BucketPlanRequest> &buckets,
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

    std::cout << "[CacheOptimizer] Sorting " << cacheableNodes.size() << " nodes by size (bytes)..." << std::endl;

    // Sort nodes by size descending
    std::vector<uint32_t> sortedNodes = cacheableNodes;
    std::sort(sortedNodes.begin(), sortedNodes.end(), [&](uint32_t a, uint32_t b)
              {
                  uint64_t sizeA = nodeMemorySizes.at(a);
                  uint64_t sizeB = nodeMemorySizes.at(b);
                  if (sizeA != sizeB)
                      return sizeA > sizeB;
                  return a < b; // stable tie-break
              });

    std::unordered_set<uint32_t> selectedCachedNodes;
    uint64_t currentCacheMem = 0;

    for (uint32_t nodeId : sortedNodes)
    {
        uint64_t nodeSize = nodeMemorySizes.at(nodeId);
        if (currentCacheMem + nodeSize <= maxCacheMemory)
        {
            selectedCachedNodes.insert(nodeId);
            currentCacheMem += nodeSize;
            std::cout << "[CacheOptimizer] Selected Node " << nodeId
                      << " | Size: " << nodeSize << " bytes" << std::endl;
        }
    }

    std::cout << "[CacheOptimizer] Final size-based cache set: " << selectedCachedNodes.size()
              << " nodes, using " << currentCacheMem << " bytes." << std::endl;

    // We still need to generate the plans for each bucket using this fixed set
    std::unordered_map<std::string, CompiledGraph> plans;
    float finalScore = 0.0f;

    std::cout << "[CacheOptimizer] Generating plans for " << buckets.size() << " buckets..." << std::endl;
    std::unordered_map<std::string, PlanningRegionState> regionStates;
    for (const BucketPlanRequest &bucket : buckets)
    {
        regionStates[bucket.key] = derivePlanningRegions(rootId, graph, bucket.bucket.regions);
        CompiledGraph plan = planner.plan(rootId, graph, bucket.bucket.regions, bucket.bucket.inputSlices, selectedCachedNodes, regionStates[bucket.key], doSaturate);

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