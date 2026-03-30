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

struct BucketPlanMemoEntry
{
    bool attempted = false;
    bool valid = false;
    bool memoryFailed = false;
    CompiledGraph plan;
    float runtime = std::numeric_limits<float>::infinity();
};

static uint32_t getCompiledLogicalNodeId(const CompiledGraph &plan, const OpInstruction &inst)
{
    if (inst.logicalNodeId != UINT32_MAX)
        return inst.logicalNodeId;

    auto physIt = plan.physicalToLogicalNodeMap.find(inst.nodeId);
    if (physIt != plan.physicalToLogicalNodeMap.end())
        return physIt->second;

    return inst.nodeId;
}

static uint32_t getCompiledLogicalNodeId(const CompiledGraph &plan, uint32_t physicalNodeId)
{
    auto nodeIt = plan.nodesMap.find(physicalNodeId);
    if (nodeIt == plan.nodesMap.end())
        return physicalNodeId;

    auto physIt = plan.physicalToLogicalNodeMap.find(physicalNodeId);
    if (physIt != plan.physicalToLogicalNodeMap.end())
        return physIt->second;

    return physicalNodeId;
}

static std::unordered_map<Backend, uint64_t> calculatePlanPeakMemoryByBackend(
    const CompiledGraph &plan,
    const std::unordered_set<uint32_t> &cachedNodes)
{
    std::unordered_map<Backend, uint64_t> currentMemByBackend;
    std::unordered_map<Backend, uint64_t> peakMemByBackend;
    std::unordered_map<uint32_t, uint32_t> uses = plan.refCounts;

    for (const OpInstruction &inst : plan.instructions)
    {
        const TensorNode &node = plan.nodesMap.at(inst.nodeId);
        const uint32_t logicalNodeId = getCompiledLogicalNodeId(plan, inst);
        const Backend backend = node.backend;

        if (cachedNodes.count(logicalNodeId) == 0)
        {
            currentMemByBackend[backend] += getSizeBytes(node.shape, node.dtype);
            peakMemByBackend[backend] = std::max(peakMemByBackend[backend], currentMemByBackend[backend]);
        }

        for (uint32_t parentId : node.parentIds)
        {
            auto useIt = uses.find(parentId);
            if (useIt == uses.end() || useIt->second == 0)
                continue;

            useIt->second--;
            if (useIt->second != 0)
                continue;

            const TensorNode &parentNode = plan.nodesMap.at(parentId);
            const uint32_t logicalParentId = getCompiledLogicalNodeId(plan, parentId);
            if (cachedNodes.count(logicalParentId) != 0)
                continue;

            uint64_t sizeBytes = getSizeBytes(parentNode.shape, parentNode.dtype);
            if (currentMemByBackend[parentNode.backend] >= sizeBytes)
                currentMemByBackend[parentNode.backend] -= sizeBytes;
            else
                currentMemByBackend[parentNode.backend] = 0;
        }
    }

    return peakMemByBackend;
}

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
        if (node.shape.empty())
            continue;
        residentByBackend[node.backend] += getSizeBytes(node.shape, node.dtype);
    }
    return residentByBackend;
}

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
        if (node.shape.empty())
            continue;
        nodeMemorySizes[nodeId] = getSizeBytes(node.shape, node.dtype);
    }
    return nodeMemorySizes;
}

static std::string encodeCachedNodeSet(const std::unordered_set<uint32_t> &cachedNodes)
{
    std::vector<uint32_t> ids(cachedNodes.begin(), cachedNodes.end());
    std::sort(ids.begin(), ids.end());

    std::stringstream ss;
    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (i > 0)
            ss << ",";
        ss << ids[i];
    }
    return ss.str();
}

static float estimatePlanRuntime(const CompiledGraph &plan, CostModel &costModel)
{
    (void)costModel;
    float totalCost = 0.0f;
    for (const auto &kv : plan.nodeCosts)
    {
        totalCost += kv.second;
    }
    return totalCost;
}

static float calculateWeightedRuntimeScore(
    const std::unordered_map<std::string, CompiledGraph> &plans,
    const std::unordered_map<std::string, uint64_t> &bucketCallCounts,
    CostModel &costModel)
{
    if (plans.empty())
        return std::numeric_limits<float>::infinity();

    double totalScore = 0.0;
    for (const auto &kv : plans)
    {
        auto countIt = bucketCallCounts.find(kv.first);
        uint64_t callCount = (countIt != bucketCallCounts.end()) ? countIt->second : 1;
        totalScore += static_cast<double>(estimatePlanRuntime(kv.second, costModel)) * static_cast<double>(callCount);
    }
    return static_cast<float>(totalScore);
}

static size_t findUpperBound(
    const std::vector<uint32_t> &cacheableNodes,
    const std::unordered_map<uint32_t, uint64_t> &nodeMemorySizes,
    const Graph &graph,
    const std::unordered_map<Backend, uint64_t> &memoryLimits)
{
    if (cacheableNodes.empty())
        return 0;

    std::unordered_map<Backend, std::vector<uint64_t>> sizesByBackend;
    for (uint32_t nodeId : cacheableNodes)
    {
        auto sizeIt = nodeMemorySizes.find(nodeId);
        if (sizeIt == nodeMemorySizes.end())
            continue;
        sizesByBackend[graph.getNode(nodeId).backend].push_back(sizeIt->second);
    }

    size_t upperBound = 0;
    for (const auto &limitPair : memoryLimits)
    {
        std::vector<uint64_t> &sizes = sizesByBackend[limitPair.first];
        std::sort(sizes.begin(), sizes.end());

        uint64_t used = 0;
        for (uint64_t sizeBytes : sizes)
        {
            if (used + sizeBytes > limitPair.second)
                break;
            used += sizeBytes;
            upperBound++;
        }
    }

    return std::min(upperBound, cacheableNodes.size());
}

static size_t findLowerBound(
    const std::vector<BucketPlanRequest> &buckets,
    const std::unordered_map<std::string, CompiledGraph> &baselinePlans,
    const std::vector<uint32_t> &cacheableNodes,
    const std::unordered_map<uint32_t, uint64_t> &nodeMemorySizes,
    const Graph &graph,
    const std::unordered_map<Backend, uint64_t> &memoryLimits)
{
    if (cacheableNodes.empty())
        return 0;

    std::unordered_map<Backend, uint64_t> maxPeaks;
    for (const BucketPlanRequest &bucket : buckets)
    {
        auto planIt = baselinePlans.find(bucket.key);
        if (planIt == baselinePlans.end())
            continue;

        std::unordered_map<Backend, uint64_t> peaks = calculatePlanPeakMemoryByBackend(planIt->second, {});
        for (const auto &peakPair : maxPeaks)
        {
            uint64_t peak = peaks[peakPair.first];
            maxPeaks[peakPair.first] = std::max(
                peakPair.second,
                peak);
        }
    }
    std::unordered_map<Backend, uint64_t> newMemoryLimits;
    for (const auto &limitPair : memoryLimits)
    {
        newMemoryLimits[limitPair.first] = limitPair.second - maxPeaks[limitPair.first];
    }

    return findUpperBound(cacheableNodes, nodeMemorySizes, graph, newMemoryLimits);
}

static bool combinationFitsResidentBudget(
    const std::unordered_map<Backend, uint64_t> &residentByBackend,
    const std::unordered_map<Backend, uint64_t> &memoryLimits)
{
    for (const auto &limitPair : memoryLimits)
    {
        auto residentIt = residentByBackend.find(limitPair.first);
        uint64_t resident = residentIt != residentByBackend.end() ? residentIt->second : 0;
        if (resident > limitPair.second)
            return false;
    }
    return true;
}

static bool combinationCanFitOptimistically(
    const std::unordered_map<Backend, uint64_t> &residentByBackend,
    const std::vector<BucketPlanRequest> &buckets,
    const std::unordered_map<std::string, CompiledGraph> &baselinePlans,
    const std::unordered_map<Backend, uint64_t> &memoryLimits)
{
    if (baselinePlans.empty())
        return true;

    for (const BucketPlanRequest &bucket : buckets)
    {
        auto baselineIt = baselinePlans.find(bucket.key);
        if (baselineIt == baselinePlans.end())
            continue;

        std::unordered_map<Backend, uint64_t> peaks = calculatePlanPeakMemoryByBackend(baselineIt->second, {});
        for (const auto &limitPair : memoryLimits)
        {
            uint64_t resident = residentByBackend.count(limitPair.first) ? residentByBackend.at(limitPair.first) : 0;
            uint64_t optimisticTransient = peaks.count(limitPair.first) ? peaks.at(limitPair.first) : 0;
            if (optimisticTransient > resident)
                optimisticTransient -= resident;
            else
                optimisticTransient = 0;

            if (resident + optimisticTransient > limitPair.second)
                return false;
        }
    }

    return true;
}

static const BucketPlanMemoEntry &getOrPlanBucket(
    uint32_t rootId,
    const Graph &graph,
    const BucketPlanRequest &bucket,
    const std::unordered_set<uint32_t> &cachedNodes,
    const std::unordered_map<Backend, uint64_t> &memoryLimits,
    CostModel &costModel,
    std::unordered_map<std::string, BucketPlanMemoEntry> &memo)
{
    const std::string cacheSetKey = encodeCachedNodeSet(cachedNodes);
    const std::string memoKey = cacheSetKey + "|" + bucket.key;

    auto memoIt = memo.find(memoKey);
    if (memoIt != memo.end())
        return memoIt->second;

    BucketPlanMemoEntry entry;
    entry.attempted = true;

    try
    {
        Graph planningGraph = graph;
        Planner planner(costModel, memoryLimits);
        entry.plan = planner.plan(rootId, planningGraph, bucket.bucket.regions, bucket.bucket.inputSlices, cachedNodes);
        entry.runtime = estimatePlanRuntime(entry.plan, costModel);
        entry.valid = true;
    }
    catch (const MemoryExhaustedError &)
    {
        entry.memoryFailed = true;
    }

    return memo.emplace(memoKey, std::move(entry)).first->second;
}

template <typename Fn>
static void enumerateCombinations(
    const std::vector<uint32_t> &nodes,
    size_t choose,
    size_t start,
    std::vector<uint32_t> &current,
    Fn &&fn)
{
    if (current.size() == choose)
    {
        fn(current);
        return;
    }

    const size_t remaining = choose - current.size();
    for (size_t i = start; i + remaining <= nodes.size(); ++i)
    {
        current.push_back(nodes[i]);
        enumerateCombinations(nodes, choose, i + 1, current, fn);
        current.pop_back();
    }
}

CacheOptimizationResult optimizeCacheCombination(
    uint32_t rootId,
    Graph &graph,
    const std::vector<BucketPlanRequest> &buckets,
    const std::unordered_map<std::string, uint64_t> &bucketCallCounts,
    const std::unordered_map<std::string, CompiledGraph> &baselinePlans,
    const std::unordered_map<Backend, uint64_t> &memoryLimits,
    CostModel &costModel)
{
    CacheOptimizationResult result;
    if (buckets.empty())
        return result;

    const std::vector<uint32_t> cacheableNodes = collectCacheableNodes(graph);
    const std::unordered_map<uint32_t, uint64_t> nodeMemorySizes = buildLogicalNodeMemorySizes(graph, cacheableNodes);

    std::unordered_map<std::string, BucketPlanMemoEntry> memo;
    const std::string emptyCacheKey = encodeCachedNodeSet({});
    for (const BucketPlanRequest &bucket : buckets)
    {
        BucketPlanMemoEntry entry;
        entry.attempted = true;

        auto baselineIt = baselinePlans.find(bucket.key);
        if (baselineIt != baselinePlans.end())
        {
            entry.valid = true;
            entry.plan = baselineIt->second;
            entry.runtime = estimatePlanRuntime(entry.plan, costModel);
        }
        else
        {
            entry.memoryFailed = true;
        }

        memo[emptyCacheKey + "|" + bucket.key] = std::move(entry);
    }

    if (cacheableNodes.empty())
    {
        if (baselinePlans.size() != buckets.size())
            return result;

        result.bestCachedNodes = {};
        result.bucketPlans = baselinePlans;
        result.runtimeScore = calculateWeightedRuntimeScore(baselinePlans, bucketCallCounts, costModel);
        result.foundValidCombination = true;
        return result;
    }

    const size_t lowerBound = findLowerBound(buckets, baselinePlans, cacheableNodes, nodeMemorySizes, graph, memoryLimits);
    const size_t upperBound = findUpperBound(cacheableNodes, nodeMemorySizes, graph, memoryLimits);

    if (lowerBound > upperBound)
        return result;

    std::cout << "[CacheOptimizer] Searching cache combinations: [" << lowerBound << ", " << upperBound << "] nodes" << std::endl;

    for (size_t k = lowerBound; k <= upperBound; ++k)
    {
        std::cout << "[CacheOptimizer] Trying combinations of size " << k << std::endl;
        std::vector<uint32_t> current;
        uint64_t total = binom(cacheableNodes.size(), k);
        ProgressTimer timer(total, "trying combinations ");
        enumerateCombinations(cacheableNodes, k, 0, current, [&](const std::vector<uint32_t> &combination)
                              {
            std::unordered_set<uint32_t> cachedNodes(combination.begin(), combination.end());
            std::unordered_map<Backend, uint64_t> residentByBackend = calculateCachedResidentMemoryByBackend(graph, cachedNodes);

            if (!combinationFitsResidentBudget(residentByBackend, memoryLimits))
                return;

            if (!combinationCanFitOptimistically(residentByBackend, buckets, baselinePlans, memoryLimits))
                return;

            double totalScore = 0.0;
            std::unordered_map<std::string, CompiledGraph> plans;

            for (const BucketPlanRequest &bucket : buckets)
            {
                const BucketPlanMemoEntry &entry = getOrPlanBucket(
                    rootId,
                    graph,
                    bucket,
                    cachedNodes,
                    memoryLimits,
                    costModel,
                    memo);

                if (!entry.valid)
                    return;

                plans[bucket.key] = entry.plan;
                auto countIt = bucketCallCounts.find(bucket.key);
                uint64_t callCount = (countIt != bucketCallCounts.end()) ? countIt->second : 1;
                totalScore += static_cast<double>(entry.runtime) * static_cast<double>(callCount);
            }

            if (!result.foundValidCombination || totalScore < result.runtimeScore)
            {
                std::cout << "[CacheOptimizer] Found better combination with score " << totalScore << std::endl;
                result.runtimeScore = static_cast<float>(totalScore);
                result.bestCachedNodes = std::move(cachedNodes);
                result.bucketPlans = std::move(plans);
                result.foundValidCombination = true;
            } 
            timer.tick(); });
    }

    return result;
}
