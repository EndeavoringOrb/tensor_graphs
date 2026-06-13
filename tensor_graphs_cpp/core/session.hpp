#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/cost_model.hpp"
#include "core/planner.hpp"
#include "core/executor.hpp"
#include "core/shapes.hpp"
#include "core/repo.hpp"
#include <unordered_map>
#include <memory>
#include <string>
#include <algorithm>
#include <cstring>
#include <set>
#include <queue>
#include <filesystem>

static std::string encodeCacheKey(
    const std::unordered_map<uint32_t, std::vector<Region>> &inputRegions)
{
    std::vector<uint32_t> ids;
    ids.reserve(inputRegions.size());
    for (const auto &pair : inputRegions)
    {
        ids.push_back(pair.first);
    }
    std::sort(ids.begin(), ids.end());

    std::stringstream ss;
    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (i > 0)
            ss << ";";
        ss << ids[i] << ":[";

        const auto &regions = inputRegions.at(ids[i]);
        const std::vector<Region> canonicalRegions = normalizeRegions(regions);
        for (size_t r = 0; r < canonicalRegions.size(); ++r)
        {
            if (r > 0)
                ss << ",";
            ss << encodeRegion(canonicalRegions[r]);
        }
        ss << "]";
    }
    return ss.str();
}

struct Session
{
    static constexpr uint32_t kCacheFileVersion = 3;

    Graph &graph;
    MemoryManager &memManager;
    CostModel costModel;
    std::unique_ptr<Executor> executor;
    uint32_t rootId;
    bool isPlanned;
    bool isCompiled;
    uint32_t nBucketSizes = 0;
    std::vector<Bucket> manualBuckets;

    std::string cachePath;
    std::vector<CompiledGraph> cachedGraphs;
    std::unordered_map<uint32_t, Backend> selectedCachedNodes;

    std::unordered_map<std::string, uint64_t> bucketCallCounts;
    std::string bucketCountsPath = "benchmarks/bucket_counts.bin";
    std::string recordsPath = "benchmarks/records.bin";

    uint32_t fullBucketIdx;
    Repo *repo;

    void ensureOutputDirectories() const
    {
        std::filesystem::create_directories("benchmarks");

        if (!bucketCountsPath.empty())
        {
            std::filesystem::path countsParent = std::filesystem::path(bucketCountsPath).parent_path();
            if (!countsParent.empty())
                std::filesystem::create_directories(countsParent);
        }

        if (!cachePath.empty())
        {
            std::filesystem::path cacheParent = std::filesystem::path(cachePath).parent_path();
            if (!cacheParent.empty())
                std::filesystem::create_directories(cacheParent);
        }
    }

    std::vector<uint32_t> collectInputNodeIds() const
    {
        std::vector<uint32_t> inputNodeIds;
        for (const auto &pair : graph.nodes)
        {
            if (pair.second.opType == OpType::INPUT)
                inputNodeIds.push_back(pair.first);
        }

        std::sort(inputNodeIds.begin(), inputNodeIds.end());
        return inputNodeIds;
    }

    void persistCache() const
    {
        if (cachePath.empty())
            return;
        ensureOutputDirectories();
        std::ofstream file(cachePath, std::ios::trunc | std::ios::binary);
        if (!file.is_open())
            return;

        BinaryWriter bw(file);

        bw.write<uint8_t>(0); // Metadata block type
        bw.write<uint32_t>(kCacheFileVersion);
        bw.write<uint32_t>(rootId);
        bw.write(selectedCachedNodes);

        for (const CompiledGraph &g : cachedGraphs)
        {
            bw.write<uint8_t>(1); // Bucket block type
            bw.write(g);
        }

        bw.write<uint8_t>(2); // Constants block type
        std::unordered_set<uint32_t> neededConstants;
        for (const CompiledGraph &g : cachedGraphs)
        {
            for (const auto &nodePair : g.nodesMap)
            {
                uint32_t logicalId = g.getLogicalId(nodePair.first);
                if (logicalId != UINT32_MAX && graph.constantStaging.count(logicalId))
                    neededConstants.insert(logicalId);
            }
        }

        std::vector<uint32_t> orderedConstants(neededConstants.begin(), neededConstants.end());
        std::sort(orderedConstants.begin(), orderedConstants.end());

        bw.write<uint32_t>(static_cast<uint32_t>(orderedConstants.size()));
        for (uint32_t logicalId : orderedConstants)
        {
            bw.write(logicalId);
            bw.write(*graph.constantStaging.at(logicalId));
        }
    }

    void addBucket(const std::unordered_map<uint32_t, std::vector<Region>> &inputDirtyRegions, const std::vector<Region> &outputNeededRegion)
    {
        manualBuckets.push_back({inputDirtyRegions, outputNeededRegion});
    }

    Session(Graph &g, MemoryManager &mem, uint32_t root, const std::string &cacheFile = "", uint32_t _nBucketSizes = 0, Repo *_repo = nullptr)
        : graph(g), memManager(mem), rootId(root), isPlanned(false), isCompiled(false), cachePath(cacheFile), nBucketSizes(_nBucketSizes), repo(_repo)
    {
        ensureOutputDirectories();
        loadCache();
    }

    void ensureFullBucket()
    {
        Bucket bucket;
        bucket.outputNeededRegion = {makeFull(graph.getNode(rootId).getShape())};
        std::vector<uint32_t> inputNodeIds = collectInputNodeIds();
        for (uint32_t nodeId : inputNodeIds)
        {
            bucket.inputDirtyRegions[nodeId] = {makeFull(graph.getNode(nodeId).getShape())};
        }

        bool hasFullBucket = false;
        for (int i = 0; i < manualBuckets.size(); i++)
        {
            const auto &mb = manualBuckets[i];
            if (mb == bucket)
            {
                hasFullBucket = true;
                fullBucketIdx = i;
                break;
            }
        }

        if (!hasFullBucket)
        {
            fullBucketIdx = manualBuckets.size();
            manualBuckets.push_back(bucket);
        }
    }

    void plan(bool doSaturate = true)
    {
        ensureOutputDirectories();
        costModel.load(recordsPath);

        ShapePropagator prop;
        prop.inferShapeRecursive(rootId, graph);

        ensureFullBucket();

        if (isPlanned)
        {
            std::cout << "[Session.compile] Using cached compilation." << std::endl;
        }
        else
        {
            std::cout << "[Session.compile] Planning new execution graph..." << std::endl;
            ensureCacheCoverage(doSaturate);
            persistCache();
            isPlanned = true;
        }
    }

    void compile(bool doSaturate = true)
    {
        plan(doSaturate);

        std::cout << "[Session.compile] Materializing persistent memory..." << std::endl;
        memManager.init();

        std::unordered_set<uint32_t> countSet;
        for (const CompiledGraph &g : cachedGraphs)
        {
            for (const auto &nodePair : g.nodesMap)
            {
                const TensorNode &node = nodePair.second;

                uint32_t logicalId = g.getLogicalId(node.id);

                if ((node.opType == OpType::INPUT || node.opType == OpType::CACHE) && (node.storageType == StorageType::PERSISTENT || node.storageType == StorageType::PINNED))
                {
                    uint32_t memId = (logicalId != UINT32_MAX) ? logicalId : node.id;
                    countSet.insert(memId);
                }
            }
        }

        ProgressTimer timer(countSet.size(), "");
        std::unordered_set<uint32_t> materialized;
        std::unordered_set<uint32_t> written;

        for (const CompiledGraph &g : cachedGraphs)
        {
            for (const auto &nodePair : g.nodesMap)
            {
                const TensorNode &node = nodePair.second;
                uint32_t physId = node.id;
                uint32_t logicalId = g.getLogicalId(physId);

                if ((node.opType == OpType::INPUT || node.opType == OpType::CACHE) && (node.storageType == StorageType::PERSISTENT || node.storageType == StorageType::PINNED))
                {
                    uint32_t memId = (logicalId != UINT32_MAX) ? logicalId : physId;

                    uint64_t sizeBytes = countElements(node.getShape()) * getDTypeSize(node.dtype);

                    if (materialized.insert(memId).second)
                    {
                        timer.tick();
                        if (node.backend != Backend::STORAGE)
                        {
                            uint64_t offset = memManager.allocate(node.backend, memId, sizeBytes, node.storageType);
                        }
                    }

                    if (written.find(memId) == written.end())
                    {
                        if (logicalId != UINT32_MAX && graph.constantStaging.count(logicalId))
                        {
                            memManager.write(node.backend, memId, graph.constantStaging.at(logicalId)->data(), sizeBytes);
                            written.insert(memId);
                        }
                        else if (g.constantStaging.count(physId))
                        {
                            memManager.write(node.backend, memId, g.constantStaging.at(physId)->data(), sizeBytes);
                            written.insert(memId);
                        }
                    }
                }
            }
        }

        executor = std::make_unique<Executor>(memManager);
        isCompiled = true;
    }

    const void *run(Bucket bucket = {}, std::function<void(uint32_t, const TensorView &, const void *)> debugCallback = nullptr,
                    bool doSaturate = true)
    {
        if (!isCompiled)
        {
            compile(doSaturate);
        }

        if (bucket.inputDirtyRegions.empty())
        {
            for (const auto &pair : graph.nodes)
            {
                const TensorNode &node = pair.second;
                if (node.opType == OpType::INPUT)
                {
                    bucket.inputDirtyRegions[pair.first] = {makeFull(pair.second.getShape())};
                }
            }
        }
        if (bucket.outputNeededRegion.empty())
        {
            bucket.outputNeededRegion = {makeFull(graph.getNode(rootId).getShape())};
        }

        const uint32_t graphIdx = getBestGraphIdx(bucket);
        std::cout << "[Session.run] chose graph: " << std::to_string(graphIdx) << std::endl;

        ProgressTimer runTimer(0, "", true);
        executor->run(cachedGraphs[graphIdx], debugCallback);
        double elapsed = runTimer.getElapsed();
        std::cout << "[Session.run] execution finished in " << std::to_string(elapsed * 1000) << "ms" << std::endl;

        const OpInstruction &lastInst = cachedGraphs[graphIdx].instructions[cachedGraphs[graphIdx].instructions.size() - 1];
        Backend backend = lastInst.backend;
        uint32_t outLogicalId = cachedGraphs[graphIdx].getLogicalId(lastInst.nodeId);
        if (!memManager.has(backend, outLogicalId))
        {
            Error::throw_err("[Session.run] execution output nodeId " + std::to_string(outLogicalId) + " not found in memory");
        }
        TensorView view = memManager.getView(cachedGraphs[graphIdx].nodesMap.at(lastInst.nodeId), outLogicalId);
        std::cout << "final output view: " << toString(view) << "\n"
                  << std::flush;
        return memManager.buffers.at(backend).arena_ptr + view.baseOffset;
    }

    void ensureCacheCoverage(bool doSaturate)
    {
        cachedGraphs.clear();
        selectedCachedNodes.clear();

        std::cout << "[Session.ensureCacheCoverage] Starting iterative cache optimization..." << std::endl;
        Planner planner(costModel, memManager.getBufferSizes());

        std::unordered_map<uint32_t, Backend> protectedCachedNodes;

        for (size_t i = 0; i < manualBuckets.size(); ++i)
        {
            const Bucket &bucket = manualBuckets[i];

            CompiledGraph plan = planner.plan(
                rootId, graph,
                bucket,
                protectedCachedNodes,
                i == fullBucketIdx ? false : doSaturate,
                false,
                repo);

            for (const auto &pair : plan.nodesMap)
            {
                if (pair.second.opType == OpType::CACHE)
                {
                    protectedCachedNodes[plan.physicalToLogicalNodeMap.at(pair.first)] = pair.second.backend;
                }
            }
        }

        std::cout << "protectedCachedNodes" << std::endl;
        for (const auto &pair : protectedCachedNodes)
        {
            std::cout << pair.first << ": " << pair.second << std::endl;
        }

        std::cout << "[Session.ensureCacheCoverage] Final replanning with " << protectedCachedNodes.size() << " protected eclasses..." << std::endl;
        for (size_t i = 0; i < manualBuckets.size(); ++i)
        {
            const Bucket &bucket = manualBuckets[i];
            CompiledGraph plan = planner.plan(
                rootId, graph,
                bucket,
                protectedCachedNodes,
                doSaturate,
                true,
                repo);
            plan.bucket = bucket;
            cachedGraphs.push_back(plan);
        }

        selectedCachedNodes = std::move(protectedCachedNodes);

        if (cachedGraphs.size() != manualBuckets.size())
        {
            Error::throw_err("[Session.ensureCacheCoverage] Planned " + std::to_string(cachedGraphs.size()) +
                             " buckets, but expected " + std::to_string(manualBuckets.size()) + ".");
        }

        persistCache();
    }

    const uint32_t getBestGraphIdx(
        const Bucket &bucket) const
    {
        uint32_t bestIdx = UINT32_MAX;
        float bestCost = std::numeric_limits<float>::max();
        for (int i = 0; i < cachedGraphs.size(); i++)
        {
            const CompiledGraph &g = cachedGraphs[i];
            bool valid = true;
            for (const auto &inputPair : bucket.inputDirtyRegions)
            {
                if (g.bucket.inputDirtyRegions.count(inputPair.first) == 0)
                {
                    valid = false;
                    break;
                }
                for (const Region &inputRegion : inputPair.second)
                {
                    bool contains = false;
                    for (const Region &gRegion : g.bucket.inputDirtyRegions.at(inputPair.first))
                    {
                        contains = contains || inputRegion <= gRegion;
                    }
                    valid = valid && contains;
                    if (!valid)
                        break;
                }
                if (!valid)
                    break;
            }
            for (const Region &outputRegion : bucket.outputNeededRegion)
            {
                bool contains = false;
                for (const Region &gRegion : g.bucket.outputNeededRegion)
                {
                    contains = contains || outputRegion <= gRegion;
                }
                valid = valid && contains;
                if (!valid)
                    break;
            }
            if (!valid)
                continue;
            const float gCost = g.cost();
            if (gCost < bestCost)
            {
                bestIdx = i;
                bestCost = gCost;
            }
        }
        if (bestIdx == UINT32_MAX)
        {
            Error::throw_err("[Session.getBestGraphIdx] couldn't find graph for input diffs");
        }
        return bestIdx;
    }

    void loadCache()
    {
        if (cachePath.empty())
            return;
        std::ifstream file(cachePath, std::ios::binary);
        if (!file.is_open())
            return;

        BinaryReader br(file);
        bool hasInvalidCache = false;
        std::string invalidCacheReason = "";
        std::vector<CompiledGraph> tempGraphs;
        std::unordered_map<uint32_t, Backend> tempSelectedCachedNodes;

        while (file.peek() != EOF)
        {
            uint8_t type;
            br.read(type);

            if (type == 0)
            {
                uint32_t version;
                uint32_t cachedRootId;
                br.read(version);
                br.read(cachedRootId);
                br.read(tempSelectedCachedNodes);

                if (version != kCacheFileVersion || cachedRootId != rootId)
                {
                    hasInvalidCache = true;
                    invalidCacheReason = "Version or RootId mismatch";
                    break;
                }
            }
            else if (type == 1)
            {
                CompiledGraph cg;
                br.read(cg);

                bool valid = true;
                for (const auto &inst : cg.instructions)
                {
                    if (inst.fullKernelId == 0 || !KernelRegistry::get().hasKernel(inst.fullKernelId))
                    {
                        valid = false;
                        break;
                    }
                    for (uint64_t kid : inst.cachedKernelIds)
                        if (kid == 0 || !KernelRegistry::get().hasKernel(kid))
                        {
                            valid = false;
                            break;
                        }
                    if (!valid)
                        break;
                }
                if (!valid)
                {
                    hasInvalidCache = true;
                    invalidCacheReason = "Invalid Kernel ID";
                    break;
                }
                cg.remapPhysIds();
                tempGraphs.push_back(cg);
            }
            else if (type == 2)
            {
                uint32_t count;
                br.read(count);
                for (uint32_t i = 0; i < count; ++i)
                {
                    uint32_t nodeId;
                    std::vector<uint8_t> data;
                    br.read(nodeId);
                    br.read(data);
                    graph.constantStaging[nodeId] = std::make_shared<std::vector<uint8_t>>(std::move(data));
                }
            }
            else
            {
                hasInvalidCache = true;
                invalidCacheReason = "Unknown block type";
                break;
            }
        }

        if (hasInvalidCache)
        {
            std::cout << "[Session.loadCache] invalid cache: " << invalidCacheReason << std::endl;
            std::ofstream clearFile(cachePath, std::ios::trunc | std::ios::binary);
            return;
        }

        if (!tempGraphs.empty())
        {
            cachedGraphs = std::move(tempGraphs);
            selectedCachedNodes = std::move(tempSelectedCachedNodes);
            isPlanned = true;
        }
    }
};