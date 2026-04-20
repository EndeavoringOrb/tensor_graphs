#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/cost_model.hpp"
#include "core/planner.hpp"
#include "core/executor.hpp"
#include "core/shapes.hpp"
#include "core/cache_optimizer.hpp"
#include <unordered_map>
#include <memory>
#include <string>
#include <algorithm>
#include <cstring>
#include <set>
#include <queue>
#include <filesystem>

namespace dirty_cache_json
{
    inline json dimToJson(const Dim &d)
    {
        return json::array({d.start, d.stop});
    }

    inline Dim dimFromJson(const json &j)
    {
        return {j[0].get<uint32_t>(), j[1].get<uint32_t>()};
    }

    inline json regionToJson(const Region &r)
    {
        json arr = json::array();
        for (const auto &d : r.region)
        {
            arr.push_back(dimToJson(d));
        }
        return arr;
    }

    inline Region regionFromJson(const json &j)
    {
        Region r;
        for (const auto &dimJson : j)
        {
            r.region.push_back(dimFromJson(dimJson));
        }
        return r;
    }

    inline json regionsToJson(const std::vector<Region> &regions)
    {
        json arr = json::array();
        for (const auto &r : regions)
        {
            arr.push_back(regionToJson(r));
        }
        return arr;
    }

    inline std::vector<Region> regionsFromJson(const json &j)
    {
        std::vector<Region> regions;
        for (const auto &rj : j)
        {
            regions.push_back(regionFromJson(rj));
        }
        return regions;
    }

    inline json bucketToJson(const DirtyBucket &bucket)
    {
        json obj;

        json regionsObj;
        for (const auto &pair : bucket.regions)
        {
            regionsObj[std::to_string(pair.first)] = regionsToJson(normalizeRegions(pair.second));
        }
        obj["regions"] = regionsObj;

        json slicesObj;
        for (const auto &pair : bucket.inputSlices)
        {
            json perNode = json::array();
            for (const auto &perParent : pair.second)
            {
                json perParentRegions = regionsToJson(normalizeRegions(perParent));
                perNode.push_back(perParentRegions);
            }
            slicesObj[std::to_string(pair.first)] = perNode;
        }
        obj["input_slices"] = slicesObj;

        return obj;
    }

    inline DirtyBucket bucketFromJson(const json &obj)
    {
        DirtyBucket bucket;

        for (auto it = obj["regions"].begin(); it != obj["regions"].end(); ++it)
        {
            uint32_t nodeId = std::stoul(it.key());
            bucket.regions[nodeId] = regionsFromJson(it.value());
        }

        for (auto it = obj["input_slices"].begin(); it != obj["input_slices"].end(); ++it)
        {
            uint32_t nodeId = std::stoul(it.key());
            std::vector<std::vector<Region>> perNode;
            for (const auto &perParent : it.value())
            {
                perNode.push_back(regionsFromJson(perParent));
            }
            bucket.inputSlices[nodeId] = perNode;
        }

        return bucket;
    }
}

struct ManualBucket
{
    std::unordered_map<uint32_t, std::vector<Region>> inputDirtyRegions;
    std::vector<Region> outputNeededRegion;
};

class Session
{
private:
    static constexpr uint32_t kCacheFileVersion = 2;

    Graph &graph;
    MemoryManager &memManager;
    CostModel costModel;
    std::unique_ptr<Executor> executor;
    uint32_t rootId;
    bool isPlanned;
    bool isCompiled;
    uint32_t nBucketSizes = 0;
    uint64_t maxCacheMemory;
    std::vector<ManualBucket> manualBuckets;

    std::string cachePath;
    std::unordered_map<std::string, CompiledGraph> cachedGraphs;
    std::unordered_map<std::string, DirtyBucket> cachedBuckets;
    std::unordered_map<uint32_t, Backend> selectedCachedNodes;

    // Bucket call tracking for cache optimization
    std::unordered_map<std::string, uint64_t> bucketCallCounts;
    std::string bucketCountsPath = "benchmarks/bucket_counts.json";
    std::string recordsPath = "benchmarks/records.jsonl";

    std::unordered_map<uint32_t, std::vector<uint8_t>> previousInputData;

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
            const TensorNode &node = pair.second;
            if (node.opType != OpType::INPUT)
                continue;
            if (graph.weightSources.count(node.id) != 0 || graph.constantStaging.count(node.id) != 0)
                continue;
            inputNodeIds.push_back(node.id);
        }

        std::sort(inputNodeIds.begin(), inputNodeIds.end());
        return inputNodeIds;
    }

    std::vector<uint32_t> buildAtomicTopoAndInferShapes()
    {
        std::vector<uint32_t> atomicTopo;
        std::unordered_set<uint32_t> visited;
        auto visit = [&](auto &self, uint32_t nodeId) -> void
        {
            if (visited.count(nodeId))
                return;
            visited.insert(nodeId);
            if (graph.hasNode(nodeId))
            {
                for (uint32_t parentId : graph.getNode(nodeId).parentIds)
                    self(self, parentId);
            }
            atomicTopo.push_back(nodeId);
        };
        visit(visit, rootId);

        ShapePropagator prop;
        for (uint32_t nodeId : atomicTopo)
        {
            if (!graph.hasNode(nodeId))
                continue;

            prop.inferShape(nodeId, graph);
        }

        return atomicTopo;
    }

    void materializePersistentInputsForPlan(const CompiledGraph &compiled)
    {
        for (const auto &nodePair : compiled.nodesMap)
        {
            const TensorNode &node = nodePair.second;
            uint32_t logicalId = compiled.getLogicalId(node.id);
            if (node.opType != OpType::INPUT || node.storageType != StorageType::PERSISTENT)
                continue;
            if (memManager.has(node.backend, logicalId))
                continue;

            uint64_t sizeBytes = getSizeBytes(node.getShape(), node.dtype);
            memManager.allocate(node.backend, logicalId, sizeBytes, StorageType::PERSISTENT);

            if (graph.constantStaging.count(node.id))
            {
                memManager.write(node.backend, logicalId, graph.constantStaging[node.id].data(), sizeBytes);
            }
            else if (graph.weightSources.count(node.id))
            {
                const auto &source = graph.weightSources.at(node.id);
                auto &loader = graph.loaders.at(source.first);
                std::vector<uint8_t> temp(sizeBytes);
                loader->loadTensor(source.second, temp.data(), sizeBytes);
                memManager.write(node.backend, logicalId, temp.data(), sizeBytes);
            }
        }
    }

    std::vector<BucketPlanRequest> enumerateCanonicalBuckets(const std::vector<uint32_t> &inputNodeIds)
    {
        std::cout << "[Session.ensureCacheCoverage] Enumerating canonical dirty buckets..." << std::endl;

        std::vector<uint32_t> atomicTopo = buildAtomicTopoAndInferShapes();
        std::unordered_map<std::string, DirtyBucket> bucketByKey;

        // 1. Always include the "Full" bucket (all inputs dirty, full output needed)
        {
            std::unordered_map<uint32_t, std::vector<Region>> fullInputRegions;
            for (uint32_t nodeId : inputNodeIds)
            {
                fullInputRegions[nodeId] = {makeFull(graph.getNode(nodeId).getShape())};
            }
            const std::string key = encodeCacheKey(fullInputRegions);

            std::unordered_map<uint32_t, std::vector<Region>> dirtyOutputRegions = fullInputRegions;
            dirtyOutputRegions[rootId] = {makeFull(graph.getNode(rootId).getShape())};
            std::unordered_map<uint32_t, std::vector<std::vector<Region>>> dirtyInputRegions;

            Graph forwardGraph = graph;
            propagateDirtyRegionsAtomic(atomicTopo, forwardGraph, dirtyOutputRegions, dirtyInputRegions);

            DirtyBucket bucket;
            bucket.regions = std::move(dirtyOutputRegions);
            bucket.inputSlices = std::move(dirtyInputRegions);
            bucketByKey[key] = std::move(bucket);
        }

        if (!manualBuckets.empty())
        {
            for (const auto &manual : manualBuckets)
            {
                const std::string key = encodeCacheKey(manual.inputDirtyRegions);
                if (bucketByKey.find(key) != bucketByKey.end())
                    continue;

                std::unordered_map<uint32_t, std::vector<Region>> dirtyOutputRegions = manual.inputDirtyRegions;
                dirtyOutputRegions[rootId] = manual.outputNeededRegion;
                std::unordered_map<uint32_t, std::vector<std::vector<Region>>> dirtyInputRegions;

                Graph forwardGraph = graph;
                propagateDirtyRegionsAtomic(atomicTopo, forwardGraph, dirtyOutputRegions, dirtyInputRegions);

                DirtyBucket bucket;
                bucket.regions = std::move(dirtyOutputRegions);
                bucket.inputSlices = std::move(dirtyInputRegions);
                bucketByKey[key] = std::move(bucket);
            }
        }
        else
        {
            struct InputOption
            {
                uint32_t nodeId;
                std::vector<std::vector<Dim>> dimSlices;
            };

            std::vector<InputOption> inputOptions;

            for (uint32_t nodeId : inputNodeIds)
            {
                if (graph.getNode(nodeId).opType != OpType::INPUT)
                    continue;

                InputOption option;
                option.nodeId = nodeId;
                for (uint32_t dimLen : graph.getNode(nodeId).getShape())
                {
                    option.dimSlices.push_back(generateSlicesForDim(dimLen, nBucketSizes));
                }
                inputOptions.push_back(std::move(option));
            }

            if (inputOptions.empty())
            {
                DirtyBucket bucket;
                bucket.regions[rootId] = makeFull(graph.getNode(rootId).getShape());
                return {{"", bucket}};
            }

            struct InputRegionSet
            {
                uint32_t nodeId;
                std::vector<std::vector<Region>> options;
            };

            std::vector<InputRegionSet> perInput;
            for (const InputOption &opt : inputOptions)
            {
                InputRegionSet regionSet;
                regionSet.nodeId = opt.nodeId;

                std::vector<Region> current = {Region{}};
                for (const auto &dimSlices : opt.dimSlices)
                {
                    std::vector<Region> next;
                    for (const Region &existing : current)
                    {
                        for (const Dim &slice : dimSlices)
                        {
                            Region region = existing;
                            region.region.push_back(slice);
                            next.push_back(region);
                        }
                    }
                    current = std::move(next);
                }

                regionSet.options.push_back({});
                for (const Region &region : current)
                {
                    regionSet.options.push_back({region});
                }

                std::cout << "[Session.ensureCacheCoverage] input node " << regionSet.nodeId
                          << " has " << regionSet.options.size() << " buckets (including clean state)." << std::endl;
                perInput.push_back(std::move(regionSet));
            }

            std::vector<size_t> indices(perInput.size(), 0);
            std::vector<size_t> sizes;
            size_t totalSize = 1;
            for (const auto &regionSet : perInput)
            {
                sizes.push_back(regionSet.options.size());
                totalSize *= regionSet.options.size();
            }

            ProgressTimer timer(totalSize, "Caching dirty region propagation: ");

            while (true)
            {
                timer.tick();

                std::unordered_map<uint32_t, std::vector<Region>> inputRegions;
                for (size_t i = 0; i < perInput.size(); ++i)
                {
                    inputRegions[perInput[i].nodeId] = perInput[i].options[indices[i]];
                }

                const std::string key = encodeCacheKey(inputRegions);
                if (bucketByKey.find(key) == bucketByKey.end())
                {
                    Graph forwardGraph = graph;
                    std::unordered_map<uint32_t, std::vector<Region>> dirtyOutputRegions = inputRegions;
                    std::unordered_map<uint32_t, std::vector<std::vector<Region>>> dirtyInputRegions;
                    propagateDirtyRegionsAtomic(atomicTopo, forwardGraph, dirtyOutputRegions, dirtyInputRegions);

                    DirtyBucket bucket;
                    bucket.regions = std::move(dirtyOutputRegions);
                    bucket.inputSlices = std::move(dirtyInputRegions);
                    bucketByKey.emplace(key, std::move(bucket));
                }

                int position = static_cast<int>(indices.size()) - 1;
                while (position >= 0)
                {
                    indices[position]++;
                    if (indices[position] < sizes[position])
                        break;
                    indices[position] = 0;
                    position--;
                }

                if (position < 0)
                    break;
            }
        }

        std::vector<std::string> orderedKeys;
        orderedKeys.reserve(bucketByKey.size());
        for (const auto &kv : bucketByKey)
            orderedKeys.push_back(kv.first);
        std::sort(orderedKeys.begin(), orderedKeys.end());

        std::vector<BucketPlanRequest> buckets;
        buckets.reserve(orderedKeys.size());
        for (const std::string &key : orderedKeys)
        {
            buckets.push_back({key, bucketByKey.at(key)});
        }
        return buckets;
    }

    void persistCache() const
    {
        if (cachePath.empty())
            return;

        ensureOutputDirectories();

        std::ofstream file(cachePath, std::ios::trunc);
        if (!file.is_open())
            return;

        json cachedNodesJson = json::object();
        for (const auto &kv : selectedCachedNodes)
        {
            cachedNodesJson[std::to_string(kv.first)] = kv.second;
        }

        json metadata;
        metadata["type"] = "metadata";
        metadata["cacheVersion"] = kCacheFileVersion;
        metadata["rootId"] = rootId;
        metadata["selectedCachedNodes"] = cachedNodesJson;
        file << metadata.dump() << "\n";

        std::vector<std::string> keys;
        keys.reserve(cachedGraphs.size());
        for (const auto &pair : cachedGraphs)
        {
            if (cachedBuckets.find(pair.first) != cachedBuckets.end())
                keys.push_back(pair.first);
        }
        std::sort(keys.begin(), keys.end());

        for (const std::string &key : keys)
        {
            json bucketEntry;
            bucketEntry["type"] = "compiled_bucket";
            bucketEntry["key"] = key;
            to_json(bucketEntry["graph"], cachedGraphs.at(key));
            bucketEntry["bucket"] = dirty_cache_json::bucketToJson(cachedBuckets.at(key));
            file << bucketEntry.dump() << "\n";
        }

        json entry;
        entry["type"] = "constants";

        json constantsObj = json::object();
        std::unordered_set<uint32_t> neededConstants;
        for (const auto &pair : cachedGraphs)
        {
            for (const auto &nodePair : pair.second.nodesMap)
            {
                uint32_t logicalId = pair.second.getLogicalId(nodePair.first);
                if (logicalId != UINT32_MAX && graph.constantStaging.count(logicalId))
                {
                    neededConstants.insert(logicalId);
                }
            }
        }

        std::vector<uint32_t> orderedConstants(neededConstants.begin(), neededConstants.end());
        std::sort(orderedConstants.begin(), orderedConstants.end());
        for (uint32_t logicalId : orderedConstants)
        {
            constantsObj[std::to_string(logicalId)] = graph.constantStaging.at(logicalId);
        }

        entry["constants"] = constantsObj;
        file << entry.dump() << "\n";
    }

public:
    void addManualBucket(const std::unordered_map<uint32_t, std::vector<Region>> &inputDirtyRegions, const std::vector<Region> &outputNeededRegion)
    {
        manualBuckets.push_back({inputDirtyRegions, outputNeededRegion});
    }

    Session(Graph &g, MemoryManager &mem, uint32_t root, const std::string &cacheFile = "", uint32_t _nBucketSizes = 0, uint64_t _maxCacheMemory = std::numeric_limits<uint64_t>::max())
        : graph(g), memManager(mem), rootId(root), isPlanned(false), isCompiled(false), cachePath(cacheFile), nBucketSizes(_nBucketSizes), maxCacheMemory(_maxCacheMemory)
    {
        ensureOutputDirectories();
        loadCache();
        loadBucketCounts();
    }

    static uint32_t nextPowerOf2(uint32_t x)
    {
        if (x == 0)
            return 1;
        x--;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return x + 1;
    }

    void compile(const std::unordered_map<uint32_t, const void *> &sampleInputs)
    {
        (void)sampleInputs;
        ensureOutputDirectories();
        costModel.load(recordsPath);
        if (isPlanned)
        {
            std::cout << "[Session.compile] Using cached compilation." << std::endl;
        }
        else
        {
            std::cout << "[Session.compile] Planning new execution graph..." << std::endl;
            ensureCacheCoverage(collectInputNodeIds());
            isPlanned = true;
        }

        std::cout << "[Session.compile] Materializing persistent memory..." << std::endl;
        memManager.init();

        // Prune the materialization step to only touch nodes actually used in any CompiledGraph
        std::unordered_set<uint32_t> countSet;
        for (const auto &pair : cachedGraphs)
        {
            for (const auto &nodePair : pair.second.nodesMap)
            {
                const TensorNode &node = nodePair.second;

                uint32_t logicalId = pair.second.getLogicalId(node.id);

                if (node.opType == OpType::INPUT && (node.storageType == StorageType::PERSISTENT || node.storageType == StorageType::PINNED))
                {
                    uint32_t memId = (logicalId != UINT32_MAX) ? logicalId : node.id;
                    countSet.insert(memId);
                }
            }
        }

        ProgressTimer timer(countSet.size(), "");
        std::unordered_set<uint32_t> materialized;

        for (const auto &pair : cachedGraphs)
        {
            for (const auto &nodePair : pair.second.nodesMap)
            {
                const TensorNode &node = nodePair.second;
                uint32_t physId = node.id;
                uint32_t logicalId = pair.second.getLogicalId(physId);

                if (node.opType == OpType::INPUT && (node.storageType == StorageType::PERSISTENT || node.storageType == StorageType::PINNED))
                {
                    uint32_t memId = (logicalId != UINT32_MAX) ? logicalId : physId;

                    if (materialized.insert(memId).second)
                    {
                        timer.tick();
                        uint64_t sizeBytes = getSizeBytes(node.getShape(), node.dtype);

                        uint64_t offset = memManager.allocate(node.backend, memId, sizeBytes, node.storageType);

                        if (logicalId != UINT32_MAX && graph.constantStaging.count(logicalId))
                        {
                            memManager.write(node.backend, memId, graph.constantStaging.at(logicalId).data(), sizeBytes);
                        }
                        else if (pair.second.constantStaging.count(physId))
                        {
                            memManager.write(node.backend, memId, pair.second.constantStaging.at(physId).data(), sizeBytes);
                        }
                        else if (logicalId != UINT32_MAX && graph.weightSources.count(logicalId))
                        {
                            const auto &source = graph.weightSources.at(logicalId);
                            auto &loader = graph.loaders.at(source.first);
                            std::vector<uint8_t> temp(sizeBytes);
                            loader->loadTensor(source.second, temp.data(), sizeBytes);
                            memManager.write(node.backend, memId, temp.data(), sizeBytes);
                        }
                    }
                }
            }
        }

        executor = std::make_unique<Executor>(memManager);
        isCompiled = true;
        persistCache();
    }

    std::unordered_map<uint32_t, std::vector<Region>> canonicalizeInputDiffs(
        const std::unordered_map<uint32_t, std::vector<Region>> &inputDiffs) const
    {
        if (inputDiffs.empty())
        {
            return {};
        }

        if (!manualBuckets.empty())
        {
            // Try to find a manual bucket that covers inputDiffs
            for (const auto &manual : manualBuckets)
            {
                bool covers = true;
                for (const auto &diffPair : inputDiffs)
                {
                    uint32_t nodeId = diffPair.first;
                    auto it = manual.inputDirtyRegions.find(nodeId);
                    if (it == manual.inputDirtyRegions.end())
                    {
                        covers = false;
                        break;
                    }
                    if (!coversRegionList(it->second, diffPair.second))
                    {
                        covers = false;
                        break;
                    }
                }
                if (covers)
                {
                    return manual.inputDirtyRegions;
                }
            }

            // Fallback to full bucket
            std::unordered_map<uint32_t, std::vector<Region>> fullInputRegions;
            std::vector<uint32_t> inputNodeIds = collectInputNodeIds();
            for (uint32_t nodeId : inputNodeIds)
            {
                fullInputRegions[nodeId] = {makeFull(graph.getNode(nodeId).getShape())};
            }
            return fullInputRegions;
        }

        std::unordered_map<uint32_t, std::vector<Region>> canonicalRegions;
        for (const auto &pair : inputDiffs)
        {
            uint32_t nodeId = pair.first;
            const auto &regionList = pair.second;

            if (regionList.empty())
                continue;

            const Region &box = regionList[0];
            Region canonical;

            for (size_t d = 0; d < box.region.size(); ++d)
            {
                uint32_t dimLen = graph.getNode(nodeId).getShape()[d];
                uint32_t start = box.region[d].start;
                uint32_t stop = box.region[d].stop;

                uint32_t targetLen = nextPowerOf2(stop - start);
                bool found = false;

                while (targetLen <= nextPowerOf2(dimLen))
                {
                    uint32_t bucketStart = (start / targetLen) * targetLen;
                    uint32_t bucketEnd = std::min(bucketStart + targetLen, dimLen);

                    if (bucketStart <= start && bucketEnd >= stop)
                    {
                        canonical.region.push_back({bucketStart, bucketEnd});
                        found = true;
                        break;
                    }
                    targetLen *= 2;
                }

                if (!found)
                {
                    canonical.region.push_back({0, dimLen});
                }
            }

            canonicalRegions[nodeId] = normalizeRegions({canonical});
        }

        return canonicalRegions;
    }

    const void *run(const std::unordered_map<uint32_t, const void *> &inputs)
    {
        if (!isCompiled)
        {
            compile(inputs);
        }

        // Dirty-bucket flow:
        // 1. compare current inputs with the previous invocation
        // 2. canonicalize the dirty regions into bucket keys
        // 3. look up or build the compiled graph for that bucket
        // 4. execute using the bucket's output regions and input slices
        std::unordered_map<uint32_t, std::vector<Region>> inputDiffs;

        for (const auto &pair : inputs)
        {
            uint32_t nodeId = pair.first;
            const void *newData = pair.second;

            if (graph.getNode(nodeId).opType != OpType::INPUT)
                continue;

            const void *oldData = nullptr;
            auto prevIt = previousInputData.find(nodeId);
            if (prevIt != previousInputData.end())
            {
                oldData = prevIt->second.data();
            }

            auto diff = computeInputDiff(oldData, newData, graph.getNode(nodeId).getShape(), graph.getNode(nodeId).dtype);
            std::cout << "Input Diffs " << pair.first << ": " << std::endl;
            for (uint32_t i = 0; i < diff.size(); i++)
            {
                Region &reg = diff[i];
                std::cout << "  " << i << ": " << toString(reg) << std::endl;
            }
            if (!diff.empty())
            {
                inputDiffs[nodeId] = diff;
            }

            uint64_t sizeBytes = getSizeBytes(graph.getNode(nodeId).getShape(), graph.getNode(nodeId).dtype);
            auto &stored = previousInputData[nodeId];
            stored.resize(sizeBytes);
            std::memcpy(stored.data(), newData, sizeBytes);

            memManager.write(graph.getNode(nodeId).backend, nodeId, newData, sizeBytes);
        }

        auto canonicalDiffs = canonicalizeInputDiffs(inputDiffs);
        if (inputDiffs.empty())
        {
            const TensorNode &rootNode = graph.getNode(rootId);
            if (memManager.has(rootNode.backend, rootId))
            {
                return memManager.read(rootNode.backend, rootId);
            }
        }

        for (const auto &pair : graph.nodes)
        {
            const TensorNode &node = pair.second;
            if (node.opType == OpType::INPUT && graph.weightSources.count(node.id) == 0 && graph.constantStaging.count(node.id) == 0)
            {
                if (canonicalDiffs.find(node.id) == canonicalDiffs.end())
                {
                    canonicalDiffs[node.id] = {};
                }
            }
        }

        std::string key = encodeCacheKey(canonicalDiffs);
        const CompiledGraph *compiled = lookupCache(canonicalDiffs);

        auto bucketIt = cachedBuckets.find(key);
        if (compiled == nullptr || bucketIt == cachedBuckets.end())
        {
            Error::throw_err("[Session.run] no compiled bucket available for dirty regions: " + key);
        }

        incrementBucketCount(key);
        saveBucketCounts();
        executor->run(inputs, *compiled, bucketIt->second);

        // get output
        const OpInstruction &lastInst = compiled->instructions[compiled->instructions.size() - 1];
        Backend backend = lastInst.backend;
        uint32_t outLogicalId = compiled->getLogicalId(lastInst.nodeId);
        if (!memManager.has(backend, outLogicalId))
        {
            Error::throw_err("[Session.run] execution output nodeId " + std::to_string(outLogicalId) + " not found in memory");
        }
        TensorNode outNode = compiled->nodesMap.at(lastInst.nodeId);
        outNode.id = outLogicalId;
        TensorView view = memManager.getView(outNode);
        std::cout << "final output view: " << toString(view) << "\n"
                  << std::flush;
        return memManager.buffers.at(backend).arena_ptr + view.baseOffset;
    }

    std::vector<Region> computeInputDiff(
        const void *oldData,
        const void *newData,
        const std::vector<uint32_t> &shape,
        DType dtype) const
    {
        // Uncomment to use baseline full plan for every run
        // Region full;
        // for (uint32_t dim : shape)
        // {
        //     full.region.push_back({0, dim});
        // }
        // return {full};

        if (shape.empty())
            return {};

        uint64_t totalElements = countElements(shape);
        uint64_t elementSize = getDTypeSize(dtype);

        if (oldData == nullptr)
        {
            Region full;
            for (uint32_t dim : shape)
            {
                full.region.push_back({0, dim});
            }
            return {full};
        }

        const uint8_t *oldBytes = static_cast<const uint8_t *>(oldData);
        const uint8_t *newBytes = static_cast<const uint8_t *>(newData);

        std::vector<uint64_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        std::vector<uint32_t> minBounds(shape.size(), UINT32_MAX);
        std::vector<uint32_t> maxBounds(shape.size(), 0);
        bool anyDirty = false;

        for (uint64_t i = 0; i < totalElements; ++i)
        {
            if (std::memcmp(oldBytes + i * elementSize, newBytes + i * elementSize, elementSize) != 0)
            {
                anyDirty = true;
                uint64_t temp = i;
                for (size_t d = 0; d < shape.size(); ++d)
                {
                    uint32_t coord = static_cast<uint32_t>(temp / strides[d]);
                    temp %= strides[d];

                    minBounds[d] = std::min(minBounds[d], coord);
                    maxBounds[d] = std::max(maxBounds[d], coord);
                }
            }
        }

        if (!anyDirty)
            return {};

        Region r;
        for (size_t d = 0; d < shape.size(); ++d)
        {
            r.region.push_back({minBounds[d], maxBounds[d] + 1});
        }

        return {r};
    }

    static std::vector<Dim> generateSlicesForDim(uint32_t dimLen, uint32_t nBucketSizes = 0)
    {
        std::set<std::pair<uint32_t, uint32_t>> unique;
        uint32_t maxSize = 1;
        while (maxSize < dimLen)
            maxSize *= 2;
        uint32_t startSize = 1;
        if (nBucketSizes != 0)
        {
            startSize = maxSize;
            for (int i = 0; i < nBucketSizes; i++)
            {
                startSize /= 2;
            }
            startSize = std::max(startSize, 1U);
        }

        for (uint32_t size = startSize; size <= maxSize; size *= 2)
        {
            for (uint32_t i = 0; i < dimLen; i += size)
            {
                uint32_t end = std::min(i + size, dimLen);
                unique.insert({i, end});
            }
        }

        std::vector<Dim> slices;
        slices.reserve(unique.size());
        for (const auto &p : unique)
        {
            slices.push_back({p.first, p.second});
        }
        return slices;
    }

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

    void ensureCacheCoverage(const std::vector<uint32_t> &inputNodeIds)
    {
        const std::vector<BucketPlanRequest> buckets = enumerateCanonicalBuckets(inputNodeIds);

        cachedGraphs.clear();
        cachedBuckets.clear();
        selectedCachedNodes.clear();

        for (const BucketPlanRequest &bucket : buckets)
        {
            cachedBuckets[bucket.key] = bucket.bucket;
        }

        std::unordered_map<uint32_t, std::vector<Region>> fullInputRegions;
        for (uint32_t inId : inputNodeIds)
        {
            fullInputRegions[inId] = makeFull(graph.getNode(inId).getShape());
        }
        std::string fullKey = encodeCacheKey(fullInputRegions);

        std::cout << "[Session.ensureCacheCoverage] Starting cache optimization..." << std::endl;
        Planner planner(costModel, memManager.getBufferSizes());
        CacheOptimizationResult optResult = optimizeCacheByFusion(
            rootId, graph, buckets, fullKey, bucketCallCounts, maxCacheMemory, planner);

        if (optResult.foundValidCombination)
        {
            selectedCachedNodes = std::move(optResult.bestCachedNodes);
            cachedGraphs = std::move(optResult.bucketPlans);
            std::cout << "[Session.ensureCacheCoverage] Selected " << selectedCachedNodes.size()
                      << " cached nodes with runtime score " << optResult.runtimeScore << std::endl;

            if (cachedGraphs.count(fullKey) == 0)
            {
                Error::throw_err("[Session::ensureCacheCoverage] full key not in generated plans");
            }
        }
        else
        {
            Error::throw_err("[Session.ensureCacheCoverage] Failed to build a valid plan for every bucket.");
        }

        if (cachedGraphs.size() != buckets.size())
        {
            Error::throw_err("[Session.ensureCacheCoverage] Planned " + std::to_string(cachedGraphs.size()) +
                             " buckets, but expected " + std::to_string(buckets.size()) + ".");
        }

        persistCache();
    }

    // TODO: fall back to larger buckets if exact match doesn't exist
    const CompiledGraph *lookupCache(
        const std::unordered_map<uint32_t, std::vector<Region>> &inputRegions) const
    {
        std::string key = encodeCacheKey(inputRegions);
        auto it = cachedGraphs.find(key);
        if (it != cachedGraphs.end())
        {
            std::cout << "found cached graph for input regions\n"
                      << std::flush;
            return &it->second;
        }
        return nullptr;
    }

    void loadCache()
    {
        if (cachePath.empty())
            return;
        std::ifstream file(cachePath);
        if (!file.is_open())
            return;

        std::string line;
        bool hasValidCache = false;
        bool hasInvalidCache = false;
        bool sawConstants = false;
        bool sawMetadata = false;
        std::string invalidCacheReason = "";
        std::unordered_map<std::string, CompiledGraph> tempGraphs;
        std::unordered_map<std::string, DirtyBucket> tempBuckets;
        std::unordered_map<uint32_t, std::vector<uint8_t>> tempStaging;
        std::unordered_map<uint32_t, Backend> tempSelectedCachedNodes;

        while (std::getline(file, line))
        {
            if (line.empty())
                continue;
            json entry;
            try
            {
                entry = json::parse(line);
            }
            catch (const std::exception &e)
            {
                hasInvalidCache = true;
                invalidCacheReason = "[Session.loadCache]: JSON parse error: " + std::string(e.what());
                break;
            }

            std::string type = entry["type"].get<std::string>();

            if (type == "metadata")
            {
                if (!entry.contains("cacheVersion") || !entry.contains("selectedCachedNodes"))
                {
                    hasInvalidCache = true;
                    invalidCacheReason = "Cache metadata is missing versioned fields.";
                    break;
                }

                const uint32_t version = entry["cacheVersion"].get<uint32_t>();
                const uint32_t cachedRootId = entry.contains("rootId") ? entry["rootId"].get<uint32_t>() : UINT32_MAX;
                if (version != kCacheFileVersion)
                {
                    hasInvalidCache = true;
                    invalidCacheReason = "Cache version " + std::to_string(version) +
                                         " does not match expected version " + std::to_string(kCacheFileVersion) + ".";
                    break;
                }
                if (cachedRootId != rootId)
                {
                    hasInvalidCache = true;
                    invalidCacheReason = "Cache rootId " + std::to_string(cachedRootId) +
                                         " does not match session rootId " + std::to_string(rootId) + ".";
                    break;
                }

                tempSelectedCachedNodes.clear();
                if (entry["selectedCachedNodes"].is_object())
                {
                    for (auto it = entry["selectedCachedNodes"].begin(); it != entry["selectedCachedNodes"].end(); ++it)
                    {
                        tempSelectedCachedNodes[std::stoul(it.key())] = it.value().get<Backend>();
                    }
                }
                else
                {
                    hasInvalidCache = true;
                    invalidCacheReason = "Cache metadata selectedCachedNodes format changed.";
                    break;
                }
                sawMetadata = true;
            }
            else if (type == "constants")
            {
                sawConstants = true;
                for (auto it = entry["constants"].begin(); it != entry["constants"].end(); ++it)
                {
                    uint32_t nodeId = std::stoul(it.key());
                    tempStaging[nodeId] = it.value().get<std::vector<uint8_t>>();
                }
            }
            else if (type == "compiled_bucket")
            {
                std::string key = entry["key"].get<std::string>();
                if (!entry.contains("graph") || !entry.contains("bucket"))
                {
                    hasInvalidCache = true;
                    invalidCacheReason = "Compiled bucket entry is missing graph or bucket payload.";
                    break;
                }

                for (const auto &instJson : entry["graph"]["instructions"])
                {
                    if (!instJson.contains("logicalNodeId") ||
                        !instJson.contains("cachedKernelIds"))
                    {
                        hasInvalidCache = true;
                        invalidCacheReason = "Compiled bucket entry is missing region-kernel metadata.";
                        break;
                    }
                }
                if (hasInvalidCache)
                    break;

                CompiledGraph cg;
                from_json(entry["graph"], cg);

                // Verify kernel IDs are still valid
                bool valid = true;
                for (const auto &inst : cg.instructions)
                {
                    if (inst.fullKernelId == 0 || !KernelRegistry::get().hasKernel(inst.fullKernelId))
                    {
                        if (inst.fullKernelId == 0)
                        {
                            invalidCacheReason = "Kernel ID 0 found in cached graph for inst.fullKernelId\n" + toString(inst);
                        }
                        else
                        {
                            invalidCacheReason = "Kernel ID " + std::to_string(inst.fullKernelId) + " not found in kernel registry for inst.fullKernelId\n" + toString(inst);
                        }
                        valid = false;
                        break;
                    }
                    for (uint64_t kid : inst.cachedKernelIds)
                    {
                        if (kid == 0 || !KernelRegistry::get().hasKernel(kid))
                        {
                            if (kid == 0)
                            {
                                invalidCacheReason = "Kernel ID 0 found in cached graph for inst.cachedKernelIds\n" + toString(inst);
                            }
                            else
                            {
                                invalidCacheReason = "Kernel ID " + std::to_string(kid) + " not found in kernel registry for inst.cachedKernelIds\n" + toString(inst);
                            }
                            valid = false;
                            break;
                        }
                    }
                    if (!valid)
                        break;
                }

                if (!valid)
                {
                    hasInvalidCache = true;
                    break;
                }

                tempGraphs[key] = std::move(cg);
                tempBuckets[key] = dirty_cache_json::bucketFromJson(entry["bucket"]);
                hasValidCache = true;
            }
            else
            {
                hasInvalidCache = true;
                invalidCacheReason = "Unknown cache entry type: " + type;
                break;
            }
        }

        if (!sawMetadata && hasValidCache)
        {
            hasInvalidCache = true;
            invalidCacheReason = "Cache file missing versioned metadata entry.";
        }

        if (!sawConstants && hasValidCache)
        {
            hasInvalidCache = true;
            invalidCacheReason = "Cache file missing 'constants' metadata (interrupted compilation).";
        }

        // If the cache contains any mismatch (e.g. from an old build format or UID 0)
        if (hasInvalidCache)
        {
            std::cout << "[Session.loadCache] invalid or stale cache detected. ignoring entire cache to force recompilation." << std::endl;
            std::cout << "[Session.loadCache] invalid cache reason: " << invalidCacheReason << std::endl;
            // Clear the file so we overwrite it instead of appending to a file full of stale entries
            std::ofstream clearFile(cachePath, std::ios::trunc);
            return;
        }

        if (hasValidCache)
        {
            cachedGraphs = std::move(tempGraphs);
            cachedBuckets = std::move(tempBuckets);
            selectedCachedNodes = std::move(tempSelectedCachedNodes);
            isPlanned = true;
            for (const auto &pair : tempStaging)
            {
                graph.constantStaging[pair.first] = pair.second;
            }
        }
    }

    void loadBucketCounts()
    {
        ensureOutputDirectories();
        std::ifstream file(bucketCountsPath);
        if (!file.is_open())
            return;

        try
        {
            json countsObj = json::parse(file);
            for (auto it = countsObj.begin(); it != countsObj.end(); ++it)
            {
                bucketCallCounts[it.key()] = it.value().get<uint64_t>();
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Session.loadBucketCounts] Failed to load bucket counts: " << e.what() << std::endl;
        }
    }

    void saveBucketCounts() const
    {
        ensureOutputDirectories();
        std::ofstream file(bucketCountsPath);
        if (!file.is_open())
            return;

        json countsObj = json::object();
        std::vector<std::string> keys;
        keys.reserve(bucketCallCounts.size());
        for (const auto &pair : bucketCallCounts)
            keys.push_back(pair.first);
        std::sort(keys.begin(), keys.end());

        for (const std::string &key : keys)
        {
            countsObj[key] = bucketCallCounts.at(key);
        }
        file << countsObj.dump(2) << "\n";
    }

    void incrementBucketCount(const std::string &bucketKey)
    {
        bucketCallCounts[bucketKey]++;
    }
};
