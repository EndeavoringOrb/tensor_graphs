#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/cost_model.hpp"
#include "core/planner.hpp"
#include "core/executor.hpp"
#include "core/shapes.hpp"
#include <unordered_map>
#include <memory>
#include <string>
#include <algorithm>
#include <cstring>
#include <set>

// ---------------------------------------------------------------------------
// JSON serialization helpers for Region / Dim / DirtyBucket
// ---------------------------------------------------------------------------

namespace dirty_cache_json // TODO: change case? not sure camelCase or PascalCase
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

    // Serialize a DirtyBucket to JSON
    inline json bucketToJson(const DirtyBucket &bucket)
    {
        json obj;

        // Regions
        json regionsObj;
        for (const auto &pair : bucket.regions)
        {
            regionsObj[std::to_string(pair.first)] = regionsToJson(pair.second);
        }
        obj["regions"] = regionsObj;

        // Input slices
        json slicesObj;
        for (const auto &pair : bucket.inputSlices)
        {
            json perNode = json::array();
            for (const auto &perOutputRegion : pair.second)
            {
                json perParent = json::array();
                for (const auto &parentRegions : perOutputRegion)
                {
                    perParent.push_back(regionsToJson(parentRegions));
                }
                perNode.push_back(perParent);
            }
            slicesObj[std::to_string(pair.first)] = perNode;
        }
        obj["input_slices"] = slicesObj;

        return obj;
    }

    // Deserialize a DirtyBucket from JSON
    inline DirtyBucket bucketFromJson(const json &obj)
    {
        DirtyBucket bucket;

        if (obj.contains("regions"))
        {
            for (auto it = obj["regions"].begin(); it != obj["regions"].end(); ++it)
            {
                uint32_t nodeId = std::stoul(it.key());
                bucket.regions[nodeId] = regionsFromJson(it.value());
            }
        }

        if (obj.contains("input_slices"))
        {
            for (auto it = obj["input_slices"].begin(); it != obj["input_slices"].end(); ++it)
            {
                uint32_t nodeId = std::stoul(it.key());
                std::vector<std::vector<std::vector<Region>>> perNode;
                for (const auto &perOutputRegion : it.value())
                {
                    std::vector<std::vector<Region>> perParent;
                    for (const auto &parentRegions : perOutputRegion)
                    {
                        perParent.push_back(regionsFromJson(parentRegions));
                    }
                    perNode.push_back(perParent);
                }
                bucket.inputSlices[nodeId] = perNode;
            }
        }

        return bucket;
    }
} // namespace dirty_cache_json

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

class Session
{
private:
    Graph &graph;
    MemoryManager &memManager;
    CostModel costModel;
    CompiledGraph compiled;
    std::unique_ptr<Executor> executor;
    uint32_t rootId;
    bool isPlanned;
    bool isCompiled;

    std::string cachePath;
    std::unordered_map<std::string, DirtyBucket> dirtyCache;

    // Map of node ID -> previous input data (for computing diffs)
    std::unordered_map<uint32_t, std::vector<uint8_t>> previousInputData;

public:
    Session(Graph &g, MemoryManager &mem, uint32_t root, const std::string &cacheFile = "")
        : graph(g), memManager(mem), rootId(root), isPlanned(false), isCompiled(false), cachePath(cacheFile)
    {
        loadCache();
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
        costModel.load("benchmarks/records.jsonl");
        if (isPlanned)
        {
            std::cout << "[Session.compile] Using cached compilation." << std::endl;
        }
        else
        {
            std::cout << "[Session.compile] Planning new execution graph..." << std::endl;
            Planner planner(costModel, 4ULL * 1024 * 1024 * 1024);
            compiled = planner.plan(rootId, graph);

            // Save immediately after planning
            saveCompiledGraph(compiled);
            isPlanned = true;
        }

        // --- THE MATERIALIZATION PASS ---
        // This must run even if we load from cache because pointers/DeviceBuffers
        // are process-specific and not persistent.
        std::cout << "[Session.compile] Materializing persistent memory..." << std::endl;
        memManager.buffers.at(Backend::CPU).init();

        for (const auto &pair : compiled.nodesMap)
        {
            uint32_t nodeId = pair.first;
            if (graph.nodes[nodeId].opType == OpType::INPUT && graph.nodes[nodeId].storageType == StorageType::PERSISTENT)
            {
                uint64_t sizeBytes = getSizeBytes(graph.nodes[nodeId].shape, graph.nodes[nodeId].dtype);

                // Physical allocation
                uint64_t offset = memManager.allocate(graph.nodes[nodeId].backend, nodeId, sizeBytes, StorageType::PERSISTENT);

                // Data loading
                if (graph.constantStaging.count(nodeId))
                {
                    memManager.write(graph.nodes[nodeId].backend, nodeId, graph.constantStaging[nodeId].data(), sizeBytes);
                }
                else if (graph.weightSources.count(nodeId))
                {
                    const auto &source = graph.weightSources.at(nodeId);
                    auto &loader = graph.loaders.at(source.first);
                    uint8_t *destPtr = memManager.buffers.at(graph.nodes[nodeId].backend).arena_ptr + offset;
                    loader->loadTensor(source.second, destPtr, sizeBytes);
                }
            }
        }

        // Re-verify cache coverage if needed (or assume persistent cache is sufficient)
        std::vector<uint32_t> inputNodeIds;
        for (const auto &pair : compiled.nodesMap)
        {
            if (pair.second.opType == OpType::INPUT && pair.second.storageType == StorageType::TRANSIENT)
            {
                inputNodeIds.push_back(pair.first);
            }
        }
        ensureCacheCoverage(inputNodeIds);

        executor = std::make_unique<Executor>(compiled, memManager, graph);
        isCompiled = true;
        graph.constantStaging.clear();
    }

    // Snap a raw dirty region to the nearest power-of-2 aligned bucket
    // and look up the cached bucket for these input diffs.
    DirtyBucket findBestBucket(
        const std::unordered_map<uint32_t, std::vector<Region>> &inputDiffs) const
    {
        if (inputDiffs.empty())
        {
            // No diffs — return an empty bucket (everything clean)
            return DirtyBucket{};
        }

        std::unordered_map<uint32_t, std::vector<Region>> canonicalRegions;

        for (const auto &pair : inputDiffs)
        {
            uint32_t nodeId = pair.first;
            const auto &regionList = pair.second;

            if (regionList.empty())
                continue;

            // Use the first region box for bucket snapping
            const Region &box = regionList[0];
            Region canonical;

            for (size_t d = 0; d < box.region.size(); ++d)
            {
                uint32_t dimLen = graph.nodes[nodeId].shape[d];
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

            canonicalRegions[nodeId] = {canonical};
        }

        const DirtyBucket *cached = lookupCache(canonicalRegions);
        if (cached)
        {
            return *cached;
        }

        // Fallback: return empty bucket (full recompute behavior)
        return DirtyBucket{};
    }

    void run(const std::unordered_map<uint32_t, const void *> &inputs)
    {
        if (!isCompiled)
        {
            compile(inputs);
        }

        // Compute input diffs against previous data
        std::unordered_map<uint32_t, std::vector<Region>> inputDiffs;

        for (const auto &pair : inputs)
        {
            uint32_t nodeId = pair.first;
            const void *newData = pair.second;

            if (graph.nodes[nodeId].opType != OpType::INPUT)
                continue;

            const void *oldData = nullptr;
            auto prevIt = previousInputData.find(nodeId);
            if (prevIt != previousInputData.end())
            {
                oldData = prevIt->second.data();
            }

            auto diff = computeInputDiff(oldData, newData, graph.nodes[nodeId].shape, graph.nodes[nodeId].dtype);
            if (!diff.empty())
            {
                inputDiffs[nodeId] = diff;
            }

            // Save a copy of the new input data
            uint64_t sizeBytes = getSizeBytes(graph.nodes[nodeId].shape, graph.nodes[nodeId].dtype);
            auto &stored = previousInputData[nodeId];
            stored.resize(sizeBytes);
            std::memcpy(stored.data(), newData, sizeBytes);

            // Write the new data to the buffer
            memManager.write(graph.nodes[nodeId].backend, nodeId, newData, sizeBytes);
        }

        // Find the best matching bucket
        DirtyBucket bucket = findBestBucket(inputDiffs);

        executor->run(inputs, bucket);
    }

    const void *getOutput(uint32_t nodeId) const
    {
        auto &buf = memManager.buffers.at(graph.nodes[nodeId].backend);
        auto it = buf.allocationMap.find(nodeId);

        if (it == buf.allocationMap.end())
            return nullptr;

        uint64_t offset = it->second->offset;
        uint64_t baseOffset = graph.nodes[nodeId].view.shape.empty() ? 0 : graph.nodes[nodeId].view.baseOffset;

#ifdef USE_CUDA
        cudaDeviceSynchronize();
#endif

        return buf.arena_ptr + offset + baseOffset;
    }

    const void *getRootOutput() const
    {
        return getOutput(rootId);
    }

    // -----------------------------------------------------------------------
    // Input diff — compute dirty regions by comparing old vs new data
    // -----------------------------------------------------------------------

    // Computes per-dimension bounding-box dirty regions between old and new data.
    // Returns a single bounding box Region that encompasses all dirty elements.
    // Returns an empty vector if no diff.
    std::vector<Region> computeInputDiff(
        const void *oldData,
        const void *newData,
        const std::vector<uint32_t> &shape,
        DType dtype) const
    {
        if (shape.empty())
            return {};

        uint64_t totalElements = countElements(shape);
        uint64_t elementSize = getDTypeSize(dtype);

        // If no old data, everything is dirty
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

        // Calculate strides for coordinate decomposition
        std::vector<uint64_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        std::vector<uint32_t> minBounds(shape.size(), UINT32_MAX);
        std::vector<uint32_t> maxBounds(shape.size(), 0);
        bool anyDirty = false;

        // Element-wise comparison across all elements
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

        // Build a single bounding box using the observed min/max extents per dimension
        Region r;
        for (size_t d = 0; d < shape.size(); ++d)
        {
            r.region.push_back({minBounds[d], maxBounds[d] + 1});
        }

        return {r};
    }

    // -----------------------------------------------------------------------
    // Power-of-2 tile generation
    // -----------------------------------------------------------------------

    // Generate all non-overlapping aligned power-of-2 tiles for a given dim length
    static std::vector<Dim> generateSlicesForDim(uint32_t dimLen)
    {
        std::set<std::pair<uint32_t, uint32_t>> unique;
        uint32_t maxSize = 1;
        while (maxSize < dimLen)
            maxSize *= 2;

        for (uint32_t size = 1; size <= maxSize; size *= 2)
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

    // -----------------------------------------------------------------------
    // Cache key encoding
    // -----------------------------------------------------------------------

    // Canonical key string from a sorted map of input node ID -> dirty regions.
    // Format: "id:[(s,e),(s,e)...];id:[(s,e),...];..."
    static std::string encodeCacheKey(
        const std::unordered_map<uint32_t, std::vector<Region>> &inputRegions)
    {
        // Sort by node ID for determinism
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
            for (size_t r = 0; r < regions.size(); ++r)
            {
                if (r > 0)
                    ss << ",";
                ss << "(";
                for (size_t d = 0; d < regions[r].region.size(); ++d)
                {
                    if (d > 0)
                        ss << ",";
                    ss << regions[r].region[d].start << "-" << regions[r].region[d].stop;
                }
                ss << ")";
            }
            ss << "]";
        }
        return ss.str();
    }

    // -----------------------------------------------------------------------
    // Cache coverage — pre-compute all dirty region permutations
    // -----------------------------------------------------------------------

    void ensureCacheCoverage(const std::vector<uint32_t> &inputNodeIds)
    {
        std::cout << "Ensuring cache coverage" << std::endl;
        // 1. For each input, generate power-of-2 tile options per dimension
        struct InputOption
        {
            uint32_t nodeId;
            std::vector<std::vector<Dim>> dimSlices; // per dimension
        };

        std::vector<InputOption> inputOptions;
        for (uint32_t nodeId : inputNodeIds)
        {
            if (graph.nodes[nodeId].opType != OpType::INPUT)
                continue;

            InputOption opt;
            opt.nodeId = nodeId;
            for (uint32_t dimLen : graph.nodes[nodeId].shape)
            {
                opt.dimSlices.push_back(generateSlicesForDim(dimLen));
            }
            inputOptions.push_back(opt);
        }

        if (inputOptions.empty())
            return;

        // 2. Generate all region combinations for each input
        //    Each input gets a list of possible region lists (either empty for clean, or 1 box)
        struct InputRegionSet
        {
            uint32_t nodeId;
            std::vector<std::vector<Region>> options;
        };

        std::vector<InputRegionSet> perInput;
        for (const auto &opt : inputOptions)
        {
            InputRegionSet irs;
            irs.nodeId = opt.nodeId;

            // Cartesian product across dimensions
            std::vector<Region> current = {Region{}};
            for (const auto &dimSlices : opt.dimSlices)
            {
                std::vector<Region> next;
                for (const auto &existing : current)
                {
                    for (const Dim &slice : dimSlices)
                    {
                        Region r = existing;
                        r.region.push_back(slice);
                        next.push_back(r);
                    }
                }
                current = std::move(next);
            }

            // Add clean state (empty list of regions)
            irs.options.push_back({});

            // Add all dirty states (list containing one region box)
            for (const auto &r : current)
            {
                irs.options.push_back({r});
            }

            std::cout << "input node " << irs.nodeId << " has " << irs.options.size() << " buckets (including clean state)." << std::endl;
            perInput.push_back(irs);
        }

        // 3. Cartesian product across all inputs
        //    Use iterative index tracking
        std::vector<size_t> indices(perInput.size(), 0);
        std::vector<size_t> sizes;
        for (const auto &irs : perInput)
        {
            sizes.push_back(irs.options.size());
        }

        std::cout << "Caching dirty region propagation" << std::endl;
        uint32_t cacheIdx = 0;
        while (true)
        {
            cacheIdx++;
            std::cout << cacheIdx << "\r";
            // Build input dirty regions for this permutation
            std::unordered_map<uint32_t, std::vector<Region>> inputRegions;
            bool allClean = true;
            for (size_t i = 0; i < perInput.size(); ++i)
            {
                const auto &option = perInput[i].options[indices[i]];
                if (!option.empty())
                {
                    inputRegions[perInput[i].nodeId] = option;
                    allClean = false;
                }
            }

            if (!allClean)
            {
                std::string key = encodeCacheKey(inputRegions);

                // Skip if already cached
                if (dirtyCache.find(key) == dirtyCache.end())
                {
                    // Forward propagate
                    auto allRegions = propagateDirtyRegions(compiled, graph, inputRegions);

                    // Build bucket
                    DirtyBucket bucket;
                    bucket.regions = allRegions;

                    // Backward propagate input slices for each dirty non-input node
                    for (const OpInstruction &inst : compiled.instructions)
                    {
                        auto regIt = allRegions.find(inst.nodeId);
                        if (regIt == allRegions.end() || regIt->second.empty())
                            continue;

                        const auto &outputRegions = regIt->second;
                        std::vector<std::vector<std::vector<Region>>> perOutputRegionSlices;

                        for (const Region &outRegion : outputRegions)
                        {
                            auto parentSlices = getInputSlices(graph, inst.nodeId, {outRegion});
                            perOutputRegionSlices.push_back(parentSlices);
                        }

                        bucket.inputSlices[inst.nodeId] = perOutputRegionSlices;
                    }

                    dirtyCache[key] = bucket;
                    saveCacheEntry(key, bucket);
                }
            }

            // Advance indices (odometer-style)
            int p = static_cast<int>(indices.size()) - 1;
            while (p >= 0)
            {
                indices[p]++;
                if (indices[p] < sizes[p])
                    break;
                indices[p] = 0;
                p--;
            }
            if (p < 0)
                break;
        }
    }

    // Lookup a cached bucket by input dirty regions. Returns nullptr if not found.
    const DirtyBucket *lookupCache(
        const std::unordered_map<uint32_t, std::vector<Region>> &inputRegions) const
    {
        std::string key = encodeCacheKey(inputRegions);
        auto it = dirtyCache.find(key);
        if (it != dirtyCache.end())
        {
            return &it->second;
        }
        return nullptr;
    }

    // -----------------------------------------------------------------------
    // Cache persistence (JSONL format)
    // -----------------------------------------------------------------------

    void saveCompiledGraph(const CompiledGraph &cg)
    {
        if (cachePath.empty())
            return;
        std::ofstream file(cachePath, std::ios::app);
        if (!file.is_open())
            return;

        json entry;
        entry["type"] = "compiled_graph";
        entry["data"] = cg;
        // Also save the graph's current ID counter to ensure future nodes don't collide
        entry["graph_count"] = graph.count;

        file << entry.dump() << "\n";
    }

    void loadCache()
    {
        if (cachePath.empty())
            return;
        std::ifstream file(cachePath);
        if (!file.is_open())
            return;

        std::string line;
        while (std::getline(file, line))
        {
            if (line.empty())
                continue;
            json entry = json::parse(line);

            if (entry.contains("type") && entry["type"] == "compiled_graph")
            {
                std::cout << "[Session.loadCache] Loading pre-compiled graph..." << std::endl;
                compiled = entry["data"].get<CompiledGraph>();
                graph.count = entry["graph_count"];

                // Re-sync the main graph's nodes with the compiled nodes
                // The planner often creates new nodes (fusions/equivalents) that aren't in the user's initial graph
                for (const auto &pair : compiled.nodesMap)
                {
                    uint32_t id = pair.first;
                    if (id >= graph.nodes.size())
                    {
                        graph.nodes.resize(id + 1);
                    }
                    graph.nodes[id] = pair.second;
                }
                isPlanned = true;
            }
            else if (entry.contains("key"))
            {
                std::string key = entry["key"].get<std::string>();
                DirtyBucket bucket = dirty_cache_json::bucketFromJson(entry["bucket"]);
                dirtyCache[key] = std::move(bucket);
            }
        }
    }

    void saveCacheEntry(const std::string &key, const DirtyBucket &bucket) const
    {
        if (cachePath.empty())
            return;

        std::ofstream file(cachePath, std::ios::app);
        if (!file.is_open())
            return;

        json entry;
        entry["key"] = key;
        entry["bucket"] = dirty_cache_json::bucketToJson(bucket);

        file << entry.dump() << "\n";
    }

    void saveFullCache() const
    {
        if (cachePath.empty())
            return;

        std::ofstream file(cachePath, std::ios::trunc);
        if (!file.is_open())
            return;

        for (const auto &pair : dirtyCache)
        {
            json entry;
            entry["key"] = pair.first;
            entry["bucket"] = dirty_cache_json::bucketToJson(pair.second);

            file << entry.dump() << "\n";
        }
    }
};
