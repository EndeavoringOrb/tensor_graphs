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
// DirtyBucket — one cached propagation result for a specific input region combo
// ---------------------------------------------------------------------------

struct DirtyBucket
{
    // Per-node output dirty regions (from forward propagation)
    std::unordered_map<uint32_t, std::vector<Region>> regions;

    // Per-node input slices (from backward propagation).
    // Outer key is node ID. Inner vector is per-output-region, each entry
    // is a vector of per-parent required regions.
    std::unordered_map<uint32_t, std::vector<std::vector<std::vector<Region>>>> inputSlices;
};

// ---------------------------------------------------------------------------
// JSON serialization helpers for Region / Dim / DirtyBucket
// ---------------------------------------------------------------------------

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
    bool isCompiled;

    std::string cachePath;
    std::unordered_map<std::string, DirtyBucket> dirtyCache;

    // Map of node ID -> previous input data (for computing diffs)
    std::unordered_map<uint32_t, std::vector<uint8_t>> previousInputData;

public:
    Session(Graph &g, MemoryManager &mem, uint32_t root, const std::string &cacheFile = "")
        : graph(g), memManager(mem), rootId(root), isCompiled(false), cachePath(cacheFile)
    {
        if (!cachePath.empty())
        {
            loadCache();
        }
    }

    void compile()
    {
        Planner planner(costModel, 4ULL * 1024 * 1024 * 1024);

        compiled = planner.plan(rootId, graph);
        executor = std::make_unique<Executor>(compiled, memManager, graph);
        isCompiled = true;
    }

    void run(const std::unordered_map<uint32_t, const void *> &inputs)
    {
        if (!isCompiled)
        {
            compile();
        }
        executor->run(inputs);
    }

    const void *getOutput(uint32_t nodeId) const
    {
        const TensorNode &node = graph.nodes[nodeId];
        auto &buf = memManager.buffers.at(node.backend);
        auto it = buf.allocationMap.find(nodeId);

        if (it == buf.allocationMap.end())
            return nullptr;

        uint64_t offset = it->second->offset;
        uint64_t baseOffset = node.view.shape.empty() ? 0 : node.view.baseOffset;

        return buf.arena.data() + offset + baseOffset;
    }

    const void *getRootOutput() const
    {
        return getOutput(rootId);
    }

    // -----------------------------------------------------------------------
    // Input diff — compute dirty regions by comparing old vs new data
    // -----------------------------------------------------------------------

    // Computes per-dimension bounding-box dirty regions between old and new data.
    // Returns multiple non-overlapping regions (one per contiguous dirty span
    // along the first dimension, with full extent on other dims).
    // Returns empty vector if no diff.
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
        uint64_t totalBytes = totalElements * elementSize;

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

        // Element-wise comparison
        // We work in terms of the first dimension's slices, each slice being
        // the product of remaining dimensions
        uint32_t dim0 = shape[0];
        uint64_t sliceElements = (dim0 > 0) ? totalElements / dim0 : 0;
        uint64_t sliceBytes = sliceElements * elementSize;

        const uint8_t *oldBytes = static_cast<const uint8_t *>(oldData);
        const uint8_t *newBytes = static_cast<const uint8_t *>(newData);

        // Find which dim-0 indices have any difference
        std::vector<bool> dim0Dirty(dim0, false);
        bool anyDirty = false;
        for (uint32_t i = 0; i < dim0; ++i)
        {
            if (std::memcmp(oldBytes + i * sliceBytes, newBytes + i * sliceBytes, sliceBytes) != 0)
            {
                dim0Dirty[i] = true;
                anyDirty = true;
            }
        }

        if (!anyDirty)
            return {};

        // Build separate regions for each contiguous dirty span along dim 0
        std::vector<Region> results;
        uint32_t spanStart = 0;
        bool inSpan = false;

        for (uint32_t i = 0; i < dim0; ++i)
        {
            if (dim0Dirty[i])
            {
                if (!inSpan)
                {
                    spanStart = i;
                    inSpan = true;
                }
            }
            else
            {
                if (inSpan)
                {
                    Region r;
                    r.region.push_back({spanStart, i});
                    for (size_t d = 1; d < shape.size(); ++d)
                    {
                        r.region.push_back({0, shape[d]});
                    }
                    results.push_back(r);
                    inSpan = false;
                }
            }
        }
        // Close trailing span
        if (inSpan)
        {
            Region r;
            r.region.push_back({spanStart, dim0});
            for (size_t d = 1; d < shape.size(); ++d)
            {
                r.region.push_back({0, shape[d]});
            }
            results.push_back(r);
        }

        return results;
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
        // 1. For each input, generate power-of-2 tile options per dimension
        struct InputOption
        {
            uint32_t nodeId;
            std::vector<std::vector<Dim>> dimSlices; // per dimension
        };

        std::vector<InputOption> inputOptions;
        for (uint32_t nodeId : inputNodeIds)
        {
            const TensorNode &node = graph.nodes[nodeId];
            if (node.opType != OpType::INPUT)
                continue;

            InputOption opt;
            opt.nodeId = nodeId;
            for (uint32_t dimLen : node.shape)
            {
                opt.dimSlices.push_back(generateSlicesForDim(dimLen));
            }
            inputOptions.push_back(opt);
        }

        if (inputOptions.empty())
            return;

        // 2. Generate all region combinations for each input
        //    Each input gets a list of possible single-box Regions
        struct InputRegionSet
        {
            uint32_t nodeId;
            std::vector<Region> options; // each option is one bounding-box Region
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
            irs.options = std::move(current);
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

        while (true)
        {
            // Build input dirty regions for this permutation
            std::unordered_map<uint32_t, std::vector<Region>> inputRegions;
            for (size_t i = 0; i < perInput.size(); ++i)
            {
                inputRegions[perInput[i].nodeId] = {perInput[i].options[indices[i]]};
            }

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
            std::string key = entry["key"].get<std::string>();
            DirtyBucket bucket = dirty_cache_json::bucketFromJson(entry["bucket"]);
            dirtyCache[key] = std::move(bucket);
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