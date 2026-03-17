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
#include <queue>

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
            regionsObj[std::to_string(pair.first)] = regionsToJson(pair.second);
        }
        obj["regions"] = regionsObj;

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

        json kernelsObj;
        for (const auto &pair : bucket.kernelIds)
        {
            json kArr = json::array();
            for (uint64_t k : pair.second)
            {
                std::stringstream ss;
                ss << "0x" << std::hex << k;
                kArr.push_back(ss.str());
            }
            kernelsObj[std::to_string(pair.first)] = kArr;
        }
        obj["kernel_ids"] = kernelsObj;

        return obj;
    }

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

        if (obj.contains("kernel_ids"))
        {
            for (auto it = obj["kernel_ids"].begin(); it != obj["kernel_ids"].end(); ++it)
            {
                uint32_t nodeId = std::stoul(it.key());
                std::vector<uint64_t> kArr;
                for (const auto &kStr : it.value())
                {
                    kArr.push_back(std::stoull(kStr.get<std::string>(), nullptr, 16));
                }
                bucket.kernelIds[nodeId] = kArr;
            }
        }

        return bucket;
    }
}

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

            saveCompiledGraph(compiled);
            isPlanned = true;
        }

        std::cout << "[Session.compile] Materializing persistent memory..." << std::endl;
        memManager.init();

        for (const auto &pair : compiled.nodesMap)
        {
            uint32_t nodeId = pair.first;
            const TensorNode &node = pair.second;

            if (node.opType == OpType::INPUT && node.storageType == StorageType::PERSISTENT)
            {
                uint64_t sizeBytes = getSizeBytes(node.shape, node.dtype);

                uint64_t offset = memManager.allocate(node.backend, nodeId, sizeBytes, StorageType::PERSISTENT);

                if (graph.constantStaging.count(nodeId))
                {
                    memManager.write(node.backend, nodeId, graph.constantStaging[nodeId].data(), sizeBytes);
                }
                else if (graph.weightSources.count(nodeId))
                {
                    const auto &source = graph.weightSources.at(nodeId);
                    auto &loader = graph.loaders.at(source.first);
                    uint8_t *destPtr = memManager.buffers.at(node.backend).arena_ptr + offset;
                    loader->loadTensor(source.second, destPtr, sizeBytes);
                }
            }
        }

        std::vector<uint32_t> inputNodeIds;
        for (const auto &pair : compiled.nodesMap)
        {
            uint32_t nodeId = pair.first;
            if (pair.second.opType == OpType::INPUT)
            {
                if (graph.weightSources.count(nodeId) == 0 && graph.constantStaging.count(nodeId) == 0)
                {
                    inputNodeIds.push_back(nodeId);
                }
            }
        }
        ensureCacheCoverage(inputNodeIds);

        for (const auto &pair : compiled.nodesMap)
        {
            uint32_t id = pair.first;
            if (id >= graph.nodes.size())
            {
                graph.nodes.resize(id + 1);
            }
            graph.nodes[id] = pair.second;
        }

        executor = std::make_unique<Executor>(compiled, memManager, graph);
        isCompiled = true;
        graph.constantStaging.clear();
    }

    DirtyBucket findBestBucket(
        const std::unordered_map<uint32_t, std::vector<Region>> &inputDiffs) const
    {
        if (inputDiffs.empty())
        {
            return DirtyBucket{};
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

        return DirtyBucket{};
    }

    void run(const std::unordered_map<uint32_t, const void *> &inputs)
    {
        if (!isCompiled)
        {
            compile(inputs);
        }

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

            uint64_t sizeBytes = getSizeBytes(graph.nodes[nodeId].shape, graph.nodes[nodeId].dtype);
            auto &stored = previousInputData[nodeId];
            stored.resize(sizeBytes);
            std::memcpy(stored.data(), newData, sizeBytes);

            memManager.write(graph.nodes[nodeId].backend, nodeId, newData, sizeBytes);
        }

        DirtyBucket bucket = findBestBucket(inputDiffs);
        executor->run(inputs, bucket);
    }

    const void *getOutput(uint32_t nodeId) const
    {
        if (isCompiled && nodeId == rootId && !compiled.instructions.empty())
        {
            nodeId = compiled.instructions.back().nodeId;
        }

        Backend backend = graph.nodes[nodeId].backend;
        uint64_t baseOffset = graph.nodes[nodeId].view.shape.empty() ? 0 : graph.nodes[nodeId].view.baseOffset;

        if (isCompiled && compiled.nodesMap.find(nodeId) != compiled.nodesMap.end())
        {
            backend = compiled.nodesMap.at(nodeId).backend;
            baseOffset = compiled.nodesMap.at(nodeId).view.shape.empty() ? 0 : compiled.nodesMap.at(nodeId).view.baseOffset;
        }

        auto &buf = memManager.buffers.at(backend);
        auto it = buf.allocationMap.find(nodeId);

        if (it == buf.allocationMap.end())
            return nullptr;

        uint64_t offset = it->second->offset;

#ifdef USE_CUDA
        if (backend == Backend::CUDA)
        {
            cudaDeviceSynchronize();
        }
#endif

        return buf.arena_ptr + offset + baseOffset;
    }

    const void *getRootOutput() const
    {
        return getOutput(rootId);
    }

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

    void getDirtySlices(const std::vector<uint32_t> &inputNodeIds)
    {
        std::cout << "getting dirty slices" << std::endl;
        struct InputOption
        {
            uint32_t nodeId;
            std::vector<std::vector<Dim>> dimSlices;
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

        std::vector<uint32_t> atomicTopo;
        std::unordered_set<uint32_t> visited;
        auto visit = [&](auto &self, uint32_t node) -> void
        {
            if (visited.count(node))
                return;
            visited.insert(node);
            if (node < graph.nodes.size())
            {
                for (uint32_t pid : graph.nodes[node].parentIds)
                    self(self, pid);
            }
            atomicTopo.push_back(node);
        };
        visit(visit, rootId);

        ShapePropagator prop;
        for (uint32_t nodeId : atomicTopo)
        {
            if (nodeId < graph.nodes.size())
            {
                prop.inferShape(nodeId, graph);
                if (graph.nodes[nodeId].view.shape.empty() && !graph.nodes[nodeId].shape.empty())
                {
                    graph.nodes[nodeId].view.shape = graph.nodes[nodeId].shape;
                    graph.nodes[nodeId].view.strides = TensorView::calcContiguousStrides(graph.nodes[nodeId].shape);
                    graph.nodes[nodeId].view.dtype = graph.nodes[nodeId].dtype;
                }
            }
        }

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

            irs.options.push_back({});

            for (const auto &r : current)
            {
                irs.options.push_back({r});
            }

            std::cout << "input node " << irs.nodeId << " has " << irs.options.size() << " buckets (including clean state)." << std::endl;
            perInput.push_back(irs);
        }

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
            std::cout << cacheIdx << "\r" << std::flush;
            std::unordered_map<uint32_t, std::vector<Region>> atomicOutputRegions;
            std::unordered_map<uint32_t, std::vector<std::vector<Region>>> atomicInputRegions; // id -> item[parentId][outputRegionId]
            for (size_t i = 0; i < perInput.size(); ++i)
            {
                const auto &option = perInput[i].options[indices[i]];
                atomicOutputRegions[perInput[i].nodeId] = option;
            }

            std::string key = encodeCacheKey(atomicOutputRegions);

            if (dirtyCache.find(key) == dirtyCache.end())
            {
                propagateDirtyRegionsAtomic(atomicTopo, graph, atomicOutputRegions, atomicInputRegions);

                std::unordered_map<uint32_t, std::vector<Region>> physicalRegions;
                for (const auto &pair : compiled.logicalNodeMap)
                {
                    uint32_t physId = pair.first;
                    uint32_t logId = pair.second;
                    if (atomicRegions.count(logId))
                    {
                        physicalRegions[physId] = atomicRegions.at(logId);
                    }
                }
                for (uint32_t inId : inputNodeIds)
                {
                    if (inputRegions.count(inId))
                    {
                        physicalRegions[inId] = inputRegions.at(inId);
                    }
                }

                DirtyBucket bucket;
                bucket.regions = physicalRegions;

                for (const OpInstruction &inst : compiled.instructions)
                {
                    auto regIt = physicalRegions.find(inst.nodeId);
                    if (regIt == physicalRegions.end() || regIt->second.empty())
                        continue;

                    const auto &outputRegions = regIt->second;
                    std::vector<std::vector<std::vector<Region>>> perOutputRegionSlices;
                    std::vector<uint64_t> regionKernels;

                    for (size_t rIdx = 0; rIdx < outputRegions.size(); ++rIdx)
                    {
                        const Region &outRegion = outputRegions[rIdx];
                        TensorNode dummyOut = compiled.nodesMap.at(inst.nodeId);

                        // Compute input slices for this output region
                        std::vector<Region> parentSlices(inst.inputNodeIds.size());
                        const TensorNode &physNode = compiled.nodesMap.at(inst.nodeId);
                        if (physNode.opType == OpType::COPY_TO || physNode.opType == OpType::CONTIGUOUS)
                        {
                            for (size_t i = 0; i < inst.inputNodeIds.size(); ++i)
                                parentSlices[i] = outRegion;
                        }
                        else if (physNode.opType == OpType::FUSED)
                        {
                            std::unordered_map<uint32_t, std::vector<Region>> reqRegions;
                            uint32_t logRoot = compiled.logicalNodeMap.at(inst.nodeId);
                            reqRegions[logRoot] = outRegion;

                            ShapePropagator bp;
                            for (auto it = atomicTopo.rbegin(); it != atomicTopo.rend(); ++it)
                            {
                                uint32_t logId = *it;
                                if (reqRegions.count(logId) && !reqRegions[logId].empty())
                                {
                                    bool isPhysicalInput = false;
                                    for (uint32_t pInId : inst.inputNodeIds)
                                    {
                                        if (compiled.logicalNodeMap.count(pInId) && compiled.logicalNodeMap.at(pInId) == logId && logId != logRoot)
                                        {
                                            isPhysicalInput = true;
                                            break;
                                        }
                                    }
                                    if (isPhysicalInput)
                                        continue;

                                    const TensorNode &logNode = graph.nodes[logId];
                                    if (logNode.opType != OpType::INPUT)
                                    {
                                        auto pRegs = bp.backward(logNode, graph, reqRegions[logId]);
                                        for (size_t k = 0; k < logNode.parentIds.size(); ++k)
                                        {
                                            reqRegions[logNode.parentIds[k]] = pRegs[k];
                                        }
                                    }
                                }
                            }

                            for (size_t i = 0; i < inst.inputNodeIds.size(); ++i)
                            {
                                if (compiled.logicalNodeMap.count(inst.inputNodeIds[i]))
                                {
                                    uint32_t logInId = compiled.logicalNodeMap.at(inst.inputNodeIds[i]);
                                    if (reqRegions.count(logInId) && !reqRegions[logInId].empty())
                                        parentSlices[i] = reqRegions[logInId];
                                    else
                                        parentSlices[i] = {outRegion};
                                }
                                else
                                {
                                    parentSlices[i] = {outRegion};
                                }
                            }
                        }
                        else
                        {
                            ShapePropagator backProp;
                            parentSlices = backProp.backward(physNode, graph, {outRegion});
                        }
                    }

                    dirtyCache[key] = bucket;
                    saveCacheEntry(key, bucket);
                }
            }

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