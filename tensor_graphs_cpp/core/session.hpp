#pragma once
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>

#include "core/common/bench_utils.hpp"
#include "core/common/thread_pool.hpp"
#include "core/cost_model.hpp"
#include "core/executor.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/plan/planner.hpp"
#include "core/repo.hpp"
#include "core/shapes.hpp"
#include "core/types.hpp"

static std::string encodeCacheKey(const std::unordered_map<uint32_t, std::vector<Region>> &inputRegions)
{
    std::vector<uint32_t> ids;
    ids.reserve(inputRegions.size());
    for (const auto &pair : inputRegions)
    {
        ids.push_back(pair.first);
    }
    std::sort(ids.begin(), ids.end());

    std::stringstream ss;
    for (uint64_t i = 0; i < ids.size(); ++i)
    {
        if (i > 0)
            ss << ";";
        ss << ids[i] << ":[";

        const auto &regions = inputRegions.at(ids[i]);
        const std::vector<Region> canonicalRegions = normalizeRegions(regions);
        for (uint64_t r = 0; r < canonicalRegions.size(); ++r)
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
    LogicalId rootId;
    bool isPlanned;
    bool isCompiled;
    uint32_t nBucketSizes = 0;
    std::vector<Bucket> manualBuckets;

    std::string cachePath;
    std::vector<CompiledGraph> cachedGraphs;
    std::unordered_map<LogicalId, MemSpace> selectedCachedNodes;

    std::unordered_map<std::string, uint64_t> bucketCallCounts;
    std::string bucketCountsPath = "benchmarks/bucket_counts.bin";
    std::string recordsPath = "benchmarks/records.bin";

    uint32_t fullBucketIdx;
    Repo *repo;
    bool disableCaching = false;
    float minCompileSeconds = 0.0f;
    std::shared_ptr<SearchDelegate> delegate = nullptr;
    bool logCostCalls = false;

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

    std::vector<LogicalId> collectInputNodeIds() const
    {
        std::vector<LogicalId> inputNodeIds;
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
        bw.write<LogicalId>(rootId);
        bw.write(selectedCachedNodes);

        for (const CompiledGraph &g : cachedGraphs)
        {
            bw.write<uint8_t>(1); // Bucket block type
            bw.write(g);
        }

        bw.write<uint8_t>(2); // Constants block type
        std::unordered_set<LogicalId> neededConstants;
        for (const auto &pair : graph.constantStaging)
        {
            neededConstants.insert(pair.first);
        }

        std::vector<LogicalId> orderedConstants(neededConstants.begin(), neededConstants.end());
        std::sort(orderedConstants.begin(), orderedConstants.end());

        bw.write<uint32_t>(static_cast<uint32_t>(orderedConstants.size()));
        for (LogicalId logicalId : orderedConstants)
        {
            bw.write(logicalId);
            bw.write(*graph.constantStaging.at(logicalId));
        }
    }

    void addBucket(const std::unordered_map<LogicalId, std::vector<Region>> &inputDirtyRegions,
                   const std::vector<Region> &outputNeededRegion)
    {
        manualBuckets.push_back({inputDirtyRegions, outputNeededRegion});
    }

    Session(Graph &g, MemoryManager &mem, LogicalId root, const std::string &cacheFile = "", uint32_t _nBucketSizes = 0,
            Repo *_repo = nullptr, bool _disableCaching = false, float _minCompileSeconds = 0.0f,
            std::shared_ptr<SearchDelegate> _delegate = nullptr, bool _logCostCalls = true)
        : graph(g), memManager(mem), rootId(root), isPlanned(false), isCompiled(false), cachePath(cacheFile),
          nBucketSizes(_nBucketSizes), repo(_repo), disableCaching(_disableCaching),
          minCompileSeconds(_minCompileSeconds), delegate(_delegate), logCostCalls(_logCostCalls),
          costModel(_logCostCalls)
    {
        ensureOutputDirectories();
        loadCache();
    }

    void ensureFullBucket()
    {
        Bucket bucket;
        bucket.outputNeededRegion = {makeFull(graph.getNode(rootId).getShape())};
        std::vector<LogicalId> inputNodeIds = collectInputNodeIds();
        for (LogicalId nodeId : inputNodeIds)
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
        costModel.setLogging(logCostCalls);
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

        // Compute exact peak allocation size required per MemSpace across all compiled graphs
        std::unordered_map<MemSpace, uint64_t> peakSizes;
        for (const CompiledGraph &g : cachedGraphs)
        {
            for (const auto &inst : g.instructions)
            {
                if (inst.outBuffer.mem_space.type != HandleType::STORAGE && inst.outBuffer.offset >= 0)
                {
                    uint64_t extent = static_cast<uint64_t>(inst.outBuffer.offset) + inst.outBuffer.size;
                    peakSizes[inst.outBuffer.mem_space] = std::max(peakSizes[inst.outBuffer.mem_space], extent);
                }
                for (const auto &inBuf : inst.inBuffers)
                {
                    if (inBuf.mem_space.type != HandleType::STORAGE && inBuf.offset >= 0)
                    {
                        uint64_t extent = static_cast<uint64_t>(inBuf.offset) + inBuf.size;
                        peakSizes[inBuf.mem_space] = std::max(peakSizes[inBuf.mem_space], extent);
                    }
                }
            }
        }

        std::cout << "[Session.compile] Materializing exact peak memory arenas..." << std::endl;
        for (const auto &pair : peakSizes)
        {
            std::cout << "  - " << pair.first << ": " << pair.second << " bytes ("
                      << (pair.second / (1024.0 * 1024.0)) << " MB)" << std::endl;
        }

        memManager.init(peakSizes);

        std::unordered_set<LogicalId> written;
        for (const CompiledGraph &g : cachedGraphs)
        {
            for (const auto &inst : g.instructions)
            {
                for (uint32_t i = 0; i < inst.children.size(); i++)
                {
                    EClassId child = inst.children[i];
                    if (!g.has_logical_id(child))
                        continue;
                    LogicalId logical_id = g.get_logical_id(child);
                    if (graph.constantStaging.count(logical_id))
                    {
                        if (written.insert(logical_id).second)
                        {
                            const TensorNode &node = graph.getNode(logical_id);
                            const ParallelBuffer &buf = inst.inBuffers[i];
                            memManager.write(buf.mem_space, buf.offset, graph.constantStaging.at(logical_id)->data(),
                                             node.getSizeBytes());
                        }
                    }
                }
            }
        }
        std::cout << "Wrote " << written.size() << " constants to memory. Graph has " << graph.constantStaging.size()
                  << " constants." << std::endl;

        executor = std::make_unique<Executor>(memManager);
        isCompiled = true;
    }

    void writeInput(LogicalId logicalId, const void *data, uint64_t size)
    {
        for (const CompiledGraph &g : cachedGraphs)
        {
            for (const auto &inst : g.instructions)
            {
                for (uint32_t i = 0; i < inst.children.size(); i++)
                {
                    EClassId child = inst.children[i];
                    if (g.has_logical_id(child) && g.get_logical_id(child) == logicalId)
                    {
                        memManager.write(inst.inBuffers[i].mem_space, inst.inBuffers[i].offset, data, size);
                        return;
                    }
                }
            }
        }
        Error::throw_err("Logical Node ID " + toString(logicalId) +
                         " not found in compiled instructions during Session::writeInput");
    }

    const void *run(Bucket bucket = {}, Debug::Callback debugCallback = nullptr, bool doSaturate = true)
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
        executor->run(cachedGraphs[graphIdx], debugCallback);

        const OpInstruction &lastInst = cachedGraphs[graphIdx].instructions.back();
        DeviceBuffer *buf = memManager.getBuffer(lastInst.outBuffer.mem_space);
        return buf->getBasePtr() + lastInst.outBuffer.offset;
    }

    void ensureCacheCoverage(bool doSaturate)
    {
        cachedGraphs.clear();
        selectedCachedNodes.clear();

        std::cout << "[Session.ensureCacheCoverage] Selecting cache nodes via SearchDelegate..." << std::endl;
        Planner planner(costModel, memManager.getMemCaps());

        std::unordered_map<LogicalId, MemSpace> bestCachedNodes;
        if (!disableCaching)
        {
            bestCachedNodes = planner.searchBestCacheNodes(rootId, graph, manualBuckets, delegate, minCompileSeconds);
        }

        std::unordered_map<LogicalId, ParallelBuffer> preallocatedBuffers;
        preallocateLogicalBuffers(bestCachedNodes, preallocatedBuffers);

        std::cout << "[Session.ensureCacheCoverage] Planning buckets with " << bestCachedNodes.size()
                  << " cached nodes across physical cores..." << std::endl;

        std::vector<LogicalId> topo = topologicalSort({rootId}, graph);
        Graph tempGraph = graph;
        planner.initBaseEGraph(rootId, tempGraph, topo, repo);

        cachedGraphs.resize(manualBuckets.size());

        ThreadPool::get().parallel_for(static_cast<uint32_t>(manualBuckets.size()), [&](uint32_t i) {
            Planner threadPlanner(costModel, memManager.getMemCaps());
            threadPlanner.baseState = planner.baseState;
            threadPlanner.baseStateInitialized = true;

            const Bucket &bucket = manualBuckets[i];
            CompiledGraph plan = threadPlanner.plan(rootId, graph, bucket, bestCachedNodes, doSaturate, true, repo,
                                                    preallocatedBuffers, minCompileSeconds, delegate);
            plan.bucket = bucket;
            cachedGraphs[i] = std::move(plan);
        });

        selectedCachedNodes = std::move(bestCachedNodes);

        if (cachedGraphs.size() != manualBuckets.size())
        {
            Error::throw_err("[Session.ensureCacheCoverage] Planned " + std::to_string(cachedGraphs.size()) +
                             " buckets, but expected " + std::to_string(manualBuckets.size()) + ".");
        }

        persistCache();
    }

    void preallocateLogicalBuffers(const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
                                   std::unordered_map<LogicalId, ParallelBuffer> &out) const
    {
        out.clear();

        struct PreAllocEntry
        {
            LogicalId logicalId;
            MemSpace memSpace;
            std::vector<uint32_t> shape;
            DType dtype;
        };
        std::vector<PreAllocEntry> entries;

        MemSpace storage = MemSpace{0, HandleType::STORAGE};
        MemSpace ram = MemSpace{1, HandleType::CPP};

        for (const auto &pair : graph.nodes)
        {
            const TensorNode &node = pair.second;
            if (node.opType != OpType::INPUT)
                continue;

            auto idtIt = graph.input_data_types.find(node.id);
            if (idtIt != graph.input_data_types.end() && idtIt->second == InputDataType::STORAGE)
                continue;

            entries.push_back({node.id, ram, node.getShape(), node.dtype});
        }

        for (const auto &kv : cachedNodes)
        {
            LogicalId logicalId = kv.first;
            MemSpace ms = kv.second;
            if (!graph.hasNode(logicalId))
                continue;
            const TensorNode &node = graph.getNode(logicalId);
            bool alreadyAdded = false;
            for (const auto &e : entries)
            {
                if (e.logicalId == logicalId)
                {
                    alreadyAdded = true;
                    break;
                }
            }
            if (alreadyAdded)
                continue;
            entries.push_back({logicalId, ms, node.getShape(), node.dtype});
        }

        std::sort(entries.begin(), entries.end(),
                  [](const PreAllocEntry &a, const PreAllocEntry &b) { return a.logicalId < b.logicalId; });

        std::unordered_map<MemSpace, uint64_t> cursor;
        BufferId nextId{0};
        for (const auto &e : entries)
        {
            if (e.memSpace == storage)
                continue;

            uint64_t size_bytes = getSizeBytes(e.shape, e.dtype);
            if (size_bytes == 0)
                continue;
            size_bytes = (size_bytes + 4095) & ~4095ULL;

            uint64_t offset = cursor[e.memSpace];
            cursor[e.memSpace] = offset + size_bytes;

            ParallelBuffer buf;
            buf.id = nextId++;
            buf.mem_space = e.memSpace;
            buf.size = size_bytes;
            buf.start = 0;
            buf.end = std::numeric_limits<uint32_t>::max();
            buf.offset = static_cast<int64_t>(offset);
            out[e.logicalId] = std::move(buf);
        }

        std::cout << "[Session.preallocateLogicalBuffers] Pre-allocated " << out.size() << " INPUT/CACHE buffers.";
        for (const auto &kv : cursor)
        {
            std::cout << " MemSpace(idx=" << kv.first.idx << ",type=" << static_cast<int>(kv.first.type)
                      << ") reserved=" << kv.second << " bytes.";
        }
        std::cout << std::endl;
    }

    const uint32_t getBestGraphIdx(const Bucket &bucket) const
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

        CacheFile cache = loadCacheFile(cachePath, /*validateKernels=*/true);

        if (!cache.isValid || cache.version != kCacheFileVersion || cache.rootId != rootId)
        {
            std::string reason = cache.isValid ? "Version or RootId mismatch" : cache.invalidReason;
            std::cout << "[Session.loadCache] invalid cache: " << reason << std::endl;
            std::ofstream clearFile(cachePath, std::ios::trunc | std::ios::binary);
            return;
        }

        for (const auto &pair : cache.constants)
        {
            graph.constantStaging[pair.first] = pair.second;
            if (graph.hasNode(pair.first))
            {
                const TensorNode &node = graph.getNode(pair.first);
                uint64_t dataHash =
                    tg_hash::computeConstantHash(node.getShape(), node.dtype, pair.second->data(), pair.second->size());
                graph.constantHashIndex[dataHash].push_back(pair.first);
            }
        }

        if (!cache.compiledGraphs.empty())
        {
            cachedGraphs = std::move(cache.compiledGraphs);
            selectedCachedNodes = std::move(cache.selectedCachedNodes);
            isPlanned = true;
        }
    }
};