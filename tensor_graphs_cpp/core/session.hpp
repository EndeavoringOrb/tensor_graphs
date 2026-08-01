#pragma once
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>

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
            Repo *_repo = nullptr, bool _disableCaching = false, float _minCompileSeconds = 0.0f)
        : graph(g), memManager(mem), rootId(root), isPlanned(false), isCompiled(false), cachePath(cacheFile),
          nBucketSizes(_nBucketSizes), repo(_repo), disableCaching(_disableCaching),
          minCompileSeconds(_minCompileSeconds)
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

        std::cout << "[Session.ensureCacheCoverage] Starting iterative cache "
                     "optimization..."
                  << std::endl;
        Planner planner(costModel, memManager.getMemCaps());

        std::unordered_map<LogicalId, MemSpace> protectedCachedNodes;

        if (!disableCaching)
        {
            for (uint64_t i = 0; i < manualBuckets.size(); ++i)
            {
                const Bucket &bucket = manualBuckets[i];

                CompiledGraph plan =
                    planner.plan(rootId, graph, bucket, protectedCachedNodes, i == fullBucketIdx ? false : doSaturate,
                                 false, repo, {}, minCompileSeconds);

                for (const auto &inst : plan.instructions)
                {
                    if (plan.has_logical_id(inst.eclass_id))
                    {
                        LogicalId logical_id = plan.get_logical_id(inst.eclass_id);
                        OpType op_type = graph.getNode(logical_id).opType;
                        if (op_type == OpType::CACHE)
                        {
                            protectedCachedNodes[logical_id] = inst.outBuffer.mem_space;
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Pre-allocate ParallelBuffers for INPUT and CACHE logical nodes
        // *outside* of Planner, before the Final replanning. This guarantees
        // that the byte offset of every persistent INPUT/CACHE buffer is
        // identical across all buckets, so Session::writeInput and the
        // constant-staging writes in Session::compile land at the same
        // physical offset regardless of which bucket's compiled graph is
        // selected at run time.
        // ------------------------------------------------------------------
        std::unordered_map<LogicalId, ParallelBuffer> preallocatedBuffers;
        preallocateLogicalBuffers(protectedCachedNodes, preallocatedBuffers);

        std::cout << "[Session.ensureCacheCoverage] Final replanning with " << protectedCachedNodes.size()
                  << " protected eclasses..." << std::endl;
        for (uint64_t i = 0; i < manualBuckets.size(); ++i)
        {
            const Bucket &bucket = manualBuckets[i];
            CompiledGraph plan = planner.plan(rootId, graph, bucket, protectedCachedNodes, doSaturate, true, repo,
                                              preallocatedBuffers, minCompileSeconds);
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

    // Allocate stable ParallelBuffers for every logical INPUT node and every
    // node in `cachedNodes` (the protected CACHE set discovered during the
    // first planning pass). STORAGE-backed INPUTs are skipped because their
    // offset is resolved dynamically inside StorageBuffer::setupInput.
    //
    // Buffers within a MemSpace are placed contiguously starting at offset 0,
    // sorted by LogicalId for determinism. The MemValidator will reduce the
    // malloc solver's mem_cap by max(offset+size) of these buffers so the
    // transient buffers land strictly above the pre-allocated region.
    void preallocateLogicalBuffers(const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
                                   std::unordered_map<LogicalId, ParallelBuffer> &out) const
    {
        out.clear();

        // Collect (LogicalId, MemSpace, shape, dtype) tuples for every logical
        // node that needs a stable buffer.
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

        // 1. All logical INPUT nodes (constants + runtime inputs + weights).
        for (const auto &pair : graph.nodes)
        {
            const TensorNode &node = pair.second;
            if (node.opType != OpType::INPUT)
                continue;

            // STORAGE-backed INPUTs (file weights) bypass the arena entirely.
            auto idtIt = graph.input_data_types.find(node.id);
            if (idtIt != graph.input_data_types.end() && idtIt->second == InputDataType::STORAGE)
                continue;

            entries.push_back({node.id, ram, node.getShape(), node.dtype});
        }

        // 2. All protected CACHE nodes (computed tensors that survived the
        // first-pass cache selection). Their MemSpace comes from the first
        // pass's `inst.outBuffer.mem_space`; if missing, default to RAM.
        for (const auto &kv : cachedNodes)
        {
            LogicalId logicalId = kv.first;
            MemSpace ms = kv.second;
            if (!graph.hasNode(logicalId))
                continue;
            const TensorNode &node = graph.getNode(logicalId);
            // Skip if this logical id is also an INPUT we already added above
            // (an INPUT can also be a cache source). Prefer the INPUT memspace.
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

        // Stable ordering: sort by LogicalId so the layout is identical across
        // runs even if iteration order over `graph.nodes` differs.
        std::sort(entries.begin(), entries.end(),
                  [](const PreAllocEntry &a, const PreAllocEntry &b) { return a.logicalId < b.logicalId; });

        // Assign offsets per MemSpace, contiguously from offset 0, with the
        // same 4096-byte alignment used by bufferize().
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
            buf.start = 0; // INPUT/CACHE buffers are alive forever.
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
        std::ifstream file(cachePath, std::ios::binary);
        if (!file.is_open())
            return;

        BinaryReader br(file);
        bool hasInvalidCache = false;
        std::string invalidCacheReason = "";
        std::vector<CompiledGraph> tempGraphs;
        std::unordered_map<LogicalId, MemSpace> tempSelectedCachedNodes;

        while (file.peek() != EOF)
        {
            uint8_t type;
            br.read(type);

            if (type == 0)
            {
                uint32_t version;
                LogicalId cachedRootId;
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
                    if (inst.kernel_id == KernelId{0} || !KernelRegistry::get().hasKernel(inst.kernel_id))
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
                tempGraphs.push_back(cg);
            }
            else if (type == 2)
            {
                uint32_t count;
                br.read(count);
                for (uint32_t i = 0; i < count; ++i)
                {
                    LogicalId nodeId;
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
