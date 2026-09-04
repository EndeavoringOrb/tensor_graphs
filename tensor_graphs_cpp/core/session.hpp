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
#include "core/plan/rule_registry.hpp"
#include "core/repo.hpp"
#include "core/shape_propagator.hpp"
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
    static constexpr uint32_t kCacheFileVersion = 4;

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
    std::vector<float> cachedBucketWeights;

    std::unordered_map<std::string, uint64_t> bucketCallCounts;
    std::string bucketCountsPath = "benchmarks/bucket_counts.bin";
    std::string recordsPath = "benchmarks/records.bin";

    uint32_t fullBucketIdx;
    Repo *repo;
    bool disableCaching = false;
    float minCompileSeconds = 0.0f;
    std::shared_ptr<SearchDelegate> delegate = nullptr;
    bool logCostCalls = false;

    Settings settings;

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
        bw.write(normalizedBucketWeights(manualBuckets));

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
                   const std::vector<Region> &outputNeededRegion, float weight = 1.0f)
    {
        Bucket bucket{inputDirtyRegions, outputNeededRegion};
        bucket.weight = weight;
        manualBuckets.push_back(std::move(bucket));
    }

    void setBucketWeights(const std::vector<float> &weights)
    {
        if (weights.size() != manualBuckets.size())
        {
            Error::throw_err("[Session.setBucketWeights] expected " + std::to_string(manualBuckets.size()) +
                             " weights, got " + std::to_string(weights.size()));
        }
        (void)normalizedBucketWeights(weights);
        for (size_t i = 0; i < weights.size(); ++i)
            manualBuckets[i].weight = weights[i];
    }

    Session(Graph &g, MemoryManager &mem, LogicalId root, const Settings &_settings, Repo *_repo = nullptr,
            std::shared_ptr<SearchDelegate> _delegate = nullptr)
        : graph(g), memManager(mem), rootId(root), settings(_settings), isPlanned(false), isCompiled(false),
          cachePath(_settings.cache_file), nBucketSizes(0), repo(_repo), disableCaching(_settings.disable_caching),
          minCompileSeconds(_settings.min_compile_seconds),
          delegate(_delegate ? _delegate : std::make_shared<HeuristicSearchDelegate>()),
          logCostCalls(_settings.log_cost_calls), costModel(_settings.log_cost_calls, _settings.records_path)
    {
        if (!settings.is_rules_defined("dispatch") || !settings.is_rules_defined("extract") ||
            !settings.is_rules_defined("bufferize") || !settings.is_rules_defined("malloc") ||
            !settings.is_rules_defined("cache") || !settings.is_rules_defined("enode"))
        {
            enableAllDefaultRules(settings, true);
        }
        ensureOutputDirectories();
        loadCache();
    }

    Session(Graph &g, MemoryManager &mem, LogicalId root, const std::string &cacheFile = "", uint32_t _nBucketSizes = 0,
            Repo *_repo = nullptr, bool _disableCaching = false, float _minCompileSeconds = 0.0f,
            std::shared_ptr<SearchDelegate> _delegate = nullptr, bool _logCostCalls = true,
            const std::string &_recordsPath = "benchmarks/records.bin")
        : graph(g), memManager(mem), rootId(root), isPlanned(false), isCompiled(false), cachePath(cacheFile),
          nBucketSizes(_nBucketSizes), repo(_repo), disableCaching(_disableCaching),
          minCompileSeconds(_minCompileSeconds),
          delegate(_delegate ? _delegate : std::make_shared<HeuristicSearchDelegate>()), logCostCalls(_logCostCalls),
          costModel(_logCostCalls, _recordsPath)
    {
        settings = Settings::get_default();
        settings.cache_file = cacheFile;
        settings.disable_caching = _disableCaching;
        settings.min_compile_seconds = _minCompileSeconds;
        settings.log_cost_calls = _logCostCalls;
        if (!_recordsPath.empty())
            settings.records_path = _recordsPath;
        if (!settings.is_rules_defined("dispatch") || !settings.is_rules_defined("extract") ||
            !settings.is_rules_defined("bufferize") || !settings.is_rules_defined("malloc") ||
            !settings.is_rules_defined("cache") || !settings.is_rules_defined("enode"))
        {
            enableAllDefaultRules(settings, true);
        }
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
        if (!settings.bucket_weights.empty())
            setBucketWeights(settings.bucket_weights);

        const std::vector<float> requestedWeights = normalizedBucketWeights(manualBuckets);
        bool cacheMatchesBuckets =
            cachedGraphs.size() == manualBuckets.size() && cachedBucketWeights.size() == requestedWeights.size();
        for (size_t i = 0; cacheMatchesBuckets && i < manualBuckets.size(); ++i)
        {
            cacheMatchesBuckets = cachedGraphs[i].bucket == manualBuckets[i] &&
                                  std::abs(cachedBucketWeights[i] - requestedWeights[i]) <= 1e-6f;
        }
        if (isPlanned && !cacheMatchesBuckets)
        {
            std::cout << "[Session.compile] Cached buckets or weights changed; replanning." << std::endl;
            cachedGraphs.clear();
            selectedCachedNodes.clear();
            cachedBucketWeights.clear();
            isPlanned = false;
        }

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
            for (const auto &pair : g.nodeViews)
            {
                uint64_t extent =
                    pair.second.offset + countElements(pair.second.getShape()) * getDTypeSize(pair.second.dtype);
                peakSizes[MemSpace{1, HandleType::CPP}] = std::max(peakSizes[MemSpace{1, HandleType::CPP}], extent);
            }
        }

        std::cout << "[Session.compile] Materializing exact peak memory arenas..." << std::endl;
        for (const auto &pair : peakSizes)
        {
            std::cout << "  - " << pair.first << ": " << pair.second << " bytes (" << (pair.second / (1024.0 * 1024.0))
                      << " MB)" << std::endl;
        }

        memManager.init(peakSizes);

        // Write all constants directly to their allocated offsets in memory
        std::unordered_set<LogicalId> written;
        for (const CompiledGraph &g : cachedGraphs)
        {
            for (const auto &pair : g.eclass_to_logical)
            {
                EClassId eclass_id = pair.first;
                LogicalId logical_id = pair.second;
                if (graph.constantStaging.count(logical_id))
                {
                    if (g.nodeViews.count(eclass_id))
                    {
                        if (written.insert(logical_id).second)
                        {
                            const TensorNode &node = graph.getNode(logical_id);
                            const TensorView &view = g.nodeViews.at(eclass_id);
                            memManager.write(MemSpace{1, HandleType::CPP}, view.offset,
                                             graph.constantStaging.at(logical_id)->data(), node.getSizeBytes());
                        }
                    }
                }
            }

            for (const auto &pair : g.constantStaging)
            {
                EClassId eclass_id = pair.first;
                if (g.nodeViews.count(eclass_id))
                {
                    const TensorView &view = g.nodeViews.at(eclass_id);
                    memManager.write(MemSpace{1, HandleType::CPP}, view.offset, pair.second->data(),
                                     pair.second->size());
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
            // 1. Direct O(1) lookup via logical_to_eclass
            auto it = g.logical_to_eclass.find(logicalId);
            if (it != g.logical_to_eclass.end())
            {
                EClassId eclass_id = it->second;
                if (g.nodeViews.count(eclass_id))
                {
                    const TensorView &view = g.nodeViews.at(eclass_id);
                    memManager.write(MemSpace{1, HandleType::CPP}, view.offset, data, size);
                    return;
                }
            }

            // 2. Scan eclass_to_logical fallback
            for (const auto &pair : g.eclass_to_logical)
            {
                if (pair.second == logicalId)
                {
                    EClassId eclass_id = pair.first;
                    if (g.nodeViews.count(eclass_id))
                    {
                        const TensorView &view = g.nodeViews.at(eclass_id);
                        memManager.write(MemSpace{1, HandleType::CPP}, view.offset, data, size);
                        return;
                    }
                }
            }

            // 3. Search instruction input buffers
            for (const auto &inst : g.instructions)
            {
                for (uint32_t i = 0; i < inst.children.size(); i++)
                {
                    EClassId child = inst.children[i];
                    auto it_l = g.eclass_to_logical.find(child);
                    if (it_l != g.eclass_to_logical.end() && it_l->second == logicalId)
                    {
                        memManager.write(inst.inBuffers[i].mem_space, inst.inBuffers[i].offset, data, size);
                        return;
                    }
                }
            }
        }
        Error::throw_err("Logical Node ID " + toString(logicalId) +
                         " not found in compiled graph during Session::writeInput");
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
        const CompiledGraph &cg = cachedGraphs[graphIdx];
        executor->run(cg, debugCallback);

        // Find the root node in CPU RAM
        for (const auto &pair : cg.eclass_to_logical)
        {
            if (pair.second == rootId)
            {
                EClassId eclass_id = pair.first;
                if (cg.nodeViews.count(eclass_id))
                {
                    const TensorView &rootView = cg.nodeViews.at(eclass_id);
                    DeviceBuffer *buf = memManager.getBuffer(MemSpace{1, HandleType::CPP});
                    if (buf && buf->getBasePtr())
                    {
                        return buf->getBasePtr() + rootView.offset;
                    }
                }
            }
        }

        if (!cg.instructions.empty())
        {
            const OpInstruction &lastInst = cg.instructions.back();
            DeviceBuffer *buf = memManager.getBuffer(lastInst.outBuffer.mem_space);
            if (buf && buf->getBasePtr())
            {
                return buf->getBasePtr() + lastInst.outBuffer.offset;
            }
        }

        Error::throw_err("Failed to retrieve valid host output pointer for root node during Session::run");
    }

    void ensureCacheCoverage(bool doSaturate)
    {
        cachedGraphs.clear();
        selectedCachedNodes.clear();

        Planner planner(costModel, settings);

        std::unordered_map<LogicalId, MemSpace> bestCachedNodes;
        if (!disableCaching)
        {
            bestCachedNodes = planner.searchBestCacheNodes(rootId, graph, manualBuckets, delegate, minCompileSeconds);
        }

        std::unordered_map<LogicalId, ParallelBuffer> preallocatedBuffers;
        planner.preallocateLogicalBuffers(graph, bestCachedNodes, preallocatedBuffers);

        std::cout << "[Session.ensureCacheCoverage] Planning buckets with " << bestCachedNodes.size()
                  << " cached nodes across physical cores..." << std::endl;

        std::vector<LogicalId> topo = topologicalSort({rootId}, graph);
        Graph tempGraph = graph;
        planner.initBaseEGraph(rootId, tempGraph, topo, repo);

        cachedGraphs.resize(manualBuckets.size());

        ThreadPool::get().parallel_for(static_cast<uint32_t>(manualBuckets.size()), [&](uint32_t i) {
            Planner threadPlanner(costModel, settings);
            threadPlanner.baseState = planner.baseState;
            threadPlanner.baseStateInitialized = true;

            const Bucket &bucket = manualBuckets[i];
            CompiledGraph plan = threadPlanner.plan(rootId, graph, bucket, bestCachedNodes, doSaturate, true, repo,
                                                    preallocatedBuffers, minCompileSeconds, delegate);
            plan.bucket = bucket;
            cachedGraphs[i] = std::move(plan);
        });

        selectedCachedNodes = std::move(bestCachedNodes);
        cachedBucketWeights = normalizedBucketWeights(manualBuckets);

        if (cachedGraphs.size() != manualBuckets.size())
        {
            Error::throw_err("[Session.ensureCacheCoverage] Planned " + std::to_string(cachedGraphs.size()) +
                             " buckets, but expected " + std::to_string(manualBuckets.size()) + ".");
        }

        persistCache();
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
            cachedBucketWeights = std::move(cache.bucketWeights);
            isPlanned = true;
        }
    }
};
