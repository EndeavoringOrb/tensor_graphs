#pragma once
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <exception>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

#include "core/common/bench_utils.hpp"
#include "core/common/execute_ref_graph.hpp"
#include "core/common/thread_pool.hpp"
#include "core/cost_model.hpp"
#include "core/executor.hpp"
#include "core/graph.hpp"
#include "core/loaders/resolver.hpp"
#include "core/loaders/tg_store.hpp"
#include "core/memory.hpp"
#include "core/plan/planner.hpp"
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
    static constexpr uint32_t kCacheFileVersion = 8;

    Graph &graph;
    MemoryManager &memManager;
    // Owned through a pointer so a caller can load it while it builds the
    // graph (parsing the records takes about a second).
    std::unique_ptr<CostModel> costModelOwner;
    CostModel &costModel;
    std::unique_ptr<Executor> executor;
    LogicalId rootId;
    bool isPlanned;
    bool isCompiled;
    bool persistent_initialized = false;
    uint32_t nBucketSizes = 0;
    std::vector<Bucket> manualBuckets;

    std::string cachePath;
    std::vector<CompiledGraph> cachedGraphs;
    std::unordered_set<BaseEClassId> selectedCachedNodes;
    std::vector<float> cachedBucketWeights;

    std::vector<const void *> rootPtrs;
    mutable std::vector<float> graphCosts;
    std::unordered_map<std::string, uint64_t> bucketCallCounts;
    std::string bucketCountsPath = "benchmarks/bucket_counts.bin";
    std::string recordsPath = "benchmarks/records.bin";
    // The records file as of the cost model's last load (size and write
    // time); plan() reloads only if it changed since (e.g. calibration ran).
    struct RecordsStamp
    {
        bool valid = false;
        std::string path;
        uintmax_t size = 0;
        std::filesystem::file_time_type write_time{};
        bool operator==(const RecordsStamp &other) const
        {
            return valid && other.valid && path == other.path && size == other.size &&
                   write_time == other.write_time;
        }
    };
    RecordsStamp loadedRecords;

    static RecordsStamp recordsStamp(const std::string &path)
    {
        RecordsStamp stamp;
        stamp.path = path;
        std::error_code error;
        stamp.size = std::filesystem::file_size(path, error);
        if (error)
            return stamp;
        stamp.write_time = std::filesystem::last_write_time(path, error);
        stamp.valid = !error;
        return stamp;
    }

    uint32_t fullBucketIdx = UINT32_MAX;
    TGStore *repo;
    std::unique_ptr<TGStore> owned_repo;
    bool disableCaching = false;
    bool disableCompilationCaching = false;
    float minCompileSeconds = 0.0f;
    std::shared_ptr<plan::Brancher> brancher = nullptr;
    bool logCostCalls = false;

    Settings settings;

    void initRepo(TGStore *_repo)
    {
        if (_repo)
        {
            repo = _repo;
            repo->enableWriting();
        }
        else
        {
            std::string g_hash = computeGraphHash(graph, {rootId});
            std::string repo_path = settings.repo_path.empty() ? ("benchmarks/repo_" + g_hash) : settings.repo_path;
            owned_repo = std::make_unique<TGStore>(repo_path, g_hash, false);
            repo = owned_repo.get();
        }
        if (repo)
        {
            TensorResolver::get().registerStore(repo->getBasePath(),
                                                std::shared_ptr<ITensorStore>(repo, [](ITensorStore *) {}));
        }
    }

    void foldCleanTensors()
    {
        if (!repo)
            return;

        repo->enableWriting();

        std::vector<LogicalId> dynamic_inputs;
        for (const auto &pair : graph.nodes)
        {
            if (pair.second.opType == OpType::INPUT &&
                graph.getInputDataType(pair.first) == InputDataType::RUNTIME)
            {
                dynamic_inputs.push_back(pair.first);
            }
        }
        for (const auto &bucket : manualBuckets)
        {
            // Zero-weight buckets (especially the automatically-added full
            // bucket) are planning witnesses, not expected runtime traffic.
            // Letting their dirty regions into this set prevents inputs that
            // stay clean in every weighted bucket from being folded into the
            // repository for those buckets.
            if (bucket.weight <= 0.0f)
                continue;
            for (const auto &pair : bucket.inputDirtyRegions)
            {
                dynamic_inputs.push_back(pair.first);
            }
        }

        RefGraphOptions ref_options;
        ref_options.only_clean_nodes = true;
        ref_options.fold_weights = settings.fold_weights;
        ref_options.dynamic_inputs = &dynamic_inputs;

        executeReferenceGraph(graph, {rootId}, *repo, ref_options);
    }

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
        if (disableCompilationCaching || cachePath.empty())
            return;
        ensureOutputDirectories();
        std::ofstream file(cachePath, std::ios::trunc | std::ios::binary);
        if (!file.is_open())
            return;

        BinaryWriter bw(file);
        {
            uint64_t graph_bytes = 0, unique_bytes = 0, global_bytes = 0;
            std::unordered_set<const void *> seen;
            for (const CompiledGraph &g : cachedGraphs)
                for (const auto &pair : g.constantStaging)
                    if (pair.second)
                    {
                        graph_bytes += pair.second->size();
                        if (seen.insert(pair.second.get()).second)
                            unique_bytes += pair.second->size();
                    }
            for (const auto &pair : graph.constantStaging)
                global_bytes += pair.second->size();
            LOG(INFO) << "[Session.persistCache] graph-constant bytes=" << graph_bytes << " unique=" << unique_bytes
                      << " (" << seen.size() << " buffers) global-constant bytes=" << global_bytes;
        }

        bw.write<uint8_t>(0); // Metadata block type
        bw.write<uint32_t>(kCacheFileVersion);
        bw.write<LogicalId>(rootId);
        bw.write(selectedCachedNodes);
        bw.write(normalizedBucketWeights(manualBuckets));

        // Graphs serialize independently: into per-graph buffers in
        // parallel, then into the file in order (the same bytes).
        std::vector<std::string> graph_blocks(cachedGraphs.size());
        ThreadPool::get().parallel_for(static_cast<uint32_t>(cachedGraphs.size()), [&](uint32_t i) {
            std::ostringstream block(std::ios::binary);
            BinaryWriter block_writer(block);
            block_writer.write<uint8_t>(1); // Bucket block type
            block_writer.write(cachedGraphs[i]);
            graph_blocks[i] = std::move(block).str();
        });
        file.flush();
        for (const std::string &block : graph_blocks)
            file.write(block.data(), static_cast<std::streamsize>(block.size()));

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

    void updateFullBucketWeight()
    {
        if (fullBucketIdx < manualBuckets.size())
            manualBuckets[fullBucketIdx].weight = manualBuckets.size() == 1 ? 1.0f : 0.0f;
    }

    void setBucketWeights(const std::vector<float> &weights)
    {
        if (weights.size() != manualBuckets.size())
        {
            Error::throw_err("[Session.setBucketWeights] expected " + std::to_string(manualBuckets.size()) +
                             " weights, got " + std::to_string(weights.size()));
        }
        const bool full_only_zero = manualBuckets.size() == 1 && fullBucketIdx == 0 && weights.size() == 1 &&
                                    weights[0] == 0.0f;
        if (!full_only_zero)
            (void)normalizedBucketWeights(weights);
        for (size_t i = 0; i < weights.size(); ++i)
            manualBuckets[i].weight = weights[i];

        updateFullBucketWeight();
    }

    Session(Graph &g, MemoryManager &mem, LogicalId root, const Settings &_settings, TGStore *_repo = nullptr,
            std::shared_ptr<plan::Brancher> _brancher = nullptr)
        : graph(g), memManager(mem), rootId(root), settings(_settings), isPlanned(false), isCompiled(false),
          cachePath(_settings.cache_file), nBucketSizes(0), repo(_repo), disableCaching(_settings.disable_caching),
          disableCompilationCaching(_settings.disable_compilation_caching),
          minCompileSeconds(_settings.min_compile_seconds),
          brancher(_brancher ? _brancher : std::make_shared<plan::HeuristicBrancher>()),
          logCostCalls(_settings.log_cost_calls),
          costModelOwner(std::make_unique<CostModel>(_settings.log_cost_calls, _settings.records_path)),
          costModel(*costModelOwner)
    {
        // plan() reloads the cost model from recordsPath if it changed.
        recordsPath = _settings.records_path;
        loadedRecords = recordsStamp(recordsPath);
        initRepo(_repo);
        ensureOutputDirectories();
        loadCache();
    }

    Session(Graph &g, MemoryManager &mem, LogicalId root, const std::string &cacheFile = "", uint32_t _nBucketSizes = 0,
            TGStore *_repo = nullptr, bool _disableCaching = false, float _minCompileSeconds = 0.0f,
            std::shared_ptr<plan::Brancher> _brancher = nullptr, bool _logCostCalls = true,
            const std::string &_recordsPath = "benchmarks/records.bin", bool _disableCompilationCaching = false,
            std::unique_ptr<CostModel> _loadedCostModel = nullptr)
        : graph(g), memManager(mem), rootId(root), isPlanned(false), isCompiled(false), cachePath(cacheFile),
          nBucketSizes(_nBucketSizes), repo(_repo), disableCaching(_disableCaching),
          disableCompilationCaching(_disableCompilationCaching),
          minCompileSeconds(_minCompileSeconds),
          brancher(_brancher ? _brancher : std::make_shared<plan::HeuristicBrancher>()), logCostCalls(_logCostCalls),
          costModelOwner(_loadedCostModel ? std::move(_loadedCostModel)
                                           : std::make_unique<CostModel>(_logCostCalls, _recordsPath)),
          costModel(*costModelOwner)
    {
        loadedRecords = recordsStamp(_recordsPath);
        settings = Settings::get_default();
        settings.cache_file = cacheFile;
        settings.disable_caching = _disableCaching;
        settings.disable_compilation_caching = _disableCompilationCaching;
        settings.min_compile_seconds = _minCompileSeconds;
        settings.log_cost_calls = _logCostCalls;
        if (!_recordsPath.empty())
            settings.records_path = _recordsPath;
        initRepo(_repo);
        ensureOutputDirectories();
        loadCache();
    }

    void ensureFullBucket()
    {
        ShapePropagator prop;
        prop.inferShapeRecursive(rootId, graph);
        
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

        // Include the full bucket in the objective only when it is the sole
        // bucket, so search has a nonzero cost to optimize in that case.
        updateFullBucketWeight();
    }

    // With `persisted` given, the plan file is written on another thread and
    // `persisted` waits for it (compile overlaps it with arena setup, which
    // only reads the plan); otherwise it is written before plan returns.
    void plan(bool doSaturate = true, std::future<void> *persisted = nullptr)
    {
        logTimeline("plan start");
        ensureOutputDirectories();
        costModel.setLogging(logCostCalls);
        // The constructor loaded the same file; reload only if it changed.
        const RecordsStamp current = recordsStamp(recordsPath);
        if (!(current == loadedRecords))
        {
            costModel.load(recordsPath);
            loadedRecords = current;
        }

        ensureFullBucket();
        if (!settings.bucket_weights.empty())
            setBucketWeights(settings.bucket_weights);
        updateFullBucketWeight();

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
            foldCleanTensors();
            logTimeline("clean tensors folded");
            ensureCacheCoverage(doSaturate);
            if (settings.saturate_only)
            {
                return;
            }
            if (persisted)
                *persisted = std::async(std::launch::async, [this]() { persistCache(); });
            else
                persistCache();
            isPlanned = true;
        }
    }

    static void logTimeline(const char *event)
    {
        LOG(INFO) << "[Session timeline] " << event << " epoch=" << std::fixed << std::setprecision(3)
                  << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
                  << std::defaultfloat;
    }

    void compile(bool doSaturate = true)
    {
        std::future<void> persisted;
        plan(doSaturate, &persisted);
        logTimeline("planned");

        // Compute exact peak allocation size required per MemSpace across all compiled graphs
        std::unordered_map<MemSpace, uint64_t> peakSizes;
        LOG(INFO) << "Bucket execution times:";
        for (const CompiledGraph &g : cachedGraphs)
        {
            g.cost(true);
            if (std::getenv("TG_PRINT_INSTRUCTIONS"))
            {
                std::unordered_map<std::string, std::pair<uint32_t, float>> k_summary;
                for (const auto &inst : g.instructions)
                {
                    float c = g.nodeCosts.count(inst.eclass_id) ? g.nodeCosts.at(inst.eclass_id) : 0.0f;
                    std::string kname = inst.debugOrigin.empty() ? ("kernel_" + std::to_string(inst.kernel_id.value)) : inst.debugOrigin;
                    k_summary[kname].first++;
                    k_summary[kname].second += c;
                }
                std::vector<std::pair<std::string, std::pair<uint32_t, float>>> sorted_k(k_summary.begin(), k_summary.end());
                std::sort(sorted_k.begin(), sorted_k.end(), [](const auto &a, const auto &b) { return a.second.second > b.second.second; });
                for (const auto &entry : sorted_k)
                {
                    LOG(INFO) << "  [inst summary] " << entry.first << ": count=" << entry.second.first << " total=" << entry.second.second << " ms";
                }
            }
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
        logTimeline("arenas materialized");

        // Write all constants directly to their allocated offsets in memory
        // TODO: currently redundant with constant writing in Executor::run? we should try only write constants here
        size_t total_written = 0;
        for (const CompiledGraph &g : cachedGraphs)
        {
            std::unordered_set<LogicalId> written;
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
                                             graph.constantStaging.at(logical_id)->data(), node.getLogicalSizeBytes());
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
            total_written += written.size();
        }
        logTimeline("constants written");
        std::cout << "Wrote " << total_written << " constants to memory across " << cachedGraphs.size()
                  << " graph(s). Graph has " << graph.constantStaging.size() << " constants." << std::endl;

        if (settings.resident_weights)
        {
            // Resident weights: write each weight at the offsets of the
            // classes holding it that no instruction computes (the
            // preallocated resident input; a bucket may instead read a
            // cached copy, which gets the same data).
            std::unordered_map<LogicalId, std::unordered_set<uint64_t>> weight_offsets;
            uint64_t weight_bytes = 0;
            for (const CompiledGraph &g : cachedGraphs)
            {
                std::unordered_set<EClassId> computed;
                for (const OpInstruction &inst : g.instructions)
                    computed.insert(inst.eclass_id);
                for (const auto &[eclass_id, logical_id] : g.eclass_to_logical)
                {
                    if (!Planner::isResidentWeightCopy(graph, logical_id) || !g.nodeViews.count(eclass_id) ||
                        computed.count(eclass_id))
                        continue;
                    const uint64_t offset = g.nodeViews.at(eclass_id).offset;
                    if (!weight_offsets[logical_id].insert(offset).second)
                        continue;
                    const std::vector<uint8_t> data =
                        TensorResolver::get().readNode(graph.getNode(logical_id).child_ids[0]);
                    if (data.size() != graph.getNode(logical_id).getLogicalSizeBytes())
                        Error::throw_err("[Session] resident weight size mismatch");
                    memManager.write(MemSpace{1, HandleType::CPP}, offset, data.data(), data.size());
                    weight_bytes += data.size();
                }
            }
            std::cout << "Wrote " << weight_offsets.size() << " resident weights (" << weight_bytes / (1024.0 * 1024.0)
                      << " MB)." << std::endl;
        }

        persistent_initialized = false;
        executor = std::make_unique<Executor>(memManager);
        if (persisted.valid())
            persisted.get();
        logTimeline("persisted");
        isCompiled = true;
        logTimeline("compiled");
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
        return runGraph(graphIdx, debugCallback);
    }

    // Runs compiled graph `graphIdx` (compile() must have run) and returns the
    // host pointer of its root tensor.
    const void *runGraph(uint32_t graphIdx, Debug::Callback debugCallback = {})
    {
        if (!isCompiled || graphIdx >= cachedGraphs.size())
            Error::throw_err("[Session.runGraph] invalid graph index " + std::to_string(graphIdx));
        if (!persistent_initialized && !selectedCachedNodes.empty() && graphIdx != fullBucketIdx)
        {
            if (fullBucketIdx >= cachedGraphs.size())
                Error::throw_err("[Session.runGraph] persistent state has no initialization graph");
            executor->run(cachedGraphs[fullBucketIdx], debugCallback);
            persistent_initialized = true;
        }
        executor->run(cachedGraphs[graphIdx], debugCallback);
        if (graphIdx == fullBucketIdx)
            persistent_initialized = true;
        return rootOutputPtr(graphIdx);
    }

    void invalidateCaches() { persistent_initialized = false; }

    // Host pointer of the root tensor written by compiled graph `graphIdx`.
    // Resolved once per graph; the arenas do not move after compile().
    const void *rootOutputPtr(uint32_t graphIdx)
    {
        if (rootPtrs.size() != cachedGraphs.size())
            rootPtrs.assign(cachedGraphs.size(), nullptr);
        if (rootPtrs[graphIdx])
            return rootPtrs[graphIdx];
        const CompiledGraph &cg = cachedGraphs[graphIdx];
        for (const auto &pair : cg.eclass_to_logical)
        {
            if (pair.second != rootId)
                continue;
            EClassId eclass_id = pair.first;
            for (const auto &inst : cg.instructions)
            {
                if (inst.eclass_id == eclass_id || inst.logical_id == rootId)
                {
                    DeviceBuffer *buf = memManager.getBuffer(inst.outBuffer.mem_space);
                    if (buf && buf->getBasePtr())
                        return rootPtrs[graphIdx] = buf->getBasePtr() + inst.outBuffer.offset;
                }
            }
            if (cg.nodeViews.count(eclass_id))
            {
                const TensorView &rootView = cg.nodeViews.at(eclass_id);
                DeviceBuffer *buf = memManager.getBuffer(MemSpace{1, HandleType::CPP});
                if (buf && buf->getBasePtr())
                    return rootPtrs[graphIdx] = buf->getBasePtr() + rootView.offset;
            }
        }
        if (!cg.instructions.empty())
        {
            const OpInstruction &lastInst = cg.instructions.back();
            DeviceBuffer *buf = memManager.getBuffer(lastInst.outBuffer.mem_space);
            if (buf && buf->getBasePtr())
                return rootPtrs[graphIdx] = buf->getBasePtr() + lastInst.outBuffer.offset;
        }
        Error::throw_err("Failed to retrieve valid host output pointer for root node during Session::run");
    }

    const std::vector<CompiledGraph> &getCachedGraphs() const { return cachedGraphs; }
    const std::unordered_set<BaseEClassId> &getSelectedCachedNodes() const { return selectedCachedNodes; }
    void setCachedGraphs(const std::vector<CompiledGraph> &graphs,
                         const std::unordered_set<BaseEClassId> &cached)
    {
        cachedGraphs = graphs;
        selectedCachedNodes = cached;
        cachedBucketWeights = normalizedBucketWeights(manualBuckets);
        persistCache();
    }

    void ensureCacheCoverage(bool doSaturate)
    {
        cachedGraphs.clear();
        selectedCachedNodes.clear();

        ensureFullBucket();
        std::vector<float> bucket_weights = normalizedBucketWeights(manualBuckets);

        Planner planner(costModel, settings);
        auto [graphs, cached_nodes] = planner.planAll(
            rootId, graph, manualBuckets, bucket_weights, doSaturate, repo, minCompileSeconds, brancher);
        logTimeline("planAll returned");

        cachedGraphs = std::move(graphs);
        selectedCachedNodes = std::move(cached_nodes);
        cachedBucketWeights = std::move(bucket_weights);

        if (settings.saturate_only)
        {
            return;
        }

        if (cachedGraphs.size() != manualBuckets.size())
        {
            Error::throw_err("[Session.ensureCacheCoverage] Planned " + std::to_string(cachedGraphs.size()) +
                             " buckets, but expected " + std::to_string(manualBuckets.size()) + ".");
        }

        // Session::plan persists the cache once planning returns.
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
            if (graphCosts.size() != cachedGraphs.size())
            {
                graphCosts.clear();
                for (const CompiledGraph &cached : cachedGraphs)
                    graphCosts.push_back(cached.cost());
            }
            const float gCost = graphCosts[i];
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
        if (disableCompilationCaching || cachePath.empty())
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
