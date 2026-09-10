#pragma once
#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
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
#include "core/plan/ortools_export.hpp"
#include "core/plan/rule_registry.hpp"
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
    static constexpr uint32_t kCacheFileVersion = 6;

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
    TGStore *repo;
    std::unique_ptr<TGStore> owned_repo;
    bool disableCaching = false;
    float minCompileSeconds = 0.0f;
    std::shared_ptr<SearchDelegate> delegate = nullptr;
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

    Session(Graph &g, MemoryManager &mem, LogicalId root, const Settings &_settings, TGStore *_repo = nullptr,
            std::shared_ptr<SearchDelegate> _delegate = nullptr)
        : graph(g), memManager(mem), rootId(root), settings(_settings), isPlanned(false), isCompiled(false),
          cachePath(_settings.cache_file), nBucketSizes(0), repo(_repo), disableCaching(_settings.disable_caching),
          minCompileSeconds(_settings.min_compile_seconds),
          delegate(_delegate ? _delegate : std::make_shared<HeuristicSearchDelegate>()),
          logCostCalls(_settings.log_cost_calls), costModel(_settings.log_cost_calls, _settings.records_path)
    {
        initRepo(_repo);
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
            TGStore *_repo = nullptr, bool _disableCaching = false, float _minCompileSeconds = 0.0f,
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
        initRepo(_repo);
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
            foldCleanTensors();
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
        LOG(INFO) << "Bucket execution times:";
        for (const CompiledGraph &g : cachedGraphs)
        {
            LOG(INFO) << g.bucket << ": ";
            g.cost(true);
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
        // TODO: currently redundant with constant writing in Executor::run? we should try only write constants here
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

    struct OrtoolsPreparedState
    {
        Planner planner;
        std::vector<Bucket> buckets;
        std::vector<EGraph> bucket_egraphs;
        std::vector<EClassId> bucket_root_eclass_ids;
        std::vector<std::unordered_map<EClassId, LogicalId>> bucket_eclass_to_logicals;
        std::vector<std::vector<ENodeInfo>> bucket_enode_infos;
        std::vector<LogicalId> candidates;
        std::vector<std::vector<uint32_t>> candidate_clean_buckets;
        std::unordered_map<LogicalId, ParallelBuffer> preallocated_buffers;

        OrtoolsPreparedState(CostModel &costModel, const Settings &settings)
            : planner(costModel, settings)
        {
        }
    };

    std::unique_ptr<OrtoolsPreparedState> prepareOrtoolsState(bool doSaturate)
    {
        auto state = std::make_unique<OrtoolsPreparedState>(costModel, settings);
        std::vector<LogicalId> topo = topologicalSort({rootId}, graph);
        Graph temp_graph = graph;
        state->planner.initBaseEGraph(rootId, temp_graph, topo, repo, doSaturate);

        state->buckets = manualBuckets;

        // Determine preallocated buffers for inputs
        state->planner.preallocateLogicalBuffers(graph, {}, state->preallocated_buffers);

        // Candidates for caching
        std::unordered_map<LogicalId, std::vector<uint32_t>> candidate_clean_map;
        if (!disableCaching)
        {
            for (uint32_t b_idx = 0; b_idx < manualBuckets.size(); ++b_idx)
            {
                const Bucket &bucket = manualBuckets[b_idx];
                std::unordered_map<LogicalId, bool> logical_dirty;
                for (LogicalId node_id : topo)
                {
                    bool is_dirty = bucket.inputDirtyRegions.count(node_id) &&
                                   !bucket.inputDirtyRegions.at(node_id).empty();
                    if (!is_dirty)
                    {
                        for (LogicalId parent_id : graph.getNode(node_id).child_ids)
                        {
                            if (logical_dirty[parent_id])
                            {
                                is_dirty = true;
                                break;
                            }
                        }
                    }
                    logical_dirty[node_id] = is_dirty;
                    if (!is_dirty)
                    {
                        candidate_clean_map[node_id].push_back(b_idx);
                    }
                }
            }

            for (LogicalId node_id : topo)
            {
                const TensorNode &node = graph.getNode(node_id);
                const bool runtime_input = node.opType == OpType::INPUT &&
                                           graph.getInputDataType(node_id) == InputDataType::RUNTIME;
                if (node.getSizeBytes() > 0 && (runtime_input || candidate_clean_map.count(node_id)))
                {
                    state->candidates.push_back(node_id);
                    state->candidate_clean_buckets.push_back(
                        candidate_clean_map.count(node_id) ? candidate_clean_map[node_id] : std::vector<uint32_t>{});
                }
            }
        }

        Engine cpu = Engine{0, EngineType::CPU};

        for (uint32_t b_idx = 0; b_idx < manualBuckets.size(); ++b_idx)
        {
            const Bucket &bucket = manualBuckets[b_idx];
            EGraph egraph = state->planner.baseState.egraph;
            auto eclassToLogical = state->planner.baseState.eclassToLogical;

            // Inject CACHE enodes for candidates that are clean in this bucket
            for (size_t c_idx = 0; c_idx < state->candidates.size(); ++c_idx)
            {
                LogicalId cand_id = state->candidates[c_idx];
                const auto &clean_b = state->candidate_clean_buckets[c_idx];
                bool is_clean = std::find(clean_b.begin(), clean_b.end(), b_idx) != clean_b.end();
                if (is_clean && state->planner.baseState.nodeToEClass.count(cand_id))
                {
                    EClassId canon_id = egraph.find(state->planner.baseState.nodeToEClass.at(cand_id));
                    bool has_cache = false;
                    for (ENodeId eid : egraph.getEClass(canon_id).enodes)
                    {
                        if (egraph.getENode(eid).getOpType() == OpType::CACHE)
                        {
                            has_cache = true;
                            break;
                        }
                    }
                    if (!has_cache)
                    {
                        const auto &cls = egraph.getEClass(canon_id);
                        ENode cache_node = ENode(KernelId{0}, OpType::CACHE, "", {}, cls.shape, cls.strides, cls.dtype,
                                                cls.mem_space, {cpu}, toString(cand_id));
                        egraph.addENode(canon_id, cache_node);
                    }
                }
            }

            std::unordered_set<EClassId> protected_classes;
            for (LogicalId cand_id : state->candidates)
            {
                if (state->planner.baseState.nodeToEClass.count(cand_id))
                {
                    protected_classes.insert(egraph.find(state->planner.baseState.nodeToEClass.at(cand_id)));
                }
            }

            state->planner.injectInputPartialPaths(egraph, graph, bucket.inputDirtyRegions, {},
                                                  state->planner.baseState.nodeToEClass, eclassToLogical);
            state->planner.injectOutputPartialPaths(egraph, graph, rootId, bucket.outputNeededRegion, {},
                                                   state->planner.baseState.nodeToEClass, eclassToLogical);

            if (doSaturate && settings.do_saturate)
            {
                state->planner.saturate(egraph, protected_classes, eclassToLogical, true, false, repo);
            }

            std::unordered_map<EClassId, LogicalId> updated_eclass_to_logical;
            for (const auto &kv : eclassToLogical)
            {
                updated_eclass_to_logical[egraph.find(kv.first)] = kv.second;
            }
            eclassToLogical = std::move(updated_eclass_to_logical);

            auto enode_infos = state->planner.computeENodeInfos(egraph, eclassToLogical, {}, false);
            state->planner.pruneEGraph(egraph, enode_infos);

            EClassId root_eclass_id = egraph.findConst(state->planner.baseState.nodeToEClass.at(rootId));

            state->bucket_egraphs.push_back(std::move(egraph));
            state->bucket_root_eclass_ids.push_back(root_eclass_id);
            state->bucket_eclass_to_logicals.push_back(std::move(eclassToLogical));
            state->bucket_enode_infos.push_back(std::move(enode_infos));
        }

        return state;
    }

    std::string exportOrtoolsProblem(bool doSaturate = true)
    {
        auto state = prepareOrtoolsState(doSaturate);
        nlohmann::json prob = ortools_export::serializeProblem(
            state->buckets, state->bucket_egraphs, state->bucket_root_eclass_ids,
            state->bucket_eclass_to_logicals, state->bucket_enode_infos, state->candidates,
            state->candidate_clean_buckets, graph, state->preallocated_buffers, settings);
        return prob.dump();
    }

    void applyOrtoolsSolution(const std::string &solution_json_str, OrtoolsPreparedState &state)
    {
        nlohmann::json sol = nlohmann::json::parse(solution_json_str);
        std::unordered_map<LogicalId, MemSpace> selected_cached;
        std::vector<ExtractionResult> extractions;
        if (!ortools_export::deserializeSolution(sol, selected_cached, extractions))
        {
            Error::throw_err("[Session.applyOrtoolsSolution] Failed to deserialize solution JSON.");
        }

        if (extractions.size() != manualBuckets.size())
        {
            Error::throw_err("[Session.applyOrtoolsSolution] Solution has " + std::to_string(extractions.size()) +
                             " buckets, expected " + std::to_string(manualBuckets.size()));
        }

        cachedGraphs.clear();
        for (size_t b = 0; b < manualBuckets.size(); ++b)
        {
            Planner thread_planner(costModel, settings);
            thread_planner.baseState = state.planner.baseState;
            thread_planner.baseStateInitialized = true;

            CompiledGraph cg = thread_planner.buildCompiledGraph(
                rootId, graph, state.bucket_egraphs[b], state.planner.baseState.nodeToEClass,
                extractions[b], selected_cached, state.bucket_eclass_to_logicals[b], state.bucket_enode_infos[b]);
            cg.bucket = manualBuckets[b];
            cachedGraphs.push_back(std::move(cg));
        }

        selectedCachedNodes = std::move(selected_cached);
        cachedBucketWeights = normalizedBucketWeights(manualBuckets);
        persistCache();
    }

    void ensureCacheCoverageOrtools(bool doSaturate)
    {
        std::cout << "[Session.ensureCacheCoverageOrtools] Saturating E-Graph and preparing OR-Tools CP-SAT problem..."
                  << std::endl;
        auto state = prepareOrtoolsState(doSaturate);
        nlohmann::json prob = ortools_export::serializeProblem(
            state->buckets, state->bucket_egraphs, state->bucket_root_eclass_ids,
            state->bucket_eclass_to_logicals, state->bucket_enode_infos, state->candidates,
            state->candidate_clean_buckets, graph, state->preallocated_buffers, settings);

        std::filesystem::create_directories("benchmarks");
        std::string prob_path = "benchmarks/ortools_problem.json";
        std::string sol_path = "benchmarks/ortools_solution.json";

        {
            std::ofstream f(prob_path);
            f << prob.dump(2);
        }
        if (std::filesystem::exists(sol_path))
        {
            std::filesystem::remove(sol_path);
        }

        bool ok = ortools_export::runOrtoolsSolverProcess(prob_path, sol_path);
        if (!ok)
        {
            Error::throw_err("[Session.ensureCacheCoverageOrtools] Solver process failed to produce solution.");
        }

        std::string sol_str;
        {
            std::ifstream f(sol_path);
            std::stringstream ss;
            ss << f.rdbuf();
            sol_str = ss.str();
        }

        applyOrtoolsSolution(sol_str, *state);
        std::cout << "[Session.ensureCacheCoverageOrtools] Successfully applied OR-Tools solution." << std::endl;
    }

    const std::vector<CompiledGraph> &getCachedGraphs() const { return cachedGraphs; }
    const std::unordered_map<LogicalId, MemSpace> &getSelectedCachedNodes() const { return selectedCachedNodes; }
    void setCachedGraphs(const std::vector<CompiledGraph> &graphs,
                         const std::unordered_map<LogicalId, MemSpace> &cached)
    {
        cachedGraphs = graphs;
        selectedCachedNodes = cached;
        cachedBucketWeights = normalizedBucketWeights(manualBuckets);
        persistCache();
    }

    void ensureCacheCoverage(bool doSaturate)
    {
        if (settings.use_ortools)
        {
            ensureCacheCoverageOrtools(doSaturate);
            return;
        }
        cachedGraphs.clear();
        selectedCachedNodes.clear();

        std::shared_ptr<SearchDelegate> search_delegate = delegate;
        if (!search_delegate)
        {
            search_delegate = std::make_shared<HeuristicSearchDelegate>();
        }

        Planner planner(costModel, settings);
        std::vector<LogicalId> topo = topologicalSort({rootId}, graph);
        Graph temp_graph = graph;
        planner.initBaseEGraph(rootId, temp_graph, topo, repo, doSaturate);

        const std::vector<float> bucket_weights = normalizedBucketWeights(manualBuckets);
        std::unordered_map<LogicalId, MemSpace> best_cached_nodes;

        std::vector<LogicalId> candidates;
        if (!disableCaching)
        {
            // A node is a cache candidate when it can remain clean in a bucket,
            // plus runtime inputs.  The latter are deliberately included even
            // when every bucket dirties a slice: their cached backing buffer is
            // the starting point for partial recomputation of decode/KV state.
            std::unordered_map<LogicalId, bool> clean_in_any_bucket;
            for (const Bucket &bucket : manualBuckets)
            {
                std::unordered_map<LogicalId, bool> logical_dirty;
                for (LogicalId node_id : topo)
                {
                    bool is_dirty = bucket.inputDirtyRegions.count(node_id) &&
                                   !bucket.inputDirtyRegions.at(node_id).empty();
                    if (!is_dirty)
                    {
                        for (LogicalId parent_id : graph.getNode(node_id).child_ids)
                        {
                            if (logical_dirty[parent_id])
                            {
                                is_dirty = true;
                                break;
                            }
                        }
                    }
                    logical_dirty[node_id] = is_dirty;
                    clean_in_any_bucket[node_id] = clean_in_any_bucket[node_id] || !is_dirty;
                }
            }

            bool has_partial_runtime_input_bucket = false;
            for (const Bucket &bucket : manualBuckets)
            {
                for (const auto &dirty_input : bucket.inputDirtyRegions)
                {
                    if (!graph.hasNode(dirty_input.first) || graph.getNode(dirty_input.first).opType != OpType::INPUT)
                        continue;
                    const auto &shape = graph.getNode(dirty_input.first).getShape();
                    for (const Region &region : dirty_input.second)
                    {
                        bool is_full = region.region.size() == shape.size();
                        for (size_t dim = 0; is_full && dim < shape.size(); ++dim)
                            is_full = region.region[dim].start == 0 && region.region[dim].stop == shape[dim];
                        has_partial_runtime_input_bucket = has_partial_runtime_input_bucket || !is_full;
                    }
                }
            }
            for (LogicalId node_id : topo)
            {
                const TensorNode &node = graph.getNode(node_id);
                const bool runtime_input = node.opType == OpType::INPUT &&
                                           graph.getInputDataType(node_id) == InputDataType::RUNTIME;
                // Decode searches should retain only their runtime input state.
                // Allowing every clean intermediate into the first heuristic
                // candidate turns one decode choice into a huge all-buffer plan.
                if (node.getSizeBytes() > 0 && (runtime_input || (!has_partial_runtime_input_bucket && clean_in_any_bucket[node_id])))
                    candidates.push_back(node_id);
            }
            std::stable_sort(candidates.begin(), candidates.end(), [&](LogicalId a, LogicalId b) {
                const bool a_runtime = graph.getNode(a).opType == OpType::INPUT &&
                                       graph.getInputDataType(a) == InputDataType::RUNTIME;
                const bool b_runtime = graph.getNode(b).opType == OpType::INPUT &&
                                       graph.getInputDataType(b) == InputDataType::RUNTIME;
                return a_runtime > b_runtime;
            });
        }

        std::vector<MemSpace> avail_mem_spaces;
        for (const auto &entry : settings.mem_caps)
        {
            if (entry.first.type != HandleType::STORAGE)
                avail_mem_spaces.push_back(entry.first);
        }
        std::sort(avail_mem_spaces.begin(), avail_mem_spaces.end(), [](const MemSpace &a, const MemSpace &b) {
            return a.type != b.type ? a.type < b.type : a.idx < b.idx;
        });

        float best_cost = TGConstants::INF;
        TimeoutChecker timeout_checker(minCompileSeconds);
        auto cache_iter = makeConfiguredCacheIterator(graph, candidates, avail_mem_spaces, search_delegate, settings,
                                                      &best_cost, &timeout_checker);
        std::unordered_map<LogicalId, MemSpace> current_cache;
        const auto search_start = std::chrono::high_resolution_clock::now();

        // Cache selection and bucket planning are one search: each selection
        // is evaluated by planning all buckets in parallel, with the same repo
        // and rewrite space used for the final compiled graphs.
        for (uint32_t eval_count = 0; cache_iter.getNextCacheSelection(current_cache); ++eval_count)
        {
            LOG(DEBUG) << "# cached nodes: " << current_cache.size();
            std::unordered_map<LogicalId, ParallelBuffer> preallocated;
            planner.preallocateLogicalBuffers(graph, current_cache, preallocated);
            std::vector<float> bucket_costs(manualBuckets.size(), TGConstants::INF);
            std::vector<CompiledGraph> candidate_graphs(manualBuckets.size());
            std::atomic<bool> failed{false};
            std::exception_ptr err_ptr = nullptr;
            std::mutex err_mutex;

            ThreadPool::get().parallel_for(static_cast<uint32_t>(manualBuckets.size()), [&](uint32_t bucket_idx) {
                try
                {
                    Planner thread_planner(costModel, settings);
                    thread_planner.baseState = planner.baseState;
                    thread_planner.baseStateInitialized = true;
                    CompiledGraph candidate = thread_planner.plan(rootId, graph, manualBuckets[bucket_idx], current_cache,
                                                                 doSaturate, true, repo, preallocated, minCompileSeconds, search_delegate);
                    bucket_costs[bucket_idx] = candidate.cost();
                    candidate.bucket = manualBuckets[bucket_idx];
                    candidate_graphs[bucket_idx] = std::move(candidate);
                }
                catch (...)
                {
                    failed.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(err_mutex);
                    if (!err_ptr)
                        err_ptr = std::current_exception();
                }
            });

            if (err_ptr)
            {
                std::rethrow_exception(err_ptr);
            }

            if (!failed.load(std::memory_order_relaxed))
            {
                double weighted_cost = 0.0;
                for (size_t bucket_idx = 0; bucket_idx < bucket_costs.size(); ++bucket_idx)
                {
                    weighted_cost += static_cast<double>(bucket_weights[bucket_idx]) * bucket_costs[bucket_idx];
                    if (search_delegate)
                        search_delegate->on_bucket_leaf_evaluated(static_cast<uint32_t>(bucket_idx), bucket_costs[bucket_idx]);
                }
                const float cost = static_cast<float>(weighted_cost);
                if (search_delegate)
                {
                    search_delegate->set_best_cost_ptr(&best_cost);
                    search_delegate->on_leaf_evaluated(cost);
                }
                if (cost < best_cost)
                {
                    best_cost = cost;
                    best_cached_nodes = current_cache;
                    cachedGraphs = std::move(candidate_graphs);
                }
            }

            if (best_cost < TGConstants::INF && minCompileSeconds == 0.0f)
                break;
            if (minCompileSeconds > 0.0f &&
                std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - search_start).count() >=
                    minCompileSeconds)
                break;
        }
        if (search_delegate)
            search_delegate->set_best_cost_ptr(nullptr);

        selectedCachedNodes = std::move(best_cached_nodes);
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
