#pragma once

#include "tests/constant_view_regression.hpp"

namespace BucketIndependence
{
inline void checkFusedOwnerIdentity()
{
    EGraph graph;
    const MemSpace ram{1, HandleType::CPP};
    const EClassId update = graph.addEClass({1, 2}, {2, 1}, DType::FLOAT32, ram);
    graph.addENode(update, ENode(KernelId{0}, OpType::INPUT, "", {}, {1, 2}, {2, 1},
                                DType::FLOAT32, ram, {Engine(0, EngineType::CPU)}, "updates"));
    const std::vector<EClassId> inputs{update, graph.addIntConst({0, 0}), graph.addIntConst({1, 2}),
                                      graph.addIntConst({1, 1}), graph.addIntConst({2, 2})};
    const KernelEntry *kernel = nullptr;
    for (const auto &[id, entry] : KernelRegistry::get().getAllKernels())
        if (entry.opName == "Scatter_F32_Rows")
            kernel = &entry;
    if (!kernel)
        Error::throw_err("[BucketIndependence] missing fused scatter kernel");
    FusionRule fusion(false);
    FusionRule::FusedCandidate candidate;
    candidate.kernel = kernel;
    candidate.target_mem_space = ram;
    candidate.mapped_engines = {Engine(0, EngineType::CPU)};
    candidate.child_need_copy.assign(5, false);
    candidate.child_need_contig.assign(5, false);
    const std::unordered_set<EClassId> protected_classes;
    RuleCtx context{graph, protected_classes, nullptr};
    std::vector<EClassId> owners;
    for (uint32_t id : {1u, 2u})
    {
        const EClassId owner = graph.addEClass({2, 2}, {2, 1}, DType::FLOAT32, ram);
        graph.getEClass(owner).base_eclass_id = BaseEClassId{id};
        const ENodeId source{static_cast<uint32_t>(graph.enodes.size())};
        const std::string tag = "scatter-into:b" + std::to_string(id);
        graph.addENode(owner, ENode(KernelId{0}, OpType::SCATTER, "", inputs, {2, 2}, {2, 1},
                                   DType::FLOAT32, ram, {Engine(0, EngineType::CPU)}, tag));
        fusion.materializeFusedCandidates(context, {candidate}, inputs, source);
        bool fused = false;
        for (ENodeId node_id : graph.getEClass(owner).enodes)
            if (graph.getENode(node_id).getKernelId() == kernel->uid)
            {
                fused = true;
                if (graph.getENode(node_id).getContentHash() != tag)
                    Error::throw_err("[BucketIndependence] fusion dropped scatter owner");
            }
        if (!fused)
            Error::throw_err("[BucketIndependence] fused owner fixture was not exercised");
        owners.push_back(owner);
    }
    graph.rebuild();
    if (graph.findConst(owners[0]) == graph.findConst(owners[1]))
        Error::throw_err("[BucketIndependence] different scatter owners became equal");
}

inline void checkCacheRequirement()
{
    EGraph graph;
    const MemSpace ram{1, HandleType::CPP};
    const EClassId owner = graph.addEClass({2}, {1}, DType::FLOAT32, ram);
    graph.getEClass(owner).base_eclass_id = BaseEClassId{1};
    graph.addENode(owner, ENode(KernelId{0}, OpType::SCATTER, "", {}, {2}, {1},
                               DType::FLOAT32, ram, {Engine(0, EngineType::CPU)}, "scatter-into:b1"));
    plan::SearchState state;
    state.bucket_egraphs.push_back(graph);
    state.buckets.push_back(Bucket{});
    state.selected_vars.resize(1);
    plan::VarInfo selected_info;
    selected_info.type = plan::VarType::SELECTED;
    selected_info.bucket_idx = 0;
    selected_info.eclass_id = owner;
    const plan::VarId selected = state.addVar(selected_info, plan::Domain::makeFixed(1, true));
    state.selected_vars[0][owner] = selected;
    plan::CacheRequirementPropagator propagator;
    std::vector<plan::VarId> work;
    if (propagator.propagate(state, selected, work))
        Error::throw_err("[BucketIndependence] scatter accepted without a cache candidate");
    plan::VarInfo cache_info;
    cache_info.type = plan::VarType::CACHED;
    cache_info.base_eclass_id = BaseEClassId{1};
    const plan::VarId cache = state.addVar(cache_info, plan::Domain::makeMask(3));
    state.cached_vars[BaseEClassId{1}] = cache;
    if (!propagator.propagate(state, selected, work) || !state.domains[cache].isFixed() ||
        state.domains[cache].fixedValue() != 1)
        Error::throw_err("[BucketIndependence] scatter did not require persistence");
}

// A cost record of runtime 1 for every registered kernel, at the kernel's
// registered dummy input shapes (workload models need real shapes).
inline void writeDummyRecords(const std::string &path)
{
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
        Error::throw_err("[BucketIndependence] cannot write dummy records to " + path);
    BinaryWriter writer(file);
    for (const auto &[uid, kernel] : KernelRegistry::get().getAllKernels())
    {
        Record record;
        record.kernelId = uid;
        record.buildContextId = BUILD_CONTEXT_ID;
        record.hwTag = HW_TAG;
        record.runTime = 1.0f;
        record.output_mem_space = kernel.output_mem_space;
        record.engines = kernel.engines;
        if (kernel.dummyShapes.size() != kernel.dtypes.size())
            Error::throw_err("[BucketIndependence] kernel " + kernel.opName + " has " +
                             std::to_string(kernel.dummyShapes.size()) + " dummy shapes for " +
                             std::to_string(kernel.dtypes.size()) + " input dtypes");
        for (size_t i = 0; i < kernel.dummyShapes.size(); ++i)
        {
            record.inputShapes.push_back(kernel.dummyShapes[i]);
            record.inputStrides.push_back(calcContiguousStrides(kernel.dummyShapes[i]));
            record.inputDTypes.push_back(kernel.dtypes[i]);
        }
        if (!kernel.dummyShapes.empty())
        {
            record.outputShape = kernel.dummyShapes.front();
            record.outputStrides = calcContiguousStrides(record.outputShape);
        }
        tg_serialize(writer, record);
    }
}

inline void run()
{
    checkCacheRequirement();
    checkFusedOwnerIdentity();
    for (bool sliding : {false, true})
    for (bool partial : {false, true})
    {
        Graph graph;
        MemoryManager memory;
        // Large enough that recomputing only the dirty rows (and keeping the
        // rest of each persistent value) is cheaper than recomputing all.
        constexpr uint32_t N = 16;
        const uint32_t late_first = N / 2;
        const LogicalId input = graph.input({N, N}, DType::FLOAT32);
        std::vector<float> mask(N * N, TGConstants::NEG_INF);
        for (uint32_t row = 0; row < N; ++row)
            for (uint32_t col = sliding && row > 0 ? row - 1 : 0; col <= row; ++col)
                mask[row * N + col] = 0;
        const LogicalId mask_id = graph.constant({N, N}, mask.data(), DType::FLOAT32);
        const LogicalId scores = graph.add(input, mask_id);
        const LogicalId weights = graph.pow(graph.fill(std::exp(1.f), {N, N}), scores);
        const LogicalId probabilities = graph.div(weights, graph.repeat(graph.sum(weights, 1), N, 1));
        const LogicalId root = graph.dot(probabilities, input);
        Settings settings;
        settings.disable_compilation_caching = true;
        // Search to the end: the first greedy plans recompute everything,
        // which would not exercise persistent updates (about 20 s per case).
        settings.min_compile_seconds = -1.0f;
        settings.log_cost_calls = false;
        const std::string run_tag = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        settings.repo_path = "logs/bucket-regression-" + run_tag;
        // Session::plan() reloads cost records from records_path, so the dummy
        // records must be in that file (records added to the session's cost
        // model directly would be dropped).
        settings.records_path = "logs/bucket-regression-records-" + run_tag + ".bin";
        writeDummyRecords(settings.records_path);
        Session session(graph, memory, root, settings);
        if (partial)
        {
            session.addBucket({{input, {makeRegion({{0, 1}, {0, N}})}}},
                              {makeRegion({{0, 1}, {0, N}})}, 3);
            session.addBucket({{input, {makeRegion({{late_first, N}, {0, N}})}}},
                              {makeRegion({{late_first, N}, {0, N}})}, 1);
        }
        const auto start = std::chrono::steady_clock::now();
        session.compile(true);
        const double compile_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (session.cachedGraphs.size() != (partial ? 3 : 1))
            Error::throw_err("[BucketIndependence] requested bucket was lost");
        if (partial && session.selectedCachedNodes.empty())
            Error::throw_err("[BucketIndependence] fixture did not exercise persistent updates");
        if (partial && (session.cachedBucketWeights[0] != .75f || session.cachedBucketWeights[1] != .25f ||
                        session.cachedBucketWeights[2] != 0))
            Error::throw_err("[BucketIndependence] workload weights changed");
        if (partial)
            for (uint32_t i = 0; i < 2; ++i)
                if (!(session.cachedGraphs[i].bucket == session.manualBuckets[i]))
                    Error::throw_err("[BucketIndependence] bucket domain/order changed");
        uint32_t callbacks = 0;
        Debug::Callback callback = [&](LogicalId, std::string &, const KernelContext &, const void *) { ++callbacks; };
        std::vector<float> values(N * N, 1);
        // First use, late/early bucket changes, changed earlier inputs, reset.
        for (uint32_t step = 0; step < 6; ++step)
        {
            const uint32_t bucket = partial ? step % 2 : session.fullBucketIdx;
            const uint32_t first = partial && bucket == 1 ? late_first : 0;
            const uint32_t end = partial && bucket == 0 ? 1 : N;
            for (uint32_t row = first; row < end; ++row)
                for (uint32_t col = 0; col < N; ++col)
                    values[row * N + col] = 0.1f * static_cast<float>(step + col + 1);
            if (step == 4)
                session.invalidateCaches();
            session.writeInput(input, values.data(), values.size() * sizeof(float));
            const float *output = static_cast<const float *>(session.runGraph(bucket, callback));
            for (uint32_t row = first; row < end; ++row)
                for (uint32_t col = 0; col < N; ++col)
                {
                    const uint32_t index = row * N + col;
                    float denominator = 0;
                    const uint32_t key_start = sliding && row > 0 ? row - 1 : 0;
                    for (uint32_t k = key_start; k <= row; ++k)
                        denominator += std::exp(values[row * N + k]);
                    float expected = 0;
                    for (uint32_t k = key_start; k <= row; ++k)
                        expected += std::exp(values[row * N + k]) / denominator * values[k * N + col];
                    if (!std::isfinite(output[index]) || std::abs(output[index] - expected) > 1e-5f)
                        Error::throw_err("[BucketIndependence] wrong masked output at " + std::to_string(index) +
                            " step=" + std::to_string(step) + " expected=" + std::to_string(expected) +
                            " actual=" + std::to_string(output[index]));
                }
        }
        if (!callbacks)
            Error::throw_err("[BucketIndependence] runGraph bypassed verification callback");
        double modeled_cost = 0;
        uint64_t peak_bytes = 0;
        if (session.cachedBucketWeights.size() != session.cachedGraphs.size())
            Error::throw_err("[BucketIndependence] missing normalized bucket weights");
        for (size_t i = 0; i < session.cachedGraphs.size(); ++i)
        {
            const CompiledGraph &compiled = session.cachedGraphs[i];
            modeled_cost += session.cachedBucketWeights[i] * compiled.cost();
            for (const auto &instruction : compiled.instructions)
                if (instruction.outBuffer.offset >= 0)
                    peak_bytes = std::max(peak_bytes, static_cast<uint64_t>(instruction.outBuffer.offset) + instruction.outBuffer.size);
        }
        std::cout << std::setprecision(9) << "BUCKET_BASELINE partial=" << partial << " sliding=" << sliding
                  << " compile_seconds=" << compile_seconds << " modeled_cost=" << modeled_cost
                  << " arena_extent_bytes=" << peak_bytes << " cached_values=" << session.selectedCachedNodes.size()
                  << " correct=1" << std::endl;
    }
    // The accepted sentinel convention must not swallow nearby finite values,
    // NaNs, or IEEE infinity as if they were the finite sentinel.
    const std::vector<float> data = {TGConstants::NEG_INF, -2e9f, -1e8f, 0.f,
        std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::infinity()};
    std::vector<uint8_t> bytes(data.size() * sizeof(float));
    std::memcpy(bytes.data(), data.data(), bytes.size());
    assertRegionListEquals(bucket_regions::unmaskedRegions(bytes, {6}),
                           {makeRegion({{1, 6}})}, "exact mask sentinel");
    std::cout << "bucket independence tests passed\n";
}
}
