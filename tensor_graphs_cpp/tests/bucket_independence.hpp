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

inline void run()
{
    checkCacheRequirement();
    checkFusedOwnerIdentity();
    for (bool sliding : {false, true})
    for (bool partial : {false, true})
    {
        Graph graph;
        MemoryManager memory;
        const LogicalId input = graph.input({4, 4}, DType::FLOAT32);
        std::vector<float> mask(16, TGConstants::NEG_INF);
        for (uint32_t row = 0; row < 4; ++row)
            for (uint32_t col = sliding && row > 0 ? row - 1 : 0; col <= row; ++col)
                mask[row * 4 + col] = 0;
        const LogicalId mask_id = graph.constant({4, 4}, mask.data(), DType::FLOAT32);
        const LogicalId scores = graph.add(input, mask_id);
        const LogicalId weights = graph.pow(graph.fill(std::exp(1.f), {4, 4}), scores);
        const LogicalId probabilities = graph.div(weights, graph.repeat(graph.sum(weights, 1), 4, 1));
        const LogicalId root = graph.dot(probabilities, input);
        Settings settings;
        settings.disable_compilation_caching = true;
        settings.records_path = "";
        settings.log_cost_calls = false;
        settings.repo_path = "logs/bucket-regression-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        Session session(graph, memory, root, settings);
        ConstantViewRegression::populateAllKernelDummyRecords(session.costModel);
        if (partial)
        {
            session.addBucket({{input, {makeRegion({{0, 1}, {0, 4}})}}},
                              {makeRegion({{0, 1}, {0, 4}})}, 3);
            session.addBucket({{input, {makeRegion({{2, 4}, {0, 4}})}}},
                              {makeRegion({{2, 4}, {0, 4}})}, 1);
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
        std::vector<float> values(16, 1);
        // First use, late/early bucket changes, changed earlier inputs, reset.
        for (uint32_t step = 0; step < 6; ++step)
        {
            const uint32_t bucket = partial ? step % 2 : session.fullBucketIdx;
            const uint32_t first = partial && bucket == 1 ? 2 : 0;
            const uint32_t end = partial && bucket == 0 ? 1 : 4;
            for (uint32_t row = first; row < end; ++row)
                for (uint32_t col = 0; col < 4; ++col)
                    values[row * 4 + col] = static_cast<float>(step + col + 1);
            if (step == 4)
                session.invalidateCaches();
            session.writeInput(input, values.data(), values.size() * sizeof(float));
            const float *output = static_cast<const float *>(session.runGraph(bucket, callback));
            for (uint32_t row = first; row < end; ++row)
                for (uint32_t col = 0; col < 4; ++col)
                {
                    const uint32_t index = row * 4 + col;
                    float denominator = 0;
                    const uint32_t key_start = sliding && row > 0 ? row - 1 : 0;
                    for (uint32_t k = key_start; k <= row; ++k)
                        denominator += std::exp(values[row * 4 + k]);
                    float expected = 0;
                    for (uint32_t k = key_start; k <= row; ++k)
                        expected += std::exp(values[row * 4 + k]) / denominator * values[k * 4 + col];
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
