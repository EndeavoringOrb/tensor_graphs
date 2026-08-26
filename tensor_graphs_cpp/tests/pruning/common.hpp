#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/constants.hpp"
#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/misc.hpp"
#include "core/plan/planner.hpp"
#include "core/plan/rule_registry.hpp"
#include "core/plan/validators/mem.hpp"
#include "core/types.hpp"
#include "tests/common.hpp"

namespace prune_test
{

// =============================================================================
// Linear Regression Line Fitting (Reusing CostModel's Matrix Ops)
// =============================================================================
struct FitResult
{
    double slope = 0.0;
    double intercept = 0.0;
    double avg_ms = 0.0;
};

inline FitResult fitLine(const std::vector<double> &x, const std::vector<double> &y)
{
    FitResult res;
    int K = static_cast<int>(x.size());
    if (K == 0)
        return res;

    double sum_y = 0.0;
    for (double v : y)
        sum_y += v;
    res.avg_ms = sum_y / K;

    if (K == 1)
    {
        res.intercept = y[0];
        res.slope = 0.0;
        return res;
    }

    CostModel::Matrix X(K, 2);
    CostModel::Matrix Y(K, 1);
    for (int i = 0; i < K; ++i)
    {
        X(i, 0) = 1.0;
        X(i, 1) = x[i];
        Y(i, 0) = y[i];
    }

    CostModel::Matrix Xt = CostModel::transpose(X);
    CostModel::Matrix XtX = CostModel::multiply(Xt, X);
    CostModel::Matrix XtY = CostModel::multiply(Xt, Y);

    if (CostModel::invert(XtX))
    {
        CostModel::Matrix W = CostModel::multiply(XtX, XtY);
        res.intercept = W(0, 0);
        res.slope = W(1, 0);
    }
    else
    {
        double mean_x = 0.0, mean_y = res.avg_ms;
        for (double v : x)
            mean_x += v;
        mean_x /= K;

        double num = 0.0, den = 0.0;
        for (int i = 0; i < K; ++i)
        {
            num += (x[i] - mean_x) * (y[i] - mean_y);
            den += (x[i] - mean_x) * (x[i] - mean_x);
        }
        res.slope = (den > 1e-12) ? (num / den) : 0.0;
        res.intercept = mean_y - res.slope * mean_x;
    }
    return res;
}

// =============================================================================
// Kernel registration helpers
// =============================================================================
struct MockKernels
{
    static constexpr uint64_t kAddInplace = 0xBB0001ULL;
    static constexpr uint64_t kMulInplace = 0xBB0002ULL;
    static constexpr uint64_t kNegInplace = 0xBB0003ULL;
    static constexpr uint64_t kFmaV1 = 0xED0001ULL;
    static constexpr uint64_t kFmaV2 = 0xED0002ULL;
    static constexpr uint64_t kFmaV3Inplace = 0xED0003ULL;
    static constexpr uint64_t kAddBigOutplace = 0xAA0001ULL;

    static bool registered;
    static void ensureRegistered()
    {
        if (registered)
            return;

        // In-place ADD/MUL/NEG for bufferize tests.
        KernelRegistry::get().registerKernel(
            KernelId{kAddInplace}, OpType::ADD, "", 2, 2, nullptr, nullptr, nullptr, {0, 1}, false, true, nullptr,
            MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
            {{8, 8}, {8, 8}}, {false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
        KernelRegistry::get().registerKernel(
            KernelId{kMulInplace}, OpType::MUL, "", 2, 2, nullptr, nullptr, nullptr, {0, 1}, false, true, nullptr,
            MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
            {{8, 8}, {8, 8}}, {false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
        KernelRegistry::get().registerKernel(KernelId{kNegInplace}, OpType::NEGATE, "", 1, 1, nullptr, nullptr, nullptr,
                                             {0}, false, true, nullptr, MemSpace(1, HandleType::CPP),
                                             {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{8, 8}}, {false},
                                             {{MemSpace(1, HandleType::CPP)}});
        // Fused FMA variants for ENode domination tests.
        KernelRegistry::get().registerKernel(
            KernelId{kFmaV1}, OpType::FUSED, "fma_v1", 2, 2, nullptr, nullptr, nullptr, {}, false, false, nullptr,
            MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
            {{8, 8}, {8, 8}}, {false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
        KernelRegistry::get().registerKernel(
            KernelId{kFmaV2}, OpType::FUSED, "fma_v2", 2, 2, nullptr, nullptr, nullptr, {}, false, false, nullptr,
            MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
            {{8, 8}, {8, 8}}, {false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
        KernelRegistry::get().registerKernel(KernelId{kFmaV3Inplace}, OpType::FUSED, "fma_v3_inplace", 2, 2, nullptr,
                                             nullptr, nullptr, {0}, false, false, nullptr, MemSpace(1, HandleType::CPP),
                                             {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                                             {{8, 8}, {8, 8}}, {false, false},
                                             {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
        // Out-of-place big ADD.
        KernelRegistry::get().registerKernel(KernelId{kAddBigOutplace}, OpType::ADD, "", 2, 2, nullptr, nullptr,
                                             nullptr, {}, false, false, nullptr, MemSpace(1, HandleType::CPP),
                                             {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                                             {{1024, 1024}, {1024, 1024}}, {false, false},
                                             {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

        registered = true;
    }
};

inline bool MockKernels::registered = false;

// =============================================================================
// Cost-model record builder for mock kernels
// =============================================================================
struct MockRecords
{
    static void populate(CostModel &costModel, float runtime = 1.0f)
    {
        MockKernels::ensureRegistered();

        for (uint64_t kid : {MockKernels::kAddInplace, MockKernels::kMulInplace, MockKernels::kNegInplace,
                             MockKernels::kAddBigOutplace})
        {
            Record r;
            r.kernelId = KernelId{kid};
            r.buildContextId = BUILD_CONTEXT_ID;
            r.hwTag = HW_TAG;
            r.outputShape = {8, 8};
            r.outputStrides = {8, 1};
            r.outputDType = DType::FLOAT32;
            r.inputShapes = (kid == MockKernels::kNegInplace) ? std::vector<std::vector<uint32_t>>{{8, 8}}
                                                              : std::vector<std::vector<uint32_t>>{{8, 8}, {8, 8}};
            r.inputStrides = (kid == MockKernels::kNegInplace) ? std::vector<std::vector<uint64_t>>{{8, 1}}
                                                               : std::vector<std::vector<uint64_t>>{{8, 1}, {8, 1}};
            r.inputDTypes = (kid == MockKernels::kNegInplace) ? std::vector<DType>{DType::FLOAT32}
                                                              : std::vector<DType>{DType::FLOAT32, DType::FLOAT32};
            r.runTime = runtime;
            costModel.records[r.kernelId].push_back(r);
        }

        // FMA variants: v1 slower (2.5ms), v2/v3 faster (0.5ms)
        Record r1;
        r1.kernelId = KernelId{MockKernels::kFmaV1};
        r1.buildContextId = BUILD_CONTEXT_ID;
        r1.hwTag = HW_TAG;
        r1.inputShapes = {{8, 8}, {8, 8}};
        r1.outputShape = {8, 8};
        r1.outputStrides = {8, 1};
        r1.inputStrides = {{8, 1}, {8, 1}};
        r1.inputDTypes = {DType::FLOAT32, DType::FLOAT32};
        r1.outputDType = DType::FLOAT32;
        r1.inputConstants = {{}, {}};
        r1.output_mem_space = MemSpace{1, HandleType::CPP};
        r1.engines = {Engine{0, EngineType::CPU}};
        r1.input_mem_spaces = {MemSpace{1, HandleType::CPP}, MemSpace{1, HandleType::CPP}};
        r1.runTime = 2.5f;
        costModel.records[r1.kernelId].push_back(r1);

        Record r2 = r1;
        r2.kernelId = KernelId{MockKernels::kFmaV2};
        r2.runTime = 0.5f;
        costModel.records[r2.kernelId].push_back(r2);

        Record r3 = r1;
        r3.kernelId = KernelId{MockKernels::kFmaV3Inplace};
        r3.runTime = 0.5f;
        costModel.records[r3.kernelId].push_back(r3);

        // Populate mock records for ALL registered kernels in KernelRegistry so no missing records warning can ever
        // occur
        populateDummyRecords(costModel, EGraph{}, runtime);
    }
};

// =============================================================================
// MockCtx
// =============================================================================
struct MockCtx
{
    CostModel costModel;
    std::unordered_map<MemSpace, uint64_t> mem_caps;
    Settings settings;
    std::unordered_map<EClassId, uint32_t> selection_map;

    MockCtx(uint64_t default_mem_cap = 1024ULL * 1024 * 1024)
        : costModel(false, ""), mem_caps({{MemSpace{1, HandleType::CPP}, default_mem_cap}})
    {
        settings.mem_caps = mem_caps;
        setupTestSettings(settings, true);
        MockKernels::ensureRegistered();
        MockRecords::populate(costModel);
    }

    EClassId build(Graph &graph, LogicalId rootId, bool strictCache = false,
                   std::function<void(EGraph &, const std::unordered_map<LogicalId, EClassId> &)> egraph_hook = nullptr)
    {
        std::vector<LogicalId> topo = topologicalSort({rootId}, graph);
        Planner planner(costModel, settings);
        planner.initBaseEGraph(rootId, graph, topo, nullptr);
        populateDummyRecords(costModel, planner.baseState.egraph);

        EGraph egraph = planner.baseState.egraph;
        auto eclassToLogical = planner.baseState.eclassToLogical;
        auto nodeToEClass = planner.baseState.nodeToEClass;

        if (egraph_hook)
        {
            egraph_hook(egraph, nodeToEClass);
        }

        std::unordered_map<LogicalId, MemSpace> cachedNodes;
        auto enodeInfos = planner.computeENodeInfos(egraph, eclassToLogical, cachedNodes, strictCache);

        this->egraph = std::move(egraph);
        this->enodeInfos = std::move(enodeInfos);
        this->planner_egraph = std::move(planner.baseState.egraph);
        this->nodeToEClass = std::move(nodeToEClass);
        this->selection_map.clear();

        for (const auto &cls : this->egraph.getClasses())
        {
            EClassId canon = this->egraph.findConst(cls.id);
            if (!this->egraph.getEClass(canon).enodes.empty())
                selection_map[canon] = 0;
        }

        return this->egraph.findConst(this->nodeToEClass.at(rootId));
    }

    EGraph egraph;
    std::vector<ENodeInfo> enodeInfos;
    EGraph planner_egraph;
    std::unordered_map<LogicalId, EClassId> nodeToEClass;
};

// =============================================================================
// Scalable Graph Builders
// =============================================================================

inline LogicalId buildWideShallow(Graph &g, int scale = 2)
{
    scale = std::max(1, scale);
    int num_inputs = 2 + 2 * scale;
    std::vector<LogicalId> in_nodes;
    for (int i = 0; i < num_inputs; ++i)
    {
        in_nodes.push_back(g.input({8, 8}, DType::FLOAT32));
    }

    std::vector<LogicalId> layer1;
    for (int i = 0; i < num_inputs; ++i)
    {
        LogicalId next_in = in_nodes[(i + 1) % num_inputs];
        if (i % 2 == 0)
            layer1.push_back(g.add(in_nodes[i], next_in));
        else
            layer1.push_back(g.mul(in_nodes[i], next_in));
    }

    std::vector<LogicalId> curr_layer = layer1;
    while (curr_layer.size() > 1)
    {
        std::vector<LogicalId> next_layer;
        for (size_t i = 0; i < curr_layer.size(); i += 2)
        {
            if (i + 1 < curr_layer.size())
            {
                if (next_layer.size() % 2 == 0)
                    next_layer.push_back(g.add(curr_layer[i], curr_layer[i + 1]));
                else
                    next_layer.push_back(g.mul(curr_layer[i], curr_layer[i + 1]));
            }
            else
            {
                next_layer.push_back(curr_layer[i]);
            }
        }
        curr_layer = next_layer;
    }

    return curr_layer[0];
}

inline LogicalId buildLinearChain(Graph &g, int scale = 1)
{
    scale = std::max(1, scale);
    int length = 2 + 2 * scale;
    LogicalId in0 = g.input({8, 8}, DType::FLOAT32);
    LogicalId in1 = g.input({8, 8}, DType::FLOAT32);
    LogicalId b1 = g.add(in0, in1);
    LogicalId b2 = g.mul(in0, in1);
    LogicalId curr = g.add(b1, b2);

    for (int i = 3; i < length; ++i)
    {
        if (i % 2 == 0)
            curr = g.add(curr, in0);
        else
            curr = g.mul(curr, in1);
    }
    return g.neg(curr);
}

inline std::vector<ParallelBuffer> buildMallocBuffers(int scale = 1)
{
    scale = std::max(1, scale);
    int N = 4 + 4 * scale;
    std::vector<ParallelBuffer> unallocated;
    for (int i = 0; i < N; ++i)
    {
        ParallelBuffer b;
        b.id = BufferId{static_cast<uint32_t>(i)};
        b.mem_space = MemSpace{1, HandleType::CPP};
        b.size = ((i % 4) + 1) * 1024ULL * 1024;
        b.start = static_cast<uint32_t>(i / 2);
        b.end = b.start + 2;
        b.offset = -1;
        unallocated.push_back(b);
    }
    return unallocated;
}

inline LogicalId buildCacheGraph(Graph &g, int scale = 1)
{
    scale = std::max(1, scale);
    int num_candidates = 2 + 2 * scale;
    LogicalId in0 = g.input({8, 8}, DType::FLOAT32);
    LogicalId in1 = g.input({8, 8}, DType::FLOAT32);

    LogicalId curr = g.add(in0, in1);
    for (int i = 1; i < num_candidates; ++i)
    {
        if (i % 3 == 0)
        {
            LogicalId tiny_in = g.input({2, 2}, DType::FLOAT32);
            LogicalId tiny_node = g.add(tiny_in, tiny_in);
            LogicalId expanded = g.reshape(tiny_node, {2, 1, 2, 1});
            expanded = g.repeat(expanded, 4, 1);
            expanded = g.repeat(expanded, 4, 3);
            expanded = g.contiguous(expanded);
            expanded = g.reshape(expanded, {8, 8});
            curr = g.add(curr, expanded);
        }
        else if (i % 2 == 0)
        {
            LogicalId fork1 = g.mul(curr, in0);
            LogicalId fork2 = g.add(curr, in1);
            curr = g.add(fork1, fork2);
        }
        else
        {
            curr = g.mul(curr, in1);
        }
    }
    return curr;
}

inline LogicalId buildDiamond(Graph &g, int scale = 1)
{
    scale = std::max(1, scale);
    LogicalId in0 = g.input({8, 8}, DType::FLOAT32);
    LogicalId in1 = g.input({8, 8}, DType::FLOAT32);
    LogicalId curr = g.add(in0, in1);

    for (int s = 0; s < scale; ++s)
    {
        LogicalId branchA = g.add(curr, in0);
        LogicalId branchB = g.add(curr, in0);
        curr = g.add(branchA, branchB);
    }
    return curr;
}

struct FmaTwins
{
    LogicalId root;
    std::vector<LogicalId> twin_roots;
    std::vector<LogicalId> inputs;
};

inline FmaTwins buildFmaTwins(Graph &g, int scale = 1)
{
    scale = std::max(1, scale);
    int num_twins = 3 * scale;
    std::vector<LogicalId> inputs;
    for (int i = 0; i < num_twins * 2; ++i)
    {
        inputs.push_back(g.input({8, 8}, DType::FLOAT32));
    }

    std::vector<LogicalId> twin_roots;
    for (int i = 0; i < num_twins; ++i)
    {
        LogicalId r = g.add(inputs[2 * i], inputs[2 * i + 1]);
        twin_roots.push_back(r);
    }

    LogicalId root = twin_roots[0];
    for (size_t i = 1; i < twin_roots.size(); ++i)
    {
        root = g.add(root, twin_roots[i]);
    }

    return {root, twin_roots, inputs};
}

inline void extendFmaTwinsEGraph(const FmaTwins &twins, EGraph &egraph_to_extend,
                                 const std::unordered_map<LogicalId, EClassId> &nodeToEClass)
{
    int num_twins = static_cast<int>(twins.twin_roots.size());
    for (int i = 0; i < num_twins; ++i)
    {
        LogicalId r = twins.twin_roots[i];
        EClassId r_cls = egraph_to_extend.findConst(nodeToEClass.at(r));
        EClassId in0_cls = egraph_to_extend.findConst(nodeToEClass.at(twins.inputs[2 * i]));
        EClassId in1_cls = egraph_to_extend.findConst(nodeToEClass.at(twins.inputs[2 * i + 1]));

        ENode v1(KernelId{MockKernels::kFmaV1}, OpType::FUSED, "fma_v1", {in0_cls, in1_cls}, {8, 8}, {8, 1},
                 DType::FLOAT32, MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}});
        ENode v2(KernelId{MockKernels::kFmaV2}, OpType::FUSED, "fma_v2", {in0_cls, in1_cls}, {8, 8}, {8, 1},
                 DType::FLOAT32, MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}});
        ENode v3(KernelId{MockKernels::kAddBigOutplace}, OpType::ADD, "", {in0_cls, in1_cls}, {1024, 1024}, {1024, 1},
                 DType::FLOAT32, MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}});
        egraph_to_extend.addENode(r_cls, v1);
        egraph_to_extend.addENode(r_cls, v2);
        egraph_to_extend.addENode(r_cls, v3);
    }
}

} // namespace prune_test