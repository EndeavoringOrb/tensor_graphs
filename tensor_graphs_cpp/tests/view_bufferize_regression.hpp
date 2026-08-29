#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/executor.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"
#include "core/plan/planner.hpp"
#include "core/plan/validators/mem.hpp"
#include "core/types.hpp"
#include "tests/common.hpp"

// =============================================================================
// Regression Test 1: is_view Check in buildCompiledGraph & Storage Output Safety
// =============================================================================
inline void testViewNotEmittedIntoInstructions()
{
    std::cout << "  - running testViewNotEmittedIntoInstructions..." << std::endl;

    CostModel costModel(false, "");
    std::unordered_map<MemSpace, uint64_t> mem_caps = {
        {MemSpace{1, HandleType::CPP}, 1024ULL * 1024 * 1024},
    };
    Settings settings = Settings::get_default();
    settings.mem_caps = mem_caps;

    Graph graph;
    // 1. Create a storage-backed weight node
    LogicalId w = graph.input({8, 16}, DType::FLOAT32);
    graph.input_data_types[w] = InputDataType::STORAGE;

    // 2. View operations on the storage weight: PERMUTE and RESHAPE
    LogicalId permDims = graph.constant({1, 0});
    LogicalId w_t = graph.permute(w, permDims);      // [16, 8] view in STORAGE
    LogicalId w_3d = graph.reshape(w_t, {1, 16, 8}); // [1, 16, 8] view in STORAGE

    // 3. Runtime input and compute kernel consuming the view
    LogicalId x = graph.input({1, 4, 16}, DType::FLOAT32);
    graph.input_data_types[x] = InputDataType::RUNTIME;
    LogicalId out = graph.dot(x, w_3d); // [1, 4, 8] in CPP

    std::vector<LogicalId> topo = topologicalSort({out}, graph);
    Planner planner(costModel, settings);
    planner.initBaseEGraph(out, graph, topo, nullptr);
    populateDummyRecords(costModel, planner.baseState.egraph);

    Bucket bucket;
    bucket.inputDirtyRegions[x] = {makeFull(graph.getNode(x).getShape())};
    bucket.outputNeededRegion = {makeFull(graph.getNode(out).getShape())};

    CompiledGraph compiled = planner.plan(out, graph, bucket, {}, true, false, nullptr);

    // Verify 1: View operations must NOT be emitted into compiled.instructions
    for (const auto &inst : compiled.instructions)
    {
        if (inst.kernel_id.value != 0 && KernelRegistry::get().hasKernel(inst.kernel_id))
        {
            const auto &kernel = KernelRegistry::get().getKernel(inst.kernel_id);
            if (kernel.is_view)
            {
                Error::throw_err("[Regression Test Failed] View kernel " + kernel.opName +
                                 " was emitted into compiled.instructions!");
            }
        }

        // Verify 2: No executable instruction should ever have a STORAGE output buffer
        if (inst.outBuffer.mem_space.type == HandleType::STORAGE)
        {
            Error::throw_err("[Regression Test Failed] Instruction has HandleType::STORAGE for outBuffer!");
        }
    }

    // Verify 3: nodeViews must still contain precalculated metadata for all views
    EClassId w_3d_eclass = planner.baseState.egraph.findConst(planner.baseState.nodeToEClass.at(w_3d));
    if (compiled.nodeViews.find(w_3d_eclass) == compiled.nodeViews.end())
    {
        Error::throw_err("[Regression Test Failed] View node missing from compiled.nodeViews!");
    }
    const TensorView &view3d = compiled.nodeViews.at(w_3d_eclass);
    std::vector<uint32_t> expectedShape = {1, 16, 8};
    if (view3d.getShape() != expectedShape)
    {
        Error::throw_err("[Regression Test Failed] View shape mismatch in nodeViews!");
    }
}

// =============================================================================
// Regression Test 2: inplace_alias Cleanup on choice == -1
// =============================================================================
inline void testInplaceAliasEraseOnNewBuffer()
{
    std::cout << "  - running testInplaceAliasEraseOnNewBuffer..." << std::endl;

    CostModel costModel(false, "");
    std::unordered_map<MemSpace, uint64_t> mem_caps = {
        {MemSpace{1, HandleType::CPP}, 1024ULL * 1024 * 1024},
    };
    Settings settings = Settings::get_default();
    settings.mem_caps = mem_caps;

    // Register safe in-place kernel for testing
    static bool registered = false;
    if (!registered)
    {
        KernelRegistry::get().registerKernel(
            KernelId{0xAA0001}, OpType::ADD, "", 2, 2, nullptr, nullptr, nullptr, {0, 1}, false, true, nullptr,
            MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
            {{8, 8}, {8, 8}}, {false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
        registered = true;
    }

    Record r;
    r.kernelId = KernelId{0xAA0001};
    r.buildContextId = BUILD_CONTEXT_ID;
    r.hwTag = HW_TAG;
    r.outputShape = {8, 8};
    r.outputStrides = {8, 1};
    r.outputDType = DType::FLOAT32;
    r.inputShapes = {{8, 8}, {8, 8}};
    r.inputStrides = {{8, 1}, {8, 1}};
    r.inputDTypes = {DType::FLOAT32, DType::FLOAT32};
    r.runTime = 1.0f;
    costModel.records[r.kernelId].push_back(r);

    Graph graph;
    LogicalId in0 = graph.input({8, 8}, DType::FLOAT32);
    LogicalId in1 = graph.input({8, 8}, DType::FLOAT32);
    LogicalId t0 = graph.add(in0, in1);
    LogicalId t1 = graph.add(t0, in1);
    LogicalId root = graph.add(t1, t0);

    std::vector<LogicalId> topo = topologicalSort({root}, graph);
    Planner planner(costModel, settings);
    planner.initBaseEGraph(root, graph, topo, nullptr);
    populateDummyRecords(costModel, planner.baseState.egraph);

    EGraph egraph = planner.baseState.egraph;
    auto enodeInfos = planner.computeENodeInfos(egraph, planner.baseState.eclassToLogical, {}, false);

    std::unordered_map<EClassId, uint32_t> selection_map;
    for (const auto &cls : egraph.getClasses())
    {
        EClassId canon = egraph.findConst(cls.id);
        if (!egraph.getEClass(canon).enodes.empty())
            selection_map[canon] = 0;
    }

    auto dispatch_iter = makeDispatchIterator(egraph, selection_map, enodeInfos);
    std::vector<EClassId> order;
    if (!dispatch_iter.getNextDispatchOrder(selection_map, order))
    {
        Error::throw_err("[Regression Test Failed] Could not generate dispatch order.");
    }

    auto buf_iter = makeBufferizeIterator(order, egraph, selection_map, enodeInfos, settings.mem_caps);

    std::vector<ParallelBuffer> bufs;
    std::unordered_map<EClassId, BufferId> eclass_to_buf;

    EClassId t1_eclass = egraph.findConst(planner.baseState.nodeToEClass.at(t1));
    EClassId t0_eclass = egraph.findConst(planner.baseState.nodeToEClass.at(t0));

    bool found_inplace = false;
    bool found_separate = false;

    while (buf_iter.getNextBufferization(bufs, eclass_to_buf))
    {
        if (eclass_to_buf.count(t1_eclass) && eclass_to_buf.count(t0_eclass))
        {
            if (eclass_to_buf[t1_eclass] == eclass_to_buf[t0_eclass])
            {
                found_inplace = true;
            }
            else
            {
                found_separate = true;
                if (buf_iter.inplace_alias.count(t1_eclass))
                {
                    Error::throw_err("[Regression Test Failed] Stale inplace_alias mapping found for choice == -1!");
                }
            }
        }
    }

    if (!found_inplace || !found_separate)
    {
        Error::throw_err(
            "[Regression Test Failed] BufferizeIterator did not explore both in-place and separate buffer options.");
    }
}

// =============================================================================
// Regression Test 3: build_buffers Full Coverage and Alias Fallback Safety
// =============================================================================
inline void testBuildBuffersCoverageAndFallback()
{
    std::cout << "  - running testBuildBuffersCoverageAndFallback..." << std::endl;

    CostModel costModel(false, "");
    std::unordered_map<MemSpace, uint64_t> mem_caps = {
        {MemSpace{1, HandleType::CPP}, 1024ULL * 1024 * 1024},
    };
    Settings settings = Settings::get_default();
    settings.mem_caps = mem_caps;

    Graph graph;
    // Nested view chain
    LogicalId in0 = graph.input({4, 8}, DType::FLOAT32);
    LogicalId v1 = graph.reshape(in0, {2, 16});
    LogicalId v2 = graph.slice(v1, {0, 0}, {2, 8}, {1, 1});
    LogicalId c1 = graph.add(v2, v2);
    LogicalId v3 = graph.permute(c1, {1, 0});
    LogicalId root = graph.neg(v3);

    std::vector<LogicalId> topo = topologicalSort({root}, graph);
    Planner planner(costModel, settings);
    planner.initBaseEGraph(root, graph, topo, nullptr);
    populateDummyRecords(costModel, planner.baseState.egraph);

    EGraph egraph = planner.baseState.egraph;
    auto enodeInfos = planner.computeENodeInfos(egraph, planner.baseState.eclassToLogical, {}, false);

    std::unordered_map<EClassId, uint32_t> selection_map;
    for (const auto &cls : egraph.getClasses())
    {
        EClassId canon = egraph.findConst(cls.id);
        if (!egraph.getEClass(canon).enodes.empty())
            selection_map[canon] = 0;
    }

    auto dispatch_iter = makeDispatchIterator(egraph, selection_map, enodeInfos);
    std::vector<EClassId> order;
    if (!dispatch_iter.getNextDispatchOrder(selection_map, order))
    {
        Error::throw_err("[Regression Test Failed] Failed to get dispatch order.");
    }

    auto buf_iter = makeBufferizeIterator(order, egraph, selection_map, enodeInfos, settings.mem_caps);
    std::vector<ParallelBuffer> out_buffers;
    std::unordered_map<EClassId, BufferId> out_eclass_to_buf;

    if (!buf_iter.getNextBufferization(out_buffers, out_eclass_to_buf))
    {
        Error::throw_err("[Regression Test Failed] getNextBufferization returned false.");
    }

    // 1. Verify that every node in `order` has an entry in `out_eclass_to_buf`
    for (EClassId eclass : order)
    {
        if (out_eclass_to_buf.find(eclass) == out_eclass_to_buf.end())
        {
            Error::throw_err("[Regression Test Failed] Missing EClass " + std::to_string(eclass.value) +
                             " from out_eclass_to_buf!");
        }
    }

    // 2. Verify all mapped buffer IDs exist in out_buffers
    std::unordered_set<BufferId> allocated_ids;
    for (const auto &buf : out_buffers)
    {
        allocated_ids.insert(buf.id);
    }
    for (const auto &kv : out_eclass_to_buf)
    {
        if (allocated_ids.find(kv.second) == allocated_ids.end())
        {
            Error::throw_err("[Regression Test Failed] out_eclass_to_buf points to unallocated BufferId " +
                             std::to_string(kv.second.value) + "!");
        }
    }

    // 3. Verify that views share the buffer with their underlying base
    EClassId in0_eclass = egraph.findConst(planner.baseState.nodeToEClass.at(in0));
    EClassId v1_eclass = egraph.findConst(planner.baseState.nodeToEClass.at(v1));
    EClassId v2_eclass = egraph.findConst(planner.baseState.nodeToEClass.at(v2));

    if (out_eclass_to_buf[v1_eclass] != out_eclass_to_buf[in0_eclass] ||
        out_eclass_to_buf[v2_eclass] != out_eclass_to_buf[in0_eclass])
    {
        Error::throw_err("[Regression Test Failed] View chain did not alias to the original input buffer!");
    }
}

// =============================================================================
// Regression Test 4: PeakMemoryPruningRule Under Tight Memory Limits
// =============================================================================
inline void testPeakMemoryPruningUnderTightCap()
{
    std::cout << "  - running testPeakMemoryPruningUnderTightCap..." << std::endl;

    CostModel costModel(false, "");
    Settings settings = Settings::get_default();

    // 16 KB cap: exactly fits in-place chain (4 buffers = 16384 bytes)
    uint64_t tight_cap = 16384ULL;
    settings.mem_caps = {{MemSpace{1, HandleType::CPP}, tight_cap}};

    Graph graph;
    LogicalId in0 = graph.input({8, 8}, DType::FLOAT32);
    LogicalId in1 = graph.input({8, 8}, DType::FLOAT32);
    LogicalId b1 = graph.add(in0, in1);
    LogicalId b2 = graph.mul(in0, in1);
    LogicalId curr = graph.add(b1, b2);
    LogicalId out = graph.neg(curr);

    std::vector<LogicalId> topo = topologicalSort({out}, graph);
    Planner planner(costModel, settings);
    planner.initBaseEGraph(out, graph, topo, nullptr);
    populateDummyRecords(costModel, planner.baseState.egraph);

    EGraph egraph = planner.baseState.egraph;
    auto enodeInfos = planner.computeENodeInfos(egraph, planner.baseState.eclassToLogical, {}, false);

    std::unordered_map<EClassId, uint32_t> selection_map;
    for (const auto &cls : egraph.getClasses())
    {
        EClassId canon = egraph.findConst(cls.id);
        if (!egraph.getEClass(canon).enodes.empty())
            selection_map[canon] = 0;
    }

    auto dispatch_iter = makeDispatchIterator(egraph, selection_map, enodeInfos);
    std::vector<EClassId> order;
    if (!dispatch_iter.getNextDispatchOrder(selection_map, order))
    {
        Error::throw_err("[Regression Test Failed] Could not generate dispatch order.");
    }

    // 1. Verify PeakMemoryPruningRule prunes all out-of-place choices under 16KB cap
    auto buf_iter = makeBufferizeIterator(order, egraph, selection_map, enodeInfos, settings.mem_caps, nullptr, nullptr,
                                          PeakMemoryPruningRule(true));
    std::vector<ParallelBuffer> bufs;
    std::unordered_map<EClassId, BufferId> eclass_to_buf;

    uint32_t yield_count = 0;
    while (buf_iter.getNextBufferization(bufs, eclass_to_buf))
    {
        yield_count++;
        BufferId overflow;
        if (!check_peak_memory(bufs, tight_cap, overflow))
        {
            Error::throw_err(
                "[Regression Test Failed] PeakMemoryPruningRule yielded a bufferization exceeding mem_cap!");
        }
    }

    if (yield_count == 0)
    {
        Error::throw_err("[Regression Test Failed] PeakMemoryPruningRule pruned all valid in-place bufferizations!");
    }

    // 2. Verify clean rejection when mem_cap is lower than minimum feasible peak (8 KB < 16 KB)
    std::unordered_map<MemSpace, uint64_t> impossible_caps = {{MemSpace{1, HandleType::CPP}, 8192ULL}};
    auto impossible_iter = makeBufferizeIterator(order, egraph, selection_map, enodeInfos, impossible_caps, nullptr,
                                                 nullptr, PeakMemoryPruningRule(true));
    if (impossible_iter.getNextBufferization(bufs, eclass_to_buf))
    {
        Error::throw_err(
            "[Regression Test Failed] BufferizeIterator yielded a bufferization under an impossible 8KB cap!");
    }
}

// =============================================================================
// Regression Test 5: Malloc Under Tight Memory Limits
// =============================================================================
inline void testMallocUnderTightCap()
{
    std::cout << "  - running testMallocUnderTightCap..." << std::endl;

    // 8 overlapping buffers with 17 MB peak demand
    std::vector<ParallelBuffer> unallocated = prune_test::buildMallocBuffers(1);

    // 1. Tight feasible cap (18 MB >= 17 MB peak)
    uint64_t tight_cap = 18ULL * 1024 * 1024;
    Settings settings = Settings::get_default();
    auto iter = makeConfiguredMallocIterator(tight_cap, unallocated, nullptr, settings);

    std::vector<ParallelBuffer> allocated;
    if (!iter.getNextAllocation(allocated))
    {
        Error::throw_err("[Regression Test Failed] Malloc failed to find valid allocation under 18 MB tight cap!");
    }

    int64_t max_peak = 0;
    for (const auto &b : allocated)
    {
        max_peak = std::max<int64_t>(max_peak, b.offset + b.size);
    }
    if (max_peak > static_cast<int64_t>(tight_cap))
    {
        Error::throw_err("[Regression Test Failed] Allocated buffers exceeded 18 MB tight cap!");
    }

    // 2. Infeasible cap (12 MB < 17 MB peak)
    uint64_t impossible_cap = 12ULL * 1024 * 1024;
    auto impossible_iter = makeConfiguredMallocIterator(impossible_cap, unallocated, nullptr, settings);
    std::vector<ParallelBuffer> impossible_alloc;
    if (impossible_iter.getNextAllocation(impossible_alloc))
    {
        Error::throw_err("[Regression Test Failed] Malloc yielded an allocation under impossible 12 MB cap!");
    }
}

// =============================================================================
// Regression Test 6: MemCapENodeDomination Under Tight Memory Limits
// =============================================================================
inline void testMemCapENodeDomination()
{
    std::cout << "  - running testMemCapENodeDomination..." << std::endl;

    uint64_t tight_mem_cap = 2ULL * 1024 * 1024; // 2 MB
    prune_test::MockCtx mock(tight_mem_cap);

    Graph g;
    auto twins = prune_test::buildFmaTwins(g, 1);
    mock.build(g, twins.root, false, [&](EGraph &egraph, const std::unordered_map<LogicalId, EClassId> &n2e) {
        prune_test::extendFmaTwinsEGraph(twins, egraph, n2e);
    });

    std::unordered_map<EClassId, LogicalId> emptyMap;
    std::unordered_map<LogicalId, MemSpace> emptyCached;
    ENodeDominationContext ctx{mock.egraph, mock.enodeInfos, emptyMap, emptyCached, mock.settings.mem_caps};

    MemCapENodeDominationRule rule(true);

    bool found_4mb_pruned = false;
    bool found_small_kept = false;

    for (uint32_t i = 0; i < mock.egraph.getENodes().size(); ++i)
    {
        ENodeId enodeId{i};
        const ENode &enode = mock.egraph.getENode(enodeId);
        bool should_prune = rule.check(enodeId, 0, ctx);

        if (enode.getKernelId() == KernelId{prune_test::MockKernels::kAddBigOutplace})
        {
            if (!should_prune)
            {
                Error::throw_err("[Regression Test Failed] 4MB AddBigOutplace kernel was not pruned under 2MB cap!");
            }
            found_4mb_pruned = true;
        }
        else if (enode.getKernelId() == KernelId{prune_test::MockKernels::kFmaV2})
        {
            if (should_prune)
            {
                Error::throw_err(
                    "[Regression Test Failed] 256-byte FmaV2 kernel was incorrectly pruned under 2MB cap!");
            }
            found_small_kept = true;
        }
    }

    if (!found_4mb_pruned || !found_small_kept)
    {
        Error::throw_err("[Regression Test Failed] Did not evaluate both large and small kernel candidates!");
    }
}

inline void runViewBufferizeRegressionTests()
{
    std::cout << "view & bufferize regression tests" << std::endl << std::flush;
    testViewNotEmittedIntoInstructions();
    testInplaceAliasEraseOnNewBuffer();
    testBuildBuffersCoverageAndFallback();
    testPeakMemoryPruningUnderTightCap();
    testMallocUnderTightCap();
    testMemCapENodeDomination();
}