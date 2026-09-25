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

    if (compiled.cost() <= 0.0f)
    {
        Error::throw_err("[Regression Test Failed] compiled.cost() " + std::to_string(compiled.cost()) +
                         " <= 0.0f");
    }

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







inline void testCompiledGraphCost()
{
    std::cout << "  - running testCompiledGraphCost..." << std::endl;

    // Test 1: Empty instructions fallback to nodeCosts sum
    {
        CompiledGraph g;
        g.nodeCosts[EClassId{1}] = 3.5f;
        g.nodeCosts[EClassId{2}] = 4.5f;
        if (std::abs(g.cost() - 8.0f) > 1e-5f)
        {
            Error::throw_err("[testCompiledGraphCost Failed] Empty instructions should sum nodeCosts!");
        }
    }

    // Test 2: Sequential execution on single engine
    {
        CompiledGraph g;
        Engine cpu{0, EngineType::CPU};

        OpInstruction inst_1;
        inst_1.eclass_id = EClassId{1};
        inst_1.engines = {cpu};
        inst_1.outBuffer.id = BufferId{10};

        OpInstruction inst_2;
        inst_2.eclass_id = EClassId{2};
        inst_2.engines = {cpu};
        inst_2.children = {EClassId{1}};
        inst_2.inBuffers.push_back(inst_1.outBuffer);
        inst_2.outBuffer.id = BufferId{20};

        g.instructions = {inst_1, inst_2};
        g.nodeCosts[EClassId{1}] = 10.0f;
        g.nodeCosts[EClassId{2}] = 5.0f;

        // inst_1 finishes at 10.0, inst_2 starts at 10.0 and finishes at 15.0
        if (std::abs(g.cost() - 15.0f) > 1e-5f)
        {
            Error::throw_err("[testCompiledGraphCost Failed] Sequential single engine makespan mismatch!");
        }
    }

    // Test 3: Parallel execution across two engines
    {
        CompiledGraph g;
        Engine cpu{0, EngineType::CPU};
        Engine gpu{0, EngineType::CUDA_GPU};

        // inst_1 on CPU takes 10.0ms
        OpInstruction inst_1;
        inst_1.eclass_id = EClassId{1};
        inst_1.engines = {cpu};
        inst_1.outBuffer.id = BufferId{10};

        // inst_2 on GPU takes 6.0ms in parallel
        OpInstruction inst_2;
        inst_2.eclass_id = EClassId{2};
        inst_2.engines = {gpu};
        inst_2.outBuffer.id = BufferId{20};

        // inst_3 on CPU takes 4.0ms, depends on both inst_1 and inst_2
        OpInstruction inst_3;
        inst_3.eclass_id = EClassId{3};
        inst_3.engines = {cpu};
        inst_3.children = {EClassId{1}, EClassId{2}};
        inst_3.inBuffers = {inst_1.outBuffer, inst_2.outBuffer};
        inst_3.outBuffer.id = BufferId{30};

        g.instructions = {inst_1, inst_2, inst_3};
        g.nodeCosts[EClassId{1}] = 10.0f;
        g.nodeCosts[EClassId{2}] = 6.0f;
        g.nodeCosts[EClassId{3}] = 4.0f;

        // inst_1: birth=0, finish=10.0 on CPU
        // inst_2: birth=0, finish=6.0 on GPU
        // inst_3: children_finish=max(10.0, 6.0)=10.0, engine_free(CPU)=10.0 => birth=10.0, finish=14.0 on CPU
        // Total makespan: max(14.0, 6.0) = 14.0ms
        if (std::abs(g.cost() - 14.0f) > 1e-5f)
        {
            Error::throw_err("[testCompiledGraphCost Failed] Parallel engine makespan mismatch!");
        }
    }

    // Test 4: View dependency through buffer_writers
    {
        CompiledGraph g;
        Engine gpu{0, EngineType::CUDA_GPU};
        Engine cpu{0, EngineType::CPU};

        // inst_1 on GPU produces Buffer 100
        OpInstruction inst_1;
        inst_1.eclass_id = EClassId{1};
        inst_1.engines = {gpu};
        inst_1.outBuffer.id = BufferId{100};

        // EClass 2 is a VIEW of EClass 1 (not in instructions), sharing Buffer 100
        // inst_3 on CPU consumes EClass 2, with inBuffers pointing to Buffer 100
        OpInstruction inst_3;
        inst_3.eclass_id = EClassId{3};
        inst_3.engines = {cpu};
        inst_3.children = {EClassId{2}};
        ParallelBuffer view_in_buf;
        view_in_buf.id = BufferId{100};
        inst_3.inBuffers = {view_in_buf};
        inst_3.outBuffer.id = BufferId{101};

        g.instructions = {inst_1, inst_3};
        g.nodeCosts[EClassId{1}] = 12.0f;
        g.nodeCosts[EClassId{3}] = 8.0f;

        // inst_1: birth=0, finish=12.0 on GPU
        // inst_3: child 2 not in eclass_engines, but inBuffers[0] has Buffer 100 written by GPU
        //        children_finish = 12.0, engine_free(CPU) = 0 => birth=12.0, finish=20.0 on CPU
        // Total makespan: max(12.0, 20.0) = 20.0ms
        if (std::abs(g.cost() - 20.0f) > 1e-5f)
        {
            Error::throw_err("[testCompiledGraphCost Failed] View buffer dependency makespan mismatch!");
        }
    }
}

inline void runViewBufferizeRegressionTests()
{
    std::cout << "view & bufferize regression tests" << std::endl << std::flush;
    testCompiledGraphCost();
    testViewNotEmittedIntoInstructions();
}
