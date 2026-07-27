// File: tensor_graphs_cpp/test.cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "core/argparse.hpp"
#include "core/common/bench_utils.hpp"
#include "core/cost_model.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/loaders/safetensors.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"
#include "core/plan/planner.hpp"
#include "core/session.hpp"
#include "core/shapes.hpp"
#include "generated/kernels_all.gen.hpp"

void fillRandom(void *ptr, uint64_t elements, DType dtype)
{
    static std::mt19937 gen(42);
    switch (dtype)
    {
    case DType::ANY:
    case DType::FLOAT32: {
        float *fptr = static_cast<float *>(ptr);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (uint64_t i = 0; i < elements; ++i)
            fptr[i] = dist(gen);
        break;
    }
    case DType::INT32: {
        int32_t *iptr = static_cast<int32_t *>(ptr);
        std::uniform_int_distribution<int32_t> dist(1, 10);
        for (uint64_t i = 0; i < elements; ++i)
            iptr[i] = dist(gen);
        break;
    }
    case DType::BOOL: {
        bool *bptr = static_cast<bool *>(ptr);
        std::uniform_int_distribution<int> dist(0, 1);
        for (uint64_t i = 0; i < elements; ++i)
            bptr[i] = dist(gen) != 0;
        break;
    }
    case DType::BF16: {
        uint16_t *bfptr = static_cast<uint16_t *>(ptr);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (uint64_t i = 0; i < elements; ++i)
        {
            float val = dist(gen);
            uint32_t f32_bits;
            std::memcpy(&f32_bits, &val, 4);
            bfptr[i] = static_cast<uint16_t>(f32_bits >> 16);
        }
        break;
    }
    default:
        Error::throw_err("[fillRandom] Unsupported DType " + toString(dtype));
    }
}

bool compareOutputs(const float *ref, const float *test, uint64_t elements, float eps = 1e-4f)
{
    for (uint64_t i = 0; i < elements; ++i)
    {
        if (std::abs(ref[i] - test[i]) > eps)
        {
            std::cout << "\nMismatch at index " << i << ": (ref)" << ref[i] << " != (test)" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool compareOutputs(const int32_t *ref, const int32_t *test, uint64_t elements, float eps = 1e-4f)
{
    for (uint64_t i = 0; i < elements; ++i)
    {
        if (ref[i] != test[i])
        {
            std::cout << "\nMismatch at index " << i << ": (ref)" << ref[i] << " != (test)" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool compareOutputs(const bool *ref, const bool *test, uint64_t elements, float eps = 1e-4f)
{
    for (uint64_t i = 0; i < elements; ++i)
    {
        if (ref[i] != test[i])
        {
            std::cout << "\nMismatch at index " << i << ": (ref)" << ref[i] << " != (test)" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

Region makeRegion(std::initializer_list<Dim> dims)
{
    Region r;
    r.region.assign(dims.begin(), dims.end());
    return r;
}

bool regionListEquals(const std::vector<Region> &actual, const std::vector<Region> &expected)
{
    const auto a = normalizeRegions(actual);
    const auto e = normalizeRegions(expected);
    if (a.size() != e.size())
        return false;
    for (uint64_t i = 0; i < a.size(); ++i)
    {
        if (!regionsMatch(a[i], e[i]))
            return false;
    }
    return true;
}

void assertRegionListEquals(const std::vector<Region> &actual, const std::vector<Region> &expected,
                            const std::string &label)
{
    if (!regionListEquals(actual, expected))
    {
        std::stringstream ss;
        ss << "[RegionTest] " << label << " expected " << encodeRegionList(expected) << " but got "
           << encodeRegionList(actual);
        Error::throw_err(ss.str());
    }
}

void runRegionMergeTests()
{
    std::cout << "region merge tests" << std::endl << std::flush;
    {
        std::vector<Region> actual = mergeRegions({makeRegion({{0, 2}}), makeRegion({{2, 4}})});
        assertRegionListEquals(actual, {makeRegion({{0, 4}})}, "1D adjacent merge");
    }
    {
        std::vector<Region> actual = mergeRegions({
            makeRegion({{0, 4}, {0, 2}}),
            makeRegion({{0, 2}, {2, 4}}),
            makeRegion({{2, 4}, {2, 4}}),
        });
        assertRegionListEquals(actual,
                               {
                                   makeRegion({{0, 4}, {0, 2}}),
                                   makeRegion({{0, 4}, {2, 4}}),
                               },
                               "two-step 2D merge");
    }
    {
        std::vector<Region> actual = mergeRegions({
            makeRegion({{0, 4}, {0, 2}}),
            makeRegion({{0, 4}, {2, 4}}),
        });
        assertRegionListEquals(actual, {makeRegion({{0, 4}, {0, 4}})}, "full 2D merge");
    }
    {
        std::vector<Region> forwardA = mergeRegions({
            makeRegion({{2, 4}, {0, 1}}),
            makeRegion({{0, 2}, {0, 1}}),
        });
        std::vector<Region> forwardB = mergeRegions({
            makeRegion({{0, 2}, {0, 1}}),
            makeRegion({{2, 4}, {0, 1}}),
        });
        if (encodeRegionList(forwardA) != encodeRegionList(forwardB))
        {
            Error::throw_err("[RegionTest] merge ordering is not deterministic");
        }
    }
}

void runShapePropagationTests()
{
    std::cout << "shape propagation tests" << std::endl << std::flush;
    ShapePropagator prop;
    auto makeIntConst = [](Graph &graph, const std::vector<int32_t> &values) -> LogicalId {
        return graph.constant({(uint32_t)values.size()}, values.data(), DType::INT32);
    };
    auto makeFloatInput = [](Graph &graph, const std::vector<uint32_t> &shape) -> LogicalId {
        return graph.input(shape, DType::FLOAT32, {});
    };
    {
        Graph graph;
        LogicalId x = makeFloatInput(graph, {4, 5});
        LogicalId axis = makeIntConst(graph, {1});
        LogicalId sumId = graph.sum(x, axis);
        prop.inferShapeRecursive(sumId, graph);
        auto forward = prop.forward(graph.getNode(sumId), graph, {{makeRegion({{1, 3}, {2, 4}})}, {}});
        assertRegionListEquals(forward, {makeRegion({{1, 3}, {0, 1}})}, "SUM forward");
        auto backward = prop.backward(graph.getNode(sumId), graph, {makeRegion({{1, 3}, {0, 1}})});
        assertRegionListEquals(backward[0], {makeRegion({{1, 3}, {0, 5}})}, "SUM backward input");
        assertRegionListEquals(backward[1], makeFull(graph.getNode(axis).getShape()), "SUM backward axis");
    }
    {
        Graph graph;
        LogicalId x = makeFloatInput(graph, {4, 5});
        LogicalId axis = makeIntConst(graph, {1});
        LogicalId maxId = graph.max(x, axis);
        prop.inferShapeRecursive(maxId, graph);
        auto forward = prop.forward(graph.getNode(maxId), graph, {{makeRegion({{0, 4}, {1, 3}})}, {}});
        assertRegionListEquals(forward, {makeRegion({{0, 4}, {0, 1}})}, "MAX forward");
        auto backward = prop.backward(graph.getNode(maxId), graph, {makeRegion({{0, 4}, {0, 1}})});
        assertRegionListEquals(backward[0], {makeRegion({{0, 4}, {0, 5}})}, "MAX backward input");
    }
    {
        Graph graph;
        LogicalId x = makeFloatInput(graph, {2, 3});
        LogicalId dims = makeIntConst(graph, {1, 0});
        LogicalId permId = graph.permute(x, dims);
        prop.inferShapeRecursive(permId, graph);
        auto forward = prop.forward(graph.getNode(permId), graph,
                                    {{makeRegion({{0, 1}, {1, 3}}), makeRegion({{1, 2}, {0, 2}})}, {}});
        assertRegionListEquals(forward,
                               {
                                   makeRegion({{1, 3}, {0, 1}}),
                                   makeRegion({{0, 2}, {1, 2}}),
                               },
                               "PERMUTE forward");
        auto backward =
            prop.backward(graph.getNode(permId), graph, {makeRegion({{1, 3}, {0, 1}}), makeRegion({{0, 2}, {1, 2}})});
        assertRegionListEquals(backward[0],
                               {
                                   makeRegion({{0, 1}, {1, 3}}),
                                   makeRegion({{1, 2}, {0, 2}}),
                               },
                               "PERMUTE backward input");
    }
    {
        Graph graph;
        LogicalId a = makeFloatInput(graph, {2, 2});
        LogicalId b = makeFloatInput(graph, {2, 2});
        LogicalId axis = makeIntConst(graph, {0});
        LogicalId concatId = graph.concat({a, b}, axis);
        prop.inferShapeRecursive(concatId, graph);
        auto forward = prop.forward(graph.getNode(concatId), graph,
                                    {{}, {makeRegion({{0, 1}, {0, 2}})}, {makeRegion({{1, 2}, {1, 2}})}});
        assertRegionListEquals(forward,
                               {
                                   makeRegion({{0, 1}, {0, 2}}),
                                   makeRegion({{3, 4}, {1, 2}}),
                               },
                               "CONCAT forward");
        auto backward =
            prop.backward(graph.getNode(concatId), graph, {makeRegion({{0, 1}, {0, 2}}), makeRegion({{3, 4}, {1, 2}})});
        assertRegionListEquals(backward[1], {makeRegion({{0, 1}, {0, 2}})}, "CONCAT backward left");
        assertRegionListEquals(backward[2], {makeRegion({{1, 2}, {1, 2}})}, "CONCAT backward right");
    }
    {
        Graph graph;
        LogicalId x = makeFloatInput(graph, {2, 2});
        LogicalId repeats = makeIntConst(graph, {3});
        LogicalId axis = makeIntConst(graph, {0});
        LogicalId repeatId = graph.repeat(x, repeats, axis);
        prop.inferShapeRecursive(repeatId, graph);
        auto forward = prop.forward(graph.getNode(repeatId), graph, {{makeRegion({{1, 2}, {0, 2}})}, {}, {}});
        assertRegionListEquals(forward, {makeRegion({{3, 6}, {0, 2}})}, "REPEAT forward");
        auto backward = prop.backward(graph.getNode(repeatId), graph, {makeRegion({{3, 6}, {0, 2}})});
        assertRegionListEquals(backward[0], {makeRegion({{1, 2}, {0, 2}})}, "REPEAT backward input");
    }
    {
        Graph graph;
        LogicalId x = makeFloatInput(graph, {8});
        LogicalId starts = makeIntConst(graph, {2});
        LogicalId ends = makeIntConst(graph, {6});
        LogicalId steps = makeIntConst(graph, {1});
        LogicalId sliceId = graph.slice(x, starts, ends, steps);
        prop.inferShapeRecursive(sliceId, graph);
        auto forward =
            prop.forward(graph.getNode(sliceId), graph, {{makeRegion({{2, 3}}), makeRegion({{5, 6}})}, {}, {}, {}});
        assertRegionListEquals(forward, {makeRegion({{0, 1}}), makeRegion({{3, 4}})}, "SLICE forward");
        auto backward = prop.backward(graph.getNode(sliceId), graph, {makeRegion({{0, 1}}), makeRegion({{3, 4}})});
        assertRegionListEquals(backward[0], {makeRegion({{2, 3}}), makeRegion({{5, 6}})}, "SLICE backward input");
    }
    {
        Graph graph;
        LogicalId target = makeFloatInput(graph, {8});
        LogicalId updates = makeFloatInput(graph, {4});
        LogicalId starts = makeIntConst(graph, {2});
        LogicalId ends = makeIntConst(graph, {6});
        LogicalId steps = makeIntConst(graph, {1});
        LogicalId scatterId = graph.scatter(target, updates, starts, ends, steps);
        prop.inferShapeRecursive(scatterId, graph);
        auto forward =
            prop.forward(graph.getNode(scatterId), graph, {{makeRegion({{0, 2}})}, {makeRegion({{1, 3}})}, {}, {}, {}});
        assertRegionListEquals(forward, {makeRegion({{0, 2}}), makeRegion({{3, 5}})}, "SCATTER forward");
        auto backward = prop.backward(graph.getNode(scatterId), graph, {makeRegion({{3, 5}})});
        assertRegionListEquals(backward[0], {makeRegion({{3, 5}})}, "SCATTER backward target");
        assertRegionListEquals(backward[1], {makeRegion({{1, 3}})}, "SCATTER backward updates");
    }
    {
        Graph graph;
        LogicalId data = makeFloatInput(graph, {4, 3});
        LogicalId idx = makeIntConst(graph, {2});
        LogicalId gatherId = graph.gather(data, idx);
        prop.inferShapeRecursive(gatherId, graph);
        auto forward =
            prop.forward(graph.getNode(gatherId), graph, {{makeRegion({{1, 3}, {0, 3}})}, {makeRegion({{0, 2}})}});
        assertRegionListEquals(forward, {makeRegion({{0, 2}, {0, 3}})}, "GATHER forward");
        auto backward = prop.backward(graph.getNode(gatherId), graph, {makeRegion({{0, 2}, {1, 3}})});
        assertRegionListEquals(backward[0], {makeRegion({{2, 3}, {1, 3}})}, "GATHER backward data");
        assertRegionListEquals(backward[1], {makeRegion({{0, 2}})}, "GATHER backward idx");
    }
    {
        Graph graph;
        LogicalId data = makeFloatInput(graph, {4, 3});
        LogicalId idxSrc = makeIntConst(graph, {1, 3, 0, 2});
        // LogicalId sliceStart = makeIntConst(graph, {0});
        // LogicalId sliceEnd = makeIntConst(graph, {2});
        // LogicalId sliceStep = makeIntConst(graph, {1});
        // LogicalId idx = graph.slice(idxSrc, sliceStart, sliceEnd, sliceStep);
        // prop.inferShapeRecursive(idx, graph);
        LogicalId gatherId = graph.gather(data, idxSrc);
        prop.inferShapeRecursive(gatherId, graph);
        auto backward = prop.backward(graph.getNode(gatherId), graph, {makeRegion({{0, 2}, {0, 3}})});
        assertRegionListEquals(backward[0], {makeRegion({{1, 2}, {0, 3}}), makeRegion({{3, 4}, {0, 3}})},
                               "GATHER backward sliced indices data");
        assertRegionListEquals(backward[1], {makeRegion({{0, 2}})}, "GATHER backward sliced indices idx");
    }
}

std::vector<float> executeReferenceGraph(LogicalId rootId, Graph &graph,
                                         const std::unordered_map<LogicalId, std::vector<uint8_t>> &rawInputData,
                                         bool forceNonContiguous = false)
{
    std::vector<LogicalId> topo = topologicalSort({rootId}, graph);
    ShapePropagator prop;
    for (LogicalId nodeId : topo)
    {
        prop.inferShape(nodeId, graph);
    }

    std::unordered_map<LogicalId, std::vector<uint8_t>> results;
    std::unordered_map<LogicalId, TensorView> views;
    for (LogicalId nodeId : topo)
    {
        const TensorNode &node = graph.getNode(nodeId);
        uint64_t elemSize = getDTypeSize(node.dtype);

        if (node.opType == OpType::INPUT || node.opType == OpType::CACHE)
        {
            TensorView view = makeView(node);
            if (forceNonContiguous)
            {
                for (auto &s : view.strides)
                    s *= 2;
            }
            views[nodeId] = view;
            uint64_t bufElements = getRequiredBufferSize(view);
            results[nodeId].resize(bufElements * elemSize, 0);
            std::vector<uint8_t> rawBytes;
            auto it = rawInputData.find(nodeId);
            if (it != rawInputData.end())
            {
                rawBytes = it->second;
            }
            else if (graph.constantStaging.count(nodeId))
            {
                rawBytes = *graph.constantStaging.at(nodeId);
            }
            else
            {
                Error::throw_err("[executeReferenceGraph] input node value not found "
                                 "in constantStaging or inputData");
            }

            uint64_t numElements = countElements(view);
            for (uint64_t i = 0; i < numElements; ++i)
            {
                uint64_t idx = getStridedIndex(i, view.getShape(), view.strides);
                std::memcpy(results[nodeId].data() + idx * elemSize, rawBytes.data() + i * elemSize, elemSize);
            }
            continue;
        }

        std::vector<const void *> inputPtrs;
        std::vector<TensorView> inputViews;
        std::vector<TensorNode> inputNodes;
        for (LogicalId pid : node.child_ids)
        {
            auto resultIt = results.find(pid);
            if (resultIt == results.end())
            {
                Error::throw_err("Parent node " + std::to_string(pid.value) + " not found in results");
            }
            inputPtrs.push_back(resultIt->second.data());
            inputViews.push_back(views[pid]);
            TensorNode inNode = graph.getNode(pid);
            inNode.strides = views[pid].strides;
            inputNodes.push_back(inNode);
        }

        TensorView outViewContig = makeView(node);
        TensorView outViewNonContig = outViewContig;
        if (forceNonContiguous)
        {
            for (auto &s : outViewNonContig.strides)
                s *= 2;
        }

        TensorNode outNodeNC = node;
        auto refs_nc = KernelRegistry::get().findMatchingKernels(
            node.opType, node.opName, inputNodes, outNodeNC, true, MemSpace{1, HandleType::CPP}, {},
            {Engine{0, EngineType::CPU}}, false, true, false, true);
        TensorView chosenOutView;
        KernelId chosenKernelUid = KernelId{0};
        if (forceNonContiguous && !refs_nc.empty())
        {
            chosenOutView = outViewNonContig;
            chosenKernelUid = refs_nc.front();
        }
        else
        {
            TensorNode outNodeC = node;
            auto refs_c = KernelRegistry::get().findMatchingKernels(
                node.opType, node.opName, inputNodes, outNodeC, true, MemSpace{1, HandleType::CPP}, {},
                {Engine{0, EngineType::CPU}}, false, true, false, true);
            if (refs_c.empty())
            {
                Error::throw_err("No reference kernel found for node " + std::to_string(nodeId.value) +
                                 " op=" + toString(node.opType) +
                                 (node.opType == OpType::FUSED ? " (" + node.opName + ")" : ""));
            }
            chosenOutView = outViewContig;
            chosenKernelUid = refs_c.front();
        }

        const KernelEntry &kernel = KernelRegistry::get().getKernel(chosenKernelUid);
        if (kernel.is_view)
        {
            TensorView dummyOutView(node, 0);
            kernel.inferView(inputNodes, dummyOutView, graph);
            LogicalId parentId = node.child_ids[0];
            results[nodeId] = results[parentId];
            chosenOutView.strides = dummyOutView.strides;
            views[nodeId] = chosenOutView;
            continue;
        }

        views[nodeId] = chosenOutView;
        uint64_t bufElements = getRequiredBufferSize(chosenOutView);
        results[nodeId].resize(bufElements * elemSize, 0);
        std::vector<void *> outputPtrs = {results[nodeId].data()};
        std::vector<TensorView> outputViews = {chosenOutView};

        if (kernel.run)
        {
            kernel.run(KernelContext(inputPtrs, outputPtrs, inputViews, outputViews));
        }
    }

    uint64_t numRootElems = countElements(graph.getNode(rootId));
    std::vector<float> finalOut(numRootElems, 0.0f);
    TensorView rootView = views[rootId];
    for (uint64_t i = 0; i < numRootElems; ++i)
    {
        uint64_t idx = getStridedIndex(i, rootView.getShape(), rootView.strides);
        if (graph.getNode(rootId).dtype == DType::FLOAT32)
        {
            std::memcpy(&finalOut[i], results[rootId].data() + idx * 4, 4);
        }
        else if (graph.getNode(rootId).dtype == DType::INT32)
        {
            int32_t val;
            std::memcpy(&val, results[rootId].data() + idx * 4, 4);
            finalOut[i] = static_cast<float>(val);
        }
        else if (graph.getNode(rootId).dtype == DType::BF16)
        {
            uint16_t val;
            std::memcpy(&val, results[rootId].data() + idx * 2, 2);
            uint32_t f32_bits = static_cast<uint32_t>(val) << 16;
            std::memcpy(&finalOut[i], &f32_bits, 4);
        }
        else if (graph.getNode(rootId).dtype == DType::BOOL)
        {
            uint8_t val;
            std::memcpy(&val, results[rootId].data() + idx, 1);
            finalOut[i] = static_cast<float>(val);
        }
        else
        {
            Error::throw_err("[executeReferenceGraph] Unsupported dtype");
        }
    }
    return finalOut;
}

std::vector<float> executeFusedKernel(const KernelEntry &kernel, const std::vector<std::vector<uint8_t>> &inputData,
                                      const std::vector<LogicalId> &inputIds, const std::vector<uint32_t> &outShape,
                                      const std::vector<uint64_t> &outStrides, DType outDType, const Graph &graph)
{
    if (inputData.size() < kernel.min_num_inputs || inputData.size() > kernel.max_num_inputs)
    {
        Error::throw_err("Fused kernel " + kernel.opName + " inputs count mismatch");
    }

    Record r;
    r.kernelId = kernel.uid;
    r.outputShape = outShape;
    r.outputStrides = outStrides;
    r.outputDType = outDType;
    r.output_mem_space = kernel.output_mem_space;
    r.engines = kernel.engines;

    for (uint64_t i = 0; i < inputIds.size(); ++i)
    {
        const TensorNode &node = graph.getNode(inputIds[i]);
        r.inputShapes.push_back(node.getShape());
        r.inputStrides.push_back(node.strides.empty() ? calcContiguousStrides(node.getShape()) : node.strides);
        r.inputDTypes.push_back(node.dtype);

        MemSpace b = {1, HandleType::CPP};
        uint64_t ruleIdx =
            std::min((uint64_t)i,
                     static_cast<uint64_t>(kernel.input_mem_spaces.empty() ? 0 : kernel.input_mem_spaces.size() - 1));
        if (ruleIdx < kernel.input_mem_spaces.size())
        {
            b = kernel.input_mem_spaces[ruleIdx];
        }
        r.input_mem_spaces.push_back(b);
    }

    PreparedKernel pk;
    pk.prepare(kernel, r, &inputData);
    pk.updateStorageContext(kernel, r, 0);
    pk.run(kernel);
    pk.synchronize();
    pk.download();

    TensorView outView;
    outView.setShape(outShape);
    outView.strides = outStrides.empty() ? calcContiguousStrides(outShape) : outStrides;
    outView.offset = 0;
    outView.dtype = outDType;

    if (kernel.is_view && !pk.inputBuffers.empty() && kernel.inferView)
    {
        std::vector<TensorNode> dummyInputs(inputData.size());
        for (uint64_t i = 0; i < inputData.size(); ++i)
        {
            dummyInputs[i].id = inputIds[i];
            dummyInputs[i].setShape(pk.inViews[i].getShape());
            dummyInputs[i].strides = pk.inViews[i].strides;
            dummyInputs[i].dtype = pk.inViews[i].dtype;
        }
        TensorView dummyOutView;
        dummyOutView.setShape(outShape);
        dummyOutView.dtype = outDType;
        kernel.inferView(dummyInputs, dummyOutView, graph);

        outView.strides = dummyOutView.strides;

        return flattenOutput(pk.inputBuffers[0].hostData.data() + outView.offset, outView.getShape(), outView.strides,
                             outView.dtype);
    }

    return flattenOutput(pk.outputBuffers[0].hostData.data() + outView.offset, outView.getShape(), outView.strides,
                         outView.dtype);
}

struct TestInputs
{
    std::vector<LogicalId> inputIds;
    std::unordered_map<LogicalId, std::vector<uint8_t>> rawInputData;
    std::vector<std::vector<uint8_t>> rawData;
};

TestInputs createTestInputs(Graph &graph, const KernelEntry &kernel)
{
    TestInputs result;
    result.rawData.resize(kernel.min_num_inputs);
    result.inputIds.resize(kernel.min_num_inputs);

    std::vector<bool> isConstantParam(kernel.min_num_inputs, false);
    std::vector<std::vector<int32_t>> constantValues(kernel.min_num_inputs);

    if (!kernel.isReference && kernel.refFactory)
    {
        Graph tempGraph;
        std::vector<LogicalId> tempInputs;
        for (uint64_t i = 0; i < kernel.min_num_inputs; ++i)
        {
            DType d = static_cast<uint32_t>(kernel.dtypes[i]) == static_cast<uint32_t>(DType::ANY) ? DType::FLOAT32
                                                                                                   : kernel.dtypes[i];
            tempInputs.push_back(tempGraph.input(kernel.dummyShapes[i], d));
        }

        kernel.refFactory(tempInputs, tempGraph);

        for (const auto &pair : tempGraph.nodes)
        {
            const TensorNode &n = pair.second;

            auto traceToInputIdx = [&](LogicalId pid) -> int {
                LogicalId curr = pid;
                while (tempGraph.hasNode(curr) && (tempGraph.getNode(curr).opType == OpType::CONTIGUOUS ||
                                                   tempGraph.getNode(curr).opType == OpType::CAST ||
                                                   tempGraph.getNode(curr).opType == OpType::RESHAPE ||
                                                   tempGraph.getNode(curr).opType == OpType::PERMUTE ||
                                                   tempGraph.getNode(curr).opType == OpType::COPY_TO))
                {
                    if (tempGraph.getNode(curr).child_ids.empty())
                        break;
                    curr = tempGraph.getNode(curr).child_ids[0];
                }
                for (uint64_t i = 0; i < kernel.min_num_inputs; ++i)
                {
                    if (tempInputs[i] == curr)
                        return (int)i;
                }
                return -1;
            };

            auto checkParam = [&](uint64_t parentIdx, const std::vector<int32_t> &defaultVals) {
                if (parentIdx < n.child_ids.size())
                {
                    int inputIdx = traceToInputIdx(n.child_ids[parentIdx]);
                    if (inputIdx >= 0)
                    {
                        isConstantParam[inputIdx] = true;
                        if (constantValues[inputIdx].empty())
                        {
                            constantValues[inputIdx] = defaultVals;
                        }
                    }
                }
            };

            if (n.opType == OpType::REPEAT)
            {
                checkParam(1, {2});
                checkParam(2, {0});
            }
            else if (n.opType == OpType::RESHAPE)
            {
                std::vector<int32_t> shapeVals;
                int srcIdx = traceToInputIdx(n.child_ids[0]);
                if (srcIdx >= 0)
                {
                    for (auto s : kernel.dummyShapes[srcIdx])
                        shapeVals.push_back((int32_t)s);
                }
                if (shapeVals.empty())
                    shapeVals = {1};
                checkParam(1, shapeVals);
            }
            else if (n.opType == OpType::PERMUTE)
            {
                std::vector<int32_t> perm;
                int srcIdx = traceToInputIdx(n.child_ids[0]);
                if (srcIdx >= 0)
                {
                    uint64_t rank = kernel.dummyShapes[srcIdx].size();
                    for (uint64_t i = 0; i < rank; ++i)
                    {
                        perm.push_back(rank == 2 ? (int32_t)(1 - i) : (int32_t)i);
                    }
                }
                if (perm.empty())
                    perm = {0};
                checkParam(1, perm);
            }
            else if (n.opType == OpType::SLICE)
            {
                std::vector<int32_t> starts, ends, steps;
                int srcIdx = traceToInputIdx(n.child_ids[0]);
                if (srcIdx >= 0)
                {
                    for (auto s : kernel.dummyShapes[srcIdx])
                    {
                        starts.push_back(0);
                        ends.push_back((int32_t)s);
                        steps.push_back(1);
                    }
                }
                else
                {
                    starts = {0};
                    ends = {2147483647};
                    steps = {1};
                }
                checkParam(1, starts);
                checkParam(2, ends);
                checkParam(3, steps);
            }
            else if (n.opType == OpType::SCATTER)
            {
                std::vector<int32_t> starts, ends, steps;
                int srcIdx = traceToInputIdx(n.child_ids[0]); // Target tensor
                if (srcIdx >= 0)
                {
                    for (auto s : kernel.dummyShapes[srcIdx])
                    {
                        starts.push_back(0);
                        ends.push_back((int32_t)s);
                        steps.push_back(1);
                    }
                }
                else
                {
                    starts = {0};
                    ends = {2147483647};
                    steps = {1};
                }
                checkParam(2, starts);
                checkParam(3, ends);
                checkParam(4, steps);
            }
            else if (n.opType == OpType::SUM || n.opType == OpType::MAX)
            {
                checkParam(1, {-1});
            }
            else if (n.opType == OpType::CONCAT)
            {
                checkParam(0, {0});
            }
            else if (n.opType == OpType::TRIU)
            {
                checkParam(1, {1});
            }
            else if (n.opType == OpType::FILL)
            {
                checkParam(1, {1});
            }
            else if (n.opType == OpType::IM2COL)
            {
                checkParam(1, {1});
                checkParam(2, {1});
                checkParam(3, {0});
            }
            else if (n.opType == OpType::ARANGE)
            {
                checkParam(0, {0});
                checkParam(1, {1});
                checkParam(2, {1});
            }
            else if (n.opType == OpType::ARGMAX)
            {
                checkParam(1, {-1});
                checkParam(2, {1});
            }
        }
    }

    for (uint64_t i = 0; i < kernel.min_num_inputs; ++i)
    {
        LogicalId id;
        DType dtype = static_cast<uint32_t>(kernel.dtypes[i]) == static_cast<uint32_t>(DType::ANY) ? DType::FLOAT32
                                                                                                   : kernel.dtypes[i];
        uint64_t elements = countElements(kernel.dummyShapes[i]);
        uint64_t sizeBytes = elements * getDTypeSize(dtype);

        if (isConstantParam[i])
        {
            std::vector<int32_t> constData(elements, 0);
            if (!constantValues[i].empty())
            {
                for (uint64_t j = 0; j < elements; ++j)
                {
                    constData[j] = constantValues[i][j % constantValues[i].size()];
                }
            }
            id = graph.constant(kernel.dummyShapes[i], constData.data(), dtype);
            result.rawData[i].resize(sizeBytes);
            std::memcpy(result.rawData[i].data(), constData.data(), sizeBytes);
        }
        else
        {
            id = graph.input(kernel.dummyShapes[i], dtype, {});
            result.rawData[i].resize(sizeBytes);
            fillRandom(result.rawData[i].data(), elements, dtype);
        }
        result.rawInputData[id] = result.rawData[i];
        result.inputIds[i] = id;
    }
    return result;
}

bool testKernelWithRecord(const KernelEntry &kernel, const Record &rec)
{
    try
    {
        if (rec.inputShapes.size() < kernel.min_num_inputs || rec.inputShapes.size() > kernel.max_num_inputs)
            return true; // Skip mismatched variadic/arity records

        // Build dummy nodes for validation against centralized matching logic
        std::vector<TensorNode> dummyInputs(rec.inputShapes.size());
        for (uint64_t idx = 0; idx < rec.inputShapes.size(); ++idx)
        {
            dummyInputs[idx].setShape(rec.inputShapes[idx]);
            dummyInputs[idx].strides = rec.inputStrides[idx];
            dummyInputs[idx].dtype = rec.inputDTypes[idx];
        }

        TensorNode dummyOutput;
        std::vector<uint32_t> outShape;
        std::vector<uint64_t> outStrides;
        DType outDType = DType::FLOAT32;

        if (!rec.outputShape.empty())
        {
            outShape = rec.outputShape;
            outStrides = rec.outputStrides;
            outDType = rec.outputDType;
            dummyOutput.setShape(outShape);
            dummyOutput.strides = outStrides;
            dummyOutput.dtype = outDType;
        }
        else
        {
            outShape = rec.inputShapes.empty() ? std::vector<uint32_t>{} : rec.inputShapes[0];
            outStrides = rec.inputStrides.empty() ? std::vector<uint64_t>{} : rec.inputStrides[0];
            outDType = rec.inputDTypes.empty() ? DType::FLOAT32 : rec.inputDTypes[0];

            dummyOutput.setShape(outShape);
            dummyOutput.strides = outStrides;
            dummyOutput.dtype = outDType;
        }

        if (!kernel.matches(dummyInputs, dummyOutput))
            return true; // Skip invalid records

        Graph graph;
        std::vector<std::vector<uint8_t>> rawData(rec.inputShapes.size());
        std::unordered_map<LogicalId, std::vector<uint8_t>> rawInputData;
        std::vector<LogicalId> inputIds(rec.inputShapes.size());

        for (uint64_t i = 0; i < rec.inputShapes.size(); ++i)
        {
            TensorView view;
            view.setShape(rec.inputShapes[i]);
            view.strides = rec.inputStrides[i];

            uint64_t elements = countElements(view.getShape());
            uint64_t bufElements = getRequiredBufferSize(view);
            uint64_t dtypeSize = getDTypeSize(rec.inputDTypes[i]);

            rawData[i].resize(bufElements * dtypeSize);

            // Contiguous array for executeReferenceGraph standard scattering
            std::vector<uint8_t> contiguousData(elements * dtypeSize);

            bool isConstant = false;
            if (i < rec.inputConstants.size() && !rec.inputConstants[i].empty() &&
                rec.inputConstants[i].size() == elements * dtypeSize)
            {
                isConstant = true;
                std::memcpy(contiguousData.data(), rec.inputConstants[i].data(), rec.inputConstants[i].size());
            }
            else
            {
                fillRandom(contiguousData.data(), elements, rec.inputDTypes[i]);
                if (rec.inputDTypes[i] == DType::INT32)
                {
                    int32_t *iptr = reinterpret_cast<int32_t *>(contiguousData.data());
                    if (kernel.opType == OpType::CONCAT || kernel.opName.find("Concat") != std::string::npos)
                    {
                        if (i == 0)
                        {
                            int32_t concat_axis = -1;
                            if (!rec.inputShapes.empty() && !rec.outputShape.empty() && rec.inputShapes.size() > 1)
                            {
                                for (uint64_t d = 0; d < rec.outputShape.size(); ++d)
                                {
                                    if (rec.outputShape[d] != rec.inputShapes[1][d])
                                    {
                                        concat_axis = (int32_t)d;
                                        break;
                                    }
                                }
                            }
                            if (concat_axis == -1)
                                concat_axis = 0;
                            for (uint64_t k = 0; k < elements; ++k)
                                iptr[k] = concat_axis;
                        }
                    }
                }
            }

            // Scatter contiguousData physically into rawData[i] using strides
            for (uint64_t k = 0; k < elements; ++k)
            {
                uint64_t idx = getStridedIndex(k, view.getShape(), view.strides);
                std::memcpy(rawData[i].data() + idx * dtypeSize, contiguousData.data() + k * dtypeSize, dtypeSize);
            }

            if (isConstant)
            {
                inputIds[i] = graph.constant(rec.inputShapes[i], contiguousData.data(), rec.inputDTypes[i]);
            }
            else
            {
                inputIds[i] = graph.input(rec.inputShapes[i], rec.inputDTypes[i], {});
                if (rec.inputDTypes[i] == DType::INT32)
                {
                    graph.constantStaging[inputIds[i]] = std::make_shared<std::vector<uint8_t>>(contiguousData);
                }
            }

            graph.getNode(inputIds[i]).strides = rec.inputStrides[i];
            rawInputData[inputIds[i]] = contiguousData;
        }

        LogicalId rootId = kernel.refFactory(inputIds, graph);

        // Reference graph will handle the continuous mapping identically internally
        std::vector<float> refOutput = executeReferenceGraph(rootId, graph, rawInputData, false);
        // Target fused execution resolves dynamically spread arrays
        std::vector<float> tgtOutput =
            executeFusedKernel(kernel, rawData, inputIds, outShape, outStrides, outDType, graph);

        if (refOutput.size() != tgtOutput.size())
        {
            std::cout << "\n[Record Test Error] Output size mismatch: ref=" << refOutput.size()
                      << " tgt=" << tgtOutput.size() << " kernel=" << kernel.opName << std::endl;
            return false;
        }

        return compareOutputs(refOutput.data(), tgtOutput.data(), refOutput.size());
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n[Record Test Exception] " << e.what() << std::endl;
        return false;
    }
}

std::unordered_map<KernelId, std::vector<Record>> loadCallRecords(const std::string &path)
{
    std::unordered_map<KernelId, std::vector<Record>> records;
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        return records;

    BinaryReader br(file);
    while (file.peek() != EOF)
    {
        Record r;
        br.read(r);
        records[r.kernelId].push_back(std::move(r));
    }
    return records;
}

void runPythonTests(std::string testDir = "tensor_graphs_cpp/tests")
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running Python Reference Kernel Tests..." << std::endl;
    std::cout << "========================================" << std::endl;
    if (!std::filesystem::exists(testDir))
    {
        std::cout << "No " << testDir << " directory found. Skipping python reference tests." << std::endl;
        return;
    }
    int passed = 0;
    int total = 0;
    std::vector<std::string> testDirs;
    for (const auto &entry : std::filesystem::directory_iterator(testDir))
    {
        if (entry.is_directory())
            testDirs.push_back(entry.path().string());
    }
    std::sort(testDirs.begin(), testDirs.end(), [](const std::string &a, const std::string &b) {
        std::string na = std::filesystem::path(a).filename().string();
        std::string nb = std::filesystem::path(b).filename().string();
        return a < b;
    });
    for (const std::string &testDir : testDirs)
    {
        total++;
        std::string infoPath = testDir + "/info.bin";
        std::string dataPath = testDir + "/data.safetensors";
        std::ifstream infoFile(infoPath, std::ios::binary);
        if (!infoFile.is_open())
            continue;

        BinaryReader br(infoFile);
        Record rec;
        br.read(rec);
        OpType opType = static_cast<OpType>(rec.kernelId.value);

        SafetensorsLoader loader(dataPath);
        std::vector<std::vector<uint8_t>> inputData;
        std::vector<TensorView> inViews;
        std::vector<TensorNode> dummyInputNodes;
        std::vector<const void *> inPtrs;
        Graph dummyGraph;
        for (uint64_t i = 0; i < rec.inputShapes.size(); ++i)
        {
            std::vector<uint32_t> shape = rec.inputShapes[i];
            std::vector<uint64_t> strides = rec.inputStrides[i];
            DType dtype = rec.inputDTypes[i];
            std::string tensorName = "input." + std::to_string(i);
            uint64_t sizeBytes = countElements(shape) * getDTypeSize(dtype);
            std::vector<uint8_t> data(sizeBytes);
            loader.loadTensor(tensorName, data.data(), sizeBytes);
            inputData.push_back(std::move(data));
            TensorView view;
            view.offset = 0;
            view.setShape(shape);
            view.strides = strides;
            view.dtype = dtype;
            inViews.push_back(view);
            TensorNode &node = dummyGraph.allocateNode(OpType::INPUT, "", dtype, {}, shape, strides, "");
            dummyInputNodes.push_back(node);
            if (dtype == DType::INT32)
                dummyGraph.constantStaging[node.id] = std::make_shared<std::vector<uint8_t>>(inputData.back());
        }
        for (auto &vec : inputData)
            inPtrs.push_back(vec.data());

        std::vector<uint32_t> outShape = rec.outputShape;
        std::vector<uint64_t> outStrides = rec.outputStrides;
        DType outDType = rec.outputDType;
        uint64_t outSizeBytes = countElements(outShape) * getDTypeSize(outDType);
        std::vector<uint8_t> expectedData(outSizeBytes);
        loader.loadTensor("output", expectedData.data(), outSizeBytes);
        std::vector<uint8_t> actualData(outSizeBytes);
        std::vector<void *> outPtrs = {actualData.data()};
        TensorView outView;
        outView.offset = 0;
        outView.setShape(outShape);
        outView.strides = outStrides;
        outView.dtype = outDType;
        std::vector<TensorView> outViews = {outView};
        TensorNode outNode;
        outNode.id = LogicalId{(uint32_t)rec.inputShapes.size()};
        outNode.dtype = outDType;
        outNode.setShape(outShape);
        outNode.strides = outStrides;

        std::cout << "Testing Python Ref " << testDir << " [" << toString(opType) << "] ... " << std::flush;
        std::vector<KernelId> matches = KernelRegistry::get().findMatchingKernels(
            opType, "", dummyInputNodes, outNode, true, MemSpace{1, HandleType::CPP}, {}, {Engine{0, EngineType::CPU}},
            false, true, false, true);
        if (matches.empty())
        {
            Error::throw_err("[runPythonTests] FAILED (No reference kernel found)");
        }
        if (matches.size() > 1)
        {
            Error::throw_err("[runPythonTests] Expected 1 kernel match, got " + std::to_string(matches.size()));
        }
        const KernelEntry &kernel = KernelRegistry::get().getKernel(matches.front());
        if (kernel.is_view)
        {
            for (uint64_t k = 0; k < dummyInputNodes.size(); ++k)
            {
                dummyInputNodes[k].strides = inViews[k].strides;
            }
            TensorView dummyOutView(outNode, 0);
            kernel.inferView(dummyInputNodes, dummyOutView, dummyGraph);
            uint64_t elements = countElements(outShape);
            if (outDType == DType::FLOAT32)
            {
                const float *src = reinterpret_cast<const float *>(inputData[0].data());
                float *dst = reinterpret_cast<float *>(actualData.data());
                for (uint64_t k = 0; k < elements; ++k)
                {
                    uint64_t srcIdx = getStridedIndex(k, dummyOutView.getShape(), dummyOutView.strides);
                    dst[k] = src[srcIdx];
                }
            }
            else if (outDType == DType::INT32)
            {
                const int32_t *src = reinterpret_cast<const int32_t *>(inputData[0].data());
                int32_t *dst = reinterpret_cast<int32_t *>(actualData.data());
                for (uint64_t k = 0; k < elements; ++k)
                {
                    uint64_t srcIdx = getStridedIndex(k, dummyOutView.getShape(), dummyOutView.strides);
                    dst[k] = src[srcIdx];
                }
            }
        }
        else if (kernel.run)
        {
            kernel.run(KernelContext(inPtrs, outPtrs, inViews, outViews));
        }

        bool ok = false;
        if (outDType == DType::FLOAT32)
        {
            ok = compareOutputs((const float *)expectedData.data(), (const float *)actualData.data(),
                                countElements(outShape));
        }
        else if (outDType == DType::INT32)
        {
            ok = compareOutputs((const int32_t *)expectedData.data(), (const int32_t *)actualData.data(),
                                countElements(outShape));
        }
        else if (outDType == DType::BOOL)
        {
            ok = compareOutputs((const bool *)expectedData.data(), (const bool *)actualData.data(),
                                countElements(outShape));
        }
        else
        {
            Error::throw_err("[runPythonTests] Unsupported type: " + (std::string)toString(outDType));
        }
        if (ok)
        {
            passed++;
            std::cout << "OK" << std::endl;
        }
        else
        {
            std::cout << "FAILED (Output Mismatch)" << std::endl;
        }
    }
    std::cout << "\n----------------------" << std::endl;
    std::cout << "Python Reference Tests Passed: " << passed << "/" << total << std::endl;
    std::cout << "----------------------\n" << std::endl;
}

std::unordered_map<KernelId, std::vector<Record>> getRecordsFromCache(const std::string &cachePath)
{
    std::unordered_map<KernelId, std::vector<Record>> recordsByUid;
    std::unordered_set<std::string> seen;

    std::ifstream file(cachePath, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Warning: Could not open cache file: " << cachePath << std::endl;
        return recordsByUid;
    }

    BinaryReader br(file);
    while (file.peek() != EOF)
    {
        uint8_t type;
        br.read(type);

        if (type == 0) // Metadata
        {
            uint32_t version;
            LogicalId cachedRootId;
            std::unordered_map<LogicalId, MemSpace> tempSelected;
            br.read(version);
            br.read(cachedRootId);
            br.read(tempSelected);
        }
        else if (type == 1) // Compiled Bucket
        {
            CompiledGraph cg;
            br.read(cg);

            for (const auto &inst : cg.instructions)
            {
                if (inst.kernel_id.value == 0)
                    continue;

                Record r;
                r.kernelId = inst.kernel_id;
                r.buildContextId = BUILD_CONTEXT_ID;
                r.hwTag = HW_TAG;
                r.runTime = 0.0f;

                const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.kernel_id);
                r.output_mem_space = kernel.output_mem_space;
                r.engines = kernel.engines;

                for (uint32_t i = 0; i < inst.children.size(); i++)
                {
                    EClassId inId = inst.children[i];
                    const TensorView &inView = cg.nodeViews.at(inId);
                    r.inputShapes.push_back(inView.getShape());
                    r.inputStrides.push_back(inView.strides);
                    r.inputDTypes.push_back(inView.dtype);

                    uint64_t ruleIdx = std::min(
                        (uint64_t)i, static_cast<uint64_t>(
                                         kernel.input_mem_spaces.empty() ? 0 : kernel.input_mem_spaces.size() - 1));
                    MemSpace ms = {1, HandleType::CPP};
                    if (!kernel.input_mem_spaces.empty() && ruleIdx < kernel.input_mem_spaces.size())
                    {
                        ms = kernel.input_mem_spaces[ruleIdx];
                    }
                    r.input_mem_spaces.push_back(ms);

                    if (cg.has_logical_id(inId))
                    {
                        LogicalId logicalId = cg.get_logical_id(inId);
                        if (cg.constantStaging.count(EClassId{logicalId.value}))
                        {
                            r.inputConstants.push_back(*cg.constantStaging.at(EClassId{logicalId.value}));
                        }
                        else if (cg.constantStaging.count(inId))
                        {
                            r.inputConstants.push_back(*cg.constantStaging.at(inId));
                        }
                        else
                        {
                            r.inputConstants.push_back({});
                        }
                    }
                    else if (cg.constantStaging.count(inId))
                    {
                        r.inputConstants.push_back(*cg.constantStaging.at(inId));
                    }
                    else
                    {
                        r.inputConstants.push_back({});
                    }
                }

                std::string sig = serializeToString(r);
                if (seen.insert(sig).second)
                {
                    recordsByUid[r.kernelId].push_back(r);
                }
            }
        }
        else if (type == 2) // Constants
        {
            uint32_t count;
            br.read(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                uint32_t n;
                std::vector<uint8_t> d;
                br.read(n);
                br.read(d);
            }
        }
        else
        {
            break;
        }
    }
    return recordsByUid;
}

int main(int argc, char *argv[])
{
    ArgParser parser("test", "Run tests.");
    parser.add_flag({"--no-records"}, "Disable record-based testing.");
    parser.add_option({"--cache"},
                      "Path to cache file. If provided, only kernel calls "
                      "present in the cache file will be tested.",
                      "");
    parser.add_positional("targetKernel", "Test only kernels whose name contain this string.", "");

    if (!parser.parse(argc, argv))
    {
        return 1;
    }

    std::string targetKernel = parser.get_positional("targetKernel");
    bool useRecords = !parser.get_flag("--no-records");
    std::string cachePath = parser.get_option("--cache");

    if (targetKernel.empty() && cachePath.empty())
    {
        runRegionMergeTests();
        runShapePropagationTests();
        // runPythonTests(); TODO: fix python tests
    }

    std::unordered_map<KernelId, std::vector<Record>> recordsByUid;
    if (!cachePath.empty())
    {
        recordsByUid = getRecordsFromCache(cachePath);
        std::cout << "Loaded kernel configurations strictly from cache." << std::endl;
    }
    else if (useRecords)
    {
        recordsByUid = loadCallRecords("benchmarks/calls.bin");
        if (recordsByUid.empty())
        {
            std::cout << "Warning: benchmarks/calls.bin not found or empty." << std::endl;
        }
    }

    std::cout << "Running Non-Reference Kernel Tests..." << std::endl;
    int passed = 0;
    int total = 0;
    int skipped = 0;

    const auto &kernels = KernelRegistry::get().getAllKernels();
    for (const auto &[uid, kernel] : kernels)
    {
        if (kernel.isReference)
            continue;

        if (!targetKernel.empty() && kernel.opName.find(targetKernel) == std::string::npos)
            continue;

        if (!cachePath.empty() && recordsByUid.find(kernel.uid) == recordsByUid.end())
            continue;

        if (!kernel.refFactory)
        {
            std::cout << "Skipping " << kernel.opName << " (no refFactory)" << std::endl;
            skipped++;
            continue;
        }
        if (kernel.dummyShapes.size() != kernel.min_num_inputs)
        {
            std::cout << "Skipping " << kernel.opName << " (dummy shapes mismatch)" << std::endl;
            skipped++;
            continue;
        }

        total++;
        std::cout << "[" << std::to_string(total) << "/" << std::to_string(kernels.size()) << "] Testing "
                  << kernel.opName << " ... " << std::flush;

        bool dummyOk = true;
        if (cachePath.empty())
        {
            // 1. Dummy Shapes Test
            Graph refGraph;
            TestInputs refInputs = createTestInputs(refGraph, kernel);
            LogicalId rootId = kernel.refFactory(refInputs.inputIds, refGraph);

            // Synchronize physical rawData with any stride changes made by refFactory
            for (uint64_t i = 0; i < kernel.min_num_inputs; ++i)
            {
                LogicalId id = refInputs.inputIds[i];
                const TensorNode &node = refGraph.getNode(id);
                if (!node.strides.empty() && node.strides != calcContiguousStrides(node.getShape()))
                {
                    TensorView view;
                    view.setShape(node.getShape());
                    view.strides = node.strides;
                    uint64_t elements = countElements(view.getShape());
                    uint64_t bufElements = getRequiredBufferSize(view);
                    uint64_t dtypeSize = getDTypeSize(node.dtype);

                    std::vector<uint8_t> newRawData(bufElements * dtypeSize, 0);
                    std::vector<uint8_t> &logicalData = refInputs.rawInputData[id];

                    for (uint64_t k = 0; k < elements; ++k)
                    {
                        uint64_t idx = getStridedIndex(k, view.getShape(), view.strides);
                        std::memcpy(newRawData.data() + idx * dtypeSize, logicalData.data() + k * dtypeSize, dtypeSize);
                    }
                    refInputs.rawData[i] = newRawData;
                }
            }

            std::vector<float> refOutput = executeReferenceGraph(rootId, refGraph, refInputs.rawInputData, false);
            uint64_t elements = refOutput.size();

            const TensorNode &rootNode = refGraph.getNode(rootId);
            std::vector<float> fusedOutput =
                executeFusedKernel(kernel, refInputs.rawData, refInputs.inputIds, rootNode.getShape(), rootNode.strides,
                                   rootNode.dtype, refGraph);

            dummyOk = false;
            if (fusedOutput.size() == elements)
                dummyOk = compareOutputs(refOutput.data(), fusedOutput.data(), elements);
        }

        // 2. Record-Based Tests
        bool recordOk = true;
        auto it = recordsByUid.find(kernel.uid);
        if ((useRecords || !cachePath.empty()) && it != recordsByUid.end())
        {
            std::cout << "\n  [Records] Testing " << it->second.size() << " configurations... " << std::flush;
            {
                ProgressTimer timer(it->second.size(), "  ");
                for (const auto &rec : it->second)
                {
                    if (!testKernelWithRecord(kernel, rec))
                    {
                        recordOk = false;
                        break;
                    }
                    timer.tick();
                }
            }
            if (recordOk)
                std::cout << "  OK" << std::endl;
            else
                std::cout << "  FAILED" << std::endl;
        }
        else if (useRecords)
        {
            std::cout << " (no records) ";
        }

        if (dummyOk && recordOk)
        {
            passed++;
            // Only print "OK" if we didn't just print a multi-line record result
            if (!useRecords || it == recordsByUid.end())
                std::cout << "OK" << std::endl;
        }
        else
        {
            std::cout << "FAILED" << std::endl;
        }
    }

    std::cout << "\n----------------------" << std::endl;
    std::cout << "Tests Passed: " << passed << "/" << total << std::endl;
    if (skipped > 0)
        std::cout << "Tests Skipped: " << skipped << std::endl;
    std::cout << "----------------------" << std::endl;
    return 0;
}