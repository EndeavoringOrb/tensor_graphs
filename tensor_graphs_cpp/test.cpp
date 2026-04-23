// tensor_graphs_cpp/test.cpp
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <unordered_set>
#include <filesystem>
#include <type_traits>
#include <fstream>
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/shapes.hpp"
#include "core/planner.hpp"
#include "core/session.hpp"
#include "core/loaders/safetensors.hpp"
#include "core/cost_model.hpp" // For Record struct
#include "generated/kernels_all.gen.hpp"

// ============================================================
// Helper Functions
// ============================================================
void fillRandom(void *ptr, size_t elements, DType dtype)
{
    static std::mt19937 gen(42);
    switch (dtype)
    {
    case DType::FLOAT32:
    {
        float *fptr = static_cast<float *>(ptr);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < elements; ++i)
            fptr[i] = dist(gen);
        break;
    }
    case DType::INT32:
    {
        int32_t *iptr = static_cast<int32_t *>(ptr);
        std::uniform_int_distribution<int32_t> dist(1, 10);
        for (size_t i = 0; i < elements; ++i)
            iptr[i] = dist(gen);
        break;
    }
    case DType::BOOL:
    {
        bool *bptr = static_cast<bool *>(ptr);
        std::uniform_int_distribution<int> dist(0, 1);
        for (size_t i = 0; i < elements; ++i)
            bptr[i] = dist(gen) != 0;
        break;
    }
    case DType::BF16:
    {
        uint16_t *bfptr = static_cast<uint16_t *>(ptr);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < elements; ++i)
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

bool compareOutputs(const float *ref, const float *test, size_t elements, float eps = 1e-4f)
{
    for (size_t i = 0; i < elements; ++i)
    {
        if (std::abs(ref[i] - test[i]) > eps)
        {
            std::cout << "\nMismatch at index " << i << ": (ref)" << ref[i] << " != (test)" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool compareOutputs(const int32_t *ref, const int32_t *test, size_t elements, float eps = 1e-4f)
{
    for (size_t i = 0; i < elements; ++i)
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
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (!regionsMatch(a[i], e[i]))
            return false;
    }
    return true;
}

void assertRegionListEquals(const std::vector<Region> &actual, const std::vector<Region> &expected, const std::string &label)
{
    if (!regionListEquals(actual, expected))
    {
        std::stringstream ss;
        ss << "[RegionTest] " << label << " expected " << encodeRegionList(expected)
           << " but got " << encodeRegionList(actual);
        Error::throw_err(ss.str());
    }
}

void runRegionMergeTests()
{
    std::cout << "region merge tests" << std::endl
              << std::flush;
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
        assertRegionListEquals(actual, {
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
    std::cout << "shape propagation tests" << std::endl
              << std::flush;
    ShapePropagator prop;
    auto makeIntConst = [](Graph &graph, const std::vector<int32_t> &values) -> uint32_t
    {
        return graph.constant({(uint32_t)values.size()}, values.data(), DType::INT32);
    };
    auto makeFloatInput = [](Graph &graph, const std::vector<uint32_t> &shape) -> uint32_t
    {
        return graph.input(shape, DType::FLOAT32, {}, StorageType::PERSISTENT);
    };
    {
        Graph graph;
        uint32_t x = makeFloatInput(graph, {4, 5});
        uint32_t axis = makeIntConst(graph, {1});
        uint32_t sumId = graph.sum(x, axis);
        prop.inferShapeRecursive(sumId, graph);
        auto forward = prop.forward(graph.getNode(sumId), graph, {{makeRegion({{1, 3}, {2, 4}})}, {}});
        assertRegionListEquals(forward, {makeRegion({{1, 3}, {0, 1}})}, "SUM forward");
        auto backward = prop.backward(graph.getNode(sumId), graph, {makeRegion({{1, 3}, {0, 1}})});
        assertRegionListEquals(backward[0], {makeRegion({{1, 3}, {0, 5}})}, "SUM backward input");
        assertRegionListEquals(backward[1], makeFull(graph.getNode(axis).getShape()), "SUM backward axis");
    }
    {
        Graph graph;
        uint32_t x = makeFloatInput(graph, {4, 5});
        uint32_t axis = makeIntConst(graph, {1});
        uint32_t maxId = graph.max(x, axis);
        prop.inferShapeRecursive(maxId, graph);
        auto forward = prop.forward(graph.getNode(maxId), graph, {{makeRegion({{0, 4}, {1, 3}})}, {}});
        assertRegionListEquals(forward, {makeRegion({{0, 4}, {0, 1}})}, "MAX forward");
        auto backward = prop.backward(graph.getNode(maxId), graph, {makeRegion({{0, 4}, {0, 1}})});
        assertRegionListEquals(backward[0], {makeRegion({{0, 4}, {0, 5}})}, "MAX backward input");
    }
    {
        Graph graph;
        uint32_t x = makeFloatInput(graph, {2, 3});
        uint32_t dims = makeIntConst(graph, {1, 0});
        uint32_t permId = graph.permute(x, dims);
        prop.inferShapeRecursive(permId, graph);
        auto forward = prop.forward(graph.getNode(permId), graph,
                                    {{makeRegion({{0, 1}, {1, 3}}), makeRegion({{1, 2}, {0, 2}})}, {}});
        assertRegionListEquals(forward, {
                                            makeRegion({{1, 3}, {0, 1}}),
                                            makeRegion({{0, 2}, {1, 2}}),
                                        },
                               "PERMUTE forward");
        auto backward = prop.backward(graph.getNode(permId), graph,
                                      {makeRegion({{1, 3}, {0, 1}}), makeRegion({{0, 2}, {1, 2}})});
        assertRegionListEquals(backward[0], {
                                                makeRegion({{0, 1}, {1, 3}}),
                                                makeRegion({{1, 2}, {0, 2}}),
                                            },
                               "PERMUTE backward input");
    }
    {
        Graph graph;
        uint32_t a = makeFloatInput(graph, {2, 2});
        uint32_t b = makeFloatInput(graph, {2, 2});
        uint32_t axis = makeIntConst(graph, {0});
        uint32_t concatId = graph.concat({a, b}, axis);
        prop.inferShapeRecursive(concatId, graph);
        auto forward = prop.forward(graph.getNode(concatId), graph,
                                    {{makeRegion({{0, 1}, {0, 2}})}, {makeRegion({{1, 2}, {1, 2}})}, {}});
        assertRegionListEquals(forward, {
                                            makeRegion({{0, 1}, {0, 2}}),
                                            makeRegion({{3, 4}, {1, 2}}),
                                        },
                               "CONCAT forward");
        auto backward = prop.backward(graph.getNode(concatId), graph,
                                      {makeRegion({{0, 1}, {0, 2}}), makeRegion({{3, 4}, {1, 2}})});
        assertRegionListEquals(backward[0], {makeRegion({{0, 1}, {0, 2}})}, "CONCAT backward left");
        assertRegionListEquals(backward[1], {makeRegion({{1, 2}, {1, 2}})}, "CONCAT backward right");
    }
    {
        Graph graph;
        uint32_t x = makeFloatInput(graph, {2, 2});
        uint32_t repeats = makeIntConst(graph, {3});
        uint32_t axis = makeIntConst(graph, {0});
        uint32_t repeatId = graph.repeat(x, repeats, axis);
        prop.inferShapeRecursive(repeatId, graph);
        auto forward = prop.forward(graph.getNode(repeatId), graph, {{makeRegion({{1, 2}, {0, 2}})}, {}, {}});
        assertRegionListEquals(forward, {makeRegion({{3, 6}, {0, 2}})}, "REPEAT forward");
        auto backward = prop.backward(graph.getNode(repeatId), graph, {makeRegion({{3, 6}, {0, 2}})});
        assertRegionListEquals(backward[0], {makeRegion({{1, 2}, {0, 2}})}, "REPEAT backward input");
    }
    {
        Graph graph;
        uint32_t x = makeFloatInput(graph, {8});
        uint32_t starts = makeIntConst(graph, {2});
        uint32_t ends = makeIntConst(graph, {6});
        uint32_t steps = makeIntConst(graph, {1});
        uint32_t sliceId = graph.slice(x, starts, ends, steps);
        prop.inferShapeRecursive(sliceId, graph);
        auto forward = prop.forward(graph.getNode(sliceId), graph, {{makeRegion({{2, 3}}), makeRegion({{5, 6}})}, {}, {}, {}});
        assertRegionListEquals(forward, {makeRegion({{0, 1}}), makeRegion({{3, 4}})}, "SLICE forward");
        auto backward = prop.backward(graph.getNode(sliceId), graph, {makeRegion({{0, 1}}), makeRegion({{3, 4}})});
        assertRegionListEquals(backward[0], {makeRegion({{2, 3}}), makeRegion({{5, 6}})}, "SLICE backward input");
    }
    {
        Graph graph;
        uint32_t target = makeFloatInput(graph, {8});
        uint32_t updates = makeFloatInput(graph, {4});
        uint32_t starts = makeIntConst(graph, {2});
        uint32_t ends = makeIntConst(graph, {6});
        uint32_t steps = makeIntConst(graph, {1});
        uint32_t scatterId = graph.scatter(target, updates, starts, ends, steps);
        prop.inferShapeRecursive(scatterId, graph);
        auto forward = prop.forward(graph.getNode(scatterId), graph, {{makeRegion({{0, 2}})}, {makeRegion({{1, 3}})}, {}, {}, {}});
        assertRegionListEquals(forward, {makeRegion({{0, 2}}), makeRegion({{3, 5}})}, "SCATTER forward");
        auto backward = prop.backward(graph.getNode(scatterId), graph, {makeRegion({{3, 5}})});
        assertRegionListEquals(backward[0], {makeRegion({{3, 5}})}, "SCATTER backward target");
        assertRegionListEquals(backward[1], {makeRegion({{1, 3}})}, "SCATTER backward updates");
    }
    {
        Graph graph;
        uint32_t data = makeFloatInput(graph, {4, 3});
        uint32_t idx = makeIntConst(graph, {2});
        uint32_t gatherId = graph.gather(data, idx);
        prop.inferShapeRecursive(gatherId, graph);
        auto forward = prop.forward(graph.getNode(gatherId), graph, {{makeRegion({{1, 3}, {0, 3}})}, {makeRegion({{0, 2}})}});
        assertRegionListEquals(forward, {makeRegion({{0, 2}, {0, 3}})}, "GATHER forward");
        auto backward = prop.backward(graph.getNode(gatherId), graph, {makeRegion({{0, 2}, {1, 3}})});
        assertRegionListEquals(backward[0], {makeRegion({{2, 3}, {1, 3}})}, "GATHER backward data");
        assertRegionListEquals(backward[1], {makeRegion({{0, 2}})}, "GATHER backward idx");
    }
    {
        Graph graph;
        uint32_t data = makeFloatInput(graph, {4, 3});
        uint32_t idxSrc = makeIntConst(graph, {1, 3, 0, 2});
        uint32_t sliceStart = makeIntConst(graph, {0});
        uint32_t sliceEnd = makeIntConst(graph, {2});
        uint32_t sliceStep = makeIntConst(graph, {1});
        uint32_t idx = graph.slice(idxSrc, sliceStart, sliceEnd, sliceStep);
        prop.inferShapeRecursive(idx, graph);
        uint32_t gatherId = graph.gather(data, idx);
        prop.inferShapeRecursive(gatherId, graph);
        auto backward = prop.backward(graph.getNode(gatherId), graph, {makeRegion({{0, 2}, {0, 3}})});
        assertRegionListEquals(backward[0], {makeRegion({{1, 2}, {0, 3}}), makeRegion({{3, 4}, {0, 3}})}, "GATHER backward sliced indices data");
        assertRegionListEquals(backward[1], {makeRegion({{0, 2}})}, "GATHER backward sliced indices idx");
    }
}

// Topological sort of graph nodes from given roots
std::vector<uint32_t> topologicalSort(const std::vector<uint32_t> &roots, const Graph &graph)
{
    std::vector<uint32_t> order;
    std::unordered_set<uint32_t> visited;
    auto visit = [&](auto &self, uint32_t node) -> void
    {
        if (visited.count(node))
            return;
        visited.insert(node);
        if (graph.hasNode(node))
        {
            for (uint32_t pid : graph.getNode(node).parentIds)
            {
                self(self, pid);
            }
        }
        order.push_back(node);
    };
    for (uint32_t root : roots)
    {
        visit(visit, root);
    }
    return order;
}

// Create a TensorView for a node
TensorView makeView(const TensorNode &node)
{
    TensorView view;
    view.setShape(node.getShape());
    view.baseOffset = 0;
    view.dtype = node.dtype;
    if (!node.strides.empty())
        view.strides = node.strides;
    else
        view.strides = calcContiguousStrides(node.getShape());
    return view;
}

size_t getRequiredBufferSize(const TensorView &view)
{
    if (view.getShape().empty())
        return 1;
    size_t maxOffset = 0;
    for (size_t i = 0; i < view.getShape().size(); ++i)
    {
        if (view.getShape()[i] > 0)
        {
            maxOffset += (view.getShape()[i] - 1) * view.strides[i];
        }
    }
    return maxOffset + 1;
}

// ============================================================
// Reference Graph Executor (Manual Traversal)
// ============================================================
std::vector<float> executeReferenceGraph(
    uint32_t rootId,
    Graph &graph,
    const std::unordered_map<uint32_t, std::vector<uint8_t>> &rawInputData,
    bool forceNonContiguous = false)
{
    std::vector<uint32_t> topo = topologicalSort({rootId}, graph);
    ShapePropagator prop;
    for (uint32_t nodeId : topo)
    {
        prop.inferShape(nodeId, graph);
    }

    std::unordered_map<uint32_t, std::vector<uint8_t>> results;
    std::unordered_map<uint32_t, TensorView> views;
    for (uint32_t nodeId : topo)
    {
        const TensorNode &node = graph.getNode(nodeId);
        uint64_t elemSize = getDTypeSize(node.dtype);

        if (node.opType == OpType::INPUT)
        {
            TensorView view = makeView(node);
            if (forceNonContiguous)
            {
                for (auto &s : view.strides)
                    s *= 2;
            }
            views[nodeId] = view;
            size_t bufElements = getRequiredBufferSize(view);
            results[nodeId].resize(bufElements * elemSize, 0);
            std::vector<uint8_t> rawBytes;
            auto it = rawInputData.find(nodeId);
            if (it != rawInputData.end())
            {
                rawBytes = it->second;
            }
            else if (graph.constantStaging.count(nodeId))
            {
                rawBytes = graph.constantStaging.at(nodeId);
            }
            else
            {
                Error::throw_err("[executeReferenceGraph] input node value not found in constantStaging or inputData");
            }

            uint64_t numElements = countElements(view);
            for (size_t i = 0; i < numElements; ++i)
            {
                uint64_t idx = getStridedIndex(i, view.getShape(), view.strides);
                std::memcpy(results[nodeId].data() + idx * elemSize,
                            rawBytes.data() + i * elemSize,
                            elemSize);
            }
            continue;
        }

        std::vector<const void *> inputPtrs;
        std::vector<TensorView> inputViews;
        std::vector<TensorNode> inputNodes;
        for (uint32_t pid : node.parentIds)
        {
            auto resultIt = results.find(pid);
            if (resultIt == results.end())
            {
                Error::throw_err("Parent node " + std::to_string(pid) + " not found in results");
            }
            inputPtrs.push_back(resultIt->second.data());
            inputViews.push_back(views[pid]);
            TensorNode inNode = graph.getNode(pid);
            inNode.strides = views[pid].strides;
            inNode.viewOffset = views[pid].baseOffset / getDTypeSize(inNode.dtype);
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
            node.opType, node.opName, node.backend,
            inputNodes, outNodeNC, {}, true);
        TensorView chosenOutView;
        uint64_t chosenKernelUid = 0;
        if (forceNonContiguous && !refs_nc.empty())
        {
            chosenOutView = outViewNonContig;
            chosenKernelUid = refs_nc.front();
        }
        else
        {
            TensorNode outNodeC = node;
            auto refs_c = KernelRegistry::get().findMatchingKernels(
                node.opType, node.opName, node.backend,
                inputNodes, outNodeC, {}, true);
            if (refs_c.empty())
            {
                Error::throw_err("No reference kernel found for node " + std::to_string(nodeId) +
                                 " op=" + toString(node.opType) +
                                 (node.opType == OpType::FUSED ? " (" + node.opName + ")" : ""));
            }
            chosenOutView = outViewContig;
            chosenKernelUid = refs_c.front();
        }

        const KernelEntry &kernel = KernelRegistry::get().getKernel(chosenKernelUid);
        if (kernel.isView)
        {
            TensorNode dummyOutNode = node;
            kernel.inferView(dummyOutNode, inputNodes, graph);
            uint32_t parentId = node.parentIds[0];
            results[nodeId] = results[parentId];
            chosenOutView.strides = dummyOutNode.strides;
            chosenOutView.baseOffset = dummyOutNode.viewOffset * elemSize;
            views[nodeId] = chosenOutView;
            continue;
        }

        views[nodeId] = chosenOutView;
        size_t bufElements = getRequiredBufferSize(chosenOutView);
        results[nodeId].resize(bufElements * elemSize, 0);
        std::vector<void *> outputPtrs = {results[nodeId].data()};
        std::vector<TensorView> outputViews = {chosenOutView};

        if (kernel.run)
        {
            kernel.run(inputPtrs, outputPtrs, inputViews, outputViews);
        }
    }

    uint64_t numRootElems = countElements(graph.getNode(rootId));
    std::vector<float> finalOut(numRootElems, 0.0f);
    TensorView rootView = views[rootId];
    for (size_t i = 0; i < numRootElems; ++i)
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

std::vector<float> flattenOutput(const void *ptr, const std::vector<uint32_t> &shape,
                                 const std::vector<uint64_t> &strides, DType dtype)
{
    std::vector<float> result;
    uint64_t elems = countElements(shape);
    result.reserve(elems);
    if (dtype == DType::FLOAT32)
    {
        const float *src = static_cast<const float *>(ptr);
        for (uint64_t i = 0; i < elems; ++i)
            result.push_back(src[getStridedIndex(i, shape, strides)]);
    }
    else if (dtype == DType::INT32)
    {
        const int32_t *src = static_cast<const int32_t *>(ptr);
        for (uint64_t i = 0; i < elems; ++i)
            result.push_back(static_cast<float>(src[getStridedIndex(i, shape, strides)]));
    }
    else if (dtype == DType::BF16)
    {
        const uint16_t *src = static_cast<const uint16_t *>(ptr);
        for (uint64_t i = 0; i < elems; ++i)
        {
            uint32_t bits = static_cast<uint32_t>(src[getStridedIndex(i, shape, strides)]) << 16;
            float val;
            std::memcpy(&val, &bits, 4);
            result.push_back(val);
        }
    }
    else
    {
        result.resize(elems, 0.0f);
    }
    return result;
}

// ============================================================
// Fused Kernel Direct Execution
// ============================================================
std::vector<float> executeFusedKernel(
    const KernelEntry &kernel,
    const std::vector<std::vector<uint8_t>> &inputData,
    const std::vector<uint32_t> &inputIds,
    const std::vector<uint32_t> &outShape,
    const std::vector<uint64_t> &outStrides,
    DType outDType,
    const Graph &graph)
{
    if (inputData.size() != kernel.numInputs)
    {
        Error::throw_err("Fused kernel " + kernel.opName + " expects " +
                         std::to_string(kernel.numInputs) + " inputs, got " +
                         std::to_string(inputData.size()));
    }

    std::vector<const void *> inputPtrs;
    std::vector<TensorView> inputViews;
    for (size_t i = 0; i < kernel.numInputs; ++i)
    {
        inputPtrs.push_back(inputData[i].data());
        TensorView view;
        const TensorNode &node = graph.getNode(inputIds[i]);
        view.setShape(node.getShape());
        view.strides = node.strides.empty() ? calcContiguousStrides(node.getShape()) : node.strides;
        view.baseOffset = node.viewOffset * getDTypeSize(node.dtype);
        view.dtype = node.dtype;
        inputViews.push_back(view);
    }

    TensorView outView;
    outView.setShape(outShape);
    outView.strides = outStrides.empty() ? calcContiguousStrides(outShape) : outStrides;
    outView.baseOffset = 0;
    outView.dtype = outDType;
    std::vector<TensorView> outputViews = {outView};

    size_t outBufElements = getRequiredBufferSize(outView);
    std::vector<uint8_t> outputData;

    if (kernel.inplace && kernel.numInputs > 0 && !kernel.inferView)
    {
        outputData = inputData[0];
        if (outputData.size() < outBufElements * getDTypeSize(outDType))
        {
            outputData.resize(outBufElements * getDTypeSize(outDType));
        }
    }
    else
    {
        outputData.resize(outBufElements * getDTypeSize(outDType), 0);
    }

    if (kernel.inplace && kernel.numInputs > 0 && kernel.inferView)
    {
        std::vector<TensorNode> dummyInputs(kernel.numInputs);
        for (size_t i = 0; i < kernel.numInputs; ++i)
        {
            dummyInputs[i].id = inputIds[i];
            dummyInputs[i].setShape(inputViews[i].getShape());
            dummyInputs[i].strides = inputViews[i].strides;
            dummyInputs[i].dtype = inputViews[i].dtype;
            dummyInputs[i].viewOffset = inputViews[i].baseOffset / getDTypeSize(inputViews[i].dtype);
        }
        TensorNode dummyOutput;
        dummyOutput.setShape(outShape);
        dummyOutput.dtype = outDType;
        kernel.inferView(dummyOutput, dummyInputs, graph);

        outView.strides = dummyOutput.strides;
        outView.baseOffset = dummyOutput.viewOffset * getDTypeSize(outDType);

        return flattenOutput(inputData[0].data() + outView.baseOffset, outView.getShape(), outView.strides, outView.dtype);
    }

#ifdef USE_CUDA
    bool anyCuda = false;
    for (Backend b : kernel.backends)
    {
        if (b == Backend::CUDA)
            anyCuda = true;
    }
    for (const auto &bb : kernel.inputBackends)
    {
        for (Backend b : bb)
        {
            if (b == Backend::CUDA)
                anyCuda = true;
        }
    }

    if (anyCuda)
    {
        bool outputNeedsCuda = false;
        for (Backend b : kernel.backends)
        {
            if (b == Backend::CUDA)
                outputNeedsCuda = true;
        }

        std::vector<void *> d_inputs;
        std::vector<void *> d_outputs;
        std::vector<bool> inputOnDevice(kernel.numInputs, false);
        for (size_t i = 0; i < inputData.size(); ++i)
        {
            bool inputNeedsCuda = false;
            size_t ruleIdx = i;
            if (kernel.isVariadic)
            {
                ruleIdx = (i == inputData.size() - 1) ? kernel.inputBackends.size() - 1 : 0;
            }

            if (ruleIdx < kernel.inputBackends.size())
            {
                for (Backend b : kernel.inputBackends[ruleIdx])
                {
                    if (b == Backend::CUDA)
                    {
                        inputNeedsCuda = true;
                        break;
                    }
                }
            }
            else
            {
                inputNeedsCuda = outputNeedsCuda;
            }

            if (kernel.opType == OpType::COPY_TO)
            {
                inputNeedsCuda = !outputNeedsCuda;
            }

            if (inputNeedsCuda)
            {
                void *d_ptr = nullptr;
                cudaMalloc(&d_ptr, inputData[i].size());
                cudaMemcpy(d_ptr, inputData[i].data(), inputData[i].size(), cudaMemcpyHostToDevice);
                d_inputs.push_back(d_ptr);
                inputOnDevice[i] = true;
            }
            else
            {
                d_inputs.push_back(const_cast<uint8_t *>(inputData[i].data()));
                inputOnDevice[i] = false;
            }
        }

        void *d_out = nullptr;
        uint64_t outBytes = outputData.size();
        if (outputNeedsCuda)
        {
            cudaMalloc(&d_out, outBytes);
            if (kernel.inplace && !kernel.inferView)
            {
                cudaMemcpy(d_out, outputData.data(), outBytes, cudaMemcpyHostToDevice);
            }
            d_outputs.push_back(d_out);
        }
        else
        {
            d_outputs.push_back(outputData.data());
        }

        std::vector<const void *> d_input_ptrs;
        for (void *p : d_inputs)
            d_input_ptrs.push_back(p);

        if (kernel.run)
            kernel.run(d_input_ptrs, d_outputs, inputViews, outputViews);

        cudaDeviceSynchronize();

        if (outputNeedsCuda)
        {
            cudaMemcpy(outputData.data(), d_out, outBytes, cudaMemcpyDeviceToHost);
            cudaFree(d_out);
        }

        for (size_t i = 0; i < d_inputs.size(); ++i)
        {
            if (inputOnDevice[i])
                cudaFree(d_inputs[i]);
        }
    }
    else
    {
        std::vector<void *> outputPtrs = {outputData.data()};
        if (kernel.run)
            kernel.run(inputPtrs, outputPtrs, inputViews, outputViews);
    }
#else
    std::vector<void *> outputPtrs = {outputData.data()};
    if (kernel.run)
        kernel.run(inputPtrs, outputPtrs, inputViews, outputViews);
#endif

    return flattenOutput(outputData.data() + outView.baseOffset, outView.getShape(), outView.strides, outView.dtype);
}

// ============================================================
// Test Setup Helpers
// ============================================================
struct TestInputs
{
    std::vector<uint32_t> inputIds;
    std::unordered_map<uint32_t, std::vector<uint8_t>> rawInputData;
    std::vector<std::vector<uint8_t>> rawData;
};

TestInputs createTestInputs(Graph &graph, const KernelEntry &kernel)
{
    TestInputs result;
    result.rawData.resize(kernel.numInputs);
    result.inputIds.resize(kernel.numInputs);
    for (size_t i = 0; i < kernel.numInputs; ++i)
    {
        uint32_t id = UINT32_MAX;
        DType dtype = kernel.dtypes[i];
        uint64_t elements = countElements(kernel.dummyShapes[i]);
        uint64_t sizeBytes = elements * getDTypeSize(dtype);
        bool isConstantParam = (kernel.opName == "Repeat_Inplace" && i > 0) ||
                               (kernel.opName == "Reshape_Inplace" && i == 1) ||
                               (kernel.opName == "Permute_CUDA_Contiguous" && i == 1) ||
                               (kernel.opName.find("SCATTER") != std::string::npos && i >= 2) ||
                               (kernel.opName.find("PartialDot") != std::string::npos && i >= 3);
        if (isConstantParam)
        {
            std::vector<int32_t> constData(elements);
            if (kernel.opName == "Repeat_Inplace")
            {
                if (i == 1)
                    constData[0] = 2;
                if (i == 2)
                    constData[0] = 0;
            }
            else if (kernel.opName == "Reshape_Inplace")
            {
                for (size_t j = 0; j < elements; ++j)
                    constData[j] = static_cast<int32_t>(kernel.dummyShapes[0][j]);
            }
            else if (kernel.opName == "Permute_CUDA_Contiguous")
            {
                size_t rank = kernel.dummyShapes[0].size();
                for (size_t j = 0; j < elements; ++j)
                {
                    if (rank == 2)
                        constData[j] = (j == 0 ? 1 : 0);
                    else
                        constData[j] = (int32_t)j;
                }
            }
            else if (kernel.opName.find("SCATTER") != std::string::npos || kernel.opName.find("PartialDot") != std::string::npos)
            {
                size_t startsIdx = kernel.opName.find("PartialDot") != std::string::npos ? 3 : 2;
                size_t endsIdx = startsIdx + 1;
                size_t stepsIdx = startsIdx + 2;

                if (i == startsIdx)
                {
                    for (size_t j = 0; j < elements; ++j)
                        constData[j] = 0;
                }
                else if (i == endsIdx)
                {
                    for (size_t j = 0; j < elements; ++j)
                    {
                        if (j < kernel.dummyShapes[0].size())
                            constData[j] = kernel.dummyShapes[0][j];
                        else
                            constData[j] = 1;
                    }
                }
                else if (i == stepsIdx)
                {
                    for (size_t j = 0; j < elements; ++j)
                        constData[j] = 1;
                }
            }
            id = graph.constant(kernel.dummyShapes[i], constData.data(), dtype);
            result.rawData[i].resize(sizeBytes);
            std::memcpy(result.rawData[i].data(), constData.data(), sizeBytes);
        }
        else
        {
            id = graph.input(kernel.dummyShapes[i], dtype, {}, StorageType::PERSISTENT);
            result.rawData[i].resize(sizeBytes);
            fillRandom(result.rawData[i].data(), elements, dtype);
        }
        result.rawInputData[id] = result.rawData[i];
        result.inputIds[i] = id;
    }
    return result;
}

// ============================================================
// JSONL / Record Testing Helpers
// ============================================================

bool testKernelWithRecord(const KernelEntry &kernel, const Record &rec)
{
    try
    {
        if (rec.inputShapes.size() != kernel.numInputs && !kernel.isVariadic)
            return true; // Skip mismatched variadic/arity records

        // Build dummy nodes for validation against centralized matching logic
        std::vector<TensorNode> dummyInputs(rec.inputShapes.size());
        for (size_t idx = 0; idx < rec.inputShapes.size(); ++idx)
        {
            dummyInputs[idx].setShape(rec.inputShapes[idx]);
            dummyInputs[idx].strides = rec.inputStrides[idx];
            dummyInputs[idx].dtype = rec.inputDTypes[idx];

            Backend b = Backend::CPU;
            if (!rec.inputBackends.empty() && idx < rec.inputBackends.size() && !rec.inputBackends[idx].empty())
                b = rec.inputBackends[idx][0];
            dummyInputs[idx].backend = b;
        }

        TensorNode dummyOutput;
        std::vector<uint32_t> outShape;
        std::vector<uint64_t> outStrides;
        DType outDType = DType::FLOAT32;

        if (!rec.outputShapes.empty())
        {
            outShape = rec.outputShapes[0];
            outStrides = rec.outputStrides[0];
            outDType = rec.outputDTypes[0];

            dummyOutput.setShape(outShape);
            dummyOutput.strides = outStrides;
            dummyOutput.dtype = outDType;
            dummyOutput.backend = rec.backends.empty() ? Backend::CPU : rec.backends[0];
        }
        else
        {
            outShape = rec.inputShapes.empty() ? std::vector<uint32_t>{} : rec.inputShapes[0];
            outStrides = rec.inputStrides.empty() ? std::vector<uint64_t>{} : rec.inputStrides[0];
            outDType = rec.inputDTypes.empty() ? DType::FLOAT32 : rec.inputDTypes[0];

            dummyOutput.setShape(outShape);
            dummyOutput.strides = outStrides;
            dummyOutput.dtype = outDType;
            dummyOutput.backend = rec.backends.empty() ? Backend::CPU : rec.backends[0];
        }

        if (!kernel.matches(dummyInputs, dummyOutput))
            return true; // Skip invalid records

        Graph graph;
        std::vector<std::vector<uint8_t>> rawData(rec.inputShapes.size());
        std::unordered_map<uint32_t, std::vector<uint8_t>> rawInputData;
        std::vector<uint32_t> inputIds(rec.inputShapes.size());

        for (size_t i = 0; i < rec.inputShapes.size(); ++i)
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
            if (i < rec.inputConstants.size() && !rec.inputConstants[i].empty() && rec.inputConstants[i].size() == elements * dtypeSize)
            {
                isConstant = true;
                std::memcpy(contiguousData.data(), rec.inputConstants[i].data(), rec.inputConstants[i].size());
            }
            else
            {
                fillRandom(contiguousData.data(), elements, rec.inputDTypes[i]);
            }

            // Scatter contiguousData physically into rawData[i] using strides
            for (size_t k = 0; k < elements; ++k)
            {
                uint64_t idx = getStridedIndex(k, view.getShape(), view.strides);
                std::memcpy(rawData[i].data() + idx * dtypeSize,
                            contiguousData.data() + k * dtypeSize,
                            dtypeSize);
            }

            if (isConstant)
            {
                inputIds[i] = graph.constant(rec.inputShapes[i], contiguousData.data(), rec.inputDTypes[i]);
            }
            else
            {
                inputIds[i] = graph.input(rec.inputShapes[i], rec.inputDTypes[i], {}, StorageType::PERSISTENT);
            }

            graph.getNode(inputIds[i]).strides = rec.inputStrides[i];
            rawInputData[inputIds[i]] = contiguousData;
        }

        uint32_t rootId = kernel.refFactory(inputIds, graph);

        // Reference graph will handle the continuous mapping identically internally
        std::vector<float> refOutput = executeReferenceGraph(rootId, graph, rawInputData, false);
        // Target fused execution resolves dynamically spread arrays
        std::vector<float> tgtOutput = executeFusedKernel(kernel, rawData, inputIds, outShape, outStrides, outDType, graph);

        if (refOutput.size() != tgtOutput.size())
        {
            std::cout << "\n[Record Test Error] Output size mismatch: ref=" << refOutput.size() << " tgt=" << tgtOutput.size() << " kernel=" << kernel.opName << std::endl;
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

std::unordered_map<uint64_t, std::vector<Record>> loadCallRecords(const std::string &path)
{
    std::unordered_map<uint64_t, std::vector<Record>> records;
    std::ifstream file(path);
    if (!file.is_open())
        return records;
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;
        json j = json::parse(line);
        Record r = j.get<Record>();
        records[r.kernelUid].push_back(std::move(r));
    }
    return records;
}

// ============================================================
// Python Test Loop
// ============================================================
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
    std::sort(testDirs.begin(), testDirs.end(), [](const std::string &a, const std::string &b)
              {
                std::string na = std::filesystem::path(a).filename().string();
                std::string nb = std::filesystem::path(b).filename().string();
                try { return std::stoi(na) < std::stoi(nb); } catch (...) { return a < b; } });
    for (const std::string &testDir : testDirs)
    {
        total++;
        std::string infoPath = testDir + "/info.json";
        std::string dataPath = testDir + "/data.safetensors";
        std::ifstream infoFile(infoPath);
        if (!infoFile.is_open())
            continue;
        json info;
        infoFile >> info;
        OpType opType = info["optype"].get<OpType>();
        SafetensorsLoader loader(dataPath);
        std::vector<std::vector<uint8_t>> inputData;
        std::vector<TensorView> inViews;
        std::vector<TensorNode> dummyInputNodes;
        std::vector<const void *> inPtrs;
        Graph dummyGraph;
        int i = 0;
        for (const auto &inpJson : info["inputs"])
        {
            std::vector<uint32_t> shape = inpJson["shape"].get<std::vector<uint32_t>>();
            std::vector<uint64_t> strides = inpJson["strides"].get<std::vector<uint64_t>>();
            DType dtype = inpJson["dtype"].get<DType>();
            std::string tensorName = "input." + std::to_string(i);
            uint64_t sizeBytes = countElements(shape) * getDTypeSize(dtype);
            std::vector<uint8_t> data(sizeBytes);
            loader.loadTensor(tensorName, data.data(), sizeBytes);
            inputData.push_back(std::move(data));
            TensorView view;
            view.baseOffset = 0;
            view.setShape(shape);
            view.strides = strides;
            view.dtype = dtype;
            inViews.push_back(view);
            TensorNode &node = dummyGraph.allocateNode(OpType::INPUT, "", dtype, {}, shape, strides, Backend::CPU, StorageType::PERSISTENT);
            dummyInputNodes.push_back(node);
            if (dtype == DType::INT32)
                dummyGraph.constantStaging[node.id] = inputData.back();
            i++;
        }
        for (auto &vec : inputData)
            inPtrs.push_back(vec.data());

        json outJson = info["output"];
        std::vector<uint32_t> outShape = outJson["shape"].get<std::vector<uint32_t>>();
        std::vector<uint64_t> outStrides = outJson["strides"].get<std::vector<uint64_t>>();
        DType outDType = outJson["dtype"].get<DType>();
        uint64_t outSizeBytes = countElements(outShape) * getDTypeSize(outDType);
        std::vector<uint8_t> expectedData(outSizeBytes);
        loader.loadTensor("output", expectedData.data(), outSizeBytes);
        std::vector<uint8_t> actualData(outSizeBytes);
        std::vector<void *> outPtrs = {actualData.data()};
        TensorView outView;
        outView.baseOffset = 0;
        outView.setShape(outShape);
        outView.strides = outStrides;
        outView.dtype = outDType;
        std::vector<TensorView> outViews = {outView};
        TensorNode outNode;
        outNode.id = i;
        outNode.dtype = outDType;
        outNode.setShape(outShape);
        outNode.strides = outStrides;
        outNode.backend = Backend::CPU;

        std::cout << "Testing Python Ref " << testDir << " [" << toString(opType) << "] ... " << std::flush;
        std::vector<uint64_t> matches = KernelRegistry::get().findMatchingKernels(
            opType, "", Backend::CPU, dummyInputNodes, outNode, {}, true);
        if (matches.empty())
        {
            Error::throw_err("[runPythonTests] FAILED (No reference kernel found)");
        }
        if (matches.size() > 1)
        {
            Error::throw_err("[runPythonTests] Expected 1 kernel match, got " + std::to_string(matches.size()));
        }
        const KernelEntry &kernel = KernelRegistry::get().getKernel(matches.front());
        if (kernel.isView)
        {
            TensorNode dummyOutNode = outNode;
            for (size_t k = 0; k < dummyInputNodes.size(); ++k)
            {
                dummyInputNodes[k].strides = inViews[k].strides;
                dummyInputNodes[k].viewOffset = inViews[k].baseOffset / getDTypeSize(dummyInputNodes[k].dtype);
            }
            kernel.inferView(dummyOutNode, dummyInputNodes, dummyGraph);
            size_t elements = countElements(outShape);
            if (outDType == DType::FLOAT32)
            {
                const float *src = reinterpret_cast<const float *>(inputData[0].data());
                float *dst = reinterpret_cast<float *>(actualData.data());
                for (size_t k = 0; k < elements; ++k)
                {
                    uint64_t srcIdx = dummyOutNode.viewOffset + getStridedIndex(k, dummyOutNode.getShape(), dummyOutNode.strides);
                    dst[k] = src[srcIdx];
                }
            }
            else if (outDType == DType::INT32)
            {
                const int32_t *src = reinterpret_cast<const int32_t *>(inputData[0].data());
                int32_t *dst = reinterpret_cast<int32_t *>(actualData.data());
                for (size_t k = 0; k < elements; ++k)
                {
                    uint64_t srcIdx = dummyOutNode.viewOffset + getStridedIndex(k, dummyOutNode.getShape(), dummyOutNode.strides);
                    dst[k] = src[srcIdx];
                }
            }
        }
        else if (kernel.run)
        {
            kernel.run(inPtrs, outPtrs, inViews, outViews);
        }

        bool ok = false;
        if (outDType == DType::FLOAT32)
        {
            ok = compareOutputs((const float *)expectedData.data(), (const float *)actualData.data(), countElements(outShape));
        }
        else if (outDType == DType::INT32)
        {
            ok = compareOutputs((const int32_t *)expectedData.data(), (const int32_t *)actualData.data(), countElements(outShape));
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
    std::cout << "----------------------\n"
              << std::endl;
}

// ============================================================
// Main Test Loop
// ============================================================
int main()
{
    runPythonTests();
    runRegionMergeTests();
    runShapePropagationTests();
    std::cout << "Running Non-Reference Kernel Tests..." << std::endl;
    int passed = 0;
    int total = 0;
    int skipped = 0;

    // Load benchmark records if available
    auto recordsByUid = loadCallRecords("benchmarks/calls.jsonl");

    const auto &kernels = KernelRegistry::get().getAllKernels();
    for (const auto &kernel : kernels)
    {
        if (kernel.isReference)
            continue;
        if (!kernel.refFactory)
        {
            std::cout << "Skipping " << kernel.opName << " (no refFactory)" << std::endl;
            skipped++;
            continue;
        }
        if (kernel.dummyShapes.size() != kernel.numInputs)
        {
            std::cout << "Skipping " << kernel.opName << " (dummy shapes mismatch)" << std::endl;
            skipped++;
            continue;
        }

        total++;
        std::cout << "Testing " << kernel.opName << " ... " << std::flush;

        // 1. Dummy Shapes Test
        Graph refGraph;
        TestInputs refInputs = createTestInputs(refGraph, kernel);
        uint32_t rootId = kernel.refFactory(refInputs.inputIds, refGraph);
        std::vector<float> refOutput = executeReferenceGraph(rootId, refGraph, refInputs.rawInputData, false);
        size_t elements = refOutput.size();

        const TensorNode &rootNode = refGraph.getNode(rootId);
        std::vector<float> fusedOutput = executeFusedKernel(kernel, refInputs.rawData, refInputs.inputIds, rootNode.getShape(), rootNode.strides, rootNode.dtype, refGraph);

        bool dummyOk = false;
        if (fusedOutput.size() == elements)
            dummyOk = compareOutputs(refOutput.data(), fusedOutput.data(), elements);

        // 2. Record-Based Tests
        bool recordOk = true;
        auto it = recordsByUid.find(kernel.uid);
        if (it != recordsByUid.end())
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
        else
        {
            std::cout << " (no records) ";
        }

        if (dummyOk && recordOk)
        {
            passed++;
            if (it == recordsByUid.end())
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