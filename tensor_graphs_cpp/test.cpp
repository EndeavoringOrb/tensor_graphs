#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <unordered_set>

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/shapes.hpp"

#include "generated/kernels_all.gen.hpp"

// ============================================================
// Helper Functions
// ============================================================

void fillRandom(void *ptr, size_t elements, DType dtype)
{
    static std::mt19937 gen(42);

    switch (dtype)
    {
    case DType::FLOAT32: {
        float *fptr = static_cast<float *>(ptr);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < elements; ++i)
        {
            fptr[i] = dist(gen);
        }
        break;
    }
    case DType::INT32: {
        int32_t *iptr = static_cast<int32_t *>(ptr);
        std::uniform_int_distribution<int32_t> dist(1, 10);
        for (size_t i = 0; i < elements; ++i)
        {
            iptr[i] = dist(gen);
        }
        break;
    }
    case DType::BOOL: {
        bool *bptr = static_cast<bool *>(ptr);
        std::uniform_int_distribution<int> dist(0, 1);
        for (size_t i = 0; i < elements; ++i)
        {
            bptr[i] = dist(gen) != 0;
        }
        break;
    }
    default:
        float *fptr = static_cast<float *>(ptr);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < elements; ++i)
        {
            fptr[i] = dist(gen);
        }
        break;
    }
}

bool compareOutputs(const float *a, const float *b, size_t elements, float eps = 1e-4f)
{
    for (size_t i = 0; i < elements; ++i)
    {
        if (std::abs(a[i] - b[i]) > eps)
        {
            std::cout << "\nMismatch at index " << i << ": " << a[i] << " != " << b[i] << std::endl;
            return false;
        }
    }
    return true;
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
        if (node < graph.nodes.size())
        {
            for (uint32_t pid : graph.nodes[node].parentIds)
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

// Create a TensorView for a node given pre-allocated data size
TensorView makeView(const TensorNode &node, size_t numElements)
{
    TensorView view;
    view.shape = node.shape;
    view.strides = TensorView::calcContiguousStrides(node.shape);
    view.baseOffset = 0;
    view.dtype = node.dtype;
    return view;
}

// ============================================================
// Reference Graph Executor (Manual Traversal)
// ============================================================

/**
 * Executes a reference graph by manually traversing nodes in topological order
 * and calling reference kernels directly. No Session involvement.
 */
std::vector<float> executeReferenceGraph(
    uint32_t rootId,
    Graph &graph,
    const std::unordered_map<uint32_t, std::vector<float>> &inputData)
{
    // 1. Topological sort from root
    std::vector<uint32_t> topo = topologicalSort({rootId}, graph);

    // 2. Run shape inference on all nodes
    ShapePropagator prop;
    for (uint32_t nodeId : topo)
    {
        prop.inferShape(nodeId, graph);
    }

    // 3. Execute each node in order, storing results
    std::unordered_map<uint32_t, std::vector<float>> results;

    for (uint32_t nodeId : topo)
    {
        const TensorNode &node = graph.nodes[nodeId];

        // INPUT nodes: copy from input data
        if (node.opType == OpType::INPUT)
        {
            auto it = inputData.find(nodeId);
            if (it != inputData.end())
            {
                results[nodeId] = it->second;
            }
            else
            {
                // For constant nodes without explicit input, allocate zero-initialized buffer
                size_t elements = countElements(node.shape);
                results[nodeId].resize(elements, 1.0f);
            }
            continue;
        }

        // Gather input pointers from parent results
        std::vector<const void *> inputPtrs;
        std::vector<TensorView> inputViews;
        for (uint32_t pid : node.parentIds)
        {
            auto resultIt = results.find(pid);
            if (resultIt == results.end())
            {
                Error::throw_err("Parent node " + std::to_string(pid) + " not found in results");
            }
            inputPtrs.push_back(resultIt->second.data());
            inputViews.push_back(makeView(graph.nodes[pid], resultIt->second.size()));
        }

        // Allocate output buffer
        size_t outElements = countElements(node.shape);
        results[nodeId].resize(outElements);

        // Find reference kernel for this operation
        std::vector<TensorNode> inputNodes;
        for (uint32_t pid : node.parentIds)
        {
            inputNodes.push_back(graph.nodes[pid]);
        }

        auto refs = KernelRegistry::get().findMatchingKernels(
            node.opType, node.opName, node.backend,
            inputNodes, node, {}, true);

        if (refs.empty())
        {
            Error::throw_err("No reference kernel found for node " + std::to_string(nodeId) +
                             " op=" + toString(node.opType) +
                             (node.opType == OpType::FUSED ? " (" + node.opName + ")" : ""));
        }

        const KernelEntry &kernel = KernelRegistry::get().getKernel(refs.front());

        std::vector<void *> outputPtrs = {results[nodeId].data()};
        std::vector<TensorView> outputViews = {makeView(node, outElements)};

        kernel.run(inputPtrs, outputPtrs, inputViews, outputViews);
    }

    return results[rootId];
}

// ============================================================
// Fused Kernel Direct Execution
// ============================================================

/**
 * Executes a fused kernel directly by calling kernel.run()
 * with prepared input/output buffers.
 *
 * @param kernel The fused kernel to execute
 * @param inputData Input data buffers
 * @param expectedOutElements Expected number of output elements (from reference graph)
 */
std::vector<float> executeFusedKernel(
    const KernelEntry &kernel,
    const std::vector<std::vector<float>> &inputData,
    size_t expectedOutElements)
{
    if (inputData.size() != kernel.numInputs)
    {
        Error::throw_err("Fused kernel " + kernel.opName + " expects " +
                         std::to_string(kernel.numInputs) + " inputs, got " +
                         std::to_string(inputData.size()));
    }

    // Prepare input pointers and views
    std::vector<const void *> inputPtrs;
    std::vector<TensorView> inputViews;
    for (size_t i = 0; i < kernel.numInputs; ++i)
    {
        inputPtrs.push_back(inputData[i].data());
        TensorView view;
        view.shape = kernel.dummyShapes[i];
        view.strides = TensorView::calcContiguousStrides(kernel.dummyShapes[i]);
        view.baseOffset = 0;
        view.dtype = kernel.dtypes[i];
        inputViews.push_back(view);
    }

    // Allocate output buffer based on expected size from reference graph
    std::vector<float> output(expectedOutElements);

    // For in-place kernels, copy the input that will be modified to the output buffer
    if (kernel.inplace && kernel.numInputs > 0)
    {
        output = inputData[0];
    }

    std::vector<void *> outputPtrs = {output.data()};
    TensorView outView;
    outView.shape = kernel.dummyShapes[0];
    outView.strides = TensorView::calcContiguousStrides(kernel.dummyShapes[0]);
    outView.baseOffset = 0;
    outView.dtype = DType::FLOAT32; // Assume FP32 output for fused kernels
    std::vector<TensorView> outputViews = {outView};

    kernel.run(inputPtrs, outputPtrs, inputViews, outputViews);

    return output;
}

// ============================================================
// Test Setup Helpers
// ============================================================

/**
 * Creates input nodes in the graph and returns their IDs.
 * Also generates random input data.
 */
struct TestInputs
{
    std::vector<uint32_t> inputIds;
    std::unordered_map<uint32_t, std::vector<float>> inputData;
    std::vector<std::vector<float>> rawData; // For fused kernel execution
};

TestInputs createTestInputs(Graph &graph, const KernelEntry &kernel)
{
    TestInputs result;
    result.rawData.resize(kernel.numInputs);
    result.inputIds.resize(kernel.numInputs);

    for (size_t i = 0; i < kernel.numInputs; ++i)
    {
        uint32_t id = graph.allocateId();
        DType dtype = kernel.dtypes[i];
        uint64_t elements = countElements(kernel.dummyShapes[i]);
        uint64_t sizeBytes = elements * getDTypeSize(dtype);

        TensorView view;
        view.shape = kernel.dummyShapes[i];
        view.strides = TensorView::calcContiguousStrides(kernel.dummyShapes[i]);
        view.baseOffset = 0;
        view.dtype = dtype;

        // For Repeat and Reshape operations, constant parameters must be created via graph.constant()
        // so they're available in constantStaging for shape inference
        bool isConstantParam = (kernel.opName == "Repeat_Inplace" && i > 0) ||
                               (kernel.opName == "Reshape_Inplace" && i == 1);

        if (isConstantParam)
        {
            // Create constant node for shape inference
            std::vector<int32_t> constData(elements);
            if (kernel.opName == "Repeat_Inplace")
            {
                if (i == 1) constData[0] = 2;  // repeats = 2
                if (i == 2) constData[0] = 0;  // axis = 0
            }
            else if (kernel.opName == "Reshape_Inplace")
            {
                // For reshape, use the dummy shape values
                for (size_t j = 0; j < elements; ++j)
                    constData[j] = static_cast<int32_t>(kernel.dummyShapes[0][j]);
            }

            id = graph.constant(kernel.dummyShapes[i], constData.data(), dtype);
            result.rawData[i].resize(elements);
            std::memcpy(result.rawData[i].data(), constData.data(), sizeBytes);
        }
        else
        {
            graph.inputWithId(id, kernel.dummyShapes[i], dtype, view, StorageType::PERSISTENT);

            // Generate random data
            result.rawData[i].resize(elements);
            fillRandom(result.rawData[i].data(), elements, dtype);
        }

        // Store in inputData map for reference graph execution
        result.inputData[id] = result.rawData[i];
        result.inputIds[i] = id;
    }

    return result;
}

// ============================================================
// Main Test Loop
// ============================================================

int main()
{
    std::cout << "Running Non-Reference Kernel Tests..." << std::endl;

    int passed = 0;
    int total = 0;
    int skipped = 0;

    const auto &kernels = KernelRegistry::get().getAllKernels();
    for (const auto &kernel : kernels)
    {
        // Skip reference implementations since they are the source of truth
        if (kernel.isReference)
        {
            continue;
        }

        // Skip kernels without refFactory (cannot verify correctness)
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

        try
        {
            // ========== REFERENCE EXECUTION ==========
            // 1. Build reference graph using refFactory
            Graph refGraph;
            TestInputs refInputs = createTestInputs(refGraph, kernel);
            uint32_t rootId = kernel.refFactory(refInputs.inputIds, refGraph);

            // 2. Execute reference graph via manual traversal
            std::vector<float> refOutput = executeReferenceGraph(rootId, refGraph, refInputs.inputData);

            // ========== FUSED KERNEL EXECUTION ==========
            // Execute fused kernel directly
            std::vector<float> fusedOutput = executeFusedKernel(kernel, refInputs.rawData, refOutput.size());

            // ========== COMPARE ==========
            size_t elements = refOutput.size();
            if (fusedOutput.size() != elements)
            {
                std::cout << "FAILED (output size mismatch: " << fusedOutput.size()
                          << " vs " << elements << ")" << std::endl;
                continue;
            }

            if (compareOutputs(refOutput.data(), fusedOutput.data(), elements))
            {
                std::cout << "OK" << std::endl;
                passed++;
            }
            else
            {
                std::cout << "FAILED" << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            std::cout << "EXCEPTION: " << e.what() << std::endl;
        }
    }

    std::cout << "\n----------------------" << std::endl;
    std::cout << "Tests Passed: " << passed << "/" << total << std::endl;
    if (skipped > 0)
    {
        std::cout << "Tests Skipped: " << skipped << std::endl;
    }
    std::cout << "----------------------" << std::endl;

    return 0;
}
