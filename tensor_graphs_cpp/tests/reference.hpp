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
#include "core/shape_propagator.hpp"
#include "generated/kernels_all.gen.hpp"

#include "tests/dispatch_domination.hpp"
#include "tests/fused.hpp"
#include "tests/input_hashcons.hpp"
#include "tests/mem_cap_prune.hpp"
#include "tests/region_merge.hpp"
#include "tests/shape_propagation.hpp"

void runRefTests(std::string testDir = "tensor_graphs_cpp/tests/data")
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
        bool ignore_in_ms = (opType != OpType::COPY_TO);
        std::vector<KernelId> matches = KernelRegistry::get().findMatchingKernels(
            opType, "", dummyInputNodes, outNode, true, MemSpace{1, HandleType::CPP}, {}, {Engine{0, EngineType::CPU}},
            false, ignore_in_ms, false, true);
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