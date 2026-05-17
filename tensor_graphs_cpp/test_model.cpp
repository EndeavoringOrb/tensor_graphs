// --- tensor_graphs_cpp/test_model.cpp ---
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <functional>
#include <filesystem>
#include <sstream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/session.hpp"
#include "core/kernels.hpp"

// Model Definitions
#include "models/gemma-3-270m.hpp"
#include "models/flux-klein-4b.hpp"

#include "generated/kernels_all.gen.hpp"
#include "generated/build_context.gen.hpp"

namespace fs = std::filesystem;

std::string getNextAvailableBinPath(const std::string &folderPath = "sessions")
{
    // Ensure the directory exists
    if (!fs::exists(folderPath))
    {
        fs::create_directories(folderPath);
    }

    int counter = 0;
    while (true)
    {
        std::ostringstream oss;
        oss << std::setw(5) << std::setfill('0') << counter << ".bin";

        fs::path fullPath = fs::path(folderPath) / oss.str();

        if (!fs::exists(fullPath))
        {
            return fullPath.string();
        }

        counter++;
    }
    return "";
}

std::vector<float> extractTensorData(const TensorView &view, const void *data)
{
    std::vector<float> result;
    uint64_t elems = countElements(view);
    result.reserve(elems);
    if (view.dtype == DType::FLOAT32)
    {
        const float *src = static_cast<const float *>(data);
        for (uint64_t i = 0; i < elems; ++i)
            result.push_back(src[getStridedIndex(i, view.getShape(), view.strides)]);
    }
    else if (view.dtype == DType::INT32)
    {
        const int32_t *src = static_cast<const int32_t *>(data);
        for (uint64_t i = 0; i < elems; ++i)
            result.push_back(static_cast<float>(src[getStridedIndex(i, view.getShape(), view.strides)]));
    }
    else if (view.dtype == DType::BF16)
    {
        const uint16_t *src = static_cast<const uint16_t *>(data);
        for (uint64_t i = 0; i < elems; ++i)
        {
            uint32_t bits = static_cast<uint32_t>(src[getStridedIndex(i, view.getShape(), view.strides)]) << 16;
            float val;
            std::memcpy(&val, &bits, 4);
            result.push_back(val);
        }
    }
    else if (view.dtype == DType::BOOL)
    {
        const uint8_t *src = static_cast<const uint8_t *>(data);
        for (uint64_t i = 0; i < elems; ++i)
            result.push_back(static_cast<float>(src[getStridedIndex(i, view.getShape(), view.strides)]));
    }
    return result;
}

struct RefIndexEntry
{
    uint64_t fileOffset;
    uint64_t numElements;
    std::string opName;
};

void runGemma(bool refOnly, bool doSaturate, std::function<void(uint32_t logicalId, const std::string &opName, const std::vector<float> &data)> callback)
{
    KernelRegistry::get().setReferenceOnly(refOnly);

    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 16ULL * 1024 * 1024 * 1024}};
#if USE_CUDA
    bufferSizes[Backend::CUDA] = 16ULL * 1024 * 1024 * 1024;
#endif
    MemoryManager mem(bufferSizes);
    Graph g;

    Gemma3ModelConfig cfg;
    cfg.n_layers = 1; // 1 layer provides a comprehensive view across all logic blocks rapidly
    uint32_t maxSeqLen = 8;

    uint32_t inputIdsId = g.input({1, maxSeqLen}, DType::INT32, {}, StorageType::PERSISTENT);

    Gemma3Model gemma(cfg, maxSeqLen, g, mem, "resources/model.safetensors");
    uint32_t rootId = gemma.build_graph(inputIdsId);

    std::vector<int32_t> input_ids = {2, 9259, 0, 0, 0, 0, 0, 0};

    // Allocate the input in persistent memory and feed into model mapping
    uint64_t sizeBytes = maxSeqLen * getDTypeSize(DType::INT32);
    mem.allocate(Backend::CPU, inputIdsId, sizeBytes, StorageType::PERSISTENT);

    std::unordered_map<uint32_t, const void *> inputs = {{inputIdsId, input_ids.data()}};

    // Use an empty string for the cache to prevent collisions between sequential runs
    Session sess(g, mem, rootId, "test_model_cache/00000.bin");

    auto debugCb = [&](uint32_t logicalId, const TensorView &view, const void *data)
    {
        if (!callback)
            return;

        std::vector<float> extracted = extractTensorData(view, data);
        std::string opName = "UNKNOWN";

        if (g.hasNode(logicalId))
        {
            auto node = g.getNode(logicalId);
            opName = toString(node.opType);
            if (node.opType == OpType::FUSED)
            {
                opName = "FUSED_" + node.opName;
            }
        }

        callback(logicalId, opName, extracted);
    };

    sess.compile(doSaturate);
    sess.run(inputs, debugCb, doSaturate);
}

int main(int argc, char *argv[])
{
    const std::string REF_FILE = "reference_tensors.bin";
    std::unordered_map<uint32_t, RefIndexEntry> refIndex;
    bool loaded = false;

    // Try to load an existing reference tensor file to save compute
    std::ifstream testFile(REF_FILE, std::ios::binary);
    if (testFile.is_open())
    {
        while (testFile.peek() != EOF)
        {
            uint32_t logicalId;
            if (!testFile.read(reinterpret_cast<char *>(&logicalId), sizeof(logicalId)))
                break;

            uint32_t nameLen;
            testFile.read(reinterpret_cast<char *>(&nameLen), sizeof(nameLen));
            std::string opName(nameLen, '\0');
            if (nameLen > 0)
                testFile.read(&opName[0], nameLen);

            uint64_t numElements;
            testFile.read(reinterpret_cast<char *>(&numElements), sizeof(numElements));

            RefIndexEntry entry;
            entry.opName = opName;
            entry.numElements = numElements;
            entry.fileOffset = testFile.tellg();

            refIndex[logicalId] = entry;

            // Skip ahead past the data so we only index metadata
            testFile.seekg(numElements * sizeof(float), std::ios::cur);
        }
        loaded = true;
    }

    if (loaded)
    {
        std::cout << ">>> Phase 1: Loaded Reference Index from " << REF_FILE << " (" << refIndex.size() << " tensors)\n";
    }
    else
    {
        std::cout << ">>> Phase 1: Running Reference Model (No Saturation) and saving to " << REF_FILE << "\n";
        std::ofstream outFile(REF_FILE, std::ios::binary | std::ios::trunc);
        if (!outFile.is_open())
        {
            std::cerr << "Failed to open " << REF_FILE << " for writing.\n";
            return 1;
        }

        runGemma(true, false, [&](uint32_t logicalId, const std::string &opName, const std::vector<float> &data)
                 {
            uint32_t nameLen = opName.size();
            uint64_t numElems = data.size();

            outFile.write(reinterpret_cast<const char *>(&logicalId), sizeof(logicalId));
            outFile.write(reinterpret_cast<const char *>(&nameLen), sizeof(nameLen));
            if (nameLen > 0) outFile.write(opName.c_str(), nameLen);
            outFile.write(reinterpret_cast<const char *>(&numElems), sizeof(numElems));

            RefIndexEntry entry;
            entry.opName = opName;
            entry.numElements = numElems;
            entry.fileOffset = outFile.tellp();
            refIndex[logicalId] = entry;

            outFile.write(reinterpret_cast<const char *>(data.data()), numElems * sizeof(float)); });
        outFile.close();
    }

    std::cout << "\n>>> Phase 2: Running Optimized Model (With Saturation)\n";
    std::cout << "\n=======================================================================================\n";
    std::cout << "Accuracy Comparison (Reference vs Optimized)\n";
    std::cout << "=======================================================================================\n";
    std::cout << std::left
              << std::setw(12) << "LogicalID"
              << std::setw(25) << "OpType"
              << std::setw(15) << "Min Diff"
              << std::setw(15) << "Max Diff"
              << std::setw(15) << "Avg Diff"
              << "\n";
    std::cout << std::string(87, '-') << "\n";

    std::ifstream inFile(REF_FILE, std::ios::binary);
    if (!inFile.is_open())
    {
        std::cerr << "Failed to open " << REF_FILE << " for reading during phase 2.\n";
        return 1;
    }

    int mismatchCount = 0;

    runGemma(false, true, [&](uint32_t logicalId, const std::string &opName, const std::vector<float> &optData)
             {
        auto it = refIndex.find(logicalId);
        if (it == refIndex.end()) return; // implicitly evaluated or dropped node

        const auto& entry = it->second;
        if (entry.numElements != optData.size())
        {
            std::cout << std::left << std::setw(12) << logicalId
                        << "SIZE MISMATCH: " << entry.numElements << " vs " << optData.size() << "\n";
            mismatchCount++;
            return;
        }

        if (optData.empty()) return;

        // Stream from the binary file only when actively performing this callback
        std::vector<float> refData(entry.numElements);
        inFile.seekg(entry.fileOffset, std::ios::beg);
        inFile.read(reinterpret_cast<char *>(refData.data()), entry.numElements * sizeof(float));

        float minDiff = std::numeric_limits<float>::max();
        float maxDiff = 0.0f;
        double sumDiff = 0.0;
        bool hasNan = false;

        for (size_t i = 0; i < refData.size(); ++i)
        {
            float diff = std::abs(refData[i] - optData[i]);
            if (std::isnan(diff))
            {
                hasNan = true;
                break;
            }
            if (diff < minDiff) minDiff = diff;
            if (diff > maxDiff) maxDiff = diff;
            sumDiff += diff;
        }

        if (hasNan)
        {
            minDiff = std::numeric_limits<float>::quiet_NaN();
            maxDiff = std::numeric_limits<float>::quiet_NaN();
            sumDiff = std::numeric_limits<double>::quiet_NaN();
        }

        float avgDiff = static_cast<float>(sumDiff / refData.size());

        if (maxDiff > 1e-2f || hasNan)
        {
            std::cout << "\033[1;31m"; // Console Red
            mismatchCount++;
        }

        std::cout << std::left
                    << std::setw(12) << logicalId
                    << std::setw(25) << entry.opName.substr(0, 24)
                    << std::setw(15) << minDiff
                    << std::setw(15) << maxDiff
                    << std::setw(15) << avgDiff
                    << "\n";

        if (maxDiff > 1e-2f || hasNan)
        {
            std::cout << "\033[0m"; // Console Reset
        } });

    std::cout << "=======================================================================================\n";
    if (mismatchCount > 0)
    {
        std::cout << "Found " << mismatchCount << " nodes with high deviation (>1e-2) or NaNs.\n";
    }
    else
    {
        std::cout << "All nodes matched perfectly or within acceptable precision limits.\n";
    }
    return 0;
}