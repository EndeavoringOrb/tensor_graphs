// File: tensor_graphs_cpp/core/cost_model.hpp
#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/misc.hpp"
#include "generated/build_context.gen.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <filesystem>
#include <mutex>

// TODO: make hardware detection better
#if defined(USE_CUDA)
#define HW_TAG "CUDA_Enabled"
#else
// Determine OS String
#if defined(TG_OS_WINDOWS)
#define PLAT_OS_STR "Windows"
#elif defined(TG_OS_MACOS)
#define PLAT_OS_STR "macOS"
#elif defined(TG_OS_LINUX)
#define PLAT_OS_STR "Linux"
#else
#define PLAT_OS_STR "UnknownOS"
#endif

// Determine Arch String
#if defined(TG_ARCH_ARM64)
#define PLAT_ARCH_STR "ARM64"
#elif defined(TG_ARCH_X64)
#define PLAT_ARCH_STR "x64"
#else
#define PLAT_ARCH_STR "UnknownArch"
#endif

#define HW_TAG PLAT_OS_STR "_" PLAT_ARCH_STR
#endif

// Uncomment the following line to enable logging calls to `benchmarks/calls.bin`
#define TENSOR_GRAPHS_LOG_COST_CALLS

struct Record
{
    uint64_t kernelUid;
    uint64_t buildContextId;
    std::string hwTag;

    std::vector<std::vector<uint32_t>> inputShapes;
    std::vector<std::vector<uint32_t>> outputShapes;
    std::vector<std::vector<uint64_t>> inputStrides;
    std::vector<std::vector<uint64_t>> outputStrides;
    std::vector<DType> inputDTypes;
    std::vector<DType> outputDTypes;
    std::vector<std::vector<uint8_t>> inputConstants;
    std::vector<Backend> backends;
    std::vector<std::vector<Backend>> inputBackends;
    float runTime;
};

inline void tg_serialize(BinaryWriter &bw, const Record &val)
{
    bw.write(val.kernelUid);
    bw.write(val.buildContextId);
    bw.write(val.hwTag);
    bw.write(val.inputShapes);
    bw.write(val.outputShapes);
    bw.write(val.inputStrides);
    bw.write(val.outputStrides);
    bw.write(val.inputDTypes);
    bw.write(val.outputDTypes);
    bw.write(val.inputConstants);
    bw.write(val.backends);
    bw.write(val.inputBackends);
    bw.write(val.runTime);
}

inline void tg_deserialize(BinaryReader &br, Record &val)
{
    br.read(val.kernelUid);
    br.read(val.buildContextId);
    br.read(val.hwTag);
    br.read(val.inputShapes);
    br.read(val.outputShapes);
    br.read(val.inputStrides);
    br.read(val.outputStrides);
    br.read(val.inputDTypes);
    br.read(val.outputDTypes);
    br.read(val.inputConstants);
    br.read(val.backends);
    br.read(val.inputBackends);
    br.read(val.runTime);
}

struct CostModel
{
    std::unordered_map<uint64_t, std::vector<Record>> records;
    std::unordered_set<size_t> loggedCalls; // <-- CHANGE TO size_t
    std::ofstream callFile;
    std::mutex logMtx;
    bool doneWarning = false;

    CostModel()
    {
#ifdef TENSOR_GRAPHS_LOG_COST_CALLS
        const std::string path = "benchmarks/calls.bin";
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        {
            std::ifstream inFile(path, std::ios::binary);
            if (inFile.is_open())
            {
                BinaryReader br(inFile);
                while (inFile.peek() != EOF)
                {
                    Record r;
                    br.read(r);
                    r.runTime = 0.0f; // normalize for hash
                    loggedCalls.insert(std::hash<std::string>{}(serializeToString(r)));
                }
            }
        }
        callFile.open(path, std::ios::app | std::ios::binary);
        if (!callFile.is_open())
            std::cerr << "Failed to open " << path << " for appending.\n";
#endif
    }

    void load(std::string benchmarkPath)
    {
        records.clear();
        std::ifstream file(benchmarkPath, std::ios::binary);
        if (!file.is_open())
            return;

        BinaryReader br(file);
        uint32_t total = 0, valid = 0;
        ProgressTimer timer(0, "loading records ");
        while (file.peek() != EOF)
        {
            timer.tick();
            Record r;
            try
            {
                br.read(r);
            }
            catch (...)
            {
                break;
            }
            total++;
            if (r.hwTag != HW_TAG || r.buildContextId != BUILD_CONTEXT_ID || !KernelRegistry::get().hasKernel(r.kernelUid))
                continue;
            valid++;
            records[r.kernelUid].push_back(std::move(r));
        }
        std::cout << "Loaded " << valid << " valid records from " << benchmarkPath << std::endl;
    }

    float interpolate(const std::vector<Record> &kernelRecords, uint64_t targetElements)
    {
        if (targetElements == 0)
            return 0.0f;

        float bestDist = std::numeric_limits<float>::infinity();
        float estimatedTime = 0.0f;

        for (const auto &r : kernelRecords)
        {
            uint64_t recElements = 0;
            for (const auto &s : r.outputShapes)
                recElements += countElements(s);
            if (recElements == 0)
                recElements = 1;

            float dist = std::abs(static_cast<float>(targetElements) - static_cast<float>(recElements));
            if (dist < bestDist)
            {
                bestDist = dist;
                estimatedTime = r.runTime * (static_cast<float>(targetElements) / static_cast<float>(recElements));
            }
        }
        return (bestDist == std::numeric_limits<float>::infinity()) ? bestDist : estimatedTime;
    }

    float estimateCost(
        uint64_t kernelUid,
        const std::vector<uint32_t> &outShape,
        const std::vector<uint64_t> &_outStrides,
        DType outDType,
        const std::vector<std::vector<uint32_t>> &inShapes,
        const std::vector<std::vector<uint64_t>> &inStrides,
        const std::vector<DType> &inDTypes,
        const std::vector<std::vector<uint8_t>> &inConstants)
    {
        std::vector<std::vector<uint32_t>> outShapes = {outShape};
        std::vector<DType> outDTypes = {outDType};
        const std::vector<std::vector<uint64_t>> outStrides = {_outStrides};

        auto it = records.find(kernelUid);
        if (it == records.end() || it->second.empty())
        {
#ifdef TENSOR_GRAPHS_LOG_COST_CALLS
            {
                Record r;
                r.kernelUid = kernelUid;
                r.buildContextId = BUILD_CONTEXT_ID;
                r.hwTag = HW_TAG;
                r.inputShapes = inShapes;
                r.outputShapes = outShapes;
                r.inputStrides = inStrides;
                r.outputStrides = outStrides;
                r.inputDTypes = inDTypes;
                r.outputDTypes = outDTypes;
                r.inputConstants = inConstants;
                const auto &entry = KernelRegistry::get().getKernel(kernelUid);
                r.backends = entry.backends;
                r.inputBackends = entry.inputBackends;
                r.runTime = 0.0f;
                std::string callStr = serializeToString(r);
                size_t callHash = std::hash<std::string>{}(callStr);

                std::lock_guard<std::mutex> lock(logMtx);
                if (loggedCalls.find(callHash) == loggedCalls.end())
                {
                    loggedCalls.insert(callHash);
                    if (callFile.is_open())
                    {
                        BinaryWriter bw(callFile);
                        bw.write(r);
                        callFile.flush();
                    }
                }
            }
#endif
            if (!doneWarning)
            {
                std::cout << "\nWARNING INF COST ESTIMATION DUE TO MISSING RECORDS\n"
                          << std::flush;
                doneWarning = true;
            }
            return std::numeric_limits<float>::infinity();
        }

        for (const auto &r : it->second)
        {
            if (r.inputShapes == inShapes && r.outputShapes == outShapes &&
                r.inputStrides == inStrides && r.outputStrides == outStrides &&
                r.inputDTypes == inDTypes && r.outputDTypes == outDTypes &&
                r.inputConstants == inConstants)
            {
                return r.runTime;
            }
        }

#ifdef TENSOR_GRAPHS_LOG_COST_CALLS
        {
            Record r;
            r.kernelUid = kernelUid;
            r.buildContextId = BUILD_CONTEXT_ID;
            r.hwTag = HW_TAG;
            r.inputShapes = inShapes;
            r.outputShapes = outShapes;
            r.inputStrides = inStrides;
            r.outputStrides = outStrides;
            r.inputDTypes = inDTypes;
            r.outputDTypes = outDTypes;
            r.inputConstants = inConstants;
            const auto &entry = KernelRegistry::get().getKernel(kernelUid);
            r.backends = entry.backends;
            r.inputBackends = entry.inputBackends;
            r.runTime = 0.0f;

            std::string callStr = serializeToString(r);
            size_t callHash = std::hash<std::string>{}(callStr);

            std::lock_guard<std::mutex> lock(logMtx);
            if (loggedCalls.find(callHash) == loggedCalls.end())
            {
                loggedCalls.insert(callHash);
                if (callFile.is_open())
                {
                    BinaryWriter bw(callFile);
                    bw.write(r);
                    callFile.flush();
                }
            }
        }
#endif
        uint64_t targetElements = countElements(outShape);
        return interpolate(it->second, targetElements);
    }
};
