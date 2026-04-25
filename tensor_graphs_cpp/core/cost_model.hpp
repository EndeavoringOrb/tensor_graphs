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

// Uncomment the following line to enable logging calls to `benchmarks/calls.jsonl`
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

inline void to_json(json &j, const Record &r)
{
    std::stringstream uid_ss, build_ss;
    uid_ss << "0x" << std::hex << r.kernelUid;
    build_ss << "0x" << std::hex << r.buildContextId;

    j = json{
        {"kernelUid", uid_ss.str()},
        {"buildContextId", build_ss.str()},
        {"hwTag", r.hwTag},
        {"inputShapes", r.inputShapes},
        {"outputShapes", r.outputShapes},
        {"inputStrides", r.inputStrides},
        {"outputStrides", r.outputStrides},
        {"inputDTypes", r.inputDTypes},
        {"outputDTypes", r.outputDTypes},
        {"inputConstants", r.inputConstants},
        {"backends", r.backends},
        {"inputBackends", r.inputBackends},
        {"runTime", r.runTime}};
}

inline void from_json(const json &j, Record &r)
{
    r.kernelUid = std::stoull(j.at("kernelUid").get<std::string>(), nullptr, 16);
    r.buildContextId = std::stoull(j.at("buildContextId").get<std::string>(), nullptr, 16);
    r.hwTag = j.at("hwTag").get<std::string>();

    r.inputShapes = j.at("inputShapes").get<std::vector<std::vector<uint32_t>>>();
    r.outputShapes = j.at("outputShapes").get<std::vector<std::vector<uint32_t>>>();
    r.inputStrides = j.at("inputStrides").get<std::vector<std::vector<uint64_t>>>();
    r.outputStrides = j.at("outputStrides").get<std::vector<std::vector<uint64_t>>>();

    r.inputDTypes = j.at("inputDTypes").get<std::vector<DType>>();
    r.outputDTypes = j.at("outputDTypes").get<std::vector<DType>>();
    r.inputConstants = j.at("inputConstants").get<std::vector<std::vector<uint8_t>>>();
    r.backends = j.at("backends").get<std::vector<Backend>>();
    r.inputBackends = j.at("inputBackends").get<std::vector<std::vector<Backend>>>();

    r.runTime = j.at("runTime").get<float>();
}

struct CostModel
{
    std::unordered_map<uint64_t, std::vector<Record>> records;
    std::unordered_set<std::string> loggedCalls;
    std::ofstream callFile;
    std::mutex logMtx;
    bool doneWarning = false;

    CostModel()
    {
#ifdef TENSOR_GRAPHS_LOG_COST_CALLS
        const std::string path = "benchmarks/calls.jsonl";
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        {
            std::ifstream inFile(path);
            if (inFile.is_open())
            {
                std::string line;
                while (std::getline(inFile, line))
                {
                    if (!line.empty())
                        loggedCalls.insert(line);
                }
            }
        }
        callFile.open(path, std::ios::app);
        if (!callFile.is_open())
            std::cerr << "Failed to open " << path << " for appending.\n";
#endif
    }

    void load(std::string benchmarkPath)
    {
        records.clear();
        std::ifstream file(benchmarkPath);
        if (!file.is_open())
            return;

        std::string line;
        uint32_t total = 0;
        uint32_t valid = 0;
        ProgressTimer timer(0, "loading records ");
        while (std::getline(file, line))
        {
            timer.tick();
            if (line.empty())
                continue;
            total++;
            auto j = json::parse(line);
            Record r = j.get<Record>();
            bool hasKernel = KernelRegistry::get().hasKernel(r.kernelUid);
            if (r.hwTag != HW_TAG || r.buildContextId != BUILD_CONTEXT_ID || !hasKernel)
                continue;
            valid++;
            records[r.kernelUid].push_back(r);
        }
        std::cout << "Loaded " << valid << " valid records out of " << total << " total records from " << benchmarkPath << std::endl;
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

                json callObj = r;
                std::string callStr = callObj.dump();

                std::lock_guard<std::mutex> lock(logMtx);
                if (loggedCalls.find(callStr) == loggedCalls.end())
                {
                    loggedCalls.insert(callStr);
                    if (callFile.is_open())
                    {
                        callFile << callStr << "\n";
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
                r.inputDTypes == inDTypes && r.outputDTypes == outDTypes)
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

            json callObj = r;
            std::string callStr = callObj.dump();

            std::lock_guard<std::mutex> lock(logMtx);
            if (loggedCalls.find(callStr) == loggedCalls.end())
            {
                loggedCalls.insert(callStr);
                if (callFile.is_open())
                {
                    callFile << callStr << "\n";
                    callFile.flush();
                }
            }
        }
#endif
        uint64_t targetElements = countElements(outShape);
        return interpolate(it->second, targetElements);
    }
};
