// File: tensor_graphs_cpp/core/cost_model.hpp
#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "generated/build_context.gen.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>

// Uncomment the following line to enable logging calls to `benchmarks/calls.jsonl`
// #define TENSOR_GRAPHS_LOG_COST_CALLS

struct Record
{
    uint64_t kernelUid;
    uint64_t buildContextId;
    std::string hwTag;

    std::vector<std::vector<uint32_t>> inputShapes;
    std::vector<std::vector<uint32_t>> outputShapes;
    std::vector<DType> inputDTypes;
    std::vector<DType> outputDTypes;
    std::vector<std::vector<uint8_t>> inputConstants;
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
        {"inputDTypes", r.inputDTypes},
        {"outputDTypes", r.outputDTypes},
        {"inputConstants", r.inputConstants},
        {"runTime", r.runTime}};
}

inline void from_json(const json &j, Record &r)
{
    r.kernelUid = std::stoull(j.at("kernelUid").get<std::string>(), nullptr, 16);
    r.buildContextId = std::stoull(j.at("buildContextId").get<std::string>(), nullptr, 16);
    r.hwTag = j.at("hwTag").get<std::string>();

    r.inputShapes = j.at("inputShapes").get<std::vector<std::vector<uint32_t>>>();
    r.outputShapes = j.at("outputShapes").get<std::vector<std::vector<uint32_t>>>();
    r.inputDTypes = j.at("inputDTypes").get<std::vector<DType>>();
    r.outputDTypes = j.at("outputDTypes").get<std::vector<DType>>();
    r.inputConstants = j.at("inputConstants").get<std::vector<std::vector<uint8_t>>>();
    r.runTime = j.at("runTime").get<float>();
}

struct CostModel
{
    std::unordered_map<uint64_t, std::vector<Record>> records;
    std::unordered_set<std::string> loggedCalls;
    std::ofstream callFile;

    CostModel()
    {
#ifdef TENSOR_GRAPHS_LOG_COST_CALLS
        const std::string path = "benchmarks/calls.jsonl";
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
        while (std::getline(file, line))
        {
            if (line.empty())
                continue;
            auto j = json::parse(line);
            Record r = j.get<Record>();
            records[r.kernelUid].push_back(r);
        }
    }

    float interpolate(const std::vector<Record> &kernelRecords, const TensorNode &node, const Graph &graph)
    {
        uint64_t targetElements = countElements(node.shape);
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

    float estimateCost(const TensorNode &node, const Graph &graph, uint64_t kernelUid)
    {
        std::vector<std::vector<uint32_t>> inShapes(node.parentIds.size());
        std::vector<std::vector<uint8_t>> inConstants(node.parentIds.size());
        std::vector<DType> inDTypes(node.parentIds.size());

        for (size_t i = 0; i < node.parentIds.size(); ++i)
        {
            uint32_t pid = node.parentIds[i];
            inShapes[i] = graph.nodes[pid].shape;

            if (graph.nodes[pid].opType == OpType::INPUT && graph.nodes[pid].storageType == StorageType::PERSISTENT)
            {
                auto stagingIt = graph.constantStaging.find(pid);
                if (stagingIt != graph.constantStaging.end())
                    inConstants[i] = stagingIt->second;
            }
        }
        std::vector<std::vector<uint32_t>> outShapes = {node.shape};
        std::vector<DType> outDTypes = {node.dtype};

#ifdef TENSOR_GRAPHS_LOG_COST_CALLS
        {
            Record r;
            r.kernelUid = kernelUid;
            r.buildContextId = BUILD_CONTEXT_ID;

// TODO: make hardware detection better
#if defined(_WIN32) || defined(_WIN64)
            r.hwTag = "Windows_ARM64";
#elif defined(__APPLE__)
            r.hwTag = "Apple_Silicon";
#else
            r.hwTag = "Linux_ARM64";
#endif

            r.inputShapes = inShapes;
            r.outputShapes = outShapes;
            r.inputDTypes = inDTypes;
            r.outputDTypes = outDTypes;
            r.inputConstants = inConstants;
            r.runTime = 0.0f;

            json callObj = r;
            std::string callStr = callObj.dump();

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

        auto it = records.find(kernelUid);
        if (it == records.end() || it->second.empty())
            return std::numeric_limits<float>::infinity();

        for (const auto &r : it->second)
        {
            if (r.inputShapes == inShapes && r.outputShapes == outShapes &&
                r.inputDTypes == inDTypes && r.outputDTypes == outDTypes)
            {
                return r.runTime;
            }
        }

        return interpolate(it->second, node, graph);
    }
};