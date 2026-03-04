#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>

// Uncomment the following line to enable logging calls to `benchmarks/calls.jsonl`
#define TENSOR_GRAPHS_LOG_COST_CALLS 1

struct Record
{
    std::vector<std::vector<uint32_t>> inputShapes;
    std::vector<std::vector<uint32_t>> outputShapes;
    std::vector<std::vector<uint8_t>> inputConstants;
    float runTime; // run time in milliseconds
};

inline void to_json(json &j, const Record &r)
{
    j = json{
        {"inputShapes", r.inputShapes},
        {"outputShapes", r.outputShapes},
        {"inputConstants", r.inputConstants},
        {"runTime", r.runTime}};
}

inline void from_json(const json &j, Record &r)
{
    r.inputShapes = j.at("inputShapes").get<std::vector<std::vector<uint32_t>>>();
    r.outputShapes = j.at("outputShapes").get<std::vector<std::vector<uint32_t>>>();
    if (j.contains("inputConstants"))
        r.inputConstants = j.at("inputConstants").get<std::vector<std::vector<uint8_t>>>();
    r.runTime = j.at("runTime").get<float>();
}

struct CostModel
{
    std::unordered_map<uint32_t, std::vector<Record>> records;
    std::unordered_set<std::string> loggedCalls;
    std::ofstream callFile;

    CostModel()
    {
#ifdef TENSOR_GRAPHS_LOG_COST_CALLS
        const std::string path = "benchmarks/calls.jsonl";

        // 1. Load existing logged calls
        {
            std::ifstream inFile(path);
            if (inFile.is_open())
            {
                std::string line;
                while (std::getline(inFile, line))
                {
                    if (!line.empty())
                    {
                        loggedCalls.insert(line);
                    }
                }
            }
        }

        // 2. Open for append
        callFile.open(path, std::ios::app);

        if (!callFile.is_open())
        {
            std::cerr << "Failed to open " << path << " for appending.\n";
        }
#endif
    }

    void load(std::string benchmarkPath)
    {
        // Clear previous state
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
            uint32_t kid = j.at("kernelId").get<uint32_t>();
            records[kid].push_back(r);
        }
    }

    // TODO: improve this. for example: sum of 2048 and sum of 1024 have same output shape but different runtimes, so current interpolate would be a bad heuristic
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
            {
                recElements += countElements(s);
            }
            if (recElements == 0)
                recElements = 1;

            float dist = std::abs(static_cast<float>(targetElements) - static_cast<float>(recElements));
            if (dist < bestDist)
            {
                bestDist = dist;
                estimatedTime = r.runTime * (static_cast<float>(targetElements) / static_cast<float>(recElements));
            }
        }

        if (bestDist == std::numeric_limits<float>::infinity())
        {
            return std::numeric_limits<float>::infinity();
        }
        return estimatedTime;
    }

    float estimateCost(const TensorNode &node, const Graph &graph, uint32_t kernelId)
    {
        std::vector<std::vector<uint32_t>> inShapes(node.parentIds.size());
        std::vector<std::vector<uint8_t>> inConstants(node.parentIds.size());

        for (size_t i = 0; i < node.parentIds.size(); ++i)
        {
            uint32_t pid = node.parentIds[i];
            inShapes[i] = graph.nodes[pid].shape;

            if (graph.nodes[pid].opType == OpType::INPUT && graph.nodes[pid].storageType == StorageType::PERSISTENT)
            {
                auto stagingIt = graph.constantStaging.find(pid);
                if (stagingIt != graph.constantStaging.end())
                {
                    inConstants[i] = stagingIt->second;
                }
            }
        }
        std::vector<std::vector<uint32_t>> outShapes = {node.shape};

#ifdef TENSOR_GRAPHS_LOG_COST_CALLS
        {
            Record r;
            r.inputShapes = inShapes;
            r.outputShapes = outShapes;
            r.inputConstants = inConstants;
            r.runTime = 0.0f;

            json callObj = r;
            callObj["kernelId"] = kernelId;
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

        auto it = records.find(kernelId);
        if (it == records.end() || it->second.empty())
        {
            return std::numeric_limits<float>::infinity();
        }

        for (const auto &r : it->second)
        {
            if (r.inputShapes == inShapes && r.outputShapes == outShapes)
            {
                return r.runTime;
            }
        }

        return interpolate(it->second, node, graph);
    }
};