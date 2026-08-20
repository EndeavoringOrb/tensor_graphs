#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "core/types.hpp"

struct WorkloadMetrics
{
    double flops = 0.0;
    double bytesRead = 0.0;
    double bytesWritten = 0.0;
};

struct Graph;

struct OpTraits
{
    OpType op_type = OpType::INPUT;
    const char *name = "";
    bool is_elementwise = false;

    void (*inferShape)(LogicalId nodeId, Graph &graph) = nullptr;
    std::vector<Region> (*forwardRegion)(const TensorNode &node, const Graph &graph,
                                         const std::vector<std::vector<Region>> &parentRegions) = nullptr;
    std::vector<std::vector<Region>> (*backwardRegion)(const TensorNode &node, const Graph &graph,
                                                       const std::vector<Region> &outputRegions) = nullptr;
    WorkloadMetrics (*computeWorkload)(const std::vector<std::vector<uint32_t>> &inShapes,
                                       const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                       DType outDType, const std::string &opName) = nullptr;
    bool (*isConstant)(uint64_t inputIdx, uint64_t numInputs) = nullptr;
    LogicalId (*buildPattern)(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType dtype) = nullptr;
};