#pragma once
#include "core/ops/common.hpp"

struct Im2ColOp
{
    static constexpr OpType op_type = OpType::IM2COL;
    static constexpr const char *name = "IM2COL";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        auto s0 = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape(); // N, C, H, W
        uint32_t k = graph.getConstantInt32(graph.getNode(nodeId).child_ids[1])[0];
        uint32_t s = graph.getConstantInt32(graph.getNode(nodeId).child_ids[2])[0];
        uint32_t p = graph.getConstantInt32(graph.getNode(nodeId).child_ids[3])[0];
        uint32_t H = s0[2];
        uint32_t W = s0[3];
        uint32_t H_out = (H + 2 * p - k) / s + 1;
        uint32_t W_out = (W + 2 * p - k) / s + 1;
        graph.getNode(nodeId).setShape({s0[0], s0[1] * k * k, H_out * W_out});
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        return forwardFull(node, graph, parentRegions);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &graph,
                                                           const std::vector<Region> &outputRegions)
    {
        return backwardFull(node, graph, outputRegions);
    }

    static WorkloadMetrics computeWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                           const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                           DType outDType, const std::string &)
    {
        return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType, 0.0);
    }

    static bool isConstant(uint64_t inputIdx, uint64_t)
    {
        return inputIdx == 1 || inputIdx == 2 || inputIdx == 3;
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.im2col(pInputs[0], pInputs[1], pInputs[2], pInputs[3]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};