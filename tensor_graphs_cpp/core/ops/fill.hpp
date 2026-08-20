#pragma once
#include "core/ops/common.hpp"

struct FillOp
{
    static constexpr OpType op_type = OpType::FILL;
    static constexpr const char *name = "FILL";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        auto target_dims = graph.getConstantInt32(graph.getNode(nodeId).child_ids[1]);
        std::vector<uint32_t> out_shape(target_dims.size());
        for (uint64_t i = 0; i < target_dims.size(); ++i)
        {
            out_shape[i] = target_dims[i];
        }
        graph.getNode(nodeId).setShape(out_shape);
        graph.getNode(nodeId).strides.assign(out_shape.size(), 0);
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
        return inputIdx == 1;
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.fill(pInputs[0], pInputs[1]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};