#pragma once
#include "core/ops/common.hpp"

struct UnpackOp
{
    static constexpr OpType op_type = OpType::UNPACK;
    static constexpr const char *name = "UNPACK";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        auto s0 = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape();
        DType in_dtype = graph.getNode(graph.getNode(nodeId).child_ids[0]).dtype;
        DType out_dtype = graph.getNode(nodeId).dtype;
        uint32_t in_bits = getDTypeNBits(in_dtype);
        uint32_t out_bits = getDTypeNBits(out_dtype);
        if (in_bits % out_bits != 0)
        {
            Error::throw_err("UNPACK requires input bits to be divisible by output bits.");
        }
        uint32_t factor = in_bits / out_bits;
        std::vector<uint32_t> out_shape = s0;
        if (!out_shape.empty())
        {
            out_shape.back() *= factor;
        }
        graph.getNode(nodeId).setShape(out_shape);
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

    static bool isConstant(uint64_t, uint64_t)
    {
        return false;
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType dtype)
    {
        return pGraph.unpack(pInputs[0], dtype);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};