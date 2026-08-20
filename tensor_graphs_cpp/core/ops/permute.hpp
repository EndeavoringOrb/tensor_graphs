#pragma once
#include "core/misc.hpp"
#include "core/ops/common.hpp"

struct PermuteOp
{
    static constexpr OpType op_type = OpType::PERMUTE;
    static constexpr const char *name = "PERMUTE";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        auto s0 = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape();
        auto dims = graph.getConstantInt32(graph.getNode(nodeId).child_ids[1]);
        std::vector<uint32_t> out_shape(dims.size());
        for (uint64_t i = 0; i < dims.size(); ++i)
        {
            out_shape[i] = s0[dims[i]];
        }
        graph.getNode(nodeId).setShape(out_shape);

        auto parentStrides = graph.getNode(graph.getNode(nodeId).child_ids[0]).strides;
        for (uint64_t i = 0; i < dims.size(); ++i)
        {
            graph.getNode(nodeId).strides[i] = parentStrides[dims[i]];
        }
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        auto dims = graph.getConstantInt32(node.child_ids[1]);

        std::vector<Region> outBoxes;
        for (const auto &reg : rA)
        {
            Region outBox;
            for (int32_t d : dims)
            {
                outBox.region.push_back(reg.region[d]);
            }
            outBoxes.push_back(outBox);
        }
        return mergeRegions(outBoxes);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &graph,
                                                           const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        auto dims = graph.getConstantInt32(node.child_ids[1]);
        std::vector<int32_t> invDims(dims.size());
        for (uint64_t i = 0; i < dims.size(); ++i)
        {
            invDims[dims[i]] = static_cast<int32_t>(i);
        }

        std::vector<Region> inBoxes;
        for (const auto &outReg : outputRegions)
        {
            Region inBox;
            for (int32_t d : invDims)
            {
                inBox.region.push_back(outReg.region[d]);
            }
            inBoxes.push_back(inBox);
        }
        return {mergeRegions(inBoxes), makeFull(graph.getNode(node.child_ids[1]).getShape())};
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
        return pGraph.permute(pInputs[0], pInputs[1]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};