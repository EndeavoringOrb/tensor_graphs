#pragma once
#include "core/misc.hpp"
#include "core/ops/common.hpp"

struct RepeatOp
{
    static constexpr OpType op_type = OpType::REPEAT;
    static constexpr const char *name = "REPEAT";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        auto s0 = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape();
        auto repeats = graph.getConstantInt32(graph.getNode(nodeId).child_ids[1])[0];
        auto axis = graph.getConstantInt32(graph.getNode(nodeId).child_ids[2])[0];
        if (axis < 0)
            axis += s0.size();
        std::vector<uint32_t> out_shape = s0;
        out_shape[axis] *= repeats;
        graph.getNode(nodeId).setShape(out_shape);

        auto parentStrides = graph.getNode(graph.getNode(nodeId).child_ids[0]).strides;
        graph.getNode(nodeId).strides = parentStrides;
        for (uint64_t d = 0; d < out_shape.size(); ++d)
        {
            if (s0[d] != out_shape[d])
            {
                graph.getNode(nodeId).strides[d] = 0;
            }
        }
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &sA = graph.getNode(node.child_ids[0]).getShape();

        int32_t repeats = graph.getConstantInt32(node.child_ids[1])[0];
        int32_t axis = graph.getConstantInt32(node.child_ids[2])[0];
        if (axis < 0)
            axis += sA.size();

        std::vector<Region> outBoxes;
        for (const auto &region : rA)
        {
            Region outBox = region;
            outBox.region[axis].start *= repeats;
            outBox.region[axis].stop *= repeats;
            outBoxes.push_back(outBox);
        }

        return mergeRegions(outBoxes);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &graph,
                                                           const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}};

        const auto &sA = graph.getNode(node.child_ids[0]).getShape();
        int32_t repeats = graph.getConstantInt32(node.child_ids[1])[0];
        int32_t axis = graph.getConstantInt32(node.child_ids[2])[0];
        if (axis < 0)
            axis += sA.size();

        std::vector<Region> inBoxes;
        for (const auto &outReg : outputRegions)
        {
            Region inBox = outReg;
            inBox.region[axis].start = inBox.region[axis].start / repeats;
            inBox.region[axis].stop = (inBox.region[axis].stop + repeats - 1) / repeats;
            inBoxes.push_back(inBox);
        }

        return {mergeRegions(inBoxes), makeFull(graph.getNode(node.child_ids[1]).getShape()),
                makeFull(graph.getNode(node.child_ids[2]).getShape())};
    }

    static WorkloadMetrics computeWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                           const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                           DType outDType, const std::string &)
    {
        return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType, 0.0);
    }

    static bool isConstant(uint64_t inputIdx, uint64_t)
    {
        return inputIdx == 1 || inputIdx == 2;
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.repeat(pInputs[0], pInputs[1], pInputs[2]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};