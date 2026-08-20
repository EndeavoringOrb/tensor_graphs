#pragma once
#include "core/misc.hpp"
#include "core/ops/common.hpp"

struct ArgmaxOp
{
    static constexpr OpType op_type = OpType::ARGMAX;
    static constexpr const char *name = "ARGMAX";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        auto s0 = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape();
        auto axis_vec = graph.getConstantInt32(graph.getNode(nodeId).child_ids[1]);
        int32_t axis = axis_vec[0];
        if (axis < 0)
            axis += s0.size();

        auto k_vec = graph.getConstantInt32(graph.getNode(nodeId).child_ids[2]);
        int32_t k = k_vec[0];

        std::vector<uint32_t> new_shape;
        for (uint64_t i = 0; i < s0.size(); ++i)
        {
            if (i == static_cast<uint64_t>(axis))
                new_shape.push_back(static_cast<uint32_t>(k));
            else
                new_shape.push_back(s0[i]);
        }
        graph.getNode(nodeId).setShape(new_shape);
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &sA = graph.getNode(node.child_ids[0]).getShape();
        int32_t axis = graph.getConstantInt32(node.child_ids[1])[0];
        if (axis < 0)
            axis += sA.size();

        int32_t k = graph.getConstantInt32(node.child_ids[2])[0];

        std::vector<Region> outBoxes;
        for (const auto &inReg : rA)
        {
            Region outBox;
            for (uint64_t d = 0; d < sA.size(); ++d)
            {
                if (static_cast<int32_t>(d) == axis)
                {
                    if (inReg.region[d].start < inReg.region[d].stop)
                        outBox.region.push_back({0, static_cast<uint32_t>(k)});
                    else
                        outBox.region.push_back({0, 0});
                }
                else
                {
                    outBox.region.push_back(inReg.region[d]);
                }
            }
            outBoxes.push_back(outBox);
        }
        return mergeRegions(outBoxes);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &graph,
                                                           const std::vector<Region> &outputRegions)
    {
        std::vector<std::vector<Region>> res(3);
        if (outputRegions.empty())
            return res;

        const auto &sA = graph.getNode(node.child_ids[0]).getShape();
        int32_t axis = graph.getConstantInt32(node.child_ids[1])[0];
        if (axis < 0)
            axis += sA.size();

        std::vector<Region> inBoxes;
        for (const auto &outReg : outputRegions)
        {
            Region inBox;
            for (uint64_t d = 0; d < sA.size(); ++d)
            {
                if (static_cast<int32_t>(d) == axis)
                {
                    if (outReg.region[d].start < outReg.region[d].stop)
                        inBox.region.push_back({0, sA[d]});
                    else
                        inBox.region.push_back({0, 0});
                }
                else
                {
                    inBox.region.push_back(outReg.region[d]);
                }
            }
            inBoxes.push_back(inBox);
        }
        res[0] = mergeRegions(inBoxes);
        res[1] = makeFull(graph.getNode(node.child_ids[1]).getShape());
        res[2] = makeFull(graph.getNode(node.child_ids[2]).getShape());
        return res;
    }

    static WorkloadMetrics computeWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                           const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                           DType outDType, const std::string &)
    {
        double inElements = inShapes.empty() ? 0.0 : static_cast<double>(countElements(inShapes[0]));
        return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType, inElements);
    }

    static bool isConstant(uint64_t inputIdx, uint64_t)
    {
        return inputIdx == 1 || inputIdx == 2;
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.argmax(pInputs[0], pInputs[1], pInputs[2]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};