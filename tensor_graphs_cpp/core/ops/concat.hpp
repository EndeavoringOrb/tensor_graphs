#pragma once
#include "core/misc.hpp"
#include "core/ops/common.hpp"

struct ConcatOp
{
    static constexpr OpType op_type = OpType::CONCAT;
    static constexpr const char *name = "CONCAT";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        LogicalId axis_id = graph.getNode(nodeId).child_ids[0];
        auto axis_vec = graph.getConstantInt32(axis_id);
        int32_t axis = axis_vec[0];
        auto s0 = graph.getNode(graph.getNode(nodeId).child_ids[1]).getShape();
        if (axis < 0)
            axis += s0.size();

        std::vector<uint32_t> out_shape = s0;
        uint32_t total_dim = s0[axis];
        for (uint64_t i = 2; i < graph.getNode(nodeId).child_ids.size(); ++i)
        {
            auto si = graph.getNode(graph.getNode(nodeId).child_ids[i]).getShape();
            total_dim += si[axis];
        }
        out_shape[axis] = total_dim;
        graph.getNode(nodeId).setShape(out_shape);
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        bool allClean = true;
        for (uint64_t i = 1; i < parentRegions.size(); ++i)
        {
            if (!parentRegions[i].empty())
            {
                allClean = false;
                break;
            }
        }
        if (allClean)
            return {};

        int32_t axis = graph.getConstantInt32(node.child_ids[0])[0];
        uint32_t rank = node.getShape().size();
        if (axis < 0)
            axis += rank;
        std::vector<Region> outBoxes;

        uint32_t current_offset = 0;
        for (uint64_t i = 1; i < node.child_ids.size(); ++i)
        {
            const auto &pShape = graph.getNode(node.child_ids[i]).getShape();
            const auto &pReg = parentRegions[i];
            for (const auto &region : pReg)
            {
                Region shifted = region;
                if (shifted.region.size() > static_cast<uint64_t>(axis))
                {
                    shifted.region[axis].start += current_offset;
                    shifted.region[axis].stop += current_offset;
                    outBoxes.push_back(shifted);
                }
            }
            current_offset += pShape[axis];
        }

        return mergeRegions(outBoxes);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &graph,
                                                           const std::vector<Region> &outputRegions)
    {
        std::vector<std::vector<Region>> res(node.child_ids.size());
        if (outputRegions.empty())
            return res;

        int32_t axis = graph.getConstantInt32(node.child_ids[0])[0];
        uint32_t rank = node.getShape().size();
        if (axis < 0)
            axis += rank;

        uint32_t current_offset = 0;
        for (uint64_t i = 1; i < node.child_ids.size(); ++i)
        {
            const auto &pShape = graph.getNode(node.child_ids[i]).getShape();
            uint32_t in_dim = pShape[axis];
            uint32_t in_end = current_offset + in_dim;

            std::vector<Region> inBoxes;
            for (const auto &outReg : outputRegions)
            {
                if (outReg.region.size() <= static_cast<uint64_t>(axis))
                    continue;
                uint32_t ov_start = std::max(outReg.region[axis].start, current_offset);
                uint32_t ov_stop = std::min(outReg.region[axis].stop, in_end);

                Region inBox = outReg;
                if (ov_start >= ov_stop)
                    continue;

                inBox.region[axis].start = ov_start - current_offset;
                inBox.region[axis].stop = ov_stop - current_offset;
                inBoxes.push_back(inBox);
            }
            res[i] = mergeRegions(inBoxes);
            current_offset = in_end;
        }
        res[0] = makeFull(graph.getNode(node.child_ids[0]).getShape());
        return res;
    }

    static WorkloadMetrics computeWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                           const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                           DType outDType, const std::string &)
    {
        return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType, 0.0);
    }

    static bool isConstant(uint64_t inputIdx, uint64_t)
    {
        return inputIdx == 0;
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        std::vector<LogicalId> concatIns;
        for (uint64_t i = 1; i < pInputs.size(); ++i)
            concatIns.push_back(pInputs[i]);
        return pGraph.concat(concatIns, pInputs[0]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};