#pragma once
#include "core/misc.hpp"
#include "core/ops/common.hpp"

struct SliceOp
{
    static constexpr OpType op_type = OpType::SLICE;
    static constexpr const char *name = "SLICE";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        auto s0 = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape();
        auto starts = graph.getConstantInt32(graph.getNode(nodeId).child_ids[1]);
        auto ends = graph.getConstantInt32(graph.getNode(nodeId).child_ids[2]);
        auto steps = graph.getConstantInt32(graph.getNode(nodeId).child_ids[3]);
        std::vector<uint32_t> out_shape(s0.size());
        for (uint64_t i = 0; i < s0.size(); ++i)
        {
            int32_t start = i < starts.size() ? starts[i] : 0;
            int32_t end = i < ends.size() ? ends[i] : s0[i];
            int32_t step = i < steps.size() ? steps[i] : 1;
            if (step == 0)
            {
                Error::throw_err("[ShapePropagator.inferShape] SLICE step cannot be zero.");
            }
            if (start < 0)
                start += s0[i];
            if (end < 0)
                end += s0[i];
            start = std::clamp<int32_t>(start, 0, s0[i]);
            end = std::clamp<int32_t>(end, 0, s0[i]);
            out_shape[i] = std::max(0, (end - start + step - 1) / step);
            if (out_shape[i] == 0)
            {
                Error::throw_err("Zero-sized dimension in tensor shape!" + toString(graph.getNode(nodeId), graph, ""));
            }
        }
        graph.getNode(nodeId).setShape(out_shape);

        auto parentStrides = graph.getNode(graph.getNode(nodeId).child_ids[0]).strides;
        for (uint64_t i = 0; i < s0.size(); ++i)
        {
            int32_t start = i < starts.size() ? starts[i] : 0;
            int32_t step = i < steps.size() ? steps[i] : 1;
            if (start < 0)
                start += s0[i];
            start = std::clamp<int32_t>(start, 0, s0[i]);
            graph.getNode(nodeId).strides[i] = parentStrides[i] * step;
        }
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &shape = graph.getNode(node.child_ids[0]).getShape();
        auto starts = graph.getConstantInt32(node.child_ids[1]);
        auto ends = graph.getConstantInt32(node.child_ids[2]);
        auto steps = graph.getConstantInt32(node.child_ids[3]);

        std::vector<Region> outBoxes;
        for (const auto &region : rA)
            outBoxes.push_back(mapSliceRegionForward(region, shape, starts, ends, steps));
        return mergeRegions(outBoxes);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &graph,
                                                           const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}, {}};

        const auto &shape = graph.getNode(node.child_ids[0]).getShape();
        auto starts = graph.getConstantInt32(node.child_ids[1]);
        auto ends = graph.getConstantInt32(node.child_ids[2]);
        auto steps = graph.getConstantInt32(node.child_ids[3]);

        std::vector<Region> inBoxes;
        for (const auto &region : outputRegions)
            inBoxes.push_back(mapSliceRegionBackward(region, shape, starts, ends, steps));

        return {mergeRegions(inBoxes), makeFull(graph.getNode(node.child_ids[1]).getShape()),
                makeFull(graph.getNode(node.child_ids[2]).getShape()),
                makeFull(graph.getNode(node.child_ids[3]).getShape())};
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
        return pGraph.slice(pInputs[0], pInputs[1], pInputs[2], pInputs[3]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};