#pragma once
#include "core/misc.hpp"
#include "core/ops/common.hpp"

struct DotOp
{
    static constexpr OpType op_type = OpType::DOT;
    static constexpr const char *name = "DOT";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        const auto &s0 = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape();
        const auto &s1 = graph.getNode(graph.getNode(nodeId).child_ids[1]).getShape();
        uint64_t r0 = s0.size();
        uint64_t r1 = s1.size();

        if (r0 != r1)
        {
            std::stringstream ss;
            ss << "[ShapePropagator.inferShape] nodeId=" + toString(nodeId) + " DOT requires equal ranks. Got " << r0
               << " (" + toString(s0) + ") and " << r1
               << " (" + toString(s1) +
                      "). Implicit broadcasting is not supported; use explicit reshape to align ranks. debugOrigin=" +
                      graph.getNode(nodeId).debugOrigin;
            Error::throw_err(ss.str());
        }

        if (r0 == 2)
        {
            if (s0[1] != s1[0])
                Error::throw_err("DOT: K-dim mismatch [M,K] @ [K,N]");
            graph.getNode(nodeId).setShape({s0[0], s1[1]});
        }
        else if (r0 == 3)
        {
            if (s0[2] != s1[1])
                Error::throw_err("DOT: K-dim mismatch [B,M,K] @ [B,K,N], " + std::to_string(s0[2]) +
                                 " != " + std::to_string(s1[1]));
            graph.getNode(nodeId).setShape({s0[0], s0[1], s1[2]});
        }
        else if (r0 == 4)
        {
            if (s0[0] != s1[0] || s0[1] != s1[1] || s0[3] != s1[2])
            {
                Error::throw_err("DOT 4D: Dimension mismatch [B,H,M,K] @ [B,H,K,N], " + std::to_string(s0[3]) +
                                 " != " + std::to_string(s1[2]));
            }
            graph.getNode(nodeId).setShape({s0[0], s0[1], s0[2], s1[3]});
        }
        else
        {
            Error::throw_err("DOT: Only Rank 2, 3, and 4 are currently supported. Got r0=" + std::to_string(r0) +
                             ", r1=" + std::to_string(r1));
        }
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        const auto &rB = parentRegions[1];
        if (rA.empty() && rB.empty())
            return {};

        const auto &sA = graph.getNode(node.child_ids[0]).getShape();
        const auto &sB = graph.getNode(node.child_ids[1]).getShape();
        const auto &outShape = node.getShape();

        std::vector<Region> outBoxes;

        for (const auto &box : rA)
        {
            Region outBox;
            if (sA.size() == 4)
            {
                outBox.region.push_back(box.region[0]);    // B
                outBox.region.push_back(box.region[1]);    // H
                outBox.region.push_back(box.region[2]);    // M
                outBox.region.push_back({0, outShape[3]}); // N (Full row needed)
            }
            else if (sA.size() == 3)
            {
                outBox.region.push_back(box.region[0]);    // B
                outBox.region.push_back(box.region[1]);    // M
                outBox.region.push_back({0, outShape[2]}); // N
            }
            else
            {
                outBox.region.push_back(box.region[0]);    // M
                outBox.region.push_back({0, outShape[1]}); // N
            }
            outBoxes.push_back(outBox);
        }

        for (const auto &box : rB)
        {
            Region outBox;
            if (sB.size() == 4)
            {
                outBox.region.push_back(box.region[0]);    // B
                outBox.region.push_back(box.region[1]);    // H
                outBox.region.push_back({0, outShape[2]}); // M
                outBox.region.push_back(box.region[3]);    // N
            }
            else if (sB.size() == 3)
            {
                outBox.region.push_back(box.region[0]);    // B
                outBox.region.push_back({0, outShape[1]}); // M
                outBox.region.push_back(box.region[2]);    // N
            }
            else
            {
                outBox.region.push_back({0, outShape[0]}); // M
                outBox.region.push_back(box.region[1]);    // N
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

        const auto &sA = graph.getNode(node.child_ids[0]).getShape();
        const auto &sB = graph.getNode(node.child_ids[1]).getShape();

        std::vector<Region> reqA, reqB;
        for (const auto &outBox : outputRegions)
        {
            Region aBox, bBox;
            if (sA.size() == 4)
            {
                aBox.region = {outBox.region[0], outBox.region[1], outBox.region[2], {0, sA[3]}};
                bBox.region = {outBox.region[0], outBox.region[1], {0, sB[2]}, outBox.region[3]};
            }
            else if (sA.size() == 3)
            {
                aBox.region.push_back(outBox.region[0]); // B
                aBox.region.push_back(outBox.region[1]); // M
                aBox.region.push_back({0, sA[2]});       // K

                bBox.region.push_back(outBox.region[0]); // B
                bBox.region.push_back({0, sB[1]});       // K
                bBox.region.push_back(outBox.region[2]); // N
            }
            else
            {
                aBox.region.push_back(outBox.region[0]); // M
                aBox.region.push_back({0, sA[1]});       // K

                bBox.region.push_back({0, sB[0]});       // K
                bBox.region.push_back(outBox.region[1]); // N
            }
            reqA.push_back(aBox);
            reqB.push_back(bBox);
        }
        return {mergeRegions(reqA), mergeRegions(reqB)};
    }

    static WorkloadMetrics computeWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                           const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                           DType outDType, const std::string &)
    {
        double kDim = inShapes.empty() || inShapes[0].empty() ? 1.0 : static_cast<double>(inShapes[0].back());
        double flops = 2.0 * static_cast<double>(countElements(outShape)) * kDim;
        return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType, flops);
    }

    static bool isConstant(uint64_t, uint64_t)
    {
        return false;
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.dot(pInputs[0], pInputs[1]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};