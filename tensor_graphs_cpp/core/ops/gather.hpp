#pragma once
#include "core/misc.hpp"
#include "core/ops/common.hpp"

struct GatherOp
{
    static constexpr OpType op_type = OpType::GATHER;
    static constexpr const char *name = "GATHER";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        auto data_shape = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape();
        auto idx_shape = graph.getNode(graph.getNode(nodeId).child_ids[1]).getShape();
        std::vector<uint32_t> out_shape = idx_shape;
        for (uint64_t i = 1; i < data_shape.size(); ++i)
        {
            out_shape.push_back(data_shape[i]);
        }
        graph.getNode(nodeId).setShape(out_shape);
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &dataReg = parentRegions[0];
        const auto &idxReg = parentRegions[1];
        if (dataReg.empty() && idxReg.empty())
            return {};

        const auto &dataShape = graph.getNode(node.child_ids[0]).getShape();
        const auto &idxShape = graph.getNode(node.child_ids[1]).getShape();
        const auto &outShape = node.getShape();

        uint32_t idxRank = static_cast<uint32_t>(idxShape.size());
        std::vector<Region> outBoxes;

        for (const auto &idxBox : idxReg)
        {
            for (const auto &dataBox : dataReg)
            {
                Region outBox;
                for (uint32_t d = 0; d < idxRank; ++d)
                {
                    if (idxBox.region.size() > d)
                        outBox.region.push_back(idxBox.region[d]);
                    else if (dataBox.region.size() > d)
                        outBox.region.push_back(dataBox.region[d]);
                    else
                        outBox.region.push_back({0, outShape[d]});
                }

                for (uint32_t d = 1; d < dataShape.size(); ++d)
                {
                    uint32_t out_d = idxRank + d - 1;
                    if (dataBox.region.size() > d)
                        outBox.region.push_back(dataBox.region[d]);
                    else if (out_d < outShape.size())
                        outBox.region.push_back({0, outShape[out_d]});
                }

                if (isValidRegion(outBox))
                    outBoxes.push_back(outBox);
            }
        }

        if (outBoxes.empty() && !dataReg.empty())
        {
            for (const auto &dataBox : dataReg)
            {
                Region outBox;
                for (uint32_t d = 0; d < idxRank; ++d)
                    outBox.region.push_back({0, outShape[d]});
                for (uint32_t d = 1; d < dataShape.size(); ++d)
                    outBox.region.push_back(dataBox.region.size() > d ? dataBox.region[d]
                                                                      : Dim{0, outShape[idxRank + d - 1]});
                if (isValidRegion(outBox))
                    outBoxes.push_back(outBox);
            }
        }

        if (outBoxes.empty() && !idxReg.empty())
        {
            for (const auto &idxBox : idxReg)
            {
                Region outBox;
                for (uint32_t d = 0; d < idxRank; ++d)
                    outBox.region.push_back(idxBox.region.size() > d ? idxBox.region[d] : Dim{0, outShape[d]});
                for (uint32_t d = 1; d < dataShape.size(); ++d)
                    outBox.region.push_back({0, outShape[idxRank + d - 1]});
                if (isValidRegion(outBox))
                    outBoxes.push_back(outBox);
            }
        }

        return mergeRegions(outBoxes);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &graph,
                                                           const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        const auto &dataShape = graph.getNode(node.child_ids[0]).getShape();
        const auto &idxShape = graph.getNode(node.child_ids[1]).getShape();
        uint32_t idxRank = static_cast<uint32_t>(idxShape.size());
        std::vector<Region> dataBoxes;
        std::vector<Region> idxBoxes;

        std::vector<int32_t> idxValues = graph.getConstantInt32(node.child_ids[1]);
        bool exactIdxValues = countElements(idxShape) == idxValues.size();

        for (const auto &outReg : outputRegions)
        {
            Region idxBox;
            for (uint32_t d = 0; d < idxRank; ++d)
                idxBox.region.push_back(outReg.region[d]);
            idxBoxes.push_back(idxBox);

            if (!exactIdxValues)
            {
                Region dataBox;
                dataBox.region.push_back({0, dataShape[0]});
                for (uint32_t d = 1; d < dataShape.size(); ++d)
                {
                    uint32_t out_d = idxRank + d - 1;
                    if (out_d < outReg.region.size())
                        dataBox.region.push_back(outReg.region[out_d]);
                    else
                        dataBox.region.push_back({0, dataShape[d]});
                }
                dataBoxes.push_back(dataBox);
                continue;
            }

            std::vector<uint32_t> regionShape;
            regionShape.reserve(outReg.region.size());
            for (const auto &dim : outReg.region)
                regionShape.push_back(dim.stop - dim.start);

            uint64_t regionCount = countElements(regionShape);
            for (uint64_t localFlat = 0; localFlat < regionCount; ++localFlat)
            {
                auto localCoords = coordsFromFlatIndex(localFlat, regionShape);
                std::vector<uint32_t> idxCoords(idxRank, 0);
                for (uint32_t d = 0; d < idxRank; ++d)
                    idxCoords[d] = outReg.region[d].start + localCoords[d];

                uint64_t idxFlat = flatIndexFromCoords(idxCoords, idxShape);
                if (idxFlat >= idxValues.size())
                    continue;

                int32_t idxValue = idxValues[static_cast<uint64_t>(idxFlat)];
                if (idxValue < 0 || static_cast<uint32_t>(idxValue) >= dataShape[0])
                    continue;

                Region dataBox;
                dataBox.region.push_back({static_cast<uint32_t>(idxValue), static_cast<uint32_t>(idxValue + 1)});
                for (uint32_t d = 1; d < dataShape.size(); ++d)
                {
                    uint32_t out_d = idxRank + d - 1;
                    if (out_d < outReg.region.size())
                        dataBox.region.push_back(outReg.region[out_d]);
                    else
                        dataBox.region.push_back({0, dataShape[d]});
                }
                dataBoxes.push_back(dataBox);
            }
        }

        if (dataBoxes.empty())
        {
            for (const auto &outReg : outputRegions)
            {
                Region dataBox;
                dataBox.region.push_back({0, dataShape[0]});
                for (uint32_t d = 1; d < dataShape.size(); ++d)
                {
                    uint32_t out_d = idxRank + d - 1;
                    if (out_d < outReg.region.size())
                        dataBox.region.push_back(outReg.region[out_d]);
                    else
                        dataBox.region.push_back({0, dataShape[d]});
                }
                dataBoxes.push_back(dataBox);
            }
        }

        return {mergeRegions(dataBoxes), mergeRegions(idxBoxes)};
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

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.gather(pInputs[0], pInputs[1]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};