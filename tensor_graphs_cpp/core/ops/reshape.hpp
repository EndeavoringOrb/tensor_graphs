#pragma once
#include "core/misc.hpp"
#include "core/ops/common.hpp"

struct ReshapeOp
{
    static constexpr OpType op_type = OpType::RESHAPE;
    static constexpr const char *name = "RESHAPE";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        auto s0 = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape();
        auto target_dims = graph.getConstantInt32(graph.getNode(nodeId).child_ids[1]);
        uint64_t total_vol = countElements(s0);
        uint64_t known_vol = 1;
        for (uint64_t i = 0; i < target_dims.size(); ++i)
        {
            if (target_dims[i] != -1)
                known_vol *= target_dims[i];
        }
        std::vector<uint32_t> out_shape(target_dims.size());
        for (uint64_t i = 0; i < target_dims.size(); ++i)
        {
            if (target_dims[i] == -1)
                out_shape[i] = total_vol / known_vol;
            else
                out_shape[i] = target_dims[i];
        }
        graph.getNode(nodeId).setShape(out_shape);
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &sA = graph.getNode(node.child_ids[0]).getShape();
        const auto &outShape = node.getShape();

        std::vector<Region> outBoxes;
        for (const auto &reg : rA)
        {
            uint64_t old_vol = 1;
            for (const auto &d : reg.region)
            {
                old_vol *= (d.stop - d.start);
            }

            uint32_t rank = static_cast<uint32_t>(sA.size());

            if (rank >= 64)
            {
                uint64_t flat_start, flat_stop;
                getFlatBounds(reg, sA, flat_start, flat_stop);
                outBoxes.push_back(unravelFlatBounds(flat_start, flat_stop, outShape));
                continue;
            }

            std::vector<uint64_t> strides(rank, 1);
            for (int i = static_cast<int>(rank) - 2; i >= 0; --i)
            {
                strides[i] = strides[i + 1] * sA[i + 1];
            }

            std::vector<uint32_t> min_coords(outShape.size(), UINT32_MAX);
            std::vector<uint32_t> max_coords(outShape.size(), 0);

            uint64_t num_corners = 1ULL << rank;
            for (uint64_t i = 0; i < num_corners; ++i)
            {
                uint64_t flat_idx = 0;
                for (uint32_t d = 0; d < rank; ++d)
                {
                    uint32_t coord = ((i >> d) & 1) ? (reg.region[d].stop - 1) : reg.region[d].start;
                    flat_idx += coord * strides[d];
                }

                uint64_t temp = flat_idx;
                for (int d = static_cast<int>(outShape.size()) - 1; d >= 0; --d)
                {
                    uint32_t c = temp % outShape[d];
                    temp /= outShape[d];
                    if (c < min_coords[d])
                        min_coords[d] = c;
                    if (c > max_coords[d])
                        max_coords[d] = c;
                }
            }

            uint64_t new_vol = 1;
            Region exact_box;
            for (uint64_t d = 0; d < outShape.size(); ++d)
            {
                exact_box.region.push_back({min_coords[d], max_coords[d] + 1});
                new_vol *= (max_coords[d] + 1 - min_coords[d]);
            }

            if (new_vol == old_vol && old_vol > 0)
            {
                outBoxes.push_back(exact_box);
            }
            else
            {
                uint64_t flat_start, flat_stop;
                getFlatBounds(reg, sA, flat_start, flat_stop);
                outBoxes.push_back(unravelFlatBounds(flat_start, flat_stop, outShape));
            }
        }
        return mergeRegions(outBoxes);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &graph,
                                                           const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};
        const auto &sA = graph.getNode(node.child_ids[0]).getShape();
        const auto &sShape = graph.getNode(node.child_ids[1]).getShape();

        std::vector<Region> inBoxes;
        for (const auto &reg : outputRegions)
        {
            uint64_t out_vol = 1;
            for (const auto &d : reg.region)
            {
                out_vol *= (d.stop - d.start);
            }

            uint32_t rank = static_cast<uint32_t>(node.getShape().size());

            if (rank >= 64)
            {
                uint64_t flat_start, flat_stop;
                getFlatBounds(reg, node.getShape(), flat_start, flat_stop);
                inBoxes.push_back(unravelFlatBounds(flat_start, flat_stop, sA));
                continue;
            }

            std::vector<uint64_t> strides(rank, 1);
            for (int i = static_cast<int>(rank) - 2; i >= 0; --i)
            {
                strides[i] = strides[i + 1] * node.getShape()[i + 1];
            }

            std::vector<uint32_t> min_coords(sA.size(), UINT32_MAX);
            std::vector<uint32_t> max_coords(sA.size(), 0);

            uint64_t num_corners = 1ULL << rank;
            for (uint64_t i = 0; i < num_corners; ++i)
            {
                uint64_t flat_idx = 0;
                for (uint32_t d = 0; d < rank; ++d)
                {
                    uint32_t coord = ((i >> d) & 1) ? (reg.region[d].stop - 1) : reg.region[d].start;
                    flat_idx += coord * strides[d];
                }

                uint64_t temp = flat_idx;
                for (int d = static_cast<int>(sA.size()) - 1; d >= 0; --d)
                {
                    uint32_t c = temp % sA[d];
                    temp /= sA[d];
                    if (c < min_coords[d])
                        min_coords[d] = c;
                    if (c > max_coords[d])
                        max_coords[d] = c;
                }
            }

            uint64_t new_vol = 1;
            Region exact_box;
            for (uint64_t d = 0; d < sA.size(); ++d)
            {
                exact_box.region.push_back({min_coords[d], max_coords[d] + 1});
                new_vol *= (max_coords[d] + 1 - min_coords[d]);
            }

            if (new_vol == out_vol && out_vol > 0)
            {
                inBoxes.push_back(exact_box);
            }
            else
            {
                uint64_t flat_start, flat_stop;
                getFlatBounds(reg, node.getShape(), flat_start, flat_stop);
                inBoxes.push_back(unravelFlatBounds(flat_start, flat_stop, sA));
            }
        }
        return {mergeRegions(inBoxes), makeFull(sShape)};
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
        return pGraph.reshape(pInputs[0], pInputs[1]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};