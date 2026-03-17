#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include <cstring>
#include <algorithm>

inline std::vector<int32_t> getConstantInt32(uint32_t id, const Graph &graph)
{
    if (graph.constantStaging.count(id))
    {
        const auto &data = graph.constantStaging.at(id);
        std::vector<int32_t> res(data.size() / sizeof(int32_t));
        std::memcpy(res.data(), data.data(), data.size());
        return res;
    }
    std::stringstream ss;
    ss << "Expected constant for shape inference but not found in staging. Node ID: " << id;
    Error::throw_err(ss.str());
}

inline std::vector<uint32_t> broadcastShapes(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b)
{
    int rankA = a.size();
    int rankB = b.size();
    int outRank = std::max(rankA, rankB);
    std::vector<uint32_t> out(outRank);
    for (int i = 0; i < outRank; ++i)
    {
        uint32_t dimA = (i < outRank - rankA) ? 1 : a[i - (outRank - rankA)];
        uint32_t dimB = (i < outRank - rankB) ? 1 : b[i - (outRank - rankB)];
        if (dimA == 1)
            out[i] = dimB;
        else if (dimB == 1)
            out[i] = dimA;
        else if (dimA == dimB)
            out[i] = dimA;
        else
        {
            std::stringstream ss;
            ss << "Cannot broadcast shapes " << toString(a) << " and " << toString(b);
            Error::throw_err(ss.str());
        }
    }
    return out;
}

inline std::vector<Region> makeFull(const std::vector<uint32_t> &shape)
{
    if (shape.empty())
        return {};
    Region r;
    for (uint32_t d : shape)
    {
        r.region.push_back({0, d});
    }
    return {r};
}

inline void getFlatBounds(const Region &region, const std::vector<uint32_t> &shape, uint64_t &flat_start, uint64_t &flat_stop)
{
    std::vector<uint64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    flat_start = 0;
    uint64_t flat_stop_minus_1 = 0;
    for (size_t i = 0; i < region.region.size(); ++i)
    {
        flat_start += region.region[i].start * strides[i];
        flat_stop_minus_1 += (region.region[i].stop - 1) * strides[i];
    }
    flat_stop = flat_stop_minus_1 + 1;
}

inline Region unravelFlatBounds(uint64_t flat_start, uint64_t flat_stop, const std::vector<uint32_t> &shape)
{
    uint64_t temp_start = flat_start;
    uint64_t temp_stop = flat_stop - 1;

    std::vector<uint32_t> coords_start(shape.size());
    std::vector<uint32_t> coords_stop(shape.size());

    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        coords_start[i] = temp_start % shape[i];
        temp_start /= shape[i];
        coords_stop[i] = temp_stop % shape[i];
        temp_stop /= shape[i];
    }

    Region region;
    bool found_diff = false;
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (found_diff)
        {
            region.region.push_back({0, shape[i]});
        }
        else
        {
            if (coords_start[i] < coords_stop[i])
            {
                region.region.push_back({coords_start[i], coords_stop[i] + 1});
                found_diff = true;
            }
            else
            {
                region.region.push_back({coords_start[i], coords_start[i] + 1});
            }
        }
    }
    return region;
}

struct ShapePropagator
{
    void inferShapeRecursive(uint32_t nodeId, Graph &graph)
    {
        if (nodeId >= graph.nodes.size())
            return;

        if (!graph.nodes[nodeId].shape.empty() && graph.nodes[nodeId].opType != OpType::RESHAPE)
            return;

        if (graph.nodes[nodeId].opType == OpType::INPUT)
            return;

        for (uint32_t pid : graph.nodes[nodeId].parentIds)
        {
            inferShapeRecursive(pid, graph);
        }

        inferShape(nodeId, graph);
    }

    void inferShape(uint32_t nodeId, Graph &graph)
    {
        if (!graph.nodes[nodeId].shape.empty() && graph.nodes[nodeId].opType != OpType::RESHAPE)
            return;
        if (graph.nodes[nodeId].opType == OpType::INPUT)
            return;

        switch (graph.nodes[nodeId].opType)
        {
        case OpType::ADD:
        case OpType::MUL:
        case OpType::DIVIDE:
        case OpType::POWER:
        {
            auto s0 = graph.nodes[graph.nodes[nodeId].parentIds[0]].shape;
            auto s1 = graph.nodes[graph.nodes[nodeId].parentIds[1]].shape;
            if (s0 != s1)
            {
                std::stringstream ss;
                ss << "[ShapePropagator.inferShape] Atomic " << toString(graph.nodes[nodeId].opType)
                   << " requires exact shape match. Got " << toString(s0)
                   << " and " << toString(s1) << ". Use explicit repeat/reshape. (Node " << graph.nodes[nodeId].id << ")";
                Error::throw_err(ss.str());
            }
            graph.nodes[nodeId].shape = s0;
            break;
        }
        case OpType::DOT:
        {
            const auto &s0 = graph.nodes[graph.nodes[nodeId].parentIds[0]].shape;
            const auto &s1 = graph.nodes[graph.nodes[nodeId].parentIds[1]].shape;
            size_t r0 = s0.size();
            size_t r1 = s1.size();

            if (r0 != r1)
            {
                std::stringstream ss;
                ss << "[ShapePropagator.inferShape] DOT requires equal ranks. Got " << r0 << " and " << r1
                   << ". Implicit broadcasting is disabled; use explicit reshape to align ranks.";
                Error::throw_err(ss.str());
            }

            if (r0 == 2)
            {
                if (s0[1] != s1[0])
                    Error::throw_err("DOT: K-dim mismatch [M,K] @ [K,N]");
                graph.nodes[nodeId].shape = {s0[0], s1[1]};
            }
            else if (r0 == 3)
            {
                if (s0[0] != s1[0])
                    Error::throw_err("DOT: Batch dim mismatch [B,M,K] @ [B,K,N]");
                if (s0[2] != s1[1])
                    Error::throw_err("DOT: K-dim mismatch [B,M,K] @ [B,K,N]");
                graph.nodes[nodeId].shape = {s0[0], s0[1], s1[2]};
            }
            else
            {
                Error::throw_err("DOT: Only Rank 2 and Rank 3 are currently supported in this framework.");
            }
            break;
        }
        case OpType::SIN:
        case OpType::COS:
        case OpType::NEGATE:
        case OpType::CAST:
        case OpType::TRIU:
        case OpType::COPY_TO:
        case OpType::CONTIGUOUS:
        case OpType::SCATTER:
        {
            graph.nodes[nodeId].shape = graph.nodes[graph.nodes[nodeId].parentIds[0]].shape;
            break;
        }
        case OpType::SUM:
        case OpType::MAX:
        {
            auto s0 = graph.nodes[graph.nodes[nodeId].parentIds[0]].shape;
            auto axis_vec = getConstantInt32(graph.nodes[nodeId].parentIds[1], graph);
            int32_t axis = axis_vec[0];
            if (axis < 0)
                axis += s0.size();

            std::vector<uint32_t> new_shape;
            for (size_t i = 0; i < s0.size(); ++i)
            {
                if (i == (size_t)axis)
                    new_shape.push_back(1);
                else
                    new_shape.push_back(s0[i]);
            }
            graph.nodes[nodeId].shape = new_shape;
            break;
        }
        case OpType::RESHAPE:
        {
            auto s0 = graph.nodes[graph.nodes[nodeId].parentIds[0]].shape;
            auto target_dims = getConstantInt32(graph.nodes[nodeId].parentIds[1], graph);
            uint64_t total_vol = countElements(s0);
            uint64_t known_vol = 1;
            for (size_t i = 0; i < target_dims.size(); ++i)
            {
                if (target_dims[i] != -1)
                    known_vol *= target_dims[i];
            }
            std::vector<uint32_t> out_shape(target_dims.size());
            for (size_t i = 0; i < target_dims.size(); ++i)
            {
                if (target_dims[i] == -1)
                    out_shape[i] = total_vol / known_vol;
                else
                    out_shape[i] = target_dims[i];
            }
            graph.nodes[nodeId].shape = out_shape;
            break;
        }
        case OpType::PERMUTE:
        {
            auto s0 = graph.nodes[graph.nodes[nodeId].parentIds[0]].shape;
            auto dims = getConstantInt32(graph.nodes[nodeId].parentIds[1], graph);
            std::vector<uint32_t> out_shape(dims.size());
            for (size_t i = 0; i < dims.size(); ++i)
            {
                out_shape[i] = s0[dims[i]];
            }
            graph.nodes[nodeId].shape = out_shape;
            break;
        }
        case OpType::GATHER:
        {
            auto data_shape = graph.nodes[graph.nodes[nodeId].parentIds[0]].shape;
            auto idx_shape = graph.nodes[graph.nodes[nodeId].parentIds[1]].shape;
            std::vector<uint32_t> out_shape = idx_shape;
            for (size_t i = 1; i < data_shape.size(); ++i)
            {
                out_shape.push_back(data_shape[i]);
            }
            graph.nodes[nodeId].shape = out_shape;
            break;
        }
        case OpType::CONCAT:
        {
            uint32_t axis_id = graph.nodes[nodeId].parentIds.back();
            auto axis_vec = getConstantInt32(axis_id, graph);
            int32_t axis = axis_vec[0];
            auto s0 = graph.nodes[graph.nodes[nodeId].parentIds[0]].shape;
            if (axis < 0)
                axis += s0.size();

            std::vector<uint32_t> out_shape = s0;
            uint32_t total_dim = s0[axis];
            for (size_t i = 1; i < graph.nodes[nodeId].parentIds.size() - 1; ++i)
            {
                auto si = graph.nodes[graph.nodes[nodeId].parentIds[i]].shape;
                total_dim += si[axis];
            }
            out_shape[axis] = total_dim;
            graph.nodes[nodeId].shape = out_shape;
            break;
        }
        case OpType::REPEAT:
        {
            auto s0 = graph.nodes[graph.nodes[nodeId].parentIds[0]].shape;
            auto repeats = getConstantInt32(graph.nodes[nodeId].parentIds[1], graph)[0];
            auto axis = getConstantInt32(graph.nodes[nodeId].parentIds[2], graph)[0];
            if (axis < 0)
                axis += s0.size();
            std::vector<uint32_t> out_shape = s0;
            out_shape[axis] *= repeats;
            graph.nodes[nodeId].shape = out_shape;
            break;
        }
        case OpType::FILL:
        {
            auto target_dims = getConstantInt32(graph.nodes[nodeId].parentIds[1], graph);
            std::vector<uint32_t> out_shape(target_dims.size());
            for (size_t i = 0; i < target_dims.size(); ++i)
            {
                out_shape[i] = target_dims[i];
            }
            graph.nodes[nodeId].shape = out_shape;
            break;
        }
        case OpType::IM2COL:
        {
            auto s0 = graph.nodes[graph.nodes[nodeId].parentIds[0]].shape; // N, C, H, W
            uint32_t k = getConstantInt32(graph.nodes[nodeId].parentIds[1], graph)[0];
            uint32_t s = getConstantInt32(graph.nodes[nodeId].parentIds[2], graph)[0];
            uint32_t p = getConstantInt32(graph.nodes[nodeId].parentIds[3], graph)[0];
            uint32_t H = s0[2];
            uint32_t W = s0[3];
            uint32_t H_out = (H + 2 * p - k) / s + 1;
            uint32_t W_out = (W + 2 * p - k) / s + 1;
            graph.nodes[nodeId].shape = {s0[0], s0[1] * k * k, H_out * W_out};
            break;
        }
        case OpType::SLICE:
        {
            auto s0 = graph.nodes[graph.nodes[nodeId].parentIds[0]].shape;
            auto starts = getConstantInt32(graph.nodes[nodeId].parentIds[1], graph);
            auto ends = getConstantInt32(graph.nodes[nodeId].parentIds[2], graph);
            auto steps = getConstantInt32(graph.nodes[nodeId].parentIds[3], graph);
            std::vector<uint32_t> out_shape(s0.size());
            for (size_t i = 0; i < s0.size(); ++i)
            {
                int32_t start = i < starts.size() ? starts[i] : 0;
                int32_t end = i < ends.size() ? ends[i] : s0[i];
                int32_t step = i < steps.size() ? steps[i] : 1;
                if (start < 0)
                    start += s0[i];
                if (end < 0)
                    end += s0[i];
                out_shape[i] = std::max(0, (end - start + step - 1) / step);
            }
            graph.nodes[nodeId].shape = out_shape;
            break;
        }
        case OpType::ARANGE:
        {
            int32_t start = getConstantInt32(graph.nodes[nodeId].parentIds[0], graph)[0];
            int32_t stop = getConstantInt32(graph.nodes[nodeId].parentIds[1], graph)[0];
            int32_t step = getConstantInt32(graph.nodes[nodeId].parentIds[2], graph)[0];
            graph.nodes[nodeId].shape = {(uint32_t)std::max(0, (stop - start + step - 1) / step)};
            break;
        }
        case OpType::FUSED:
            break; // TODO: we should never hit this, only infer shapes for atomic, fused should inherit from atomic that it is replacing
        default:
            break;
        }
    }

    std::vector<Region> forwardElementwise(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        std::vector<Region> outputRegions;
        auto regionExists = [&](const Region &r)
        {
            for (const auto &existing : outputRegions)
            {
                if (regionsMatch(existing, r))
                    return true;
            }
            return false;
        };
        for (const auto &pr : parentRegions)
        {
            for (const auto &region : pr)
            {
                if (!regionExists(region))
                {
                    outputRegions.push_back(region);
                }
            }
        }
        return outputRegions;
    }

    std::vector<std::vector<Region>> backwardElementwise(const TensorNode &node, const std::vector<Region> &outputRegions)
    {
        std::vector<std::vector<Region>> inputRegions(node.parentIds.size());
        for (size_t i = 0; i < node.parentIds.size(); ++i)
        {
            inputRegions[i] = outputRegions;
        }
        return inputRegions;
    }

    std::vector<Region> forwardFull(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        for (const auto &p : parentRegions)
        {
            if (!p.empty())
                return makeFull(node.shape);
        }
        return {};
    }

    std::vector<std::vector<Region>> backwardFull(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        std::vector<std::vector<Region>> res(node.parentIds.size());
        if (outputRegions.empty())
            return res;
        for (size_t i = 0; i < node.parentIds.size(); ++i)
        {
            res[i] = makeFull(graph.nodes[node.parentIds[i]].shape);
        }
        return res;
    }

    std::vector<Region> forwardDot(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        const auto &rB = parentRegions[1];
        if (rA.empty() && rB.empty())
            return {};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        const auto &sB = graph.nodes[node.parentIds[1]].shape;
        const auto &outShape = node.shape;

        std::vector<Region> outBoxes;

        if (!rA.empty())
        {
            for (const auto &box : rA)
            {
                Region outBox;
                if (sA.size() == 3)
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
        }
        if (!rB.empty())
        {
            for (const auto &box : rB)
            {
                Region outBox;
                if (sB.size() == 3)
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
        }
        return outBoxes;
    }

    std::vector<std::vector<Region>> backwardDot(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        const auto &sB = graph.nodes[node.parentIds[1]].shape;

        std::vector<Region> reqA, reqB;
        for (const auto &outBox : outputRegions)
        {
            Region aBox, bBox;
            if (sA.size() == 3)
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
        return {reqA, reqB};
    }

    std::vector<Region> forwardReshape(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        const auto &outShape = node.shape;

        std::vector<Region> outBoxes;
        for (const auto &reg : rA)
        {
            uint64_t old_vol = 1;
            for (const auto &d : reg.region)
            {
                old_vol *= (d.stop - d.start);
            }

            uint32_t rank = sA.size();

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
            for (size_t d = 0; d < outShape.size(); ++d)
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
        return outBoxes;
    }

    std::vector<std::vector<Region>> backwardReshape(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};
        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        const auto &sShape = graph.nodes[node.parentIds[1]].shape;

        std::vector<Region> inBoxes;
        for (const auto &reg : outputRegions)
        {
            uint64_t out_vol = 1;
            for (const auto &d : reg.region)
            {
                out_vol *= (d.stop - d.start);
            }

            uint32_t rank = node.shape.size();

            if (rank >= 64)
            {
                uint64_t flat_start, flat_stop;
                getFlatBounds(reg, node.shape, flat_start, flat_stop);
                inBoxes.push_back(unravelFlatBounds(flat_start, flat_stop, sA));
                continue;
            }

            std::vector<uint64_t> strides(rank, 1);
            for (int i = static_cast<int>(rank) - 2; i >= 0; --i)
            {
                strides[i] = strides[i + 1] * node.shape[i + 1];
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
            for (size_t d = 0; d < sA.size(); ++d)
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
                getFlatBounds(reg, node.shape, flat_start, flat_stop);
                inBoxes.push_back(unravelFlatBounds(flat_start, flat_stop, sA));
            }
        }
        return {inBoxes, makeFull(sShape)};
    }

    std::vector<Region> forwardReduce(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        int32_t axis = getConstantInt32(node.parentIds[1], graph)[0];
        if (axis < 0)
            axis += sA.size();

        Region outBox;
        for (size_t d = 0; d < sA.size(); ++d)
        {
            if ((int32_t)d == axis)
            {
                if (rA[0].region[d].start < rA[0].region[d].stop)
                {
                    outBox.region.push_back({0, 1});
                }
                else
                {
                    outBox.region.push_back({0, 0});
                }
            }
            else
            {
                outBox.region.push_back(rA[0].region[d]);
            }
        }
        return {outBox};
    }

    std::vector<std::vector<Region>> backwardReduce(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        int32_t axis = getConstantInt32(node.parentIds[1], graph)[0];
        if (axis < 0)
            axis += sA.size();

        Region inBox;
        for (size_t d = 0; d < sA.size(); ++d)
        {
            if ((int32_t)d == axis)
            {
                if (outputRegions[0].region[d].start < outputRegions[0].region[d].stop)
                {
                    inBox.region.push_back({0, sA[d]});
                }
                else
                {
                    inBox.region.push_back({0, 0});
                }
            }
            else
            {
                inBox.region.push_back(outputRegions[0].region[d]);
            }
        }
        return {{inBox}, makeFull(graph.nodes[node.parentIds[1]].shape)};
    }

    std::vector<Region> forwardPermute(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        auto dims = getConstantInt32(node.parentIds[1], graph);

        Region outBox;
        for (int32_t d : dims)
        {
            outBox.region.push_back(rA[0].region[d]);
        }
        return {outBox};
    }

    std::vector<std::vector<Region>> backwardPermute(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        auto dims = getConstantInt32(node.parentIds[1], graph);
        std::vector<int32_t> invDims(dims.size());
        for (size_t i = 0; i < dims.size(); ++i)
        {
            invDims[dims[i]] = i;
        }

        Region inBox;
        for (int32_t d : invDims)
        {
            inBox.region.push_back(outputRegions[0].region[d]);
        }
        return {{inBox}, makeFull(graph.nodes[node.parentIds[1]].shape)};
    }

    std::vector<Region> forwardGather(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &dataReg = parentRegions[0];
        const auto &idxReg = parentRegions[1];
        if (dataReg.empty() && idxReg.empty())
            return {};

        const auto &dataShape = graph.nodes[node.parentIds[0]].shape;
        const auto &idxShape = graph.nodes[node.parentIds[1]].shape;
        const auto &outShape = node.shape;

        Region outBox;
        uint32_t idxRank = idxShape.size();

        for (uint32_t d = 0; d < idxRank; ++d)
        {
            if (!idxReg.empty() && idxReg[0].region.size() > d)
            {
                outBox.region.push_back(idxReg[0].region[d]);
            }
            else if (!dataReg.empty())
            {
                outBox.region.push_back({0, outShape[d]});
            }
            else
            {
                outBox.region.push_back({0, 0});
            }
        }

        for (uint32_t d = 1; d < dataShape.size(); ++d)
        {
            uint32_t out_d = idxRank + d - 1;
            if (!dataReg.empty() && dataReg[0].region.size() > d)
            {
                outBox.region.push_back(dataReg[0].region[d]);
            }
            else if (!idxReg.empty())
            {
                outBox.region.push_back({0, outShape[out_d]});
            }
            else
            {
                outBox.region.push_back({0, 0});
            }
        }

        bool valid = true;
        for (const auto &dim : outBox.region)
        {
            if (dim.start >= dim.stop)
            {
                valid = false;
                break;
            }
        }
        if (!valid)
            return {};
        return {outBox};
    }

    std::vector<std::vector<Region>> backwardGather(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        const auto &outReg = outputRegions[0];
        const auto &dataShape = graph.nodes[node.parentIds[0]].shape;
        const auto &idxShape = graph.nodes[node.parentIds[1]].shape;
        uint32_t idxRank = idxShape.size();

        Region dataBox;
        dataBox.region.push_back({0, dataShape[0]});
        for (uint32_t d = 1; d < dataShape.size(); ++d)
        {
            uint32_t out_d = idxRank + d - 1;
            if (out_d < outReg.region.size())
            {
                dataBox.region.push_back(outReg.region[out_d]);
            }
            else
            {
                dataBox.region.push_back({0, dataShape[d]});
            }
        }

        Region idxBox;
        for (uint32_t d = 0; d < idxRank; ++d)
        {
            idxBox.region.push_back(outReg.region[d]);
        }

        return {{dataBox}, {idxBox}};
    }

    std::vector<Region> forwardConcat(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        bool allClean = true;
        for (size_t i = 0; i < parentRegions.size() - 1; ++i)
        {
            if (!parentRegions[i].empty())
            {
                allClean = false;
                break;
            }
        }
        if (allClean)
            return {};

        int32_t axis = getConstantInt32(node.parentIds.back(), graph)[0];
        uint32_t rank = node.shape.size();
        if (axis < 0)
            axis += rank;

        Region outBox;
        for (uint32_t d = 0; d < rank; ++d)
        {
            outBox.region.push_back({node.shape[d], 0}); // min, max
        }

        uint32_t current_offset = 0;
        for (size_t i = 0; i < node.parentIds.size() - 1; ++i)
        {
            const auto &pShape = graph.nodes[node.parentIds[i]].shape;
            const auto &pReg = parentRegions[i];
            if (!pReg.empty())
            {
                for (uint32_t d = 0; d < rank; ++d)
                {
                    uint32_t start = pReg[0].region[d].start;
                    uint32_t stop = pReg[0].region[d].stop;
                    if (d == (uint32_t)axis)
                    {
                        start += current_offset;
                        stop += current_offset;
                    }
                    outBox.region[d].start = std::min(outBox.region[d].start, start);
                    outBox.region[d].stop = std::max(outBox.region[d].stop, stop);
                }
            }
            current_offset += pShape[axis];
        }

        for (const auto &dim : outBox.region)
        {
            if (dim.start >= dim.stop)
                return {};
        }
        return {outBox};
    }

    std::vector<std::vector<Region>> backwardConcat(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        std::vector<std::vector<Region>> res(node.parentIds.size());
        if (outputRegions.empty())
            return res;

        int32_t axis = getConstantInt32(node.parentIds.back(), graph)[0];
        uint32_t rank = node.shape.size();
        if (axis < 0)
            axis += rank;

        uint32_t out_start = outputRegions[0].region[axis].start;
        uint32_t out_stop = outputRegions[0].region[axis].stop;

        uint32_t current_offset = 0;
        for (size_t i = 0; i < node.parentIds.size() - 1; ++i)
        {
            const auto &pShape = graph.nodes[node.parentIds[i]].shape;
            uint32_t in_dim = pShape[axis];
            uint32_t in_end = current_offset + in_dim;

            uint32_t ov_start = std::max(out_start, current_offset);
            uint32_t ov_stop = std::min(out_stop, in_end);

            if (ov_start >= ov_stop)
            {
                res[i] = {};
            }
            else
            {
                Region inBox = outputRegions[0];
                inBox.region[axis].start = ov_start - current_offset;
                inBox.region[axis].stop = ov_stop - current_offset;
                res[i] = {inBox};
            }
            current_offset = in_end;
        }
        res.back() = makeFull(graph.nodes[node.parentIds.back()].shape);
        return res;
    }

    std::vector<Region> forwardRepeat(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;

        int32_t repeats = getConstantInt32(node.parentIds[1], graph)[0];
        int32_t axis = getConstantInt32(node.parentIds[2], graph)[0];
        if (axis < 0)
            axis += sA.size();

        Region outBox = rA[0];
        outBox.region[axis].start *= repeats;
        outBox.region[axis].stop *= repeats;

        return {outBox};
    }

    std::vector<std::vector<Region>> backwardRepeat(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        int32_t repeats = getConstantInt32(node.parentIds[1], graph)[0];
        int32_t axis = getConstantInt32(node.parentIds[2], graph)[0];
        if (axis < 0)
            axis += sA.size();

        Region inBox = outputRegions[0];
        inBox.region[axis].start = inBox.region[axis].start / repeats;
        inBox.region[axis].stop = (inBox.region[axis].stop + repeats - 1) / repeats;

        return {{inBox}, makeFull(graph.nodes[node.parentIds[1]].shape), makeFull(graph.nodes[node.parentIds[2]].shape)};
    }

    std::vector<Region> forward(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        for (uint32_t pid : node.parentIds)
        {
            if (pid >= graph.nodes.size())
            {
                std::stringstream ss;
                ss << "[ShapePropagator.forward] Invalid parent ID " << pid << " for OpType " << node.opType;
                Error::throw_err(ss.str());
            }
        }
        switch (node.opType)
        {
        case OpType::ADD:
        case OpType::MUL:
        case OpType::DIVIDE:
        case OpType::POWER:
        case OpType::SIN:
        case OpType::COS:
        case OpType::NEGATE:
        case OpType::CAST:
        case OpType::TRIU:
        case OpType::COPY_TO:
        case OpType::CONTIGUOUS:
        case OpType::SCATTER:
            return forwardElementwise(node, graph, parentRegions);
        case OpType::DOT:
            return forwardDot(node, graph, parentRegions);
        case OpType::SUM:
        case OpType::MAX:
            return forwardReduce(node, graph, parentRegions);
        case OpType::RESHAPE:
            return forwardReshape(node, graph, parentRegions);
        case OpType::PERMUTE:
            return forwardPermute(node, graph, parentRegions);
        case OpType::GATHER:
            return forwardGather(node, graph, parentRegions);
        case OpType::CONCAT:
            return forwardConcat(node, graph, parentRegions);
        case OpType::REPEAT:
            return forwardRepeat(node, graph, parentRegions);
        case OpType::ARANGE:
        case OpType::FILL:
        case OpType::SLICE:
        case OpType::IM2COL:
            return forwardFull(node, graph, parentRegions);
        default:
            std::stringstream ss;
            ss << "[ShapePropagator.forward] Unsupported OpType for ShapePropagator.forward: " << toString(node.opType);
            Error::throw_err(ss.str());
        }
    }

    std::vector<std::vector<Region>> backward(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        switch (node.opType)
        {
        case OpType::ADD:
        case OpType::MUL:
        case OpType::DIVIDE:
        case OpType::POWER:
        case OpType::SIN:
        case OpType::COS:
        case OpType::NEGATE:
        case OpType::CAST:
        case OpType::TRIU:
        case OpType::COPY_TO:
        case OpType::CONTIGUOUS:
        case OpType::SCATTER:
            return backwardElementwise(node, outputRegions);
        case OpType::DOT:
            return backwardDot(node, graph, outputRegions);
        case OpType::SUM:
        case OpType::MAX:
            return backwardReduce(node, graph, outputRegions);
        case OpType::RESHAPE:
            return backwardReshape(node, graph, outputRegions);
        case OpType::PERMUTE:
            return backwardPermute(node, graph, outputRegions);
        case OpType::GATHER:
            return backwardGather(node, graph, outputRegions);
        case OpType::CONCAT:
            return backwardConcat(node, graph, outputRegions);
        case OpType::REPEAT:
            return backwardRepeat(node, graph, outputRegions);
        case OpType::ARANGE:
        case OpType::FILL:
        case OpType::SLICE:
        case OpType::IM2COL:
            return backwardFull(node, graph, outputRegions);
        default:
            std::stringstream ss;
            ss << "[ShapePropagator.backward] Unsupported OpType for ShapePropagator.backward: " << toString(node.opType);
            Error::throw_err(ss.str());
        }
    }
};