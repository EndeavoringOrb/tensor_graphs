#pragma once
#include <algorithm>
#include <cstring>

#include "core/graph.hpp"
#include "core/types.hpp"

inline bool isElementwise(OpType op)
{
    switch (op)
    {
    case OpType::ADD:
    case OpType::MUL:
    case OpType::DIVIDE:
    case OpType::POWER:
    case OpType::SIN:
    case OpType::COS:
    case OpType::NEGATE:
    case OpType::CAST:
    case OpType::COPY_TO:
    case OpType::CONTIGUOUS:
    case OpType::LOG:
    case OpType::LT:
    case OpType::EQ:
    case OpType::AND:
    case OpType::OR:
    case OpType::NOT:
        return true;
    default:
        return false;
    }
}

inline std::vector<uint32_t> coordsFromFlatIndex(uint64_t flatIndex, const std::vector<uint32_t> &shape)
{
    std::vector<uint32_t> coords(shape.size(), 0);
    uint64_t temp = flatIndex;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        if (shape[static_cast<uint64_t>(i)] == 0)
            return coords;
        coords[static_cast<uint64_t>(i)] = static_cast<uint32_t>(temp % shape[static_cast<uint64_t>(i)]);
        temp /= shape[static_cast<uint64_t>(i)];
    }
    return coords;
}

inline uint64_t flatIndexFromCoords(const std::vector<uint32_t> &coords, const std::vector<uint32_t> &shape)
{
    uint64_t flatIndex = 0;
    uint64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        flatIndex += static_cast<uint64_t>(coords[static_cast<uint64_t>(i)]) * stride;
        stride *= shape[static_cast<uint64_t>(i)];
    }
    return flatIndex;
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

inline void getFlatBounds(const Region &region, const std::vector<uint32_t> &shape, uint64_t &flat_start,
                          uint64_t &flat_stop)
{
    std::vector<uint64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    flat_start = 0;
    uint64_t flat_stop_minus_1 = 0;
    for (uint64_t i = 0; i < region.region.size(); ++i)
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
    for (uint64_t i = 0; i < shape.size(); ++i)
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

inline int32_t normalizeAxisIndex(int32_t axis, uint32_t rank)
{
    if (axis < 0)
        axis += static_cast<int32_t>(rank);
    return axis;
}

inline bool isValidRegion(const Region &region)
{
    for (const auto &dim : region.region)
    {
        if (dim.start >= dim.stop)
            return false;
    }
    return true;
}

inline Region mapSliceRegionForward(const Region &region, const std::vector<uint32_t> &shape,
                                    const std::vector<int32_t> &starts, const std::vector<int32_t> &ends,
                                    const std::vector<int32_t> &steps)
{
    Region out;
    for (uint64_t d = 0; d < shape.size(); ++d)
    {
        int32_t start = d < starts.size() ? starts[d] : 0;
        int32_t end = d < ends.size() ? ends[d] : static_cast<int32_t>(shape[d]);
        int32_t step = d < steps.size() ? steps[d] : 1;
        if (step <= 0)
            Error::throw_err("[ShapePropagator] slice step must be positive");
        if (start < 0)
            start += static_cast<int32_t>(shape[d]);
        if (end < 0)
            end += static_cast<int32_t>(shape[d]);
        start = std::clamp<int32_t>(start, 0, static_cast<int32_t>(shape[d]));
        end = std::clamp<int32_t>(end, 0, static_cast<int32_t>(shape[d]));
        uint32_t lo = region.region[d].start;
        uint32_t hi = region.region[d].stop;
        if (lo >= hi)
        {
            out.region.push_back({0, 0});
            continue;
        }

        int64_t outStart = static_cast<int64_t>(lo) - start;
        if (outStart <= 0)
            outStart = 0;
        else
            outStart = (outStart + step - 1) / step;

        int64_t outStop = static_cast<int64_t>(hi) - start;
        if (outStop <= 0)
            outStop = 0;
        else
            outStop = (outStop + step - 1) / step;

        uint32_t outShapeDim = end > start ? static_cast<uint32_t>((end - start + step - 1) / step) : 0;
        out.region.push_back({std::min<uint32_t>(static_cast<uint32_t>(std::max<int64_t>(0, outStart)), outShapeDim),
                              std::min<uint32_t>(static_cast<uint32_t>(std::max<int64_t>(0, outStop)), outShapeDim)});
    }
    return out;
}

inline Region mapSliceRegionBackward(const Region &region, const std::vector<uint32_t> &shape,
                                     const std::vector<int32_t> &starts, const std::vector<int32_t> &ends,
                                     const std::vector<int32_t> &steps)
{
    Region out;
    for (uint64_t d = 0; d < shape.size(); ++d)
    {
        int32_t start = d < starts.size() ? starts[d] : 0;
        int32_t end = d < ends.size() ? ends[d] : static_cast<int32_t>(shape[d]);
        int32_t step = d < steps.size() ? steps[d] : 1;
        if (step <= 0)
            Error::throw_err("[ShapePropagator] slice step must be positive");
        if (start < 0)
            start += static_cast<int32_t>(shape[d]);
        if (end < 0)
            end += static_cast<int32_t>(shape[d]);
        start = std::clamp<int32_t>(start, 0, static_cast<int32_t>(shape[d]));
        end = std::clamp<int32_t>(end, 0, static_cast<int32_t>(shape[d]));

        int64_t lo = static_cast<int64_t>(start) + static_cast<int64_t>(region.region[d].start) * step;
        int64_t hi = static_cast<int64_t>(start) + static_cast<int64_t>(region.region[d].stop) * step;
        out.region.push_back({static_cast<uint32_t>(std::clamp<int64_t>(lo, 0, shape[d])),
                              static_cast<uint32_t>(std::clamp<int64_t>(hi, 0, shape[d]))});
    }
    return out;
}

struct ShapePropagator
{
    void inferShapeRecursive(LogicalId nodeId, Graph &graph)
    {
        if (!graph.hasNode(nodeId))
            return;

        if (!graph.getNode(nodeId).getShape().empty())
            return;

        if (graph.getNode(nodeId).opType == OpType::INPUT)
            return;

        for (LogicalId pid : graph.getNode(nodeId).child_ids)
        {
            inferShapeRecursive(pid, graph);
        }

        inferShape(nodeId, graph);
    }

    void inferShape(LogicalId nodeId, Graph &graph)
    {
        if (!graph.hasNode(nodeId))
            return;
        if (!graph.getNode(nodeId).getShape().empty())
            return;
        if (graph.getNode(nodeId).opType == OpType::INPUT)
            return;

        switch (graph.getNode(nodeId).opType)
        {
        case OpType::ADD:
        case OpType::MUL:
        case OpType::DIVIDE:
        case OpType::POWER:
        case OpType::LT:
        case OpType::EQ:
        case OpType::AND:
        case OpType::OR: {
            auto s0 = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape();
            auto s1 = graph.getNode(graph.getNode(nodeId).child_ids[1]).getShape();
            if (s0 != s1)
            {
                std::stringstream ss;
                ss << "[ShapePropagator.inferShape] Atomic " << toString(graph.getNode(nodeId).opType)
                   << " requires exact shape match. Got " << toString(s0) << " and " << toString(s1)
                   << ". Use explicit repeat/reshape. (Node " << graph.getNode(nodeId).id
                   << "). " + graph.getNode(nodeId).debugOrigin;
                Error::throw_err(ss.str());
            }
            graph.getNode(nodeId).setShape(s0);
            break;
        }
        case OpType::DOT: {
            const auto &s0 = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape();
            const auto &s1 = graph.getNode(graph.getNode(nodeId).child_ids[1]).getShape();
            uint64_t r0 = s0.size();
            uint64_t r1 = s1.size();

            if (r0 != r1)
            {
                std::stringstream ss;
                ss << "[ShapePropagator.inferShape] nodeId=" + toString(nodeId) + " DOT requires equal ranks. Got "
                   << r0 << " (" + toString(s0) + ") and " << r1
                   << " (" + toString(s1) +
                          "). Implicit broadcasting is not supported; use explicit "
                          "reshape "
                          "to align ranks. debugOrigin=" +
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
                    Error::throw_err("DOT 4D: Dimension mismatch [B,H,M,K] @ [B,H,K,N], " + std::to_string(s0[2]) +
                                     " != " + std::to_string(s1[1]));
                }
                graph.getNode(nodeId).setShape({s0[0], s0[1], s0[2], s1[3]});
            }
            else
            {
                Error::throw_err("DOT: Only Rank 2 and Rank 3 are currently supported "
                                 "in this framework. Got r0=" +
                                 std::to_string(r0) + ", r1=" + std::to_string(r1));
            }
            break;
        }
        case OpType::SIN:
        case OpType::COS:
        case OpType::NEGATE:
        case OpType::CAST:
        case OpType::TRIU:
        case OpType::CONTIGUOUS:
        case OpType::SCATTER:
        case OpType::LOG:
        case OpType::NOT: {
            graph.getNode(nodeId).setShape(graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape());
            break;
        }
        case OpType::COPY_TO: {
            graph.getNode(nodeId).setShape(graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape());
            graph.getNode(nodeId).strides = graph.getNode(graph.getNode(nodeId).child_ids[0]).strides;
            break;
        }
        case OpType::SUM:
        case OpType::MAX: {
            auto s0 = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape();
            auto axis_vec = graph.getConstantInt32(graph.getNode(nodeId).child_ids[1]);
            int32_t axis = axis_vec[0];
            if (axis < 0)
                axis += s0.size();

            std::vector<uint32_t> new_shape;
            for (uint64_t i = 0; i < s0.size(); ++i)
            {
                if (i == (uint64_t)axis)
                    new_shape.push_back(1);
                else
                    new_shape.push_back(s0[i]);
            }
            graph.getNode(nodeId).setShape(new_shape);
            break;
        }
        case OpType::RESHAPE: {
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
            break;
        }
        case OpType::PERMUTE: {
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
            break;
        }
        case OpType::GATHER: {
            auto data_shape = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape();
            auto idx_shape = graph.getNode(graph.getNode(nodeId).child_ids[1]).getShape();
            std::vector<uint32_t> out_shape = idx_shape;
            for (uint64_t i = 1; i < data_shape.size(); ++i)
            {
                out_shape.push_back(data_shape[i]);
            }
            graph.getNode(nodeId).setShape(out_shape);
            break;
        }
        case OpType::CONCAT: {
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
            break;
        }
        case OpType::REPEAT: {
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
            break;
        }
        case OpType::FILL: {
            auto target_dims = graph.getConstantInt32(graph.getNode(nodeId).child_ids[1]);
            std::vector<uint32_t> out_shape(target_dims.size());
            for (uint64_t i = 0; i < target_dims.size(); ++i)
            {
                out_shape[i] = target_dims[i];
            }
            graph.getNode(nodeId).setShape(out_shape);
            break;
        }
        case OpType::IM2COL: {
            auto s0 = graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape(); // N, C, H, W
            uint32_t k = graph.getConstantInt32(graph.getNode(nodeId).child_ids[1])[0];
            uint32_t s = graph.getConstantInt32(graph.getNode(nodeId).child_ids[2])[0];
            uint32_t p = graph.getConstantInt32(graph.getNode(nodeId).child_ids[3])[0];
            uint32_t H = s0[2];
            uint32_t W = s0[3];
            uint32_t H_out = (H + 2 * p - k) / s + 1;
            uint32_t W_out = (W + 2 * p - k) / s + 1;
            graph.getNode(nodeId).setShape({s0[0], s0[1] * k * k, H_out * W_out});
            break;
        }
        case OpType::SLICE: {
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
                    Error::throw_err("Zero-sized dimension in tensor shape!" +
                                     toString(graph.getNode(nodeId), graph, ""));
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
            break;
        }
        case OpType::ARANGE: {
            int32_t start = graph.getConstantInt32(graph.getNode(nodeId).child_ids[0])[0];
            int32_t stop = graph.getConstantInt32(graph.getNode(nodeId).child_ids[1])[0];
            int32_t step = graph.getConstantInt32(graph.getNode(nodeId).child_ids[2])[0];
            graph.getNode(nodeId).setShape({(uint32_t)std::max(0, (stop - start + step - 1) / step)});
            break;
        }
        case OpType::ARGMAX: {
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
                if (i == (uint64_t)axis)
                    new_shape.push_back((uint32_t)k);
                else
                    new_shape.push_back(s0[i]);
            }
            graph.getNode(nodeId).setShape(new_shape);
            break;
        }
        case OpType::UNPACK: {
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
            break;
        }
        case OpType::FUSED:
            Error::throw_err("this should not happen. only atomic nodes should have their shape inferred.");
        default:
            break;
        }

        for (auto d : graph.getNode(nodeId).getShape())
        {
            if (d == 0)
            {
                Error::throw_err("Zero-sized dimension in tensor shape!" + toString(graph.getNode(nodeId), graph, ""));
            }
        }
    }

    std::vector<Region> forwardElementwise(const TensorNode &node, const Graph &graph,
                                           const std::vector<std::vector<Region>> &parentRegions)
    {
        std::vector<Region> outputRegions;
        auto regionExists = [&](const Region &r) {
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
        return mergeRegions(outputRegions);
    }

    std::vector<std::vector<Region>> backwardElementwise(const uint32_t n_children,
                                                         const std::vector<Region> &outputRegions)
    {
        std::vector<std::vector<Region>> inputRegions(n_children);
        std::vector<Region> merged = mergeRegions(outputRegions);
        for (uint64_t i = 0; i < n_children; ++i)
        {
            inputRegions[i] = merged;
        }
        return inputRegions;
    }

    std::vector<Region> forwardFull(const TensorNode &node, const Graph &graph,
                                    const std::vector<std::vector<Region>> &parentRegions)
    {
        for (const auto &p : parentRegions)
        {
            if (!p.empty())
                return makeFull(node.getShape());
        }
        return {};
    }

    std::vector<std::vector<Region>> backwardFull(const TensorNode &node, const Graph &graph,
                                                  const std::vector<Region> &outputRegions)
    {
        std::vector<std::vector<Region>> res(node.child_ids.size());
        if (outputRegions.empty())
            return res;
        for (uint64_t i = 0; i < node.child_ids.size(); ++i)
        {
            res[i] = makeFull(graph.getNode(node.child_ids[i]).getShape());
        }
        return res;
    }

    std::vector<Region> forwardDot(const TensorNode &node, const Graph &graph,
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
        return mergeRegions(outBoxes);
    }

    std::vector<std::vector<Region>> backwardDot(const TensorNode &node, const Graph &graph,
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
                aBox.region = {outBox.region[0], outBox.region[1], outBox.region[2], {0, sA[3]}}; // [B, H, M, K]
                bBox.region = {outBox.region[0], outBox.region[1], {0, sB[2]}, outBox.region[3]}; // [B, H, K, N]
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

    std::vector<Region> forwardReshape(const TensorNode &node, const Graph &graph,
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

    std::vector<std::vector<Region>> backwardReshape(const TensorNode &node, const Graph &graph,
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

            uint32_t rank = node.getShape().size();

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

    std::vector<Region> forwardReduce(const TensorNode &node, const Graph &graph,
                                      const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &sA = graph.getNode(node.child_ids[0]).getShape();
        int32_t axis = graph.getConstantInt32(node.child_ids[1])[0];
        if (axis < 0)
            axis += sA.size();

        std::vector<Region> outBoxes;
        for (const auto &inReg : rA)
        {
            Region outBox;
            for (uint64_t d = 0; d < sA.size(); ++d)
            {
                if ((int32_t)d == axis)
                {
                    if (inReg.region[d].start < inReg.region[d].stop)
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
                    outBox.region.push_back(inReg.region[d]);
                }
            }
            outBoxes.push_back(outBox);
        }
        return mergeRegions(outBoxes);
    }

    std::vector<std::vector<Region>> backwardReduce(const TensorNode &node, const Graph &graph,
                                                    const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

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
                if ((int32_t)d == axis)
                {
                    if (outReg.region[d].start < outReg.region[d].stop)
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
                    inBox.region.push_back(outReg.region[d]);
                }
            }
            inBoxes.push_back(inBox);
        }
        return {mergeRegions(inBoxes), makeFull(graph.getNode(node.child_ids[1]).getShape())};
    }

    std::vector<Region> forwardPermute(const TensorNode &node, const Graph &graph,
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

    std::vector<std::vector<Region>> backwardPermute(const TensorNode &node, const Graph &graph,
                                                     const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        auto dims = graph.getConstantInt32(node.child_ids[1]);
        std::vector<int32_t> invDims(dims.size());
        for (uint64_t i = 0; i < dims.size(); ++i)
        {
            invDims[dims[i]] = i;
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

    std::vector<Region> forwardGather(const TensorNode &node, const Graph &graph,
                                      const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &dataReg = parentRegions[0];
        const auto &idxReg = parentRegions[1];
        if (dataReg.empty() && idxReg.empty())
            return {};

        const auto &dataShape = graph.getNode(node.child_ids[0]).getShape();
        const auto &idxShape = graph.getNode(node.child_ids[1]).getShape();
        const auto &outShape = node.getShape();

        uint32_t idxRank = idxShape.size();
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

    std::vector<std::vector<Region>> backwardGather(const TensorNode &node, const Graph &graph,
                                                    const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        const auto &dataShape = graph.getNode(node.child_ids[0]).getShape();
        const auto &idxShape = graph.getNode(node.child_ids[1]).getShape();
        uint32_t idxRank = idxShape.size();
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

    std::vector<Region> forwardConcat(const TensorNode &node, const Graph &graph,
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

    std::vector<std::vector<Region>> backwardConcat(const TensorNode &node, const Graph &graph,
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

    std::vector<Region> forwardRepeat(const TensorNode &node, const Graph &graph,
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

    std::vector<std::vector<Region>> backwardRepeat(const TensorNode &node, const Graph &graph,
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

    std::vector<Region> forwardSlice(const TensorNode &node, const Graph &graph,
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

    std::vector<std::vector<Region>> backwardSlice(const TensorNode &node, const Graph &graph,
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

    std::vector<Region> forwardScatter(const TensorNode &node, const Graph &graph,
                                       const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &targetRegions = parentRegions[0];
        const auto &updateRegions = parentRegions[1];
        if (targetRegions.empty() && updateRegions.empty())
            return {};

        const auto &targetShape = graph.getNode(node.child_ids[0]).getShape();
        auto starts = graph.getConstantInt32(node.child_ids[2]);
        auto ends = graph.getConstantInt32(node.child_ids[3]);
        auto steps = graph.getConstantInt32(node.child_ids[4]);

        std::vector<Region> outBoxes;
        for (const auto &region : targetRegions)
            outBoxes.push_back(region);
        for (const auto &region : updateRegions)
            outBoxes.push_back(mapSliceRegionBackward(region, targetShape, starts, ends, steps));
        return mergeRegions(outBoxes);
    }

    std::vector<std::vector<Region>> backwardScatter(const TensorNode &node, const Graph &graph,
                                                     const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}, {}, {}};

        const auto &targetShape = graph.getNode(node.child_ids[0]).getShape();
        auto starts = graph.getConstantInt32(node.child_ids[2]);
        auto ends = graph.getConstantInt32(node.child_ids[3]);
        auto steps = graph.getConstantInt32(node.child_ids[4]);

        std::vector<Region> targetBoxes;
        std::vector<Region> updateBoxes;
        for (const auto &region : outputRegions)
        {
            targetBoxes.push_back(region);
            updateBoxes.push_back(mapSliceRegionForward(region, targetShape, starts, ends, steps));
        }

        return {mergeRegions(targetBoxes), mergeRegions(updateBoxes),
                makeFull(graph.getNode(node.child_ids[2]).getShape()),
                makeFull(graph.getNode(node.child_ids[3]).getShape()),
                makeFull(graph.getNode(node.child_ids[4]).getShape())};
    }

    std::vector<Region> forward(const TensorNode &node, const Graph &graph,
                                const std::vector<std::vector<Region>> &parentRegions)
    {
        for (LogicalId pid : node.child_ids)
        {
            if (!graph.hasNode(pid))
            {
                std::stringstream ss;
                ss << "[ShapePropagator.forward] Invalid parent ID " << pid << " for OpType " << node.opType;
                Error::throw_err(ss.str());
            }
        }
        if (isElementwise(node.opType))
        {
            return forwardElementwise(node, graph, parentRegions);
        }
        switch (node.opType)
        {
        case OpType::TRIU: {
            if (!parentRegions[1].empty())
                return makeFull(node.getShape());
            return mergeRegions(parentRegions[0]);
        }
        case OpType::SCATTER: {
            if (!parentRegions[2].empty() || !parentRegions[3].empty() || !parentRegions[4].empty())
                return makeFull(node.getShape());
            return forwardScatter(node, graph, parentRegions);
        }
        case OpType::DOT:
            return forwardDot(node, graph, parentRegions);
        case OpType::SUM:
        case OpType::MAX:
            return forwardReduce(node, graph, parentRegions);
        case OpType::ARGMAX: {
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
                    if ((int32_t)d == axis)
                    {
                        if (inReg.region[d].start < inReg.region[d].stop)
                        {
                            outBox.region.push_back({0, (uint32_t)k});
                        }
                        else
                        {
                            outBox.region.push_back({0, 0});
                        }
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
        case OpType::IM2COL:
        case OpType::UNPACK: // TODO: make more conservative
            return forwardFull(node, graph, parentRegions);
        case OpType::SLICE:
            return forwardSlice(node, graph, parentRegions);
        default:
            std::stringstream ss;
            ss << "[ShapePropagator.forward] Unsupported OpType for "
                  "ShapePropagator.forward: "
               << toString(node.opType);
            Error::throw_err(ss.str());
        }
    }

    std::vector<std::vector<Region>> backward(const TensorNode &node, const Graph &graph,
                                              const std::vector<Region> &outputRegions)
    {
        if (isElementwise(node.opType) || node.opType == OpType::INPUT)
        {
            return backwardElementwise(node.child_ids.size(), outputRegions);
        }
        switch (node.opType)
        {
        case OpType::TRIU:
            return {mergeRegions(outputRegions), makeFull(graph.getNode(node.child_ids[1]).getShape())};
        case OpType::SCATTER:
            return backwardScatter(node, graph, outputRegions);
        case OpType::DOT:
            return backwardDot(node, graph, outputRegions);
        case OpType::SUM:
        case OpType::MAX:
            return backwardReduce(node, graph, outputRegions);
        case OpType::ARGMAX: {
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
                    if ((int32_t)d == axis)
                    {
                        if (outReg.region[d].start < outReg.region[d].stop)
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
        case OpType::IM2COL:
        case OpType::UNPACK: // TODO: make more conservative
            return backwardFull(node, graph, outputRegions);
        case OpType::SLICE:
            return backwardSlice(node, graph, outputRegions);
        default:
            std::stringstream ss;
            ss << "[ShapePropagator.backward] Unsupported OpType for "
                  "ShapePropagator.backward: "
               << toString(node.opType);
            Error::throw_err(ss.str());
        }
    }
};