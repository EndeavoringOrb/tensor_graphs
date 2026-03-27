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

inline std::vector<uint32_t> coordsFromFlatIndex(uint64_t flatIndex, const std::vector<uint32_t> &shape)
{
    std::vector<uint32_t> coords(shape.size(), 0);
    uint64_t temp = flatIndex;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        if (shape[static_cast<size_t>(i)] == 0)
            return coords;
        coords[static_cast<size_t>(i)] = static_cast<uint32_t>(temp % shape[static_cast<size_t>(i)]);
        temp /= shape[static_cast<size_t>(i)];
    }
    return coords;
}

inline uint64_t flatIndexFromCoords(const std::vector<uint32_t> &coords, const std::vector<uint32_t> &shape)
{
    uint64_t flatIndex = 0;
    uint64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        flatIndex += static_cast<uint64_t>(coords[static_cast<size_t>(i)]) * stride;
        stride *= shape[static_cast<size_t>(i)];
    }
    return flatIndex;
}

inline bool evaluateInt32TensorForPlanning(uint32_t nodeId, const Graph &graph, std::vector<int32_t> &outValues)
{
    if (!graph.hasNode(nodeId))
        return false;

    const TensorNode &node = graph.getNode(nodeId);
    if (node.dtype != DType::INT32)
        return false;

    switch (node.opType)
    {
    case OpType::INPUT:
    {
        auto it = graph.constantStaging.find(nodeId);
        if (it == graph.constantStaging.end())
            return false;
        const auto &data = it->second;
        if (data.size() % sizeof(int32_t) != 0)
            return false;
        outValues.resize(data.size() / sizeof(int32_t));
        std::memcpy(outValues.data(), data.data(), data.size());
        return true;
    }
    case OpType::COPY_TO:
    case OpType::CONTIGUOUS:
    case OpType::CAST:
    case OpType::RESHAPE:
        return evaluateInt32TensorForPlanning(node.parentIds[0], graph, outValues);
    case OpType::ARANGE:
    {
        std::vector<int32_t> startVals, stopVals, stepVals;
        if (!evaluateInt32TensorForPlanning(node.parentIds[0], graph, startVals) ||
            !evaluateInt32TensorForPlanning(node.parentIds[1], graph, stopVals) ||
            !evaluateInt32TensorForPlanning(node.parentIds[2], graph, stepVals) ||
            startVals.empty() || stopVals.empty() || stepVals.empty())
        {
            return false;
        }

        int32_t start = startVals[0];
        int32_t stop = stopVals[0];
        int32_t step = stepVals[0];
        if (step == 0)
            return false;

        outValues.clear();
        if (step > 0)
        {
            for (int32_t v = start; v < stop; v += step)
                outValues.push_back(v);
        }
        else
        {
            for (int32_t v = start; v > stop; v += step)
                outValues.push_back(v);
        }
        return true;
    }
    case OpType::FILL:
    {
        std::vector<int32_t> valueVals;
        if (!evaluateInt32TensorForPlanning(node.parentIds[0], graph, valueVals) || valueVals.empty())
            return false;
        outValues.assign(static_cast<size_t>(countElements(node.shape)), valueVals[0]);
        return true;
    }
    case OpType::PERMUTE:
    {
        std::vector<int32_t> parentValues;
        if (!evaluateInt32TensorForPlanning(node.parentIds[0], graph, parentValues))
            return false;
        const auto &parentShape = graph.getNode(node.parentIds[0]).shape;
        auto dims = getConstantInt32(node.parentIds[1], graph);
        if (dims.size() != node.shape.size())
            return false;

        outValues.resize(static_cast<size_t>(countElements(node.shape)));
        for (uint64_t outFlat = 0; outFlat < outValues.size(); ++outFlat)
        {
            auto outCoords = coordsFromFlatIndex(outFlat, node.shape);
            std::vector<uint32_t> inCoords(parentShape.size(), 0);
            for (size_t i = 0; i < dims.size(); ++i)
                inCoords[static_cast<size_t>(dims[i])] = outCoords[i];
            outValues[static_cast<size_t>(outFlat)] = parentValues[static_cast<size_t>(flatIndexFromCoords(inCoords, parentShape))];
        }
        return true;
    }
    case OpType::SLICE:
    {
        std::vector<int32_t> parentValues;
        if (!evaluateInt32TensorForPlanning(node.parentIds[0], graph, parentValues))
            return false;
        const auto &parentShape = graph.getNode(node.parentIds[0]).shape;
        auto starts = getConstantInt32(node.parentIds[1], graph);
        auto ends = getConstantInt32(node.parentIds[2], graph);
        auto steps = getConstantInt32(node.parentIds[3], graph);

        outValues.resize(static_cast<size_t>(countElements(node.shape)));
        for (uint64_t outFlat = 0; outFlat < outValues.size(); ++outFlat)
        {
            auto outCoords = coordsFromFlatIndex(outFlat, node.shape);
            std::vector<uint32_t> inCoords(parentShape.size(), 0);
            for (size_t d = 0; d < parentShape.size(); ++d)
            {
                int32_t start = d < starts.size() ? starts[d] : 0;
                int32_t end = d < ends.size() ? ends[d] : static_cast<int32_t>(parentShape[d]);
                int32_t step = d < steps.size() ? steps[d] : 1;
                if (step <= 0)
                    return false;
                if (start < 0)
                    start += static_cast<int32_t>(parentShape[d]);
                if (end < 0)
                    end += static_cast<int32_t>(parentShape[d]);
                start = std::clamp<int32_t>(start, 0, static_cast<int32_t>(parentShape[d]));
                end = std::clamp<int32_t>(end, 0, static_cast<int32_t>(parentShape[d]));
                (void)end;
                inCoords[d] = static_cast<uint32_t>(start + static_cast<int32_t>(outCoords[d]) * step);
            }
            outValues[static_cast<size_t>(outFlat)] = parentValues[static_cast<size_t>(flatIndexFromCoords(inCoords, parentShape))];
        }
        return true;
    }
    case OpType::CONCAT:
    {
        int32_t axis = getConstantInt32(node.parentIds.back(), graph)[0];
        if (axis < 0)
            axis += static_cast<int32_t>(node.shape.size());
        if (axis < 0 || static_cast<size_t>(axis) >= node.shape.size())
            return false;

        std::vector<std::vector<int32_t>> parentValues(node.parentIds.size() - 1);
        for (size_t i = 0; i + 1 < node.parentIds.size(); ++i)
        {
            if (!evaluateInt32TensorForPlanning(node.parentIds[i], graph, parentValues[i]))
                return false;
        }

        outValues.resize(static_cast<size_t>(countElements(node.shape)));
        std::vector<uint32_t> offsets;
        offsets.reserve(node.parentIds.size() - 1);
        uint32_t currentOffset = 0;
        for (size_t i = 0; i + 1 < node.parentIds.size(); ++i)
        {
            offsets.push_back(currentOffset);
            currentOffset += graph.getNode(node.parentIds[i]).shape[static_cast<size_t>(axis)];
        }

        for (uint64_t outFlat = 0; outFlat < outValues.size(); ++outFlat)
        {
            auto outCoords = coordsFromFlatIndex(outFlat, node.shape);
            uint32_t parentAxisCoord = outCoords[static_cast<size_t>(axis)];
            size_t parentIndex = 0;
            for (; parentIndex < offsets.size(); ++parentIndex)
            {
                uint32_t parentAxisExtent = graph.getNode(node.parentIds[parentIndex]).shape[static_cast<size_t>(axis)];
                if (parentAxisCoord < offsets[parentIndex] + parentAxisExtent)
                    break;
            }
            if (parentIndex >= offsets.size())
                return false;

            auto inCoords = outCoords;
            inCoords[static_cast<size_t>(axis)] = parentAxisCoord - offsets[parentIndex];
            outValues[static_cast<size_t>(outFlat)] = parentValues[parentIndex][static_cast<size_t>(flatIndexFromCoords(inCoords, graph.getNode(node.parentIds[parentIndex]).shape))];
        }
        return true;
    }
    case OpType::REPEAT:
    {
        std::vector<int32_t> parentValues;
        if (!evaluateInt32TensorForPlanning(node.parentIds[0], graph, parentValues))
            return false;
        const auto &parentShape = graph.getNode(node.parentIds[0]).shape;
        int32_t repeats = getConstantInt32(node.parentIds[1], graph)[0];
        int32_t axis = getConstantInt32(node.parentIds[2], graph)[0];
        if (axis < 0)
            axis += static_cast<int32_t>(parentShape.size());
        if (repeats <= 0 || axis < 0 || static_cast<size_t>(axis) >= parentShape.size())
            return false;

        outValues.resize(static_cast<size_t>(countElements(node.shape)));
        for (uint64_t outFlat = 0; outFlat < outValues.size(); ++outFlat)
        {
            auto outCoords = coordsFromFlatIndex(outFlat, node.shape);
            auto inCoords = outCoords;
            inCoords[static_cast<size_t>(axis)] = outCoords[static_cast<size_t>(axis)] / static_cast<uint32_t>(repeats);
            outValues[static_cast<size_t>(outFlat)] = parentValues[static_cast<size_t>(flatIndexFromCoords(inCoords, parentShape))];
        }
        return true;
    }
    default:
        return false;
    }
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

inline int32_t normalizeAxisIndex(int32_t axis, uint32_t rank)
{
    if (axis < 0)
        axis += static_cast<int32_t>(rank);
    return axis;
}

inline Region makeEmptyRegion(size_t rank)
{
    Region region;
    for (size_t i = 0; i < rank; ++i)
        region.region.push_back({0, 0});
    return region;
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

inline Region clampRegionToShape(Region region, const std::vector<uint32_t> &shape)
{
    for (size_t i = 0; i < region.region.size() && i < shape.size(); ++i)
    {
        region.region[i].start = std::min(region.region[i].start, shape[i]);
        region.region[i].stop = std::min(region.region[i].stop, shape[i]);
    }
    return region;
}

inline Region shiftRegionAxis(const Region &region, size_t axis, int64_t delta)
{
    Region out = region;
    if (axis < out.region.size())
    {
        int64_t start = static_cast<int64_t>(out.region[axis].start) + delta;
        int64_t stop = static_cast<int64_t>(out.region[axis].stop) + delta;
        out.region[axis].start = static_cast<uint32_t>(std::max<int64_t>(0, start));
        out.region[axis].stop = static_cast<uint32_t>(std::max<int64_t>(0, stop));
    }
    return out;
}

inline Region mapSliceRegionForward(const Region &region, const std::vector<uint32_t> &shape,
                                    const std::vector<int32_t> &starts, const std::vector<int32_t> &ends,
                                    const std::vector<int32_t> &steps)
{
    Region out;
    for (size_t d = 0; d < shape.size(); ++d)
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
    for (size_t d = 0; d < shape.size(); ++d)
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
    void inferShapeRecursive(uint32_t nodeId, Graph &graph)
    {
        if (!graph.hasNode(nodeId))
            return;

        if (!graph.getNode(nodeId).shape.empty())
            return;

        if (graph.getNode(nodeId).opType == OpType::INPUT)
            return;

        for (uint32_t pid : graph.getNode(nodeId).parentIds)
        {
            inferShapeRecursive(pid, graph);
        }

        inferShape(nodeId, graph);
    }

    void inferShape(uint32_t nodeId, Graph &graph)
    {
        if (!graph.hasNode(nodeId))
            return;
        if (!graph.getNode(nodeId).shape.empty())
            return;
        if (graph.getNode(nodeId).opType == OpType::INPUT)
            return;

        switch (graph.getNode(nodeId).opType)
        {
        case OpType::ADD:
        case OpType::MUL:
        case OpType::DIVIDE:
        case OpType::POWER:
        {
            auto s0 = graph.getNode(graph.getNode(nodeId).parentIds[0]).shape;
            auto s1 = graph.getNode(graph.getNode(nodeId).parentIds[1]).shape;
            if (s0 != s1)
            {
                std::stringstream ss;
                ss << "[ShapePropagator.inferShape] Atomic " << toString(graph.getNode(nodeId).opType)
                   << " requires exact shape match. Got " << toString(s0)
                   << " and " << toString(s1) << ". Use explicit repeat/reshape. (Node " << graph.getNode(nodeId).id << ")";
                Error::throw_err(ss.str());
            }
            graph.getNode(nodeId).shape = s0;
            break;
        }
        case OpType::DOT:
        {
            const auto &s0 = graph.getNode(graph.getNode(nodeId).parentIds[0]).shape;
            const auto &s1 = graph.getNode(graph.getNode(nodeId).parentIds[1]).shape;
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
                if (s0[2] != s1[1])
                    Error::throw_err("DOT: K-dim mismatch[B,M,K] @ [B,K,N]");
                graph.getNode(nodeId).shape = {s0[0], s0[1], s1[2]};
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
            graph.getNode(nodeId).shape = graph.getNode(graph.getNode(nodeId).parentIds[0]).shape;
            break;
        }
        case OpType::SUM:
        case OpType::MAX:
        {
            auto s0 = graph.getNode(graph.getNode(nodeId).parentIds[0]).shape;
            auto axis_vec = getConstantInt32(graph.getNode(nodeId).parentIds[1], graph);
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
            graph.getNode(nodeId).shape = new_shape;
            break;
        }
        case OpType::RESHAPE:
        {
            auto s0 = graph.getNode(graph.getNode(nodeId).parentIds[0]).shape;
            auto target_dims = getConstantInt32(graph.getNode(nodeId).parentIds[1], graph);
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
            graph.getNode(nodeId).shape = out_shape;
            break;
        }
        case OpType::PERMUTE:
        {
            auto s0 = graph.getNode(graph.getNode(nodeId).parentIds[0]).shape;
            auto dims = getConstantInt32(graph.getNode(nodeId).parentIds[1], graph);
            std::vector<uint32_t> out_shape(dims.size());
            for (size_t i = 0; i < dims.size(); ++i)
            {
                out_shape[i] = s0[dims[i]];
            }
            graph.getNode(nodeId).shape = out_shape;
            break;
        }
        case OpType::GATHER:
        {
            auto data_shape = graph.getNode(graph.getNode(nodeId).parentIds[0]).shape;
            auto idx_shape = graph.getNode(graph.getNode(nodeId).parentIds[1]).shape;
            std::vector<uint32_t> out_shape = idx_shape;
            for (size_t i = 1; i < data_shape.size(); ++i)
            {
                out_shape.push_back(data_shape[i]);
            }
            graph.getNode(nodeId).shape = out_shape;
            break;
        }
        case OpType::CONCAT:
        {
            uint32_t axis_id = graph.getNode(nodeId).parentIds.back();
            auto axis_vec = getConstantInt32(axis_id, graph);
            int32_t axis = axis_vec[0];
            auto s0 = graph.getNode(graph.getNode(nodeId).parentIds[0]).shape;
            if (axis < 0)
                axis += s0.size();

            std::vector<uint32_t> out_shape = s0;
            uint32_t total_dim = s0[axis];
            for (size_t i = 1; i < graph.getNode(nodeId).parentIds.size() - 1; ++i)
            {
                auto si = graph.getNode(graph.getNode(nodeId).parentIds[i]).shape;
                total_dim += si[axis];
            }
            out_shape[axis] = total_dim;
            graph.getNode(nodeId).shape = out_shape;
            break;
        }
        case OpType::REPEAT:
        {
            auto s0 = graph.getNode(graph.getNode(nodeId).parentIds[0]).shape;
            auto repeats = getConstantInt32(graph.getNode(nodeId).parentIds[1], graph)[0];
            auto axis = getConstantInt32(graph.getNode(nodeId).parentIds[2], graph)[0];
            if (axis < 0)
                axis += s0.size();
            std::vector<uint32_t> out_shape = s0;
            out_shape[axis] *= repeats;
            graph.getNode(nodeId).shape = out_shape;
            break;
        }
        case OpType::FILL:
        {
            auto target_dims = getConstantInt32(graph.getNode(nodeId).parentIds[1], graph);
            std::vector<uint32_t> out_shape(target_dims.size());
            for (size_t i = 0; i < target_dims.size(); ++i)
            {
                out_shape[i] = target_dims[i];
            }
            graph.getNode(nodeId).shape = out_shape;
            break;
        }
        case OpType::IM2COL:
        {
            auto s0 = graph.getNode(graph.getNode(nodeId).parentIds[0]).shape; // N, C, H, W
            uint32_t k = getConstantInt32(graph.getNode(nodeId).parentIds[1], graph)[0];
            uint32_t s = getConstantInt32(graph.getNode(nodeId).parentIds[2], graph)[0];
            uint32_t p = getConstantInt32(graph.getNode(nodeId).parentIds[3], graph)[0];
            uint32_t H = s0[2];
            uint32_t W = s0[3];
            uint32_t H_out = (H + 2 * p - k) / s + 1;
            uint32_t W_out = (W + 2 * p - k) / s + 1;
            graph.getNode(nodeId).shape = {s0[0], s0[1] * k * k, H_out * W_out};
            break;
        }
        case OpType::SLICE:
        {
            auto s0 = graph.getNode(graph.getNode(nodeId).parentIds[0]).shape;
            auto starts = getConstantInt32(graph.getNode(nodeId).parentIds[1], graph);
            auto ends = getConstantInt32(graph.getNode(nodeId).parentIds[2], graph);
            auto steps = getConstantInt32(graph.getNode(nodeId).parentIds[3], graph);
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
            graph.getNode(nodeId).shape = out_shape;
            break;
        }
        case OpType::ARANGE:
        {
            int32_t start = getConstantInt32(graph.getNode(nodeId).parentIds[0], graph)[0];
            int32_t stop = getConstantInt32(graph.getNode(nodeId).parentIds[1], graph)[0];
            int32_t step = getConstantInt32(graph.getNode(nodeId).parentIds[2], graph)[0];
            graph.getNode(nodeId).shape = {(uint32_t)std::max(0, (stop - start + step - 1) / step)};
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
        return mergeRegions(outputRegions);
    }

    std::vector<std::vector<Region>> backwardElementwise(const TensorNode &node, const std::vector<Region> &outputRegions)
    {
        std::vector<std::vector<Region>> inputRegions(node.parentIds.size());
        for (size_t i = 0; i < node.parentIds.size(); ++i)
        {
            inputRegions[i] = mergeRegions(outputRegions);
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
            res[i] = makeFull(graph.getNode(node.parentIds[i]).shape);
        }
        return res;
    }

    std::vector<Region> forwardDot(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        const auto &rB = parentRegions[1];
        if (rA.empty() && rB.empty())
            return {};

        const auto &sA = graph.getNode(node.parentIds[0]).shape;
        const auto &sB = graph.getNode(node.parentIds[1]).shape;
        const auto &outShape = node.shape;

        std::vector<Region> outBoxes;

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
        return {mergeRegions(reqA), mergeRegions(reqB)};
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
        return mergeRegions(outBoxes);
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
        return {mergeRegions(inBoxes), makeFull(sShape)};
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

        std::vector<Region> outBoxes;
        for (const auto &inReg : rA)
        {
            Region outBox;
            for (size_t d = 0; d < sA.size(); ++d)
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

    std::vector<std::vector<Region>> backwardReduce(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        int32_t axis = getConstantInt32(node.parentIds[1], graph)[0];
        if (axis < 0)
            axis += sA.size();

        std::vector<Region> inBoxes;
        for (const auto &outReg : outputRegions)
        {
            Region inBox;
            for (size_t d = 0; d < sA.size(); ++d)
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
        return {mergeRegions(inBoxes), makeFull(graph.nodes[node.parentIds[1]].shape)};
    }

    std::vector<Region> forwardPermute(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        auto dims = getConstantInt32(node.parentIds[1], graph);

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
        return {mergeRegions(inBoxes), makeFull(graph.nodes[node.parentIds[1]].shape)};
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
                    outBox.region.push_back(dataBox.region.size() > d ? dataBox.region[d] : Dim{0, outShape[idxRank + d - 1]});
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

    std::vector<std::vector<Region>> backwardGather(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}};

        const auto &dataShape = graph.nodes[node.parentIds[0]].shape;
        const auto &idxShape = graph.nodes[node.parentIds[1]].shape;
        uint32_t idxRank = idxShape.size();
        std::vector<Region> dataBoxes;
        std::vector<Region> idxBoxes;

        std::vector<int32_t> idxValues;
        bool exactIdxValues = evaluateInt32TensorForPlanning(node.parentIds[1], graph, idxValues) &&
                              countElements(idxShape) == idxValues.size();

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

                int32_t idxValue = idxValues[static_cast<size_t>(idxFlat)];
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
        std::vector<Region> outBoxes;

        uint32_t current_offset = 0;
        for (size_t i = 0; i < node.parentIds.size() - 1; ++i)
        {
            const auto &pShape = graph.nodes[node.parentIds[i]].shape;
            const auto &pReg = parentRegions[i];
            for (const auto &region : pReg)
            {
                Region shifted = region;
                if (shifted.region.size() > static_cast<size_t>(axis))
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

    std::vector<std::vector<Region>> backwardConcat(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        std::vector<std::vector<Region>> res(node.parentIds.size());
        if (outputRegions.empty())
            return res;

        int32_t axis = getConstantInt32(node.parentIds.back(), graph)[0];
        uint32_t rank = node.shape.size();
        if (axis < 0)
            axis += rank;

        uint32_t current_offset = 0;
        for (size_t i = 0; i < node.parentIds.size() - 1; ++i)
        {
            const auto &pShape = graph.nodes[node.parentIds[i]].shape;
            uint32_t in_dim = pShape[axis];
            uint32_t in_end = current_offset + in_dim;

            std::vector<Region> inBoxes;
            for (const auto &outReg : outputRegions)
            {
                if (outReg.region.size() <= static_cast<size_t>(axis))
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

    std::vector<std::vector<Region>> backwardRepeat(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}};

        const auto &sA = graph.nodes[node.parentIds[0]].shape;
        int32_t repeats = getConstantInt32(node.parentIds[1], graph)[0];
        int32_t axis = getConstantInt32(node.parentIds[2], graph)[0];
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

        return {mergeRegions(inBoxes), makeFull(graph.nodes[node.parentIds[1]].shape), makeFull(graph.nodes[node.parentIds[2]].shape)};
    }

    std::vector<Region> forwardSlice(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &rA = parentRegions[0];
        if (rA.empty())
            return {};

        const auto &shape = graph.nodes[node.parentIds[0]].shape;
        auto starts = getConstantInt32(node.parentIds[1], graph);
        auto ends = getConstantInt32(node.parentIds[2], graph);
        auto steps = getConstantInt32(node.parentIds[3], graph);

        std::vector<Region> outBoxes;
        for (const auto &region : rA)
            outBoxes.push_back(mapSliceRegionForward(region, shape, starts, ends, steps));
        return mergeRegions(outBoxes);
    }

    std::vector<std::vector<Region>> backwardSlice(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}, {}};

        const auto &shape = graph.nodes[node.parentIds[0]].shape;
        auto starts = getConstantInt32(node.parentIds[1], graph);
        auto ends = getConstantInt32(node.parentIds[2], graph);
        auto steps = getConstantInt32(node.parentIds[3], graph);

        std::vector<Region> inBoxes;
        for (const auto &region : outputRegions)
            inBoxes.push_back(mapSliceRegionBackward(region, shape, starts, ends, steps));

        return {mergeRegions(inBoxes), makeFull(graph.nodes[node.parentIds[1]].shape), makeFull(graph.nodes[node.parentIds[2]].shape), makeFull(graph.nodes[node.parentIds[3]].shape)};
    }

    std::vector<Region> forwardScatter(const TensorNode &node, const Graph &graph, const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &targetRegions = parentRegions[0];
        const auto &updateRegions = parentRegions[1];
        if (targetRegions.empty() && updateRegions.empty())
            return {};

        const auto &targetShape = graph.nodes[node.parentIds[0]].shape;
        auto starts = getConstantInt32(node.parentIds[2], graph);
        auto ends = getConstantInt32(node.parentIds[3], graph);
        auto steps = getConstantInt32(node.parentIds[4], graph);

        std::vector<Region> outBoxes;
        for (const auto &region : targetRegions)
            outBoxes.push_back(region);
        for (const auto &region : updateRegions)
            outBoxes.push_back(mapSliceRegionBackward(region, targetShape, starts, ends, steps));
        return mergeRegions(outBoxes);
    }

    std::vector<std::vector<Region>> backwardScatter(const TensorNode &node, const Graph &graph, const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}, {}, {}};

        const auto &targetShape = graph.nodes[node.parentIds[0]].shape;
        auto starts = getConstantInt32(node.parentIds[2], graph);
        auto ends = getConstantInt32(node.parentIds[3], graph);
        auto steps = getConstantInt32(node.parentIds[4], graph);

        std::vector<Region> targetBoxes;
        std::vector<Region> updateBoxes;
        for (const auto &region : outputRegions)
        {
            targetBoxes.push_back(region);
            updateBoxes.push_back(mapSliceRegionForward(region, targetShape, starts, ends, steps));
        }

        return {mergeRegions(targetBoxes), mergeRegions(updateBoxes), makeFull(graph.nodes[node.parentIds[2]].shape), makeFull(graph.nodes[node.parentIds[3]].shape), makeFull(graph.nodes[node.parentIds[4]].shape)};
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
        case OpType::COPY_TO:
        case OpType::CONTIGUOUS:
            return forwardElementwise(node, graph, parentRegions);
        case OpType::TRIU:
        {
            if (!parentRegions[1].empty())
                return makeFull(node.shape);
            return mergeRegions(parentRegions[0]);
        }
        case OpType::SCATTER:
        {
            if (!parentRegions[2].empty() || !parentRegions[3].empty() || !parentRegions[4].empty())
                return makeFull(node.shape);
            return forwardScatter(node, graph, parentRegions);
        }
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
        case OpType::IM2COL:
            return forwardFull(node, graph, parentRegions);
        case OpType::SLICE:
            return forwardSlice(node, graph, parentRegions);
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
        case OpType::COPY_TO:
        case OpType::CONTIGUOUS:
            return backwardElementwise(node, outputRegions);
        case OpType::TRIU:
            return {mergeRegions(outputRegions), makeFull(graph.nodes[node.parentIds[1]].shape)};
        case OpType::SCATTER:
            return backwardScatter(node, graph, outputRegions);
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
        case OpType::IM2COL:
            return backwardFull(node, graph, outputRegions);
        case OpType::SLICE:
            return backwardSlice(node, graph, outputRegions);
        default:
            std::stringstream ss;
            ss << "[ShapePropagator.backward] Unsupported OpType for ShapePropagator.backward: " << toString(node.opType);
            Error::throw_err(ss.str());
        }
    }
};
