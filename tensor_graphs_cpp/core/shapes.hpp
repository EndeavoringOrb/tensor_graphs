#pragma once
#include <algorithm>
#include <cstring>

#include "core/graph.hpp"
#include "core/types.hpp"

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

inline std::vector<Region> forwardElementwise(const TensorNode &node, const Graph &graph,
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

inline std::vector<std::vector<Region>> backwardElementwise(const uint32_t n_children,
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

inline std::vector<Region> forwardFull(const TensorNode &node, const Graph &graph,
                                       const std::vector<std::vector<Region>> &parentRegions)
{
    for (const auto &p : parentRegions)
    {
        if (!p.empty())
            return makeFull(node.getShape());
    }
    return {};
}

inline std::vector<std::vector<Region>> backwardFull(const TensorNode &node, const Graph &graph,
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

inline void inferShapeElementwise(LogicalId nodeId, Graph &graph, const char *opName = "ELEMENTWISE")
{
    const auto &child_ids = graph.getNode(nodeId).child_ids;
    if (child_ids.empty())
        return;
    const auto &s0 = graph.getNode(child_ids[0]).getShape();
    for (size_t i = 1; i < child_ids.size(); ++i)
    {
        const auto &si = graph.getNode(child_ids[i]).getShape();
        if (s0 != si)
        {
            std::stringstream ss;
            ss << "[ShapePropagator.inferShape] Atomic " << opName << " requires exact shape match. Got "
               << toString(s0) << " and " << toString(si) << ". Use explicit repeat/reshape. (Node "
               << graph.getNode(nodeId).id << "). " << graph.getNode(nodeId).debugOrigin;
            Error::throw_err(ss.str());
        }
    }
    graph.getNode(nodeId).setShape(s0);
}

inline void inferShapeUnary(LogicalId nodeId, Graph &graph)
{
    graph.getNode(nodeId).setShape(graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape());
}