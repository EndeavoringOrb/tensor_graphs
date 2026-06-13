// tensor_graphs_cpp/core/misc.hpp
#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/timer.hpp"
#include <unordered_set>

inline std::string toString(const std::vector<uint32_t> &shape)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

inline std::string toString(const std::vector<int32_t> &shape)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

inline std::string toString(const std::vector<uint64_t> &shape)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

inline std::string toString(const std::vector<int64_t> &shape)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

inline std::string toString(const TensorNode &node, const std::string &prefix = "")
{
    std::stringstream ss;
    ss << prefix << "Node " << node.id << " [" << toString(node.opType);
    if (node.opType == OpType::FUSED)
    {
        ss << " (" << node.opName << ")";
    }
    ss << "]\n"
       << prefix << "  DType:        " << toString(node.dtype) << "\n"
       << prefix << "  Shape:        " << toString(node.getShape()) << "\n"
       << prefix << "  Strides:      " << toString(node.strides) << "\n"
       << prefix << "  Backend:      " << node.backend << "\n"
       << prefix << "  Contiguous:   " << (isContiguous(node) ? "true" : "false") << "\n"
       << prefix << "  Storage Type: " << toString(node.storageType);
    return ss.str();
}

/**
 * Helper to format a TensorNode's metadata and its parents' metadata into a string.
 * This encapsulates the logging logic used in estimateCost and interpolate.
 */
inline std::string toString(const TensorNode &node, const Graph &graph, const std::string &prefix = "")
{
    std::stringstream ss;
    ss << prefix << "Node " << node.id << " [" << toString(node.opType) << "]\n"
       << prefix << "  Origin:     " << node.debugOrigin << "\n"
       << prefix << "  DType:      " << toString(node.dtype) << "\n"
       << prefix << "  Shape:      " << toString(node.getShape()) << "\n"
       << prefix << "  Strides:    " << toString(node.strides) << "\n"
       << prefix << "  Backend:    " << node.backend << "\n"
       << prefix << "  Contiguous: " << (isContiguous(node) ? "true" : "false") << "\n"
       << prefix << "  Parents (" << node.parentIds.size() << "):";

    if (node.parentIds.empty())
    {
        ss << " None";
    }
    else
    {
        for (size_t i = 0; i < node.parentIds.size(); ++i)
        {
            uint32_t pid = node.parentIds[i];
            if (graph.hasNode(pid))
            {
                const auto &parent = graph.getNode(pid);
                ss << "\n"
                   << prefix << "    [" << i << "] Parent ID " << pid
                   << "\n"
                   << toString(parent, (std::string) "    ");
            }
            else
            {
                ss << "\n"
                   << prefix << "[" << i << "] Parent ID " << pid << " [OUT OF BOUNDS/NOT FOUND]";
            }
        }
    }
    return ss.str();
}

/**
 * Convenience wrapper to print node info directly to std::cout
 */
inline void printNode(const TensorNode &node, const Graph &graph, const std::string &label = "NODE INFO")
{
    std::cout << "--- " << label << " ---\n"
              << toString(node, graph)
              << "\n-----------------------" << std::endl;
}

inline std::string toString(const Region &reg)
{
    std::stringstream ss;
    ss << "[";
    for (uint32_t i = 0; i < reg.region.size(); i++)
    {
        if (i > 0)
        {
            ss << ", ";
        }
        const Dim &dim = reg.region[i];
        ss << "(" << dim.start << ", " << dim.stop << ")";
    }
    return ss.str();
}

inline std::string toString(const OpInstruction &inst)
{
    std::stringstream ss;
    ss << "OpInstruction\n"
       << "  Node ID: " << inst.nodeId << "\n"
       << "  Full Kernel ID: " << inst.fullKernelId << "\n"
       << "  Input Node IDs: " << toString(inst.inputNodeIds) << "\n"
       << "  Inplace Input Index: " << inst.inplaceInputIndex << "\n"
       << "  Backend: " << inst.backend << "\n";
    return ss.str();
}

inline std::string toString(const TensorView &view)
{
    std::stringstream ss;
    ss << "TensorView\n"
       << "  baseOffset: " << view.baseOffset << "\n"
       << "  shape: " << toString(view.getShape()) << "\n"
       << "  strides: " << toString(view.strides) << "\n"
       << "  dtype: " << view.dtype << "\n";
    return ss.str();
}

// Intersect two 1D intervals
inline Dim intersectDims(const Dim &a, const Dim &b)
{
    return {std::max(a.start, b.start), std::min(a.stop, b.stop)};
}

// Intersect two N-D Regions
inline Region intersectRegions(const Region &r1, const Region &r2)
{
    if (r1.region.size() != r2.region.size())
        return Region(); // Rank mismatch

    Region result;
    for (size_t i = 0; i < r1.region.size(); ++i)
    {
        result.region.push_back(intersectDims(r1.region[i], r2.region[i]));
    }
    return result;
}

inline bool regionsMatch(const Region &r1, const Region &r2)
{
    if (r1.region.size() != r2.region.size())
        return false;
    for (size_t i = 0; i < r1.region.size(); ++i)
    {
        if (r1.region[i].start != r2.region[i].start ||
            r1.region[i].stop != r2.region[i].stop)
        {
            return false;
        }
    }
    return true;
}

inline bool coversDim(const Dim &outer, const Dim &inner)
{
    return outer.start <= inner.start && outer.stop >= inner.stop;
}

inline bool coversRegion(const Region &outer, const Region &inner)
{
    if (outer.region.size() != inner.region.size())
        return false;
    for (size_t i = 0; i < outer.region.size(); i++)
    {
        if (!coversDim(outer.region[i], inner.region[i]))
            return false;
    }
    return true;
}

inline bool coversRegionList(const std::vector<Region> &outer, const std::vector<Region> &inner)
{
    if (inner.empty())
        return true;
    for (const auto &innerReg : inner)
    {
        bool found = false;
        for (const auto &outerReg : outer)
        {
            if (coversRegion(outerReg, innerReg))
            {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }
    return true;
}

// Check if two intervals overlap or are adjacent
inline bool intervalsOverlapOrAdjacent(const Dim &a, const Dim &b)
{
    return a.stop >= b.start && b.stop >= a.start;
}

inline std::string regionGroupKeyExcludingDim(const Region &region, size_t excludeDim)
{
    std::stringstream ss;
    for (size_t i = 0; i < region.region.size(); ++i)
    {
        if (i == excludeDim)
            continue;
        ss << region.region[i].start << "-" << region.region[i].stop << "|";
    }
    return ss.str();
}

inline std::vector<Region> mergeRegionsAlongDim(const std::vector<Region> &regions, size_t mergeDim)
{
    if (regions.empty())
        return {};

    std::map<std::string, std::vector<Region>> groups;
    for (const auto &region : regions)
    {
        groups[regionGroupKeyExcludingDim(region, mergeDim)].push_back(region);
    }

    std::vector<Region> merged;
    for (const auto &groupPair : groups)
    {
        std::vector<Region> group = groupPair.second;
        std::sort(group.begin(), group.end(), [mergeDim](const Region &a, const Region &b)
                  {
                      if (a.region.size() != b.region.size())
                          return a.region.size() < b.region.size();
                      for (size_t i = 0; i < a.region.size(); ++i)
                      {
                          if (i == mergeDim)
                              continue;
                          if (a.region[i].start != b.region[i].start)
                              return a.region[i].start < b.region[i].start;
                          if (a.region[i].stop != b.region[i].stop)
                              return a.region[i].stop < b.region[i].stop;
                      }
                      if (a.region[mergeDim].start != b.region[mergeDim].start)
                          return a.region[mergeDim].start < b.region[mergeDim].start;
                      return a.region[mergeDim].stop < b.region[mergeDim].stop; });

        Region current = group.front();
        for (size_t i = 1; i < group.size(); ++i)
        {
            if (intervalsOverlapOrAdjacent(current.region[mergeDim], group[i].region[mergeDim]))
            {
                current.region[mergeDim].start = std::min(current.region[mergeDim].start, group[i].region[mergeDim].start);
                current.region[mergeDim].stop = std::max(current.region[mergeDim].stop, group[i].region[mergeDim].stop);
            }
            else
            {
                merged.push_back(current);
                current = group[i];
            }
        }
        merged.push_back(current);
    }

    return normalizeRegions(std::move(merged));
}

// Merge regions by repeatedly coalescing along each dimension when all other
// dimensions are identical and the merge dimension overlaps or is adjacent.
//
// Examples:
// f([(0,2),(2,4)]) -> [(0,4)]
// f([((0,4),(0,2)),((0,2),(2,4)),((2,4),(2,4))]) -> [((0,4),(0,2)),((0,4),(2,4))]
// f([((0,4),(0,2)),((0,4),(2,4))]) -> [((0,4),(0,4))]
inline std::vector<Region> mergeRegions(const std::vector<Region> &regions)
{
    if (regions.empty())
        return {};

    std::vector<Region> result = normalizeRegions(regions);
    if (result.empty())
        return result;

    const int rank = result.front().region.size();
    for (int dim = rank - 1; dim >= 0; dim--) // TODO: maybe merge top down? could discover different optimizations in InfinityDomination if we merge different ways?
    {
        std::vector<Region> next = mergeRegionsAlongDim(result, dim);
        if (encodeRegionList(next) != encodeRegionList(result))
        {
            result = std::move(next);
        }
    }

    return normalizeRegions(std::move(result));
}

// Merge two vectors of regions together
inline std::vector<Region> mergeRegions(const std::vector<Region> &regions1, const std::vector<Region> &regions2)
{
    if (regions1.empty())
        return regions2;
    if (regions2.empty())
        return regions1;

    std::vector<Region> combined = regions1;
    combined.insert(combined.end(), regions2.begin(), regions2.end());
    return mergeRegions(combined);
}

// Intersect two lists of regions (The "AND" operation)
inline std::vector<Region> intersectRegionLists(const std::vector<Region> &list1, const std::vector<Region> &list2)
{
    if (list1.empty() || list2.empty())
        return {};

    std::vector<Region> intersections;
    for (const auto &r1 : list1)
    {
        for (const auto &r2 : list2)
        {
            Region inter = intersectRegions(r1, r2);
            intersections.push_back(inter);
        }
    }
    // Clean up overlapping results
    return mergeRegions(intersections);
}

inline std::string toString(const std::unordered_map<Backend, uint64_t> &map)
{
    std::stringstream ss;
    ss << "{";
    for (auto it = map.begin(); it != map.end(); ++it)
    {
        ss << toString(it->first) << ": " << it->second << " bytes";
        if (std::next(it) != map.end())
            ss << ", ";
    }
    ss << "}";
    return ss.str();
}

inline std::string toStringHex(uint64_t val)
{
    std::stringstream ss;
    ss << std::hex << val;
    return ss.str();
}

// Topological sort of graph nodes from given roots
inline std::vector<uint32_t> topologicalSort(const std::vector<uint32_t> &roots, const Graph &graph)
{
    std::vector<uint32_t> order;
    std::unordered_set<uint32_t> visited;
    auto visit = [&](auto &self, uint32_t node) -> void
    {
        if (visited.count(node))
            return;
        visited.insert(node);
        if (graph.hasNode(node))
        {
            for (uint32_t pid : graph.getNode(node).parentIds)
            {
                self(self, pid);
            }
        }
        order.push_back(node);
    };
    for (uint32_t root : roots)
    {
        visit(visit, root);
    }
    return order;
}

// Create a TensorView for a node
inline TensorView makeView(const TensorNode &node)
{
    TensorView view;
    view.setShape(node.getShape());
    view.baseOffset = 0;
    view.dtype = node.dtype;
    if (!node.strides.empty())
        view.strides = node.strides;
    else
        view.strides = calcContiguousStrides(node.getShape());
    return view;
}

inline size_t getRequiredBufferSize(const TensorView &view)
{
    if (view.getShape().empty())
        return 1;
    size_t maxOffset = 0;
    for (size_t i = 0; i < view.getShape().size(); ++i)
    {
        if (view.getShape()[i] > 0)
        {
            maxOffset += (view.getShape()[i] - 1) * view.strides[i];
        }
    }
    return maxOffset + 1;
}

inline std::vector<float> flattenOutput(const void *ptr, const std::vector<uint32_t> &shape,
                                        const std::vector<uint64_t> &strides, DType dtype)
{
    std::vector<float> result;
    uint64_t elems = countElements(shape);
    result.reserve(elems);
    if (dtype == DType::FLOAT32)
    {
        const float *src = static_cast<const float *>(ptr);
        for (uint64_t i = 0; i < elems; ++i)
            result.push_back(src[getStridedIndex(i, shape, strides)]);
    }
    else if (dtype == DType::INT32)
    {
        const int32_t *src = static_cast<const int32_t *>(ptr);
        for (uint64_t i = 0; i < elems; ++i)
            result.push_back(static_cast<float>(src[getStridedIndex(i, shape, strides)]));
    }
    else if (dtype == DType::BF16)
    {
        const uint16_t *src = static_cast<const uint16_t *>(ptr);
        for (uint64_t i = 0; i < elems; ++i)
        {
            uint32_t bits = static_cast<uint32_t>(src[getStridedIndex(i, shape, strides)]) << 16;
            float val;
            std::memcpy(&val, &bits, 4);
            result.push_back(val);
        }
    }
    else
    {
        result.resize(elems, 0.0f);
    }
    return result;
}

inline std::vector<uint8_t> flattenBytes(const void *ptr, const std::vector<uint32_t> &shape,
                                         const std::vector<uint64_t> &strides, DType dtype)
{
    uint64_t elems = countElements(shape);
    uint64_t elemSize = getDTypeSize(dtype);
    std::vector<uint8_t> result(elems * elemSize);

    for (uint64_t i = 0; i < elems; ++i)
    {
        uint64_t idx = getStridedIndex(i, shape, strides);
        std::memcpy(result.data() + i * elemSize, static_cast<const uint8_t *>(ptr) + idx * elemSize, elemSize);
    }
    return result;
}