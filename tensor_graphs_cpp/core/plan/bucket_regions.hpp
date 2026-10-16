// tensor_graphs_cpp/core/plan/bucket_regions.hpp
#pragma once
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>

#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

#include "core/graph.hpp"
#include "core/loaders/tg_store.hpp"
#include "core/misc.hpp"
#include "core/shape_propagator.hpp"
#include "core/types.hpp"

// Diagnostics switch, read once (getenv takes the CRT environment lock).
inline bool debugEnvTgDebugMissing()
{
    static const bool enabled = std::getenv("TG_DEBUG_MISSING") != nullptr;
    return enabled;
}

// Per-bucket regions of every logical node, from the op region traits:
// - dirty:  elements that may differ from the previous run (forward from the
//           bucket's dirty input regions);
// - needed: elements that can influence the bucket's needed output region,
//           plus every dirty element of every non-output value that does not
//           depend on input padding (buckets keep cached values valid),
//           closed backward over inputs.
// Elements outside `needed` do not matter for this run; elements outside
// `dirty` keep their previous (cached) value.
//
// Additive masks: in add(x, m) where m is a known float constant, elements
// with m == TGConstants::NEG_INF are masked by the explicit sentinel convention. Their sum stays far below every
// unmasked value of a softmax row under the mask-sentinel approximation.
// This is not an identity for arbitrary input values (NEG_INF is finite).
// Represents the unmasked coordinate set:
//   coord[col_axis] < coord[row_axis] + diag_k   (when upper_masked == true, i.e., TRIU(NEG_INF, diag_k))
struct TriuBound
{
    uint32_t row_axis = 0;
    uint32_t col_axis = 0;
    int32_t diag_k = 1;

    bool operator==(const TriuBound &other) const
    {
        return row_axis == other.row_axis && col_axis == other.col_axis && diag_k == other.diag_k;
    }
    bool operator!=(const TriuBound &other) const
    {
        return !(*this == other);
    }
};

// Masked elements therefore neither carry dirtiness forward nor make x
// needed.
// Bucket-independent facts used by computeBucketRegions: masks of add nodes
// and the finite/support regions of the value-support analysis. They depend
// only on the graph and its constant data, so one computation serves every
// bucket of a compile.
struct RegionFacts
{
    std::unordered_map<LogicalId, std::pair<uint32_t, std::vector<Region>>> masks;
    std::unordered_map<LogicalId, std::vector<Region>> finite, support;
    std::unordered_map<LogicalId, TriuBound> triuMasks;
    std::unordered_map<LogicalId, TriuBound> triuFinite, triuSupport;
    // Elements the full output needs (bucket-independent), computed by the
    // first computeBucketRegions call that needs them.
    struct LiveCache
    {
        std::mutex mutex;
        LogicalId root;
        std::shared_ptr<const std::unordered_map<LogicalId, std::vector<Region>>> live;
    };
    std::shared_ptr<LiveCache> liveCache = std::make_shared<LiveCache>();
};

struct BucketRegions
{
    static constexpr float kMaskedValue = TGConstants::NEG_INF;

    // Bucket-independent value facts (masks, finite and support regions) of
    // the graph; null when region analysis did not run.
    const RegionFacts *facts = nullptr;
    std::shared_ptr<const RegionFacts> ownedFacts;

    std::unordered_map<LogicalId, std::vector<Region>> dirty;
    std::unordered_map<LogicalId, std::vector<Region>> needed;

    const std::vector<Region> &dirtyOf(LogicalId id) const
    {
        auto it = dirty.find(id);
        return it == dirty.end() ? empty_ : it->second;
    }
    const std::vector<Region> &neededOf(LogicalId id) const
    {
        auto it = needed.find(id);
        return it == needed.end() ? empty_ : it->second;
    }

    // Elements that must be (re)computed this run: dirty & needed.
    const std::vector<Region> &liveOf(LogicalId id) const;
    // True if `box` contains every live element of `id`.
    bool coversLive(LogicalId id, const Region &box) const;

  private:
    std::vector<Region> empty_;
    // liveOf results; a BucketRegions belongs to one bucket's saturation.
    mutable std::unordered_map<LogicalId, std::vector<Region>> live_;
};

namespace bucket_regions
{

inline std::vector<Region> dropEmpty(std::vector<Region> regions)
{
    regions.erase(std::remove_if(regions.begin(), regions.end(),
                                 [](const Region &r) {
                                     if (r.region.empty())
                                         return true;
                                     for (const Dim &d : r.region)
                                         if (d.start >= d.stop)
                                             return true;
                                     return false;
                                 }),
                  regions.end());
    return regions;
}

inline std::vector<Region> intersect(const std::vector<Region> &a, const std::vector<Region> &b)
{
    return dropEmpty(intersectRegionLists(a, b));
}

inline std::vector<Region> unite(const std::vector<Region> &a, const std::vector<Region> &b)
{
    return dropEmpty(mergeRegions(a, b));
}

// a minus b, as boxes: each box of a is split only against the boxes of b
// that intersect it, one dimension at a time.
inline std::vector<Region> subtract(const std::vector<Region> &a, const std::vector<Region> &b)
{
    auto overlaps = [](const Region &x, const Region &y) {
        if (x.region.size() != y.region.size())
            Error::throw_err("[bucket_regions::subtract] rank mismatch");
        for (size_t d = 0; d < x.region.size(); ++d)
            if (x.region[d].stop <= y.region[d].start || y.region[d].stop <= x.region[d].start)
                return false;
        return true;
    };
    const std::vector<Region> cuts = dropEmpty(b);
    std::vector<Region> result;
    std::vector<const Region *> hits;
    for (const Region &box : dropEmpty(a))
    {
        hits.clear();
        for (const Region &cut : cuts)
            if (overlaps(box, cut))
                hits.push_back(&cut);
        if (hits.empty())
        {
            result.push_back(box);
            continue;
        }
        std::vector<Region> pieces{box}, next;
        for (const Region *cut : hits)
        {
            next.clear();
            for (const Region &piece : pieces)
            {
                if (!overlaps(piece, *cut))
                {
                    next.push_back(piece);
                    continue;
                }
                // Peel the parts of `rest` outside the cut, one dimension at a time.
                Region rest = piece;
                for (size_t d = 0; d < piece.region.size(); ++d)
                {
                    const Dim r = rest.region[d];
                    const Dim &c = cut->region[d];
                    if (r.start < c.start)
                    {
                        Region below = rest;
                        below.region[d] = {r.start, c.start};
                        next.push_back(below);
                    }
                    if (c.stop < r.stop)
                    {
                        Region above = rest;
                        above.region[d] = {c.stop, r.stop};
                        next.push_back(above);
                    }
                    rest.region[d] = {std::max(r.start, c.start), std::min(r.stop, c.stop)};
                }
            }
            pieces.swap(next);
            if (pieces.empty())
                break;
        }
        result.insert(result.end(), pieces.begin(), pieces.end());
    }
    return dropEmpty(mergeRegions(result));
}

// Regions of a float tensor whose values are not masked: one run per row of
// the last dimension, merged.
inline std::vector<Region> unmaskedRegions(const std::vector<uint8_t> &bytes, const std::vector<uint32_t> &shape)
{
    const uint64_t count = countElements(shape);
    if (bytes.size() != count * sizeof(float) || shape.empty())
        Error::throw_err("[bucket_regions] mask data does not match its shape " + toString(shape));
    const float *data = reinterpret_cast<const float *>(bytes.data());
    const uint32_t cols = shape.back();
    const uint64_t rows = count / cols;
    std::vector<Region> regions;
    for (uint64_t row = 0; row < rows; ++row)
    {
        std::vector<uint32_t> prefix(shape.size() - 1);
        uint64_t rest = row;
        for (int64_t d = static_cast<int64_t>(shape.size()) - 2; d >= 0; --d)
        {
            prefix[d] = static_cast<uint32_t>(rest % shape[d]);
            rest /= shape[d];
        }
        uint32_t c = 0;
        while (c < cols)
        {
            while (c < cols && data[row * cols + c] == BucketRegions::kMaskedValue)
                ++c;
            const uint32_t start = c;
            while (c < cols && data[row * cols + c] != BucketRegions::kMaskedValue)
                ++c;
            if (c > start)
            {
                Region r;
                for (uint32_t p : prefix)
                    r.region.push_back({p, p + 1});
                r.region.push_back({start, c});
                regions.push_back(std::move(r));
            }
        }
    }
    return mergeRegions(regions);
}

// Bounding box of a non-empty region list.
inline Region boundingBox(const std::vector<Region> &regions)
{
    if (regions.empty())
        Error::throw_err("[bucket_regions] bounding box of no regions");
    Region box = regions.front();
    for (const Region &r : regions)
    {
        if (r.region.size() != box.region.size())
            Error::throw_err("[bucket_regions] bounding box of regions with different ranks");
        for (size_t d = 0; d < box.region.size(); ++d)
        {
            box.region[d].start = std::min(box.region[d].start, r.region[d].start);
            box.region[d].stop = std::max(box.region[d].stop, r.region[d].stop);
        }
    }
    return box;
}

inline bool isFull(const Region &r, const std::vector<uint32_t> &shape)
{
    if (r.region.size() != shape.size())
        return false;
    for (size_t d = 0; d < shape.size(); ++d)
        if (r.region[d].start != 0 || r.region[d].stop != shape[d])
            return false;
    return true;
}

} // namespace bucket_regions

inline const std::vector<Region> &BucketRegions::liveOf(LogicalId id) const
{
    auto it = live_.find(id);
    if (it != live_.end())
        return it->second;
    return live_[id] = bucket_regions::intersect(dirtyOf(id), neededOf(id));
}

inline bool BucketRegions::coversLive(LogicalId id, const Region &box) const
{
    const std::vector<Region> &live = liveOf(id);
    return !live.empty() && coversRegionList({box}, live);
}

namespace bucket_regions
{
// Range along `axis` of the parts of `valid` inside `box` (box's other axes).
// With everyRow, every position of the other axes inside `box` must keep at
// least one valid element, else nullopt. nullopt when nothing is valid.
inline std::optional<Dim> axisRange(const std::vector<Region> &valid, const Region &box, size_t axis, bool everyRow)
{
    std::optional<Dim> range;
    std::vector<Region> rows;
    for (const Region &v : valid)
    {
        if (v.region.size() != box.region.size())
            Error::throw_err("[axisRange] rank mismatch");
        Region hit = intersectRegions(v, box);
        if (dropEmpty({hit}).empty())
            continue;
        const Dim d = hit.region[axis];
        range = range ? Dim{std::min(range->start, d.start), std::max(range->stop, d.stop)} : d;
        hit.region[axis] = {0, 1};
        rows.push_back(hit);
    }
    if (!range || !everyRow)
        return range;
    Region boxRows = box;
    boxRows.region[axis] = {0, 1};
    const std::vector<Region> merged = mergeRegions(rows);
    if (merged.size() != 1 || !regionsMatch(merged.front(), boxRows))
        return std::nullopt;
    return range;
}

inline std::optional<Region> intersectBoxWithTriuBound(const Region &box, const TriuBound &b)
{
    if (b.row_axis >= box.region.size() || b.col_axis >= box.region.size())
        Error::throw_err("[intersectBoxWithTriuBound] triu bound axis out of range for box");
    const uint32_t r0 = box.region[b.row_axis].start;
    const uint32_t r1 = box.region[b.row_axis].stop;
    const uint32_t c0 = box.region[b.col_axis].start;
    const uint32_t c1 = box.region[b.col_axis].stop;
    if (r0 >= r1 || c0 >= c1)
        return std::nullopt;

    const int64_t r_min = std::max<int64_t>(static_cast<int64_t>(r0), static_cast<int64_t>(c0) - b.diag_k + 1);
    if (r_min >= static_cast<int64_t>(r1))
        return std::nullopt;
    const uint32_t restricted_r0 = static_cast<uint32_t>(std::max<int64_t>(0, r_min));

    const int64_t c_limit = static_cast<int64_t>(r1 - 1) + b.diag_k;
    if (c_limit <= static_cast<int64_t>(c0))
        return std::nullopt;
    const uint32_t restricted_stop = std::min<uint32_t>(c1, static_cast<uint32_t>(std::max<int64_t>(0, c_limit)));
    if (c0 >= restricted_stop)
        return std::nullopt;

    Region res = box;
    res.region[b.row_axis].start = restricted_r0;
    res.region[b.col_axis].stop = restricted_stop;
    return res;
}

inline std::vector<Region> boundedIntersect(const std::vector<Region> &needed, const TriuBound &b)
{
    std::vector<Region> out;
    for (const Region &box : dropEmpty(needed))
    {
        auto hit = intersectBoxWithTriuBound(box, b);
        if (hit)
            out.push_back(*hit);
    }
    return dropEmpty(mergeRegions(out));
}

// Per box of `needed`, the bounding box of its part inside `valid` (dropped
// when empty). A superset of the exact intersection with one box per input
// box: masks are staircases, and exact intersections would split every box
// into one per row.
inline std::vector<Region> boundedIntersect(const std::vector<Region> &needed, const std::vector<Region> &valid)
{
    std::vector<Region> out;
    for (const Region &box : bucket_regions::dropEmpty(needed))
    {
        const std::vector<Region> part = bucket_regions::intersect({box}, valid);
        if (!part.empty())
            out.push_back(bucket_regions::boundingBox(part));
    }
    return bucket_regions::dropEmpty(mergeRegions(out));
}

inline std::optional<Dim> axisRange(const TriuBound &b, const Region &box, size_t axis, bool everyRow)
{
    if (b.row_axis >= box.region.size() || b.col_axis >= box.region.size())
        Error::throw_err("[axisRange] triu bound axis out of range for box");
    const uint32_t r0 = box.region[b.row_axis].start;
    const uint32_t r1 = box.region[b.row_axis].stop;
    const uint32_t c0 = box.region[b.col_axis].start;
    const uint32_t c1 = box.region[b.col_axis].stop;
    if (r0 >= r1 || c0 >= c1)
        return std::nullopt;

    if (axis == b.col_axis)
    {
        const int64_t max_c_stop = std::min<int64_t>(c1, static_cast<int64_t>(r1 - 1) + b.diag_k);
        if (static_cast<int64_t>(c0) >= max_c_stop)
            return std::nullopt;
        if (everyRow)
        {
            const int64_t r0_limit = static_cast<int64_t>(r0) + b.diag_k;
            if (static_cast<int64_t>(c0) >= r0_limit)
                return std::nullopt;
        }
        return Dim{c0, static_cast<uint32_t>(max_c_stop)};
    }
    else if (axis == b.row_axis)
    {
        const int64_t min_r_start = std::max<int64_t>(static_cast<int64_t>(r0), static_cast<int64_t>(c0) - b.diag_k + 1);
        if (min_r_start >= static_cast<int64_t>(r1))
            return std::nullopt;
        if (everyRow)
        {
            const int64_t c1_req_r = static_cast<int64_t>(c1 - 1) - b.diag_k + 1;
            if (c1_req_r >= static_cast<int64_t>(r1))
                return std::nullopt;
        }
        return Dim{static_cast<uint32_t>(std::max<int64_t>(0, min_r_start)), r1};
    }
    return std::nullopt;
}

inline std::optional<std::vector<int32_t>> getInt32Constant(LogicalId id, const Graph &graph, TGStore *repo)
{
    if (graph.constantStaging.count(id))
        return graph.getConstantInt32(id);
    if (repo && repo->isValid() && repo->has(id))
    {
        auto bytes = repo->read(id);
        const auto &node = graph.getNode(id);
        uint64_t numElements = countElements(node.getShape());
        if (bytes.size() == numElements * sizeof(int32_t))
        {
            std::vector<int32_t> res(numElements);
            std::memcpy(res.data(), bytes.data(), bytes.size());
            return res;
        }
    }
    return std::nullopt;
}

inline std::optional<TriuBound> mapTriuBoundForward(const TensorNode &node, const Graph &graph,
                                                    const TriuBound &bound, TGStore *repo = nullptr)
{
    switch (node.opType)
    {
    case OpType::CONTIGUOUS:
        return bound;
    case OpType::REPEAT: {
        if (node.child_ids.size() < 3)
            return std::nullopt;
        auto repeats_vec = getInt32Constant(node.child_ids[1], graph, repo);
        auto axis_vec = getInt32Constant(node.child_ids[2], graph, repo);
        if (!repeats_vec || repeats_vec->empty() || !axis_vec || axis_vec->empty())
            return std::nullopt;
        const int32_t repeats = (*repeats_vec)[0];
        int32_t axis = (*axis_vec)[0];
        const size_t in_rank = graph.getNode(node.child_ids[0]).getShape().size();
        if (axis < 0)
            axis += static_cast<int32_t>(in_rank);
        if (repeats == 1 || (static_cast<uint32_t>(axis) != bound.row_axis &&
                             static_cast<uint32_t>(axis) != bound.col_axis))
        {
            return bound;
        }
        return std::nullopt;
    }
    case OpType::PERMUTE: {
        if (node.child_ids.size() < 2)
            return std::nullopt;
        auto dims_vec = getInt32Constant(node.child_ids[1], graph, repo);
        if (!dims_vec)
            return std::nullopt;
        const auto &dims = *dims_vec;
        uint32_t new_row = UINT32_MAX, new_col = UINT32_MAX;
        for (uint32_t i = 0; i < dims.size(); ++i)
        {
            if (static_cast<uint32_t>(dims[i]) == bound.row_axis)
                new_row = i;
            if (static_cast<uint32_t>(dims[i]) == bound.col_axis)
                new_col = i;
        }
        if (new_row != UINT32_MAX && new_col != UINT32_MAX)
        {
            return TriuBound{new_row, new_col, bound.diag_k};
        }
        return std::nullopt;
    }
    case OpType::RESHAPE: {
        const auto &in_shape = graph.getNode(node.child_ids[0]).getShape();
        const auto &out_shape = node.getShape();
        if (in_shape.size() >= 2 && out_shape.size() >= 2)
        {
            const size_t in_rank = in_shape.size();
            const size_t out_rank = out_shape.size();
            if (bound.row_axis == in_rank - 2 && bound.col_axis == in_rank - 1 &&
                in_shape[in_rank - 2] == out_shape[out_rank - 2] &&
                in_shape[in_rank - 1] == out_shape[out_rank - 1])
            {
                return TriuBound{static_cast<uint32_t>(out_rank - 2),
                                 static_cast<uint32_t>(out_rank - 1), bound.diag_k};
            }
        }
        return std::nullopt;
    }
    case OpType::SLICE: {
        if (node.child_ids.size() < 4)
            return std::nullopt;
        auto starts_vec = getInt32Constant(node.child_ids[1], graph, repo);
        auto steps_vec = getInt32Constant(node.child_ids[3], graph, repo);
        if (!starts_vec || !steps_vec)
            return std::nullopt;
        const auto &starts = *starts_vec;
        const auto &steps = *steps_vec;
        if (bound.row_axis < steps.size() && bound.col_axis < steps.size() &&
            steps[bound.row_axis] == 1 && steps[bound.col_axis] == 1 &&
            bound.row_axis < starts.size() && bound.col_axis < starts.size())
        {
            const int32_t new_k = bound.diag_k + starts[bound.row_axis] - starts[bound.col_axis];
            return TriuBound{bound.row_axis, bound.col_axis, new_k};
        }
        return std::nullopt;
    }
    default:
        return std::nullopt;
    }
}

inline std::optional<TriuBound> matchTriuMask(LogicalId id, const Graph &graph, TGStore *repo = nullptr)
{
    if (graph.getNode(id).getShape().empty())
    {
        ShapePropagator prop;
        prop.inferShapeRecursive(id, const_cast<Graph &>(graph));
    }
    const TensorNode &node = graph.getNode(id);
    const auto &shape = node.getShape();
    if (node.opType == OpType::TRIU && node.child_ids.size() == 2 && shape.size() >= 2)
    {
        std::optional<int32_t> diag_k;
        LogicalId k_id = node.child_ids[1];
        auto staged_k = graph.constantStaging.find(k_id);
        if (staged_k != graph.constantStaging.end() && staged_k->second->size() >= sizeof(int32_t))
        {
            int32_t k_val;
            std::memcpy(&k_val, staged_k->second->data(), sizeof(int32_t));
            diag_k = k_val;
        }
        else if (repo && repo->isValid() && repo->has(k_id))
        {
            auto bytes = repo->read(k_id);
            if (bytes.size() >= sizeof(int32_t))
            {
                int32_t k_val;
                std::memcpy(&k_val, bytes.data(), sizeof(int32_t));
                diag_k = k_val;
            }
        }
        if (diag_k)
        {
            auto checkFillScalar = [&](LogicalId data_id) -> bool {
                LogicalId curr = data_id;
                for (int depth = 0; depth < 8; ++depth)
                {
                    const TensorNode &n = graph.getNode(curr);
                    if (n.opType == OpType::FILL && n.child_ids.size() >= 2)
                    {
                        LogicalId val_id = n.child_ids[0];
                        const TensorNode &val_node = graph.getNode(val_id);
                        if (val_node.dtype != DType::FLOAT32)
                            return false;
                        auto st = graph.constantStaging.find(val_id);
                        if (st != graph.constantStaging.end() && st->second->size() >= sizeof(float))
                        {
                            float f_val;
                            std::memcpy(&f_val, st->second->data(), sizeof(float));
                            return f_val == BucketRegions::kMaskedValue;
                        }
                        if (repo && repo->isValid() && repo->has(val_id))
                        {
                            auto b = repo->read(val_id);
                            if (b.size() >= sizeof(float))
                            {
                                float f_val;
                                std::memcpy(&f_val, b.data(), sizeof(float));
                                return f_val == BucketRegions::kMaskedValue;
                            }
                        }
                        return false;
                    }
                    if ((n.opType == OpType::CONTIGUOUS || n.opType == OpType::REPEAT ||
                         n.opType == OpType::RESHAPE || n.opType == OpType::PERMUTE ||
                         n.opType == OpType::SLICE) && !n.child_ids.empty())
                    {
                        curr = n.child_ids[0];
                    }
                    else
                    {
                        return false;
                    }
                }
                return false;
            };

            if (checkFillScalar(node.child_ids[0]))
            {
                TriuBound bound;
                bound.row_axis = static_cast<uint32_t>(shape.size() - 2);
                bound.col_axis = static_cast<uint32_t>(shape.size() - 1);
                bound.diag_k = *diag_k;
                return bound;
            }
        }
    }

    const bool isView = (node.opType == OpType::CONTIGUOUS || node.opType == OpType::REPEAT ||
                         node.opType == OpType::RESHAPE || node.opType == OpType::PERMUTE ||
                         node.opType == OpType::SLICE);
    if (isView && !node.child_ids.empty())
    {
        auto child_bound = matchTriuMask(node.child_ids[0], graph, repo);
        if (child_bound)
        {
            return mapTriuBoundForward(node, graph, *child_bound, repo);
        }
    }
    return std::nullopt;
}
} // namespace bucket_regions

inline RegionFacts computeRegionFacts(const Graph &graph, const std::vector<LogicalId> &topo, TGStore *repo)
{
    using namespace bucket_regions;
    ShapePropagator prop;
    RegionFacts facts;
    auto &masks = facts.masks;
    auto &finite = facts.finite;
    auto &support = facts.support;
    // Unmasked regions of the mask operand of add(x, mask), per add node.
    // Unmasked regions of a float tensor with known values: from its staged or
    // folded data (in logical order), or through repeat/reshape/contiguous
    // from such a tensor. Empty optional if its values are not known.
    std::unordered_map<LogicalId, std::optional<std::vector<Region>>> unmaskedMemo;
    std::function<std::optional<std::vector<Region>>(LogicalId, uint32_t)> unmaskedOf =
        [&](LogicalId id, uint32_t depth) -> std::optional<std::vector<Region>> {
        auto memo = unmaskedMemo.find(id);
        if (memo != unmaskedMemo.end())
            return memo->second;
        std::optional<std::vector<Region>> result;
        const TensorNode &node = graph.getNode(id);
        if (node.dtype == DType::FLOAT32)
        {
            const uint64_t bytesNeeded = countElements(node.getShape()) * sizeof(float);
            std::vector<uint8_t> bytes;
            auto staged = graph.constantStaging.find(id);
            if (staged != graph.constantStaging.end())
                bytes = *staged->second;
            else if (repo && repo->has(id))
                bytes = repo->read(id);
            if (!bytes.empty() && bytes.size() == bytesNeeded)
                result = unmaskedRegions(bytes, node.getShape());
            else if (depth > 0 && !node.child_ids.empty() &&
                     (node.opType == OpType::REPEAT || node.opType == OpType::RESHAPE ||
                      node.opType == OpType::CONTIGUOUS))
            {
                auto source = unmaskedOf(node.child_ids[0], depth - 1);
                if (source)
                {
                    std::vector<std::vector<Region>> parents(node.child_ids.size());
                    parents[0] = *source;
                    result = dropEmpty(prop.forward(node, graph, parents));
                }
            }
        }
        unmaskedMemo[id] = result;
        return result;
    };
    for (LogicalId id : topo)
    {
        const TensorNode &node = graph.getNode(id);
        if (node.opType != OpType::ADD || node.child_ids.size() != 2)
            continue;
        for (uint32_t m = 0; m < 2; ++m)
        {
            const auto &shape = graph.getNode(node.child_ids[m]).getShape();
            if (shape != node.getShape())
                continue;
            auto triu = matchTriuMask(node.child_ids[m], graph, repo);
            if (triu)
            {
                masks[id] = {m, makeFull(shape)};
                facts.triuMasks[id] = *triu;
                break;
            }
            std::optional<std::vector<Region>> unmasked = unmaskedOf(node.child_ids[m], 8);
            if (!unmasked)
                continue;
            if (unmasked->size() == 1 && isFull(unmasked->front(), shape))
                continue;
            masks[id] = {m, std::move(*unmasked)};
            break;
        }
    }

    // Value support. finite[x]: where x is not masked (~ -1e9); support[x]:
    // where x may be nonzero. Absent means the whole tensor. A masked score
    // stays far below its row's maximum, so exp() of it (pow with a base > 1)
    // is exactly 0, and products and quotients keep those zeros. A dot then
    // only reads the rows of its second operand at columns where its first
    // operand can be nonzero.
    auto fullOf = [&](LogicalId id) { return makeFull(graph.getNode(id).getShape()); };
    auto finiteOf = [&](LogicalId id) {
        auto it = finite.find(id);
        return it == finite.end() ? fullOf(id) : it->second;
    };
    auto supportOf = [&](LogicalId id) {
        auto it = support.find(id);
        return it == support.end() ? fullOf(id) : it->second;
    };
    auto isFullList = [&](LogicalId id, const std::vector<Region> &r) {
        return r.size() == 1 && isFull(r.front(), graph.getNode(id).getShape());
    };
    // Smallest value of a constant float tensor (through view-like ops).
    std::function<std::optional<float>(LogicalId, uint32_t)> constantMin = [&](LogicalId id,
                                                                               uint32_t depth) -> std::optional<float> {
        const TensorNode &node = graph.getNode(id);
        if (node.dtype != DType::FLOAT32)
            return std::nullopt;
        std::vector<uint8_t> bytes;
        auto staged = graph.constantStaging.find(id);
        if (staged != graph.constantStaging.end())
            bytes = *staged->second;
        else if (repo && repo->has(id))
            bytes = repo->read(id);
        if (!bytes.empty() && bytes.size() % sizeof(float) == 0)
        {
            const float *v = reinterpret_cast<const float *>(bytes.data());
            float lo = v[0];
            for (size_t i = 1; i < bytes.size() / sizeof(float); ++i)
                lo = std::min(lo, v[i]);
            return lo;
        }
        if (depth > 0 && !node.child_ids.empty() &&
            (node.opType == OpType::REPEAT || node.opType == OpType::RESHAPE ||
             node.opType == OpType::CONTIGUOUS || node.opType == OpType::FILL))
            return constantMin(node.child_ids[0], depth - 1);
        return std::nullopt;
    };
    auto mapForward = [&](const TensorNode &node, const std::function<std::vector<Region>(LogicalId)> &of) {
        std::vector<std::vector<Region>> parents(node.child_ids.size());
        for (size_t c = 0; c < node.child_ids.size(); ++c)
        {
            const bool tensorInput = node.opType == OpType::CONCAT ? c > 0 : c == 0;
            if (tensorInput)
                parents[c] = of(node.child_ids[c]);
        }
        return dropEmpty(prop.forward(node, graph, parents));
    };
    for (LogicalId id : topo)
    {
        const TensorNode &node = graph.getNode(id);
        std::vector<Region> fin, sup;
        bool hasFin = false, hasSup = false;
        std::optional<TriuBound> triuFin, triuSup;
        switch (node.opType)
        {
        case OpType::ADD: {
            if (node.child_ids.size() != 2)
                break;
            auto mask = masks.find(id);
            auto triuMaskIt = facts.triuMasks.find(id);
            if (triuMaskIt != facts.triuMasks.end())
            {
                triuFin = triuMaskIt->second;
                hasFin = true;
            }
            else if (mask != masks.end())
            {
                fin = intersect(mask->second.second, finiteOf(node.child_ids[1 - mask->second.first]));
                hasFin = true;
            }
            else
            {
                auto t0 = facts.triuFinite.find(node.child_ids[0]);
                auto t1 = facts.triuFinite.find(node.child_ids[1]);
                if (t0 != facts.triuFinite.end())
                    triuFin = t0->second;
                else if (t1 != facts.triuFinite.end())
                    triuFin = t1->second;
                fin = intersect(finiteOf(node.child_ids[0]), finiteOf(node.child_ids[1]));
                hasFin = true;
            }
            break;
        }
        case OpType::POWER: {
            const auto base = constantMin(node.child_ids[0], 8);
            if (base && *base > 1.0f)
            {
                auto t = facts.triuFinite.find(node.child_ids[1]);
                if (t != facts.triuFinite.end())
                    triuSup = t->second;
                sup = finiteOf(node.child_ids[1]);
                hasSup = true;
            }
            break;
        }
        case OpType::MUL: {
            auto t0 = facts.triuSupport.find(node.child_ids[0]);
            auto t1 = facts.triuSupport.find(node.child_ids[1]);
            if (t0 != facts.triuSupport.end())
                triuSup = t0->second;
            else if (t1 != facts.triuSupport.end())
                triuSup = t1->second;
            sup = intersect(supportOf(node.child_ids[0]), supportOf(node.child_ids[1]));
            hasSup = true;
            break;
        }
        case OpType::DIVIDE: {
            auto t0 = facts.triuSupport.find(node.child_ids[0]);
            if (t0 != facts.triuSupport.end())
                triuSup = t0->second;
            sup = supportOf(node.child_ids[0]);
            hasSup = true;
            break;
        }
        case OpType::REPEAT:
        case OpType::RESHAPE:
        case OpType::PERMUTE:
        case OpType::CONTIGUOUS:
        case OpType::SLICE: {
            if (!node.child_ids.empty())
            {
                LogicalId c0 = node.child_ids[0];
                auto tFin = facts.triuFinite.find(c0);
                if (tFin != facts.triuFinite.end())
                {
                    triuFin = mapTriuBoundForward(node, graph, tFin->second, repo);
                }
                auto tSup = facts.triuSupport.find(c0);
                if (tSup != facts.triuSupport.end())
                {
                    triuSup = mapTriuBoundForward(node, graph, tSup->second, repo);
                }
            }
            fin = mapForward(node, finiteOf);
            sup = mapForward(node, supportOf);
            hasFin = hasSup = true;
            break;
        }
        case OpType::CONCAT: {
            fin = mapForward(node, finiteOf);
            sup = mapForward(node, supportOf);
            hasFin = hasSup = true;
            break;
        }
        default:
            break;
        }
        if (triuFin)
            facts.triuFinite[id] = *triuFin;
        if (triuSup)
            facts.triuSupport[id] = *triuSup;
        if (hasFin && !isFullList(id, fin))
            finite[id] = std::move(fin);
        if (hasSup && !isFullList(id, sup))
            support[id] = std::move(sup);
    }
    return facts;
}

using bucket_regions::boundedIntersect;

// paddedInputs: input -> axis along which the input is padded past each
// bucket's dirty region (e.g. token ids past the request). Values depending
// on that padding need not be kept valid (see `needed` below).
inline BucketRegions computeBucketRegions(const Graph &graph, LogicalId rootId, const Bucket &bucket,
                                          const std::vector<LogicalId> &topo, TGStore *repo,
                                          const RegionFacts *facts = nullptr,
                                          const std::unordered_map<LogicalId, uint32_t> *paddedInputs = nullptr)
{
    using namespace bucket_regions;
    BucketRegions out;
    ShapePropagator prop;

    if (facts)
        out.facts = facts;
    else
    {
        out.ownedFacts = std::make_shared<const RegionFacts>(computeRegionFacts(graph, topo, repo));
        out.facts = out.ownedFacts.get();
    }
    const RegionFacts &f = *out.facts;
    const auto &masks = f.masks;
    const auto &finite = f.finite;
    const auto &support = f.support;
    // Rank-3 dot A[b,m,k] x B[b,k,n] with partial support of A.
    auto dotSupport = [&](const TensorNode &node) -> const std::vector<Region> * {
        if (node.opType != OpType::DOT || node.child_ids.size() != 2 || node.getShape().size() != 3)
            return nullptr;
        auto it = support.find(node.child_ids[0]);
        return it == support.end() ? nullptr : &it->second;
    };
    auto dotTriuSupport = [&](const TensorNode &node) -> const TriuBound * {
        if (node.opType != OpType::DOT || node.child_ids.size() != 2 || node.getShape().size() != 3)
            return nullptr;
        auto it = f.triuSupport.find(node.child_ids[0]);
        return it == f.triuSupport.end() ? nullptr : &it->second;
    };
    // Output rows made dirty by dirty rows of B.
    auto dotForwardFromB = [&](const std::vector<Region> &suppA, const std::vector<Region> &dirtyB,
                               uint32_t mRows) {
        std::vector<Region> outBoxes;
        for (const Region &b : dirtyB)
            for (const Region &a : suppA)
            {
                Region probe;
                probe.region = {b.region[0], {0, mRows}, b.region[1]};
                Region hit = intersectRegions(a, probe);
                if (dropEmpty({hit}).empty())
                    continue;
                Region o;
                o.region = {hit.region[0], hit.region[1], b.region[2]};
                outBoxes.push_back(o);
            }
        return dropEmpty(mergeRegions(outBoxes));
    };
    auto dotForwardFromBTriu = [&](const TriuBound &bound, const std::vector<Region> &dirtyB,
                                   uint32_t mRows) {
        std::vector<Region> outBoxes;
        for (const Region &b : dirtyB)
        {
            if (b.region.size() != 3)
                continue;
            const uint32_t k0 = b.region[1].start;
            const int64_t m_min = std::max<int64_t>(0, static_cast<int64_t>(k0) - bound.diag_k + 1);
            if (m_min < static_cast<int64_t>(mRows))
            {
                Region o;
                o.region = {b.region[0], {static_cast<uint32_t>(m_min), mRows}, b.region[2]};
                outBoxes.push_back(o);
            }
        }
        return dropEmpty(mergeRegions(outBoxes));
    };
    // Rows of B needed for the needed output region.
    auto dotNeededB = [&](const std::vector<Region> &suppA, const std::vector<Region> &neededOut, uint32_t kCols) {
        std::vector<Region> boxes;
        for (const Region &o : neededOut)
            for (const Region &a : suppA)
            {
                Region probe;
                probe.region = {o.region[0], o.region[1], {0, kCols}};
                Region hit = intersectRegions(a, probe);
                if (dropEmpty({hit}).empty())
                    continue;
                Region bBox;
                bBox.region = {hit.region[0], hit.region[2], o.region[2]};
                boxes.push_back(bBox);
            }
        return dropEmpty(mergeRegions(boxes));
    };
    auto dotNeededBTriu = [&](const TriuBound &bound, const std::vector<Region> &neededOut,
                              uint32_t kCols) {
        std::vector<Region> boxes;
        for (const Region &o : neededOut)
        {
            if (o.region.size() != 3)
                continue;
            const uint32_t m1 = o.region[1].stop;
            if (m1 == 0)
                continue;
            const int64_t k_limit = std::max<int64_t>(0, static_cast<int64_t>(m1 - 1) + bound.diag_k);
            const uint32_t k_stop = std::min<uint32_t>(kCols, static_cast<uint32_t>(k_limit));
            if (k_stop > 0)
            {
                Region bBox;
                bBox.region = {o.region[0], {0, k_stop}, o.region[2]};
                boxes.push_back(bBox);
            }
        }
        return dropEmpty(mergeRegions(boxes));
    };

    if (debugEnvTgDebugMissing())
    {
        std::cout << "[debug support] finite=" << finite.size() << " support=" << support.size() << std::endl;
        std::cout << "[debug masks] masks=" << masks.size() << std::endl;
    }
    // Elements affected by the given input regions (forward through the ops).
    auto forwardFrom = [&](const std::unordered_map<LogicalId, std::vector<Region>> &seeds) {
        std::unordered_map<LogicalId, std::vector<Region>> reach;
        static const std::vector<Region> none;
        auto of = [&](LogicalId id) -> const std::vector<Region> & {
            auto it = reach.find(id);
            return it == reach.end() ? none : it->second;
        };
        for (LogicalId id : topo)
        {
            const TensorNode &node = graph.getNode(id);
            if (node.opType == OpType::INPUT)
            {
                auto it = seeds.find(id);
                if (it != seeds.end() && !it->second.empty())
                    reach[id] = dropEmpty(mergeRegions(it->second));
                continue;
            }
            std::vector<std::vector<Region>> childRegions;
            bool anyReached = false;
            for (LogicalId child : node.child_ids)
            {
                childRegions.push_back(of(child));
                anyReached = anyReached || !childRegions.back().empty();
            }
            if (!anyReached)
                continue;
            std::vector<Region> regions;
            if (const TriuBound *triuSuppA = dotTriuSupport(node))
            {
                std::vector<std::vector<Region>> fromA = {childRegions[0], {}};
                regions = dropEmpty(prop.forward(node, graph, fromA));
                if (!childRegions[1].empty())
                    regions = unite(regions, dotForwardFromBTriu(*triuSuppA, childRegions[1],
                                                                 graph.getNode(node.child_ids[0]).getShape()[1]));
            }
            else if (const std::vector<Region> *suppA = dotSupport(node))
            {
                std::vector<std::vector<Region>> fromA = {childRegions[0], {}};
                regions = dropEmpty(prop.forward(node, graph, fromA));
                if (!childRegions[1].empty())
                    regions = unite(regions, dotForwardFromB(*suppA, childRegions[1],
                                                             graph.getNode(node.child_ids[0]).getShape()[1]));
            }
            else
                regions = dropEmpty(prop.forward(node, graph, childRegions));
            auto mask = masks.find(id);
            if (mask != masks.end() && of(node.child_ids[mask->second.first]).empty())
            {
                auto triuIt = f.triuMasks.find(id);
                if (triuIt != f.triuMasks.end())
                {
                    std::vector<Region> restricted;
                    for (const auto &r : regions)
                    {
                        auto hit = intersectBoxWithTriuBound(r, triuIt->second);
                        if (hit)
                            restricted.push_back(*hit);
                    }
                    regions = dropEmpty(mergeRegions(restricted));
                }
                else
                {
                    regions = intersect(regions, mask->second.second);
                }
            }
            if (!regions.empty())
                reach[id] = std::move(regions);
        }
        return reach;
    };
    static const bool time_regions = std::getenv("TG_TIME_REGIONS") != nullptr;
    auto phase_start = std::chrono::steady_clock::now();
    auto phase = [&](const char *name) {
        if (!time_regions)
            return;
        const auto now = std::chrono::steady_clock::now();
        std::cout << "[regions timing] " << name << " "
                  << std::chrono::duration<double, std::milli>(now - phase_start).count() << " ms" << std::endl;
        phase_start = now;
    };
    out.dirty = forwardFrom(bucket.inputDirtyRegions);
    phase("forward dirty");

    // Padding past this bucket's dirty region: values depending on it are
    // recomputed by whichever later run first covers it (that run's dirty
    // region reaches them), so this run need not keep them valid.
    std::unordered_map<LogicalId, std::vector<Region>> padding;
    if (paddedInputs)
    {
        std::unordered_map<LogicalId, std::vector<Region>> paddingSeeds;
        for (const auto &[inputId, axis] : *paddedInputs)
        {
            auto it = bucket.inputDirtyRegions.find(inputId);
            if (it == bucket.inputDirtyRegions.end() || it->second.empty())
                continue;
            const std::vector<uint32_t> &shape = graph.getNode(inputId).getShape();
            if (axis >= shape.size())
                Error::throw_err("[computeBucketRegions] padded input axis out of range");
            uint32_t end = 0;
            for (const Region &r : it->second)
            {
                if (r.region.size() != shape.size())
                    Error::throw_err("[computeBucketRegions] dirty region rank mismatch for a padded input");
                end = std::max(end, r.region[axis].stop);
            }
            if (end >= shape[axis])
                continue;
            Region pad = makeFull(shape).front();
            pad.region[axis] = {end, shape[axis]};
            paddingSeeds[inputId] = {pad};
        }
        padding = forwardFrom(paddingSeeds);
        phase("forward padding");
        // Elements that depend on padding are no longer dirty: no run reads
        // them as clean (a later run that needs them has the padding in its
        // dirty region) and none needs them now.
        for (auto it = out.dirty.begin(); it != out.dirty.end();)
        {
            auto pad = padding.find(it->first);
            if (pad != padding.end())
                it->second = subtract(it->second, pad->second);
            if (it->second.empty())
                it = out.dirty.erase(it);
            else
                ++it;
        }
        phase("subtract padding");
    }

    // Backward from the root's needed region. With `live`, every non-output
    // value is also needed on its dirty elements that do not depend on input
    // padding and that some run can read (live: needed for the full output).
    auto backward = [&](std::unordered_map<LogicalId, std::vector<Region>> &target,
                        const std::vector<Region> &rootNeeded,
                        const std::unordered_map<LogicalId, std::vector<Region>> *live) {
        static const std::vector<Region> none;
        auto neededIn = [&](LogicalId id) -> const std::vector<Region> & {
            auto it = target.find(id);
            return it == target.end() ? none : it->second;
        };
        target[rootId] = dropEmpty(mergeRegions(rootNeeded));
        uint32_t widenings = 0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            const LogicalId id = *it;
            const TensorNode &node = graph.getNode(id);
            // Buckets keep cached values valid: every value but the output is
            // recomputed on its whole dirty region (not only where the output
            // needs it), so whatever a bucket leaves in a persistent buffer
            // matches the current inputs and any later bucket may read it as
            // clean.
            if (live && id != rootId && !out.dirtyOf(id).empty())
            {
                std::vector<Region> keep = out.dirtyOf(id);
                auto alive = live->find(id);
                keep = alive == live->end() ? std::vector<Region>{} : intersect(keep, alive->second);
                // Masked elements (outside the finite facts) only matter through
                // being far below the unmasked ones, which a stale masked value
                // still is: they need not be refreshed.
                auto triuFin = f.triuFinite.find(id);
                if (triuFin != f.triuFinite.end())
                    keep = boundedIntersect(keep, triuFin->second);
                else
                {
                    auto fin = finite.find(id);
                    if (fin != finite.end())
                        keep = intersect(keep, fin->second);
                }
                // Elements outside a value's support are exactly zero in every run,
                // so their cached value stays right.
                auto triuSupp = f.triuSupport.find(id);
                if (triuSupp != f.triuSupport.end())
                    keep = boundedIntersect(keep, triuSupp->second);
                else
                {
                    auto supp = support.find(id);
                    if (supp != support.end())
                        keep = intersect(keep, supp->second);
                }
                if (!keep.empty())
                    target[id] = unite(neededIn(id), keep);
            }
            const std::vector<Region> needed = neededIn(id);
            if (needed.empty() || node.child_ids.empty())
                continue;
            std::vector<std::vector<Region>> childNeeded = prop.backward(node, graph, needed);
            if (childNeeded.size() != node.child_ids.size())
                Error::throw_err("[computeBucketRegions] backward region count mismatch for " + toString(node.opType));
            if (const TriuBound *triuSuppA = dotTriuSupport(node))
            {
                childNeeded[1] = dotNeededBTriu(*triuSuppA, needed, graph.getNode(node.child_ids[0]).getShape()[2]);
                childNeeded[0] = boundedIntersect(childNeeded[0], *triuSuppA);
            }
            else if (const std::vector<Region> *suppA = dotSupport(node))
            {
                childNeeded[1] = dotNeededB(*suppA, needed, graph.getNode(node.child_ids[0]).getShape()[2]);
                // Columns of A outside its support are exactly zero in every run
                // (the masks are graph constants), so they are not needed.
                childNeeded[0] = boundedIntersect(childNeeded[0], *suppA);
            }
            // A sum needs its input only where it can be nonzero; a max only
            // where it is unmasked, for boxes whose every row keeps an unmasked
            // element (masked ones sit ~1e9 below them).
            if ((node.opType == OpType::SUM || node.opType == OpType::MAX) && node.child_ids.size() == 2)
            {
                auto triuFact = node.opType == OpType::SUM ? f.triuSupport.find(node.child_ids[0])
                                                           : f.triuFinite.find(node.child_ids[0]);
                auto axisData = graph.constantStaging.find(node.child_ids[1]);
                if (triuFact != (node.opType == OpType::SUM ? f.triuSupport.end() : f.triuFinite.end()) &&
                    axisData != graph.constantStaging.end() && axisData->second->size() == sizeof(int32_t))
                {
                    int32_t axis;
                    std::memcpy(&axis, axisData->second->data(), sizeof(axis));
                    const int32_t rank = static_cast<int32_t>(graph.getNode(node.child_ids[0]).getShape().size());
                    if (axis < 0)
                        axis += rank;
                    if (axis < 0 || axis >= rank)
                        Error::throw_err("[computeBucketRegions] reduction axis out of range");
                    if (node.opType == OpType::SUM)
                        childNeeded[0] = boundedIntersect(childNeeded[0], triuFact->second);
                    else
                    {
                        std::vector<Region> restricted;
                        for (const Region &box : dropEmpty(childNeeded[0]))
                        {
                            if (bucket_regions::axisRange(triuFact->second, box, static_cast<size_t>(axis), true))
                            {
                                auto hit = intersectBoxWithTriuBound(box, triuFact->second);
                                if (hit)
                                    restricted.push_back(*hit);
                            }
                            else
                                restricted.push_back(box);
                        }
                        childNeeded[0] = dropEmpty(mergeRegions(restricted));
                    }
                }
                else
                {
                    const auto &facts = node.opType == OpType::SUM ? support : finite;
                    auto valid = facts.find(node.child_ids[0]);
                    if (valid != facts.end() && axisData != graph.constantStaging.end() &&
                        axisData->second->size() == sizeof(int32_t))
                    {
                        int32_t axis;
                        std::memcpy(&axis, axisData->second->data(), sizeof(axis));
                        const int32_t rank = static_cast<int32_t>(graph.getNode(node.child_ids[0]).getShape().size());
                        if (axis < 0)
                            axis += rank;
                        if (axis < 0 || axis >= rank)
                            Error::throw_err("[computeBucketRegions] reduction axis out of range");
                        if (node.opType == OpType::SUM)
                            childNeeded[0] = boundedIntersect(childNeeded[0], valid->second);
                        else
                        {
                            std::vector<Region> restricted;
                            for (const Region &box : dropEmpty(childNeeded[0]))
                            {
                                if (bucket_regions::axisRange(valid->second, box, static_cast<size_t>(axis), true))
                                {
                                    for (Region r : boundedIntersect({box}, valid->second))
                                        restricted.push_back(r);
                                }
                                else
                                    restricted.push_back(box);
                            }
                            childNeeded[0] = dropEmpty(mergeRegions(restricted));
                        }
                    }
                }
            }
            if (debugEnvTgDebugMissing() && widenings < 12)
            {
                const bool outPartial = !(needed.size() == 1 && isFull(needed.front(), node.getShape()));
                for (size_t c = 0; c < node.child_ids.size() && outPartial; ++c)
                {
                    const TensorNode &child = graph.getNode(node.child_ids[c]);
                    const auto r = dropEmpty(childNeeded[c]);
                    if (r.size() == 1 && isFull(r.front(), child.getShape()) && countElements(child.getShape()) > 64 &&
                        graph.constantStaging.count(node.child_ids[c]) == 0)
                    {
                        std::cout << "[debug widen] " << toString(node.opType) << " " << id.value << " out "
                                  << toString(node.getShape()) << " needed " << encodeRegionList(needed) << " -> child "
                                  << c << " " << toString(child.opType) << " " << toString(child.getShape()) << std::endl;
                        ++widenings;
                    }
                }
            }
            auto mask = masks.find(id);
            for (size_t c = 0; c < node.child_ids.size(); ++c)
            {
                std::vector<Region> regions = dropEmpty(childNeeded[c]);
                if (mask != masks.end() && c != mask->second.first)
                    regions = intersect(regions, mask->second.second);
                if (regions.empty())
                    continue;
                auto &slot = target[node.child_ids[c]];
                slot = unite(slot, regions);
            }
        }
    };
    if (paddedInputs)
    {
        // Buckets keep cached values valid (see `needed` above).
        std::shared_ptr<const std::unordered_map<LogicalId, std::vector<Region>>> live;
        {
            std::lock_guard<std::mutex> lock(f.liveCache->mutex);
            if (!f.liveCache->live || f.liveCache->root != rootId)
            {
                auto computed = std::make_shared<std::unordered_map<LogicalId, std::vector<Region>>>();
                backward(*computed, makeFull(graph.getNode(rootId).getShape()), nullptr);
                f.liveCache->live = std::move(computed);
                f.liveCache->root = rootId;
            }
            live = f.liveCache->live;
        }
        phase("live");
        // Dirty elements no output can ever depend on (e.g. attention scores
        // that the mask removes) are never read: they need not be refreshed,
        // so they are not dirty either.
        for (auto it = out.dirty.begin(); it != out.dirty.end();)
        {
            auto alive = live->find(it->first);
            if (it->first != rootId)
                it->second = alive == live->end() ? std::vector<Region>{} : intersect(it->second, alive->second);
            if (it->second.empty())
                it = out.dirty.erase(it);
            else
                ++it;
        }
        phase("dirty within live");
        backward(out.needed, bucket.outputNeededRegion, live.get());
        phase("backward needed");
    }
    else
        backward(out.needed, bucket.outputNeededRegion, nullptr);

    if (debugEnvTgDebugMissing())
    {
        static const bool full_lists = std::getenv("TG_DEBUG_REGIONS_FULL") != nullptr;
        auto box = [](const std::vector<Region> &r) {
            if (r.empty())
                return std::string("-");
            return full_lists ? encodeRegionList(r) : encodeRegion(boundingBox(r));
        };
        for (LogicalId id : topo)
        {
            const TensorNode &node = graph.getNode(id);
            static const bool all_nodes = std::getenv("TG_DEBUG_REGIONS_ALL") != nullptr;
            if (node.opType != OpType::DOT && !all_nodes)
                continue;
            std::cout << "[debug regions] " << toString(node.opType) << " " << id.value << " " << toString(node.getShape())
                      << " dirty=" << box(out.dirtyOf(id)) << " needed=" << box(out.neededOf(id))
                      << " live=" << box(out.liveOf(id)) << std::endl;
        }
    }
    return out;
}
