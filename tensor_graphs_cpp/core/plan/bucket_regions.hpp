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
// with m <= kMaskedValue are masked. Their sum stays far below every
// unmasked value of a softmax row, so exp() of it is exactly 0 whatever x is.
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
    static constexpr float kMaskedValue = -1e8f;

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

    // After editing dirty/needed (family normalization).
    void invalidateLive()
    {
        live_.clear();
    }

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
            while (c < cols && data[row * cols + c] <= BucketRegions::kMaskedValue)
                ++c;
            const uint32_t start = c;
            while (c < cols && data[row * cols + c] > BucketRegions::kMaskedValue)
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
        switch (node.opType)
        {
        case OpType::ADD: {
            if (node.child_ids.size() != 2)
                break;
            auto mask = masks.find(id);
            fin = mask != masks.end() ? intersect(mask->second.second, finiteOf(node.child_ids[1 - mask->second.first]))
                                      : intersect(finiteOf(node.child_ids[0]), finiteOf(node.child_ids[1]));
            hasFin = true;
            break;
        }
        case OpType::POWER: {
            const auto base = constantMin(node.child_ids[0], 8);
            if (base && *base > 1.0f)
            {
                sup = finiteOf(node.child_ids[1]);
                hasSup = true;
            }
            break;
        }
        case OpType::MUL:
            sup = intersect(supportOf(node.child_ids[0]), supportOf(node.child_ids[1]));
            hasSup = true;
            break;
        case OpType::DIVIDE:
            sup = supportOf(node.child_ids[0]);
            hasSup = true;
            break;
        case OpType::REPEAT:
        case OpType::RESHAPE:
        case OpType::PERMUTE:
        case OpType::CONTIGUOUS:
        case OpType::SLICE:
        case OpType::CONCAT:
            fin = mapForward(node, finiteOf);
            sup = mapForward(node, supportOf);
            hasFin = hasSup = true;
            break;
        default:
            break;
        }
        if (hasFin && !isFullList(id, fin))
            finite[id] = std::move(fin);
        if (hasSup && !isFullList(id, sup))
            support[id] = std::move(sup);
    }
    return facts;
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
            if (const std::vector<Region> *suppA = dotSupport(node))
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
                regions = intersect(regions, mask->second.second);
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
                auto fin = finite.find(id);
                if (fin != finite.end())
                    keep = intersect(keep, fin->second);
                // Elements outside a value's support are exactly zero in every run,
                // so their cached value stays right.
                auto supp = support.find(id);
                if (supp != support.end())
                    keep = intersect(keep, supp->second);
                if (!keep.empty())
                    target[id] = unite(neededIn(id), keep);
            }
            const std::vector<Region> needed = neededIn(id);
            if (needed.empty() || node.child_ids.empty())
                continue;
            std::vector<std::vector<Region>> childNeeded = prop.backward(node, graph, needed);
            if (childNeeded.size() != node.child_ids.size())
                Error::throw_err("[computeBucketRegions] backward region count mismatch for " + toString(node.opType));
            if (const std::vector<Region> *suppA = dotSupport(node))
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
                const auto &facts = node.opType == OpType::SUM ? support : finite;
                auto valid = facts.find(node.child_ids[0]);
                auto axisData = graph.constantStaging.find(node.child_ids[1]);
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


// Shift families: buckets whose inputs are one bucket shifted by a per-member
// amount (e.g. the same token block at different positions). The normalizer
// rewrites their regions so that every box bound either moves with the shift
// (bound = bound_first + c * (shift - shift_first), one integer c per box
// dimension for start and stop alike) or is the same in every member: other
// dimensions are widened to their hull over the family (more elements dirty
// or needed: recomputed or kept, always sound). Members then differ only by
// moving bounds, so one plan serves them all after relocating the constants
// derived from those bounds.
//
// Members are added one at a time (regions of a whole family need not be
// held at once); add() returns false when a member's region lists do not
// line up with the first one's (different nodes or box counts), and the
// caller splits the family.
class FamilyRegionNormalizer
{
    struct DimState
    {
        int64_t s0 = 0, e0 = 0;
        int64_t coef = 0;
        bool coefKnown = false;
        bool moving = true;
        uint32_t lo = UINT32_MAX, hi = 0;
    };
    // Per node: per-box states while every member has the same box count;
    // otherwise (or when the node has no region in some member) the node
    // is described by its bounding box (a superset).
    struct NodeState
    {
        std::vector<std::vector<DimState>> boxes;
        std::vector<DimState> bbox;
        bool bboxMode = false;
        size_t present = 0;
    };
    std::unordered_map<LogicalId, NodeState> dirty_, needed_;
    int64_t shift0_ = 0;
    size_t count_ = 0;

    static void initDims(std::vector<DimState> &dims, const Region &box)
    {
        dims.assign(box.region.size(), DimState{});
        for (size_t d = 0; d < dims.size(); ++d)
        {
            dims[d].s0 = dims[d].lo = box.region[d].start;
            dims[d].e0 = dims[d].hi = box.region[d].stop;
        }
    }

    static void updateDims(std::vector<DimState> &dims, const Region &box, int64_t dt)
    {
        for (size_t d = 0; d < dims.size(); ++d)
        {
            DimState &st = dims[d];
            const int64_t s = box.region[d].start, e = box.region[d].stop;
            st.lo = std::min<uint32_t>(st.lo, static_cast<uint32_t>(s));
            st.hi = std::max<uint32_t>(st.hi, static_cast<uint32_t>(e));
            if (!st.moving)
                continue;
            const int64_t ds = s - st.s0, de = e - st.e0;
            if (ds != de || (dt == 0 && ds != 0) || (dt != 0 && ds % dt != 0))
            {
                st.moving = false;
                continue;
            }
            if (dt == 0)
                continue;
            if (!st.coefKnown)
            {
                st.coef = ds / dt;
                st.coefKnown = true;
            }
            else if (ds / dt != st.coef)
                st.moving = false;
        }
    }

    void addMap(std::unordered_map<LogicalId, NodeState> &states,
                const std::unordered_map<LogicalId, std::vector<Region>> &regions, int64_t shift)
    {
        const int64_t dt = shift - shift0_;
        for (const auto &[id, boxes] : regions)
        {
            if (boxes.empty())
                continue;
            NodeState &ns = states[id];
            const Region box = bucket_regions::boundingBox(boxes);
            if (ns.present == 0)
            {
                initDims(ns.bbox, box);
                // A node first seen after the first member cannot move.
                if (count_ > 0)
                {
                    ns.bboxMode = true;
                    for (DimState &st : ns.bbox)
                        st.moving = false;
                }
                else
                    for (const Region &r : boxes)
                    {
                        ns.boxes.emplace_back();
                        initDims(ns.boxes.back(), r);
                    }
            }
            else
            {
                if (ns.bbox.size() != box.region.size())
                    Error::throw_err("[FamilyRegionNormalizer] region rank changes across members");
                updateDims(ns.bbox, box, dt);
                if (!ns.bboxMode && ns.boxes.size() != boxes.size())
                    ns.bboxMode = true;
                if (!ns.bboxMode)
                    for (size_t j = 0; j < boxes.size(); ++j)
                        updateDims(ns.boxes[j], boxes[j], dt);
            }
            ++ns.present;
        }
    }

    void finishMember(std::unordered_map<LogicalId, NodeState> &states)
    {
        // Nodes without regions in this member: bounding box, not moving.
        for (auto &[id, ns] : states)
            if (ns.present != count_ + 1)
            {
                ns.bboxMode = true;
                for (DimState &st : ns.bbox)
                    st.moving = false;
                ns.present = count_ + 1;
            }
    }

    static void applyMap(const std::unordered_map<LogicalId, NodeState> &states,
                         std::unordered_map<LogicalId, std::vector<Region>> &regions)
    {
        std::unordered_map<LogicalId, std::vector<Region>> out;
        for (const auto &[id, ns] : states)
        {
            auto it = regions.find(id);
            const std::vector<Region> own = it == regions.end() ? std::vector<Region>{} : it->second;
            if (ns.bboxMode)
            {
                Region box;
                box.region.resize(ns.bbox.size());
                const Region ownBox = own.empty() ? Region{} : bucket_regions::boundingBox(own);
                for (size_t d = 0; d < ns.bbox.size(); ++d)
                {
                    const DimState &st = ns.bbox[d];
                    box.region[d] = st.moving ? ownBox.region[d] : Dim{st.lo, st.hi};
                }
                out[id] = {box};
                continue;
            }
            if (own.size() != ns.boxes.size())
                Error::throw_err("[FamilyRegionNormalizer] member box count differs from its family's");
            std::vector<Region> boxes = own;
            for (size_t j = 0; j < boxes.size(); ++j)
                for (size_t d = 0; d < boxes[j].region.size(); ++d)
                {
                    const DimState &st = ns.boxes[j][d];
                    if (!st.moving)
                        boxes[j].region[d] = {st.lo, st.hi};
                }
            out[id] = std::move(boxes);
        }
        for (const auto &[id, boxes] : regions)
            if (!boxes.empty() && !states.count(id))
                Error::throw_err("[FamilyRegionNormalizer] member has regions on a node its family has none on");
        regions = std::move(out);
    }

  public:
    bool add(const BucketRegions &regions, int64_t shift)
    {
        if (count_ == 0)
            shift0_ = shift;
        addMap(dirty_, regions.dirty, shift);
        addMap(needed_, regions.needed, shift);
        finishMember(dirty_);
        finishMember(needed_);
        ++count_;
        return true;
    }

    // Rewrites one member's own regions to the family form.
    bool apply(BucketRegions &regions) const
    {
        if (count_ == 0)
            Error::throw_err("[FamilyRegionNormalizer] apply before any member was added");
        applyMap(dirty_, regions.dirty);
        applyMap(needed_, regions.needed);
        regions.invalidateLive();
        return true;
    }

    size_t size() const
    {
        return count_;
    }

    const std::string &why() const
    {
        static const std::string none = "regions always line up (bounding boxes)";
        return none;
    }
};
