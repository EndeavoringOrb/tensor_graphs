// tensor_graphs_cpp/core/plan/bucket_families.hpp
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/egraph.hpp"
#include "core/types.hpp"

// Shift families (see FamilyRegionNormalizer): one plan is searched for a
// representative member; every other member's compiled graph is the
// representative's with the values derived from moving region bounds
// relocated. The relocation is read off the difference between the
// representative's compiled graph and a witness member's (same extraction
// applied to the witness's own egraph), and checked against a third member.
namespace bucket_families
{

// Values of a compiled graph that move with the family shift: per unit of
// shift, view byte offsets and integer constant elements change by a fixed
// amount.
struct GraphRelocation
{
    struct ViewOffset
    {
        EClassId id;
        int64_t perShift;
    };
    struct ConstantElements
    {
        EClassId id;
        DType dtype;
        std::vector<std::pair<uint32_t, int64_t>> perShift; // element index, change per unit shift
    };
    std::vector<ViewOffset> views;
    std::vector<ConstantElements> constants;
};

inline bool sameBuffer(const ParallelBuffer &a, const ParallelBuffer &b)
{
    return a.id == b.id && a.mem_space == b.mem_space && a.size == b.size && a.start == b.start && a.end == b.end &&
           a.offset == b.offset;
}

// Everything but view offsets and constant bytes must match.
inline bool sameStructure(const CompiledGraph &a, const CompiledGraph &b, std::string &why)
{
    if (a.instructions.size() != b.instructions.size())
    {
        why = "instruction count " + std::to_string(a.instructions.size()) + " vs " +
              std::to_string(b.instructions.size());
        return false;
    }
    for (size_t i = 0; i < a.instructions.size(); ++i)
    {
        const OpInstruction &x = a.instructions[i], &y = b.instructions[i];
        bool same = x.eclass_id == y.eclass_id && x.logical_id == y.logical_id && x.kernel_id == y.kernel_id &&
                    x.children == y.children && sameBuffer(x.outBuffer, y.outBuffer) &&
                    x.inBuffers.size() == y.inBuffers.size() && x.engines == y.engines;
        for (size_t k = 0; same && k < x.inBuffers.size(); ++k)
            same = sameBuffer(x.inBuffers[k], y.inBuffers[k]);
        if (!same)
        {
            why = "instruction " + std::to_string(i) + " differs (" + x.debugOrigin + ")";
            return false;
        }
    }
    if (a.nodeViews.size() != b.nodeViews.size())
    {
        why = "view count differs";
        return false;
    }
    for (const auto &[id, va] : a.nodeViews)
    {
        auto it = b.nodeViews.find(id);
        if (it == b.nodeViews.end() || it->second.getShape() != va.getShape() || it->second.strides != va.strides ||
            it->second.dtype != va.dtype)
        {
            why = "view of class " + std::to_string(id.value) + " differs in shape, strides or dtype";
            return false;
        }
    }
    if (a.constantStaging.size() != b.constantStaging.size())
    {
        why = "constant count differs";
        return false;
    }
    for (const auto &[id, data] : a.constantStaging)
    {
        auto it = b.constantStaging.find(id);
        if (it == b.constantStaging.end() || it->second->size() != data->size())
        {
            why = "constant of class " + std::to_string(id.value) + " missing or resized";
            return false;
        }
    }
    if (a.eclass_to_logical != b.eclass_to_logical)
    {
        why = "class to logical map differs";
        return false;
    }
    return true;
}

inline bool readInteger(const std::vector<uint8_t> &bytes, DType dtype, uint32_t index, int64_t &out)
{
    if (dtype == DType::INT32)
    {
        int32_t v;
        std::memcpy(&v, bytes.data() + static_cast<size_t>(index) * sizeof(v), sizeof(v));
        out = v;
        return true;
    }
    if (dtype == DType::INT64)
    {
        int64_t v;
        std::memcpy(&v, bytes.data() + static_cast<size_t>(index) * sizeof(v), sizeof(v));
        out = v;
        return true;
    }
    return false;
}

inline uint32_t integerCount(const std::vector<uint8_t> &bytes, DType dtype)
{
    return static_cast<uint32_t>(bytes.size() / (dtype == DType::INT64 ? 8 : 4));
}

// Relocation taking `a` (shift sa) to `b` (shift sb). False with a reason when
// the graphs differ in more than linearly moving offsets and integers.
inline bool deriveRelocation(const CompiledGraph &a, int64_t sa, const CompiledGraph &b, int64_t sb,
                             GraphRelocation &out, std::string &why)
{
    const int64_t dt = sb - sa;
    if (dt == 0)
        Error::throw_err("[bucket_families] witness has the representative's shift");
    if (!sameStructure(a, b, why))
        return false;
    out = GraphRelocation{};
    for (const auto &[id, va] : a.nodeViews)
    {
        const int64_t d = static_cast<int64_t>(b.nodeViews.at(id).offset) - static_cast<int64_t>(va.offset);
        if (d == 0)
            continue;
        if (d % dt != 0)
        {
            why = "view offset of class " + std::to_string(id.value) + " moves by " + std::to_string(d) +
                  " for a shift of " + std::to_string(dt);
            return false;
        }
        out.views.push_back({id, d / dt});
    }
    for (const auto &[id, data] : a.constantStaging)
    {
        const std::vector<uint8_t> &other = *b.constantStaging.at(id);
        if (other == *data)
            continue;
        auto view = a.nodeViews.find(id);
        const DType dtype = view == a.nodeViews.end() ? DType::INT32 : view->second.dtype;
        if (dtype != DType::INT32 && dtype != DType::INT64)
        {
            why = "non-integer constant of class " + std::to_string(id.value) + " differs";
            return false;
        }
        GraphRelocation::ConstantElements ce{id, dtype, {}};
        const uint32_t n = integerCount(*data, dtype);
        for (uint32_t i = 0; i < n; ++i)
        {
            int64_t x, y;
            readInteger(*data, dtype, i, x);
            readInteger(other, dtype, i, y);
            if (x == y)
                continue;
            if ((y - x) % dt != 0)
            {
                why = "constant element of class " + std::to_string(id.value) + " moves by " +
                      std::to_string(y - x) + " for a shift of " + std::to_string(dt);
                return false;
            }
            ce.perShift.push_back({i, (y - x) / dt});
        }
        out.constants.push_back(std::move(ce));
    }
    return true;
}

// `base` (a member at some shift) moved by `delta` units of shift.
inline CompiledGraph applyRelocation(const CompiledGraph &base, const GraphRelocation &rel, int64_t delta)
{
    CompiledGraph out = base;
    for (const auto &v : rel.views)
    {
        TensorView &view = out.nodeViews.at(v.id);
        const int64_t offset = static_cast<int64_t>(view.offset) + v.perShift * delta;
        if (offset < 0)
            Error::throw_err("[bucket_families] relocated view offset below zero");
        view.offset = static_cast<uint64_t>(offset);
    }
    for (const auto &c : rel.constants)
    {
        auto data = std::make_shared<std::vector<uint8_t>>(*base.constantStaging.at(c.id));
        for (const auto &[index, perShift] : c.perShift)
        {
            int64_t v;
            readInteger(*data, c.dtype, index, v);
            v += perShift * delta;
            if (c.dtype == DType::INT32)
            {
                if (v < INT32_MIN || v > INT32_MAX)
                    Error::throw_err("[bucket_families] relocated constant out of int32 range");
                const int32_t w = static_cast<int32_t>(v);
                std::memcpy(data->data() + static_cast<size_t>(index) * sizeof(w), &w, sizeof(w));
            }
            else
                std::memcpy(data->data() + static_cast<size_t>(index) * sizeof(v), &v, sizeof(v));
        }
        out.constantStaging[c.id] = std::move(data);
    }
    return out;
}

// Exact equality of what the executor uses (offsets and constant bytes too).
inline bool sameCompiledGraph(const CompiledGraph &a, const CompiledGraph &b, std::string &why)
{
    if (!sameStructure(a, b, why))
        return false;
    for (const auto &[id, va] : a.nodeViews)
        if (b.nodeViews.at(id).offset != va.offset)
        {
            why = "view offset of class " + std::to_string(id.value) + ": " + std::to_string(va.offset) + " vs " +
                  std::to_string(b.nodeViews.at(id).offset);
            return false;
        }
    for (const auto &[id, data] : a.constantStaging)
        if (*b.constantStaging.at(id) != *data)
        {
            why = "constant bytes of class " + std::to_string(id.value) + " differ";
            return false;
        }
    return true;
}

// Same enode, except that constant inputs may hold different values (their
// content hash is the value's).
inline bool sameENodeModuloConstants(const EGraph &a, const EGraph &b, ENodeId e)
{
    const ENode &x = a.getENode(e), &y = b.getENode(e);
    if (x == y)
        return true;
    if (x.getOpType() != OpType::INPUT || y.getOpType() != OpType::INPUT)
        return false;
    const EClassId cx = a.findConst(a.getENodeEClass(e)), cy = b.findConst(b.getENodeEClass(e));
    if (cx != cy || !a.constantStaging.count(cx) || !b.constantStaging.count(cy))
        return false;
    return x.getKernelId() == y.getKernelId() && x.getOpName() == y.getOpName() && x.getChildren() == y.getChildren() &&
           x.getShape() == y.getShape() && x.getStrides() == y.getStrides() && x.getDType() == y.getDType() &&
           x.getMemSpace() == y.getMemSpace() && x.getEngines() == y.getEngines();
}

// Two egraphs with the same classes and enodes in the same order (constant
// values aside), so an extraction of one applies to the other.
inline bool alignedEGraphs(const EGraph &a, const EGraph &b, std::string &why)
{
    // First enode that differs (creation order), for the reason.
    const size_t common = std::min(a.getENodes().size(), b.getENodes().size());
    for (uint32_t e = 0; e < common; ++e)
        if (!sameENodeModuloConstants(a, b, ENodeId{e}))
        {
            const ENode &x = a.getENode(ENodeId{e}), &y = b.getENode(ENodeId{e});
            why = "enode " + std::to_string(e) + " differs: " + toString(x.getOpType()) + ":" + x.getOpName() + " " +
                  toString(x.getShape()) + " from " + x.getDebugOrigin() + " vs " + toString(y.getOpType()) + ":" +
                  y.getOpName() + " " + toString(y.getShape()) + " from " + y.getDebugOrigin();
            return false;
        }
    if (a.getClasses().size() != b.getClasses().size() || a.getENodes().size() != b.getENodes().size())
    {
        why = "sizes differ: classes " + std::to_string(a.getClasses().size()) + " vs " +
              std::to_string(b.getClasses().size()) + ", enodes " + std::to_string(a.getENodes().size()) + " vs " +
              std::to_string(b.getENodes().size());
        return false;
    }
    for (uint32_t c = 0; c < a.getClasses().size(); ++c)
    {
        const EClassId id{c};
        if (a.findConst(id) != b.findConst(id))
        {
            why = "class " + std::to_string(c) + " has different representatives";
            return false;
        }
        if (a.findConst(id) != id)
            continue;
        const EClass &x = a.getEClass(id), &y = b.getEClass(id);
        std::string field;
        if (x.enodes != y.enodes)
            field = "enodes (" + std::to_string(x.enodes.size()) + " vs " + std::to_string(y.enodes.size()) + ")";
        else if (x.shape != y.shape || x.strides != y.strides || x.dtype != y.dtype || x.mem_space != y.mem_space)
            field = "layout";
        else if (x.base_eclass_id != y.base_eclass_id)
            field = "base class";
        else if (x.is_clean != y.is_clean)
            field = std::string("is_clean (") + (x.is_clean ? "clean" : "dirty") + " in the representative)";
        if (!field.empty())
        {
            why = "class " + std::to_string(c) + " (shape " + toString(x.shape) + ", logical " +
                  std::to_string(x.logical_id.value) + ") differs in " + field;
            return false;
        }
        if ((a.constantStaging.count(id) != 0) != (b.constantStaging.count(id) != 0))
        {
            why = "class " + std::to_string(c) + " is constant in one egraph only";
            return false;
        }
    }
    return true;
}

} // namespace bucket_families
