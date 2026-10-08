// tensor_graphs_cpp/core/plan/propagators/base.hpp
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/constants.hpp"
#include "core/kernels.hpp"
#include "core/logging.hpp"
#include "core/plan/domain.hpp"
#include "core/plan/search_state.hpp"

namespace plan
{

constexpr uint8_t varTypeMask(VarType type)
{
    return static_cast<uint8_t>(1u << static_cast<uint8_t>(type));
}

constexpr uint8_t kAllVarTypesMask =
    varTypeMask(VarType::CACHED) | varTypeMask(VarType::SELECTED) |
    varTypeMask(VarType::START) | varTypeMask(VarType::OFFSET);

enum class StartSelectionGuard : uint8_t
{
    NONE,
    NON_OPTIONAL,
    FIXED_POSITIVE,
    NON_OPTIONAL_WITH_CONSUMERS,
    FIXED_NON_OPTIONAL_WITH_CONSUMERS
};

class Propagator
{
  public:
    virtual ~Propagator() = default;
    virtual std::string name() const = 0;
    virtual uint8_t interestedVarTypes() const { return kAllVarTypesMask; }
    virtual StartSelectionGuard startSelectionGuard() const { return StartSelectionGuard::NONE; }

    // Shrinks variable domains in state. Returns false on contradiction.
    virtual bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) = 0;
};

// ============================================================================
// Helper functions for op inspection and graph queries
// ============================================================================

inline bool isOpCacheOrScatter(const ENode &enode)
{
    if (enode.getOpType() == OpType::CACHE || enode.getOpType() == OpType::SCATTER)
        return true;
    KernelId kid = enode.getKernelId();
    if (kid.value != 0 && KernelRegistry::get().hasKernel(kid))
    {
        const auto &entry = KernelRegistry::get().getKernel(kid);
        if (entry.opType == OpType::CACHE || entry.opType == OpType::SCATTER)
            return true;
        if (entry.opType == OpType::FUSED && entry.refFactory)
        {
            Graph kGraph;
            std::vector<LogicalId> kInputs;
            for (uint64_t i = 0; i < entry.min_num_inputs; ++i)
                kInputs.push_back(kGraph.input(entry.dummyShapes.size() > i ? entry.dummyShapes[i] : std::vector<uint32_t>{1},
                                               entry.dtypes.size() > i ? entry.dtypes[i] : DType::FLOAT32));
            entry.refFactory(kInputs, kGraph);
            for (const auto &pair : kGraph.nodes)
            {
                if (pair.second.opType == OpType::CACHE || pair.second.opType == OpType::SCATTER)
                    return true;
            }
        }
    }
    return false;
}

inline bool isOpRootScatterOrCache(const ENode &enode)
{
    if (enode.getOpType() == OpType::CACHE || enode.getOpType() == OpType::SCATTER)
        return true;
    KernelId kid = enode.getKernelId();
    if (kid.value != 0 && KernelRegistry::get().hasKernel(kid))
    {
        const auto &entry = KernelRegistry::get().getKernel(kid);
        if (entry.opType == OpType::SCATTER)
            return true;
        if (entry.opType == OpType::FUSED && entry.refFactory)
        {
            Graph kGraph;
            std::vector<LogicalId> kInputs;
            for (uint64_t i = 0; i < entry.min_num_inputs; ++i)
                kInputs.push_back(kGraph.input(entry.dummyShapes.size() > i ? entry.dummyShapes[i] : std::vector<uint32_t>{1},
                                               entry.dtypes.size() > i ? entry.dtypes[i] : DType::FLOAT32));
            LogicalId rootId = entry.refFactory(kInputs, kGraph);
            if (kGraph.getNode(rootId).opType == OpType::SCATTER)
                return true;
        }
    }
    return false;
}

} // namespace plan
