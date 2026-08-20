#pragma once
#include "core/ops/common.hpp"

struct FusedOp
{
    static constexpr OpType op_type = OpType::FUSED;
    static constexpr const char *name = "FUSED";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId, Graph &)
    {
        Error::throw_err("Only atomic nodes should have their shape inferred directly.");
    }

    static std::vector<Region> forwardRegion(const TensorNode &, const Graph &,
                                             const std::vector<std::vector<Region>> &)
    {
        Error::throw_err("Region forward is not supported on FUSED op.");
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &, const Graph &,
                                                           const std::vector<Region> &)
    {
        Error::throw_err("Region backward is not supported on FUSED op.");
    }

    static bool isConstant(uint64_t, uint64_t)
    {
        return false;
    }

    static LogicalId buildPattern(Graph &, const std::vector<LogicalId> &, DType)
    {
        return LogicalId();
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,        name,    is_elementwise, inferShape,  forwardRegion,
                        backwardRegion, nullptr, isConstant,     buildPattern};
    }
};