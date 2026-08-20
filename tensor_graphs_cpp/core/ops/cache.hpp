#pragma once
#include "core/ops/common.hpp"

struct CacheOp
{
    static constexpr OpType op_type = OpType::CACHE;
    static constexpr const char *name = "CACHE";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId, Graph &)
    {
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        return forwardElementwise(node, graph, parentRegions);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &,
                                                           const std::vector<Region> &outputRegions)
    {
        return backwardElementwise(node.child_ids.size(), outputRegions);
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