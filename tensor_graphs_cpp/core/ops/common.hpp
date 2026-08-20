#pragma once
#include <algorithm>
#include <sstream>
#include <vector>

#include "core/graph.hpp"
#include "core/ops/op_def.hpp"
#include "core/shapes.hpp"
#include "core/types.hpp"

namespace op_common
{
inline WorkloadMetrics defaultWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                       const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                       DType outDType, double flops)
{
    WorkloadMetrics m;
    m.bytesWritten = static_cast<double>(countElements(outShape) * getDTypeSize(outDType));
    for (size_t i = 0; i < inShapes.size(); ++i)
    {
        DType dt = (i < inDTypes.size()) ? inDTypes[i] : DType::FLOAT32;
        m.bytesRead += static_cast<double>(countElements(inShapes[i]) * getDTypeSize(dt));
    }
    m.flops = flops;
    return m;
}

inline bool noStructuralConstants(uint64_t, uint64_t)
{
    return false;
}

inline LogicalId noPattern(Graph &, const std::vector<LogicalId> &, DType)
{
    return LogicalId();
}
} // namespace op_common

// Base template providing shared implementations for all element-wise ops
template <typename Derived> struct ElementwiseOpBase
{
    static constexpr bool is_elementwise = true;

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

    static WorkloadMetrics computeWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                           const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                           DType outDType, const std::string &)
    {
        return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType,
                                          static_cast<double>(countElements(outShape)));
    }

    static OpTraits traits()
    {
        return OpTraits{Derived::op_type,         Derived::name,          Derived::is_elementwise,
                        Derived::inferShape,      Derived::forwardRegion, Derived::backwardRegion,
                        Derived::computeWorkload, Derived::isConstant,    Derived::buildPattern};
    }
};

// Base for binary element-wise ops (ADD, MUL, DIVIDE, POWER, LT, EQ, AND, OR)
template <typename Derived> struct ElementwiseBinaryOp : public ElementwiseOpBase<Derived>
{
    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        inferShapeElementwise(nodeId, graph, Derived::name);
    }
};

// Base for unary element-wise ops (SIN, COS, NEGATE, LOG, NOT, CAST, CONTIGUOUS, COPY_TO)
template <typename Derived> struct ElementwiseUnaryOp : public ElementwiseOpBase<Derived>
{
    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        inferShapeUnary(nodeId, graph);
    }
};