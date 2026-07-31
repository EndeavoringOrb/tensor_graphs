#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchFillView(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return true;
}

inline void inferViewFill(const std::vector<TensorNode> &inputs, TensorView &output, const Graph &graph)
{
    output.strides.assign(output.getShape().size(), 0);
}

REGISTER_REF_KERNEL_VIEW(OpType::FILL, 2, 2, matchFillView, inferViewFill, MemSpace(1, HandleType::CPP),
                         {Engine(0, EngineType::CPU)}, {DType::ANY, DType::INT32}, {{1}, {1}}, {false, false},
                         {MemSpace(1, HandleType::CPP), MemSpace(1, HandleType::CPP)});