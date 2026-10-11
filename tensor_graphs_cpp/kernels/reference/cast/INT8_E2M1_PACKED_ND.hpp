#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchCastINT8_E2M1_PACKED_INT8(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (output.dtype != DType::E2M1_PACKED_INT8)
        return false;
    return true;
}

inline void inferViewCastINT8_E2M1_PACKED_INT8(const std::vector<TensorNode> &inputs, TensorView &output,
                                               const Graph &graph)
{
    output.strides = inputs[0].strides;
}

REGISTER_REF_KERNEL_VIEW(OpType::CAST, 1, 1, matchCastINT8_E2M1_PACKED_INT8, inferViewCastINT8_E2M1_PACKED_INT8,
                         MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::INT8}, {{8, 32}}, {false},
                         {MemSpace(1, HandleType::CPP)});