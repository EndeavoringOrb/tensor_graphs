#pragma once
#include "core/types.hpp"

inline bool isLast2DimsContiguous(const TensorNode &node)
{
    const auto &s = node.getShape();
    const auto &st = node.strides;
    if (s.size() < 2 || st.size() < 2)
        return false;
    uint32_t lastDim = s.back();
    return st.back() == 1 && st[st.size() - 2] == lastDim;
}
