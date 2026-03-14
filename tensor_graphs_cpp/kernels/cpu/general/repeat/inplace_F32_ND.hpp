#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchRepeatF32_Inplace_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 3)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].view.baseOffset != output.view.baseOffset)
        return false;
    if (inputs[0].storageType == StorageType::PERSISTENT)
        return false;
    auto it = refCounts.find(inputs[0].id);
    if (it == refCounts.end() || it->second != 1)
        return false;
    return true;
}

inline void runRepeatF32_Inplace_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                    const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    return;
}

inline uint32_t refFactoryRepeatF32_Inplace_ND(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.repeat(inputs[0], inputs[1], inputs[2]);
}

REGISTER_KERNEL_INPLACE("Repeat_Inplace", 3, Backend::CPU, matchRepeatF32_Inplace_ND, runRepeatF32_Inplace_ND, refFactoryRepeatF32_Inplace_ND, {DType::FLOAT32, DType::INT32, DType::INT32}, {{1}, {1}, {1}}, {false, false, false});