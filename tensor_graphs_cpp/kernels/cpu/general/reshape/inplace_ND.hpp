// File: tensor_graphs_cpp/kernels/cpu/general/reshape/inplace_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchReshapeInplaceND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2)
        return false;
    if (!inputs[0].view.isContiguous())
        return false;
    if (inputs[0].view.baseOffset != output.view.baseOffset || inputs[0].view.dtype != output.view.dtype)
        return false;
    if (inputs[0].storageType == StorageType::PERSISTENT)
        return false;
    auto it = refCounts.find(inputs[0].id);
    if (it == refCounts.end() || it->second != 1)
        return false;
    return countElements(inputs[0].shape) == countElements(output.shape);
}

inline void runReshapeInplaceND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    return;
}

inline uint32_t refFactoryReshapeInplaceND(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.reshape(inputs[0], inputs[1]);
}

REGISTER_KERNEL_INPLACE("Reshape_Inplace", 2, Backend::CPU, matchReshapeInplaceND, runReshapeInplaceND, refFactoryReshapeInplaceND, {DType::FLOAT32, DType::INT32}, {{1}, {1}}, {false, false});