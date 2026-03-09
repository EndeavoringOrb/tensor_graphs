#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchReshapeInplaceND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;
    if (inputs[0].view.baseOffset != output.view.baseOffset || inputs[0].view.dtype != output.view.dtype)
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

REGISTER_KERNEL_INPLACE("Reshape_Inplace", 2, Backend::CPU, matchReshapeInplaceND, runReshapeInplaceND, refFactoryReshapeInplaceND, {DType::FLOAT32, DType::INT32}, {{1}, {1}}, {false, false}); // TODO: for this kernel input[0] doesn't have to have dtype float32, maybe we should update signature to allow list of dtypes instead of single dtype