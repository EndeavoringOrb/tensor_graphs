#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>

bool matchDotF32_3D_CUDA(const std::vector<TensorNode> &inputs, const TensorNode &output);

void runDotF32_3D_CUDA(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                       const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews);

#ifdef USE_CUDA
REGISTER_REF_KERNEL(OpType::DOT, Backend::CUDA, matchDotF32_3D_CUDA, runDotF32_3D_CUDA);
#endif