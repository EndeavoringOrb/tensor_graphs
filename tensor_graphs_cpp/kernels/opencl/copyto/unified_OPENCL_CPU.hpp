#pragma once
#include "core/hardware.hpp"
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchUnifiedCopy_OpenCL_CPU(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    auto &caps = HardwareCaps::get();
    // Only valid on OpenCL with unified memory
    if (!caps.has_opencl || !caps.has_unified_memory)
        return false;

    return inputs[0].getShape() == output.getShape() && inputs[0].strides == output.strides && inputs[0].dtype == output.dtype;
}

inline void inferViewUnified_OpenCL_CPU(TensorNode &node, const std::vector<TensorNode> &inputs, const Graph &graph)
{
    node.strides = inputs[0].strides;
    node.viewOffset = inputs[0].viewOffset;
}

inline uint32_t refFactoryUnifiedCopy_OPENCL_CPU(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.copyto(inputs[0], Backend::CPU);
}

REGISTER_KERNEL_VIEW("UnifiedCopyTo_OPENCL_CPU", 1, matchUnifiedCopy_OpenCL_CPU, refFactoryUnifiedCopy_OPENCL_CPU, inferViewUnified_OpenCL_CPU, {Backend::CPU}, {DType::ANY}, {{1024}}, {false}, {{Backend::OPENCL}});