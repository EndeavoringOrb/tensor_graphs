#pragma once
#include "core/hardware.hpp"
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchUnifiedCopy_CPU_OpenCL(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    auto &caps = HardwareCaps::get();
    // Only valid on OpenCL with unified memory
    if (!caps.has_opencl || !caps.has_unified_memory)
        return false;

    return inputs[0].getShape() == output.getShape() && inputs[0].strides == output.strides && inputs[0].dtype == output.dtype;
}

inline void inferViewUnified_CPU_OpenCL(TensorNode &node, const std::vector<TensorNode> &inputs, const Graph &graph)
{
    node.strides = inputs[0].strides;
    node.viewOffset = inputs[0].viewOffset;
}

inline uint32_t refFactoryUnifiedCopy_CPU_OPENCL(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.copyto(inputs[0], Backend::OPENCL);
}

REGISTER_KERNEL_VIEW("UnifiedCopyTo_CPU_OPENCL", 1, matchUnifiedCopy_CPU_OpenCL, refFactoryUnifiedCopy_CPU_OPENCL, inferViewUnified_CPU_OpenCL, {Backend::OPENCL}, {DType::ANY}, {{1024}}, {false}, {{Backend::CPU}});