#pragma once
#include "core/hardware.hpp"
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

inline bool matchCopy_OpenCL_CPU(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape() == output.getShape() &&
           inputs[0].strides == output.strides &&
           inputs[0].dtype == output.dtype &&
           isContiguous(output);
}

inline void runCopy_OpenCL_CPU(const KernelContext &ctx)
{
    // Ensure the GPU has finished writing to the mapped buffer
    clFinish(OpenCLState::get().queue);

    const uint8_t *src = static_cast<const uint8_t *>(ctx.inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(ctx.outputs[0]);

    uint64_t numElements = countElements(ctx.inViews[0].getShape());
    uint64_t elemSize = getDTypeSize(ctx.inViews[0].dtype);

    std::memcpy(dst, src, numElements * elemSize);
}

REGISTER_REF_KERNEL(OpType::COPY_TO, 1, matchCopy_OpenCL_CPU, runCopy_OpenCL_CPU, {Backend::CPU}, {DType::ANY}, {{8, 32}}, {true}, {{Backend::OPENCL}});