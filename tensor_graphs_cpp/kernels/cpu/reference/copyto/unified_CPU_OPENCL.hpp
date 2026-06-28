#pragma once
#include "core/hardware.hpp"
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

inline bool matchCopy_CPU_OpenCL(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape() == output.getShape() &&
           inputs[0].strides == output.strides &&
           inputs[0].dtype == output.dtype &&
           isContiguous(output);
}

inline void runCopy_CPU_OpenCL(const KernelContext &ctx)
{
    // Because the OpenCL arena is mapped via CL_MEM_ALLOC_HOST_PTR,
    // we can copy data into it directly using standard CPU memcpy.
    // The GPU will then compute directly on this buffer.
    const uint8_t *src = static_cast<const uint8_t *>(ctx.inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(ctx.outputs[0]);

    uint64_t numElements = countElements(ctx.inViews[0].getShape());
    uint64_t elemSize = getDTypeSize(ctx.inViews[0].dtype);

    std::memcpy(dst, src, numElements * elemSize);
}

REGISTER_REF_KERNEL(OpType::COPY_TO, 1, matchCopy_CPU_OpenCL, runCopy_CPU_OpenCL, {Backend::OPENCL}, {DType::ANY}, {{8, 32}}, {true}, {{Backend::CPU}});