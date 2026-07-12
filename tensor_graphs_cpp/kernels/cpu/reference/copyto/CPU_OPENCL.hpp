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
    const uint8_t *src_host_ptr = static_cast<const uint8_t *>(ctx.inputs[0]);
    cl_mem dst_device_buf = ctx.cl_outputs[0];

    uint64_t numElements = countElements(ctx.inViews[0].getShape());
    uint64_t elemSize = getDTypeSize(ctx.inViews[0].dtype);
    size_t size_bytes = numElements * elemSize;

    if (size_bytes == 0)
        return;

    cl_int err = clEnqueueWriteBuffer(
        OpenCLState::get().queue,
        dst_device_buf,
        CL_TRUE, // Blocking write
        0,
        size_bytes,
        src_host_ptr,
        0, nullptr, nullptr);

    if (err != CL_SUCCESS)
    {
        Error::throw_err("OpenCL: clEnqueueWriteBuffer failed in runCopy_CPU_OpenCL");
    }
}

REGISTER_REF_KERNEL(OpType::COPY_TO, 1, matchCopy_CPU_OpenCL, runCopy_CPU_OpenCL, {Backend::OPENCL}, {DType::ANY}, {{8, 32}}, {true}, {{Backend::CPU}});