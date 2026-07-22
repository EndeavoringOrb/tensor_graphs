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
    cl_mem src_device_buf = ctx.cl_inputs[0];
    uint8_t *dst_host_ptr = static_cast<uint8_t *>(ctx.outputs[0]);

    uint64_t numElements = countElements(ctx.inViews[0].getShape());
    uint64_t elemSize = getDTypeSize(ctx.inViews[0].dtype);
    uint64_t size_bytes = numElements * elemSize;

    if (size_bytes == 0)
        return;

    // Enqueue a blocking read. The driver handles both synchronization and cache invalidation.
    cl_int err = clEnqueueReadBuffer(
        OpenCLState::get().queue,
        src_device_buf,
        CL_TRUE, // Blocking read
        0,
        size_bytes,
        dst_host_ptr,
        0, nullptr, nullptr);

    if (err != CL_SUCCESS)
    {
        Error::throw_err("OpenCL: clEnqueueReadBuffer failed in runCopy_OpenCL_CPU");
    }
}

REGISTER_REF_KERNEL(OpType::COPY_TO, 1, matchCopy_OpenCL_CPU, runCopy_OpenCL_CPU, {Backend::CPU}, {DType::ANY}, {{8, 32}}, {true}, {{Backend::OPENCL}});