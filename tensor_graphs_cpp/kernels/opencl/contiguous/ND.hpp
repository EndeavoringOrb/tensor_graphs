#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "kernels/opencl/opencl_utils.hpp"

struct ContiguousParamsOpenCL
{
    uint32_t rank;
    uint32_t padding;
    uint32_t shape[8];
    uint64_t in_strides[8];
};

inline bool matchContiguous_OpenCL_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &in = inputs[0];

    if (in.dtype != output.dtype)
        return false;

    // Supports up to rank 8 limit
    if (in.getShape().size() > 8 || in.getShape().empty())
        return false;

    if (in.getShape() != output.getShape())
        return false;

    // Output must be contiguous
    if (!isContiguous(output))
        return false;

    return true;
}

inline void runContiguous_OpenCL_ND(const KernelContext &ctx)
{
    uint64_t numElements = countElements(ctx.outViews[0].getShape());
    uint64_t elemSize = getDTypeSize(ctx.inViews[0].dtype);
    if (numElements == 0)
        return;

    ContiguousParamsOpenCL p;
    std::memset(&p, 0, sizeof(p));
    p.rank = (uint32_t)ctx.outViews[0].getShape().size();

    for (uint32_t i = 0; i < p.rank; ++i)
    {
        p.shape[i] = ctx.outViews[0].getShape()[i];
        p.in_strides[i] = ctx.inViews[0].strides[i];
    }

    cl_kernel k = OpenCL::getKernel("kernels/opencl/contiguous/contiguous.cl", "contiguous_generic");

    OpenCL::setArgBuffer(k, 0, ctx.cl_inputs[0]);
    OpenCL::setArgBuffer(k, 1, ctx.cl_outputs[0]);
    clSetKernelArg(k, 2, sizeof(uint64_t), &numElements);
    clSetKernelArg(k, 3, sizeof(uint64_t), &elemSize);
    clSetKernelArg(k, 4, sizeof(ContiguousParamsOpenCL), &p);

    size_t local_work_size = 256;
    size_t global_work_size = ((numElements + local_work_size - 1) / local_work_size) * local_work_size;

    cl_int err = clEnqueueNDRangeKernel(OpenCLState::get().queue, k, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        Error::throw_err("OpenCL: Failed to enqueue Contiguous_OpenCL_ND");
    }

    clFinish(OpenCLState::get().queue);
}

inline uint32_t refFactoryContiguous_OpenCL_ND(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.contiguous(inputs[0]);
}

REGISTER_KERNEL("Contiguous_OpenCL_ND", 1, matchContiguous_OpenCL_ND, runContiguous_OpenCL_ND, refFactoryContiguous_OpenCL_ND, {Backend::OPENCL},
                {DType::ANY},       // Input DType
                {{1024, 640}},      // Dummy shape
                {false},            // Input does NOT require contiguity
                {{Backend::OPENCL}} // Input backends
);