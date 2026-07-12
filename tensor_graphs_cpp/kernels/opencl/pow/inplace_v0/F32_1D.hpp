#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "kernels/opencl/opencl_utils.hpp"

inline bool matchPowF32_OpenCL_1D_Inplace(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runPowF32_OpenCL_1D_Inplace(const KernelContext &ctx)
{
    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0)
        return;

    cl_kernel k = OpenCL::getKernel("kernels/opencl/pow/inplace/pow.cl", "pow_f32_nd_inplace");
    OpenCL::setArgBuffer(k, 0, ctx.cl_outputs[0]);
    OpenCL::setArgBuffer(k, 1, ctx.cl_inputs[1]);
    clSetKernelArg(k, 2, sizeof(uint64_t), &n);

    size_t local_work_size = 256;
    size_t global_work_size = ((n + local_work_size - 1) / local_work_size) * local_work_size;
    cl_int err = clEnqueueNDRangeKernel(OpenCLState::get().queue, k, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        Error::throw_err("OpenCL: Failed to enqueue Pow_F32_1D_Inplace");

    clFinish(OpenCLState::get().queue);
}

inline uint32_t refFactoryPowF32_1D_OpenCL_Inplace(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.pow(inputs[0], inputs[1]);
}

REGISTER_KERNEL_INPLACE("Pow_F32_1D_OpenCL_inplace", 2, matchPowF32_OpenCL_1D_Inplace, runPowF32_OpenCL_1D_Inplace, refFactoryPowF32_1D_OpenCL_Inplace, {Backend::OPENCL}, {DType::FLOAT32, DType::FLOAT32}, {{1024}, {1024}}, {true, true}, {{Backend::OPENCL}, {Backend::OPENCL}});