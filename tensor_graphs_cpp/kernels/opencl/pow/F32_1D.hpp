#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "kernels/opencl/opencl_utils.hpp"

inline bool matchPowF32_OpenCL_1D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runPowF32_OpenCL_1D(const KernelContext &ctx)
{
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    const float *B = static_cast<const float *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0)
        return;

    cl_kernel k = OpenCL::getKernel("kernels/opencl/pow/pow.cl", "pow_f32_nd");
    OpenCL::setArgSVM(k, 0, A);
    OpenCL::setArgSVM(k, 1, B);
    OpenCL::setArgSVM(k, 2, Out);
    clSetKernelArg(k, 3, sizeof(uint64_t), &n);

    size_t local_work_size = 256;
    size_t global_work_size = ((n + local_work_size - 1) / local_work_size) * local_work_size;
    cl_int err = clEnqueueNDRangeKernel(OpenCLState::get().queue, k, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        Error::throw_err("OpenCL: Failed to enqueue Pow_F32_1D");

    clFinish(OpenCLState::get().queue);
}

inline uint32_t refFactoryPowF32_1D_OpenCL(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.pow(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Pow_F32_1D_OpenCL", 2, matchPowF32_OpenCL_1D, runPowF32_OpenCL_1D, refFactoryPowF32_1D_OpenCL, {Backend::OPENCL}, {DType::FLOAT32, DType::FLOAT32}, {{1024}, {1024}}, {true, true}, {{Backend::OPENCL}, {Backend::OPENCL}});