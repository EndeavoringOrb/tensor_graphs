#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "kernels/opencl/opencl_utils.hpp"

inline bool matchDotF32_3D_OpenCL(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (output.dtype != DType::FLOAT32)
        return false;
    const auto &s0 = inputs[0].getShape();
    const auto &s1 = inputs[1].getShape();
    const auto &so = output.getShape();
    if (s0.size() != 3 || s1.size() != 3 || so.size() != 3)
        return false;
    if (s0[0] != s1[0] || s0[2] != s1[1])
        return false;
    if (so[0] != s0[0] || so[1] != s0[1] || so[2] != s1[2])
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runDotF32_3D_OpenCL(const KernelContext &ctx)
{
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    const float *B = static_cast<const float *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    uint64_t B_count = ctx.inViews[0].getShape()[0];
    uint64_t M = ctx.inViews[0].getShape()[1];
    uint64_t K = ctx.inViews[0].getShape()[2];
    uint64_t N = ctx.inViews[1].getShape()[2];

    cl_kernel k = OpenCL::getKernel("kernels/opencl/dot/dot.cl", "dot_f32_3d");

    OpenCL::setArgSVM(k, 0, A);
    OpenCL::setArgSVM(k, 1, B);
    OpenCL::setArgSVM(k, 2, Out);
    clSetKernelArg(k, 3, sizeof(uint64_t), &B_count);
    clSetKernelArg(k, 4, sizeof(uint64_t), &M);
    clSetKernelArg(k, 5, sizeof(uint64_t), &K);
    clSetKernelArg(k, 6, sizeof(uint64_t), &N);

    size_t local_work_size[3] = {16, 16, 1};
    size_t global_work_size[3] = {
        ((N + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0],
        ((M + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1],
        (size_t)B_count};

    cl_int err = clEnqueueNDRangeKernel(OpenCLState::get().queue, k, 3, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        Error::throw_err("OpenCL: Failed to enqueue Dot_F32_3D");

    clFinish(OpenCLState::get().queue);
}

inline uint32_t refFactoryDotF32_3D_OpenCL(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.dot(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Dot_F32_3D_OpenCL", 2, matchDotF32_3D_OpenCL, runDotF32_3D_OpenCL, refFactoryDotF32_3D_OpenCL, {Backend::OPENCL}, {DType::FLOAT32, DType::FLOAT32}, {{1, 16, 32}, {1, 32, 16}}, {true, true}, {{Backend::OPENCL}, {Backend::OPENCL}});