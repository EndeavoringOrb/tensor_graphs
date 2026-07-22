#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "kernels/opencl/opencl_utils.hpp"

inline bool matchDotF32_3D_OpenCL_v1(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

inline void runDotF32_3D_OpenCL_v1(const KernelContext &ctx)
{
    uint64_t B_count = ctx.inViews[0].getShape()[0];
    uint64_t M = ctx.inViews[0].getShape()[1];
    uint64_t K = ctx.inViews[0].getShape()[2];
    uint64_t N = ctx.inViews[1].getShape()[2];

    cl_kernel k = OpenCL::getKernel("kernels/opencl/dot/v1/dot.cl", "dot_f32_3d");

    OpenCL::setArgBuffer(k, 0, ctx.cl_inputs[0]);
    OpenCL::setArgBuffer(k, 1, ctx.cl_inputs[1]);
    OpenCL::setArgBuffer(k, 2, ctx.cl_outputs[0]);
    clSetKernelArg(k, 3, sizeof(uint64_t), &B_count);
    clSetKernelArg(k, 4, sizeof(uint64_t), &M);
    clSetKernelArg(k, 5, sizeof(uint64_t), &K);
    clSetKernelArg(k, 6, sizeof(uint64_t), &N);

    uint64_t local_work_size[3] = {16, 16, 1};
    uint64_t global_work_size[3] = {
        ((N + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0],
        ((M + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1],
        (uint64_t)B_count};

    cl_int err = clEnqueueNDRangeKernel(OpenCLState::get().queue, k, 3, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        Error::throw_err("OpenCL: Failed to enqueue Dot_F32_3D_v1");

    clFinish(OpenCLState::get().queue);
}

inline LogicalId refFactoryDotF32_3D_OpenCL_v1(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.dot(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Dot_F32_3D_OpenCL_v1", 2, 2, matchDotF32_3D_OpenCL_v1, runDotF32_3D_OpenCL_v1, refFactoryDotF32_3D_OpenCL_v1, MemSpace(1, HandleType::OPENCL), {Engine(0, EngineType::QUALCOMM_IGPU)}, {DType::FLOAT32, DType::FLOAT32}, {{1, 16, 32}, {1, 32, 16}}, {true, true}, {{MemSpace(1, HandleType::OPENCL)}, {MemSpace(1, HandleType::OPENCL)}});