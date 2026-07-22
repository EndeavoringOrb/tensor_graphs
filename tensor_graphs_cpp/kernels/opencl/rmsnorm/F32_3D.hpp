// tensor_graphs_cpp/kernels/opencl/rmsnorm/F32_3D.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "kernels/opencl/opencl_utils.hpp"

inline bool matchJinaRMSNorm_F32_3D_OpenCL(const std::vector<TensorNode> &inputs,
                                           const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1)
        return false;
    if (inputs[0].getShape()[2] != inputs[1].getShape()[0])
        return false;
    return isContiguous(output);
}

inline void runJinaRMSNorm_F32_3D_OpenCL(const KernelContext &ctx)
{
    const auto &shape = ctx.inViews[0].getShape();
    uint32_t B = shape[0];
    uint32_t S = shape[1];
    uint32_t D = shape[2];
    uint32_t outer_size = B * S;
    float eps = 1e-5f;

    cl_kernel k = OpenCL::getKernel("kernels/opencl/rmsnorm/rmsnorm.cl", "rmsnorm_f32_3d");
    OpenCL::setArgBuffer(k, 0, ctx.cl_inputs[0]);
    OpenCL::setArgBuffer(k, 1, ctx.cl_inputs[1]);
    OpenCL::setArgBuffer(k, 2, ctx.cl_outputs[0]);
    clSetKernelArg(k, 3, sizeof(uint32_t), &outer_size);
    clSetKernelArg(k, 4, sizeof(uint32_t), &D);
    clSetKernelArg(k, 5, sizeof(float), &eps);

    uint64_t local_work_size = 256;
    uint64_t global_work_size = ((outer_size + local_work_size - 1) / local_work_size) * local_work_size;
    cl_int err = clEnqueueNDRangeKernel(OpenCLState::get().queue, k, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        Error::throw_err("OpenCL: Failed to enqueue RMSNorm_OpenCL");

    clFinish(OpenCLState::get().queue);
}

inline uint32_t refFactoryJinaRMSNorm_F32_3D_OpenCL(const std::vector<uint32_t> &inputs,
                                                    Graph &g)
{
    uint32_t x_id = inputs[0];
    uint32_t w_id = inputs[1];

    const auto &shape = g.getNode(x_id).getShape();
    uint32_t B = shape[0];
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    auto expand_scalar_1S1 = [&](float val) -> uint32_t
    {
        uint32_t node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh[] = {1, 1, 1};
        uint32_t out = g.reshape(node, g.constant({3}, sh, DType::INT32));
        if (S > 1)
        {
            int32_t rep = (int32_t)S;
            int32_t ax = 1;
            out = g.repeat(out,
                           g.constant({1}, &rep, DType::INT32),
                           g.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    auto repeat_d_axis2 = [&](uint32_t node) -> uint32_t
    {
        int32_t rep = (int32_t)D;
        int32_t ax = 2;
        return g.repeat(node,
                        g.constant({1}, &rep, DType::INT32),
                        g.constant({1}, &ax, DType::INT32));
    };

    auto expand_1d_1SD = [&](uint32_t vec) -> uint32_t
    {
        int32_t sh[] = {1, 1, (int32_t)D};
        uint32_t out = g.reshape(vec, g.constant({3}, sh, DType::INT32));
        if (S > 1)
        {
            int32_t rep = (int32_t)S;
            int32_t ax = 1;
            out = g.repeat(out,
                           g.constant({1}, &rep, DType::INT32),
                           g.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    uint32_t x_sq = g.mul(x_id, x_id);
    int32_t ax_val = -1;
    uint32_t axis_node = g.constant({1}, &ax_val, DType::INT32);
    uint32_t sum_sq = g.sum(x_sq, axis_node);

    float d_float = (float)D;
    uint32_t n_node = expand_scalar_1S1(d_float);
    uint32_t mean_sq = g.div(sum_sq, n_node);

    uint32_t eps_node = expand_scalar_1S1(1e-5f);
    uint32_t mean_sq_plus_eps = g.add(mean_sq, eps_node);
    uint32_t sqrt_node = expand_scalar_1S1(0.5f);
    uint32_t std = g.pow(mean_sq_plus_eps, sqrt_node);

    uint32_t one_node = expand_scalar_1S1(1.0f);
    uint32_t inv_std = g.div(one_node, std);
    uint32_t inv_std_expanded = repeat_d_axis2(inv_std);

    uint32_t x_norm = g.mul(x_id, inv_std_expanded);
    uint32_t w_exp = expand_1d_1SD(w_id);
    return g.mul(x_norm, w_exp);
}

REGISTER_KERNEL("JinaRMSNorm_F32_3D_OpenCL", 2,
                matchJinaRMSNorm_F32_3D_OpenCL, runJinaRMSNorm_F32_3D_OpenCL,
                refFactoryJinaRMSNorm_F32_3D_OpenCL,
                {Backend::OPENCL},
                {DType::FLOAT32, DType::FLOAT32},
                {{1, 1024, 768}, {768}},
                {true, true},
                {{Backend::OPENCL}, {Backend::OPENCL}});