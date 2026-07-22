// tensor_graphs_cpp/kernels/opencl/gelu/F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "kernels/opencl/opencl_utils.hpp"

inline bool matchJinaGeluExact_F32_3D_OpenCL(const std::vector<TensorNode> &inputs,
                                             const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3)
        return false;
    return isContiguous(output);
}

inline void runJinaGeluExact_F32_3D_OpenCL(const KernelContext &ctx)
{
    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0)
        return;

    cl_kernel k = OpenCL::getKernel("kernels/opencl/gelu/gelu.cl", "gelu_f32_nd");
    OpenCL::setArgBuffer(k, 0, ctx.cl_inputs[0]);
    OpenCL::setArgBuffer(k, 1, ctx.cl_outputs[0]);
    clSetKernelArg(k, 2, sizeof(uint64_t), &n);

    uint64_t local_work_size = 256;
    uint64_t global_work_size = ((n + local_work_size - 1) / local_work_size) * local_work_size;
    cl_int err = clEnqueueNDRangeKernel(OpenCLState::get().queue, k, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        Error::throw_err("OpenCL: Failed to enqueue Gelu_OpenCL");

    clFinish(OpenCLState::get().queue);
}

inline uint32_t refFactoryJinaGeluExact_F32_3D_OpenCL(const std::vector<uint32_t> &inputs,
                                                      Graph &g)
{
    uint32_t x_id = inputs[0];
    const auto &shape = g.getNode(x_id).getShape();
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    auto expand_scalar_SD = [&](float val) -> uint32_t
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
        if (D > 1)
        {
            int32_t rep = (int32_t)D;
            int32_t ax = 2;
            out = g.repeat(out,
                           g.constant({1}, &rep, DType::INT32),
                           g.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    uint32_t inv_sqrt2 = expand_scalar_SD(0.7071067811865475f);
    uint32_t half = expand_scalar_SD(0.5f);
    uint32_t eps_node = expand_scalar_SD(1e-12f);
    uint32_t p_node = expand_scalar_SD(0.3275911f);
    uint32_t one_node = expand_scalar_SD(1.0f);
    uint32_t a1 = expand_scalar_SD(0.254829592f);
    uint32_t a2 = expand_scalar_SD(-0.284496736f);
    uint32_t a3 = expand_scalar_SD(1.421413741f);
    uint32_t a4 = expand_scalar_SD(-1.453152027f);
    uint32_t a5 = expand_scalar_SD(1.061405429f);
    uint32_t e_node = expand_scalar_SD(2.718281828459045f);

    uint32_t x_scaled = g.mul(x_id, inv_sqrt2);
    uint32_t xs_sq = g.mul(x_scaled, x_scaled);
    uint32_t abs_xs = g.pow(xs_sq, half);
    uint32_t abs_xs_eps = g.add(abs_xs, eps_node);
    uint32_t sign_xs = g.div(x_scaled, abs_xs_eps);

    uint32_t p_abs = g.mul(p_node, abs_xs);
    uint32_t denom = g.add(one_node, p_abs);
    uint32_t t = g.div(one_node, denom);

    uint32_t t2 = g.mul(t, t);
    uint32_t t3 = g.mul(t2, t);
    uint32_t t4 = g.mul(t3, t);
    uint32_t t5 = g.mul(t4, t);

    uint32_t poly = g.mul(a1, t);
    poly = g.add(poly, g.mul(a2, t2));
    poly = g.add(poly, g.mul(a3, t3));
    poly = g.add(poly, g.mul(a4, t4));
    poly = g.add(poly, g.mul(a5, t5));

    uint32_t neg_xs_sq = g.neg(xs_sq);
    uint32_t exp_neg_xs_sq = g.pow(e_node, neg_xs_sq);

    uint32_t product = g.mul(poly, exp_neg_xs_sq);
    uint32_t erf_pos = g.add(one_node, g.neg(product));
    uint32_t erf_val = g.mul(sign_xs, erf_pos);

    uint32_t one_plus_erf = g.add(one_node, erf_val);
    uint32_t half_x = g.mul(x_id, half);
    return g.mul(half_x, one_plus_erf);
}

REGISTER_KERNEL("JinaGeluExact_F32_3D_OpenCL", 1,
                matchJinaGeluExact_F32_3D_OpenCL, runJinaGeluExact_F32_3D_OpenCL,
                refFactoryJinaGeluExact_F32_3D_OpenCL,
                {Backend::OPENCL},
                {DType::FLOAT32},
                {{1, 1024, 3072}},
                {true},
                {{Backend::OPENCL}});