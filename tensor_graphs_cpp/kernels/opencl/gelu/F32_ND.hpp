// tensor_graphs_cpp/kernels/opencl/gelu/F32_ND.hpp
#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/opencl/opencl_utils.hpp"

inline bool matchJinaGeluExact_F32_3D_OpenCL(const std::vector<TensorNode> &inputs, const TensorNode &output)
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
    cl_int err = clEnqueueNDRangeKernel(OpenCLState::get().queue, k, 1, nullptr, &global_work_size, &local_work_size, 0,
                                        nullptr, nullptr);
    if (err != CL_SUCCESS)
        Error::throw_err("OpenCL: Failed to enqueue Gelu_OpenCL");

    clFinish(OpenCLState::get().queue);
}

inline LogicalId refFactoryJinaGeluExact_F32_3D_OpenCL(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x_id = inputs[0];
    const auto &shape = g.getNode(x_id).getShape();
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    auto expand_scalar_SD = [&](float val) -> LogicalId {
        LogicalId node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh[] = {1, 1, 1};
        LogicalId out = g.reshape(node, g.constant({3}, sh, DType::INT32));
        if (S > 1)
        {
            int32_t rep = (int32_t)S;
            int32_t ax = 1;
            out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));
        }
        if (D > 1)
        {
            int32_t rep = (int32_t)D;
            int32_t ax = 2;
            out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    LogicalId inv_sqrt2 = expand_scalar_SD(0.7071067811865475f);
    LogicalId half = expand_scalar_SD(0.5f);
    LogicalId eps_node = expand_scalar_SD(1e-12f);
    LogicalId p_node = expand_scalar_SD(0.3275911f);
    LogicalId one_node = expand_scalar_SD(1.0f);
    LogicalId a1 = expand_scalar_SD(0.254829592f);
    LogicalId a2 = expand_scalar_SD(-0.284496736f);
    LogicalId a3 = expand_scalar_SD(1.421413741f);
    LogicalId a4 = expand_scalar_SD(-1.453152027f);
    LogicalId a5 = expand_scalar_SD(1.061405429f);
    LogicalId e_node = expand_scalar_SD(2.718281828459045f);

    LogicalId x_scaled = g.mul(x_id, inv_sqrt2);
    LogicalId xs_sq = g.mul(x_scaled, x_scaled);
    LogicalId abs_xs = g.pow(xs_sq, half);
    LogicalId abs_xs_eps = g.add(abs_xs, eps_node);
    LogicalId sign_xs = g.div(x_scaled, abs_xs_eps);

    LogicalId p_abs = g.mul(p_node, abs_xs);
    LogicalId denom = g.add(one_node, p_abs);
    LogicalId t = g.div(one_node, denom);

    LogicalId t2 = g.mul(t, t);
    LogicalId t3 = g.mul(t2, t);
    LogicalId t4 = g.mul(t3, t);
    LogicalId t5 = g.mul(t4, t);

    LogicalId poly = g.mul(a1, t);
    poly = g.add(poly, g.mul(a2, t2));
    poly = g.add(poly, g.mul(a3, t3));
    poly = g.add(poly, g.mul(a4, t4));
    poly = g.add(poly, g.mul(a5, t5));

    LogicalId neg_xs_sq = g.neg(xs_sq);
    LogicalId exp_neg_xs_sq = g.pow(e_node, neg_xs_sq);

    LogicalId product = g.mul(poly, exp_neg_xs_sq);
    LogicalId erf_pos = g.add(one_node, g.neg(product));
    LogicalId erf_val = g.mul(sign_xs, erf_pos);

    LogicalId one_plus_erf = g.add(one_node, erf_val);
    LogicalId half_x = g.mul(x_id, half);
    return g.mul(half_x, one_plus_erf);
}

REGISTER_KERNEL("JinaGeluExact_F32_3D_OpenCL", 1, 1, matchJinaGeluExact_F32_3D_OpenCL, runJinaGeluExact_F32_3D_OpenCL,
                refFactoryJinaGeluExact_F32_3D_OpenCL, MemSpace(1, HandleType::OPENCL),
                {Engine(1, EngineType::QUALCOMM_IGPU)}, {DType::FLOAT32}, {{1, 1024, 3072}}, {true},
                {{MemSpace(1, HandleType::OPENCL)}});