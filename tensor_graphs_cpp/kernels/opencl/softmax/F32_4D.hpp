// tensor_graphs_cpp/kernels/opencl/softmax/F32_4D.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "kernels/opencl/opencl_utils.hpp"

inline bool matchSoftmaxF32_4D_OpenCL(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape().size() == 4 && isContiguous(output);
}

inline void runSoftmaxF32_4D_OpenCL(const KernelContext &ctx)
{
    const auto &shape = ctx.inViews[0].getShape();
    uint32_t outer_size = shape[0] * shape[1] * shape[2];
    uint32_t dim_size = shape[3];

    cl_kernel k = OpenCL::getKernel("kernels/opencl/softmax/softmax.cl", "softmax_f32_4d");
    OpenCL::setArgBuffer(k, 0, ctx.cl_inputs[0]);
    OpenCL::setArgBuffer(k, 1, ctx.cl_outputs[0]);
    clSetKernelArg(k, 2, sizeof(uint32_t), &outer_size);
    clSetKernelArg(k, 3, sizeof(uint32_t), &dim_size);

    uint64_t local_work_size = 256;
    uint64_t global_work_size = ((outer_size + local_work_size - 1) / local_work_size) * local_work_size;
    cl_int err = clEnqueueNDRangeKernel(OpenCLState::get().queue, k, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        Error::throw_err("OpenCL: Failed to enqueue Softmax_4D_OpenCL");

    clFinish(OpenCLState::get().queue);
}

inline LogicalId refFactorySoftmax4D_OpenCL(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    auto s = g.getNode(x).getShape();
    int32_t ax = -1;
    LogicalId axis_node = g.constant({1}, &ax, DType::INT32);
    LogicalId m_rep = g.constant({1}, (int32_t *)&s[3], DType::INT32);
    LogicalId ax_rep = g.constant({1}, (int32_t *)&ax, DType::INT32);

    LogicalId max_s = g.repeat(g.max(x, axis_node), m_rep, ax_rep);
    LogicalId shifted = g.add(x, g.neg(max_s));

    float e_v = 2.718281828f;
    LogicalId e_n = g.constant({1}, &e_v, DType::FLOAT32);
    int32_t sh4[] = {1, 1, 1, 1};
    LogicalId e_b = g.reshape(e_n, g.constant({4}, sh4, DType::INT32));
    for (int i = 0; i < 4; ++i)
    {
        int32_t r = (int32_t)s[i];
        if (r <= 1)
            continue;
        int32_t a = i;
        e_b = g.repeat(e_b, g.constant({1}, &r, DType::INT32),
                       g.constant({1}, &a, DType::INT32));
    }

    LogicalId exps = g.pow(e_b, shifted);
    LogicalId sums = g.repeat(g.sum(exps, axis_node), m_rep, ax_rep);
    return g.div(exps, sums);
}

REGISTER_KERNEL("Softmax_4D_OpenCL", 1, 1, matchSoftmaxF32_4D_OpenCL, runSoftmaxF32_4D_OpenCL, refFactorySoftmax4D_OpenCL, MemSpace(1, HandleType::OPENCL), {Engine(0, EngineType::QUALCOMM_IGPU)},
    {DType::FLOAT32},
    {{1, 24, 1536, 1536}},
    {true},
    {{MemSpace(1, HandleType::OPENCL)}});