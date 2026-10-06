#include "kernels/fused/softmax_4d/ref.hpp"
// tensor_graphs_cpp/kernels/fused/softmax_4d/opencl/F32_4D_OpenCL.hpp
#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/utils/opencl_utils.hpp"

inline bool matchSoftmaxF32_4D_OpenCL(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape().size() == 4 && isContiguous(output);
}

inline void runSoftmaxF32_4D_OpenCL(const KernelContext &ctx)
{
    const auto &shape = ctx.inViews[0].getShape();
    uint32_t outer_size = shape[0] * shape[1] * shape[2];
    uint32_t dim_size = shape[3];

    cl_kernel k = OpenCL::getKernel("kernels/fused/softmax_4d/opencl/softmax.cl", "softmax_f32_4d");
    OpenCL::setArgBuffer(k, 0, ctx.cl_inputs[0]);
    OpenCL::setArgBuffer(k, 1, ctx.cl_outputs[0]);
    clSetKernelArg(k, 2, sizeof(uint32_t), &outer_size);
    clSetKernelArg(k, 3, sizeof(uint32_t), &dim_size);

    uint64_t local_work_size = 256;
    uint64_t global_work_size = ((outer_size + local_work_size - 1) / local_work_size) * local_work_size;
    cl_int err = clEnqueueNDRangeKernel(OpenCLState::get().queue, k, 1, nullptr, &global_work_size, &local_work_size, 0,
                                        nullptr, nullptr);
    if (err != CL_SUCCESS)
        Error::throw_err("OpenCL: Failed to enqueue Softmax_4D_OpenCL");
}



REGISTER_KERNEL("Softmax_4D_OpenCL", 1, 1, matchSoftmaxF32_4D_OpenCL, runSoftmaxF32_4D_OpenCL,
                refFactorySoftmax4D, {0}, MemSpace(1, HandleType::OPENCL),
                {Engine(1, EngineType::QUALCOMM_IGPU)}, {DType::FLOAT32}, {{1, 24, 1536, 1536}}, {true},
                {{MemSpace(1, HandleType::OPENCL)}});