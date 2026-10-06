#include "kernels/fused/jina_gelu_exact_f32_3d/ref.hpp"
// tensor_graphs_cpp/kernels/fused/jina_gelu_exact_f32_3d/opencl/F32_ND_OpenCL.hpp
#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/utils/opencl_utils.hpp"

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

    cl_kernel k = OpenCL::getKernel("kernels/fused/jina_gelu_exact_f32_3d/opencl/gelu.cl", "gelu_f32_nd");
    OpenCL::setArgBuffer(k, 0, ctx.cl_inputs[0]);
    OpenCL::setArgBuffer(k, 1, ctx.cl_outputs[0]);
    clSetKernelArg(k, 2, sizeof(uint64_t), &n);

    uint64_t local_work_size = 256;
    uint64_t global_work_size = ((n + local_work_size - 1) / local_work_size) * local_work_size;
    cl_int err = clEnqueueNDRangeKernel(OpenCLState::get().queue, k, 1, nullptr, &global_work_size, &local_work_size, 0,
                                        nullptr, nullptr);
    if (err != CL_SUCCESS)
        Error::throw_err("OpenCL: Failed to enqueue Gelu_OpenCL");
}



REGISTER_KERNEL("JinaGeluExact_F32_3D_OpenCL", 1, 1, matchJinaGeluExact_F32_3D_OpenCL, runJinaGeluExact_F32_3D_OpenCL,
                refFactoryJinaGeluExact_F32_3D, {0}, MemSpace(1, HandleType::OPENCL),
                {Engine(1, EngineType::QUALCOMM_IGPU)}, {DType::FLOAT32}, {{1, 1024, 3072}}, {true},
                {{MemSpace(1, HandleType::OPENCL)}});