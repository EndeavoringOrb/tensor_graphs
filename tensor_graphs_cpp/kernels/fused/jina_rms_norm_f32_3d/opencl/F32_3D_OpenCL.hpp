#include "kernels/fused/jina_rms_norm_f32_3d/ref.hpp"
// tensor_graphs_cpp/kernels/fused/jina_rms_norm_f32_3d/opencl/F32_3D_OpenCL.hpp
#pragma once

#ifdef TG_USE_OPENCL
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/utils/opencl_utils.hpp"

inline bool matchJinaRMSNorm_F32_3D_OpenCL(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

    cl_kernel k = OpenCL::getKernel("kernels/fused/jina_rms_norm_f32_3d/opencl/rmsnorm.cl", "rmsnorm_f32_3d");
    OpenCL::setArgBuffer(k, 0, ctx.cl_inputs[0]);
    OpenCL::setArgBuffer(k, 1, ctx.cl_inputs[1]);
    OpenCL::setArgBuffer(k, 2, ctx.cl_outputs[0]);
    clSetKernelArg(k, 3, sizeof(uint32_t), &outer_size);
    clSetKernelArg(k, 4, sizeof(uint32_t), &D);
    clSetKernelArg(k, 5, sizeof(float), &eps);

    uint64_t local_work_size = 256;
    uint64_t global_work_size = ((outer_size + local_work_size - 1) / local_work_size) * local_work_size;
    cl_int err = clEnqueueNDRangeKernel(OpenCLState::get().queue, k, 1, nullptr, &global_work_size, &local_work_size, 0,
                                        nullptr, nullptr);
    if (err != CL_SUCCESS)
        Error::throw_err("OpenCL: Failed to enqueue RMSNorm_OpenCL");
}



REGISTER_KERNEL("JinaRMSNorm_F32_3D_OpenCL", 2, 2, matchJinaRMSNorm_F32_3D_OpenCL, runJinaRMSNorm_F32_3D_OpenCL,
                refFactoryJinaRMSNorm_F32_3D, {0}, MemSpace(1, HandleType::OPENCL),
                {Engine(1, EngineType::QUALCOMM_IGPU)}, {DType::FLOAT32, DType::FLOAT32}, {{1, 1024, 768}, {768}},
                {true, true}, {{MemSpace(1, HandleType::OPENCL)}, {MemSpace(1, HandleType::OPENCL)}});
#endif
