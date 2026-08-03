#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"
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
    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0)
        return;

    cl_kernel k = OpenCL::getKernel("kernels/opencl/pow/pow.cl", "pow_f32_nd");
    OpenCL::setArgBuffer(k, 0, ctx.cl_inputs[0]);
    OpenCL::setArgBuffer(k, 1, ctx.cl_inputs[1]);
    OpenCL::setArgBuffer(k, 2, ctx.cl_outputs[0]);
    clSetKernelArg(k, 3, sizeof(uint64_t), &n);

    uint64_t local_work_size = 256;
    uint64_t global_work_size = ((n + local_work_size - 1) / local_work_size) * local_work_size;
    cl_int err = clEnqueueNDRangeKernel(OpenCLState::get().queue, k, 1, nullptr, &global_work_size, &local_work_size, 0,
                                        nullptr, nullptr);
    if (err != CL_SUCCESS)
        Error::throw_err("OpenCL: Failed to enqueue Pow_F32_1D");
}

inline LogicalId refFactoryPowF32_1D_OpenCL(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.pow(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Pow_F32_1D_OpenCL", 2, 2, matchPowF32_OpenCL_1D, runPowF32_OpenCL_1D, refFactoryPowF32_1D_OpenCL,
                MemSpace(1, HandleType::OPENCL), {Engine(1, EngineType::QUALCOMM_IGPU)},
                {DType::FLOAT32, DType::FLOAT32}, {{1024}, {1024}}, {true, true},
                {{MemSpace(1, HandleType::OPENCL)}, {MemSpace(1, HandleType::OPENCL)}});