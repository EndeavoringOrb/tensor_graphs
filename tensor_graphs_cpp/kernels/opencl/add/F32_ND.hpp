#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <CL/cl.h>

// Simple program helper to build kernels
inline cl_program compileProgram(cl_context context, cl_device_id device, const char *source)
{
    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    return program;
}

inline bool matchAddF32_OpenCL_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runAddF32_OpenCL_ND(const KernelContext &ctx)
{
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    const float *B = static_cast<const float *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0)
        return;

    OpenCLState &ocl = OpenCLState::get();

    // Simple static kernel cache
    static cl_program program = nullptr;
    static cl_kernel kernel = nullptr;
    if (!program)
    {
        const char *source = R"(
            __kernel void add_f32_nd(__global const float* A, __global const float* B, __global float* Out, const ulong n) {
                ulong idx = get_global_id(0);
                if (idx < n) {
                    Out[idx] = A[idx] + B[idx];
                }
            }
        )";
        program = compileProgram(ocl.context, ocl.device, source);
        cl_int err;
        kernel = clCreateKernel(program, "add_f32_nd", &err);
    }

    // Set SVM arguments directly (no clEnqueueWriteBuffer needed!)
    clSetKernelArgSVMPointer(kernel, 0, A);
    clSetKernelArgSVMPointer(kernel, 1, B);
    clSetKernelArgSVMPointer(kernel, 2, Out);
    clSetKernelArg(kernel, 3, sizeof(cl_ulong), &n);

    size_t globalSize = n;
    size_t localSize = 256;
    // Align global size
    if (globalSize % localSize != 0)
    {
        globalSize = ((globalSize + localSize - 1) / localSize) * localSize;
    }

    clEnqueueNDRangeKernel(ocl.queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
    clFinish(ocl.queue); // Ensure execution is complete before returning control
}

inline uint32_t refFactoryAddF32_ND_OpenCL(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.add(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Add_F32_ND_OpenCL", 2, matchAddF32_OpenCL_ND, runAddF32_OpenCL_ND, refFactoryAddF32_ND_OpenCL,
                {Backend::OPENCL},
                {DType::FLOAT32, DType::FLOAT32},
                {{1024}, {1024}},
                {true, true},
                {{Backend::OPENCL}, {Backend::OPENCL}});