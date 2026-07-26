#pragma once
#include <iostream>
#include <string>
#include <unordered_map>

#include "core/memory.hpp"
#include "generated/opencl_kernels.gen.hpp"

namespace OpenCL
{
inline cl_program buildProgram(const std::string &source)
{
    OpenCLState::get().init();

    cl_context ctx = OpenCLState::get().context;
    cl_device_id device = OpenCLState::get().device;

    const char *src_ptr = source.c_str();
    uint64_t src_len = source.length();

    cl_int err;
    cl_program program = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS)
    {
        Error::throw_err("OpenCL: Failed to create program");
    }

    err = clBuildProgram(program, 1, &device, "-cl-std=CL2.0", nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        uint64_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string build_log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], nullptr);
        Error::throw_err("OpenCL: Build failed:\n" + build_log);
    }
    return program;
}

inline cl_kernel getKernel(const std::string &file_path, const std::string &kernel_name)
{
    static std::unordered_map<std::string, cl_program> programs;
    static std::unordered_map<std::string, cl_kernel> kernels;

    std::string key = file_path + ":" + kernel_name;
    if (kernels.count(key))
    {
        return kernels[key];
    }

    if (programs.count(file_path) == 0)
    {
        auto it = OPENCL_SOURCE_MAP.find(file_path);
        if (it == OPENCL_SOURCE_MAP.end())
        {
            Error::throw_err("OpenCL source file not found in generated map: " + file_path);
        }
        programs[file_path] = buildProgram(it->second);
    }

    cl_int err;
    cl_kernel kernel = clCreateKernel(programs[file_path], kernel_name.c_str(), &err);
    if (err != CL_SUCCESS)
    {
        Error::throw_err("OpenCL: Failed to create kernel " + kernel_name);
    }

    kernels[key] = kernel;
    return kernel;
}

inline void setArgBuffer(cl_kernel k, cl_uint index, cl_mem buffer)
{
    cl_int err = clSetKernelArg(k, index, sizeof(cl_mem), &buffer);
    if (err != CL_SUCCESS)
    {
        Error::throw_err("OpenCL: Failed to set buffer kernel arg " + std::to_string(index));
    }
}
} // namespace OpenCL