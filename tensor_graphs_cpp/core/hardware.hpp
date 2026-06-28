#pragma once
#include <string>
#include <iostream>
#include <thread>
#include "core/types.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <CL/cl.h>

struct HardwareCaps
{
        bool has_unified_memory = false;
        bool has_cuda = false;
        bool has_neon = false;
        bool has_opencl = false; // New
        bool is_adreno = false;  // New
        std::string hw_tag;
        size_t num_threads = 1;

        static HardwareCaps &get()
        {
                static HardwareCaps instance;
                static bool initialized = false;
                if (!initialized)
                {
                        instance.probe();
                        initialized = true;
                }
                return instance;
        }

private:
        void probe()
        {
                // 1. Detect CPU Architecture & SIMD
#if defined(TG_HAS_NEON)
                has_neon = true;
#endif
                num_threads = std::thread::hardware_concurrency();

                // 2. Detect CUDA
#ifdef USE_CUDA
                int deviceCount = 0;
                if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0)
                {
                        has_cuda = true;
                        int isIntegrated = 0;
                        cudaDeviceGetAttribute(&isIntegrated, cudaDevAttrIntegrated, 0);
                        int canMapHostMemory = 0;
                        cudaDeviceGetAttribute(&canMapHostMemory, cudaDevAttrCanMapHostMemory, 0);
                        if (isIntegrated && canMapHostMemory)
                        {
                                has_unified_memory = true;
                        }
                }
#endif

                // 3. Detect OpenCL & Adreno GPU
                cl_uint numPlatforms = 0;
                if (clGetPlatformIDs(0, nullptr, &numPlatforms) == CL_SUCCESS && numPlatforms > 0)
                {
                        std::vector<cl_platform_id> platforms(numPlatforms);
                        clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

                        for (auto platform : platforms)
                        {
                                cl_uint numDevices = 0;
                                if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) == CL_SUCCESS && numDevices > 0)
                                {
                                        std::vector<cl_device_id> devices(numDevices);
                                        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);

                                        for (auto device : devices)
                                        {
                                                char deviceName[256];
                                                clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
                                                std::string name(deviceName);

                                                has_opencl = true;
                                                // Check if the device is Qualcomm Adreno
                                                if (name.find("Adreno") != std::string::npos)
                                                {
                                                        is_adreno = true;
                                                        // Qualcomm Adreno supports Shared Virtual Memory (SVM).
                                                        // We can flag unified memory optimization.
                                                        has_unified_memory = true;
                                                }
                                        }
                                }
                        }
                }

                // 4. Generate HW_TAG
                std::string os = "UnknownOS";
#if defined(TG_OS_WINDOWS)
                os = "Win";
#elif defined(TG_OS_MACOS)
                os = "Mac";
#elif defined(TG_OS_LINUX)
                os = "Lin";
#endif

                std::string arch = "UnknownArch";
#if defined(TG_ARCH_ARM64)
                arch = "ARM64";
#elif defined(TG_ARCH_X64)
                arch = "x64";
#endif

                hw_tag = os + "_" + arch;
                if (has_cuda)
                        hw_tag += "_CUDA";
                if (is_adreno)
                        hw_tag += "_Adreno";
                if (has_unified_memory)
                        hw_tag += "_UM";

                std::cout << "[Hardware] Probed: " << hw_tag << " (Threads: " << num_threads << ")" << std::endl;
        }
};