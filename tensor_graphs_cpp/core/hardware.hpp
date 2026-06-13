#pragma once
#include <string>
#include <iostream>
#include <thread>
#include "core/types.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

struct HardwareCaps
{
    bool has_unified_memory = false;
    bool has_cuda = false;
    bool has_neon = false;
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

#include <thread>
        num_threads = std::thread::hardware_concurrency();

// 2. Detect CUDA & Unified Memory
#ifdef USE_CUDA
        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0)
        {
            has_cuda = true;
            int isIntegrated = 0;
            // cudaDevAttrIntegrated is the safest check for true physically unified memory.
            // Discrete GPUs with Managed Memory will return 0 here.
            cudaDeviceGetAttribute(&isIntegrated, cudaDevAttrIntegrated, 0);

            int canMapHostMemory = 0;
            cudaDeviceGetAttribute(&canMapHostMemory, cudaDevAttrCanMapHostMemory, 0);

            // We enable the optimization ONLY if it's an integrated SoC (Spark/Jetson/Grace)
            if (isIntegrated && canMapHostMemory)
            {
                has_unified_memory = true;
            }
        }
#endif

        // 3. Generate HW_TAG for Cost Model
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

        hw_tag = os + "_" + arch + (has_cuda ? "_CUDA" : "_NoCUDA");
        if (has_unified_memory)
            hw_tag += "_UM";

        std::cout << "[Hardware] Probed: " << hw_tag << " (Threads: " << num_threads << ")" << std::endl;
    }
};