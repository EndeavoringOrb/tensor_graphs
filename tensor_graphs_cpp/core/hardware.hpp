#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/logging.hpp"
#include "core/types.hpp"

#if defined(TG_OS_WINDOWS)
#include <windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#elif defined(TG_OS_LINUX)
#include <unistd.h>
#elif defined(TG_OS_MACOS)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#ifdef TG_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef TG_USE_OPENCL
inline void queryOpenCLDeviceLimits(cl_device_id device)
{
    char deviceName[256] = {0};
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);

    std::cout << "\n========================================" << std::endl;
    if (err == CL_SUCCESS)
    {
        std::cout << "Device: " << deviceName << std::endl;
    }
    else
    {
        std::cout << "Device: [Failed to retrieve device name. Error: " << err << "]" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    cl_device_svm_capabilities svm_caps = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr);
    if (err == CL_SUCCESS)
    {
        std::cout << "OpenCL SVM Capabilities Bitfield: " << svm_caps << std::endl;
        if (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
            std::cout << "  - CL_DEVICE_SVM_COARSE_GRAIN_BUFFER Supported" << std::endl;
        if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
            std::cout << "  - CL_DEVICE_SVM_FINE_GRAIN_BUFFER Supported" << std::endl;
        if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)
            std::cout << "  - CL_DEVICE_SVM_FINE_GRAIN_SYSTEM Supported" << std::endl;
        if (svm_caps & CL_DEVICE_SVM_ATOMICS)
            std::cout << "  - CL_DEVICE_SVM_ATOMICS Supported" << std::endl;
        if (svm_caps == 0)
            std::cout << "  - No SVM capabilities supported on this device/driver configuration." << std::endl;
    }

    cl_ulong max_alloc = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, nullptr);
    if (err == CL_SUCCESS)
    {
        std::cout << "Max Single Allocation Size (CL_DEVICE_MAX_MEM_ALLOC_SIZE): " << max_alloc << " bytes ("
                  << (double)max_alloc / (1024.0 * 1024.0) << " MB)" << std::endl;
    }
}
#endif

struct OpenCLState
{
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    std::string device_name;
    std::string platform_name;
    std::string platform_vendor;
    cl_ulong max_mem_alloc_size = 0;
    bool initialized = false;

    static OpenCLState &get()
    {
        static OpenCLState instance;
        return instance;
    }

    void init()
    {
#ifdef TG_USE_OPENCL
        if (initialized)
            return;

        cl_uint numPlatforms = 0;
        if (clGetPlatformIDs(0, nullptr, &numPlatforms) != CL_SUCCESS || numPlatforms == 0)
            return;

        std::vector<cl_platform_id> platforms(numPlatforms);
        clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

        cl_device_id selectedDevice = nullptr;
        cl_platform_id selectedPlatform = nullptr;
        std::string selectedPlatName;
        std::string selectedPlatVendor;
        bool foundNative = false;

        for (auto plat : platforms)
        {
            char pName[256] = {0};
            char pVendor[256] = {0};
            clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof(pName), pName, nullptr);
            clGetPlatformInfo(plat, CL_PLATFORM_VENDOR, sizeof(pVendor), pVendor, nullptr);

            std::string platNameStr(pName);
            std::string platVendStr(pVendor);

            bool isCompatibilityLayer = (platNameStr.find("OpenCLOn12") != std::string::npos ||
                                         platVendStr.find("Microsoft") != std::string::npos);

            cl_uint numDevices = 0;
            if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) == CL_SUCCESS && numDevices > 0)
            {
                std::vector<cl_device_id> devices(numDevices);
                clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);

                // Prioritize native vendor platforms over compatibility layers
                if (!selectedDevice || (!isCompatibilityLayer && !foundNative))
                {
                    selectedDevice = devices[0];
                    selectedPlatform = plat;
                    selectedPlatName = platNameStr;
                    selectedPlatVendor = platVendStr;
                    if (!isCompatibilityLayer)
                    {
                        foundNative = true;
                    }
                }
            }
        }

        if (selectedDevice)
        {
            device = selectedDevice;
            platform = selectedPlatform;
            platform_name = selectedPlatName;
            platform_vendor = selectedPlatVendor;

            char devName[256] = {0};
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(devName), devName, nullptr);
            device_name = std::string(devName);

            cl_ulong max_alloc = 0;
            if (clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, nullptr) ==
                CL_SUCCESS)
            {
                max_mem_alloc_size = max_alloc;
            }

            cl_int err;
            context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            if (err == CL_SUCCESS && context)
            {
                queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
                if (err == CL_SUCCESS && queue)
                {
                    initialized = true;
                }
            }
        }
#endif
    }
};

struct HardwareCaps
{
    bool has_unified_memory = false;
    bool has_cuda = false;
    bool has_neon = false;
    bool has_opencl = false;
    bool is_adreno = false;
    std::string hw_tag;
    uint64_t num_threads = 1;
    uint32_t num_cuda_devices = 0;
    uint64_t opencl_max_alloc = 0;

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
#if defined(TG_HAS_NEON)
        has_neon = true;
#endif
        num_threads = get_num_threads();

#ifdef TG_USE_CUDA
        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0)
        {
            has_cuda = true;
            num_cuda_devices = static_cast<uint32_t>(deviceCount);
            for (int dev = 0; dev < deviceCount; ++dev)
            {
                int isIntegrated = 0;
                cudaDeviceGetAttribute(&isIntegrated, cudaDevAttrIntegrated, dev);
                int canMapHostMemory = 0;
                cudaDeviceGetAttribute(&canMapHostMemory, cudaDevAttrCanMapHostMemory, dev);
                if (isIntegrated && canMapHostMemory)
                {
                    has_unified_memory = true;
                }
            }
        }
#endif

#ifdef TG_USE_OPENCL
        auto &cl_state = OpenCLState::get();
        cl_state.init();
        if (cl_state.initialized && cl_state.device)
        {
            has_opencl = true;
            opencl_max_alloc = cl_state.max_mem_alloc_size;
            queryOpenCLDeviceLimits(cl_state.device);

            if (cl_state.device_name.find("Adreno") != std::string::npos)
            {
                is_adreno = true;
                has_unified_memory = true;
            }
        }
#endif

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

        std::cout << "[Hardware] Probed: " << hw_tag << " (Threads: " << num_threads
                  << ", CUDA GPUs: " << num_cuda_devices << ")" << std::endl;
    }
};

struct System
{
    HardwareCaps caps;
    std::vector<Engine> engines;
    std::vector<MemSpace> mem_spaces;
    std::unordered_map<MemSpace, uint64_t> default_buffer_sizes;

    static System &get()
    {
        static System instance;
        static bool initialized = false;
        if (!initialized)
        {
            instance.detect();
            initialized = true;
        }
        return instance;
    }

    const std::unordered_map<MemSpace, uint64_t> &getBufferSizes() const
    {
        return default_buffer_sizes;
    }

    const std::vector<Engine> &getAvailableEngines() const
    {
        return engines;
    }

    const std::vector<MemSpace> &getAvailableMemSpaces() const
    {
        return mem_spaces;
    }

    uint32_t getNumCudaDevices() const
    {
        return caps.num_cuda_devices;
    }

  private:
    void detect()
    {
        caps = HardwareCaps::get();
        default_buffer_sizes.clear();
        engines.clear();
        mem_spaces.clear();

        // 1. Storage default
        MemSpace storage{0, HandleType::STORAGE};
        default_buffer_sizes[storage] = 0;
        mem_spaces.push_back(storage);

        // 2. Host RAM Detection
        uint64_t total_ram = 16ULL * 1024 * 1024 * 1024;
#if defined(TG_OS_WINDOWS)
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&memInfo))
        {
            total_ram = memInfo.ullTotalPhys;
        }
#elif defined(TG_OS_LINUX)
        uint64_t pages = sysconf(_SC_PHYS_PAGES);
        uint64_t page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0)
        {
            total_ram = pages * page_size;
        }
#elif defined(TG_OS_MACOS)
        int64_t mac_mem = 0;
        size_t len = sizeof(mac_mem);
        if (sysctlbyname("hw.memsize", &mac_mem, &len, nullptr, 0) == 0 && mac_mem > 0)
        {
            total_ram = static_cast<uint64_t>(mac_mem);
        }
#endif
        uint64_t cpu_buffer_size = std::max<uint64_t>((uint64_t)(total_ram * 0.75), 4ULL * 1024 * 1024 * 1024);

        MemSpace cpu_ms{1, HandleType::CPP};
        default_buffer_sizes[cpu_ms] = cpu_buffer_size;
        mem_spaces.push_back(cpu_ms);

        engines.push_back(Engine{0, EngineType::CPU, {cpu_ms}});

        // 3. CUDA GPUs Detection
#ifdef TG_USE_CUDA
        if (caps.has_cuda && caps.num_cuda_devices > 0)
        {
            for (uint32_t dev = 0; dev < caps.num_cuda_devices; ++dev)
            {
                cudaSetDevice(dev);
                size_t free_mem = 0, total_mem = 0;
                cudaMemGetInfo(&free_mem, &total_mem);
                uint64_t vram_size =
                    (total_mem > 0) ? static_cast<uint64_t>(total_mem * 0.90) : 8ULL * 1024 * 1024 * 1024;

                MemSpace cuda_ms{dev, HandleType::CUDA};
                default_buffer_sizes[cuda_ms] = vram_size;
                mem_spaces.push_back(cuda_ms);

                // 1. Compute Engine (SMs) operates strictly on device VRAM
                engines.push_back(Engine{dev, EngineType::CUDA_GPU, {cuda_ms}});

                // 2. DMA Engine handles Host <-> Device and P2P transfers
                std::unordered_set<MemSpace> dma_supported = {cuda_ms, cpu_ms};
                for (uint32_t peer_dev = 0; peer_dev < caps.num_cuda_devices; ++peer_dev)
                {
                    if (peer_dev != dev)
                    {
                        dma_supported.insert(MemSpace{peer_dev, HandleType::CUDA});
                    }
                }
                engines.push_back(Engine{dev, EngineType::CUDA_DMA, dma_supported});
            }
        }
#endif

        // 4. OpenCL Detection
#ifdef TG_USE_OPENCL
        if (caps.has_opencl)
        {
            MemSpace cl_ms{1, HandleType::OPENCL};
            uint64_t opencl_mem = caps.opencl_max_alloc > 0 ? caps.opencl_max_alloc : (1ULL * 1024 * 1024 * 1024);
            default_buffer_sizes[cl_ms] = opencl_mem;
            mem_spaces.push_back(cl_ms);

            engines.push_back(Engine{0, EngineType::QUALCOMM_IGPU, {cl_ms}});
        }
#endif
        std::cout << "[System] Initialized with " << mem_spaces.size() << " MemSpaces and " << engines.size()
                  << " Engines:\n";
        for (const auto &mem_space : mem_spaces)
        {
            uint64_t size_bytes = default_buffer_sizes.count(mem_space) ? default_buffer_sizes.at(mem_space) : 0;
            double size_mb = static_cast<double>(size_bytes) / (1024.0 * 1024.0);
            std::cout << "  - " << mem_space << ": " << size_bytes << " bytes (" << size_mb << " MB)\n";
            LOG(INFO) << mem_space << ": " << size_bytes << " bytes (" << size_mb << " MB)";
        }
        for (const auto &engine : engines)
        {
            std::cout << "  - " << engine << "\n";
            LOG(INFO) << engine;
        }
    }
};