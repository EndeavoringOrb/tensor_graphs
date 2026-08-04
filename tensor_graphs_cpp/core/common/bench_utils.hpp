#pragma once

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef TG_USE_CUDA
#include <cuda_runtime.h>
#endif

#include "core/cost_model.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"
#include "core/types.hpp"

#ifdef TG_OS_WINDOWS
#include <fcntl.h>
#include <io.h>
#include <share.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

// =============================================================================
// Extensible Unified RAII Device Buffer
// =============================================================================
struct BenchBuffer
{
    MemSpace mem_space = {1, HandleType::CPP};
    uint64_t bytes = 0;
    std::vector<uint8_t> hostData;
    void *devicePtr = nullptr;
    cl_mem clMem = nullptr;

    BenchBuffer() = default;
    ~BenchBuffer()
    {
        free();
    }

    // Disable copy
    BenchBuffer(const BenchBuffer &) = delete;
    BenchBuffer &operator=(const BenchBuffer &) = delete;

    // Support move
    BenchBuffer(BenchBuffer &&o) noexcept
    {
        *this = std::move(o);
    }

    BenchBuffer &operator=(BenchBuffer &&o) noexcept
    {
        if (this != &o)
        {
            free();
            mem_space = o.mem_space;
            bytes = o.bytes;
            hostData = std::move(o.hostData);
            devicePtr = o.devicePtr;
            clMem = o.clMem;
            o.devicePtr = nullptr;
            o.clMem = nullptr;
            o.bytes = 0;
        }
        return *this;
    }

    void allocate(MemSpace ms, uint64_t size)
    {
        free();
        mem_space = ms;
        bytes = size == 0 ? 1 : size;

        hostData.resize(bytes, 0);

        if (mem_space.type == HandleType::CUDA)
        {
#ifdef TG_USE_CUDA
            cudaError_t err = cudaMalloc(&devicePtr, bytes);
            if (err != cudaSuccess)
            {
                Error::throw_err("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
            }
#else
            Error::throw_err("CUDA backend requested but TG_USE_CUDA is not defined.");
#endif
        }
        else if (mem_space.type == HandleType::OPENCL)
        {
#ifdef TG_USE_OPENCL
            OpenCLState::get().init();
            cl_context ctx = OpenCLState::get().context;
            if (!ctx)
            {
                Error::throw_err("OpenCL context not initialized.");
            }
            devicePtr = hostData.data();
            cl_int err;
            clMem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, bytes, devicePtr, &err);
            if (err != CL_SUCCESS || !clMem)
            {
                Error::throw_err("clCreateBuffer failed to allocate memory of size " + std::to_string(bytes) +
                                 ". Error: " + std::to_string(err));
            }
#else
            Error::throw_err("OPENCL backend requested but TG_USE_OPENCL is not defined.");
#endif
        }
        else
        {
            devicePtr = hostData.data();
        }
    }

    void upload()
    {
        if (mem_space.type == HandleType::CUDA)
        {
#ifdef TG_USE_CUDA
            cudaError_t err = cudaMemcpy(devicePtr, hostData.data(), bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                Error::throw_err("cudaMemcpy HostToDevice failed: " + std::string(cudaGetErrorString(err)));
            }
#endif
        }
        else if (mem_space.type == HandleType::OPENCL)
        {
#ifdef TG_USE_OPENCL
            if (clMem && !hostData.empty())
            {
                cl_int err = clEnqueueWriteBuffer(OpenCLState::get().queue, clMem,
                                                  CL_TRUE, // Blocking write to guarantee host-to-device visibility
                                                  0, bytes, hostData.data(), 0, nullptr, nullptr);
                if (err != CL_SUCCESS)
                {
                    Error::throw_err("clEnqueueWriteBuffer failed with error: " + std::to_string(err));
                }
            }
#else
            Error::throw_err("OPENCL backend requested but TG_USE_OPENCL is not defined.");
#endif
        }
    }

    void download()
    {
        if (mem_space.type == HandleType::CUDA)
        {
#ifdef TG_USE_CUDA
            cudaError_t err = cudaMemcpy(hostData.data(), devicePtr, bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                Error::throw_err("cudaMemcpy DeviceToHost failed: " + std::string(cudaGetErrorString(err)));
            }
#endif
        }
        else if (mem_space.type == HandleType::OPENCL)
        {
#ifdef TG_USE_OPENCL
            if (clMem && !hostData.empty())
            {
                cl_int err = clEnqueueReadBuffer(OpenCLState::get().queue, clMem,
                                                 CL_TRUE, // Blocking read to guarantee device-to-host visibility
                                                 0, bytes, hostData.data(), 0, nullptr, nullptr);
                if (err != CL_SUCCESS)
                {
                    Error::throw_err("clEnqueueReadBuffer failed with error: " + std::to_string(err));
                }
            }
#else
            Error::throw_err("OPENCL backend requested but TG_USE_OPENCL is not defined.");
#endif
        }
    }

    void free()
    {
        if (devicePtr)
        {
            if (mem_space.type == HandleType::CUDA)
            {
#ifdef TG_USE_CUDA
                cudaFree(devicePtr);
#endif
            }
            else if (mem_space.type == HandleType::OPENCL)
            {
#ifdef TG_USE_OPENCL
                if (clMem)
                {
                    clReleaseMemObject(clMem);
                    clMem = nullptr;
                }
#else
                Error::throw_err("OPENCL backend requested but TG_USE_OPENCL is not defined.");
#endif
            }
            devicePtr = nullptr;
        }
        bytes = 0;
        hostData.clear();
    }

    const void *getReadPtr() const
    {
        return (mem_space.type == HandleType::STORAGE) ? nullptr : devicePtr;
    }

    void *getWritePtr()
    {
        return devicePtr;
    }
};

struct StorageFiles
{
    std::vector<std::string> paths;
    std::vector<int> fds;

    StorageFiles() = default;

    StorageFiles(const StorageFiles &) = delete;
    StorageFiles &operator=(const StorageFiles &) = delete;

    StorageFiles(StorageFiles &&other) noexcept : paths(std::move(other.paths)), fds(std::move(other.fds))
    {
    }

    StorageFiles &operator=(StorageFiles &&other) noexcept
    {
        if (this != &other)
        {
            clear();
            paths = std::move(other.paths);
            fds = std::move(other.fds);
        }
        return *this;
    }

    void clear()
    {
        for (int fd : fds)
        {
            if (fd >= 0)
            {
#ifdef TG_OS_WINDOWS
                _close(fd);
#else
                close(fd);
#endif
            }
        }
        fds.clear();
        for (const auto &path : paths)
        {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
        paths.clear();
    }

    ~StorageFiles()
    {
        clear();
    }
};

inline StorageFiles createStorageInputs(const Record &r, const KernelEntry &kernel, int runIdx,
                                        const std::vector<BenchBuffer> *inputBuffers = nullptr)
{
    StorageFiles sf;
    std::vector<char> dummyBuf(1024 * 1024, 0);

    for (uint64_t idx = 0; idx < r.inputShapes.size(); ++idx)
    {
        MemSpace b = {1, HandleType::CPP};
        if (!r.input_mem_spaces.empty() && idx < r.input_mem_spaces.size())
            b = r.input_mem_spaces[idx];

        if (b.type == HandleType::STORAGE)
        {
            uint64_t elements = countElements(r.inputShapes[idx]);
            uint64_t bytes = elements * getDTypeSize(r.inputDTypes[idx]);
            if (bytes == 0)
            {
                Error::throw_err("[createStorageInputs] got 0 bytes for file size");
            }

            std::string path = "benchmarks/dummy_storage_" + std::to_string(sf.fds.size()) + "_" +
                               std::to_string(r.kernelId.value) + "_run_" + std::to_string(runIdx) + ".bin";
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open())
            {
                std::cerr << "Failed to create dummy storage file: " << path << std::endl;
                continue;
            }

            // Write prepared host-side data if available; otherwise, write fallback
            // zeroes
            if (inputBuffers && idx < inputBuffers->size() && !(*inputBuffers)[idx].hostData.empty())
            {
                out.write(reinterpret_cast<const char *>((*inputBuffers)[idx].hostData.data()), bytes);
            }
            else
            {
                uint64_t written = 0;
                while (written < bytes)
                {
                    uint64_t toWrite = std::min<uint64_t>(dummyBuf.size(), bytes - written);
                    out.write(dummyBuf.data(), toWrite);
                    written += toWrite;
                }
            }
            out.close();
            sf.paths.push_back(path);

            int fd = -1;
#ifdef TG_OS_WINDOWS
            _wsopen_s(&fd, std::filesystem::path(path).c_str(), _O_RDONLY | _O_BINARY, _SH_DENYNO, 0);
#else
            fd = open(path.c_str(), O_RDONLY);
#endif
            if (fd < 0)
            {
                std::cerr << "Failed to open dummy storage file for reading: " << path << std::endl;
            }
            sf.fds.push_back(fd);
        }
    }
    return sf;
}

inline void synchronizeHandle(HandleType handle)
{
    if (handle == HandleType::CUDA)
    {
#ifdef TG_USE_CUDA
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            Error::throw_err("CUDA Synchronization failed: " + std::string(cudaGetErrorString(err)));
        }
#else
        Error::throw_err("CUDA backend requested but TG_USE_CUDA is not defined.");
#endif
    }
    else if (handle == HandleType::OPENCL)
    {
#ifdef TG_USE_OPENCL
        clFinish(OpenCLState::get().queue);
#else
        Error::throw_err("OPENCL backend requested but TG_USE_OPENCL is not defined.");
#endif
    }
}

// =============================================================================
// PreparedKernel helper encapsulating allocation, copy, and execution context
// =============================================================================
struct PreparedKernel
{
    std::vector<BenchBuffer> inputBuffers;
    std::vector<BenchBuffer> outputBuffers;
    std::vector<const void *> inPtrs;
    std::vector<TensorView> inViews;
    std::vector<void *> outPtrs;
    std::vector<TensorView> outViews;
    KernelContext ctx;
    StorageFiles sf;

    void prepare(const KernelEntry &kernel, const Record &r,
                 const std::vector<std::vector<uint8_t>> *explicitInputData = nullptr)
    {
        inputBuffers.resize(r.inputShapes.size());
        inPtrs.assign(r.inputShapes.size(), nullptr);
        inViews.resize(r.inputShapes.size());

        outputBuffers.resize(1);
        outPtrs.assign(1, nullptr);
        outViews.resize(1);

        for (uint64_t idx = 0; idx < r.inputShapes.size(); ++idx)
        {
            uint64_t maxIndex = 0;
            for (uint64_t d = 0; d < r.inputShapes[idx].size(); ++d)
            {
                if (r.inputShapes[idx][d] > 0)
                {
                    maxIndex += (r.inputShapes[idx][d] - 1) * r.inputStrides[idx][d];
                }
            }
            uint64_t elements = r.inputShapes[idx].empty() ? 1 : maxIndex + 1;

            if (elements == 0)
                elements = 1;
            uint64_t bytes = elements * getDTypeSize(r.inputDTypes[idx]);

            MemSpace b = {1, HandleType::CPP};
            if (!r.input_mem_spaces.empty() && idx < r.input_mem_spaces.size())
                b = r.input_mem_spaces[idx];

            inputBuffers[idx].allocate(b, bytes);

            if (explicitInputData && idx < explicitInputData->size() && !(*explicitInputData)[idx].empty())
            {
                std::memcpy(inputBuffers[idx].hostData.data(), (*explicitInputData)[idx].data(),
                            std::min(bytes, (uint64_t)(*explicitInputData)[idx].size()));
            }
            else if (idx < r.inputConstants.size() && !r.inputConstants[idx].empty() &&
                     r.inputConstants[idx].size() == bytes)
            {
                std::memcpy(inputBuffers[idx].hostData.data(), r.inputConstants[idx].data(), bytes);
            }
            else
            {
                if (r.inputDTypes[idx] == DType::FLOAT32)
                {
                    float *fptr = reinterpret_cast<float *>(inputBuffers[idx].hostData.data());
                    for (uint64_t k = 0; k < elements; ++k)
                        fptr[k] = 1.0f;
                }
                else if (r.inputDTypes[idx] == DType::INT32)
                {
                    int32_t *iptr = reinterpret_cast<int32_t *>(inputBuffers[idx].hostData.data());
                    if (kernel.opType == OpType::PERMUTE || kernel.opName.find("Permute") != std::string::npos)
                    {
                        if (idx == 1 && r.inputShapes.size() > 0 && r.inputShapes[0].size() == r.outputShape.size() &&
                            elements == r.inputShapes[0].size())
                        {
                            std::vector<bool> used(elements, false);
                            for (uint64_t k = 0; k < elements; ++k)
                            {
                                uint64_t found_d = k;
                                for (uint64_t d = 0; d < elements; ++d)
                                {
                                    if (!used[d] && r.inputShapes[0][d] == r.outputShape[k])
                                    {
                                        found_d = d;
                                        break;
                                    }
                                }
                                used[found_d] = true;
                                iptr[k] = found_d;
                            }
                        }
                        else
                        {
                            for (uint64_t k = 0; k < elements; ++k)
                                iptr[k] = k;
                        }
                    }
                    else if (kernel.opType == OpType::CONCAT || kernel.opName.find("Concat") != std::string::npos)
                    {
                        if (idx == r.inputShapes.size() - 1)
                        {
                            int32_t concat_axis = -1;
                            if (!r.inputShapes.empty() && !r.outputShape.empty())
                            {
                                for (uint64_t d = 0; d < r.outputShape.size(); ++d)
                                {
                                    if (r.outputShape[d] != r.inputShapes[0][d])
                                    {
                                        concat_axis = (int32_t)d;
                                        break;
                                    }
                                }
                            }
                            if (concat_axis == -1)
                                concat_axis = 0;
                            for (uint64_t k = 0; k < elements; ++k)
                                iptr[k] = concat_axis;
                        }
                        else
                        {
                            for (uint64_t k = 0; k < elements; ++k)
                                iptr[k] = 1;
                        }
                    }
                    else
                    {
                        for (uint64_t k = 0; k < elements; ++k)
                            iptr[k] = 1;
                    }
                }
                else if (r.inputDTypes[idx] == DType::BF16)
                {
                    uint16_t *bptr = reinterpret_cast<uint16_t *>(inputBuffers[idx].hostData.data());
                    for (uint64_t k = 0; k < elements; ++k)
                        bptr[k] = 0x3F80;
                }
                else
                {
                    std::memset(inputBuffers[idx].hostData.data(), 1, bytes);
                }
            }

            inputBuffers[idx].upload();
            inPtrs[idx] = inputBuffers[idx].getReadPtr();

            inViews[idx].setShape(r.inputShapes[idx]);
            inViews[idx].strides = r.inputStrides[idx];
            inViews[idx].offset = 0;
            inViews[idx].dtype = r.inputDTypes[idx];
        }

        {
            uint64_t maxIndex = 0;
            for (uint64_t d = 0; d < r.outputShape.size(); ++d)
            {
                if (r.outputShape[d] > 0)
                {
                    maxIndex += (r.outputShape[d] - 1) * r.outputStrides[d];
                }
            }
            uint64_t elements = r.outputShape.empty() ? 1 : maxIndex + 1;

            if (elements == 0)
                elements = 1;
            uint64_t bytes = elements * getDTypeSize(r.outputDType);

            MemSpace outBackend = r.output_mem_space;
            outputBuffers[0].allocate(outBackend, bytes);

            outPtrs[0] = outputBuffers[0].getWritePtr();

            outViews[0].setShape(r.outputShape);
            outViews[0].strides = r.outputStrides;
            outViews[0].offset = 0;
            outViews[0].dtype = r.outputDType;
        }

        ctx.inputs = inPtrs;
        ctx.outputs = outPtrs;
        ctx.inViews = inViews;
        ctx.outViews = outViews;
        ctx.fd.assign(inPtrs.size(), -1);

        for (uint64_t idx = 0; idx < inputBuffers.size(); ++idx)
        {
            ctx.cl_inputs.push_back(inputBuffers[idx].clMem);
        }
        for (uint64_t idx = 0; idx < outputBuffers.size(); ++idx)
        {
            ctx.cl_outputs.push_back(outputBuffers[idx].clMem);
        }
    }

    void updateStorageContext(const KernelEntry &kernel, const Record &r, int runIdx)
    {
        sf = createStorageInputs(r, kernel, runIdx, &inputBuffers);
        uint64_t storageInIdx = 0;
        for (uint64_t idx = 0; idx < r.inputShapes.size(); ++idx)
        {
            MemSpace b = {1, HandleType::CPP};
            if (!r.input_mem_spaces.empty() && idx < r.input_mem_spaces.size())
                b = r.input_mem_spaces[idx];

            if (b.type == HandleType::STORAGE)
            {
                if (storageInIdx < sf.fds.size())
                {
                    ctx.fd[idx] = sf.fds[storageInIdx++];
                }
            }
        }
    }

    void run(const KernelEntry &kernel)
    {
        if (kernel.run)
        {
            kernel.run(ctx);
        }
    }

    void synchronize()
    {
        for (const auto &buf : inputBuffers)
        {
            synchronizeHandle(buf.mem_space.type);
        }
        for (const auto &buf : outputBuffers)
        {
            synchronizeHandle(buf.mem_space.type);
        }
    }

    void download()
    {
        for (auto &buf : outputBuffers)
        {
            buf.download();
        }
    }
};

std::unordered_map<KernelId, std::vector<Record>> loadCallRecords(const std::string &path)
{
    std::unordered_map<KernelId, std::vector<Record>> records;
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        return records;

    BinaryReader br(file);
    while (file.peek() != EOF)
    {
        Record r;
        br.read(r);
        records[r.kernelId].push_back(std::move(r));
    }
    return records;
}

struct CacheFile
{
    uint32_t version = 0;
    LogicalId rootId;
    std::unordered_map<LogicalId, MemSpace> selectedCachedNodes;
    std::vector<CompiledGraph> compiledGraphs;
    std::unordered_map<LogicalId, std::shared_ptr<std::vector<uint8_t>>> constants;
    bool isValid = true;
    std::string invalidReason;
};

inline CacheFile loadCacheFile(const std::string &cachePath, bool validateKernels = false)
{
    CacheFile cache;
    if (cachePath.empty())
    {
        cache.isValid = false;
        cache.invalidReason = "Cache path is empty";
        return cache;
    }

    std::ifstream file(cachePath, std::ios::binary);
    if (!file.is_open())
    {
        cache.isValid = false;
        cache.invalidReason = "Could not open cache file: " + cachePath;
        return cache;
    }

    BinaryReader br(file);
    while (file.peek() != EOF)
    {
        uint8_t type = 0;
        br.read(type);

        if (type == 0) // Metadata
        {
            br.read(cache.version);
            br.read(cache.rootId);
            br.read(cache.selectedCachedNodes);
        }
        else if (type == 1) // Compiled Bucket
        {
            CompiledGraph cg;
            br.read(cg);

            if (validateKernels)
            {
                for (const auto &inst : cg.instructions)
                {
                    if (inst.kernel_id == KernelId{0} || !KernelRegistry::get().hasKernel(inst.kernel_id))
                    {
                        cache.isValid = false;
                        cache.invalidReason = "Invalid Kernel ID: " + toString(inst.kernel_id);
                        break;
                    }
                }
                if (!cache.isValid)
                    break;
            }
            cache.compiledGraphs.push_back(std::move(cg));
        }
        else if (type == 2) // Constants
        {
            uint32_t count = 0;
            br.read(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                LogicalId nodeId;
                std::vector<uint8_t> data;
                br.read(nodeId);
                br.read(data);
                cache.constants[nodeId] = std::make_shared<std::vector<uint8_t>>(std::move(data));
            }
        }
        else
        {
            cache.isValid = false;
            cache.invalidReason = "Unknown block type";
            break;
        }
    }

    return cache;
}

inline std::unordered_map<KernelId, std::vector<Record>> getRecordsFromCache(const std::string &cachePath)
{
    std::unordered_map<KernelId, std::vector<Record>> recordsByUid;
    std::unordered_set<std::string> seen;

    CacheFile cache = loadCacheFile(cachePath, /*validateKernels=*/false);
    if (!cache.isValid)
    {
        std::cerr << "Warning: " << cache.invalidReason << std::endl;
        return recordsByUid;
    }

    for (const auto &cg : cache.compiledGraphs)
    {
        for (const auto &inst : cg.instructions)
        {
            if (inst.kernel_id.value == 0 || !KernelRegistry::get().hasKernel(inst.kernel_id))
                continue;

            Record r;
            r.kernelId = inst.kernel_id;
            r.buildContextId = BUILD_CONTEXT_ID;
            r.hwTag = HW_TAG;
            r.runTime = 0.0f;

            const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.kernel_id);
            r.output_mem_space = kernel.output_mem_space;
            r.engines = kernel.engines;

            if (cg.nodeViews.count(inst.eclass_id))
            {
                const TensorView &outView = cg.nodeViews.at(inst.eclass_id);
                r.outputShape = outView.getShape();
                r.outputStrides = outView.strides;
                r.outputDType = outView.dtype;
            }

            for (uint32_t i = 0; i < inst.children.size(); i++)
            {
                EClassId inId = inst.children[i];
                if (!cg.nodeViews.count(inId))
                    continue;

                const TensorView &inView = cg.nodeViews.at(inId);
                r.inputShapes.push_back(inView.getShape());
                r.inputStrides.push_back(inView.strides);
                r.inputDTypes.push_back(inView.dtype);

                uint64_t ruleIdx = std::min(
                    (uint64_t)i,
                    static_cast<uint64_t>(kernel.input_mem_spaces.empty() ? 0 : kernel.input_mem_spaces.size() - 1));
                MemSpace ms = {1, HandleType::CPP};
                if (!kernel.input_mem_spaces.empty() && ruleIdx < kernel.input_mem_spaces.size())
                {
                    ms = kernel.input_mem_spaces[ruleIdx];
                }
                r.input_mem_spaces.push_back(ms);

                if (cg.has_logical_id(inId))
                {
                    LogicalId logicalId = cg.get_logical_id(inId);
                    if (cg.constantStaging.count(EClassId{logicalId.value}))
                    {
                        r.inputConstants.push_back(*cg.constantStaging.at(EClassId{logicalId.value}));
                    }
                    else if (cg.constantStaging.count(inId))
                    {
                        r.inputConstants.push_back(*cg.constantStaging.at(inId));
                    }
                    else if (cache.constants.count(logicalId))
                    {
                        r.inputConstants.push_back(*cache.constants.at(logicalId));
                    }
                    else
                    {
                        r.inputConstants.push_back({});
                    }
                }
                else if (cg.constantStaging.count(inId))
                {
                    r.inputConstants.push_back(*cg.constantStaging.at(inId));
                }
                else
                {
                    r.inputConstants.push_back({});
                }
            }

            std::string sig = serializeToString(r);
            if (seen.insert(sig).second)
            {
                recordsByUid[r.kernelId].push_back(r);
            }
        }
    }
    return recordsByUid;
}