#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/memory.hpp"
#include "core/types.hpp"

#ifdef TG_USE_CUDA
#include <cuda_runtime.h>
#endif

class Synchronizer
{
  private:
    std::unordered_map<uint32_t, Engine> buffer_last_writer;
    std::unordered_set<Engine> busy_engines;
#ifdef TG_USE_CUDA
    std::unordered_map<uint32_t, cudaStream_t> cuda_streams;
#endif

  public:
    Synchronizer() = default;

    ~Synchronizer()
    {
#ifdef TG_USE_CUDA
        for (auto &kv : cuda_streams)
        {
            if (kv.second)
            {
                cudaSetDevice(kv.first);
                cudaStreamSynchronize(kv.second);
                cudaStreamDestroy(kv.second);
            }
        }
#endif
    }

#ifdef TG_USE_CUDA
    cudaStream_t getCudaStream(uint32_t dev_idx)
    {
        auto it = cuda_streams.find(dev_idx);
        if (it != cuda_streams.end())
            return it->second;
        cudaSetDevice(dev_idx);
        cudaStream_t s = nullptr;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cuda_streams[dev_idx] = s;
        return s;
    }
#endif

    void syncBefore(const OpInstruction &inst, const Engine &current_engine)
    {
        std::unordered_set<Engine> engines_to_sync;

        for (const ParallelBuffer &inBuf : inst.inBuffers)
        {
            auto it = buffer_last_writer.find(inBuf.id.value);
            if (it != buffer_last_writer.end())
            {
                const Engine &writer_engine = it->second;
                if (writer_engine != current_engine && busy_engines.count(writer_engine))
                {
                    engines_to_sync.insert(writer_engine);
                }
            }
        }

        for (const Engine &engine : engines_to_sync)
        {
            syncEngine(engine);
        }
    }

    void markExecuted(const OpInstruction &inst, const Engine &current_engine, bool issued_work)
    {
        if (issued_work && current_engine.type != EngineType::CPU)
        {
            busy_engines.insert(current_engine);
        }
        buffer_last_writer[inst.outBuffer.id.value] = current_engine;
    }

    void syncEngine(const Engine &engine)
    {
        if (!busy_engines.count(engine))
            return;

        if (engine.type == EngineType::CUDA_GPU)
        {
#ifdef TG_USE_CUDA
            cudaSetDevice(engine.idx);
            auto it = cuda_streams.find(engine.idx);
            if (it != cuda_streams.end() && it->second != nullptr)
            {
                cudaStreamSynchronize(it->second);
            }
            else
            {
                cudaDeviceSynchronize();
            }
#endif
        }
        else if (engine.type == EngineType::QUALCOMM_IGPU)
        {
#ifdef TG_USE_OPENCL
            if (OpenCLState::get().initialized)
            {
                clFinish(OpenCLState::get().queue);
            }
#endif
        }

        busy_engines.erase(engine);
    }

    void syncAll()
    {
        std::vector<Engine> to_sync(busy_engines.begin(), busy_engines.end());
        for (const Engine &engine : to_sync)
        {
            syncEngine(engine);
        }
    }
};