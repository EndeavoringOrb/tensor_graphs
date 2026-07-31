#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/memory.hpp"
#include "core/types.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <CL/cl.h>

class Synchronizer
{
  private:
    std::unordered_map<uint32_t, EngineType> buffer_last_writer;
    std::unordered_set<EngineType> busy_engines;

  public:
    Synchronizer() = default;

    // Checks if we need to synchronize any engines before executing an instruction
    void syncBefore(const OpInstruction &inst, EngineType current_engine)
    {
        std::unordered_set<EngineType> engines_to_sync;

        // Check if any input was last written by a different asynchronous engine
        for (const ParallelBuffer &inBuf : inst.inBuffers)
        {
            auto it = buffer_last_writer.find(inBuf.id.value);
            if (it != buffer_last_writer.end())
            {
                EngineType writer_engine = it->second;
                if (writer_engine != current_engine && busy_engines.count(writer_engine))
                {
                    engines_to_sync.insert(writer_engine);
                }
            }
        }

        for (EngineType engine : engines_to_sync)
        {
            syncEngine(engine);
        }
    }

    // Records that an engine has completed an instruction and claimed the output
    void markExecuted(const OpInstruction &inst, EngineType current_engine, bool issued_work)
    {
        if (issued_work && current_engine != EngineType::CPU)
        {
            busy_engines.insert(current_engine);
        }

        // Track who wrote to this memory space
        buffer_last_writer[inst.outBuffer.id.value] = current_engine;
    }

    // Handles the engine-specific synchronization call
    void syncEngine(EngineType engine)
    {
        if (!busy_engines.count(engine))
            return;

        if (engine == EngineType::CUDA_GPU)
        {
#ifdef USE_CUDA
            cudaDeviceSynchronize();
#endif
        }
        else if (engine == EngineType::QUALCOMM_IGPU)
        {
            if (OpenCLState::get().initialized)
            {
                clFinish(OpenCLState::get().queue);
            }
        }
        // Future async engines can be added here

        busy_engines.erase(engine);
    }

    // Perform a full wait on all currently busy engines
    void syncAll()
    {
        // Copy the set to avoid modifying it while iterating
        std::vector<EngineType> to_sync(busy_engines.begin(), busy_engines.end());
        for (EngineType engine : to_sync)
        {
            syncEngine(engine);
        }
    }
};