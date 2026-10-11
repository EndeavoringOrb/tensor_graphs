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
    std::unordered_map<uint32_t, std::vector<Engine>> buffer_last_writers;
    std::unordered_set<Engine> busy_engines;

    struct PendingAccess
    {
        ParallelBuffer buffer;
        bool writes;
    };
    std::unordered_map<Engine, std::vector<PendingAccess>> pending_accesses;

    static bool overlapsMemory(const ParallelBuffer &a, const ParallelBuffer &b)
    {
        if (a.mem_space != b.mem_space || a.size == 0 || b.size == 0)
            return false;
        if (a.offset < 0 || b.offset < 0)
            return a.id == b.id;
        return a.offset <= b.offset ? static_cast<uint64_t>(b.offset - a.offset) < a.size
                                    : static_cast<uint64_t>(a.offset - b.offset) < b.size;
    }

#ifdef TG_USE_CUDA
    std::unordered_map<Engine, cudaStream_t> cuda_streams;
    std::unordered_map<uint32_t, cudaEvent_t> buffer_events; // buffer_id -> event
#endif

  public:
    Synchronizer() = default;

    ~Synchronizer()
    {
#ifdef TG_USE_CUDA
        for (auto &kv : buffer_events)
        {
            if (kv.second)
            {
                cudaEventDestroy(kv.second);
            }
        }
        for (auto &kv : cuda_streams)
        {
            if (kv.second)
            {
                cudaSetDevice(kv.first.idx);
                cudaStreamSynchronize(kv.second);
                cudaStreamDestroy(kv.second);
            }
        }
#endif
    }

#ifdef TG_USE_CUDA
    cudaStream_t getCudaStream(const Engine &engine)
    {
        auto it = cuda_streams.find(engine);
        if (it != cuda_streams.end())
            return it->second;

        cudaSetDevice(engine.idx);
        cudaStream_t s = nullptr;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cuda_streams[engine] = s;
        return s;
    }

    cudaEvent_t getBufferEvent(uint32_t buffer_id, uint32_t dev_idx)
    {
        auto it = buffer_events.find(buffer_id);
        if (it != buffer_events.end())
            return it->second;

        cudaSetDevice(dev_idx);
        cudaEvent_t ev = nullptr;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        buffer_events[buffer_id] = ev;
        return ev;
    }
#endif

    void syncBefore(const OpInstruction &inst, const std::vector<Engine> &inst_engines)
    {
        if (inst_engines.empty())
            return;

        const Engine &primary_engine = inst_engines[0];

        // Allocation lifetimes follow dispatch order, but device work can still
        // be using an arena range when a different buffer reuses it. Buffer IDs
        // alone cannot describe these read/write and write/write dependencies.
        std::vector<Engine> conflicting_engines;
        for (const auto &entry : pending_accesses)
        {
            if (entry.first == primary_engine)
                continue;
            bool conflict = false;
            for (const PendingAccess &access : entry.second)
            {
                conflict = overlapsMemory(access.buffer, inst.outBuffer);
                if (!conflict && access.writes)
                {
                    for (const ParallelBuffer &input : inst.inBuffers)
                        conflict = conflict || overlapsMemory(access.buffer, input);
                }
                if (conflict)
                    break;
            }
            if (conflict)
                conflicting_engines.push_back(entry.first);
        }
        for (const Engine &engine : conflicting_engines)
            syncEngine(engine);

        for (const ParallelBuffer &inBuf : inst.inBuffers)
        {
            auto it = buffer_last_writers.find(inBuf.id.value);
            if (it == buffer_last_writers.end())
                continue;

            const std::vector<Engine> &writers = it->second;
            for (const Engine &writer_engine : writers)
            {
                if (!busy_engines.count(writer_engine))
                    continue;

                // In-stream operations on the primary engine are serialized by default
                if (writer_engine == primary_engine)
                    continue;

#ifdef TG_USE_CUDA
                bool writer_is_cuda =
                    (writer_engine.type == EngineType::CUDA_GPU || writer_engine.type == EngineType::CUDA_DMA);

                if (writer_is_cuda)
                {
                    auto evIt = buffer_events.find(inBuf.id.value);
                    if (evIt != buffer_events.end())
                    {
                        bool any_cuda_consumer = false;
                        for (const Engine &cur_eng : inst_engines)
                        {
                            if (cur_eng.type == EngineType::CUDA_GPU || cur_eng.type == EngineType::CUDA_DMA)
                            {
                                any_cuda_consumer = true;
                                if (cur_eng != writer_engine)
                                {
                                    cudaSetDevice(cur_eng.idx);
                                    cudaStream_t cur_stream = getCudaStream(cur_eng);
                                    cudaStreamWaitEvent(cur_stream, evIt->second, 0);
                                }
                            }
                        }

                        if (!any_cuda_consumer)
                        {
                            // CPU / Host consumer waiting on CUDA producer
                            cudaEventSynchronize(evIt->second);
                        }
                        continue;
                    }
                }
#endif
                // Fallback sync for OpenCL or non-event paths
                syncEngine(writer_engine);
            }
        }
    }

    void markExecuted(const OpInstruction &inst, const std::vector<Engine> &inst_engines, bool issued_work)
    {
        if (inst_engines.empty())
            return;

        const Engine &primary_engine = inst_engines[0];

        if (issued_work)
        {
#ifdef TG_USE_CUDA
            // CPU-dispatched uploads use the default stream. A pageable H2D
            // copy can return after staging, before its device write finishes;
            // our nonblocking consumer streams do not implicitly wait for it.
            if (primary_engine.type == EngineType::CPU && inst.outBuffer.mem_space.type == HandleType::CUDA)
            {
                cudaSetDevice(inst.outBuffer.mem_space.idx);
                const cudaError_t result = cudaStreamSynchronize(nullptr);
                if (result != cudaSuccess)
                    Error::throw_err(cudaGetErrorString(result));
            }
#endif
            for (const Engine &eng : inst_engines)
            {
                if (eng.type != EngineType::CPU)
                {
                    busy_engines.insert(eng);
                    auto &accesses = pending_accesses[eng];
                    for (const ParallelBuffer &input : inst.inBuffers)
                        accesses.push_back({input, false});
                    accesses.push_back({inst.outBuffer, true});
                }
            }

#ifdef TG_USE_CUDA
            if (primary_engine.type == EngineType::CUDA_GPU || primary_engine.type == EngineType::CUDA_DMA)
            {
                cudaSetDevice(primary_engine.idx);
                cudaStream_t stream = getCudaStream(primary_engine);
                cudaEvent_t ev = getBufferEvent(inst.outBuffer.id.value, primary_engine.idx);
                cudaEventRecord(ev, stream);

                // Synchronize auxiliary CUDA engines (e.g. source DMA engine) with the completion event
                for (size_t i = 1; i < inst_engines.size(); ++i)
                {
                    const Engine &other_eng = inst_engines[i];
                    if (other_eng.type == EngineType::CUDA_GPU || other_eng.type == EngineType::CUDA_DMA)
                    {
                        cudaSetDevice(other_eng.idx);
                        cudaStream_t other_stream = getCudaStream(other_eng);
                        cudaStreamWaitEvent(other_stream, ev, 0);
                    }
                }
            }
#endif
        }

        buffer_last_writers[inst.outBuffer.id.value] = inst_engines;
    }

    void syncEngine(const Engine &engine)
    {
        if (!busy_engines.count(engine))
            return;

#ifdef TG_USE_CUDA
        if (engine.type == EngineType::CUDA_GPU || engine.type == EngineType::CUDA_DMA)
        {
            cudaSetDevice(engine.idx);
            auto it = cuda_streams.find(engine);
            if (it != cuda_streams.end() && it->second != nullptr)
            {
                cudaStreamSynchronize(it->second);
            }
            else
            {
                cudaDeviceSynchronize();
            }
        }
#endif
#ifdef TG_USE_OPENCL
        if (engine.type == EngineType::QUALCOMM_IGPU)
        {
            if (OpenCLState::get().initialized)
            {
                clFinish(OpenCLState::get().queue);
            }
        }
#endif
        busy_engines.erase(engine);
        pending_accesses.erase(engine);
    }

    void syncEngines(const std::vector<Engine> &engines)
    {
        for (const Engine &eng : engines)
        {
            syncEngine(eng);
        }
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
