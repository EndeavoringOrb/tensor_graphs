#pragma once

#include "core/synchronizer.hpp"

#ifdef TG_USE_CUDA
#include <atomic>
#include <chrono>
#include <thread>

inline void CUDART_CB completeDelayedAccess(void *data)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    static_cast<std::atomic<bool> *>(data)->store(true);
}

inline void testCudaArenaReuse(bool prior_write, bool next_write, bool same_id)
{
    std::atomic<bool> completed{false};
    Synchronizer sync;
    const Engine gpu{0, EngineType::CUDA_GPU};
    const Engine cpu{0, EngineType::CPU};
    const MemSpace device{0, HandleType::CUDA};
    ParallelBuffer original{BufferId{1}, device, 64, 0, 1, 128};
    ParallelBuffer reused{BufferId{same_id ? 1u : 2u}, device, 32, 2, 3, 144};
    ParallelBuffer unrelated{BufferId{3}, device, 64, 0, 3, 1024};

    OpInstruction producer;
    producer.inBuffers = {prior_write ? unrelated : original};
    producer.outBuffer = prior_write ? original : unrelated;
    const cudaError_t result = cudaLaunchHostFunc(sync.getCudaStream(gpu), completeDelayedAccess, &completed);
    if (result != cudaSuccess)
        Error::throw_err(cudaGetErrorString(result));
    sync.markExecuted(producer, {gpu}, true);

    OpInstruction consumer;
    consumer.inBuffers = {next_write ? unrelated : reused};
    consumer.outBuffer = next_write ? reused : unrelated;
    // Keep the other range disjoint so only the intended hazard triggers a wait.
    consumer.inBuffers[0].offset = next_write ? 2048 : reused.offset;
    consumer.outBuffer.offset = next_write ? reused.offset : 2048;
    sync.syncBefore(consumer, {cpu});
    if (!completed.load())
        Error::throw_err("CUDA arena reuse did not wait for the outstanding access");
}

inline void testCpuCudaUploadCompletion()
{
    std::atomic<bool> completed{false};
    Synchronizer sync;
    OpInstruction upload;
    upload.outBuffer = {BufferId{1}, MemSpace{0, HandleType::CUDA}, 64, 0, 1, 0};
    cudaSetDevice(0);
    const cudaError_t result = cudaLaunchHostFunc(nullptr, completeDelayedAccess, &completed);
    if (result != cudaSuccess)
        Error::throw_err(cudaGetErrorString(result));
    sync.markExecuted(upload, {Engine{0, EngineType::CPU}}, true);
    const bool upload_completed = completed.load();
    // Drain even on failure so the callback cannot outlive its stack data.
    cudaStreamSynchronize(nullptr);
    if (!upload_completed)
        Error::throw_err("CPU-dispatched CUDA upload was marked complete before its device write");
}
#endif

inline void runCudaSyncRegressionTests()
{
#ifdef TG_USE_CUDA
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
        std::cout << "Skipping CUDA synchronization tests: no CUDA device" << std::endl;
        return;
    }
    testCudaArenaReuse(false, true, false); // outstanding read, newly allocated writer
    testCudaArenaReuse(true, true, false);  // outstanding write, newly allocated writer
    testCudaArenaReuse(true, false, false); // aliased producer/consumer with different IDs
    testCudaArenaReuse(false, true, true);  // in-place write after outstanding read
    testCpuCudaUploadCompletion();
    std::cout << "CUDA synchronization regression tests passed" << std::endl;
#else
    std::cout << "Skipping CUDA synchronization tests: CUDA disabled" << std::endl;
#endif
}
