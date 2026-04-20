#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <thread>
#include <vector>
#include <algorithm>
#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

inline bool matchAddF32_3D_Broadcast0_Inplace(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 3)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    if (!isContiguous(inputs[0]))
        return false;
    if (inputs[0].storageType == StorageType::PERSISTENT)
        return false;

    if (inputs[1].strides[0] != 0)
        return false;
    if (inputs[1].strides[2] != 1)
        return false;
    if (inputs[1].strides[1] != inputs[1].getShape()[2])
        return false;

    return true;
}

inline void runAddF32_3D_Broadcast0_Inplace(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                            const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    float *out = static_cast<float *>(outputs[0]);
    const float *b = static_cast<const float *>(inputs[1]);

    uint32_t B = inViews[0].getShape()[0];
    uint32_t M = inViews[0].getShape()[1];
    uint32_t N = inViews[0].getShape()[2];

    uint64_t mn_size = M * N;
    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;
    if (num_threads > B) num_threads = B;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]() {
            uint32_t start_b = (B * t) / num_threads;
            uint32_t end_b = (B * (t + 1)) / num_threads;
            
            for (uint32_t batch = start_b; batch < end_b; ++batch)
            {
                float *out_batch = out + batch * mn_size;
                uint32_t i = 0;
#if defined(TG_HAS_NEON)
                for (; i + 4 <= mn_size; i += 4)
                {
                    float32x4_t va = vld1q_f32(out_batch + i);
                    float32x4_t vb = vld1q_f32(b + i); 
                    vst1q_f32(out_batch + i, vaddq_f32(va, vb));
                }
#endif
                for (; i < mn_size; ++i)
                {
                    out_batch[i] += b[i];
                }
            }
        });
    }
    for (auto &w : workers) w.join();
}

inline uint32_t refFactoryAdd3D_Broadcast0_Inplace(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.add(inputs[0], inputs[1]);
}

REGISTER_KERNEL_INPLACE("Add_3D_Broadcast0_inplace", 2, matchAddF32_3D_Broadcast0_Inplace, runAddF32_3D_Broadcast0_Inplace, refFactoryAdd3D_Broadcast0_Inplace, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 8, 2048}, {1, 8, 2048}}, {true, false}, {{Backend::CPU}, {Backend::CPU}});