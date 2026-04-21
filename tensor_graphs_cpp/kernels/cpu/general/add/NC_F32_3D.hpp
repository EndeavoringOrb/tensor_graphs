// File: tensor_graphs_cpp/kernels/cpu/general/add/NC_F32_3D.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <thread>
#include <vector>
#include <algorithm>

inline bool matchAddNC_F32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 3)
        return false;
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    return true;
}

inline void runAddNC_F32_3D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                            const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *a = static_cast<const float *>(inputs[0]);
    const float *b = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    uint32_t B = inViews[0].getShape()[0];
    uint32_t M = inViews[0].getShape()[1];
    uint32_t N = inViews[0].getShape()[2];

    uint64_t a_str0 = inViews[0].strides[0];
    uint64_t a_str1 = inViews[0].strides[1];
    uint64_t a_str2 = inViews[0].strides[2];

    uint64_t b_str0 = inViews[1].strides[0];
    uint64_t b_str1 = inViews[1].strides[1];
    uint64_t b_str2 = inViews[1].strides[2];

    uint64_t out_str0 = outViews[0].strides[0];
    uint64_t out_str1 = outViews[0].strides[1];
    uint64_t out_str2 = outViews[0].strides[2];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    uint32_t b_per_thread = (B + num_threads - 1) / num_threads;

    if (b_per_thread == 0)
    {
        b_per_thread = 1;
        num_threads = B;
    }

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t start_b = t * b_per_thread;
            uint32_t end_b = std::min(start_b + b_per_thread, B);

            for (uint32_t i = start_b; i < end_b; ++i) {
                for (uint32_t j = 0; j < M; ++j) {
                    const float* a_row = a + i * a_str0 + j * a_str1;
                    const float* b_row = b + i * b_str0 + j * b_str1;
                    float* out_row = out + i * out_str0 + j * out_str1;

                    if (a_str2 == 1 && b_str2 == 1 && out_str2 == 1) {
                        for (uint32_t k = 0; k < N; ++k) {
                            out_row[k] = a_row[k] + b_row[k];
                        }
                    } else {
                        for (uint32_t k = 0; k < N; ++k) {
                            out_row[k * out_str2] = a_row[k * a_str2] + b_row[k * b_str2];
                        }
                    }
                }
            } });
    }
    for (auto &w : workers)
        w.join();
}

inline uint32_t refFactoryAddNC_F32_3D(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.add(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Add_NC_F32_3D", 2, matchAddNC_F32_3D, runAddNC_F32_3D, refFactoryAddNC_F32_3D, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 1, 1}, {1, 1, 1}}, {false, false}, {{Backend::CPU}, {Backend::CPU}});