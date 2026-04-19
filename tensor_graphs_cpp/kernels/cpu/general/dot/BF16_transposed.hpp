#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <thread>
#include <vector>
#include <cstring>
#include <algorithm>

inline bool matchDotTransposedBF16(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::BF16)
        return false;
    if (output.dtype != DType::FLOAT32)
        return false;

    auto sX = inputs[0].getShape();
    auto sW = inputs[1].getShape();
    auto sOut = output.getShape();

    if (sX.size() != 3 || sW.size() != 2 || sOut.size() != 3)
        return false;
    if (sX[2] != sW[1])
        return false;
    if (sOut[0] != sX[0] || sOut[1] != sX[1] || sOut[2] != sW[0])
        return false;

    return true;
}

inline void runDotTransposedBF16(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                 const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *X = static_cast<const float *>(inputs[0]);
    const uint16_t *W = static_cast<const uint16_t *>(inputs[1]);
    float *Out = static_cast<float *>(outputs[0]);

    uint32_t B = inViews[0].getShape()[0];
    uint32_t S = inViews[0].getShape()[1];
    uint32_t InDim = inViews[0].getShape()[2];
    uint32_t OutDim = inViews[1].getShape()[0];

    uint32_t total_rows = B * S;
    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    uint32_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t start_row = t * rows_per_thread;
            uint32_t end_row = std::min(start_row + rows_per_thread, total_rows);

            for (uint32_t row_idx = start_row; row_idx < end_row; ++row_idx) {
                const float* x_row = X + row_idx * InDim;
                float* out_row = Out + row_idx * OutDim;

                for (uint32_t o = 0; o < OutDim; ++o) {
                    const uint16_t* w_row = W + o * InDim;
                    float sum = 0.0f;
                    
                    for (uint32_t i = 0; i < InDim; ++i) {
                        uint32_t bits = static_cast<uint32_t>(w_row[i]) << 16;
                        float w_f32;
                        std::memcpy(&w_f32, &bits, 4);
                        sum += x_row[i] * w_f32;
                    }
                    out_row[o] = sum;
                }
            } });
    }

    for (auto &thread : workers)
    {
        thread.join();
    }
}

inline uint32_t refFactoryDotTransposedBF16(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t w_cast = graph.cast(inputs[1], DType::FLOAT32);

    int32_t perm_dims[] = {1, 0};
    uint32_t dims_node = graph.constant({2}, perm_dims, DType::INT32);
    uint32_t w_t = graph.permute(w_cast, dims_node);

    w_t = graph.contiguous(w_t);

    auto w_shape = graph.getNode(inputs[1]).getShape();
    int32_t s3[] = {1, (int32_t)w_shape[1], (int32_t)w_shape[0]};
    uint32_t w_3d = graph.reshape(w_t, graph.constant({3}, s3, DType::INT32));

    return graph.dot(inputs[0], w_3d);
}

REGISTER_KERNEL("Dot_Transposed_BF16", 2, matchDotTransposedBF16, runDotTransposedBF16, refFactoryDotTransposedBF16, {Backend::CPU}, {DType::FLOAT32, DType::BF16}, {{1, 8, 640}, {256, 640}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});