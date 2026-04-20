#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>
#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchBF16TransposedGEMM_v3(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::BF16)
        return false;
    auto sX = inputs[0].getShape();
    auto sW = inputs[1].getShape();
    auto sO = output.getShape();
    if (sX.size() != 3 || sW.size() != 2 || sO.size() != 3)
        return false;
    if (sX[2] != sW[1] || sO[2] != sW[0])
        return false;
    if (sO[0] != sX[0] || sO[1] != sX[1])
        return false;
    return isContiguous(output);
}

inline void runBF16TransposedGEMM_v3(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                     const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *X = static_cast<const float *>(inputs[0]);
    const uint16_t *W = static_cast<const uint16_t *>(inputs[1]);
    float *Out = static_cast<float *>(outputs[0]);
    uint32_t B = inViews[0].getShape()[0];
    uint32_t S = inViews[0].getShape()[1];
    uint32_t K = inViews[0].getShape()[2];
    uint32_t N = inViews[1].getShape()[0];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    if (num_threads > N)
        num_threads = N;

    std::vector<std::thread> workers;
    uint32_t n_per_thread = (N + num_threads - 1) / num_threads;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t start_n = t * n_per_thread;
            uint32_t end_n = std::min(start_n + n_per_thread, N);
            
            for (uint32_t b = 0; b < B; ++b) {
                for (uint32_t s_start = 0; s_start < S; s_start += 8) {
                    uint32_t s_end = std::min(s_start + 8, S);
                    
                    for (uint32_t n = start_n; n < end_n; ++n) {
                        const uint16_t* w_row = W + n * K;
                        float32x4_t acc[8];
                        for (uint32_t s = s_start; s < s_end; ++s) {
                            acc[s - s_start] = vdupq_n_f32(0.0f);
                        }
                        
                        uint32_t k = 0;
                        for (; k + 4 <= K; k += 4) {
                            uint16x4_t vbf16 = vld1_u16(w_row + k);
                            float32x4_t vw = vreinterpretq_f32_u32(vshll_n_u16(vbf16, 16));
                            for (uint32_t s = s_start; s < s_end; ++s) {
                                float32x4_t vx = vld1q_f32(X + b * S * K + s * K + k);
                                acc[s - s_start] = vfmaq_f32(acc[s - s_start], vx, vw);
                            }
                        }
                        
                        float sum[8];
                        for (uint32_t s = s_start; s < s_end; ++s) {
                            sum[s - s_start] = vaddvq_f32(acc[s - s_start]);
                        }
                        
                        for (; k < K; ++k) {
                            uint32_t bits = (uint32_t)w_row[k] << 16;
                            float wf; std::memcpy(&wf, &bits, 4);
                            for (uint32_t s = s_start; s < s_end; ++s) {
                                sum[s - s_start] += X[b * S * K + s * K + k] * wf;
                            }
                        }
                        
                        for (uint32_t s = s_start; s < s_end; ++s) {
                            Out[b * S * N + s * N + n] = sum[s - s_start];
                        }
                    }
                }
            } });
    }
    for (auto &worker : workers)
        worker.join();
}

inline uint32_t refFactoryBF16TransposedGEMM_v3(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t w_cast = graph.cast(inputs[1], DType::FLOAT32);
    int32_t perm[] = {1, 0};
    uint32_t w_t = graph.contiguous(graph.permute(w_cast, graph.constant({2}, perm, DType::INT32)));
    auto w_shape = graph.getNode(inputs[1]).getShape();
    int32_t s3[] = {1, (int32_t)w_shape[1], (int32_t)w_shape[0]};
    return graph.dot(inputs[0], graph.reshape(w_t, graph.constant({3}, s3, DType::INT32)));
}

REGISTER_KERNEL("BF16_Transposed_GEMM_NEON_v3", 2, matchBF16TransposedGEMM_v3, runBF16TransposedGEMM_v3, refFactoryBF16TransposedGEMM_v3, {Backend::CPU}, {DType::FLOAT32, DType::BF16}, {{1, 8, 640}, {262144, 640}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});
#endif