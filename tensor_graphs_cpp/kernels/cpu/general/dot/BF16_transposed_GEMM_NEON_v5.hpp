// File: tensor_graphs_cpp/kernels/cpu/general/dot/BF16_transposed_GEMM_NEON_v5.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>

inline bool matchBF16TransposedGEMM_v5(const std::vector<TensorNode> &inputs, const TensorNode &output)
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
    return isContiguous(output);
}

inline void runBF16TransposedGEMM_v5(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                     const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *X = static_cast<const float *>(inputs[0]);
    const uint16_t *W = static_cast<const uint16_t *>(inputs[1]);
    float *Out = static_cast<float *>(outputs[0]);

    const uint32_t B = inViews[0].getShape()[0];
    const uint32_t S = inViews[0].getShape()[1];
    const uint32_t K = inViews[0].getShape()[2];
    const uint32_t N = inViews[1].getShape()[0];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    num_threads = std::min(num_threads, N / 4 + 1);

    std::vector<std::thread> workers;
    uint32_t n_block = (N + num_threads - 1) / num_threads;
    n_block = (n_block + 3) & ~3;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t n_start = t * n_block;
            if (n_start >= N) return;
            uint32_t n_end = std::min(n_start + n_block, N);
            uint32_t s_rem = S & ~3;
            uint32_t n_rem = n_end & ~3;

            for (uint32_t b = 0; b < B; ++b) {
                // 4x4 blocks -> S is outermost to keep X in cache!
                for (uint32_t s = 0; s < s_rem; s += 4) {
                    const float* x_ptr = X + b * S * K + s * K;
                    for (uint32_t n = n_start; n < n_rem; n += 4) {
                        float32x4_t acc00 = vdupq_n_f32(0), acc01 = vdupq_n_f32(0), acc02 = vdupq_n_f32(0), acc03 = vdupq_n_f32(0);
                        float32x4_t acc10 = vdupq_n_f32(0), acc11 = vdupq_n_f32(0), acc12 = vdupq_n_f32(0), acc13 = vdupq_n_f32(0);
                        float32x4_t acc20 = vdupq_n_f32(0), acc21 = vdupq_n_f32(0), acc22 = vdupq_n_f32(0), acc23 = vdupq_n_f32(0);
                        float32x4_t acc30 = vdupq_n_f32(0), acc31 = vdupq_n_f32(0), acc32 = vdupq_n_f32(0), acc33 = vdupq_n_f32(0);

                        const uint16_t* w_ptr = W + n * K;

                        for (uint32_t k = 0; k < (K & ~3); k += 4) {
                            float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w_ptr + 0 * K + k), 16));
                            float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w_ptr + 1 * K + k), 16));
                            float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w_ptr + 2 * K + k), 16));
                            float32x4_t w3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w_ptr + 3 * K + k), 16));

                            float32x4_t x0 = vld1q_f32(x_ptr + 0 * K + k);
                            float32x4_t x1 = vld1q_f32(x_ptr + 1 * K + k);
                            float32x4_t x2 = vld1q_f32(x_ptr + 2 * K + k);
                            float32x4_t x3 = vld1q_f32(x_ptr + 3 * K + k);

                            acc00 = vfmaq_f32(acc00, x0, w0); acc01 = vfmaq_f32(acc01, x0, w1);
                            acc02 = vfmaq_f32(acc02, x0, w2); acc03 = vfmaq_f32(acc03, x0, w3);
                            acc10 = vfmaq_f32(acc10, x1, w0); acc11 = vfmaq_f32(acc11, x1, w1);
                            acc12 = vfmaq_f32(acc12, x1, w2); acc13 = vfmaq_f32(acc13, x1, w3);
                            acc20 = vfmaq_f32(acc20, x2, w0); acc21 = vfmaq_f32(acc21, x2, w1);
                            acc22 = vfmaq_f32(acc22, x2, w2); acc23 = vfmaq_f32(acc23, x2, w3);
                            acc30 = vfmaq_f32(acc30, x3, w0); acc31 = vfmaq_f32(acc31, x3, w1);
                            acc32 = vfmaq_f32(acc32, x3, w2); acc33 = vfmaq_f32(acc33, x3, w3);
                        }

                        auto store_4 = [&](uint32_t row_s, float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3) {
                            float* out_ptr = Out + b * S * N + (s + row_s) * N + n;
                            float res[4] = {vaddvq_f32(a0), vaddvq_f32(a1), vaddvq_f32(a2), vaddvq_f32(a3)};
                            for (uint32_t k = (K & ~3); k < K; ++k) {
                                float xv = x_ptr[row_s * K + k];
                                for (int i = 0; i < 4; ++i) {
                                    uint32_t bits = (uint32_t)W[(n + i) * K + k] << 16;
                                    float wv; std::memcpy(&wv, &bits, 4);
                                    res[i] += xv * wv;
                                }
                            }
                            vst1q_f32(out_ptr, vld1q_f32(res));
                        };

                        store_4(0, acc00, acc01, acc02, acc03);
                        store_4(1, acc10, acc11, acc12, acc13);
                        store_4(2, acc20, acc21, acc22, acc23);
                        store_4(3, acc30, acc31, acc32, acc33);
                    }
                }
                
                for (uint32_t s = s_rem; s < S; ++s) {
                    const float* x_ptr = X + b * S * K + s * K;
                    for (uint32_t n = n_start; n < n_rem; n += 4) {
                        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0), acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
                        const uint16_t* w_ptr = W + n * K;

                        for (uint32_t k = 0; k < (K & ~3); k += 4) {
                            float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w_ptr + 0 * K + k), 16));
                            float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w_ptr + 1 * K + k), 16));
                            float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w_ptr + 2 * K + k), 16));
                            float32x4_t w3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w_ptr + 3 * K + k), 16));

                            float32x4_t x0 = vld1q_f32(x_ptr + k);

                            acc0 = vfmaq_f32(acc0, x0, w0);
                            acc1 = vfmaq_f32(acc1, x0, w1);
                            acc2 = vfmaq_f32(acc2, x0, w2);
                            acc3 = vfmaq_f32(acc3, x0, w3);
                        }

                        float* out_ptr = Out + b * S * N + s * N + n;
                        float res[4] = {vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)};
                        
                        for (uint32_t k = (K & ~3); k < K; ++k) {
                            float xv = x_ptr[k];
                            for (int i = 0; i < 4; ++i) {
                                uint32_t bits = (uint32_t)W[(n + i) * K + k] << 16;
                                float wv; std::memcpy(&wv, &bits, 4);
                                res[i] += xv * wv;
                            }
                        }
                        vst1q_f32(out_ptr, vld1q_f32(res));
                    }
                }

                for (uint32_t n = n_rem; n < n_end; ++n) {
                    for (uint32_t s = 0; s < S; ++s) {
                        float sum = 0.0f;
                        const uint16_t* w_ptr = W + n * K;
                        for (uint32_t k = 0; k < K; ++k) {
                            uint32_t bits = (uint32_t)w_ptr[k] << 16;
                            float wf; std::memcpy(&wf, &bits, 4);
                            sum += X[b * S * K + s * K + k] * wf;
                        }
                        Out[b * S * N + s * N + n] = sum;
                    }
                }
            } });
    }
    for (auto &worker : workers)
        worker.join();
}

inline uint32_t refFactoryBF16TransposedGEMM_v5(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t w_cast = graph.cast(inputs[1], DType::FLOAT32);
    int32_t perm[] = {1, 0};
    uint32_t w_t = graph.contiguous(graph.permute(w_cast, graph.constant({2}, perm, DType::INT32)));
    auto w_shape = graph.getNode(inputs[1]).getShape();
    int32_t s3[] = {1, (int32_t)w_shape[1], (int32_t)w_shape[0]};
    return graph.dot(inputs[0], graph.reshape(w_t, graph.constant({3}, s3, DType::INT32)));
}

REGISTER_KERNEL("BF16_Transposed_GEMM_NEON_v5", 2, matchBF16TransposedGEMM_v5, runBF16TransposedGEMM_v5, refFactoryBF16TransposedGEMM_v5, {Backend::CPU}, {DType::FLOAT32, DType::BF16}, {{1, 8, 64}, {1024, 64}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});
#endif