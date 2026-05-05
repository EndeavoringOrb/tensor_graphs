// File: tensor_graphs_cpp/kernels/cpu/general/dot/partial_BF16_transposed_GEMM_NEON.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "core/graph.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <thread>
#include <vector>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchPartialBF16TransposedGEMM(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Signature: [cache, A, W, sC, eC, tC, sA, eA, tA, sW, eW, tW] (12 inputs)
    if (inputs.size() != 12)
        return false;

    // Check main tensor dtypes
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 ||
        inputs[2].dtype != DType::BF16 || output.dtype != DType::FLOAT32)
        return false;

    // Check coordinate tensor dtypes
    for (int i = 3; i < 12; ++i)
    {
        if (inputs[i].dtype != DType::INT32)
            return false;
    }

    const auto &shape_C = inputs[0].getShape();
    const auto &shape_A = inputs[1].getShape();
    const auto &shape_W = inputs[2].getShape();

    // Cache and A are 3D, W is 2D
    if (shape_C.size() != 3 || shape_A.size() != 3 || shape_W.size() != 2)
        return false;

    return true;
}

inline void runPartialBF16TransposedGEMM(const std::vector<const void *> &inputs,
                                         const std::vector<void *> &outputs,
                                         const std::vector<TensorView> &inViews,
                                         const std::vector<TensorView> &outViews)
{
    const float *target_ptr = static_cast<const float *>(inputs[0]);
    const float *A_ptr = static_cast<const float *>(inputs[1]);
    const uint16_t *W_ptr = static_cast<const uint16_t *>(inputs[2]);

    const int32_t *startsC_raw = static_cast<const int32_t *>(inputs[3]);
    const int32_t *endsC_raw = static_cast<const int32_t *>(inputs[4]);
    const int32_t *stepsC_raw = static_cast<const int32_t *>(inputs[5]);

    const int32_t *startsA_raw = static_cast<const int32_t *>(inputs[6]);
    const int32_t *endsA_raw = static_cast<const int32_t *>(inputs[7]);
    const int32_t *stepsA_raw = static_cast<const int32_t *>(inputs[8]);

    const int32_t *startsW_raw = static_cast<const int32_t *>(inputs[9]);
    const int32_t *endsW_raw = static_cast<const int32_t *>(inputs[10]);
    const int32_t *stepsW_raw = static_cast<const int32_t *>(inputs[11]);

    float *out_cache_ptr = static_cast<float *>(outputs[0]);

    const TensorView &view_cache = inViews[0];
    const TensorView &view_A = inViews[1];
    const TensorView &view_W = inViews[2];

    int32_t startsC[3], endsC[3], stepsC[3];
    int32_t startsA[3], endsA[3], stepsA[3];
    int32_t startsW[3], endsW[3], stepsW[3];

    for (int i = 0; i < 3; ++i)
    {
        startsC[i] = (inViews[3].getShape().empty() || i >= inViews[3].getShape()[0]) ? 0 : startsC_raw[i];
        endsC[i] = (inViews[4].getShape().empty() || i >= inViews[4].getShape()[0]) ? view_cache.getShape()[i] : endsC_raw[i];
        stepsC[i] = (inViews[5].getShape().empty() || i >= inViews[5].getShape()[0]) ? 1 : stepsC_raw[i];

        startsA[i] = (inViews[6].getShape().empty() || i >= inViews[6].getShape()[0]) ? 0 : startsA_raw[i];
        endsA[i] = (inViews[7].getShape().empty() || i >= inViews[7].getShape()[0]) ? view_A.getShape()[i] : endsA_raw[i];
        stepsA[i] = (inViews[8].getShape().empty() || i >= inViews[8].getShape()[0]) ? 1 : stepsA_raw[i];
    }

    for (int i = 0; i < 3; ++i)
    {
        startsW[i] = (inViews[9].getShape().empty() || i >= inViews[9].getShape()[0]) ? 0 : startsW_raw[i];
        endsW[i] = (inViews[10].getShape().empty() || i >= inViews[10].getShape()[0]) ? (i == 0 ? 1 : (i == 1 ? view_W.getShape()[1] : view_W.getShape()[0])) : endsW_raw[i];
        stepsW[i] = (inViews[11].getShape().empty() || i >= inViews[11].getShape()[0]) ? 1 : stepsW_raw[i];
    }

    auto get_dim = [](int32_t s, int32_t e, int32_t st, uint32_t dim_len) -> uint32_t
    {
        if (s < 0)
            s += dim_len;
        if (e < 0)
            e += dim_len;
        s = std::max(0, std::min(s, (int32_t)dim_len));
        e = std::max(0, std::min(e, (int32_t)dim_len));
        if (st <= 0 || s >= e)
            return 0;
        return (e - s + st - 1) / st;
    };

    uint32_t slice_B = get_dim(startsC[0], endsC[0], stepsC[0], view_cache.getShape()[0]);
    uint32_t slice_S = get_dim(startsC[1], endsC[1], stepsC[1], view_cache.getShape()[1]);
    uint32_t slice_N = get_dim(startsC[2], endsC[2], stepsC[2], view_cache.getShape()[2]);

    uint32_t slice_K = get_dim(startsA[2], endsA[2], stepsA[2], view_A.getShape()[2]);

    if (slice_B == 0 || slice_S == 0 || slice_N == 0 || slice_K == 0)
        return;

    int32_t startC_b = startsC[0] < 0 ? startsC[0] + view_cache.getShape()[0] : startsC[0];
    int32_t startC_s = startsC[1] < 0 ? startsC[1] + view_cache.getShape()[1] : startsC[1];
    int32_t startC_n = startsC[2] < 0 ? startsC[2] + view_cache.getShape()[2] : startsC[2];
    int32_t stepC_b = stepsC[0], stepC_s = stepsC[1], stepC_n = stepsC[2];

    int32_t startA_b = startsA[0] < 0 ? startsA[0] + view_A.getShape()[0] : startsA[0];
    int32_t startA_s = startsA[1] < 0 ? startsA[1] + view_A.getShape()[1] : startsA[1];
    int32_t startA_k = startsA[2] < 0 ? startsA[2] + view_A.getShape()[2] : startsA[2];
    int32_t stepA_b = stepsA[0], stepA_s = stepsA[1], stepA_k = stepsA[2];

    int32_t startW_n = startsW[2] < 0 ? startsW[2] + view_W.getShape()[0] : startsW[2];
    int32_t startW_k = startsW[1] < 0 ? startsW[1] + view_W.getShape()[1] : startsW[1];
    int32_t stepW_n = stepsW[2], stepW_k = stepsW[1];

    const int64_t strideA_B = view_A.strides[0];
    const int64_t strideA_S = view_A.strides[1];
    const int64_t strideA_K = view_A.strides[2];

    const int64_t strideW_N = view_W.strides[0];
    const int64_t strideW_K = view_W.strides[1];

    const int64_t strideC_B = view_cache.strides[0];
    const int64_t strideC_S = view_cache.strides[1];
    const int64_t strideC_N = view_cache.strides[2];

    bool can_simd = (strideC_N == 1 && stepC_n == 1 &&
                     strideW_K == 1 && stepW_k == 1 &&
                     strideA_K == 1 && stepA_k == 1);

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    num_threads = std::min(num_threads, slice_N / 4 + 1);

    std::vector<std::thread> workers;
    uint32_t n_block = (slice_N + num_threads - 1) / num_threads;
    n_block = (n_block + 3) & ~3; // Align to 4 for vectorized columns

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t n_start = t * n_block;
            if (n_start >= slice_N) return;
            uint32_t n_end = std::min(n_start + n_block, slice_N);
            uint32_t s_rem = slice_S & ~3;
            uint32_t n_rem = n_end & ~3;

            for (uint32_t b = 0; b < slice_B; ++b) {
                int32_t b_C = startC_b + b * stepC_b;
                int32_t b_A = startA_b + b * stepA_b;

                if (can_simd) {
                    // S loop in chunks of 4
                    for (uint32_t s = 0; s < s_rem; s += 4) {
                        int32_t s_C = startC_s + s * stepC_s;
                        int32_t s_A = startA_s + s * stepA_s;
                        
                        for (uint32_t n = n_start; n < n_rem; n += 4) {
                            int32_t n_C = startC_n + n; // step is 1
                            int32_t n_W = startW_n + n; // step is 1

                            float32x4_t acc00 = vdupq_n_f32(0), acc01 = vdupq_n_f32(0), acc02 = vdupq_n_f32(0), acc03 = vdupq_n_f32(0);
                            float32x4_t acc10 = vdupq_n_f32(0), acc11 = vdupq_n_f32(0), acc12 = vdupq_n_f32(0), acc13 = vdupq_n_f32(0);
                            float32x4_t acc20 = vdupq_n_f32(0), acc21 = vdupq_n_f32(0), acc22 = vdupq_n_f32(0), acc23 = vdupq_n_f32(0);
                            float32x4_t acc30 = vdupq_n_f32(0), acc31 = vdupq_n_f32(0), acc32 = vdupq_n_f32(0), acc33 = vdupq_n_f32(0);

                            for (uint32_t k = 0; k < (slice_K & ~3); k += 4) {
                                int32_t k_W = startW_k + k; 
                                int32_t k_A = startA_k + k; 

                                float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W_ptr + (n_W + 0) * strideW_N + k_W), 16));
                                float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W_ptr + (n_W + 1) * strideW_N + k_W), 16));
                                float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W_ptr + (n_W + 2) * strideW_N + k_W), 16));
                                float32x4_t w3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W_ptr + (n_W + 3) * strideW_N + k_W), 16));

                                float32x4_t x0 = vld1q_f32(A_ptr + b_A * strideA_B + (s_A + 0 * stepA_s) * strideA_S + k_A);
                                float32x4_t x1 = vld1q_f32(A_ptr + b_A * strideA_B + (s_A + 1 * stepA_s) * strideA_S + k_A);
                                float32x4_t x2 = vld1q_f32(A_ptr + b_A * strideA_B + (s_A + 2 * stepA_s) * strideA_S + k_A);
                                float32x4_t x3 = vld1q_f32(A_ptr + b_A * strideA_B + (s_A + 3 * stepA_s) * strideA_S + k_A);

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
                                float* out_ptr = out_cache_ptr + b_C * strideC_B + (s_C + row_s * stepC_s) * strideC_S + n_C;
                                float res[4] = {vaddvq_f32(a0), vaddvq_f32(a1), vaddvq_f32(a2), vaddvq_f32(a3)};
                                
                                for (uint32_t k = (slice_K & ~3); k < slice_K; ++k) {
                                    float xv = A_ptr[b_A * strideA_B + (s_A + row_s * stepA_s) * strideA_S + startA_k + k];
                                    for (int i = 0; i < 4; ++i) {
                                        uint32_t bits = (uint32_t)W_ptr[(n_W + i) * strideW_N + startW_k + k] << 16;
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

                    // S remainder block (Hits directly when seq_len=1 in partial decoding caches!)
                    for (uint32_t s = s_rem; s < slice_S; ++s) {
                        int32_t s_C = startC_s + s * stepC_s;
                        int32_t s_A = startA_s + s * stepA_s;

                        for (uint32_t n = n_start; n < n_rem; n += 4) {
                            int32_t n_C = startC_n + n;
                            int32_t n_W = startW_n + n;

                            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0), acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

                            for (uint32_t k = 0; k < (slice_K & ~3); k += 4) {
                                int32_t k_W = startW_k + k;
                                int32_t k_A = startA_k + k;

                                float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W_ptr + (n_W + 0) * strideW_N + k_W), 16));
                                float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W_ptr + (n_W + 1) * strideW_N + k_W), 16));
                                float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W_ptr + (n_W + 2) * strideW_N + k_W), 16));
                                float32x4_t w3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W_ptr + (n_W + 3) * strideW_N + k_W), 16));

                                float32x4_t x0 = vld1q_f32(A_ptr + b_A * strideA_B + s_A * strideA_S + k_A);

                                acc0 = vfmaq_f32(acc0, x0, w0);
                                acc1 = vfmaq_f32(acc1, x0, w1);
                                acc2 = vfmaq_f32(acc2, x0, w2);
                                acc3 = vfmaq_f32(acc3, x0, w3);
                            }

                            float* out_ptr = out_cache_ptr + b_C * strideC_B + s_C * strideC_S + n_C;
                            float res[4] = {vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)};

                            for (uint32_t k = (slice_K & ~3); k < slice_K; ++k) {
                                float xv = A_ptr[b_A * strideA_B + s_A * strideA_S + startA_k + k];
                                for (int i = 0; i < 4; ++i) {
                                    uint32_t bits = (uint32_t)W_ptr[(n_W + i) * strideW_N + startW_k + k] << 16;
                                    float wv; std::memcpy(&wv, &bits, 4);
                                    res[i] += xv * wv;
                                }
                            }
                            vst1q_f32(out_ptr, vld1q_f32(res));
                        }
                    }

                    // N remainder
                    for (uint32_t n = n_rem; n < n_end; ++n) {
                        int32_t n_C = startC_n + n;
                        int32_t n_W = startW_n + n;

                        for (uint32_t s = 0; s < slice_S; ++s) {
                            int32_t s_C = startC_s + s * stepC_s;
                            int32_t s_A = startA_s + s * stepA_s;

                            float sum = 0.0f;
                            for (uint32_t k = 0; k < slice_K; ++k) {
                                int32_t k_A = startA_k + k;
                                int32_t k_W = startW_k + k;

                                uint32_t bits = (uint32_t)W_ptr[n_W * strideW_N + k_W] << 16;
                                float wf; std::memcpy(&wf, &bits, 4);
                                sum += A_ptr[b_A * strideA_B + s_A * strideA_S + k_A] * wf;
                            }
                            out_cache_ptr[b_C * strideC_B + s_C * strideC_S + n_C] = sum;
                        }
                    }
                } else {
                    // Scalar fallback if strides aren't 1
                    for (uint32_t s = 0; s < slice_S; ++s) {
                        int32_t s_C = startC_s + s * stepC_s;
                        int32_t s_A = startA_s + s * stepA_s;

                        for (uint32_t n = n_start; n < n_end; ++n) {
                            int32_t n_C = startC_n + n * stepC_n;
                            int32_t n_W = startW_n + n * stepW_n;

                            float sum = 0.0f;
                            for (uint32_t k = 0; k < slice_K; ++k) {
                                int32_t k_A = startA_k + k * stepA_k;
                                int32_t k_W = startW_k + k * stepW_k;

                                float a_val = A_ptr[b_A * strideA_B + s_A * strideA_S + k_A * strideA_K];
                                uint16_t w_bf16 = W_ptr[n_W * strideW_N + k_W * strideW_K];
                                uint32_t bits = (uint32_t)w_bf16 << 16;
                                float w_val; std::memcpy(&w_val, &bits, 4);
                                sum += a_val * w_val;
                            }
                            out_cache_ptr[b_C * strideC_B + s_C * strideC_S + n_C * strideC_N] = sum;
                        }
                    }
                }
            } });
    }

    for (auto &th : workers)
        th.join();
}

inline uint32_t refFactoryPartialBF16TransposedGEMM(const std::vector<uint32_t> &inIds, Graph &graph)
{
    // inIds: [cache, A, W, sC, eC, tC, sA, eA, tA, sW, eW, tW]
    uint32_t sliceA = graph.slice(inIds[1], inIds[6], inIds[7], inIds[8]);
    uint32_t contigA = graph.contiguous(sliceA);

    uint32_t w_cast = graph.cast(inIds[2], DType::FLOAT32);
    int32_t perm[] = {1, 0};
    uint32_t w_t = graph.contiguous(graph.permute(w_cast, graph.constant({2}, perm, DType::INT32)));

    auto w_full_shape = graph.getNode(inIds[2]).getShape();
    int32_t N_full = w_full_shape.size() > 0 ? w_full_shape[0] : 1;
    int32_t K_full = w_full_shape.size() > 1 ? w_full_shape[1] : 1;

    int32_t s3[] = {1, K_full, N_full};
    uint32_t w_3d = graph.reshape(w_t, graph.constant({3}, s3, DType::INT32));

    uint32_t sliceW = graph.slice(w_3d, inIds[9], inIds[10], inIds[11]);
    uint32_t contigW = graph.contiguous(sliceW);

    uint32_t dot = graph.dot(contigA, contigW);
    uint32_t contigDot = graph.contiguous(dot);

    return graph.scatter(inIds[0], contigDot, inIds[3], inIds[4], inIds[5]);
}

REGISTER_KERNEL_INPLACE(
    "Scatter_BF16_Transposed_GEMM_NEON",
    12,
    matchPartialBF16TransposedGEMM,
    runPartialBF16TransposedGEMM,
    refFactoryPartialBF16TransposedGEMM,
    {Backend::CPU},
    {DType::FLOAT32, DType::FLOAT32, DType::BF16,
     DType::INT32, DType::INT32, DType::INT32,
     DType::INT32, DType::INT32, DType::INT32,
     DType::INT32, DType::INT32, DType::INT32},
    {{1, 8, 2048}, {1, 8, 2048}, {2048, 2048}, {3}, {3}, {3}, {3}, {3}, {3}, {3}, {3}, {3}},
    {false, false, false, false, false, false, false, false, false, false, false, false},
    {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});

#endif