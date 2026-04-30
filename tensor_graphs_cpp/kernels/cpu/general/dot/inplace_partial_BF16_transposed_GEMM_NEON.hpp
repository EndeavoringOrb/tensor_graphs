// File: tensor_graphs_cpp/kernels/cpu/general/dot/inplace_partial_BF16_transposed_GEMM_NEON.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "core/common/partial.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>

inline bool matchPartialBF16TransposedGEMM(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // inputs: target, A, W, starts, ends, steps
    if (inputs.size() != 6)
        return false;

    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || inputs[2].dtype != DType::BF16)
        return false;

    if (inputs[3].dtype != DType::INT32 || inputs[4].dtype != DType::INT32 || inputs[5].dtype != DType::INT32)
        return false;

    auto sTarget = inputs[0].getShape();
    auto sA = inputs[1].getShape();
    auto sW = inputs[2].getShape();

    if (sTarget.size() != 3 || sA.size() != 3 || sW.size() != 2)
        return false;

    // Target is [B, S, N], A is [B, S, K], W is [N, K]
    if (sTarget[0] != sA[0])
        return false;
    if (sTarget[1] != sA[1])
        return false;
    if (sA[2] != sW[1])
        return false; // K matches
    if (sTarget[2] != sW[0])
        return false; // N matches

    if (inputs[0].storageType == StorageType::PERSISTENT)
        return false;

    if (!isContiguous(inputs[1]))
        return false;
    if (!isContiguous(inputs[2]))
        return false;

    return true;
}

inline void runPartialBF16TransposedGEMM(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *target = static_cast<const float *>(inputs[0]);
    const float *X = static_cast<const float *>(inputs[1]);
    const uint16_t *W = static_cast<const uint16_t *>(inputs[2]);
    const int32_t *starts = static_cast<const int32_t *>(inputs[3]);
    const int32_t *ends = static_cast<const int32_t *>(inputs[4]);
    const int32_t *steps = static_cast<const int32_t *>(inputs[5]);
    float *out = static_cast<float *>(outputs[0]);

    const auto &out_shape = outViews[0].getShape(); // target shape [B, S, N]
    const auto &out_strides = outViews[0].strides;

    // Step 1: Copy target to output if different buffers (inplace copy-on-write)
    partial_ops::copy_target_if_needed(target, out, out_shape, inViews[0].strides, out_strides);

    // Compute slice shape and adjusted starts/steps for the output region
    uint32_t starts_size = inViews[3].getShape().empty() ? 0 : inViews[3].getShape()[0];
    uint32_t ends_size = inViews[4].getShape().empty() ? 0 : inViews[4].getShape()[0];
    uint32_t steps_size = inViews[5].getShape().empty() ? 0 : inViews[5].getShape()[0];

    std::vector<uint32_t> slice_shape(out_shape.size());
    std::vector<int32_t> adj_starts(out_shape.size());
    std::vector<int32_t> adj_steps(out_shape.size());

    for (size_t d = 0; d < out_shape.size(); ++d)
    {
        int32_t st = (d < starts_size) ? starts[d] : 0;
        int32_t en = (d < ends_size) ? ends[d] : (int32_t)out_shape[d];
        int32_t sp = (d < steps_size) ? steps[d] : 1;
        if (st < 0)
            st += out_shape[d];
        if (en < 0)
            en += out_shape[d];
        adj_starts[d] = st;
        adj_steps[d] = sp;
        slice_shape[d] = static_cast<uint32_t>(std::max(0, (en - st + sp - 1) / sp));
    }

    uint32_t slice_B = slice_shape[0];
    uint32_t slice_S = slice_shape[1];
    uint32_t slice_N = slice_shape[2];

    if (slice_B == 0 || slice_S == 0 || slice_N == 0)
        return;

    const uint32_t S_full = inViews[1].getShape()[1];
    const uint32_t K = inViews[1].getShape()[2];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    num_threads = std::min(num_threads, slice_N / 4 + 1);

    std::vector<std::thread> workers;
    uint32_t n_block = (slice_N + num_threads - 1) / num_threads;
    n_block = (n_block + 3) & ~3;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t sn_start = t * n_block;
            if (sn_start >= slice_N) return;
            uint32_t sn_end = std::min(sn_start + n_block, slice_N);
            uint32_t ss_rem = slice_S & ~3;
            uint32_t sn_rem = sn_end & ~3;

            for (uint32_t sb = 0; sb < slice_B; ++sb) {
                uint32_t b = adj_starts[0] + sb * adj_steps[0];
                
                // 4x4 blocks -> S is outermost to keep X in cache!
                for (uint32_t ss = 0; ss < ss_rem; ss += 4) {
                    uint32_t s0 = adj_starts[1] + (ss + 0) * adj_steps[1];
                    uint32_t s1 = adj_starts[1] + (ss + 1) * adj_steps[1];
                    uint32_t s2 = adj_starts[1] + (ss + 2) * adj_steps[1];
                    uint32_t s3 = adj_starts[1] + (ss + 3) * adj_steps[1];
                    
                    const float* x_ptr0 = X + b * S_full * K + s0 * K;
                    const float* x_ptr1 = X + b * S_full * K + s1 * K;
                    const float* x_ptr2 = X + b * S_full * K + s2 * K;
                    const float* x_ptr3 = X + b * S_full * K + s3 * K;

                    for (uint32_t sn = sn_start; sn < sn_rem; sn += 4) {
                        uint32_t n0 = adj_starts[2] + (sn + 0) * adj_steps[2];
                        uint32_t n1 = adj_starts[2] + (sn + 1) * adj_steps[2];
                        uint32_t n2 = adj_starts[2] + (sn + 2) * adj_steps[2];
                        uint32_t n3 = adj_starts[2] + (sn + 3) * adj_steps[2];

                        float32x4_t acc00 = vdupq_n_f32(0), acc01 = vdupq_n_f32(0), acc02 = vdupq_n_f32(0), acc03 = vdupq_n_f32(0);
                        float32x4_t acc10 = vdupq_n_f32(0), acc11 = vdupq_n_f32(0), acc12 = vdupq_n_f32(0), acc13 = vdupq_n_f32(0);
                        float32x4_t acc20 = vdupq_n_f32(0), acc21 = vdupq_n_f32(0), acc22 = vdupq_n_f32(0), acc23 = vdupq_n_f32(0);
                        float32x4_t acc30 = vdupq_n_f32(0), acc31 = vdupq_n_f32(0), acc32 = vdupq_n_f32(0), acc33 = vdupq_n_f32(0);

                        for (uint32_t k = 0; k < (K & ~3); k += 4) {
                            float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W + n0 * K + k), 16));
                            float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W + n1 * K + k), 16));
                            float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W + n2 * K + k), 16));
                            float32x4_t w3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W + n3 * K + k), 16));

                            float32x4_t vx0 = vld1q_f32(x_ptr0 + k);
                            float32x4_t vx1 = vld1q_f32(x_ptr1 + k);
                            float32x4_t vx2 = vld1q_f32(x_ptr2 + k);
                            float32x4_t vx3 = vld1q_f32(x_ptr3 + k);

                            acc00 = vfmaq_f32(acc00, vx0, w0); acc01 = vfmaq_f32(acc01, vx0, w1);
                            acc02 = vfmaq_f32(acc02, vx0, w2); acc03 = vfmaq_f32(acc03, vx0, w3);
                            acc10 = vfmaq_f32(acc10, vx1, w0); acc11 = vfmaq_f32(acc11, vx1, w1);
                            acc12 = vfmaq_f32(acc12, vx1, w2); acc13 = vfmaq_f32(acc13, vx1, w3);
                            acc20 = vfmaq_f32(acc20, vx2, w0); acc21 = vfmaq_f32(acc21, vx2, w1);
                            acc22 = vfmaq_f32(acc22, vx2, w2); acc23 = vfmaq_f32(acc23, vx2, w3);
                            acc30 = vfmaq_f32(acc30, vx3, w0); acc31 = vfmaq_f32(acc31, vx3, w1);
                            acc32 = vfmaq_f32(acc32, vx3, w2); acc33 = vfmaq_f32(acc33, vx3, w3);
                        }

                        auto store_4 = [&](uint32_t s_idx, const float* x_ptr, float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3) {
                            float res[4] = {vaddvq_f32(a0), vaddvq_f32(a1), vaddvq_f32(a2), vaddvq_f32(a3)};
                            uint32_t n_idxs[4] = {n0, n1, n2, n3};
                            for (uint32_t k = (K & ~3); k < K; ++k) {
                                float xv = x_ptr[k];
                                for (int i = 0; i < 4; ++i) {
                                    uint32_t bits = (uint32_t)W[n_idxs[i] * K + k] << 16;
                                    float wv; std::memcpy(&wv, &bits, 4);
                                    res[i] += xv * wv;
                                }
                            }
                            
                            uint64_t base_idx = b * out_strides[0] + s_idx * out_strides[1];
                            out[base_idx + n0 * out_strides[2]] = res[0];
                            out[base_idx + n1 * out_strides[2]] = res[1];
                            out[base_idx + n2 * out_strides[2]] = res[2];
                            out[base_idx + n3 * out_strides[2]] = res[3];
                        };

                        store_4(s0, x_ptr0, acc00, acc01, acc02, acc03);
                        store_4(s1, x_ptr1, acc10, acc11, acc12, acc13);
                        store_4(s2, x_ptr2, acc20, acc21, acc22, acc23);
                        store_4(s3, x_ptr3, acc30, acc31, acc32, acc33);
                    }
                }
                
                // ss_rem to slice_S
                for (uint32_t ss = ss_rem; ss < slice_S; ++ss) {
                    uint32_t s = adj_starts[1] + ss * adj_steps[1];
                    const float* x_ptr = X + b * S_full * K + s * K;
                    
                    for (uint32_t sn = sn_start; sn < sn_rem; sn += 4) {
                        uint32_t n0 = adj_starts[2] + (sn + 0) * adj_steps[2];
                        uint32_t n1 = adj_starts[2] + (sn + 1) * adj_steps[2];
                        uint32_t n2 = adj_starts[2] + (sn + 2) * adj_steps[2];
                        uint32_t n3 = adj_starts[2] + (sn + 3) * adj_steps[2];

                        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0), acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

                        for (uint32_t k = 0; k < (K & ~3); k += 4) {
                            float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W + n0 * K + k), 16));
                            float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W + n1 * K + k), 16));
                            float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W + n2 * K + k), 16));
                            float32x4_t w3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W + n3 * K + k), 16));

                            float32x4_t vx = vld1q_f32(x_ptr + k);

                            acc0 = vfmaq_f32(acc0, vx, w0);
                            acc1 = vfmaq_f32(acc1, vx, w1);
                            acc2 = vfmaq_f32(acc2, vx, w2);
                            acc3 = vfmaq_f32(acc3, vx, w3);
                        }

                        float res[4] = {vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)};
                        uint32_t n_idxs[4] = {n0, n1, n2, n3};
                        for (uint32_t k = (K & ~3); k < K; ++k) {
                            float xv = x_ptr[k];
                            for (int i = 0; i < 4; ++i) {
                                uint32_t bits = (uint32_t)W[n_idxs[i] * K + k] << 16;
                                float wv; std::memcpy(&wv, &bits, 4);
                                res[i] += xv * wv;
                            }
                        }
                        
                        uint64_t base_idx = b * out_strides[0] + s * out_strides[1];
                        out[base_idx + n0 * out_strides[2]] = res[0];
                        out[base_idx + n1 * out_strides[2]] = res[1];
                        out[base_idx + n2 * out_strides[2]] = res[2];
                        out[base_idx + n3 * out_strides[2]] = res[3];
                    }
                }

                // sn_rem to sn_end
                for (uint32_t sn = sn_rem; sn < sn_end; ++sn) {
                    uint32_t n = adj_starts[2] + sn * adj_steps[2];
                    for (uint32_t ss = 0; ss < slice_S; ++ss) {
                        uint32_t s = adj_starts[1] + ss * adj_steps[1];
                        float sum = 0.0f;
                        const uint16_t* w_ptr = W + n * K;
                        const float* x_ptr = X + b * S_full * K + s * K;
                        for (uint32_t k = 0; k < K; ++k) {
                            uint32_t bits = (uint32_t)w_ptr[k] << 16;
                            float wf; std::memcpy(&wf, &bits, 4);
                            sum += x_ptr[k] * wf;
                        }
                        out[b * out_strides[0] + s * out_strides[1] + n * out_strides[2]] = sum;
                    }
                }
            } });
    }
    for (auto &worker : workers)
        worker.join();
}

inline uint32_t refFactoryPartialBF16TransposedGEMM(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t target = inputs[0];
    uint32_t A = inputs[1];
    uint32_t W = inputs[2];
    uint32_t st = inputs[3];
    uint32_t en = inputs[4];
    uint32_t step = inputs[5];

    uint32_t w_cast = graph.cast(W, DType::FLOAT32);
    int32_t perm[] = {1, 0};
    uint32_t w_t = graph.contiguous(graph.permute(w_cast, graph.constant({2}, perm, DType::INT32)));
    auto w_shape = graph.getNode(W).getShape();
    int32_t s3[] = {1, (int32_t)w_shape[1], (int32_t)w_shape[0]};
    uint32_t B = graph.reshape(w_t, graph.constant({3}, s3, DType::INT32));

    uint32_t op_res = graph.dot(graph.contiguous(graph.slice(A, st, en, step)),
                                graph.contiguous(graph.slice(B, st, en, step)));

    return graph.scatter(target, graph.contiguous(op_res), st, en, step);
}

REGISTER_KERNEL_INPLACE("Partial_BF16_Transposed_GEMM_NEON", 6, matchPartialBF16TransposedGEMM, runPartialBF16TransposedGEMM, refFactoryPartialBF16TransposedGEMM, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32, DType::BF16, DType::INT32, DType::INT32, DType::INT32}, {{1, 8, 1024}, {1, 8, 64}, {1024, 64}, {3}, {3}, {3}}, {false, false, false, false, false, false}, {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});

#endif