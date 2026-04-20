// File: tensor_graphs_cpp/kernels/cpu/general/dot/partial_bf16.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <thread>
#include <algorithm>
#include <cstring>

inline bool matchPartialDotBF16_3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 6)
        return false;
    // 0: cache, 1: A_partial, 2: B_partial (BF16)
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || inputs[2].dtype != DType::BF16)
        return false;

    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 3 || inputs[2].getShape().size() != 2)
        return false;

    // Verify K matches
    if (inputs[1].getShape()[2] != inputs[2].getShape()[1])
        return false;

    // Verify N matches cache
    if (inputs[0].getShape()[2] != inputs[2].getShape()[0])
        return false;

    if (inputs[0].storageType == StorageType::PERSISTENT)
        return false;

    if (inputs[0].getShape() != output.getShape())
        return false;

    return true;
}

inline void runPartialDotBF16_3D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                 const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *A = static_cast<const float *>(inputs[1]);
    const uint16_t *W = static_cast<const uint16_t *>(inputs[2]);
    const int32_t *starts = static_cast<const int32_t *>(inputs[3]);
    const int32_t *steps = static_cast<const int32_t *>(inputs[5]);
    float *out = static_cast<float *>(outputs[0]); // Aliases cache

    uint32_t Bp = inViews[1].getShape()[0];
    uint32_t Mp = inViews[1].getShape()[1];
    uint32_t K = inViews[1].getShape()[2];
    uint32_t N_dim = inViews[2].getShape()[0]; // W is [N, K]

    int64_t b_start = starts[0];
    int64_t m_start = starts[1];
    int64_t n_start = starts[2];
    int64_t b_step = steps[0];
    int64_t m_step = steps[1];
    int64_t n_step = steps[2];

    int64_t cB_stride = outViews[0].strides[0];
    int64_t cM_stride = outViews[0].strides[1];
    int64_t cN_stride = outViews[0].strides[2];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    if (num_threads > N_dim)
        num_threads = N_dim;

    std::vector<std::thread> workers;
    uint32_t n_per_thread = (N_dim + num_threads - 1) / num_threads;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t start_n = t * n_per_thread;
            uint32_t end_n = std::min(start_n + n_per_thread, N_dim);
            
            for (uint32_t bp = 0; bp < Bp; ++bp) {
                for (uint32_t mp = 0; mp < Mp; ++mp) {
                    const float *a_row = A + (bp * Mp + mp) * K;
                    float *c_row = out + (b_start + bp * b_step) * cB_stride + 
                                   (m_start + mp * m_step) * cM_stride + n_start * cN_stride;

                    for (uint32_t n = start_n; n < end_n; ++n) {
                        const uint16_t* w_row = W + n * K;
                        float sum = 0.0f;
                        
                        uint32_t k = 0;
#if defined(TG_HAS_NEON)
                        float32x4_t acc = vdupq_n_f32(0.0f);
                        for (; k + 4 <= K; k += 4) {
                            uint16x4_t vbf16 = vld1_u16(w_row + k);
                            float32x4_t vw = vreinterpretq_f32_u32(vshll_n_u16(vbf16, 16));
                            float32x4_t vx = vld1q_f32(a_row + k);
                            acc = vfmaq_f32(acc, vx, vw);
                        }
                        sum = vaddvq_f32(acc);
#endif
                        for (; k < K; ++k) {
                            uint32_t bits = (uint32_t)w_row[k] << 16;
                            float wf; std::memcpy(&wf, &bits, 4);
                            sum += a_row[k] * wf;
                        }
                        
                        c_row[n * n_step] = sum;
                    }
                }
            } });
    }
    for (auto &worker : workers)
        worker.join();
}

inline uint32_t refFactoryPartialDotBF16(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t cache = inputs[0];
    uint32_t A_partial = inputs[1];
    uint32_t B_bf16 = inputs[2];
    uint32_t starts = inputs[3];
    uint32_t ends = inputs[4];
    uint32_t steps = inputs[5];

    uint32_t w_cast = graph.cast(B_bf16, DType::FLOAT32);
    int32_t perm[] = {1, 0};
    uint32_t w_t = graph.contiguous(graph.permute(w_cast, graph.constant({2}, perm, DType::INT32)));
    auto w_shape = graph.getNode(B_bf16).getShape();
    int32_t s3[] = {1, (int32_t)w_shape[1], (int32_t)w_shape[0]};
    uint32_t w_3d = graph.reshape(w_t, graph.constant({3}, s3, DType::INT32));

    uint32_t dot_res = graph.dot(A_partial, w_3d);
    return graph.scatter(cache, dot_res, starts, ends, steps);
}

REGISTER_KERNEL_INPLACE("PartialDot_BF16_3D", 6, matchPartialDotBF16_3D, runPartialDotBF16_3D, refFactoryPartialDotBF16,
                        {Backend::CPU},
                        {DType::FLOAT32, DType::FLOAT32, DType::BF16, DType::INT32, DType::INT32, DType::INT32},
                        {{1, 8, 262144}, {1, 8, 640}, {262144, 640}, {3}, {3}, {3}},
                        {false, true, true, false, false, false},
                        {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});

#endif