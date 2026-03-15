#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchDotF32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2) return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32) return false;
    const auto &s0 = inputs[0].shape;
    const auto &s1 = inputs[1].shape;
    const auto &so = output.shape;
    if (s0.size() != 3 || s1.size() != 3 || so.size() != 3) return false;
    if (s0[0] != s1[0] || s0[2] != s1[1]) return false;
    if (so[0] != s0[0] || so[1] != s0[1] || so[2] != s1[2]) return false;
    return true;
}

inline void runDotF32_3D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *A = static_cast<const float *>(inputs[0]);
    const float *B = static_cast<const float *>(inputs[1]);
    float *Out = static_cast<float *>(outputs[0]);

    uint32_t B_count = inViews[0].shape[0];
    uint32_t M = inViews[0].shape[1];
    uint32_t K = inViews[0].shape[2];
    uint32_t N = inViews[1].shape[2];

    for (uint32_t b = 0; b < B_count; ++b) {
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < K; ++k) {
                    uint64_t a_flat = (uint64_t)b * M * K + m * K + k;
                    uint64_t b_flat = (uint64_t)b * K * N + k * N + n;
                    sum += A[getStridedIndex(a_flat, inViews[0].shape, inViews[0].strides)] * 
                           B[getStridedIndex(b_flat, inViews[1].shape, inViews[1].strides)];
                }
                uint64_t out_flat = (uint64_t)b * M * N + m * N + n;
                Out[getStridedIndex(out_flat, outViews[0].shape, outViews[0].strides)] = sum;
            }
        }
    }
}

REGISTER_REF_KERNEL(OpType::DOT, Backend::CPU, matchDotF32_3D, runDotF32_3D);