#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ----------------------------------------------------------------------------
// CUDA kernel: Fused HC Pre‑processing
// Inputs:
//   x       : [B, S, HC, D]          (B=1, HC=4 typically)
//   fn_w    : [HC*D, MIX_DIM]         (MIX_DIM = HC*(2+HC))
//   scale   : [HC]                    (for sigmoid scaling)
//   base    : [HC]                    (for sigmoid base)
//   hc_scale: [HC*HC]                 (for Sinkhorn)
//   hc_base : [HC*HC]
//   eps     : scalar
// Outputs:
//   y       : [B, S, D]               (pre‑applied and reduced)
//   post    : [B, S, HC]              (post‑activations)
//   comb    : [B, S, HC, HC]          (Sinkhorn matrix)
// ----------------------------------------------------------------------------
__global__ void fused_hc_pre_kernel(
    const float* __restrict__ x,          // [B, S, HC, D]
    const float* __restrict__ fn_w,       // [HC*D, MIX_DIM]
    const float* __restrict__ scale,      // [HC]
    const float* __restrict__ base,       // [HC]
    const float* __restrict__ hc_scale,   // [HC*HC]
    const float* __restrict__ hc_base,    // [HC*HC]
    float eps,
    float* __restrict__ y,                // [B, S, D]
    float* __restrict__ post,             // [B, S, HC]
    float* __restrict__ comb,             // [B, S, HC, HC]
    uint32_t B, uint32_t S, uint32_t HC, uint32_t D, uint32_t MIX_DIM)
{
    uint32_t s = blockIdx.x;  // assuming grid over S
    if (s >= S) return;

    uint32_t tid = threadIdx.x;
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;

    // Step 1: Compute rsqrt over flattened x: shape [B, S, HC*D]
    float thread_sum_sq = 0.0f;
    for (uint32_t i = tid; i < HC * D; i += blockDim.x) {
        float val = x[s * HC * D + i];
        thread_sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum_sq += __shfl_down_sync(0xffffffff, thread_sum_sq, offset);
    }

    __shared__ float warp_sums[32];
    if (lane == 0) {
        warp_sums[warp_id] = thread_sum_sq;
    }
    __syncthreads();

    __shared__ float rsqrt_val;
    if (tid == 0) {
        float total_sum_sq = 0.0f;
        uint32_t num_warps = (blockDim.x + 31) / 32;
        for (uint32_t w = 0; w < num_warps; ++w) {
            total_sum_sq += warp_sums[w];
        }
        float mean_sq = total_sum_sq / static_cast<float>(HC * D);
        rsqrt_val = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    // Step 2: Compute mixes = (x_flat * fn_w) * rsqrt
    float mixes[24]; // MIX_DIM max 24 (HC=4)
    for (uint32_t t = 0; t < MIX_DIM && t < 24; ++t) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < HC * D; ++i) {
            sum += x[s * HC * D + i] * fn_w[i * MIX_DIM + t];
        }
        mixes[t] = sum * rsqrt_val;
    }

    // Step 3: Split mixes into pre (HC), post (HC), comb (HC*HC)
    float pre[4], post_arr[4], comb_flat[16]; // HC=4
    for (uint32_t i = 0; i < HC && i < 4; ++i) {
        pre[i] = mixes[i];
        post_arr[i] = mixes[HC + i];
    }
    for (uint32_t i = 0; i < HC * HC && i < 16; ++i) {
        comb_flat[i] = mixes[2 * HC + i];
    }

    // Step 4: Apply scale & base to pre, post, comb
    float sig_pre[4], sig_post[4];
    for (uint32_t i = 0; i < HC && i < 4; ++i) {
        float p = pre[i] * scale[i] + base[i];
        sig_pre[i] = (1.0f / (1.0f + expf(-p))) + eps; // sigmoid + eps
        float q = post_arr[i] * scale[HC + i] + base[HC + i];
        sig_post[i] = 2.0f * (1.0f / (1.0f + expf(-q))); // 2*sigmoid
    }
    float comb_mat[4][4];
    for (uint32_t i = 0; i < HC && i < 4; ++i) {
        for (uint32_t j = 0; j < HC && j < 4; ++j) {
            comb_mat[i][j] = comb_flat[i * HC + j] * hc_scale[i * HC + j] + hc_base[i * HC + j];
        }
    }

    // Step 5: Sinkhorn iterations (HC <= 4, cheap per token)
    float sinkhorn[4][4];
    for (uint32_t i = 0; i < HC && i < 4; ++i) {
        for (uint32_t j = 0; j < HC && j < 4; ++j) {
            sinkhorn[i][j] = comb_mat[i][j];
        }
    }

    // Apply softmax along last dim
    for (uint32_t i = 0; i < HC && i < 4; ++i) {
        float maxv = -1e30f;
        for (uint32_t j = 0; j < HC && j < 4; ++j) {
            maxv = fmaxf(maxv, sinkhorn[i][j]);
        }
        float sum = 0.0f;
        for (uint32_t j = 0; j < HC && j < 4; ++j) {
            sinkhorn[i][j] = expf(sinkhorn[i][j] - maxv);
            sum += sinkhorn[i][j];
        }
        for (uint32_t j = 0; j < HC && j < 4; ++j) {
            sinkhorn[i][j] = (sinkhorn[i][j] / sum) + eps;
        }
    }

    // Normalize columns and repeat for HC_SINKHORN_ITERS (e.g., 20)
    for (uint32_t iter = 0; iter < 20; ++iter) {
        // row normalize
        for (uint32_t i = 0; i < HC && i < 4; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < HC && j < 4; ++j) {
                sum += sinkhorn[i][j];
            }
            float inv = 1.0f / (sum + eps);
            for (uint32_t j = 0; j < HC && j < 4; ++j) {
                sinkhorn[i][j] *= inv;
            }
        }
        // col normalize
        for (uint32_t j = 0; j < HC && j < 4; ++j) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < HC && i < 4; ++i) {
                sum += sinkhorn[i][j];
            }
            float inv = 1.0f / (sum + eps);
            for (uint32_t i = 0; i < HC && i < 4; ++i) {
                sinkhorn[i][j] *= inv;
            }
        }
    }

    // Store comb and post outputs from thread 0
    if (tid == 0) {
        for (uint32_t i = 0; i < HC && i < 4; ++i) {
            for (uint32_t j = 0; j < HC && j < 4; ++j) {
                comb[s * HC * HC + i * HC + j] = sinkhorn[i][j];
            }
        }
        for (uint32_t hc = 0; hc < HC && hc < 4; ++hc) {
            post[s * HC + hc] = sig_post[hc];
        }
    }

    // Step 6: Compute y = sum_hc (pre_weight * x) over HC dimension
    for (uint32_t d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t hc = 0; hc < HC && hc < 4; ++hc) {
            float val = x[s * HC * D + hc * D + d];
            acc += sig_pre[hc] * val;
        }
        y[s * D + d] = acc;
    }
}

// ----------------------------------------------------------------------------
// Host-side matching and run functions
// ----------------------------------------------------------------------------
inline bool matchFusedHCPre_DeepSeek_CUDA(const std::vector<TensorNode>& inputs,
                                          const TensorNode& output) {
    if (inputs[0].getShape().size() != 4) return false;
    return isContiguous(output);
}

inline void runFusedHCPre_DeepSeek_CUDA(const KernelContext& ctx) {
    const float* x = static_cast<const float*>(ctx.inputs[0]);
    const float* fn_w = static_cast<const float*>(ctx.inputs[1]);
    const float* scale = static_cast<const float*>(ctx.inputs[2]);
    const float* base = static_cast<const float*>(ctx.inputs[3]);
    const float* hc_scale = static_cast<const float*>(ctx.inputs[4]);
    const float* hc_base = static_cast<const float*>(ctx.inputs[5]);
    float* y = static_cast<float*>(ctx.outputs[0]);
    float* post = static_cast<float*>(ctx.outputs[1]);
    float* comb = static_cast<float*>(ctx.outputs[2]);

    const auto& shape_x = ctx.inViews[0].getShape();
    uint32_t B = shape_x[0];
    uint32_t S = shape_x[1];
    uint32_t HC = shape_x[2];
    uint32_t D = shape_x[3];
    uint32_t MIX_DIM = HC * (2 + HC);

    float eps = 1e-6f;

    dim3 grid(S, 1, 1);
    dim3 block(256, 1, 1);
    size_t shared_size = 0;

    fused_hc_pre_kernel<<<grid, block, shared_size>>>(
        x, fn_w, scale, base, hc_scale, hc_base, eps,
        y, post, comb, B, S, HC, D, MIX_DIM);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in FusedHCPre_CUDA: " +
                         std::string(cudaGetErrorString(err)));
    }
}

// ----------------------------------------------------------------------------
// Reference Factory: builds the exact graph pattern for HC pre
// ----------------------------------------------------------------------------
inline LogicalId refFactoryFusedHCPre_DeepSeek(const std::vector<LogicalId>& inputs,
                                               Graph& graph) {
    Error::throw_err("refFactoryFusedHCPre_DeepSeek not yet implemented");
    return LogicalId{};
}

REGISTER_KERNEL("Fused_HC_Pre_DeepSeek_CUDA", 6, 6,
                matchFusedHCPre_DeepSeek_CUDA,
                runFusedHCPre_DeepSeek_CUDA,
                refFactoryFusedHCPre_DeepSeek,
                MemSpace(2, HandleType::CUDA),
                {Engine(0, EngineType::CUDA_GPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::FLOAT32,
                 DType::FLOAT32, DType::FLOAT32},
                {{1, 8, 4, 4096}, {16384, 24}, {4}, {4}, {16}, {16}},
                {true, true, true, true, true, true},
                {{MemSpace(2, HandleType::CUDA)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)}});
#endif // TG_USE_CUDA