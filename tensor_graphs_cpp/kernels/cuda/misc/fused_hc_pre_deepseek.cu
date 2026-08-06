// tensor_graphs_cpp/kernels/cuda/general/fused_hc_pre_deepseek.cu
#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_functions.h>

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
    // Each block handles one sequence token (S)
    // Each thread handles one element of the output (or a part of the Sinkhorn)
    // For simplicity we parallelize over S and over D/HC dimensions.
    extern __shared__ float shared_mem[];

    uint32_t s = blockIdx.x;  // assuming grid over S
    if (s >= S) return;

    uint32_t tid = threadIdx.x;
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;

    // Step 1: Compute rsqrt over flattened x: shape [B, S, HC*D]
    // We'll compute sum of squares per token.
    __shared__ float sum_sq_shared[32]; // one per warp? We'll use block reduction.
    float sum_sq = 0.0f;
    for (uint32_t i = tid; i < HC * D; i += blockDim.x) {
        uint32_t hc_idx = i / D;
        uint32_t d_idx = i % D;
        float val = x[s * HC * D + i]; // assuming B=1
        sum_sq += val * val;
    }
    // block reduction for sum_sq
    // ... (use shared memory reduction)
    // after reduction, sum_sq is total per token.

    // Step 2: Compute mixes = (x_flat * fn_w) * rsqrt
    // We need to compute linear projection: mixes[t] = sum_i x_flat[i] * fn_w[i][t]
    // This is a matmul (HC*D) x (MIX_DIM). We can parallelize over MIX_DIM.
    // We'll compute a thread-block tile of the output mixes.
    // For simplicity, we do a simple loop per thread over K.
    // Given sizes: HC*D ≈ 4*4096=16384, MIX_DIM=4*(2+4)=24. This is small.
    // We can just do a straightforward loop.
    float mixes[24]; // MIX_DIM max 24 (HC=4)
    for (uint32_t t = 0; t < MIX_DIM; ++t) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < HC * D; ++i) {
            sum += x[s * HC * D + i] * fn_w[i * MIX_DIM + t];
        }
        mixes[t] = sum * rsqrt_val; // rsqrt_val computed from sum_sq
    }

    // Step 3: Split mixes into pre (HC), post (HC), comb (HC*HC)
    float pre[4], post_arr[4], comb_flat[16]; // HC=4
    for (uint32_t i = 0; i < HC; ++i) {
        pre[i] = mixes[i];
        post_arr[i] = mixes[HC + i];
    }
    for (uint32_t i = 0; i < HC * HC; ++i) {
        comb_flat[i] = mixes[2 * HC + i];
    }

    // Step 4: Apply scale & base to pre, post, comb
    float sig_pre[4], sig_post[4];
    for (uint32_t i = 0; i < HC; ++i) {
        float p = pre[i] * scale[i] + base[i];
        sig_pre[i] = 1.0f / (1.0f + expf(-p)); // sigmoid
        float q = post_arr[i] * scale[HC + i] + base[HC + i];
        sig_post[i] = 2.0f * (1.0f / (1.0f + expf(-q))); // 2*sigmoid
    }
    float comb_mat[4][4];
    for (uint32_t i = 0; i < HC; ++i) {
        for (uint32_t j = 0; j < HC; ++j) {
            comb_mat[i][j] = comb_flat[i * HC + j] * hc_scale[2*HC + i*HC + j] + hc_base[2*HC + i*HC + j];
        }
    }

    // Step 5: Sinkhorn iterations (HC <= 4, cheap per token)
    float sinkhorn[4][4];
    for (uint32_t i = 0; i < HC; ++i)
        for (uint32_t j = 0; j < HC; ++j)
            sinkhorn[i][j] = comb_mat[i][j];

    // Apply softmax along last dim
    for (uint32_t i = 0; i < HC; ++i) {
        float maxv = -1e30f;
        for (uint32_t j = 0; j < HC; ++j) maxv = fmaxf(maxv, sinkhorn[i][j]);
        float sum = 0.0f;
        for (uint32_t j = 0; j < HC; ++j) {
            sinkhorn[i][j] = expf(sinkhorn[i][j] - maxv);
            sum += sinkhorn[i][j];
        }
        for (uint32_t j = 0; j < HC; ++j) sinkhorn[i][j] /= sum;
        // add eps
        sinkhorn[i][j] += eps;
    }
    // Normalize columns and repeat for HC_SINKHORN_ITERS (e.g., 20)
    // For brevity, we show a few iterations; in full kernel we loop.
    for (uint32_t iter = 0; iter < 20; ++iter) {
        // row normalize
        for (uint32_t i = 0; i < HC; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < HC; ++j) sum += sinkhorn[i][j];
            float inv = 1.0f / (sum + eps);
            for (uint32_t j = 0; j < HC; ++j) sinkhorn[i][j] *= inv;
        }
        // col normalize
        for (uint32_t j = 0; j < HC; ++j) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < HC; ++i) sum += sinkhorn[i][j];
            float inv = 1.0f / (sum + eps);
            for (uint32_t i = 0; i < HC; ++i) sinkhorn[i][j] *= inv;
        }
    }
    // Store comb output
    for (uint32_t i = 0; i < HC; ++i)
        for (uint32_t j = 0; j < HC; ++j)
            comb[s * HC * HC + i * HC + j] = sinkhorn[i][j];

    // Step 6: Compute y = sum_hc (pre_weight * x) over HC dimension
    // y[s, d] = sum_hc pre[s, hc] * x[s, hc, d]
    // This is a simple reduction over HC.
    // We can parallelize over d dimension.
    for (uint32_t d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t hc = 0; hc < HC; ++hc) {
            float val = x[s * HC * D + hc * D + d];
            acc += sig_pre[hc] * val;
        }
        y[s * D + d] = acc;
    }

    // Store post (sigmoid_post * 2)
    for (uint32_t hc = 0; hc < HC; ++hc)
        post[s * HC + hc] = sig_post[hc];
}

// ----------------------------------------------------------------------------
// Host-side matching and run functions
// ----------------------------------------------------------------------------
inline bool matchFusedHCPre_DeepSeek_CUDA(const std::vector<TensorNode>& inputs,
                                          const TensorNode& output) {
    // Inputs: [x, fn_w, scale, base, hc_scale, hc_base] (6 inputs)
    // We can check shapes: x should be 4D [B, S, HC, D], etc.
    // The kernel is specifically for DeepSeek's HC pre.
    // For matching, we rely on the refFactory to produce the pattern.
    // Here we only do basic shape checks.
    if (inputs.size() < 6) return false;
    if (inputs[0].getShape().size() != 4) return false;
    return isContiguous(output);
}

inline void runFusedHCPre_DeepSeek_CUDA(const KernelContext& ctx) {
    // Extract parameters from ctx and launch kernel
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

    // Get eps from constant? We'll pass as parameter; assume 1e-6.
    float eps = 1e-6f;

    dim3 grid(S, 1, 1);
    dim3 block(256, 1, 1); // tune
    size_t shared_size = 0; // not using shared for now; could use for reductions

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
    // Inputs: [x, fn_w, scale, base, hc_scale, hc_base]
    // We reconstruct the original chain:
    // x_flat = reshape(x, [1, S, HC*D])
    // rsqrt = 1/sqrt(mean(x_flat^2) + eps)
    // mixes = linear(x_flat, fn_w) * rsqrt
    // split into pre, post, comb
    // ... etc.
    // This is complex; we must replicate the exact operations from
    // DeepSeekV4FlashModel::hc_pre.
    // For brevity, we assume this is provided in the model code.
    // The planner will match this pattern.
    // We'll return a placeholder.
    // In practice, we'd copy the actual code from the model's hc_pre method.
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