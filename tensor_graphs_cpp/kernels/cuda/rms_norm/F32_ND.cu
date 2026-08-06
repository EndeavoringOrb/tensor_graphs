#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// CUDA kernel: fused RMSNorm
//   x : [dim0, seq_len, dim_size]   (contiguous)
//   w : [dim_size]                  (contiguous)
//   out: same shape as x
// ---------------------------------------------------------------------------
__global__ void rmsnorm_f32_nd_kernel(const float* __restrict__ x,
                                      const float* __restrict__ weight,
                                      float* __restrict__ out,
                                      uint32_t dim0,
                                      uint32_t seq_len,
                                      uint32_t dim_size,
                                      float eps)
{
    uint32_t row = blockIdx.x;          // one block per row (dim0 * seq_len)
    uint32_t tid = threadIdx.x;
    uint32_t total_rows = dim0 * seq_len;
    if (row >= total_rows) return;

    const float* x_row = x + (uint64_t)row * dim_size;
    float* out_row = out + (uint64_t)row * dim_size;

    // ---------------------------------------------------------------------
    // 1. Sum of squares across the last dimension (dim_size)
    // ---------------------------------------------------------------------
    float sum = 0.0f;
    for (uint32_t i = tid; i < dim_size; i += blockDim.x) {
        float v = x_row[i];
        sum += v * v;
    }

    // Shared memory for reduction (size = blockDim.x)
    extern __shared__ float sh[];
    sh[tid] = sum;
    __syncthreads();

    // Block reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh[tid] += sh[tid + s];
        }
        __syncthreads();
    }

    // ---------------------------------------------------------------------
    // 2. Compute inv_std = 1 / sqrt(mean_sq + eps)
    // ---------------------------------------------------------------------
    float inv_std;
    if (tid == 0) {
        float mean_sq = sh[0] / (float)dim_size;
        inv_std = 1.0f / sqrtf(mean_sq + eps);
        sh[0] = inv_std;   // store inv_std in shared memory for broadcast
    }
    __syncthreads();
    inv_std = sh[0];

    // ---------------------------------------------------------------------
    // 3. Apply: out = x * inv_std * weight
    // ---------------------------------------------------------------------
    for (uint32_t i = tid; i < dim_size; i += blockDim.x) {
        out_row[i] = x_row[i] * inv_std * weight[i];
    }
}

// ---------------------------------------------------------------------------
// Match function: verifies the RMSNorm pattern
//   inputs[0] = x (3D), inputs[1] = weight (1D)
//   output has same shape as x
// ---------------------------------------------------------------------------
inline bool matchRMSNorm_F32_CUDA_ND(const std::vector<TensorNode>& inputs,
                                     const TensorNode& output)
{
    // Output must be FLOAT32 and contiguous
    if (output.dtype != DType::FLOAT32) return false;
    if (!isContiguous(output)) return false;

    // Input shapes: x must be 3D, weight must be 1D
    const auto& sX = inputs[0].getShape();
    const auto& sW = inputs[1].getShape();
    if (sX.size() != 3) return false;
    if (sW.size() != 1) return false;
    if (sX[2] != sW[0]) return false;          // last dim of x equals weight size
    if (sX != output.getShape()) return false; // output shape matches x

    return true;
}

// ---------------------------------------------------------------------------
// Run function: launches the CUDA kernel
// ---------------------------------------------------------------------------
inline void runRMSNorm_F32_CUDA_ND(const KernelContext& ctx)
{
    const float* x      = static_cast<const float*>(ctx.inputs[0]);
    const float* weight = static_cast<const float*>(ctx.inputs[1]);
    float* out          = static_cast<float*>(ctx.outputs[0]);

    const auto& shape = ctx.inViews[0].getShape();
    uint32_t dim0     = shape[0];
    uint32_t seq_len  = shape[1];
    uint32_t dim_size = shape[2];

    // eps is read from the constant node, but in this kernel we use a fixed value.
    // The decomposition uses eps_fp32 = g.constant({1}, &eps, DType::FLOAT32)
    // We can either pass eps as a kernel argument or hardcode it.
    // Here we hardcode 1e-6 as per DeepSeekV4FlashConfig::norm_eps.
    float eps = 1e-6f;

    // Determine block size: use at most 256 threads, but cap at dim_size.
    uint32_t blockSize = std::min<uint32_t>(dim_size, 256);
    // Round up to a multiple of warp size (32) for better occupancy.
    blockSize = ((blockSize + 31) / 32) * 32;
    if (blockSize == 0) blockSize = 32;

    uint32_t total_rows = dim0 * seq_len;
    uint32_t gridSize   = total_rows;   // one block per row

    // Shared memory size: blockSize floats
    size_t shmem = blockSize * sizeof(float);

    rmsnorm_f32_nd_kernel<<<gridSize, blockSize, shmem>>>(
        x, weight, out, dim0, seq_len, dim_size, eps
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in RMSNorm_F32_CUDA_ND: " +
                         std::string(cudaGetErrorString(err)));
    }
}

// ---------------------------------------------------------------------------
// Reference factory: reproduces the exact graph decomposition of
// DeepSeekV4FlashModel::rms_norm().
//   inputs[0] = x (3D)
//   inputs[1] = weight (1D)
// Returns the LogicalId of the RMSNorm output.
// ---------------------------------------------------------------------------
inline LogicalId refFactoryRMSNorm_F32_CUDA_ND(const std::vector<LogicalId>& inputs,
                                               Graph& graph)
{
    LogicalId x_id = inputs[0];
    LogicalId w_id = inputs[1];

    const auto& xShape = graph.getNode(x_id).getShape();
    uint32_t dim0     = xShape[0];
    uint32_t seq_len  = xShape[1];
    uint32_t dim_size = xShape[2];

    // Helper: expand a scalar constant to a 3D tensor with shape [dim0, seq_len, 1]
    auto expand_scalar_to_3d_1 = [&](float val) -> LogicalId {
        LogicalId node = graph.constant({1}, &val, DType::FLOAT32);
        int32_t sh3[] = {1, 1, 1};
        LogicalId out = graph.reshape(node, graph.constant({3}, sh3, DType::INT32));
        // repeat along axis 0 and 1 if needed
        if (dim0 > 1) {
            int32_t rep = (int32_t)dim0;
            int32_t ax = 0;
            out = graph.repeat(out, graph.constant({1}, &rep, DType::INT32),
                               graph.constant({1}, &ax, DType::INT32));
        }
        if (seq_len > 1) {
            int32_t rep = (int32_t)seq_len;
            int32_t ax = 1;
            out = graph.repeat(out, graph.constant({1}, &rep, DType::INT32),
                               graph.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    // 1. x_sq = x * x
    LogicalId x_sq = graph.mul(x_id, x_id);

    // 2. sum_sq = sum(x_sq, axis=-1)  => shape [dim0, seq_len, 1]
    int32_t axis_val = -1;
    LogicalId axis_node = graph.constant({1}, &axis_val, DType::INT32);
    LogicalId sum_sq = graph.sum(x_sq, axis_node);

    // 3. mean_sq = sum_sq / dim_size
    //    expand scalar dim_size to [dim0, seq_len, 1]
    float dim_size_f = (float)dim_size;
    LogicalId n_node = expand_scalar_to_3d_1(dim_size_f);
    LogicalId mean_sq = graph.div(sum_sq, n_node);

    // 4. var = mean_sq + eps
    float eps = 1e-6f;   // matches norm_eps in DeepSeekV4FlashConfig
    LogicalId eps_node = expand_scalar_to_3d_1(eps);
    LogicalId var = graph.add(mean_sq, eps_node);

    // 5. std = sqrt(var) = pow(var, 0.5)
    float half = 0.5f;
    LogicalId half_node = expand_scalar_to_3d_1(half);
    LogicalId std = graph.pow(var, half_node);

    // 6. inv_std = 1 / std   (shape [dim0, seq_len, 1])
    float one = 1.0f;
    LogicalId one_node = expand_scalar_to_3d_1(one);
    LogicalId inv_std = graph.div(one_node, std);

    // 7. inv_std_expanded = repeat(inv_std, dim_size, axis=2)
    //    => shape [dim0, seq_len, dim_size]
    int32_t rep_dim = (int32_t)dim_size;
    int32_t ax2 = 2;
    LogicalId inv_std_expanded = graph.repeat(inv_std,
                                              graph.constant({1}, &rep_dim, DType::INT32),
                                              graph.constant({1}, &ax2, DType::INT32));

    // 8. x_norm = x * inv_std_expanded
    LogicalId x_norm = graph.mul(x_id, inv_std_expanded);

    // 9. weight_expanded: reshape weight to [1, 1, dim_size], repeat to [dim0, seq_len, dim_size]
    int32_t w_shape[] = {1, 1, (int32_t)dim_size};
    LogicalId w_reshaped = graph.reshape(w_id, graph.constant({3}, w_shape, DType::INT32));
    LogicalId w_exp = w_reshaped;
    if (dim0 > 1) {
        int32_t rep0 = (int32_t)dim0;
        int32_t ax0 = 0;
        w_exp = graph.repeat(w_exp, graph.constant({1}, &rep0, DType::INT32),
                             graph.constant({1}, &ax0, DType::INT32));
    }
    if (seq_len > 1) {
        int32_t rep1 = (int32_t)seq_len;
        int32_t ax1 = 1;
        w_exp = graph.repeat(w_exp, graph.constant({1}, &rep1, DType::INT32),
                             graph.constant({1}, &ax1, DType::INT32));
    }

    // 10. out = x_norm * w_exp
    return graph.mul(x_norm, w_exp);
}

// ---------------------------------------------------------------------------
// Kernel registration
// ---------------------------------------------------------------------------
REGISTER_KERNEL(
    "RMSNorm_F32_ND_CUDA",
    2,                                             // min inputs
    2,                                             // max inputs
    matchRMSNorm_F32_CUDA_ND,
    runRMSNorm_F32_CUDA_ND,
    refFactoryRMSNorm_F32_CUDA_ND,
    {0},
    MemSpace(2, HandleType::CUDA),                 // output memory space
    {Engine(0, EngineType::CUDA_GPU)},             // engines
    {DType::FLOAT32, DType::FLOAT32},              // input dtypes
    {{1, 8, 2048}, {2048}},                        // dummy shapes
    {true, true},                                  // requires contiguous inputs
    {{MemSpace(2, HandleType::CUDA)},              // input mem spaces (CUDA)
     {MemSpace(2, HandleType::CUDA)}}              // weight also on CUDA
);

#endif // TG_USE_CUDA