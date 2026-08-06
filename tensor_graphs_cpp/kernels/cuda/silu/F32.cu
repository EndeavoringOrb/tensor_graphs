// tensor_graphs_cpp/kernels/cuda/fused/silu.cu
#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// CUDA kernel: numerically stable SiLU (Swish)
//   silu(x) = x / (1 + exp(-x))    for x >= 0
//   silu(x) = x * exp(x) / (1 + exp(x))   for x < 0  (avoids overflow)
// ---------------------------------------------------------------------------
__global__ void fused_silu_f32_nd_kernel(const float* __restrict__ x,
                                         float* __restrict__ out,
                                         uint64_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = x[idx];
    float result;
    if (val >= 0.0f) {
        // Stable for x >= 0: exp(-x) in (0,1]
        result = val / (1.0f + expf(-val));
    } else {
        // For x < 0: use x * exp(x) / (1 + exp(x))
        float exp_x = expf(val);
        result = val * exp_x / (1.0f + exp_x);
    }
    out[idx] = result;
}

// ---------------------------------------------------------------------------
// Match function – validates shapes and contiguity.
// The kernel accepts any rank, but requires the output to be contiguous.
// Input contiguity is enforced by the registration macro.
// ---------------------------------------------------------------------------
inline bool matchFusedSilu_CUDA(const std::vector<TensorNode> &inputs,
                                const TensorNode &output) {
    if (inputs.size() != 1) return false;
    if (inputs[0].getShape() != output.getShape()) return false;
    if (output.dtype != DType::FLOAT32) return false;
    if (!isContiguous(output)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Run function – launches the CUDA kernel.
// ---------------------------------------------------------------------------
inline void runFusedSilu_CUDA(const KernelContext &ctx) {
    const float* x = static_cast<const float*>(ctx.inputs[0]);
    float* out = static_cast<float*>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    fused_silu_f32_nd_kernel<<<numBlocks, blockSize>>>(x, out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in FusedSilu_CUDA: " +
                         std::string(cudaGetErrorString(err)));
    }
}

// ---------------------------------------------------------------------------
// Reference Factory – exactly replicates the graph built by
// DeepSeekV4FlashModel::silu.
//
// Given input tensor x, it constructs:
//   neg_one = fill(-1, shape)
//   neg_x   = mul(x, neg_one)
//   e_node  = fill(e, shape)
//   exp_neg = pow(e_node, neg_x)
//   one     = fill(1, shape)
//   den     = add(one, exp_neg)
//   sig     = div(one, den)
//   result  = mul(x, sig)
// ---------------------------------------------------------------------------
inline LogicalId refFactoryFusedSilu_CUDA(const std::vector<LogicalId> &inputs,
                                          Graph &g) {
    if (inputs.size() != 1) {
        Error::throw_err("FusedSilu_CUDA requires exactly 1 input.");
    }

    LogicalId x_id = inputs[0];
    const TensorNode &x_node = g.getNode(x_id);
    const std::vector<uint32_t> &shape = x_node.getShape();
    DType dtype = x_node.dtype;  // should be FLOAT32

    // Helper to create a fill node: constant scalar -> fill with shape
    auto fill_scalar = [&](float val) -> LogicalId {
        LogicalId scalar = g.constant({1}, &val, dtype);
        // Build a shape tensor constant (INT32) from the actual shape
        std::vector<int32_t> shape_int(shape.begin(), shape.end());
        LogicalId shape_node = g.constant({(uint32_t)shape.size()},
                                          shape_int.data(), DType::INT32);
        return g.fill(scalar, shape_node);
    };

    // Build the exact decomposition
    LogicalId neg_one = fill_scalar(-1.0f);
    LogicalId neg_x = g.mul(x_id, neg_one);

    float e_val = 2.718281828459045f;
    LogicalId e_node = fill_scalar(e_val);
    LogicalId exp_neg = g.pow(e_node, neg_x);

    LogicalId one_node = fill_scalar(1.0f);
    LogicalId den = g.add(one_node, exp_neg);
    LogicalId sig = g.div(one_node, den);

    return g.mul(x_id, sig);
}

// ---------------------------------------------------------------------------
// Registration – CUDA backend, F32 only, contiguous input/output.
// ---------------------------------------------------------------------------
REGISTER_KERNEL("FusedSilu_CUDA",
                1,                         // min_num_inputs
                1,                         // max_num_inputs
                matchFusedSilu_CUDA,
                runFusedSilu_CUDA,
                refFactoryFusedSilu_CUDA,
                MemSpace(2, HandleType::CUDA),         // output memory space
                {Engine(0, EngineType::CUDA_GPU)},     // engines
                {DType::FLOAT32},                      // input dtypes
                {{1}},                                 // dummy shape (1D, will be replaced)
                {true},                                // input must be contiguous
                {MemSpace(2, HandleType::CUDA)}        // input memory space
);

#endif // TG_USE_CUDA