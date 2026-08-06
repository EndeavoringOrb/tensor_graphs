#ifdef TG_USE_CUDA
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "core/shapes.hpp"

namespace MXFP4CUDA {

// E2M1 FP4 lookup table (16 values stored in fast CUDA constant memory)
// Bit 3 = Sign bit (0: positive, 1: negative)
// Bits [2:0] = Magnitude
__device__ __constant__ float FP4_E2M1_TABLE[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

/**
 * Fused MXFP4 Weight Dequantization Kernel
 * Each thread processes 1 byte (2 FP4 elements).
 * Since block scale size is 32, every consecutive pair of elements (even col, odd col)
 * is guaranteed to belong to the exact same scale block.
 */
__global__ void fused_mxfp4_dequant_kernel(
    const uint8_t* __restrict__ packed_weights, // [out_d, in_d / 2]
    const float* __restrict__ scales,            // [out_d, in_d / 32]
    float* __restrict__ out,                     // [out_d, in_d]
    uint32_t out_d,
    uint32_t in_d)
{
    // Global byte index across all packed weight data
    uint64_t byte_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total_bytes = ((uint64_t)out_d * in_d) >> 1; // total_elements / 2

    if (byte_idx >= total_bytes) return;

    // Output element starting index (2 elements per thread)
    uint64_t elem_idx = byte_idx << 1; 

    // Compute 2D coordinates
    uint32_t row = (uint32_t)(elem_idx / in_d);
    uint32_t col = (uint32_t)(elem_idx % in_d);

    // Read 1 packed byte containing 2 FP4 numbers
    uint8_t byte_val = packed_weights[byte_idx];
    uint8_t low_fp4  = byte_val & 0x0F;        // Element at `col`
    uint8_t high_fp4 = (byte_val >> 4) & 0x0F; // Element at `col + 1`

    // Lookup FP4 float representations
    float w0 = FP4_E2M1_TABLE[low_fp4];
    float w1 = FP4_E2M1_TABLE[high_fp4];

    // Compute scale index: 1 scale per 32 contiguous elements along `in_d`
    uint32_t scale_w = in_d >> 5; // in_d / 32
    uint32_t scale_col = col >> 5; // col / 32
    uint64_t scale_idx = (uint64_t)row * scale_w + scale_col;

    float scale_val = scales[scale_idx];

    // Write final dequantized FP32 values
    out[elem_idx]     = w0 * scale_val;
    out[elem_idx + 1] = w1 * scale_val;
}

} // namespace MXFP4CUDA

// ---------------------------------------------------------------------------
// Match Function for Framework Integration
// ---------------------------------------------------------------------------
inline bool matchFusedMXFP4_CUDA(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::FLOAT32) return false;

    // inputs[0]: Packed weights [out_d, in_d / 2]
    // inputs[1]: Scales [out_d, in_d / 32]
    const auto &sW = inputs[0].getShape();
    const auto &sS = inputs[1].getShape();
    const auto &sO = output.getShape();

    if (sW.size() != 2 || sS.size() != 2 || sO.size() != 2) return false;

    uint32_t out_d = sO[0];
    uint32_t in_d  = sO[1];

    if (sW[0] != out_d || sW[1] != (in_d / 2)) return false;
    if (sS[0] != out_d || sS[1] != (in_d / 32)) return false;

    if (!isContiguous(output)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Kernel Runner
// ---------------------------------------------------------------------------
inline void runFusedMXFP4_CUDA(const KernelContext &ctx) {
    const uint8_t *packed_weights = static_cast<const uint8_t *>(ctx.inputs[0]);
    const float *scales           = static_cast<const float *>(ctx.inputs[1]);
    float *out                    = static_cast<float *>(ctx.outputs[0]);

    const auto &out_shape = ctx.outViews[0].getShape();
    uint32_t out_d = out_shape[0];
    uint32_t in_d  = out_shape[1];

    uint64_t total_elements = (uint64_t)out_d * in_d;
    uint64_t total_bytes    = total_elements >> 1;

    if (total_bytes == 0) return;

    int blockSize = 256;
    int numBlocks = (int)((total_bytes + blockSize - 1) / blockSize);

    MXFP4CUDA::fused_mxfp4_dequant_kernel<<<numBlocks, blockSize>>>(
        packed_weights, scales, out, out_d, in_d
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in FusedMXFP4_CUDA: " + std::string(cudaGetErrorString(err)));
    }
}

// ---------------------------------------------------------------------------
// Reference Factory (for E-Graph Matching)
// ---------------------------------------------------------------------------
inline LogicalId refFactoryFusedMXFP4_CUDA(const std::vector<LogicalId> &inputs, Graph &graph) {
    // inputs[0]: raw packed weight [out_d, in_d / 2]
    // inputs[1]: raw scale [out_d, in_d / 32]
    LogicalId packed = graph.cast(inputs[0], DType::E2M1_PACKED_INT8);
    LogicalId unpacked = graph.unpack(packed, DType::E2M1);
    LogicalId unpacked_f32 = graph.cast(unpacked, DType::FLOAT32);

    LogicalId scale_f32 = graph.cast(inputs[1], DType::FLOAT32);

    // Infer shapes for intermediate nodes before accessing getShape()
    ShapePropagator propagator;
    propagator.inferShapeRecursive(unpacked_f32, graph);

    auto out_shape = graph.getNode(unpacked_f32).getShape();
    uint32_t out_d = out_shape[0];
    uint32_t in_d  = out_shape[1];
    uint32_t scale_w = in_d / 32;

    int32_t sh3_scale[] = {(int32_t)out_d, (int32_t)scale_w, 1};
    LogicalId scale_reshaped = graph.reshape(scale_f32, graph.constant({3}, sh3_scale, DType::INT32));
    
    int32_t rep32[] = {32};
    int32_t ax2[] = {2};
    LogicalId scale_repeated = graph.repeat(scale_reshaped, graph.constant({1}, rep32, DType::INT32), graph.constant({1}, ax2, DType::INT32));
    
    int32_t sh2_final[] = {(int32_t)out_d, (int32_t)in_d};
    LogicalId scale_final = graph.reshape(scale_repeated, graph.constant({2}, sh2_final, DType::INT32));

    return graph.mul(unpacked_f32, scale_final);
}

// Register as a fused CUDA kernel
REGISTER_KERNEL(
    "Fused_MXFP4_Dequant_CUDA", 2, 2,
    matchFusedMXFP4_CUDA, runFusedMXFP4_CUDA, refFactoryFusedMXFP4_CUDA,
    MemSpace(2, HandleType::CUDA),
    {Engine(0, EngineType::CUDA_GPU)},
    {DType::INT8, DType::FLOAT32},
    {{2048, 1024}, {2048, 64}},
    {true, true},
    {{MemSpace(2, HandleType::CUDA)}, {MemSpace(2, HandleType::CUDA)}}
);

#endif