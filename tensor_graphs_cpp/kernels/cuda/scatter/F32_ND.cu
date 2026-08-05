#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

namespace ScatterCUDA {
    constexpr int MAX_RANK = 8;
    struct ScatterParams {
        uint32_t ndim;
        uint32_t upd_shape[MAX_RANK];
        uint32_t out_shape[MAX_RANK];
        uint64_t out_strides[MAX_RANK];
        int32_t starts[MAX_RANK];
        int32_t steps[MAX_RANK];
    };

    __global__ void scatter_f32_nd_kernel(const float* updates, float* Out, uint64_t n_updates, ScatterParams p) {
        uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_updates) return;

        uint64_t temp = idx;
        uint64_t out_phys_idx = 0;

        for (int d = (int)p.ndim - 1; d >= 0; --d) {
            uint32_t coord = temp % p.upd_shape[d];
            temp /= p.upd_shape[d];

            int32_t s = p.starts[d];
            if (s < 0) s += p.out_shape[d];
            int32_t st = p.steps[d];

            uint32_t target_coord = s + coord * st;
            out_phys_idx += (uint64_t)target_coord * p.out_strides[d];
        }

        Out[out_phys_idx] = updates[idx];
    }
}

inline bool matchScatterF32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::FLOAT32) return false;
    if (inputs[0].getShape().size() != inputs[1].getShape().size() || inputs[0].getShape().size() != output.getShape().size()) return false;
    uint32_t rank = static_cast<uint32_t>(inputs[0].getShape().size());
    if (rank > ScatterCUDA::MAX_RANK) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runScatterF32_CUDA_ND(const KernelContext &ctx) {
    const float *target = static_cast<const float *>(ctx.inputs[0]);
    const float *updates = static_cast<const float *>(ctx.inputs[1]);
    const int32_t *starts = static_cast<const int32_t *>(ctx.inputs[2]);
    const int32_t *steps = static_cast<const int32_t *>(ctx.inputs[4]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const auto &out_shape = ctx.outViews[0].getShape();
    const auto &upd_shape = ctx.inViews[1].getShape();
    uint64_t n_target = countElements(out_shape);

    if (target != Out && n_target > 0) {
        cudaMemcpy(Out, target, n_target * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    uint64_t n_updates = countElements(upd_shape);
    if (n_updates == 0) return;

    ScatterCUDA::ScatterParams p;
    p.ndim = static_cast<uint32_t>(upd_shape.size());
    for (uint32_t d = 0; d < p.ndim; ++d) {
        p.upd_shape[d] = upd_shape[d];
        p.out_shape[d] = out_shape[d];
        p.out_strides[d] = ctx.outViews[0].strides[d];
        p.starts[d] = (d < ctx.inViews[2].getShape()[0]) ? starts[d] : 0;
        p.steps[d] = (d < ctx.inViews[4].getShape()[0]) ? steps[d] : 1;
    }

    int blockSize = 256;
    int numBlocks = (n_updates + blockSize - 1) / blockSize;

    ScatterCUDA::scatter_f32_nd_kernel<<<numBlocks, blockSize>>>(updates, Out, n_updates, p);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Scatter_F32_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

inline LogicalId refFactoryScatterF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.scatter(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
}

REGISTER_KERNEL("Scatter_F32_ND_CUDA", 5, 5, matchScatterF32_CUDA_ND, runScatterF32_CUDA_ND, refFactoryScatterF32_ND_CUDA, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::FLOAT32, DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32}, {{8, 32}, {8, 32}, {2}, {2}, {2}}, {false, false, false, false, false}, {{MemSpace(2, HandleType::CUDA)}, {MemSpace(2, HandleType::CUDA)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif