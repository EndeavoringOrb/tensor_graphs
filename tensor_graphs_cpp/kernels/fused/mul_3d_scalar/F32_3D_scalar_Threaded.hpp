#pragma once
#include <algorithm>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/mul_3d_scalar/ref.hpp"
inline bool matchMulFP32_3D_Scalar_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != 1)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    return isContiguous(output);
}

inline void runMulFP32_3D_Scalar_Threaded(const KernelContext &ctx)
{
    const float *data3D = static_cast<const float *>(ctx.inputs[0]);
    float scalarValue = *static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    uint64_t totalElements = countElements(ctx.inViews[0].getShape());
    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint64_t chunk = (totalElements + num_threads - 1) / num_threads;
        uint64_t start = t * chunk;
        uint64_t end = std::min(start + chunk, totalElements);
        for (uint64_t i = start; i < end; ++i)
            out[i] = data3D[i] * scalarValue;
    });
}



REGISTER_KERNEL("Mul_3D_Scalar_Threaded", 2, 2, matchMulFP32_3D_Scalar_Threaded, runMulFP32_3D_Scalar_Threaded,
                refFactoryMul3D_Scalar, {0}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32}, {{1, 1, 1}, {1}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});