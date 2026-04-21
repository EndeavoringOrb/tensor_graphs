// File: tensor_graphs_cpp/kernels/cpu/general/mul/F32_3D_scalar_Threaded.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>
#include <thread>
#include <algorithm>

inline bool matchMulFP32_3D_Scalar_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != 1)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    return isContiguous(output);
}

inline void runMulFP32_3D_Scalar_Threaded(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                          const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *data3D = static_cast<const float *>(inputs[0]);
    float scalarValue = *static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    uint64_t totalElements = countElements(inViews[0].getShape());
    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    uint64_t chunk = (totalElements + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint64_t start = t * chunk;
            uint64_t end = std::min(start + chunk, totalElements);
            for (uint64_t i = start; i < end; ++i)
                out[i] = data3D[i] * scalarValue; });
    }
    for (auto &w : workers)
        w.join();
}

inline uint32_t refFactoryMul3D_Scalar_Threaded(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Mul 3D+Scalar requires 2 inputs");

    const auto &shape3D = graph.getNode(inputs[0]).getShape();
    int32_t reshape_dims[] = {1, 1, 1};
    uint32_t out = graph.reshape(inputs[1], graph.constant({3}, reshape_dims, DType::INT32));

    for (int i = 0; i < 3; ++i)
    {
        int32_t rep = (int32_t)shape3D[i];
        int32_t axis = i;
        out = graph.repeat(out, graph.constant({1}, &rep, DType::INT32), graph.constant({1}, &axis, DType::INT32));
    }
    return graph.mul(inputs[0], out);
}

REGISTER_KERNEL("Mul_3D_Scalar_Threaded", 2, matchMulFP32_3D_Scalar_Threaded, runMulFP32_3D_Scalar_Threaded, refFactoryMul3D_Scalar_Threaded, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 1, 1}, {1}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});