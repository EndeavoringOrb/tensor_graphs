#pragma once
#include "core/common/partial.hpp"

inline bool matchPartialPowInplace(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs.size() == 6 && inputs[0].dtype == DType::FLOAT32 && inputs[0].storageType != StorageType::PERSISTENT;
}

inline uint32_t refFactoryPartialPow(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t target = inputs[0], A = inputs[1], B = inputs[2], st = inputs[3], en = inputs[4], step = inputs[5];
    uint32_t op_res = graph.pow(graph.contiguous(graph.slice(A, st, en, step)), graph.contiguous(graph.slice(B, st, en, step)));
    return graph.scatter(target, graph.contiguous(op_res), st, en, step);
}

inline void runPartialPowInplace(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                 const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *target = (const float *)inputs[0], *A = (const float *)inputs[1], *B = (const float *)inputs[2];
    const int32_t *st = (const int32_t *)inputs[3], *en = (const int32_t *)inputs[4], *step = (const int32_t *)inputs[5];
    float *out = (float *)outputs[0];
    const auto &out_shape = outViews[0].getShape();
    partial_ops::copy_target_if_needed(target, out, out_shape, inViews[0].strides, outViews[0].strides);
    std::vector<uint32_t> sl_sh;
    partial_ops::compute_slice_shape(out_shape, st, en, step, inViews[3].getShape()[0], inViews[4].getShape()[0], inViews[5].getShape()[0], sl_sh);
    uint64_t n = countElements(sl_sh);
    for (uint64_t i = 0; i < n; ++i)
    {
        uint64_t temp = i, out_idx = 0, a_idx = 0, b_idx = 0;
        for (int d = (int)sl_sh.size() - 1; d >= 0; --d)
        {
            uint32_t coord = temp % sl_sh[d];
            temp /= sl_sh[d];
            int32_t s = (d < (int)inViews[3].getShape()[0]) ? st[d] : 0;
            if (s < 0)
                s += out_shape[d];
            int32_t stp = (d < (int)inViews[5].getShape()[0]) ? step[d] : 1;
            uint64_t oc = s + (uint64_t)coord * stp;
            out_idx += oc * outViews[0].strides[d];
            a_idx += oc * inViews[1].strides[d];
            b_idx += oc * inViews[2].strides[d];
        }
        out[out_idx] = std::pow(A[a_idx], B[b_idx]);
    }
}

REGISTER_KERNEL_INPLACE("Partial_Pow_inplace", 6, matchPartialPowInplace, runPartialPowInplace, refFactoryPartialPow, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32}, {{8, 32}, {8, 32}, {8, 32}, {8}, {8}, {8}}, {false, false, false, false, false, false}, {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});