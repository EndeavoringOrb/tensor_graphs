// File: tensor_graphs_cpp/kernels/cpu/reference/argmax/I32_ND.hpp
#pragma once
#include <algorithm>
#include <numeric>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchArgmaxI32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return isContiguous(output);
}

inline void runArgmaxI32_ND(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    int32_t axis = *static_cast<const int32_t *>(ctx.inputs[1]);
    int32_t k = *static_cast<const int32_t *>(ctx.inputs[2]);
    int32_t *out = static_cast<int32_t *>(ctx.outputs[0]);

    const auto &inShape = ctx.inViews[0].getShape();
    int ndim = static_cast<int>(inShape.size());
    if (axis < 0)
        axis += ndim;

    uint64_t outer = 1, mid = inShape[axis], inner = 1;
    for (int i = 0; i < axis; ++i)
        outer *= inShape[i];
    for (int i = axis + 1; i < ndim; ++i)
        inner *= inShape[i];

    for (uint64_t o = 0; o < outer; ++o)
    {
        for (uint64_t i = 0; i < inner; ++i)
        {
            // Gather indices and values along the target axis
            std::vector<std::pair<float, int32_t>> candidates(mid);
            for (uint32_t m = 0; m < mid; ++m)
            {
                uint64_t inIdx = (o * mid + m) * inner + i;
                candidates[m] = {in[inIdx], static_cast<int32_t>(m)};
            }

            // Sort descending by value, stable sort to preserve order of equal
            // elements
            std::stable_sort(candidates.begin(), candidates.end(),
                             [](const std::pair<float, int32_t> &a, const std::pair<float, int32_t> &b) {
                                 return a.first > b.first;
                             });

            // Write top-K indices to the output
            for (int32_t j = 0; j < k; ++j)
            {
                uint64_t outIdx = (o * k + j) * inner + i;
                out[outIdx] = (j < (int32_t)mid) ? candidates[j].second : -1;
            }
        }
    }
}

REGISTER_REF_KERNEL(OpType::ARGMAX, 3, 3, matchArgmaxI32_ND, runArgmaxI32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::INT32, DType::INT32}, {{8, 32}, {1}, {1}},
                    {true, false, false},
                    {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});