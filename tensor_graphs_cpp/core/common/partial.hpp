#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>
#include <algorithm>

namespace partial_ops
{
    // Ensures the target buffer is correctly prepared in the output buffer
    inline void copy_target_if_needed(const float *target, float *out, const std::vector<uint32_t> &shape,
                                      const std::vector<uint64_t> &tgt_strides, const std::vector<uint64_t> &out_strides)
    {
        if (target != out)
        {
            uint64_t n = countElements(shape);
            for (uint64_t i = 0; i < n; ++i)
            {
                out[getStridedIndex(i, shape, out_strides)] = target[getStridedIndex(i, shape, tgt_strides)];
            }
        }
    }

    // Calculates the shape of the slice being updated
    inline void compute_slice_shape(const std::vector<uint32_t> &out_shape,
                                    const int32_t *starts, const int32_t *ends, const int32_t *steps,
                                    uint32_t starts_size, uint32_t ends_size, uint32_t steps_size,
                                    std::vector<uint32_t> &slice_shape)
    {
        slice_shape.resize(out_shape.size());
        for (size_t i = 0; i < out_shape.size(); ++i)
        {
            int32_t st = (i < starts_size) ? starts[i] : 0;
            int32_t en = (i < ends_size) ? ends[i] : (int32_t)out_shape[i];
            int32_t step = (i < steps_size) ? steps[i] : 1;
            if (st < 0)
                st += out_shape[i];
            if (en < 0)
                en += out_shape[i];
            slice_shape[i] = static_cast<uint32_t>(std::max(0, (en - st + step - 1) / step));
        }
    }
}