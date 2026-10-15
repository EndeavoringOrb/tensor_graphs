// Sum/Max_F32_LastAxis_NEON(_Threaded) on a row view of a larger tensor
// (non-unit leading strides) against a scalar reference.
#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline void runReduceLastAxisTests()
{
    std::cout << "reduce last axis tests" << std::endl;
    // Parent [3, 10, 37]; view rows 2..7 of axis 1: shape [3, 5, 37], strides [370, 37, 1].
    const uint32_t A = 3, P = 10, R0 = 2, R = 5, N = 37;
    std::vector<float> parent(static_cast<size_t>(A) * P * N);
    for (size_t i = 0; i < parent.size(); ++i)
        parent[i] = std::sin(0.7f * static_cast<float>(i)) * 3.0f;
    const float *view_ptr = parent.data() + static_cast<size_t>(R0) * N;
    const int32_t axis = -1;

    for (const char *name : {"Sum_F32_LastAxis_NEON", "Sum_F32_LastAxis_NEON_Threaded", "Max_F32_LastAxis_NEON",
                             "Max_F32_LastAxis_NEON_Threaded"})
    {
        const KernelEntry *kernel = nullptr;
        for (const auto &[uid, entry] : KernelRegistry::get().getAllKernels())
            if (entry.opName == name)
                kernel = &entry;
        if (!kernel)
            Error::throw_err(std::string("[ReduceLastAxis] kernel not registered: ") + name);
        const bool is_sum = std::string(name).rfind("Sum", 0) == 0;

        TensorView in;
        in.setShape({A, R, N});
        in.strides = {static_cast<uint64_t>(P) * N, N, 1};
        in.dtype = DType::FLOAT32;
        TensorView ax;
        ax.setShape({1});
        ax.strides = {1};
        ax.dtype = DType::INT32;
        TensorView outv;
        outv.setShape({A, R, 1});
        outv.strides = calcContiguousStrides(outv.getShape());
        outv.dtype = DType::FLOAT32;
        std::vector<float> out(static_cast<size_t>(A) * R, -7.0f);
        KernelContext ctx({view_ptr, &axis}, {out.data()}, {in, ax}, {outv});
        kernel->run(ctx);

        for (uint32_t a = 0; a < A; ++a)
            for (uint32_t r = 0; r < R; ++r)
            {
                const float *row = view_ptr + static_cast<size_t>(a) * P * N + static_cast<size_t>(r) * N;
                double ref = is_sum ? 0.0 : -std::numeric_limits<double>::infinity();
                for (uint32_t i = 0; i < N; ++i)
                    ref = is_sum ? ref + row[i] : std::max(ref, static_cast<double>(row[i]));
                const float got = out[static_cast<size_t>(a) * R + r];
                if (!(std::fabs(got - ref) <= 1e-4 * (1.0 + std::fabs(ref))))
                    Error::throw_err(std::string("[ReduceLastAxis] ") + name + " mismatch at (" + std::to_string(a) +
                                     "," + std::to_string(r) + "): " + std::to_string(got) + " vs " +
                                     std::to_string(ref));
            }
    }
    std::cout << "reduce last axis tests passed" << std::endl;
}
