// Scatter_F32_Rows(_Threaded): an update box at a nonzero offset inside a
// larger output must land exactly there and leave the rest untouched.
#pragma once

#include <cstring>
#include <iostream>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline void runScatterRowsTests()
{
    std::cout << "scatter rows tests" << std::endl;
    const std::vector<uint32_t> outShape = {2, 5, 6};
    const std::vector<uint32_t> updShape = {1, 3, 4};
    const std::vector<int32_t> starts = {1, 1, 2}, ends = {2, 4, 6}, steps = {1, 1, 1}, shape = {2, 5, 6};

    std::vector<float> updates(countElements(updShape));
    for (size_t i = 0; i < updates.size(); ++i)
        updates[i] = 100.0f + static_cast<float>(i);

    for (const char *name : {"Scatter_F32_Rows", "Scatter_F32_Rows_Threaded"})
    {
        const KernelEntry *kernel = nullptr;
        for (const auto &[uid, entry] : KernelRegistry::get().getAllKernels())
            if (entry.opName == name)
                kernel = &entry;
        if (!kernel)
            Error::throw_err(std::string("[ScatterRows] kernel not registered: ") + name);

        std::vector<float> out(countElements(outShape), -1.0f);
        auto view = [](const std::vector<uint32_t> &s, DType d) {
            TensorView v;
            v.setShape(s);
            v.strides = calcContiguousStrides(s);
            v.dtype = d;
            return v;
        };
        KernelContext ctx({updates.data(), starts.data(), ends.data(), steps.data(), shape.data()}, {out.data()},
                          {view(updShape, DType::FLOAT32), view({3}, DType::INT32), view({3}, DType::INT32),
                           view({3}, DType::INT32), view({3}, DType::INT32)},
                          {view(outShape, DType::FLOAT32)});
        kernel->run(ctx);

        for (uint32_t a = 0; a < outShape[0]; ++a)
            for (uint32_t b = 0; b < outShape[1]; ++b)
                for (uint32_t c = 0; c < outShape[2]; ++c)
                {
                    const bool inside = a == 1 && b >= 1 && b < 4 && c >= 2 && c < 6;
                    const float expected =
                        inside ? updates[(b - 1) * updShape[2] + (c - 2)] : -1.0f;
                    const float got = out[(a * outShape[1] + b) * outShape[2] + c];
                    if (got != expected)
                        Error::throw_err(std::string("[ScatterRows] ") + name + " wrong value at (" +
                                         std::to_string(a) + "," + std::to_string(b) + "," + std::to_string(c) +
                                         "): " + std::to_string(got) + " expected " + std::to_string(expected));
                }
    }
    std::cout << "scatter rows tests passed" << std::endl;
}
