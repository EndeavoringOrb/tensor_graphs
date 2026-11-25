// BF16_Transposed_GEMM_NEON_WStream_Threaded against a scalar reference on
// shapes that exercise every remainder path (rows, W rows and K not multiples
// of 4) and a batch above 1.
#pragma once

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline void runBF16TransposedWStreamTests()
{
    std::cout << "bf16 transposed wstream tests" << std::endl;
    for (const char *name : {"BF16_Transposed_GEMM_NEON_WStream_Threaded", "BF16_Transposed_GEMM_NEON_WStream_v2_Threaded",
                             "BF16_Transposed_GEMM_NEON_WStream_v3_Threaded"})
    {
    const KernelEntry *kernel = nullptr;
    for (const auto &[uid, entry] : KernelRegistry::get().getAllKernels())
        if (entry.opName == name)
            kernel = &entry;
    if (!kernel)
        Error::throw_err(std::string("[BF16TransposedWStream] kernel not registered: ") + name);

    struct Case
    {
        uint32_t B, S, K, N;
    };
    for (const Case &c : {Case{1, 1, 640, 1000}, Case{1, 8, 640, 4096}, Case{2, 7, 67, 13}, Case{1, 13, 3, 6},
                          Case{1, 5, 64, 3}})
    {
        std::vector<float> X(static_cast<size_t>(c.B) * c.S * c.K);
        std::vector<uint16_t> W(static_cast<size_t>(c.N) * c.K);
        for (size_t i = 0; i < X.size(); ++i)
            X[i] = std::sin(0.37f * static_cast<float>(i)) * 1.5f;
        for (size_t i = 0; i < W.size(); ++i)
        {
            const float v = std::cos(0.11f * static_cast<float>(i)) * 0.75f;
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            W[i] = static_cast<uint16_t>(bits >> 16);
        }
        std::vector<float> out(static_cast<size_t>(c.B) * c.S * c.N, -7.0f);
        auto view = [](const std::vector<uint32_t> &s, DType d) {
            TensorView v;
            v.setShape(s);
            v.strides = calcContiguousStrides(s);
            v.dtype = d;
            return v;
        };
        KernelContext ctx({X.data(), W.data()}, {out.data()},
                          {view({c.B, c.S, c.K}, DType::FLOAT32), view({c.N, c.K}, DType::BF16)},
                          {view({c.B, c.S, c.N}, DType::FLOAT32)});
        kernel->run(ctx);

        for (uint32_t b = 0; b < c.B; ++b)
            for (uint32_t s = 0; s < c.S; ++s)
                for (uint32_t n = 0; n < c.N; ++n)
                {
                    double ref = 0.0, mag = 0.0;
                    for (uint32_t k = 0; k < c.K; ++k)
                    {
                        const uint32_t bits = static_cast<uint32_t>(W[static_cast<size_t>(n) * c.K + k]) << 16;
                        float w;
                        std::memcpy(&w, &bits, sizeof(w));
                        const double term = static_cast<double>(X[(static_cast<size_t>(b) * c.S + s) * c.K + k]) * w;
                        ref += term;
                        mag += std::fabs(term);
                    }
                    const float got = out[(static_cast<size_t>(b) * c.S + s) * c.N + n];
                    if (!(std::fabs(got - ref) <= 1e-5 * mag + 1e-6))
                        Error::throw_err(std::string("[BF16TransposedWStream] ") + name + " mismatch B=" + std::to_string(c.B) +
                                         " S=" + std::to_string(c.S) + " K=" + std::to_string(c.K) +
                                         " N=" + std::to_string(c.N) + " at (" + std::to_string(b) + "," +
                                         std::to_string(s) + "," + std::to_string(n) + "): " + std::to_string(got) +
                                         " vs " + std::to_string(ref));
                }
    }
    }
    std::cout << "bf16 transposed wstream tests passed" << std::endl;
}
