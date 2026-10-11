// CPU AVX2 last-axis / general-axis F32 sum reduction for SUM.
// The goal's RMS-norm square-sum calls reduce the last (contiguous) axis over
// 256..640 element rows; a vectorized multi-accumulator CPU reduction beats
// the launch-bound CUDA baseline on all four configurations.
#if defined(TG_HAS_AVX2)
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#pragma GCC push_options
#pragma GCC target("avx2,fma")
#include <immintrin.h>

inline LogicalId refFactorySumAxisCpu(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // SumOp::inferShape reads the staged axis constant; stage a neutral
    // default when the axis input is a plain graph input during reference
    // discovery.
    if (!graph.constantStaging.count(inputs[1]))
    {
        auto data = std::make_shared<std::vector<uint8_t>>(sizeof(int32_t));
        const int32_t zero = 0;
        std::memcpy(data->data(), &zero, sizeof(int32_t));
        graph.constantStaging[inputs[1]] = data;
    }
    return graph.sum(inputs[0], inputs[1]);
}

inline bool sumAxisPackedStrides(const std::vector<uint64_t> &strides, const std::vector<uint32_t> &shape)
{
    uint64_t expected = 1;
    for (size_t d = shape.size(); d-- > 0;)
    {
        if (strides[d] != expected)
            return false;
        expected *= shape[d];
    }
    return true;
}

inline bool matchSumAxisCpu(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != output.getShape().size())
        return false;
    if (output.dtype != DType::FLOAT32)
        return false;
    if (!isContiguous(output))
        return false;
    if (!sumAxisPackedStrides(inputs[0].strides, inputs[0].getShape()))
        return false;
    for (size_t d = 0; d < output.getShape().size(); ++d)
    {
        if (output.getShape()[d] != inputs[0].getShape()[d] && output.getShape()[d] != 1)
            return false;
    }
    return true;
}

__attribute__((target("avx2,fma"))) inline float sumAxisRow(const float *row, uint64_t n)
{
    const __m256 zero = _mm256_setzero_ps();
    __m256 acc0 = zero, acc1 = zero, acc2 = zero, acc3 = zero;
    uint64_t i = 0;
    for (; i + 32 <= n; i += 32)
    {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(row + i));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(row + i + 8));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(row + i + 16));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(row + i + 24));
    }
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(row + i));
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    // Horizontal sum.
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    float total = _mm_cvtss_f32(s);
    for (; i < n; ++i)
        total += row[i];
    return total;
}

__attribute__((target("avx2,fma"))) inline void sumAxisInnerRange(const float *base, float *outBase,
                                                                  uint64_t outerPos, uint64_t innerCount,
                                                                  uint64_t innerStrideElems, uint64_t reduceCount,
                                                                  uint64_t outInnerStrideElems, uint64_t start,
                                                                  uint64_t count)
{
    // Reduces dimension `axis` for output positions [start, start+count) laid
    // out as (outerPos * innerCount + position) over the trailing axis.
    for (uint64_t k = start; k < start + count; ++k)
    {
        const uint64_t o = k / innerCount;
        const uint64_t inner = k % innerCount;
        const float *src = base + o * reduceCount * innerStrideElems + inner;
        float acc = 0.0f;
        for (uint64_t r = 0; r < reduceCount; ++r)
            acc += src[r * innerStrideElems];
        outBase[k / innerCount * outInnerStrideElems + (k % innerCount)] = acc;
        (void)outerPos;
    }
}

inline void runSumAxisCpu(const KernelContext &ctx)
{
    const TensorView &inView = ctx.inViews[0];
    const TensorView &outView = ctx.outViews[0];
    const uint32_t ndim = inView.getShape().size();
    if (ndim == 0)
        return;

    // Read the axis scalar (constants live in a host buffer; handle a storage
    // descriptor defensively).
    int32_t axisValue = 0;
    if (ctx.fd.size() > 1 && ctx.fd[1] >= 0)
    {
        ssize_t got = ::pread(ctx.fd[1], &axisValue, sizeof(int32_t), static_cast<off_t>(ctx.inViews[1].offset));
        if (got != static_cast<ssize_t>(sizeof(int32_t)))
            Error::throw_err("Sum_Axis_F32_AVX2: failed to read axis from storage");
    }
    else
    {
        std::memcpy(&axisValue, ctx.inputs[1], sizeof(int32_t));
    }
    int64_t axis = axisValue < 0 ? static_cast<int64_t>(ndim) + axisValue : static_cast<int64_t>(axisValue);
    if (axis < 0 || axis >= static_cast<int64_t>(ndim))
        Error::throw_err("Sum_Axis_F32_AVX2: axis out of range: " + std::to_string(axisValue));

    const auto &inShape = inView.getShape();
    const float *base = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    uint64_t outer = 1;
    for (uint64_t d = 0; d < static_cast<uint64_t>(axis); ++d)
        outer *= inShape[d];
    const uint64_t reduce = inShape[axis];
    uint64_t inner = 1;
    for (uint64_t d = static_cast<uint64_t>(axis) + 1; d < ndim; ++d)
        inner *= inShape[d];
    const uint64_t outInner = (axis + 1 < ndim) ? outView.strides[axis] : 1;

    if (reduce == 0)
    {
        const uint64_t total = outer * inner;
        for (uint64_t k = 0; k < total; ++k)
        {
            const uint64_t o = k / inner;
            const uint64_t i = k % inner;
            out[o * outInner + i] = 0.0f;
        }
        return;
    }

    if (axis + 1 == ndim)
    {
        // Fast path: reduce the contiguous last dimension, one row per output.
        const uint64_t rows = outer;
        const uint64_t rowLen = reduce;
        uint32_t numThreads = ThreadPool::get().get_num_threads();
        if (numThreads == 0)
            numThreads = std::max(1U, std::thread::hardware_concurrency());
        const bool threadIt = rows * rowLen >= 131072 && numThreads > 1;
        if (!threadIt)
        {
            for (uint64_t r = 0; r < rows; ++r)
                out[r * outInner] = sumAxisRow(base + r * rowLen, rowLen);
            return;
        }
        uint64_t chunk = (rows + numThreads - 1) / numThreads;
        ThreadPool::get().parallel_for(numThreads, [=](uint32_t t) {
            uint64_t start = t * chunk;
            if (start >= rows)
                return;
            uint64_t end = std::min(start + chunk, rows);
            for (uint64_t r = start; r < end; ++r)
                out[r * outInner] = sumAxisRow(base + r * rowLen, rowLen);
        });
        return;
    }

    // General path: strided reduction along an interior axis.
    const uint64_t total = outer * inner;
    const uint64_t innerStrideElems = inner; // packed input: stride[axis] == inner
    uint32_t numThreads = ThreadPool::get().get_num_threads();
    if (numThreads == 0)
        numThreads = std::max(1U, std::thread::hardware_concurrency());
    const bool threadIt = total * reduce >= 131072 && numThreads > 1;
    if (!threadIt)
    {
        sumAxisInnerRange(base, out, 0, inner, innerStrideElems, reduce, outInner, 0, total);
        return;
    }
    uint64_t chunk = (total + numThreads - 1) / numThreads;
    ThreadPool::get().parallel_for(numThreads, [=](uint32_t t) {
        uint64_t start = t * chunk;
        if (start >= total)
            return;
        sumAxisInnerRange(base, out, 0, inner, innerStrideElems, reduce, outInner, start,
                          std::min(start + chunk, total) - start);
    });
}

REGISTER_KERNEL("Sum_Axis_F32_AVX2", 2, 2, matchSumAxisCpu, runSumAxisCpu, refFactorySumAxisCpu,
                {0}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::INT32}, {{8, 32}, {1}}, {true, false},
                {MemSpace(1, HandleType::CPP), MemSpace(1, HandleType::CPP)});
#pragma GCC pop_options
#endif // TG_HAS_AVX2
