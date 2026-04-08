#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

/**
 * KERNEL: CAST INT32 -> FLOAT32 (ND, Contiguous)
 * ---------------------------------------------------------
 * This kernel performs a standard numerical cast from 32-bit
 * integers to 32-bit floating point numbers.
 */

inline bool matchCastI32_F32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 1)
        return false;

    // Check Dtypes: Input must be INT32, Output must be FLOAT32
    if (inputs[0].dtype != DType::INT32 || output.dtype != DType::FLOAT32)
        return false;

    // Check Shape Identity
    if (inputs[0].getShape() != output.getShape())
        return false;

    // Reference implementation requires contiguity
    if (!isContiguous(inputs[0]) || !isContiguous(output))
        return false;

    return true;
}

inline void runCastI32_F32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const int32_t *src = static_cast<const int32_t *>(inputs[0]);
    float *dst = static_cast<float *>(outputs[0]);

    uint64_t numElements = countElements(inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        dst[i] = static_cast<float>(src[i]);
    }
}

// Register as a CPU kernel for the CAST operation
REGISTER_REF_KERNEL(OpType::CAST, 1, matchCastI32_F32_ND, runCastI32_F32_ND, {Backend::CPU}, {DType::INT32}, {{8, 32}}, {false}, {{Backend::CPU}});