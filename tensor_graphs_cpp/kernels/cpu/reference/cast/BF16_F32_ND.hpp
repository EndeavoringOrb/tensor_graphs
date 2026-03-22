#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

/**
 * ---------------------------------------------------------
 * KERNEL: CAST BF16 -> FLOAT32 (ND, Contiguous)
 * ---------------------------------------------------------
 * This kernel converts Bfloat16 tensors to Float32.
 * BF16 consists of 1 sign bit, 8 exponent bits, and 7 mantissa bits.
 * To convert to F32, we shift the bits left by 16.
 */

/**
 * Match Function:
 * Validates that input is BF16, output is F32, shapes match, and both are contiguous.
 */
inline bool matchCastBF16_F32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 1)
        return false;

    // Check Dtypes
    if (inputs[0].dtype != DType::BF16 || output.dtype != DType::FLOAT32)
        return false;

    // Check Shape Identity
    if (inputs[0].shape != output.shape)
        return false;

    // Reference implementation requires contiguity for flat iteration
    if (!inputs[0].view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

/**
 * Run Function:
 * Iterates through all elements, performing bit-shifting for conversion.
 */
inline void runCastBF16_F32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                               const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    // BF16 is stored as uint16_t raw bits
    const uint16_t *src = static_cast<const uint16_t *>(inputs[0]);
    float *dst = static_cast<float *>(outputs[0]);

    uint64_t numElements = countElements(inViews[0].shape);

    for (uint64_t i = 0; i < numElements; ++i)
    {
        // 1. Shift bits left by 16 to move BF16 bits to the top of a 32-bit word
        uint32_t f32_bits = static_cast<uint32_t>(src[i]) << 16;

        // 2. Safely bit_cast to float using memcpy to avoid strict aliasing issues
        float val;
        std::memcpy(&val, &f32_bits, sizeof(float));

        dst[i] = val;
    }
}

// Register as a CPU kernel for the CAST operation
REGISTER_REF_KERNEL(OpType::CAST, matchCastBF16_F32_ND, runCastBF16_F32_ND, {Backend::CPU});