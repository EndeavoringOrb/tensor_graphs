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
inline bool matchCastBF16_F32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{

    // Check Shape Identity
    if (inputs[0].getShape() != output.getShape())
        return false;

    // Reference implementation requires contiguity for flat iteration
    if (!isContiguous(output))
        return false;

    return true;
}

/**
 * Run Function:
 * Iterates through all elements, performing bit-shifting for conversion.
 */
inline void runCastBF16_F32_ND(const KernelContext &ctx)
{
    // BF16 is stored as uint16_t raw bits
    const uint16_t *src = static_cast<const uint16_t *>(ctx.inputs[0]);
    float *dst = static_cast<float *>(ctx.outputs[0]);

    uint64_t numElements = countElements(ctx.inViews[0].getShape());

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
REGISTER_REF_KERNEL(OpType::CAST, 1, matchCastBF16_F32_ND, runCastBF16_F32_ND, {Backend::CPU}, {DType::BF16}, {{8, 32}}, {true}, {{Backend::CPU}});
