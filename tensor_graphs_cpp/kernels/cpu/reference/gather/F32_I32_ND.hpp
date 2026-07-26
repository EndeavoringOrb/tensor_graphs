#pragma once
#include <cstring>

#include "core/kernels.hpp"
#include "core/types.hpp"

/**
 * KERNEL: GATHER F32 (Data) + I32 (Indices)
 * Performs: output = data[indices]
 * Logic: For each index in the indices tensor, copy a 'row' from the data
 * tensor.
 */

inline bool matchGatherF32_I32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // inputs[0] = data, inputs[1] = indices

    // Simple check: data must be at least 1D
    if (inputs[0].getShape().empty())
        return false;

    // Reference implementation requires contiguity
    if (!isContiguous(output))
        return false;

    return true;
}

inline void runGatherF32_I32_ND(const KernelContext &ctx)
{
    const float *data = static_cast<const float *>(ctx.inputs[0]);
    const int32_t *indices = static_cast<const int32_t *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const std::vector<uint32_t> &dataShape = ctx.inViews[0].getShape();
    const std::vector<uint32_t> &idxShape = ctx.inViews[1].getShape();

    uint32_t vocabSize = dataShape[0];
    uint64_t rowSize = 1;
    for (uint64_t i = 1; i < dataShape.size(); ++i)
        rowSize *= dataShape[i];

    uint64_t numIndices = countElements(idxShape);

    for (uint64_t i = 0; i < numIndices; ++i)
    {
        int32_t idx = indices[i];

        // Basic bounds checking
        if (idx < 0 || (uint32_t)idx >= vocabSize)
        {
            // In a real system, you might want to zero out or throw
            std::memset(out + (i * rowSize), 0, rowSize * sizeof(float));
            continue;
        }

        // Copy the row
        std::memcpy(out + (i * rowSize), data + (idx * rowSize), rowSize * sizeof(float));
    }
}

REGISTER_REF_KERNEL(OpType::GATHER, 2, 2, matchGatherF32_I32_ND, runGatherF32_I32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::INT32}, {{8, 32}, {8}}, {true, true},
                    {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
