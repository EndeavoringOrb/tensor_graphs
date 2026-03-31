#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

/**
 * KERNEL: GATHER F32 (Data) + I32 (Indices)
 * Performs: output = data[indices]
 * Logic: For each index in the indices tensor, copy a 'row' from the data tensor.
 */

inline bool matchGatherF32_I32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2)
        return false;

    // inputs[0] = data, inputs[1] = indices
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::INT32 || output.dtype != DType::FLOAT32)
        return false;

    // Simple check: data must be at least 1D
    if (inputs[0].getShape().empty())
        return false;

    // Reference implementation requires contiguity
    if (!isContiguous(inputs[0]) || !isContiguous(inputs[1]) || !isContiguous(output))
        return false;

    return true;
}

inline void runGatherF32_I32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *data = static_cast<const float *>(inputs[0]);
    const int32_t *indices = static_cast<const int32_t *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    const std::vector<uint32_t> &dataShape = inViews[0].getShape();
    const std::vector<uint32_t> &idxShape = inViews[1].getShape();

    uint32_t vocabSize = dataShape[0];
    uint64_t rowSize = 1;
    for (size_t i = 1; i < dataShape.size(); ++i)
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

REGISTER_REF_KERNEL(OpType::GATHER, matchGatherF32_I32_ND, runGatherF32_I32_ND, {Backend::CPU});