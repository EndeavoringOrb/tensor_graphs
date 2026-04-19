#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

inline bool matchGatherBF16(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;
    if (inputs[0].dtype != DType::BF16 || inputs[1].dtype != DType::INT32)
        return false;
    if (output.dtype != DType::FLOAT32)
        return false;

    if (inputs[0].getShape().empty())
        return false;

    return true;
}

inline void runGatherBF16(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                          const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const uint16_t *data = static_cast<const uint16_t *>(inputs[0]);
    const int32_t *indices = static_cast<const int32_t *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    const auto &dataShape = inViews[0].getShape();
    const auto &idxShape = inViews[1].getShape();

    uint32_t vocabSize = dataShape[0];
    uint64_t rowSize = 1;
    for (size_t i = 1; i < dataShape.size(); ++i)
        rowSize *= dataShape[i];

    uint64_t numIndices = countElements(idxShape);

    for (uint64_t i = 0; i < numIndices; ++i)
    {
        int32_t idx = indices[i];
        if (idx < 0 || (uint32_t)idx >= vocabSize)
        {
            std::memset(out + (i * rowSize), 0, rowSize * sizeof(float));
            continue;
        }

        const uint16_t *src_row = data + (idx * rowSize);
        float *dst_row = out + (i * rowSize);
        for (uint64_t j = 0; j < rowSize; ++j)
        {
            uint32_t bits = static_cast<uint32_t>(src_row[j]) << 16;
            std::memcpy(&dst_row[j], &bits, 4);
        }
    }
}

inline uint32_t refFactoryGatherBF16(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t casted = graph.cast(inputs[0], DType::FLOAT32);
    return graph.gather(casted, inputs[1]);
}

REGISTER_KERNEL("Gather_BF16", 2, matchGatherBF16, runGatherBF16, refFactoryGatherBF16, {Backend::CPU}, {DType::BF16, DType::INT32}, {{262144, 640}, {1, 8}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});