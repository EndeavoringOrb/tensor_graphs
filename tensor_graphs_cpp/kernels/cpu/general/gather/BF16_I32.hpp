#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

inline bool matchGatherBF16(const std::vector<TensorNode> &inputs, const TensorNode &output)
{

    if (inputs[0].getShape().empty())
        return false;

    return true;
}

inline void runGatherBF16(const KernelContext &ctx)
{
    const uint16_t *data = static_cast<const uint16_t *>(ctx.inputs[0]);
    const int32_t *indices = static_cast<const int32_t *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &dataShape = ctx.inViews[0].getShape();
    const auto &idxShape = ctx.inViews[1].getShape();

    uint32_t vocabSize = dataShape[0];
    uint64_t rowSize = 1;
    for (uint64_t i = 1; i < dataShape.size(); ++i)
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

inline LogicalId refFactoryGatherBF16(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId casted = graph.cast(inputs[0], DType::FLOAT32);
    return graph.gather(casted, inputs[1]);
}

REGISTER_KERNEL("Gather_BF16", 2, 2, matchGatherBF16, runGatherBF16, refFactoryGatherBF16, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::BF16, DType::INT32}, {{262144, 640}, {1, 8}}, {true, true}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});