#pragma once
#include <cstring>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchCopyTo_CPU_CPU(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &in = inputs[0];

    if (in.dtype != output.dtype)
        return false;

    if (in.getShape() != output.getShape())
        return false;

    if (in.strides != output.strides)
        return false;

    if (!isContiguous(output))
        return false;

    return true;
}

inline void runCopyTo_CPU_CPU(const KernelContext &ctx)
{
    const uint8_t *src = static_cast<const uint8_t *>(ctx.inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(ctx.outputs[0]);

    uint64_t numElements = countElements(ctx.inViews[0].getShape());
    uint64_t elemSize = getDTypeSize(ctx.inViews[0].dtype);

    std::memcpy(dst, src, numElements * elemSize);
}

// Register the kernel for the COPY_TO operation on the CPU backend
REGISTER_REF_KERNEL(OpType::COPY_TO, 1, 1, matchCopyTo_CPU_CPU, runCopyTo_CPU_CPU, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::ANY}, {{8, 32}}, {true}, {{MemSpace(1, HandleType::CPP)}});