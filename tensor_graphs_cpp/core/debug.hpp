#pragma once
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "core/types.hpp"
#include "core/memory.hpp"

namespace Debug
{
    inline void checkNan(const TensorNode &node, const MemoryManager &mem, const std::string &context, const CompiledGraph &compiled = {})
    {
#ifndef DEBUG
        return;
#endif
        // INT32 and BOOL cannot represent NaNs, so we skip them
        if (node.dtype != DType::FLOAT32 && node.dtype != DType::BF16)
        {
            return;
        }

        TensorView view = mem.getView(node, compiled);

        auto it = mem.buffers.find(node.backend);
        if (it == mem.buffers.end())
            return;
        const uint8_t *basePtr = it->second.arena_ptr + view.baseOffset;
        if (!basePtr)
            return;

#ifdef USE_CUDA
        std::vector<uint8_t> hostData;
        if (node.backend == Backend::CUDA)
        {
            uint64_t maxOffset = 0;
            for (size_t i = 0; i < view.getShape().size(); ++i)
            {
                if (view.getShape()[i] > 0)
                {
                    maxOffset += (view.getShape()[i] - 1) * view.strides[i];
                }
            }
            uint64_t sizeBytes = (maxOffset + 1) * getDTypeSize(node.dtype);
            hostData.resize(sizeBytes);
            cudaMemcpy(hostData.data(), basePtr, sizeBytes, cudaMemcpyDeviceToHost);
            basePtr = hostData.data();
        }
#endif

        uint64_t numElements = countElements(node);

        if (node.dtype == DType::FLOAT32)
        {
            const float *data = reinterpret_cast<const float *>(basePtr);
            for (uint64_t i = 0; i < numElements; ++i)
            {
                uint64_t idx = getStridedIndex(i, view.getShape(), view.strides);
                if (std::isnan(data[idx]))
                {
                    std::stringstream ss;
                    ss << "[NaN Detection] Found NaN in node " << node.id
                       << " (" << toString(node.opType) << (node.opType == OpType::FUSED ? " " + node.opName : "") << ")"
                       << " during \"" << context
                       << "\" at element index " << i << " (flat index " << idx << ")";
                    Error::throw_err(ss.str());
                }
            }
        }
        else if (node.dtype == DType::BF16)
        {
            const uint16_t *data = reinterpret_cast<const uint16_t *>(basePtr);
            for (uint64_t i = 0; i < numElements; ++i)
            {
                uint64_t idx = getStridedIndex(i, view.getShape(), view.strides);
                uint16_t bits = data[idx];
                bool is_nan = ((bits & 0x7F80) == 0x7F80) && ((bits & 0x007F) != 0);
                if (is_nan)
                {
                    std::stringstream ss;
                    ss << "[NaN Detection] Found BF16 NaN in node " << node.id
                       << " (" << toString(node.opType) << ")"
                       << " during " << context
                       << " at element index " << i << " (flat index " << idx << ")";
                    Error::throw_err(ss.str());
                }
            }
        }
    }
}