#pragma once
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "core/types.hpp"
#include "core/memory.hpp"

namespace Debug
{
    inline void checkNan(const TensorNode &node, const MemoryManager &mem, const std::string &context)
    {
#ifndef DEBUG
        return;
#endif
        // INT32 and BOOL cannot represent NaNs, so we skip them
        if (node.dtype != DType::FLOAT32 && node.dtype != DType::BF16)
        {
            return;
        }

        const uint8_t *basePtr = mem.read(node.backend, node.id);
        if (!basePtr)
            return;

#ifdef USE_CUDA
        std::vector<uint8_t> hostData;
        if (node.backend == Backend::CUDA)
        {
            TensorView view = mem.getView(node);
            uint64_t numElements = countElements(view.shape);
            uint64_t sizeBytes = numElements * getDTypeSize(node.dtype);
            hostData.resize(sizeBytes);
            cudaMemcpy(hostData.data(), basePtr, sizeBytes, cudaMemcpyDeviceToHost);
            basePtr = hostData.data();
        }
#endif

        TensorView view = mem.getView(node);
        uint64_t numElements = countElements(view.shape);

        // This check assumes contiguous layout for debugging simplicity.
        // If your tensor is heavily sliced/permuted, this might need a strided iterator.
        if (node.dtype == DType::FLOAT32)
        {
            const float *data = reinterpret_cast<const float *>(basePtr);
            for (uint64_t i = 0; i < numElements; ++i)
            {
                if (std::isnan(data[i]))
                {
                    std::stringstream ss;
                    ss << "[NaN Detection] Found NaN in node " << node.id
                       << " (" << toString(node.opType) << (node.opType == OpType::FUSED ? " " + node.opName : "") << ")"
                       << " during \"" << context
                       << "\" at element index " << i;
                    Error::throw_err(ss.str());
                }
            }
        }
        else if (node.dtype == DType::BF16)
        {
            const uint16_t *data = reinterpret_cast<const uint16_t *>(basePtr);
            for (uint64_t i = 0; i < numElements; ++i)
            {
                // BF16 NaN: Exponent (bits 7-14) is all 1s, Mantissa (bits 0-6) is non-zero
                uint16_t bits = data[i];
                bool is_nan = ((bits & 0x7F80) == 0x7F80) && ((bits & 0x007F) != 0);
                if (is_nan)
                {
                    std::stringstream ss;
                    ss << "[NaN Detection] Found BF16 NaN in node " << node.id
                       << " (" << toString(node.opType) << ")"
                       << " during " << context
                       << " at element index " << i;
                    Error::throw_err(ss.str());
                }
            }
        }
    }
}