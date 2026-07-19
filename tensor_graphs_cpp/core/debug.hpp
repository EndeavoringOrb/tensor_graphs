// tensor_graphs_cpp/core/debug.hpp
#pragma once
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <cstring>
#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"

namespace Debug
{
    using Callback = std::function<void(uint32_t logicalId, const TensorNode &node, const KernelContext &ctx, const void *data)>;

    inline void checkNan(const TensorNode &node, const MemoryManager &mem, const std::string &context)
    {
#ifndef DEBUG_CHECKNAN
        return;
#endif
        // Skip checking model weights/inputs for NaNs, as they are out of our control
        if (node.opType == OpType::INPUT)
        {
            return;
        }

        // INT32 and BOOL cannot represent NaNs, so we skip them
        if (node.dtype != DType::FLOAT32 && node.dtype != DType::BF16)
        {
            return;
        }

        TensorView view = mem.getView(node, node.id);

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
                    std::cerr << "[NaN Detection] Found NaN in node " << node.id
                              << " (" << toString(node.opType) << (node.opType == OpType::FUSED ? " " + node.opName : "") << ")"
                              << " during \"" << context
                              << "\" at element index " << i << " (flat index " << idx << ")";
                    return;
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
                    std::cerr << "[NaN Detection] Found BF16 NaN in node " << node.id
                              << " (" << toString(node.opType) << ")"
                              << " during " << context
                              << " at element index " << i << " (flat index " << idx << ")";
                    return;
                }
            }
        }
    }

    struct RefIndexEntry
    {
        uint64_t fileOffset;
        uint64_t numElements;
        std::string opName;
    };

    class ReferenceVerifier
    {
    private:
        std::string mode;
        std::string filePath;
        std::ofstream refOutFile;
        std::ifstream refInFile;
        std::unordered_map<std::string, RefIndexEntry> refIndex;
        std::unordered_map<uint32_t, int> callCounts;
        int mismatchCount = 0;
        bool initialized = false;

    public:
        ReferenceVerifier() : mode("none") {}

        ~ReferenceVerifier()
        {
            close();
        }

        bool init(const std::string &writePath, const std::string &comparePath)
        {
            close();
            mismatchCount = 0;
            callCounts.clear();
            refIndex.clear();
            initialized = false;

            if (!writePath.empty())
            {
                mode = "write";
                filePath = writePath;

                std::filesystem::path p(filePath);
                if (p.has_parent_path() && !std::filesystem::exists(p.parent_path()))
                {
                    std::filesystem::create_directories(p.parent_path());
                }

                refOutFile.open(filePath, std::ios::binary | std::ios::trunc);
                if (!refOutFile.is_open())
                {
                    std::cerr << "Failed to open reference file for writing: " << filePath << "\n";
                    return false;
                }
                std::cout << ">>> Writing Reference Tensors to " << filePath << "\n";
                initialized = true;
            }
            else if (!comparePath.empty())
            {
                mode = "compare";
                filePath = comparePath;

                refInFile.open(filePath, std::ios::binary);
                if (!refInFile.is_open())
                {
                    std::cerr << "Failed to open reference file for reading: " << filePath << "\n";
                    return false;
                }

                while (refInFile.peek() != EOF)
                {
                    uint32_t logicalId;
                    if (!refInFile.read(reinterpret_cast<char *>(&logicalId), sizeof(logicalId)))
                        break;

                    uint32_t nameLen;
                    refInFile.read(reinterpret_cast<char *>(&nameLen), sizeof(nameLen));
                    std::string opName(nameLen, '\0');
                    if (nameLen > 0)
                        refInFile.read(&opName[0], nameLen);

                    uint64_t numElements;
                    refInFile.read(reinterpret_cast<char *>(&numElements), sizeof(numElements));

                    RefIndexEntry entry;
                    entry.opName = opName;
                    entry.numElements = numElements;
                    entry.fileOffset = refInFile.tellg();

                    int iter = callCounts[logicalId]++;
                    std::string key = std::to_string(logicalId) + "_" + std::to_string(iter);
                    refIndex[key] = entry;

                    refInFile.seekg(numElements * sizeof(float), std::ios::cur);
                }
                callCounts.clear(); // Reset for actual execution
                std::cout << ">>> Loaded Reference Index from " << filePath << " (" << refIndex.size() << " tensors)\n";

                std::cout << "\n=======================================================================================\n";
                std::cout << "Accuracy Comparison (Reference vs Optimized)\n";
                std::cout << "=======================================================================================\n";
                std::cout << std::left
                          << std::setw(15) << "Node_Iter"
                          << std::setw(25) << "OpType"
                          << std::setw(15) << "Min Diff"
                          << std::setw(15) << "Max Diff"
                          << std::setw(15) << "Avg Diff"
                          << "Details\n";
                std::cout << std::string(120, '-') << "\n";
                initialized = true;
            }
            else
            {
                mode = "none";
            }
            return true;
        }

        void close()
        {
            if (refOutFile.is_open())
                refOutFile.close();
            if (refInFile.is_open())
                refInFile.close();
        }

        std::string getMode() const { return mode; }
        int getMismatchCount() const { return mismatchCount; }

        void verify(uint32_t logicalId, const TensorNode &node, const KernelContext &ctx, const void *data, Graph *graph)
        {
            if (mode == "none" || !graph)
                return;

            const TensorView &view = ctx.outViews[0];
            std::vector<float> optData = flattenOutput(data, view.getShape(), view.strides, view.dtype);
            std::string opName = "UNKNOWN";
            if (graph->hasNode(logicalId))
            {
                auto node = graph->getNode(logicalId);
                opName = toString(node.opType);
                if (node.opType == OpType::FUSED)
                {
                    opName = "FUSED_" + node.opName;
                }
            }

            int iter = callCounts[logicalId]++;
            std::string key = std::to_string(logicalId) + "_" + std::to_string(iter);

            if (mode == "write")
            {
                uint32_t nameLen = opName.size();
                uint64_t numElems = optData.size();

                refOutFile.write(reinterpret_cast<const char *>(&logicalId), sizeof(logicalId));
                refOutFile.write(reinterpret_cast<const char *>(&nameLen), sizeof(nameLen));
                if (nameLen > 0)
                    refOutFile.write(opName.c_str(), nameLen);
                refOutFile.write(reinterpret_cast<const char *>(&numElems), sizeof(numElems));

                refOutFile.write(reinterpret_cast<const char *>(optData.data()), numElems * sizeof(float));
            }
            else if (mode == "compare")
            {
                float minDiff = std::numeric_limits<float>::max();
                float maxDiff = 0.0f;
                double sumDiff = 0.0;
                float avgDiff = 0.0f;
                bool hasNan = false;
                std::string opName = node.opType == OpType::FUSED ? "FUSED_" + node.opName : toString(node.opType);

                auto it = refIndex.find(key);
                if (it != refIndex.end())
                {
                    const auto &entry = it->second;
                    if (entry.numElements != optData.size())
                    {
                        std::cout << std::left << std::setw(15) << key
                                  << "SIZE MISMATCH: " << entry.numElements << " vs " << optData.size() << "\n";
                        mismatchCount++;
                        return;
                    }

                    std::vector<float> refData(entry.numElements);
                    refInFile.seekg(entry.fileOffset, std::ios::beg);
                    refInFile.read(reinterpret_cast<char *>(refData.data()), entry.numElements * sizeof(float));

                    for (size_t i = 0; i < refData.size(); ++i)
                    {
                        float diff = std::abs(refData[i] - optData[i]);
                        if (std::isnan(diff))
                        {
                            hasNan = true;
                            break;
                        }
                        if (diff < minDiff)
                            minDiff = diff;
                        if (diff > maxDiff)
                            maxDiff = diff;
                        sumDiff += diff;
                    }

                    if (hasNan)
                    {
                        minDiff = std::numeric_limits<float>::quiet_NaN();
                        maxDiff = std::numeric_limits<float>::quiet_NaN();
                        sumDiff = std::numeric_limits<double>::quiet_NaN();
                    }

                    avgDiff = static_cast<float>(sumDiff / refData.size());

                    if (maxDiff > 1e-2f || hasNan)
                    {
                        std::cout << "\033[1;31m";
                        mismatchCount++;
                    }
                }

                std::cout << std::left
                          << std::setw(15) << key
                          << std::setw(25) << opName.substr(0, 24)
                          << std::setw(15) << minDiff
                          << std::setw(15) << maxDiff
                          << std::setw(15) << avgDiff
                          << "id=" << node.id
                          << ", dtype=" << toString(view.dtype)
                          << ", shape=" << toString(view.getShape())
                          << ", strides=" << toString(view.strides)
                          << ", offset=" << view.baseOffset
                          << ", debugOrigin=" << node.debugOrigin
                          << "\n";

                // Print detailed information for all input-related buffers
                size_t maxInputs = std::max({ctx.inputs.size(), ctx.inViews.size(), ctx.fd.size(), ctx.cl_inputs.size()});
                if (maxInputs > 0)
                {
                    std::cout << "  Inputs:\n";
                    for (size_t i = 0; i < maxInputs; ++i)
                    {
                        std::cout << "    [" << i << "] ";

                        if (i < node.child_ids.size()) {
                            std::cout << "id=" << node.child_ids[i];
                        }

                        if (i < ctx.inputs.size())
                            std::cout << ", ptr=" << ctx.inputs[i];
                        else
                            std::cout << ", ptr=N/A";

                        if (i < ctx.inViews.size())
                        {
                            const auto &inView = ctx.inViews[i];
                            std::cout << ", dtype=" << toString(inView.dtype)
                                      << ", shape=" << toString(inView.getShape())
                                      << ", strides=" << toString(inView.strides)
                                      << ", offset=" << inView.baseOffset;
                        }
                        if (i < ctx.fd.size())
                        {
                            std::cout << ", fd=" << ctx.fd[i];
                        }
                        if (i < ctx.cl_inputs.size())
                        {
                            std::cout << ", cl_mem=" << ctx.cl_inputs[i];
                        }
                        std::cout << "\n";
                    }
                }

                // Print detailed information for all output-related buffers
                size_t maxOutputs = std::max({ctx.outputs.size(), ctx.outViews.size(), ctx.cl_outputs.size()});
                if (maxOutputs > 0)
                {
                    std::cout << "  Outputs:\n";
                    for (size_t i = 0; i < maxOutputs; ++i)
                    {
                        std::cout << "    [" << i << "] ";
                        if (i < ctx.outputs.size())
                            std::cout << "ptr=" << ctx.outputs[i];
                        else
                            std::cout << "ptr=N/A";

                        if (i < ctx.outViews.size())
                        {
                            const auto &outView = ctx.outViews[i];
                            std::cout << ", dtype=" << toString(outView.dtype)
                                      << ", shape=" << toString(outView.getShape())
                                      << ", strides=" << toString(outView.strides)
                                      << ", offset=" << outView.baseOffset;
                        }
                        if (i < ctx.cl_outputs.size())
                        {
                            std::cout << ", cl_mem=" << ctx.cl_outputs[i];
                        }
                        std::cout << "\n";
                    }
                }

                if (maxDiff > 1e-2f || hasNan)
                {
                    std::cout << "\033[0m";
                }
            }
        }

        void printSummary() const
        {
            if (mode == "compare")
            {
                std::cout << "=======================================================================================\n";
                if (mismatchCount > 0)
                {
                    std::cout << "Found " << mismatchCount << " nodes with high deviation (>1e-2) or NaNs.\n";
                }
                else
                {
                    std::cout << "All nodes matched perfectly or within acceptable precision limits.\n";
                }
            }
        }
    };
}