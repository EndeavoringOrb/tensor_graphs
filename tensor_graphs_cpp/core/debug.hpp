// tensor_graphs_cpp/core/debug.hpp
#pragma once
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/memory.hpp"
#include "core/misc.hpp"
#include "core/shapes.hpp"
#include "core/types.hpp"

namespace Debug
{
using Callback =
    std::function<void(LogicalId logicalId, std::string &kernel_name, const KernelContext &ctx, const void *data)>;

template <typename ErrorHandler>
inline void _checkValues(const std::vector<const void *> &ptrs, const std::vector<TensorView> &views,
                         const std::string &context, ErrorHandler &&reportError)
{
    for (uint64_t i = 0; i < ptrs.size(); ++i)
    {
        if (!ptrs[i])
            continue;
        uint64_t numElements = countElements(views[i]);

        const void *host_ptr = ptrs[i];
#ifdef USE_CUDA
        std::vector<uint8_t> temp_host_data;
        cudaPointerAttributes attrs;
        if (cudaPointerGetAttributes(&attrs, ptrs[i]) == cudaSuccess && attrs.type == cudaMemoryTypeDevice)
        {
            uint64_t sizeBytes = getRequiredBufferSize(views[i]) * getDTypeSize(views[i].dtype);
            temp_host_data.resize(sizeBytes);
            cudaMemcpy(temp_host_data.data(), ptrs[i], sizeBytes, cudaMemcpyDeviceToHost);
            host_ptr = temp_host_data.data();
        }
#endif

        if (views[i].dtype == DType::FLOAT32)
        {
            const float *data = static_cast<const float *>(host_ptr);
            for (uint64_t j = 0; j < numElements; ++j)
            {
                uint64_t idx = getStridedIndex(j, views[i].getShape(), views[i].strides);
                if (std::isnan(data[idx]) || std::isinf(data[idx]))
                {
                    reportError(i, j, idx);
                }
            }
        }
        else if (views[i].dtype == DType::BF16)
        {
            const uint16_t *data = static_cast<const uint16_t *>(host_ptr);
            for (uint64_t j = 0; j < numElements; ++j)
            {
                uint64_t idx = getStridedIndex(j, views[i].getShape(), views[i].strides);
                uint16_t bits = data[idx];
                bool is_inf = (bits & 0x7F80) == 0x7F80 && (bits & 0x007F) == 0;
                bool is_nan = ((bits & 0x7F80) == 0x7F80) && ((bits & 0x007F) != 0);
                if (is_inf || is_nan)
                {
                    reportError(i, j, idx);
                }
            }
        }
    }
}

inline void checkValues(const std::vector<const void *> &ptrs, const std::vector<TensorView> &views,
                        const std::string &context)
{
    _checkValues(ptrs, views, context, [&](uint64_t bufferIdx, uint64_t elemIdx, uint64_t flatIdx) {
        std::string msg = "[NaN/Inf Detection] Found NaN/Inf during \"" + context + "\" in buffer " +
                          std::to_string(bufferIdx) + " at element " + std::to_string(elemIdx) + " (flat index " +
                          std::to_string(flatIdx) + ")";
        std::cerr << "\n" << msg << "\n";
        Error::throw_err(msg);
    });
}

inline void checkValues(const std::vector<const void *> &out_ptrs, const std::vector<TensorView> &out_views,
                        const std::vector<const void *> &in_ptrs, const std::vector<TensorView> &in_views,
                        const KernelEntry &kernel, const std::string &context)
{
    _checkValues(out_ptrs, out_views, context, [&](uint64_t bufferIdx, uint64_t elemIdx, uint64_t flatIdx) {
        std::string msg = "[NaN/Inf Detection] Found NaN/Inf during \"" + context + "\" in buffer " +
                          std::to_string(bufferIdx) + " at element " + std::to_string(elemIdx) + " (flat index " +
                          std::to_string(flatIdx) + ")";

        if (isElementwise(kernel.opType))
        {
            ShapePropagator prop;
            std::vector<uint32_t> coords = coordsFromFlatIndex(elemIdx, out_views[bufferIdx].getShape());
            Region outReg;
            for (uint32_t c : coords)
            {
                outReg.region.push_back({c, c + 1});
            }
            std::vector<std::vector<Region>> inRegions = prop.backwardElementwise(in_ptrs.size(), {outReg});

            msg += "\n  Relevant Input Values:";
            for (size_t in_i = 0; in_i < in_ptrs.size(); ++in_i)
            {
                if (!in_ptrs[in_i] || in_i >= inRegions.size())
                    continue;

                const void *in_host_ptr = in_ptrs[in_i];
#ifdef USE_CUDA
                std::vector<uint8_t> temp_in_host_data;
                cudaPointerAttributes in_attrs;
                if (cudaPointerGetAttributes(&in_attrs, in_ptrs[in_i]) == cudaSuccess &&
                    in_attrs.type == cudaMemoryTypeDevice)
                {
                    uint64_t inSizeBytes = getRequiredBufferSize(in_views[in_i]) * getDTypeSize(in_views[in_i].dtype);
                    temp_in_host_data.resize(inSizeBytes);
                    cudaMemcpy(temp_in_host_data.data(), in_ptrs[in_i], inSizeBytes, cudaMemcpyDeviceToHost);
                    in_host_ptr = temp_in_host_data.data();
                }
#endif

                msg += "\n    Input " + std::to_string(in_i) + ":";
                const auto &regions = inRegions[in_i];
                int printCount = 0;
                for (const Region &r : regions)
                {
                    std::vector<uint32_t> rShape;
                    for (const Dim &d : r.region)
                        rShape.push_back(d.stop - d.start);
                    uint64_t rElems = countElements(rShape);
                    for (uint64_t localFlat = 0; localFlat < rElems && printCount < 50; ++localFlat, ++printCount)
                    {
                        auto localCoords = coordsFromFlatIndex(localFlat, rShape);
                        std::vector<uint32_t> absCoords = localCoords;
                        for (size_t d = 0; d < r.region.size(); ++d)
                        {
                            absCoords[d] += r.region[d].start;
                        }
                        uint64_t inFlatIdx = flatIndexFromCoords(absCoords, in_views[in_i].getShape());
                        uint64_t inStridedIdx =
                            getStridedIndex(inFlatIdx, in_views[in_i].getShape(), in_views[in_i].strides);

                        msg += "\n      " + toString(absCoords) + " = ";
                        if (in_views[in_i].dtype == DType::FLOAT32)
                        {
                            msg += std::to_string(static_cast<const float *>(in_host_ptr)[inStridedIdx]);
                        }
                        else if (in_views[in_i].dtype == DType::INT32)
                        {
                            msg += std::to_string(static_cast<const int32_t *>(in_host_ptr)[inStridedIdx]);
                        }
                        else if (in_views[in_i].dtype == DType::BF16)
                        {
                            uint16_t bits = static_cast<const uint16_t *>(in_host_ptr)[inStridedIdx];
                            uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
                            float val;
                            std::memcpy(&val, &f32_bits, 4);
                            msg += std::to_string(val);
                        }
                        else
                        {
                            msg += "?";
                        }
                    }
                    if (printCount >= 50)
                    {
                        msg += "\n      ... (truncated)";
                        break;
                    }
                }
            }
        }

        std::cerr << "\n" << msg << "\n";
        Error::throw_err(msg);
    });
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
    std::unordered_map<LogicalId, int> callCounts;
    int mismatchCount = 0;
    bool initialized = false;

  public:
    ReferenceVerifier() : mode("none")
    {
    }

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
                LogicalId logicalId;
                if (!refInFile.read(reinterpret_cast<char *>(&logicalId.value), sizeof(logicalId.value)))
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
                std::string key = toString(logicalId) + "_" + std::to_string(iter);
                refIndex[key] = entry;

                refInFile.seekg(numElements * sizeof(float), std::ios::cur);
            }
            callCounts.clear(); // Reset for actual execution
            std::cout << ">>> Loaded Reference Index from " << filePath << " (" << refIndex.size() << " tensors)\n";

            std::cout << "\n========================================================="
                         "==============================\n";
            std::cout << "Accuracy Comparison (Reference vs Optimized)\n";
            std::cout << "==========================================================="
                         "============================\n";
            std::cout << std::left << std::setw(15) << "Node_Iter" << std::setw(25) << "OpType" << std::setw(15)
                      << "Min Diff" << std::setw(15) << "Max Diff" << std::setw(15) << "Avg Diff"
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

    std::string getMode() const
    {
        return mode;
    }
    int getMismatchCount() const
    {
        return mismatchCount;
    }

    void verify(LogicalId logicalId, std::string &kernel_name, const KernelContext &ctx, const void *data, Graph *graph)
    {
        if (mode == "none" || !graph || logicalId == LogicalId())
            return;

        const TensorView &view = ctx.outViews[0];
        std::vector<float> optData = flattenOutput(data, view.getShape(), view.strides, view.dtype);
        std::string opName = kernel_name;
        TensorNode node;
        if (graph->hasNode(logicalId))
        {
            node = graph->getNode(logicalId);
        }

        int iter = callCounts[logicalId]++;
        std::string key = toString(logicalId) + "_" + std::to_string(iter);

        if (mode == "write")
        {
            uint32_t nameLen = opName.size();
            uint64_t numElems = optData.size();

            refOutFile.write(reinterpret_cast<const char *>(&logicalId.value), sizeof(logicalId.value));
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

            auto it = refIndex.find(key);
            if (it == refIndex.end())
            {
                return;
            }
            const auto &entry = it->second;
            if (entry.numElements != optData.size())
            {
                std::cout << std::left << std::setw(15) << key << "SIZE MISMATCH: " << entry.numElements << " vs "
                          << optData.size() << "\n";
                mismatchCount++;
                return;
            }

            std::vector<float> refData(entry.numElements);
            refInFile.seekg(entry.fileOffset, std::ios::beg);
            refInFile.read(reinterpret_cast<char *>(refData.data()), entry.numElements * sizeof(float));

            for (uint64_t i = 0; i < refData.size(); ++i)
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

            std::cout << std::left << std::setw(15) << key << std::setw(25) << opName.substr(0, 32) << std::setw(15)
                      << minDiff << std::setw(15) << maxDiff << std::setw(15) << avgDiff << "id=" << node.id
                      << ", dtype=" << toString(view.dtype) << ", shape=" << toString(view.getShape())
                      << ", strides=" << toString(view.strides) << ", offset=" << view.offset
                      << ", debugOrigin=" << node.debugOrigin << "\n";

            // Print detailed information for all input-related buffers
            uint64_t maxInputs = std::max({ctx.inputs.size(), ctx.inViews.size(), ctx.fd.size(), ctx.cl_inputs.size()});
            if (maxInputs > 0)
            {
                std::cout << "  Inputs:\n";
                for (uint64_t i = 0; i < maxInputs; ++i)
                {
                    std::cout << "    [" << i << "] ";

                    if (i < node.child_ids.size())
                    {
                        std::cout << "id=" << node.child_ids[i];
                    }

                    if (i < ctx.inputs.size())
                        std::cout << ", ptr=" << ctx.inputs[i];
                    else
                        std::cout << ", ptr=N/A";

                    if (i < ctx.inViews.size())
                    {
                        const auto &inView = ctx.inViews[i];
                        std::cout << ", dtype=" << toString(inView.dtype) << ", shape=" << toString(inView.getShape())
                                  << ", strides=" << toString(inView.strides) << ", offset=" << inView.offset;
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
            uint64_t maxOutputs = std::max({ctx.outputs.size(), ctx.outViews.size(), ctx.cl_outputs.size()});
            if (maxOutputs > 0)
            {
                std::cout << "  Outputs:\n";
                for (uint64_t i = 0; i < maxOutputs; ++i)
                {
                    std::cout << "    [" << i << "] ";
                    if (i < ctx.outputs.size())
                        std::cout << "ptr=" << ctx.outputs[i];
                    else
                        std::cout << "ptr=N/A";

                    if (i < ctx.outViews.size())
                    {
                        const auto &outView = ctx.outViews[i];
                        std::cout << ", dtype=" << toString(outView.dtype) << ", shape=" << toString(outView.getShape())
                                  << ", strides=" << toString(outView.strides) << ", offset=" << outView.offset;
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
            std::cout << "==========================================================="
                         "============================\n";
            if (mismatchCount > 0)
            {
                std::cout << "Found " << mismatchCount << " nodes with high deviation (>1e-2) or NaNs.\n";
            }
            else
            {
                std::cout << "All nodes matched perfectly or within acceptable "
                             "precision limits.\n";
            }
        }
    }
};
} // namespace Debug