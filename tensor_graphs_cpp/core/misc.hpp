#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/timer.hpp"

inline std::string toString(const std::vector<uint32_t> &shape)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

inline std::string toString(const std::vector<int32_t> &shape)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

inline std::string toString(const std::vector<uint64_t> &shape)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

inline std::string toString(const std::vector<int64_t> &shape)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

inline std::string toString(const TensorNode &node, const std::string &prefix = "")
{
    std::stringstream ss;
    ss << prefix << "Node " << node.id << " [" << toString(node.opType);
    if (node.opType == OpType::FUSED)
    {
        ss << " (" << node.opName << ")";
    }
    ss << "]\n"
       << prefix << "  DType:        " << toString(node.dtype) << "\n"
       << prefix << "  Shape:        " << toString(node.shape) << "\n"
       << prefix << "  Backend:      " << node.backend << "\n"
       << prefix << "  Contiguous:   " << (node.view.isContiguous() ? "true" : "false") << "\n"
       << prefix << "  Storage Type: " << toString(node.storageType);
    return ss.str();
}

/**
 * Helper to format a TensorNode's metadata and its parents' metadata into a string.
 * This encapsulates the logging logic used in estimateCost and interpolate.
 */
inline std::string toString(const TensorNode &node, const Graph &graph, const std::string &prefix = "")
{
    std::stringstream ss;
    ss << prefix << "Node " << node.id << " [" << toString(node.opType) << "]\n"
       << prefix << "  DType:      " << toString(node.dtype) << "\n"
       << prefix << "  Shape:      " << toString(node.shape) << "\n"
       << prefix << "  Backend:    " << node.backend << "\n"
       << prefix << "  Contiguous: " << (node.view.isContiguous() ? "true" : "false") << "\n"
       << prefix << "  Parents (" << node.parentIds.size() << "):";

    if (node.parentIds.empty())
    {
        ss << " None";
    }
    else
    {
        for (size_t i = 0; i < node.parentIds.size(); ++i)
        {
            uint32_t pid = node.parentIds[i];
            if (graph.hasNode(pid))
            {
                const auto &parent = graph.getNode(pid);
                ss << "\n"
                   << prefix << "    [" << i << "] Parent ID " << pid
                   << "\n"
                   << toString(parent, (std::string) "    ");
            }
            else
            {
                ss << "\n"
                   << prefix << "[" << i << "] Parent ID " << pid << " [OUT OF BOUNDS/NOT FOUND]";
            }
        }
    }
    return ss.str();
}

/**
 * Convenience wrapper to print node info directly to std::cout
 */
inline void printNode(const TensorNode &node, const Graph &graph, const std::string &label = "NODE INFO")
{
    std::cout << "--- " << label << " ---\n"
              << toString(node, graph)
              << "\n-----------------------" << std::endl;
}

std::string toString(const Region &reg)
{
    std::stringstream ss;
    ss << "[";
    for (uint32_t i = 0; i < reg.region.size(); i++)
    {
        if (i > 0)
        {
            ss << ", ";
        }
        const Dim &dim = reg.region[i];
        ss << "(" << dim.start << ", " << dim.stop << ")";
    }
    return ss.str();
}

std::string toString(const OpInstruction &inst)
{
    std::stringstream ss;
    ss << "OpInstruction\n"
       << "  Node ID: " << inst.nodeId << "\n"
       << "  Full Kernel ID: " << inst.fullKernelId << "\n"
       << "  Input Node IDs: " << toString(inst.inputNodeIds) << "\n"
       << "  Inplace Input Index: " << inst.inplaceInputIndex << "\n"
       << "  Backend: " << inst.backend << "\n";
    return ss.str();
}

// calculate binomial coefficient "n choose k"
uint64_t binom(uint64_t n, uint64_t k)
{
    if (k > n)
        return 0;
    k = std::min(k, n - k); // symmetry

    uint64_t result = 1;

    for (uint64_t i = 1; i <= k; ++i)
    {
        uint64_t num = n - k + i; // grows
        uint64_t den = i;

        uint64_t g = std::gcd(num, den);
        num /= g;
        den /= g;

        // reduce result with denominator before multiplying
        g = std::gcd(result, den);
        result /= g;
        den /= g;

        // now safe to multiply
        result *= num;
        result /= den;
    }

    return result;
}