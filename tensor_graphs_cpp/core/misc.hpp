#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"

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
            if (pid < graph.nodes.size())
            {
                const auto &parent = graph.nodes[pid];
                ss << "\n"
                   << prefix << "    [" << i << "] Parent ID " << pid
                   << " (" << toString(parent.opType) << ", " << toString(parent.shape) << ")";
            }
            else
            {
                ss << "\n"
                   << prefix << "    [" << i << "] Parent ID " << pid << " [OUT OF BOUNDS]";
            }
        }
    }
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