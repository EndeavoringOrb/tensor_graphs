#include "core/types.hpp"
#include "core/graph.hpp"

/**
 * Helper to format a TensorNode's metadata and its parents' metadata into a string.
 * This encapsulates the logging logic used in estimateCost and interpolate.
 */
inline std::string nodeToString(const TensorNode& node, const Graph& graph, const std::string& prefix = "") {
    std::stringstream ss;
    ss << prefix << "Node " << node.id << " [" << toString(node.opType) << "]\n"
       << prefix << "  DType:      " << toString(node.dtype) << "\n"
       << prefix << "  Shape:      " << toString(node.shape) << "\n"
       << prefix << "  Backend:    " << node.backend << "\n"
       << prefix << "  Contiguous: " << (node.view.isContiguous() ? "true" : "false") << "\n"
       << prefix << "  Parents (" << node.parentIds.size() << "):";
    
    if (node.parentIds.empty()) {
        ss << " None";
    } else {
        for (size_t i = 0; i < node.parentIds.size(); ++i) {
            uint32_t pid = node.parentIds[i];
            if (pid < graph.nodes.size()) {
                const auto& parent = graph.nodes[pid];
                ss << "\n" << prefix << "    [" << i << "] Parent ID " << pid 
                   << " (" << toString(parent.opType) << ", " << toString(parent.shape) << ")";
            } else {
                ss << "\n" << prefix << "    [" << i << "] Parent ID " << pid << " [OUT OF BOUNDS]";
            }
        }
    }
    return ss.str();
}

/**
 * Convenience wrapper to print node info directly to std::cout
 */
inline void printNode(const TensorNode& node, const Graph& graph, const std::string& label = "NODE INFO") {
    std::cout << "--- " << label << " ---\n" 
              << nodeToString(node, graph) 
              << "\n-----------------------" << std::endl;
}