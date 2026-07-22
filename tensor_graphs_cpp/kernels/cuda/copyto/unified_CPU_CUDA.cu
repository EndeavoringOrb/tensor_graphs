#ifdef USE_CUDA
#include "core/hardware.hpp"
#include "core/kernels.hpp"

inline bool matchUnifiedCopy(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    auto &caps = HardwareCaps::get();
    // Only valid on integrated chips (DGX Spark)
    if (!caps.has_unified_memory)
        return false;

    return inputs[0].getShape() == output.getShape() && inputs[0].strides == output.strides && inputs[0].dtype == output.dtype;
}

inline void inferViewUnified(TensorNode &node, const std::vector<TensorNode> &inputs, const Graph &graph)
{
    node.strides = inputs[0].strides;
}

inline LogicalId refFactoryUnifiedCopy(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // This kernel matches a cross-backend COPY_TO. We use CUDA as the target backend
    // for the reference pattern.
    return graph._copyto(inputs[0]);
}

// Registered as a View to ensure the Planner always prefers this over a copy on unified memory systems
REGISTER_KERNEL_VIEW("UnifiedCopyTo_CPU_CUDA", 1, 1, matchUnifiedCopy, refFactoryUnifiedCopy, inferViewUnified, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
    {DType::ANY},
    {{1024}},
    {false},
    {{MemSpace(1, HandleType::CPP)}});
#endif