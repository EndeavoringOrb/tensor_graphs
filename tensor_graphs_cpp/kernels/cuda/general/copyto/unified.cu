#ifdef USE_CUDA
#include "core/hardware.hpp"
#include "core/kernels.hpp"

inline bool matchUnifiedCopy(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    auto &caps = HardwareCaps::get();
    // Only valid on integrated chips (DGX Spark)
    if (!caps.has_unified_memory)
        return false;

    // Only for cross-backend movement (CPU <-> CUDA)
    bool isCross = (inputs[0].backend != output.backend);
    return isCross && inputs[0].getShape() == output.getShape() && inputs[0].strides == output.strides;
}

inline void inferViewUnified(TensorNode &node, const std::vector<TensorNode> &inputs, const Graph &graph)
{
    node.strides = inputs[0].strides;
    node.viewOffset = inputs[0].viewOffset;
}

// Registered as a Reference View to ensure the Planner always prefers this over a copy
REGISTER_REF_KERNEL_VIEW(
    OpType::COPY_TO, 1, matchUnifiedCopy, inferViewUnified,
    {Backend::CPU, Backend::CUDA},
    {DType::ANY},
    {{1024}},
    {false},
    {{Backend::CPU, Backend::CUDA}});
#endif