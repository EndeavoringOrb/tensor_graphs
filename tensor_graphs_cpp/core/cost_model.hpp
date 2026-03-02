#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"

struct CostModel
{
    float estimateCost(const TensorNode &node, const Graph &graph, uint32_t kernelId) {
        // TODO: implement
        // TODO: allow specification of exact kernel based on some kernel entry index. Use kernel registry entry index, and when a new kernel is added we invalidate the .jsonl benchmark data
        return 0.0f;
    }
};