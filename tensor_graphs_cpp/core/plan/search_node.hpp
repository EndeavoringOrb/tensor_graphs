// tensor_graphs_cpp/core/plan/search_node.hpp
#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/plan/domain.hpp"
#include "core/plan/search_state.hpp"

namespace plan
{

struct SearchNode
{
    uint32_t id = 0;
    uint32_t parent_id = UINT32_MAX;
    std::vector<std::pair<VarId, Domain>> delta;

    float lower_bound = 0.0f;
    float priority = 0.0f;
    uint32_t depth = 0;
    size_t trail_marker = 0;

    SearchNode() = default;
    SearchNode(uint32_t id, uint32_t parent_id, std::vector<std::pair<VarId, Domain>> delta, float lower_bound,
               float priority, uint32_t depth, size_t trail_marker = 0)
        : id(id), parent_id(parent_id), delta(std::move(delta)), lower_bound(lower_bound), priority(priority),
          depth(depth), trail_marker(trail_marker)
    {
    }
};

struct SearchNodeCompare
{
    bool operator()(const std::shared_ptr<SearchNode> &a, const std::shared_ptr<SearchNode> &b) const
    {
        // Min-heap: smaller priority comes first. If priorities match, deeper node comes first (dive).
        if (a->priority != b->priority)
            return a->priority > b->priority;
        return a->depth < b->depth;
    }
};

} // namespace plan
