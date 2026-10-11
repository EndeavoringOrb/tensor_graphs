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
    std::pair<VarId, Domain> delta;

    float lower_bound = 0.0f;
    float priority = 0.0f;
    uint32_t depth = 0;
    size_t trail_marker = 0;

    SearchNode() = default;
    SearchNode(uint32_t id, uint32_t parent_id, std::pair<VarId, Domain> delta, float lower_bound,
               float priority, uint32_t depth, size_t trail_marker = 0)
        : id(id), parent_id(parent_id), delta(std::move(delta)), lower_bound(lower_bound), priority(priority),
          depth(depth), trail_marker(trail_marker)
    {
    }
};

struct SearchNodeCompare
{
    bool has_incumbent = false;

    SearchNodeCompare() = default;
    explicit SearchNodeCompare(bool has_incumbent) : has_incumbent(has_incumbent) {}

    bool operator()(const std::shared_ptr<SearchNode> &a, const std::shared_ptr<SearchNode> &b) const
    {
        if (!has_incumbent)
        {
            // Dive phase: deepest first. If priorities match, break tie by lower priority (e.g. Left over Right).
            if (a->depth != b->depth)
                return a->depth < b->depth;
            return a->priority > b->priority;
        }
        else
        {
            // Optimization phase: smaller priority comes first. If priorities match, deeper node comes first.
            if (a->priority != b->priority)
                return a->priority > b->priority;
            return a->depth < b->depth;
        }
    }
};

} // namespace plan
