// tensor_graphs_cpp/core/plan/search_node.hpp
#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <limits>
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
    // NaN until measured; lazy nodes use the parent box restricted by delta.
    double log2_volume = std::numeric_limits<double>::quiet_NaN();
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
    // Proof phase: the open node with the largest box comes first (its
    // stored log2 volume, or the shallowest node when volumes are not
    // measured), so the regions that dominate the frontier are refreshed and
    // split first.
    bool largest_box_first = false;

    SearchNodeCompare() = default;
    explicit SearchNodeCompare(bool has_incumbent) : has_incumbent(has_incumbent) {}

    bool operator()(const std::shared_ptr<SearchNode> &a, const std::shared_ptr<SearchNode> &b) const
    {
        if (largest_box_first)
        {
            const bool measured = !std::isnan(a->log2_volume) && !std::isnan(b->log2_volume);
            if (measured && a->log2_volume != b->log2_volume)
                return a->log2_volume < b->log2_volume;
            if (a->depth != b->depth)
                return a->depth > b->depth;
            return a->priority > b->priority;
        }
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
