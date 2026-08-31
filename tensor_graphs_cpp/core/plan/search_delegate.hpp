// tensor_graphs_cpp/core/plan/search_delegate.hpp
#pragma once
#include "core/graph.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

struct ActionFeatureCache
{
    float is_cached = 0.0f; // 1.0f if cached, 0.0f if not cached
    uint64_t size = 0;      // bytes
    float num_users = 0.0f;
    uint32_t logical_id = 0;
    MemSpace mem_space;
    uint64_t mem_cap = 0;
};

struct ActionFeatureExtractDispatch
{
    float cost;
    float min_dp_cp_cost = 0.0f;
    float dp_cost = 0.0f;
    float rev_cp_cost = 0.0f;
    float dp_mem = 0.0f; // Sethi-Ullman peak memory estimate in bytes
    uint64_t size;       // n elements * dtype size
    std::vector<uint32_t> engine_idxs;
    uint32_t num_nodes = 0;
    uint32_t num_edges = 0;
    MemSpace mem_space;
    uint64_t mem_cap = 0;
};

struct ActionFeatureMalloc
{
    uint64_t size = 0;  // bytes
    uint32_t start = 0; // birth time (idx into dispatch order of first eclass that uses this)
    uint32_t end = 0;   // death time (idx into dispatch order of last eclass that uses this)
    MemSpace mem_space;
    uint64_t mem_cap = 0;
};

struct ActionFeatureBufferize
{
    float is_new_buffer;
    uint64_t size;
    uint64_t parent_size;
    float parent_birth_time;
    MemSpace mem_space;
    uint64_t mem_cap = 0;
};

struct ActionFeatureFrontier
{
    uint32_t eclass_id = 0;
    uint32_t num_enodes = 0;
    float min_dp_cp_cost = 0.0f;
    float min_dp_cost = 0.0f;
    float min_dp_mem = 0.0f; // Minimum Sethi-Ullman peak memory across enodes
    uint64_t size = 0;       // bytes
    DType dtype = DType::FLOAT32;
    MemSpace mem_space;
    uint64_t mem_cap = 0;
};

class SearchDelegate
{
  public:
    virtual ~SearchDelegate() = default;

    virtual bool fast_fail() const
    {
        return false;
    }

    virtual void push_state()
    {
    }
    virtual void pop_state()
    {
    }
    virtual void on_leaf_evaluated(float cost)
    {
    }

    virtual void init_cache_graph(const std::vector<float> &node_features, const std::vector<uint32_t> &edge_src,
                                  const std::vector<uint32_t> &edge_dst)
    {
    }

    virtual void init_egraph(const std::vector<float> &node_features, const std::vector<uint32_t> &edge_src,
                             const std::vector<uint32_t> &edge_dst)
    {
    }

    virtual void init_dispatch_graph(const std::vector<float> &node_features, const std::vector<uint32_t> &edge_src,
                                     const std::vector<uint32_t> &edge_dst)
    {
    }

    virtual void init_bufferize_graph(const std::vector<float> &node_features, const std::vector<uint32_t> &edge_src,
                                      const std::vector<uint32_t> &edge_dst)
    {
    }

    virtual void init_malloc_graph(const std::vector<float> &node_features, const std::vector<uint32_t> &edge_src,
                                   const std::vector<uint32_t> &edge_dst)
    {
    }

    virtual std::vector<uint32_t> order_cache(const std::vector<ActionFeatureCache> &choices)
    {
        std::vector<uint32_t> res(choices.size());
        for (uint32_t i = 0; i < choices.size(); ++i)
            res[i] = i;
        return res;
    }

    virtual std::vector<uint32_t> order_enodes(const std::vector<ActionFeatureExtractDispatch> &enodes)
    {
        std::vector<uint32_t> res(enodes.size());
        for (uint32_t i = 0; i < enodes.size(); ++i)
            res[i] = i;
        return res;
    }

    virtual std::vector<uint32_t> order_dispatch(const std::vector<ActionFeatureExtractDispatch> &ready_nodes)
    {
        std::vector<uint32_t> res(ready_nodes.size());
        for (uint32_t i = 0; i < ready_nodes.size(); ++i)
            res[i] = i;
        return res;
    }

    virtual std::vector<uint32_t> order_bufferize(const std::vector<ActionFeatureBufferize> &choices)
    {
        std::vector<uint32_t> res(choices.size());
        for (uint32_t i = 0; i < choices.size(); ++i)
            res[i] = i;
        return res;
    }

    virtual std::vector<uint32_t> order_malloc(const std::vector<ActionFeatureMalloc> &avail_buffers)
    {
        std::vector<uint32_t> res(avail_buffers.size());
        for (uint32_t i = 0; i < avail_buffers.size(); ++i)
            res[i] = i;
        return res;
    }

    virtual std::vector<uint32_t> order_frontier(const std::vector<ActionFeatureFrontier> &frontier)
    {
        std::vector<uint32_t> res(frontier.size());
        for (uint32_t i = 0; i < frontier.size(); ++i)
            res[i] = i;
        return res;
    }
};

class HeuristicSearchDelegate : public SearchDelegate
{
  public:
    std::vector<uint32_t> order_cache(const std::vector<ActionFeatureCache> &choices) override
    {
        std::vector<uint32_t> res(choices.size());
        std::iota(res.begin(), res.end(), 0);
        return res;
    }

    std::vector<uint32_t> order_enodes(const std::vector<ActionFeatureExtractDispatch> &enodes) override
    {
        std::vector<uint32_t> res(enodes.size());
        std::iota(res.begin(), res.end(), 0);
        std::stable_sort(res.begin(), res.end(), [&](uint32_t a, uint32_t b) {
            // 1. Prioritize options with lower Sethi-Ullman peak memory usage to fit in capacity sooner
            if (enodes[a].dp_mem != enodes[b].dp_mem)
                return enodes[a].dp_mem < enodes[b].dp_mem;
            // 2. Subtree workload cost
            if (enodes[a].dp_cost != enodes[b].dp_cost)
                return enodes[a].dp_cost < enodes[b].dp_cost;
            // 3. Direct execution cost
            return enodes[a].cost < enodes[b].cost;
        });
        return res;
    }

    std::vector<uint32_t> order_dispatch(const std::vector<ActionFeatureExtractDispatch> &ready_nodes) override
    {
        LOG(DEBUG) << "order_dispatch start";
        std::vector<uint32_t> res(ready_nodes.size());
        std::iota(res.begin(), res.end(), 0);
        std::stable_sort(res.begin(), res.end(), [&](uint32_t a, uint32_t b) {
            // 1. Critical-path first (depth-first: consume and free tensors ASAP)
            if (ready_nodes[a].min_dp_cp_cost != ready_nodes[b].min_dp_cp_cost)
                return ready_nodes[a].min_dp_cp_cost > ready_nodes[b].min_dp_cp_cost;

            // 2. Reverse Critical Path tie-breaker (Distance to Output)
            if (ready_nodes[a].rev_cp_cost != ready_nodes[b].rev_cp_cost)
                return ready_nodes[a].rev_cp_cost > ready_nodes[b].rev_cp_cost;

            // 3. Sethi-Ullman peak memory demand
            if (ready_nodes[a].dp_mem != ready_nodes[b].dp_mem)
                return ready_nodes[a].dp_mem < ready_nodes[b].dp_mem;

            // 4. Memory footprint tie-breaker (smaller footprint first)
            if (ready_nodes[a].size != ready_nodes[b].size)
                return ready_nodes[a].size < ready_nodes[b].size;

            return ready_nodes[a].cost < ready_nodes[b].cost;
        });
        LOG(DEBUG) << "order_dispatch end";
        return res;
    }

    std::vector<uint32_t> order_bufferize(const std::vector<ActionFeatureBufferize> &choices) override
    {
        std::vector<uint32_t> res(choices.size());
        std::iota(res.begin(), res.end(), 0);
        std::stable_sort(res.begin(), res.end(),
                         [&](uint32_t a, uint32_t b) { return choices[a].is_new_buffer < choices[b].is_new_buffer; });
        return res;
    }

    std::vector<uint32_t> order_malloc(const std::vector<ActionFeatureMalloc> &avail_buffers) override
    {
        std::vector<uint32_t> res(avail_buffers.size());
        std::iota(res.begin(), res.end(), 0);
        std::stable_sort(res.begin(), res.end(),
                         [&](uint32_t a, uint32_t b) { return avail_buffers[a].size > avail_buffers[b].size; });
        return res;
    }

    std::vector<uint32_t> order_frontier(const std::vector<ActionFeatureFrontier> &frontier) override
    {
        std::vector<uint32_t> res(frontier.size());
        std::iota(res.begin(), res.end(), 0);
        std::stable_sort(res.begin(), res.end(), [&](uint32_t a, uint32_t b) {
            // 1. Minimum Remaining Values (MRV): fewest candidate ENodes first
            if (frontier[a].num_enodes != frontier[b].num_enodes)
                return frontier[a].num_enodes < frontier[b].num_enodes;

            // 2. Critical Path First: highest critical path cost first (Fail-First)
            if (frontier[a].min_dp_cp_cost != frontier[b].min_dp_cp_cost)
                return frontier[a].min_dp_cp_cost > frontier[b].min_dp_cp_cost;

            // 3. Sethi-Ullman memory requirement (Fail-First: decide heavier memory subtrees first)
            if (frontier[a].min_dp_mem != frontier[b].min_dp_mem)
                return frontier[a].min_dp_mem > frontier[b].min_dp_mem;

            // 4. Tie-breaker: largest buffer size first
            return frontier[a].size > frontier[b].size;
        });
        return res;
    }
};