// tensor_graphs_cpp/core/plan/search_delegate.hpp
#pragma once
#include "core/graph.hpp"
#include "core/types.hpp"
#include <cstdint>
#include <vector>

struct ActionFeatureExtractDispatch
{
    float cost;
    uint64_t size; // n elements * dtype size
    MemSpace mem_space;
    std::vector<uint32_t> engine_idxs;
    Graph graph; // refFactory or single op graph
};

struct ActionFeatureMalloc
{
    uint64_t size = 0;  // bytes
    uint32_t start = 0; // birth time (idx into dispatch order of first eclass that uses this)
    uint32_t end = 0;   // death time (idx into dispatch order of last eclass that uses this)
};

struct ActionFeatureBufferize
{
    float is_new_buffer;
    uint64_t size;
    uint64_t parent_size;
    float parent_birth_time;
};

class SearchDelegate
{
  public:
    virtual ~SearchDelegate() = default;

    virtual void push_state()
    {
    }
    virtual void pop_state()
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
};