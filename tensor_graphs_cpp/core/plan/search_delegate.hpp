#pragma once
#include <cstdint>
#include <vector>

struct ActionFeature
{
    uint32_t id;
    float cost;
    float size;
    uint32_t op_type;
};

class SearchDelegate
{
public:
    virtual ~SearchDelegate() = default;

    virtual void push_state() {}
    virtual void pop_state() {}

    virtual std::vector<uint32_t> order_enodes(uint32_t eclass_id, const std::vector<ActionFeature> &enodes)
    {
        std::vector<uint32_t> res(enodes.size());
        for (uint32_t i = 0; i < enodes.size(); ++i)
            res[i] = i;
        return res;
    }

    virtual std::vector<uint32_t> order_dispatch(const std::vector<ActionFeature> &ready_nodes)
    {
        std::vector<uint32_t> res(ready_nodes.size());
        for (uint32_t i = 0; i < ready_nodes.size(); ++i)
            res[i] = i;
        return res;
    }

    virtual std::vector<uint32_t> order_malloc(const std::vector<ActionFeature> &avail_buffers)
    {
        std::vector<uint32_t> res(avail_buffers.size());
        for (uint32_t i = 0; i < avail_buffers.size(); ++i)
            res[i] = i;
        return res;
    }
};