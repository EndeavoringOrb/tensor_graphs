#pragma once
#include "core/types.hpp"

struct Graph
{
    uint32_t count = 0;
    std::vector<TensorNode> nodes;

    uint32_t allocateId() noexcept { return count++; }

    uint32_t input(std::vector<uint32_t> shape, DType dtype, TensorView view)
    {
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::INPUT;
        node.dtype = dtype;
        node.shape = shape;
        node.view = view;
        nodes.push_back(node);
        return node.id;
    }

    uint32_t add(uint32_t id0, uint32_t id1)
    {
        if (nodes[id0].dtype != nodes[id1].dtype)
        {
            std::stringstream ss;
            ss << "DType mismatch in add: " << nodes[id0].dtype << ", " << nodes[id1].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::ADD;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1};
        nodes.push_back(node);
        return node.id;
    }
};