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
            ss << "[Graph.add] DType mismatch: " << nodes[id0].dtype << ", " << nodes[id1].dtype;
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

    uint32_t mul(uint32_t id0, uint32_t id1)
    {
        if (nodes[id0].dtype != nodes[id1].dtype)
        {
            std::stringstream ss;
            ss << "[Graph.mul] DType mismatch: " << nodes[id0].dtype << ", " << nodes[id1].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::MUL;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t div(uint32_t id0, uint32_t id1)
    {
        if (nodes[id0].dtype != nodes[id1].dtype)
        {
            std::stringstream ss;
            ss << "[Graph.div] DType mismatch: " << nodes[id0].dtype << ", " << nodes[id1].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::DIVIDE;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t dot(uint32_t id0, uint32_t id1)
    {
        if (nodes[id0].dtype != nodes[id1].dtype)
        {
            std::stringstream ss;
            ss << "[Graph.dot] DType mismatch: " << nodes[id0].dtype << ", " << nodes[id1].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::DOT;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t sin(uint32_t id0)
    {
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::SIN;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t cos(uint32_t id0)
    {
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::COS;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t neg(uint32_t id0)
    {
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::NEGATE;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t pow(uint32_t id0, uint32_t id1)
    {
        if (nodes[id0].dtype != nodes[id1].dtype)
        {
            std::stringstream ss;
            ss << "[Graph.pow] DType mismatch: " << nodes[id0].dtype << ", " << nodes[id1].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::POWER;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t sum(uint32_t id0, uint32_t id1, uint32_t id2)
    {
        if (nodes[id1].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.sum] Expected " << DType::INT32 << " for input 1, got: " << nodes[id1].dtype;
            throw std::runtime_error(ss.str());
        }
        if (nodes[id2].dtype != DType::BOOL)
        {
            std::stringstream ss;
            ss << "[Graph.sum] Expected " << DType::BOOL << " for input 2, got: " << nodes[id2].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::SUM;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1, id2};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t max(uint32_t id0, uint32_t id1, uint32_t id2)
    {
        if (nodes[id1].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.max] Expected " << DType::INT32 << " for input 1, got: " << nodes[id1].dtype;
            throw std::runtime_error(ss.str());
        }
        if (nodes[id2].dtype != DType::BOOL)
        {
            std::stringstream ss;
            ss << "[Graph.max] Expected " << DType::BOOL << " for input 2, got: " << nodes[id2].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::MAX;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1, id2};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t reshape(uint32_t id0, uint32_t id1)
    {
        if (nodes[id1].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.reshape] Expected " << DType::INT32 << " for input 1, got: " << nodes[id1].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::RESHAPE;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t permute(uint32_t id0, uint32_t id1)
    {
        if (nodes[id1].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.permute] Expected " << DType::INT32 << " for input 1, got: " << nodes[id1].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::PERMUTE;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t slice(uint32_t id0, uint32_t id1, uint32_t id2, uint32_t id3)
    {
        if (nodes[id1].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.slice] Expected " << DType::INT32 << " for input 1, got: " << nodes[id1].dtype;
            throw std::runtime_error(ss.str());
        }
        if (nodes[id2].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.slice] Expected " << DType::INT32 << " for input 2, got: " << nodes[id2].dtype;
            throw std::runtime_error(ss.str());
        }
        if (nodes[id3].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.slice] Expected " << DType::INT32 << " for input 3, got: " << nodes[id3].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::SLICE;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1, id2, id3};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t concat(std::vector<uint32_t> ids, uint32_t id1)
    {
        if (ids.size() == 0) {
            throw std::runtime_error("[Graph.concat] Expected at least one input tensor, got None.");
        }
        for (int i = 0; i < ids.size(); i++) {
            uint32_t id = ids[i];
            if (nodes[ids[0]].dtype != nodes[id].dtype)
            {
                std::stringstream ss;
                ss << "[Graph.concat] DType mismatch between tensor 0 and tensor " << i << ": " << nodes[ids[0]].dtype << ", " << nodes[id].dtype;
                throw std::runtime_error(ss.str());
            }
        }
        if (nodes[id1].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.concat] Expected " << DType::INT32 << " for input 1, got: " << nodes[id1].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::CONCAT;
        node.dtype = nodes[ids[0]].dtype;
        node.parentIds = ids;
        node.parentIds.push_back(id1);
        nodes.push_back(node);
        return node.id;
    }

    uint32_t cast(uint32_t id0, DType dtype) {
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::CAST;
        node.dtype = dtype;
        node.parentIds = {id0};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t repeat(uint32_t id0, uint32_t repeats_id, uint32_t axis_id) {
        if (nodes[repeats_id].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.repeat] Expected " << DType::INT32 << " for input 1, got: " << nodes[repeats_id].dtype;
            throw std::runtime_error(ss.str());
        }
        if (nodes[axis_id].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.repeat] Expected " << DType::INT32 << " for input 2, got: " << nodes[axis_id].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::REPEAT;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, repeats_id, axis_id};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t arange(uint32_t id0, uint32_t id1, uint32_t id2, uint32_t id3)
    {
        if (nodes[id1].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.arange] Expected " << DType::INT32 << " for input 1, got: " << nodes[id1].dtype;
            throw std::runtime_error(ss.str());
        }
        if (nodes[id2].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.arange] Expected " << DType::INT32 << " for input 2, got: " << nodes[id2].dtype;
            throw std::runtime_error(ss.str());
        }
        if (nodes[id3].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.arange] Expected " << DType::INT32 << " for input 3, got: " << nodes[id3].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::ARANGE;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1, id2, id3};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t triu(uint32_t id0, uint32_t k_id) {
        if (nodes[k_id].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.triu] Expected " << DType::INT32 << " for input 1, got: " << nodes[k_id].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::TRIU;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, k_id};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t gather(uint32_t id0, uint32_t indices_id) {
        if (nodes[indices_id].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.gather] Expected " << DType::INT32 << " for input 1, got: " << nodes[indices_id].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::GATHER;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, indices_id};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t 
};