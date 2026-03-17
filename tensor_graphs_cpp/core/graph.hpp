#pragma once
#include "core/types.hpp"
#include "core/loaders/safetensors.hpp"
#include "core/memory.hpp"
#include <vector>
#include <stdexcept>
#include <sstream>

struct MemoryManager;

struct Graph
{
    uint32_t count = 0;
    std::vector<TensorNode> nodes;
    std::unordered_map<std::string, std::shared_ptr<SafetensorsLoader>> loaders;     // Mapping of path -> Loader instance
    std::unordered_map<uint32_t, std::pair<std::string, std::string>> weightSources; // Mapping of nodeId -> {path, tensor_name}

    std::unordered_map<uint32_t, std::vector<uint8_t>> constantStaging;

    uint32_t allocateId() noexcept { return count++; }

    void registerLoader(const std::string &path)
    {
        if (loaders.find(path) == loaders.end())
        {
            loaders[path] = std::make_shared<SafetensorsLoader>(path);
        }
    }

    uint32_t constant(const std::vector<uint32_t> &shape, const void *dataPtr, DType dtype)
    {
        uint64_t sizeBytes = getSizeBytes(shape, dtype);
        uint32_t id = allocateId();
        SHA256 sha;
        sha.update(static_cast<const uint8_t *>(dataPtr), sizeBytes);

        std::vector<uint8_t> buffer(sizeBytes);
        std::memcpy(buffer.data(), dataPtr, sizeBytes);
        constantStaging[id] = std::move(buffer);

        TensorView view;
        view.shape = shape;
        view.strides = TensorView::calcContiguousStrides(shape);
        view.baseOffset = 0;
        view.dtype = dtype;

        uint32_t nodeId = inputWithId(id, shape, dtype, view, StorageType::PERSISTENT);
        nodes[nodeId].contentHash = sha.digest();
        return nodeId;
    }

    uint32_t weight(const std::string &path, const std::string &name)
    {
        registerLoader(path);
        auto &loader = loaders.at(path);
        if (!loader->hasTensor(name))
        {
            Error::throw_err("Tensor '" + name + "' not found in: " + path);
        }

        const auto &meta = loader->getMetadata(name);
        uint32_t id = allocateId();
        weightSources[id] = {path, name};

        TensorView view;
        view.shape = meta.shape;
        view.strides = TensorView::calcContiguousStrides(meta.shape);
        view.baseOffset = 0;
        view.dtype = meta.dtype;

        uint32_t nodeId = inputWithId(id, meta.shape, meta.dtype, view, StorageType::PERSISTENT);
        SHA256 sha;
        sha.update(path + "::" + name);
        nodes[nodeId].contentHash = sha.digest();
        return nodeId;
    }

    uint32_t input(std::vector<uint32_t> shape, DType dtype, TensorView view, StorageType storageType = StorageType::PERSISTENT)
    {
        return inputWithId(allocateId(), shape, dtype, view, storageType);
    }

    uint32_t inputWithId(uint32_t id, std::vector<uint32_t> shape, DType dtype, TensorView view, StorageType storageType = StorageType::PERSISTENT)
    {
        TensorNode node = TensorNode();
        node.id = id;
        node.opType = OpType::INPUT;
        node.dtype = dtype;
        node.shape = shape;
        node.view = view;
        node.storageType = storageType;
        nodes.push_back(node);
        return node.id;
    }

    uint32_t contiguous(uint32_t id0)
    {
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::CONTIGUOUS;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t add(uint32_t id0, uint32_t id1)
    {
        if (nodes[id0].dtype != nodes[id1].dtype)
        {
            std::stringstream ss;
            ss << "[Graph.add] DType mismatch: " << nodes[id0].dtype << ", " << nodes[id1].dtype;
            Error::throw_err(ss.str());
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
            Error::throw_err(ss.str());
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
            Error::throw_err(ss.str());
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
            Error::throw_err(ss.str());
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
            Error::throw_err(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::POWER;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t sum(uint32_t id0, uint32_t id1)
    {
        if (nodes[id1].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.sum] Expected " << DType::INT32 << " for input 1, got: " << nodes[id1].dtype;
            Error::throw_err(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::SUM;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t max(uint32_t id0, uint32_t id1)
    {
        if (nodes[id1].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.max] Expected " << DType::INT32 << " for input 1, got: " << nodes[id1].dtype;
            Error::throw_err(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::MAX;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t reshape(uint32_t id0, uint32_t id1)
    {
        if (nodes[id1].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.reshape] Expected " << DType::INT32 << " for input 1, got: " << nodes[id1].dtype;
            Error::throw_err(ss.str());
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
            Error::throw_err(ss.str());
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
            Error::throw_err(ss.str());
        }
        if (nodes[id2].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.slice] Expected " << DType::INT32 << " for input 2, got: " << nodes[id2].dtype;
            Error::throw_err(ss.str());
        }
        if (nodes[id3].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.slice] Expected " << DType::INT32 << " for input 3, got: " << nodes[id3].dtype;
            Error::throw_err(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::SLICE;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1, id2, id3};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t scatter(uint32_t id0, uint32_t id1, uint32_t id2, uint32_t id3, uint32_t id4)
    {
        if (nodes[id2].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.scatter] Expected INT32 for starts, got: " << toString(nodes[id2].dtype);
            Error::throw_err(ss.str());
        }
        if (nodes[id3].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.scatter] Expected INT32 for ends, got: " << toString(nodes[id3].dtype);
            Error::throw_err(ss.str());
        }
        if (nodes[id4].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.scatter] Expected INT32 for steps, got: " << toString(nodes[id4].dtype);
            Error::throw_err(ss.str());
        }
        if (nodes[id0].dtype != nodes[id1].dtype)
        {
            std::stringstream ss;
            ss << "[Graph.scatter] DType mismatch between target (" << toString(nodes[id0].dtype) << ") and updates (" << toString(nodes[id1].dtype) << ")";
            Error::throw_err(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::SCATTER;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1, id2, id3, id4};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t concat(std::vector<uint32_t> ids, uint32_t id1)
    {
        if (ids.size() == 0)
        {
            Error::throw_err("[Graph.concat] Expected at least 1 input tensor, got 0.");
        }
        for (int i = 0; i < ids.size(); i++)
        {
            uint32_t id = ids[i];
            if (nodes[ids[0]].dtype != nodes[id].dtype)
            {
                std::stringstream ss;
                ss << "[Graph.concat] DType mismatch between tensor 0 and tensor " << i << ": " << nodes[ids[0]].dtype << ", " << nodes[id].dtype;
                Error::throw_err(ss.str());
            }
        }
        if (nodes[id1].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.concat] Expected " << DType::INT32 << " for input 1, got: " << nodes[id1].dtype;
            Error::throw_err(ss.str());
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

    uint32_t cast(uint32_t id0, DType dtype)
    {
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::CAST;
        node.dtype = dtype;
        node.parentIds = {id0};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t repeat(uint32_t id0, uint32_t repeats_id, uint32_t axis_id)
    {
        if (nodes[repeats_id].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.repeat] Expected " << DType::INT32 << " for input 1, got: " << nodes[repeats_id].dtype;
            Error::throw_err(ss.str());
        }
        if (nodes[axis_id].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.repeat] Expected " << DType::INT32 << " for input 2, got: " << nodes[axis_id].dtype;
            Error::throw_err(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::REPEAT;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, repeats_id, axis_id};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t arange(uint32_t id1, uint32_t id2, uint32_t id3)
    {
        if (nodes[id1].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.arange] Expected " << DType::INT32 << " for input 1, got: " << nodes[id1].dtype;
            Error::throw_err(ss.str());
        }
        if (nodes[id2].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.arange] Expected " << DType::INT32 << " for input 2, got: " << nodes[id2].dtype;
            Error::throw_err(ss.str());
        }
        if (nodes[id3].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.arange] Expected " << DType::INT32 << " for input 3, got: " << nodes[id3].dtype;
            Error::throw_err(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::ARANGE;
        node.dtype = DType::INT32;
        node.parentIds = {id1, id2, id3};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t triu(uint32_t id0, uint32_t k_id)
    {
        if (nodes[k_id].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.triu] Expected " << DType::INT32 << " for input 1, got: " << nodes[k_id].dtype;
            Error::throw_err(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::TRIU;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, k_id};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t gather(uint32_t id0, uint32_t indices_id)
    {
        if (nodes[indices_id].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.gather] Expected " << DType::INT32 << " for input 1, got: " << nodes[indices_id].dtype;
            Error::throw_err(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::GATHER;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, indices_id};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t fill(uint32_t value_id, uint32_t shape_id)
    {
        if (nodes[shape_id].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.fill] Expected " << DType::INT32 << " for input 1, got: " << nodes[shape_id].dtype;
            Error::throw_err(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::FILL;
        node.dtype = nodes[value_id].dtype;
        node.parentIds = {value_id, shape_id};
        nodes.push_back(node);
        return node.id;
    }

    uint32_t copyto(uint32_t id0, Backend backend)
    {
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::COPY_TO;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0};
        node.backend = backend;
        nodes.push_back(node);
        return node.id;
    }

    uint32_t im2col(uint32_t input_id, uint32_t kernel_size_id, uint32_t stride_id, uint32_t padding_id)
    {
        if (nodes[kernel_size_id].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.im2col] Expected " << DType::INT32 << " for input 1, got: " << nodes[kernel_size_id].dtype;
            Error::throw_err(ss.str());
        }
        if (nodes[stride_id].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.im2col] Expected " << DType::INT32 << " for input 2, got: " << nodes[stride_id].dtype;
            Error::throw_err(ss.str());
        }
        if (nodes[padding_id].dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.im2col] Expected " << DType::INT32 << " for input 3, got: " << nodes[padding_id].dtype;
            Error::throw_err(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::IM2COL;
        node.dtype = nodes[input_id].dtype;
        node.parentIds = {input_id, kernel_size_id, stride_id, padding_id};
        nodes.push_back(node);
        return node.id;
    }
};

struct LogicalGraph
{
    Graph graph;
    std::unordered_map<std::string, std::vector<uint32_t>> fusionMap;
    std::unordered_map<uint32_t, uint32_t> estimatedRefCounts;
};