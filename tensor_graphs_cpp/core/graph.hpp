#pragma once
#include "core/types.hpp"
#include "core/loaders/safetensors.hpp"
#include "core/memory.hpp"
#include <vector>
#include <stdexcept>
#include <sstream>
#include <deque>

struct MemoryManager;

struct IdAllocator
{
    uint32_t nextId = 0;
    uint32_t allocate() { return nextId++; }
};

struct Graph
{
    std::unordered_map<uint32_t, TensorNode> nodes;
    std::shared_ptr<IdAllocator> allocator;
    std::unordered_map<std::string, std::shared_ptr<SafetensorsLoader>> loaders;     // Mapping of path -> Loader instance
    std::unordered_map<uint32_t, std::pair<std::string, std::string>> weightSources; // Mapping of nodeId -> {path, tensor_name}

    std::unordered_map<uint32_t, std::vector<uint8_t>> constantStaging;

    Graph() : allocator(std::make_shared<IdAllocator>()) {}

    bool hasNode(uint32_t id) const
    {
        return nodes.find(id) != nodes.end();
    }

    TensorNode &getNode(uint32_t id)
    {
        return nodes.at(id);
    }

    const TensorNode &getNode(uint32_t id) const
    {
        return nodes.at(id);
    }

    TensorNode &allocateNode()
    {
        uint32_t id = allocator->allocate();
        nodes[id] = TensorNode();
        nodes[id].id = id;
        return nodes[id];
    }

    void registerLoader(const std::string &path)
    {
        if (loaders.find(path) == loaders.end())
        {
            loaders[path] = std::make_shared<SafetensorsLoader>(path);
        }
    }

    uint32_t constant(TensorNode &node, const std::vector<uint32_t> &shape, const void *dataPtr, DType dtype)
    {
        uint64_t sizeBytes = getSizeBytes(shape, dtype);
        uint32_t id = node.id;
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

        node.opType = OpType::INPUT;
        node.dtype = dtype;
        node.shape = shape;
        node.view = view;
        node.storageType = StorageType::PERSISTENT;
        node.contentHash = sha.digest();
        return id;
    }

    uint32_t constant(const std::vector<uint32_t> &shape, const void *dataPtr, DType dtype)
    {
        return constant(allocateNode(), shape, dataPtr, dtype);
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
        TensorNode &node = allocateNode();
        uint32_t id = node.id;
        weightSources[id] = {path, name};

        TensorView view;
        view.shape = meta.shape;
        view.strides = TensorView::calcContiguousStrides(meta.shape);
        view.baseOffset = 0;
        view.dtype = meta.dtype;

        node.opType = OpType::INPUT;
        node.dtype = meta.dtype;
        node.shape = meta.shape;
        node.view = view;
        node.storageType = StorageType::PERSISTENT;

        SHA256 sha;
        sha.update(path + "::" + name);
        node.contentHash = sha.digest();
        return id;
    }

    uint32_t input(TensorNode &node, std::vector<uint32_t> shape, DType dtype, TensorView view, StorageType storageType = StorageType::PERSISTENT)
    {
        node.opType = OpType::INPUT;
        node.dtype = dtype;
        node.shape = shape;
        node.view = view;
        node.storageType = storageType;
        return node.id;
    }

    uint32_t input(std::vector<uint32_t> shape, DType dtype, TensorView view, StorageType storageType = StorageType::PERSISTENT)
    {
        return input(allocateNode(), shape, dtype, view, storageType);
    }

    uint32_t inputWithId(uint32_t id, std::vector<uint32_t> shape, DType dtype, TensorView view, StorageType storageType = StorageType::PERSISTENT)
    {
        if (id >= allocator->nextId)
        {
            allocator->nextId = id + 1;
        }
        if (!hasNode(id))
        {
            nodes[id] = TensorNode();
            nodes[id].id = id;
        }
        return input(getNode(id), shape, dtype, view, storageType);
    }

    uint32_t contiguous(uint32_t id0)
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::CONTIGUOUS;
        node.dtype = dtype;
        node.parentIds = {id0};
        return node.id;
    }

    uint32_t add(uint32_t id0, uint32_t id1)
    {
        if (getNode(id0).dtype != getNode(id1).dtype)
        {
            std::stringstream ss;
            ss << "[Graph.add] DType mismatch: " << getNode(id0).dtype << ", " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::ADD;
        node.dtype = dtype;
        node.parentIds = {id0, id1};
        return node.id;
    }

    uint32_t mul(uint32_t id0, uint32_t id1)
    {
        if (getNode(id0).dtype != getNode(id1).dtype)
        {
            std::stringstream ss;
            ss << "[Graph.mul] DType mismatch: " << getNode(id0).dtype << ", " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::MUL;
        node.dtype = dtype;
        node.parentIds = {id0, id1};
        return node.id;
    }

    uint32_t div(uint32_t id0, uint32_t id1)
    {
        if (getNode(id0).dtype != getNode(id1).dtype)
        {
            std::stringstream ss;
            ss << "[Graph.div] DType mismatch: " << getNode(id0).dtype << ", " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::DIVIDE;
        node.dtype = dtype;
        node.parentIds = {id0, id1};
        return node.id;
    }

    uint32_t dot(uint32_t id0, uint32_t id1)
    {
        if (getNode(id0).dtype != getNode(id1).dtype)
        {
            std::stringstream ss;
            ss << "[Graph.dot] DType mismatch: " << getNode(id0).dtype << ", " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::DOT;
        node.dtype = dtype;
        node.parentIds = {id0, id1};
        return node.id;
    }

    uint32_t sin(uint32_t id0)
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::SIN;
        node.dtype = dtype;
        node.parentIds = {id0};
        return node.id;
    }

    uint32_t cos(uint32_t id0)
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::COS;
        node.dtype = dtype;
        node.parentIds = {id0};
        return node.id;
    }

    uint32_t neg(uint32_t id0)
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::NEGATE;
        node.dtype = dtype;
        node.parentIds = {id0};
        return node.id;
    }

    uint32_t pow(uint32_t id0, uint32_t id1)
    {
        if (getNode(id0).dtype != getNode(id1).dtype)
        {
            std::stringstream ss;
            ss << "[Graph.pow] DType mismatch: " << getNode(id0).dtype << ", " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::POWER;
        node.dtype = dtype;
        node.parentIds = {id0, id1};
        return node.id;
    }

    uint32_t sum(uint32_t id0, uint32_t id1)
    {
        if (getNode(id1).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.sum] Expected " << DType::INT32 << " for input 1, got: " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.id = node.id; // redundant but following pattern
        node.opType = OpType::SUM;
        node.dtype = dtype;
        node.parentIds = {id0, id1};
        return node.id;
    }

    uint32_t max(uint32_t id0, uint32_t id1)
    {
        if (getNode(id1).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.max] Expected " << DType::INT32 << " for input 1, got: " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::MAX;
        node.dtype = dtype;
        node.parentIds = {id0, id1};
        return node.id;
    }

    uint32_t reshape(uint32_t id0, uint32_t id1)
    {
        if (getNode(id1).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.reshape] Expected " << DType::INT32 << " for input 1, got: " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::RESHAPE;
        node.dtype = dtype;
        node.parentIds = {id0, id1};
        return node.id;
    }

    uint32_t permute(uint32_t id0, uint32_t id1)
    {
        if (getNode(id1).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.permute] Expected " << DType::INT32 << " for input 1, got: " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::PERMUTE;
        node.dtype = dtype;
        node.parentIds = {id0, id1};
        return node.id;
    }

    uint32_t slice(uint32_t id0, uint32_t id1, uint32_t id2, uint32_t id3)
    {
        if (getNode(id1).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.slice] Expected " << DType::INT32 << " for input 1, got: " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        if (getNode(id2).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.slice] Expected " << DType::INT32 << " for input 2, got: " << getNode(id2).dtype;
            Error::throw_err(ss.str());
        }
        if (getNode(id3).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.slice] Expected " << DType::INT32 << " for input 3, got: " << getNode(id3).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::SLICE;
        node.dtype = dtype;
        node.parentIds = {id0, id1, id2, id3};
        return node.id;
    }

    uint32_t scatter(uint32_t id0, uint32_t id1, uint32_t id2, uint32_t id3, uint32_t id4)
    {
        if (getNode(id2).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.scatter] Expected INT32 for starts, got: " << toString(getNode(id2).dtype);
            Error::throw_err(ss.str());
        }
        if (getNode(id3).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.scatter] Expected INT32 for ends, got: " << toString(getNode(id3).dtype);
            Error::throw_err(ss.str());
        }
        if (getNode(id4).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.scatter] Expected INT32 for steps, got: " << toString(getNode(id4).dtype);
            Error::throw_err(ss.str());
        }
        if (getNode(id0).dtype != getNode(id1).dtype)
        {
            std::stringstream ss;
            ss << "[Graph.scatter] DType mismatch between target (" << toString(getNode(id0).dtype) << ") and updates (" << toString(getNode(id1).dtype) << ")";
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::SCATTER;
        node.dtype = dtype;
        node.parentIds = {id0, id1, id2, id3, id4};
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
            if (getNode(ids[0]).dtype != getNode(id).dtype)
            {
                std::stringstream ss;
                ss << "[Graph.concat] DType mismatch between tensor 0 and tensor " << i << ": " << getNode(ids[0]).dtype << ", " << getNode(id).dtype;
                Error::throw_err(ss.str());
            }
        }
        if (getNode(id1).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.concat] Expected " << DType::INT32 << " for input 1, got: " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(ids[0]).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::CONCAT;
        node.dtype = dtype;
        node.parentIds = ids;
        node.parentIds.push_back(id1);
        return node.id;
    }

    uint32_t cast(uint32_t id0, DType dtype)
    {
        TensorNode &node = allocateNode();
        node.opType = OpType::CAST;
        node.dtype = dtype;
        node.parentIds = {id0};
        return node.id;
    }

    uint32_t repeat(uint32_t id0, uint32_t repeats_id, uint32_t axis_id)
    {
        if (getNode(repeats_id).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.repeat] Expected " << DType::INT32 << " for input 1, got: " << getNode(repeats_id).dtype;
            Error::throw_err(ss.str());
        }
        if (getNode(axis_id).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.repeat] Expected " << DType::INT32 << " for input 2, got: " << getNode(axis_id).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::REPEAT;
        node.dtype = dtype;
        node.parentIds = {id0, repeats_id, axis_id};
        return node.id;
    }

    uint32_t arange(uint32_t id1, uint32_t id2, uint32_t id3)
    {
        if (getNode(id1).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.arange] Expected " << DType::INT32 << " for input 1, got: " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        if (getNode(id2).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.arange] Expected " << DType::INT32 << " for input 2, got: " << getNode(id2).dtype;
            Error::throw_err(ss.str());
        }
        if (getNode(id3).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.arange] Expected " << DType::INT32 << " for input 3, got: " << getNode(id3).dtype;
            Error::throw_err(ss.str());
        }
        TensorNode &node = allocateNode();
        node.opType = OpType::ARANGE;
        node.dtype = DType::INT32;
        node.parentIds = {id1, id2, id3};
        return node.id;
    }

    uint32_t triu(uint32_t id0, uint32_t k_id)
    {
        if (getNode(k_id).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.triu] Expected " << DType::INT32 << " for input 1, got: " << getNode(k_id).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::TRIU;
        node.dtype = dtype;
        node.parentIds = {id0, k_id};
        return node.id;
    }

    uint32_t gather(uint32_t id0, uint32_t indices_id)
    {
        if (getNode(indices_id).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.gather] Expected " << DType::INT32 << " for input 1, got: " << getNode(indices_id).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::GATHER;
        node.dtype = dtype;
        node.parentIds = {id0, indices_id};
        return node.id;
    }

    uint32_t fill(uint32_t value_id, uint32_t shape_id)
    {
        if (getNode(shape_id).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.fill] Expected " << DType::INT32 << " for input 1, got: " << getNode(shape_id).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(value_id).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::FILL;
        node.dtype = dtype;
        node.parentIds = {value_id, shape_id};
        return node.id;
    }

    uint32_t copyto(uint32_t id0, Backend backend)
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::COPY_TO;
        node.dtype = dtype;
        node.parentIds = {id0};
        node.backend = backend;
        return node.id;
    }

    uint32_t im2col(uint32_t input_id, uint32_t kernel_size_id, uint32_t stride_id, uint32_t padding_id)
    {
        if (getNode(kernel_size_id).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.im2col] Expected " << DType::INT32 << " for input 1, got: " << getNode(kernel_size_id).dtype;
            Error::throw_err(ss.str());
        }
        if (getNode(stride_id).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.im2col] Expected " << DType::INT32 << " for input 2, got: " << getNode(stride_id).dtype;
            Error::throw_err(ss.str());
        }
        if (getNode(padding_id).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.im2col] Expected " << DType::INT32 << " for input 3, got: " << getNode(padding_id).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(input_id).dtype;
        TensorNode &node = allocateNode();
        node.opType = OpType::IM2COL;
        node.dtype = dtype;
        node.parentIds = {input_id, kernel_size_id, stride_id, padding_id};
        return node.id;
    }
};

struct LogicalGraph
{
    Graph graph;
    std::unordered_map<std::string, std::vector<uint32_t>> fusionMap;
    std::unordered_map<uint32_t, uint32_t> estimatedRefCounts;
};