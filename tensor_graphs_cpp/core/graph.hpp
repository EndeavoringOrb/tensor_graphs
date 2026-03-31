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

    TensorNode &allocateNode(OpType _opType, std::string _opName, DType _dtype, std::vector<uint32_t> _parentIds, std::vector<uint32_t> _shape = {}, std::vector<int64_t> _strides = {}, Backend _backend = Backend::CPU, StorageType _storageType = StorageType::PERSISTENT, std::string _contentHash = "")
    {
        uint32_t id = allocator->allocate();
        nodes[id] = TensorNode(id, _opType, _opName, _dtype, _parentIds, _shape, _strides, _backend, _storageType, _contentHash);
        return nodes[id];
    }

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
        
        SHA256 sha;
        sha.update(static_cast<const uint8_t *>(dataPtr), sizeBytes);

        TensorNode &node = allocateNode(OpType::INPUT, "", dtype, {}, shape, {}, Backend::CPU, StorageType::PERSISTENT, sha.digest());
        uint32_t id = node.id;

        std::vector<uint8_t> buffer(sizeBytes);
        std::memcpy(buffer.data(), dataPtr, sizeBytes);
        constantStaging[id] = std::move(buffer);

        return id;
    }

    uint32_t weight(const std::string &path, const std::string &name)
    {
        registerLoader(path);
        auto &loader = loaders.at(path);
        if (!loader->hasTensor(name))
        {
            Error::throw_err("Tensor '" + name + "' not found in: " + path);
        }

        SHA256 sha;
        sha.update(path + "::" + name);

        const auto &meta = loader->getMetadata(name);
        TensorNode &node = allocateNode(OpType::INPUT, "", meta.dtype, {}, meta.shape, {}, Backend::CPU, StorageType::PERSISTENT, sha.digest());
        uint32_t id = node.id;
        weightSources[id] = {path, name};

        return id;
    }

    uint32_t input(std::vector<uint32_t> shape, DType dtype, std::vector<int64_t> strides = {}, StorageType storageType = StorageType::PERSISTENT)
    {
        TensorNode &node = allocateNode(OpType::INPUT, "", dtype, {}, shape, strides, Backend::CPU, storageType);
        return node.id;
    }

    uint32_t contiguous(uint32_t id0)
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::CONTIGUOUS, "", dtype, {id0});
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
        TensorNode &node = allocateNode(OpType::ADD, "", dtype, {id0, id1});
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
        TensorNode &node = allocateNode(OpType::MUL, "", dtype, {id0, id1});
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
        TensorNode &node = allocateNode(OpType::DIVIDE, "", dtype, {id0, id1});
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
        TensorNode &node = allocateNode(OpType::DOT, "", dtype, {id0, id1});
        return node.id;
    }

    uint32_t sin(uint32_t id0)
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::SIN, "", dtype, {id0});
        return node.id;
    }

    uint32_t cos(uint32_t id0)
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::COS, "", dtype, {id0});
        return node.id;
    }

    uint32_t neg(uint32_t id0)
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::NEGATE, "", dtype, {id0});
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
        TensorNode &node = allocateNode(OpType::POWER, "", dtype, {id0, id1});
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
        TensorNode &node = allocateNode(OpType::SUM, "", dtype, {id0, id1});
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
        TensorNode &node = allocateNode(OpType::MAX, "", dtype, {id0, id1});
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
        TensorNode &node = allocateNode(OpType::RESHAPE, "", dtype, {id0, id1});
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
        TensorNode &node = allocateNode(OpType::PERMUTE, "", dtype, {id0, id1});
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
        TensorNode &node = allocateNode(OpType::SLICE, "", dtype, {id0, id1, id2, id3});
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
        TensorNode &node = allocateNode(OpType::SCATTER, "", dtype, {id0, id1, id2, id3, id4});
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
        ids.push_back(id1);
        TensorNode &node = allocateNode(OpType::CONCAT, "", dtype, ids);
        return node.id;
    }

    uint32_t cast(uint32_t id0, DType dtype)
    {
        TensorNode &node = allocateNode(OpType::CAST, "", dtype, {id0});
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
        TensorNode &node = allocateNode(OpType::REPEAT, "", dtype, {id0, repeats_id, axis_id});
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
        TensorNode &node = allocateNode(OpType::ARANGE, "", DType::INT32, {id1, id2, id3});
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
        TensorNode &node = allocateNode(OpType::TRIU, "", dtype, {id0, k_id});
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
        TensorNode &node = allocateNode(OpType::GATHER, "", dtype, {id0, indices_id});
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
        TensorNode &node = allocateNode(OpType::FILL, "", dtype, {value_id, shape_id});
        return node.id;
    }

    uint32_t copyto(uint32_t id0, Backend backend)
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::COPY_TO, "", dtype, {id0}, {}, {}, backend);
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
        TensorNode &node = allocateNode(OpType::IM2COL, "", dtype, {input_id, kernel_size_id, stride_id, padding_id});
        return node.id;
    }
};

struct LogicalGraph
{
    Graph graph;
    std::unordered_map<std::string, std::vector<uint32_t>> fusionMap;
    std::unordered_map<uint32_t, uint32_t> estimatedRefCounts;
};