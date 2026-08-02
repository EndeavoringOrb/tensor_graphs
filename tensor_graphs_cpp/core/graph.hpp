#pragma once
#include <deque>
#include <filesystem>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "core/loaders/loader.hpp"
#include "core/memory.hpp"
#include "core/types.hpp"

struct MemoryManager;

enum class InputDataType : uint32_t
{
    STORAGE,
    CONSTANT,
    RUNTIME
};

struct Graph
{
    std::unordered_map<LogicalId, TensorNode> nodes;

    std::unordered_map<LogicalId, InputDataType> input_data_types;
    std::unordered_map<LogicalId, std::shared_ptr<std::vector<uint8_t>>> constantStaging;
    std::unordered_map<uint64_t, std::vector<LogicalId>> constantHashIndex;

    Graph()
    {
    }

    bool hasNode(LogicalId id) const
    {
        return nodes.find(id) != nodes.end();
    }

    TensorNode &getNode(LogicalId id)
    {
        return nodes.at(id);
    }

    const TensorNode &getNode(LogicalId id) const
    {
        return nodes.at(id);
    }

    InputDataType getInputDataType(LogicalId id) const
    {
        return input_data_types.at(id);
    }

    inline std::vector<int32_t> getConstantInt32(LogicalId id) const
    {
        if (constantStaging.count(id))
        {
            const auto &data = *constantStaging.at(id);
            const auto &node = getNode(id);
            uint64_t numElements = countElements(node.getShape());
            std::vector<int32_t> res(numElements);
            const int32_t *src = reinterpret_cast<const int32_t *>(data.data());
            for (uint64_t i = 0; i < numElements; ++i)
            {
                res[i] = src[getStridedIndex(i, node.getShape(),
                                             node.strides)]; // TODO: does this need getStridedIndex?
            }
            return res;
        }
        std::stringstream ss;
        ss << "Expected constant for shape inference but not found in staging. "
              "Node ID: "
           << id;
        Error::throw_err(ss.str());
    }

    TensorNode &allocateNode(OpType _opType, std::string _opName, DType _dtype, std::vector<LogicalId> _child_ids,
                             std::vector<uint32_t> _shape = {}, std::vector<uint64_t> _strides = {},
                             std::string _contentHash = "", std::source_location loc = std::source_location::current())
    {
        LogicalId id = LogicalIdAllocator::allocate();
        std::string origin = toString(loc);
        nodes[id] = TensorNode(id, _opType, _opName, _dtype, _child_ids, _shape, _strides, _contentHash, origin);
        return nodes[id];
    }

    LogicalId constant(const std::vector<uint32_t> &shape, const void *dataPtr, DType dtype,
                       std::source_location loc = std::source_location::current())
    {
        uint64_t sizeBytes = getSizeBytes(shape, dtype);
        uint64_t dataHash = tg_hash::computeConstantHash(shape, dtype, dataPtr, sizeBytes);

        auto it = constantHashIndex.find(dataHash);
        if (it != constantHashIndex.end())
        {
            for (LogicalId candidateId : it->second)
            {
                auto stagingIt = constantStaging.find(candidateId);
                if (stagingIt == constantStaging.end())
                    continue;

                const TensorNode &node = getNode(candidateId);
                if (node.dtype == dtype && node.getShape() == shape && stagingIt->second->size() == sizeBytes &&
                    (sizeBytes == 0 || std::memcmp(stagingIt->second->data(), dataPtr, sizeBytes) == 0))
                {
                    return candidateId;
                }
            }
        }

        SHA256 sha;
        sha.update(static_cast<const uint8_t *>(dataPtr), sizeBytes);

        TensorNode &node = allocateNode(OpType::INPUT, "", dtype, {}, shape, {}, sha.digest(), loc);
        LogicalId id = node.id;

        auto buffer = std::make_shared<std::vector<uint8_t>>(sizeBytes);
        if (sizeBytes > 0)
        {
            std::memcpy(buffer->data(), dataPtr, sizeBytes);
        }
        constantStaging[id] = buffer;

        input_data_types[id] = InputDataType::CONSTANT;
        constantHashIndex[dataHash].push_back(id);

        return id;
    }

    LogicalId weight(const std::string &path, const std::string &name,
                     std::source_location loc = std::source_location::current())
    {
        if (!FileRegistry::get().hasTensor(path, name))
        {
            Error::throw_err("Tensor '" + name + "' not found in: " + path);
        }

        SHA256 sha;
        sha.update(path + "::" + name);

        const auto &meta = FileRegistry::get().getMetadata(path, name);
        TensorNode &node = allocateNode(OpType::INPUT, name, meta.dtype, {}, meta.shape, {}, sha.digest(), loc);
        FileRegistry::get().registerNode(node.id, path, name);
        input_data_types[node.id] = InputDataType::STORAGE;
        TensorNode &copyNode = allocateNode(OpType::COPY_TO, "", meta.dtype, {node.id}, {}, {}, "", loc);

        return copyNode.id;
    }

    LogicalId input(std::vector<uint32_t> shape, DType dtype, std::vector<uint64_t> strides = {},
                    std::source_location loc = std::source_location::current())
    {
        TensorNode &node = allocateNode(OpType::INPUT, "", dtype, {}, shape, strides, "", loc);
        input_data_types[node.id] = InputDataType::RUNTIME;
        return node.id;
    }

    LogicalId _copyto(LogicalId id0, std::source_location loc = std::source_location::current())
    {
        TensorNode &node = allocateNode(OpType::COPY_TO, "", getNode(id0).dtype, {id0}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId contiguous(LogicalId id0, std::source_location loc = std::source_location::current())
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::CONTIGUOUS, "", dtype, {id0}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId add(LogicalId id0, LogicalId id1, std::source_location loc = std::source_location::current())
    {
        if (getNode(id0).dtype != getNode(id1).dtype)
        {
            std::stringstream ss;
            ss << "[Graph.add] DType mismatch: " << getNode(id0).dtype << ", " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::ADD, "", dtype, {id0, id1}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId mul(LogicalId id0, LogicalId id1, std::source_location loc = std::source_location::current())
    {
        if (getNode(id0).dtype != getNode(id1).dtype)
        {
            std::stringstream ss;
            ss << "[Graph.mul] DType mismatch: " << getNode(id0).dtype << ", " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::MUL, "", dtype, {id0, id1}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId div(LogicalId id0, LogicalId id1, std::source_location loc = std::source_location::current())
    {
        if (getNode(id0).dtype != getNode(id1).dtype)
        {
            std::stringstream ss;
            ss << "[Graph.div] DType mismatch: " << getNode(id0).dtype << ", " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::DIVIDE, "", dtype, {id0, id1}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId dot(LogicalId id0, LogicalId id1, std::source_location loc = std::source_location::current())
    {
        if (getNode(id0).dtype != getNode(id1).dtype)
        {
            std::stringstream ss;
            ss << "[Graph.dot] DType mismatch: " << getNode(id0).dtype << ", " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::DOT, "", dtype, {id0, id1}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId sin(LogicalId id0, std::source_location loc = std::source_location::current())
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::SIN, "", dtype, {id0}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId cos(LogicalId id0, std::source_location loc = std::source_location::current())
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::COS, "", dtype, {id0}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId neg(LogicalId id0, std::source_location loc = std::source_location::current())
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::NEGATE, "", dtype, {id0}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId pow(LogicalId id0, LogicalId id1, std::source_location loc = std::source_location::current())
    {
        if (getNode(id0).dtype != getNode(id1).dtype)
        {
            std::stringstream ss;
            ss << "[Graph.pow] DType mismatch: " << getNode(id0).dtype << ", " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::POWER, "", dtype, {id0, id1}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId sum(LogicalId id0, LogicalId id1, std::source_location loc = std::source_location::current())
    {
        if (getNode(id1).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.sum] Expected " << DType::INT32 << " for input 1, got: " << getNode(id1).dtype;
            Error::throw_err(ss.str(), loc);
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::SUM, "", dtype, {id0, id1}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId max(LogicalId id0, LogicalId id1, std::source_location loc = std::source_location::current())
    {
        if (getNode(id1).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.max] Expected " << DType::INT32 << " for input 1, got: " << getNode(id1).dtype;
            Error::throw_err(ss.str(), loc);
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::MAX, "", dtype, {id0, id1}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId reshape(LogicalId id0, LogicalId id1, std::source_location loc = std::source_location::current())
    {
        if (getNode(id1).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.reshape] Expected " << DType::INT32 << " for input 1, got: " << getNode(id1).dtype;
            Error::throw_err(ss.str(), loc);
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::RESHAPE, "", dtype, {id0, id1}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId permute(LogicalId id0, LogicalId id1, std::source_location loc = std::source_location::current())
    {
        if (getNode(id1).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.permute] Expected " << DType::INT32 << " for input 1, got: " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::PERMUTE, "", dtype, {id0, id1}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId slice(LogicalId id0, LogicalId id1, LogicalId id2, LogicalId id3,
                    std::source_location loc = std::source_location::current())
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
        TensorNode &node = allocateNode(OpType::SLICE, "", dtype, {id0, id1, id2, id3}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId scatter(LogicalId id0, LogicalId id1, LogicalId id2, LogicalId id3, LogicalId id4,
                      std::source_location loc = std::source_location::current())
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
            ss << "[Graph.scatter] DType mismatch between target (" << toString(getNode(id0).dtype) << ") and updates ("
               << toString(getNode(id1).dtype) << ")";
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::SCATTER, "", dtype, {id0, id1, id2, id3, id4}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId concat(std::vector<LogicalId> ids, LogicalId id1,
                     std::source_location loc = std::source_location::current())
    {
        if (ids.size() == 0)
        {
            Error::throw_err("[Graph.concat] Expected at least 1 input tensor, got 0.");
        }
        for (int i = 0; i < ids.size(); i++)
        {
            LogicalId id = ids[i];
            if (getNode(ids[0]).dtype != getNode(id).dtype)
            {
                std::stringstream ss;
                ss << "[Graph.concat] DType mismatch between tensor 0 and tensor " << i << ": " << getNode(ids[0]).dtype
                   << ", " << getNode(id).dtype;
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
        std::vector<LogicalId> children;
        children.push_back(id1);
        children.insert(children.end(), ids.begin(), ids.end());
        TensorNode &node = allocateNode(OpType::CONCAT, "", dtype, children, {}, {}, "", loc);
        return node.id;
    }

    LogicalId cast(LogicalId id0, DType dtype, std::source_location loc = std::source_location::current())
    {
        TensorNode &node = allocateNode(OpType::CAST, "", dtype, {id0}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId repeat(LogicalId id0, LogicalId repeats_id, LogicalId axis_id,
                     std::source_location loc = std::source_location::current())
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
        TensorNode &node = allocateNode(OpType::REPEAT, "", dtype, {id0, repeats_id, axis_id}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId arange(LogicalId id1, LogicalId id2, LogicalId id3,
                     std::source_location loc = std::source_location::current())
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
        TensorNode &node = allocateNode(OpType::ARANGE, "", DType::INT32, {id1, id2, id3}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId triu(LogicalId id0, LogicalId k_id, std::source_location loc = std::source_location::current())
    {
        if (getNode(k_id).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.triu] Expected " << DType::INT32 << " for input 1, got: " << getNode(k_id).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::TRIU, "", dtype, {id0, k_id}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId gather(LogicalId id0, LogicalId indices_id, std::source_location loc = std::source_location::current())
    {
        if (getNode(indices_id).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.gather] Expected " << DType::INT32 << " for input 1, got: " << getNode(indices_id).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::GATHER, "", dtype, {id0, indices_id}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId fill(LogicalId value_id, LogicalId shape_id, std::source_location loc = std::source_location::current())
    {
        if (getNode(shape_id).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.fill] Expected " << DType::INT32 << " for input 1, got: " << getNode(shape_id).dtype;
            Error::throw_err(ss.str());
        }
        DType dtype = getNode(value_id).dtype;
        TensorNode &node = allocateNode(OpType::FILL, "", dtype, {value_id, shape_id}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId im2col(LogicalId input_id, LogicalId kernel_size_id, LogicalId stride_id, LogicalId padding_id,
                     std::source_location loc = std::source_location::current())
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
        TensorNode &node =
            allocateNode(OpType::IM2COL, "", dtype, {input_id, kernel_size_id, stride_id, padding_id}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId log(LogicalId id0, std::source_location loc = std::source_location::current())
    {
        DType dtype = getNode(id0).dtype;
        TensorNode &node = allocateNode(OpType::LOG, "", dtype, {id0}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId argmax(LogicalId id0, LogicalId dim_id, LogicalId k_id,
                     std::source_location loc = std::source_location::current())
    {
        if (getNode(dim_id).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.argmax] Expected " << DType::INT32 << " for input 1, got: " << getNode(dim_id).dtype;
            Error::throw_err(ss.str());
        }
        if (getNode(k_id).dtype != DType::INT32)
        {
            std::stringstream ss;
            ss << "[Graph.argmax] Expected " << DType::INT32 << " for input 2, got: " << getNode(k_id).dtype;
            Error::throw_err(ss.str());
        }
        TensorNode &node = allocateNode(OpType::ARGMAX, "", DType::INT32, {id0, dim_id, k_id}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId lt(LogicalId id0, LogicalId id1, std::source_location loc = std::source_location::current())
    {
        if (getNode(id0).dtype != getNode(id1).dtype)
        {
            std::stringstream ss;
            ss << "[Graph.lt] DType mismatch: " << getNode(id0).dtype << ", " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        TensorNode &node = allocateNode(OpType::LT, "", DType::BOOL, {id0, id1}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId eq(LogicalId id0, LogicalId id1, std::source_location loc = std::source_location::current())
    {
        if (getNode(id0).dtype != getNode(id1).dtype)
        {
            std::stringstream ss;
            ss << "[Graph.eq] DType mismatch: " << getNode(id0).dtype << ", " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        TensorNode &node = allocateNode(OpType::EQ, "", DType::BOOL, {id0, id1}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId logical_and(LogicalId id0, LogicalId id1, std::source_location loc = std::source_location::current())
    {
        if (getNode(id0).dtype != DType::BOOL || getNode(id1).dtype != DType::BOOL)
        {
            std::stringstream ss;
            ss << "[Graph.logical_and] Inputs must be BOOL. Got " << getNode(id0).dtype << " and "
               << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        TensorNode &node = allocateNode(OpType::AND, "", DType::BOOL, {id0, id1}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId logical_or(LogicalId id0, LogicalId id1, std::source_location loc = std::source_location::current())
    {
        if (getNode(id0).dtype != DType::BOOL || getNode(id1).dtype != DType::BOOL)
        {
            std::stringstream ss;
            ss << "[Graph.logical_or] Inputs must be BOOL. Got " << getNode(id0).dtype << " and " << getNode(id1).dtype;
            Error::throw_err(ss.str());
        }
        TensorNode &node = allocateNode(OpType::OR, "", DType::BOOL, {id0, id1}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId logical_not(LogicalId id0, std::source_location loc = std::source_location::current())
    {
        if (getNode(id0).dtype != DType::BOOL)
        {
            std::stringstream ss;
            ss << "[Graph.logical_not] Input must be BOOL. Got " << getNode(id0).dtype;
            Error::throw_err(ss.str());
        }
        TensorNode &node = allocateNode(OpType::NOT, "", DType::BOOL, {id0}, {}, {}, "", loc);
        return node.id;
    }

    LogicalId unpack(LogicalId id0, DType dtype, std::source_location loc = std::source_location::current())
    {
        TensorNode &node = allocateNode(OpType::UNPACK, "", dtype, {id0}, {}, {}, "", loc);
        return node.id;
    }

    // Higher level stuff
    LogicalId repeat(LogicalId id, uint32_t repeats, uint32_t axis,
                     std::source_location loc = std::source_location::current())
    {
        if (repeats <= 1)
            return id;
        int32_t r = repeats, a = axis;
        return repeat(id, constant({1}, &r, DType::INT32), constant({1}, &a, DType::INT32), loc);
    }

    LogicalId fill(const LogicalId scalar_id, const std::vector<uint32_t> &shape,
                   std::source_location loc = std::source_location::current())
    {
        std::vector<int32_t> shape_int(shape.begin(), shape.end());
        LogicalId shape_node = constant({(uint32_t)shape_int.size()}, shape_int.data(), DType::INT32);
        return fill(scalar_id, shape_node, loc);
    }

    LogicalId fill(const float value, const std::vector<uint32_t> &shape,
                   std::source_location loc = std::source_location::current())
    {
        std::vector<int32_t> shape_int(shape.begin(), shape.end());
        LogicalId shape_node = constant({(uint32_t)shape_int.size()}, shape_int.data(), DType::INT32);
        return fill(constant({1}, &value, DType::FLOAT32), shape_node, loc);
    }

    LogicalId fill(const int32_t value, const std::vector<uint32_t> &shape,
                   std::source_location loc = std::source_location::current())
    {
        std::vector<int32_t> shape_int(shape.begin(), shape.end());
        LogicalId shape_node = constant({(uint32_t)shape_int.size()}, shape_int.data(), DType::INT32);
        return fill(constant({1}, &value, DType::INT32), shape_node, loc);
    }

    LogicalId concat(std::vector<LogicalId> ids, uint32_t axis,
                     std::source_location loc = std::source_location::current())
    {
        return concat(ids, constant({1}, &axis, DType::INT32), loc);
    }

    LogicalId relu(LogicalId scores, const std::vector<uint32_t> &shape,
                   std::source_location loc = std::source_location::current())
    {
        // 1. Create a zero tensor with matching shape
        LogicalId zeros = fill(0.0f, shape);

        // 2. Element-wise comparison: (0 < scores) -> BOOL tensor
        LogicalId is_positive = lt(zeros, scores);

        // 3. Cast BOOL -> FLOAT32 (1.0f for true, 0.0f for false)
        LogicalId mask_f32 = cast(is_positive, DType::FLOAT32);

        // 4. Element-wise multiply: x * (x > 0)
        LogicalId relu_scores = mul(scores, mask_f32);

        return relu_scores;
    }

    LogicalId constant(const std::vector<int32_t> &vals)
    {
        return constant({(uint32_t)vals.size()}, vals.data(), DType::INT32);
    }
};

inline bool isIsomorphic(const Graph &g1, LogicalId root1, const Graph &g2, LogicalId root2)
{
    const TensorNode &n1 = g1.getNode(root1);
    const TensorNode &n2 = g2.getNode(root2);

    if (n1.opType != n2.opType)
        return false;
    if (n1.opType == OpType::FUSED && n1.opName != n2.opName)
        return false;

    if (n1.opType == OpType::INPUT)
    {
        bool n1_has_hash = !n1.contentHash.empty();
        bool n2_has_hash = !n2.contentHash.empty();
        if (n1_has_hash && n2_has_hash)
            return n1.contentHash == n2.contentHash;
        return true;
    }

    if (n1.child_ids.size() != n2.child_ids.size())
        return false;

    for (uint64_t i = 0; i < n1.child_ids.size(); ++i)
    {
        if (!isIsomorphic(g1, n1.child_ids[i], g2, n2.child_ids[i]))
            return false;
    }

    return true;
}