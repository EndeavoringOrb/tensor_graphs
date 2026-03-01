#pragma once
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <sstream>
#include <fstream>
#include <vector>
#include <limits>
#include <cctype>
#include <list>
#include <json.hpp>
using json = nlohmann::json;

uint64_t countElements(std::vector<uint32_t> shape)
{
    uint64_t count = 1;
    for (uint32_t val : shape)
    {
        count *= val;
    }
    return count;
}

enum class DType : uint32_t
{
    FLOAT32,
    INT32,
    BF16,
    _COUNT
};

enum class OpType : uint32_t
{
    INPUT,

    ADD,
    MUL,
    DIVIDE,
    DOT,
    SQRT,
    SIN,
    COS,
    EXP,
    NEGATE,
    POWER,
    SUM,
    MAX,
    RESHAPE,
    PERMUTE,
    SLICE,
    CONCAT,
    CAST,
    REPEAT,
    ARANGE,
    TRIU,
    GATHER,
    FILL,
    WHERE,
    COPY_TO,
    IM2COL,

    CONV2D,
    FMA,
    GELU,
    GROUPNORM,
    RMS_NORM,
    ROPE_2D_CONSECUTIVE,
    ROPE,
    SIGMOID,
    SILU,
    SOFTMAX,
    TANH,
    UPSAMPLE_NEAREST
};

inline constexpr bool isAtomic(OpType type)
{
    return static_cast<uint32_t>(type) < static_cast<uint32_t>(OpType::CONV2D);
}

enum class Backend : uint32_t
{
    CPU
};

enum class StorageType : uint32_t
{
    TRANSIENT,
    PERSISTENT
};

// ---------------------------------------------------------
// CUSTOM ERROR TYPES
// ---------------------------------------------------------

struct TensorGraphError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

struct ViewOpValidationError : public TensorGraphError
{
    uint32_t nodeId = 0;
    OpType opType;
    std::vector<uint32_t> shape;
    size_t dimIndex;

    ViewOpValidationError(const std::string &msg, uint32_t nid, OpType op,
                          const std::vector<uint32_t> &s, size_t dim = 0)
        : TensorGraphError(msg), nodeId(nid), opType(op), shape(s), dimIndex(dim) {}
};

struct ShapeMismatchError : public TensorGraphError
{
    uint32_t nodeId = 0;
    std::vector<uint32_t> expectedShape;
    std::vector<uint32_t> actualShape;

    ShapeMismatchError(const std::string &msg, uint32_t nid,
                       const std::vector<uint32_t> &expected,
                       const std::vector<uint32_t> &actual)
        : TensorGraphError(msg), nodeId(nid),
          expectedShape(expected), actualShape(actual) {}
};

struct MemoryAllocationError : public TensorGraphError
{
    uint64_t requestedSize = 0;

    MemoryAllocationError(const std::string &msg, uint64_t size)
        : TensorGraphError(msg), requestedSize(size) {}
};

struct Dim
{
    uint32_t start;
    uint32_t stop;
};

struct Region
{
    std::vector<Dim> region;
};

inline bool regionsMatch(const Region &r1, const Region &r2)
{
    if (r1.region.size() != r2.region.size())
        return false;
    for (size_t i = 0; i < r1.region.size(); ++i)
    {
        if (r1.region[i].start != r2.region[i].start ||
            r1.region[i].stop != r2.region[i].stop)
        {
            return false;
        }
    }
    return true;
}

struct TensorView
{
    uint64_t baseOffset; // Offset into the MemoryManager's DeviceBuffer
    std::vector<uint32_t> shape;
    std::vector<int64_t> strides; // Strides in terms of elements, not bytes

    // Check if the physical layout perfectly matches the logical layout
    bool isContiguous() const
    {
        int64_t expectedStride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
        {
            if (shape[i] == 1)
                continue; // Stride doesn't matter for size 1
            if (strides[i] != expectedStride)
                return false;
            expectedStride *= shape[i];
        }
        return true;
    }

    // Helper to generate default contiguous strides for a given shape
    static std::vector<int64_t> calcContiguousStrides(const std::vector<uint32_t> &targetShape)
    {
        std::vector<int64_t> newStrides(targetShape.size());
        int64_t stride = 1;
        for (int i = static_cast<int>(targetShape.size()) - 1; i >= 0; --i)
        {
            newStrides[i] = stride;
            stride *= targetShape[i];
        }
        return newStrides;
    }
};

struct TensorNode
{
    uint32_t id;
    OpType opType;
    DType dtype;
    std::vector<uint32_t> parentIds;
    std::vector<uint32_t> shape;
    Backend backend = Backend::CPU;
    TensorView view;
};

inline const char *toString(DType dtype)
{
    switch (dtype)
    {
    case DType::FLOAT32:
        return "F32";
    case DType::INT32:
        return "I32";
    case DType::BF16:
        return "BF16";
    default:
        return "UNKNOWN_DTYPE";
    }
}

inline DType fromString(const std::string &str)
{
    for (uint32_t i = 0; i < static_cast<uint32_t>(DType::_COUNT); ++i)
    {
        DType dtype = static_cast<DType>(i);
        if (toString(dtype) == str)
            return dtype;
    }
    throw std::runtime_error("Unknown dtype: " + str); // TODO: make this throw custom error, and catch for that instead of generic runtime_error
}

inline const char *toString(OpType op)
{
    switch (op)
    {
    case OpType::INPUT:
        return "INPUT";
    case OpType::ADD:
        return "ADD";
    case OpType::MUL:
        return "MUL";
    case OpType::DIVIDE:
        return "DIVIDE";
    case OpType::DOT:
        return "DOT";
    case OpType::SQRT:
        return "SQRT";
    case OpType::SIN:
        return "SIN";
    case OpType::COS:
        return "COS";
    case OpType::EXP:
        return "EXP";
    case OpType::NEGATE:
        return "NEGATE";
    case OpType::POWER:
        return "POWER";
    case OpType::SUM:
        return "SUM";
    case OpType::MAX:
        return "MAX";
    case OpType::RESHAPE:
        return "RESHAPE";
    case OpType::PERMUTE:
        return "PERMUTE";
    case OpType::SLICE:
        return "SLICE";
    case OpType::CONCAT:
        return "CONCAT";
    case OpType::CAST:
        return "CAST";
    case OpType::REPEAT:
        return "REPEAT";
    case OpType::ARANGE:
        return "ARANGE";
    case OpType::TRIU:
        return "TRIU";
    case OpType::GATHER:
        return "GATHER";
    case OpType::FILL:
        return "FILL";
    case OpType::WHERE:
        return "WHERE";
    case OpType::COPY_TO:
        return "COPY_TO";
    case OpType::IM2COL:
        return "IM2COL";
    case OpType::CONV2D:
        return "CONV2D";
    case OpType::FMA:
        return "FMA";
    case OpType::GELU:
        return "GELU";
    case OpType::GROUPNORM:
        return "GROUPNORM";
    case OpType::RMS_NORM:
        return "RMS_NORM";
    case OpType::ROPE_2D_CONSECUTIVE:
        return "ROPE_2D_CONSECUTIVE";
    case OpType::ROPE:
        return "ROPE";
    case OpType::SIGMOID:
        return "SIGMOID";
    case OpType::SILU:
        return "SILU";
    case OpType::SOFTMAX:
        return "SOFTMAX";
    case OpType::TANH:
        return "TANH";
    case OpType::UPSAMPLE_NEAREST:
        return "UPSAMPLE_NEAREST";
    default:
        return "UNKNOWN_OP";
    }
}

inline const char *toString(Backend backend)
{
    switch (backend)
    {
    case Backend::CPU:
        return "CPU";
    default:
        return "UNKNOWN_BACKEND";
    }
}

inline const char *toString(StorageType storage)
{
    switch (storage)
    {
    case StorageType::TRANSIENT:
        return "TRANSIENT";
    case StorageType::PERSISTENT:
        return "PERSISTENT";
    default:
        return "UNKNOWN_STORAGE";
    }
}

inline std::string toString(const std::vector<uint32_t> &shape)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

inline std::ostream &operator<<(std::ostream &os, DType dtype) { return os << toString(dtype); }
inline std::ostream &operator<<(std::ostream &os, OpType op) { return os << toString(op); }
inline std::ostream &operator<<(std::ostream &os, Backend backend) { return os << toString(backend); }
inline std::ostream &operator<<(std::ostream &os, StorageType storage) { return os << toString(storage); }

struct ViewOps
{
    static TensorView repeat(uint32_t nodeId, const TensorView &input,
                             uint32_t dim, uint32_t repeats)
    {
        if (dim >= input.shape.size())
        {
            std::stringstream ss;
            ss << "[ViewOps.repeat] Dimension " << dim << " out of bounds for shape ";
            ss << toString(input.shape) << " (node " << nodeId << ")";
            throw ViewOpValidationError(ss.str(), nodeId, OpType::REPEAT,
                                        input.shape, dim);
        }

        if (input.shape[dim] != 1)
        {
            std::stringstream ss;
            ss << "[ViewOps.repeat] Can only broadcast dimension of size 1. "
               << "Dimension " << dim << " has size " << input.shape[dim]
               << ", requested repeats=" << repeats << " (node " << nodeId << ")";
            throw ViewOpValidationError(ss.str(), nodeId, OpType::REPEAT,
                                        input.shape, dim);
        }

        TensorView output = input;
        output.shape[dim] = repeats;
        output.strides[dim] = 0; // Broadcasting stride
        return output;
    }

    static TensorView reshape(uint32_t nodeId, const TensorView &input,
                              const std::vector<uint32_t> &newShape)
    {
        if (!input.isContiguous())
        {
            std::stringstream ss;
            ss << "[ViewOps.reshape] Cannot reshape non-contiguous tensor. "
               << "Current strides: [";
            for (size_t i = 0; i < input.strides.size(); ++i)
            {
                if (i > 0)
                    ss << ", ";
                ss << input.strides[i];
            }
            ss << "] (node " << nodeId << ")";
            throw ViewOpValidationError(ss.str(), nodeId, OpType::RESHAPE,
                                        input.shape);
        }

        uint64_t inputElements = countElements(input.shape);
        uint64_t outputElements = countElements(newShape);

        if (inputElements != outputElements)
        {
            std::stringstream ss;
            ss << "[ViewOps.reshape] Shape mismatch: expected "
               << outputElements << " elements, got " << inputElements
               << ". Reshaping " << toString(input.shape)
               << " to " << toString(newShape) << " (node " << nodeId << ")";
            throw ShapeMismatchError(ss.str(), nodeId, input.shape, newShape);
        }

        TensorView output = input;
        output.shape = newShape;
        output.strides = TensorView::calcContiguousStrides(newShape);
        return output;
    }

    static TensorView permute(const TensorView &input, const std::vector<uint32_t> &dims)
    {
        if (dims.size() != input.shape.size())
        {
            throw ViewOpValidationError(
                "[ViewOps.permute] Dimension count mismatch",
                0, OpType::PERMUTE, input.shape);
        }

        for (uint32_t d : dims)
        {
            if (d >= input.shape.size())
            {
                std::stringstream ss;
                ss << "[ViewOps.permute] Dimension " << d << " out of bounds for rank "
                   << input.shape.size();
                throw ViewOpValidationError(ss.str(), 0, OpType::PERMUTE,
                                            input.shape);
            }
        }

        TensorView output;
        output.baseOffset = input.baseOffset;
        output.shape.resize(dims.size());
        output.strides.resize(dims.size());

        for (size_t i = 0; i < dims.size(); ++i)
        {
            output.shape[i] = input.shape[dims[i]];
            output.strides[i] = input.strides[dims[i]];
        }
        return output;
    }

    static TensorView slice(const TensorView &input, uint32_t dim,
                            uint32_t start, uint32_t stop, uint32_t step)
    {
        if (dim >= input.shape.size())
        {
            throw ViewOpValidationError(
                "[ViewOps.slice] Dimension out of bounds",
                0, OpType::SLICE, input.shape, dim);
        }

        if (start > stop || stop > input.shape[dim])
        {
            std::stringstream ss;
            ss << "[ViewOps.slice] Invalid slice range: [" << start << ", "
               << stop << ") for dimension " << dim << " with size "
               << input.shape[dim];
            throw ViewOpValidationError(ss.str(), 0, OpType::SLICE,
                                        input.shape, dim);
        }

        if (step == 0)
        {
            throw ViewOpValidationError(
                "[ViewOps.slice] Step cannot be zero",
                0, OpType::SLICE, input.shape, dim);
        }

        TensorView output = input;
        output.baseOffset += start * input.strides[dim];
        output.shape[dim] = (stop - start + step - 1) / step;
        output.strides[dim] *= step;

        return output;
    }
};

struct MemBlock
{
    uint64_t offset;
    uint64_t sizeBytes;
    uint32_t nodeId; // Use UINT32_MAX to denote a free segment
    StorageType storageType;
    int32_t refCount;
    bool isLocked;
    float cost; // Track compute cost

    bool isFree() const
    {
        return nodeId == UINT32_MAX;
    }
};

struct DeviceBuffer
{
    std::vector<uint8_t> arena;
    uint64_t sizeBytes;

    // Unified physical memory layout representation
    std::list<MemBlock> blocks;
    // O(1) lookup map pointing directly to the list node for fast updates
    std::unordered_map<uint32_t, std::list<MemBlock>::iterator> allocationMap;

    DeviceBuffer(uint64_t _sizeBytes) : sizeBytes(_sizeBytes)
    {
        // Initialize with one massive free block
        MemBlock initialFree;
        initialFree.offset = 0;
        initialFree.sizeBytes = _sizeBytes;
        initialFree.nodeId = UINT32_MAX;
        initialFree.cost = 0.0f;
        initialFree.isLocked = false;
        blocks.push_back(initialFree);
    }

    // Arena not resized at construction so we can do allocation planning without actually making arena huge.
    void init()
    {
        arena.resize(sizeBytes);
    }

    std::list<MemBlock>::iterator findFreeSlot(uint64_t _sizeBytes)
    {
        for (auto it = blocks.begin(); it != blocks.end(); ++it)
        {
            if (it->isFree() && it->sizeBytes >= _sizeBytes)
            {
                return it;
            }
        }
        return blocks.end();
    }

    bool tryEvict(uint64_t needed)
    {
        auto left = blocks.begin();
        auto right = blocks.begin();
        uint64_t currentSize = 0;
        float currentCost = 0.0f;

        auto bestLeft = blocks.end();
        auto bestRight = blocks.end();
        float bestCost = std::numeric_limits<float>::infinity();
        uint64_t bestSize = std::numeric_limits<uint64_t>::max();

        while (right != blocks.end())
        {
            // Locked blocks act as an impenetrable wall. Reset the window.
            if (right->isLocked)
            {
                right++;
                left = right;
                currentSize = 0;
                currentCost = 0.0f;
                continue;
            }

            currentSize += right->sizeBytes;
            currentCost += right->cost;

            // Once the window meets our size requirement, evaluate and shrink
            while (currentSize >= needed)
            {
                // Save it if it is strictly cheaper, OR same cost but wastes less physical space
                if (currentCost < bestCost || (currentCost == bestCost && currentSize < bestSize))
                {
                    bestCost = currentCost;
                    bestSize = currentSize;
                    bestLeft = left;
                    bestRight = right;
                }

                // Break early if we've shrunk it to a single block
                if (left == right)
                {
                    break;
                }

                // Shrink from the left
                currentSize -= left->sizeBytes;
                currentCost -= left->isFree() ? 0.0f : left->cost;
                left++;
            }
            right++;
        }

        if (bestLeft != blocks.end())
        {
            auto it = bestLeft;
            uint64_t mergeOffset = bestLeft->offset;
            auto end_evict = std::next(bestRight);

            while (it != end_evict)
            {
                if (!it->isFree())
                {
                    allocationMap.erase(it->nodeId);
                }
                it++;
            }

            MemBlock mergedFree;
            mergedFree.offset = mergeOffset;
            mergedFree.sizeBytes = bestSize;
            mergedFree.nodeId = UINT32_MAX;
            mergedFree.cost = 0.0f;
            mergedFree.isLocked = false;

            blocks.insert(bestLeft, mergedFree);
            blocks.erase(bestLeft, end_evict);

            return true;
        }

        return false;
    }

    uint64_t allocate(uint32_t nodeId, uint64_t _sizeBytes, StorageType storageType, int32_t refCount, float cost)
    {
        // 1. If it's already cached, lock it and update
        auto mapIt = allocationMap.find(nodeId);
        if (mapIt != allocationMap.end())
        {
            auto blockIt = mapIt->second;
            blockIt->refCount = refCount;
            blockIt->isLocked = true;
            blockIt->cost = cost;
            return blockIt->offset;
        }

        // 2. See if there is space already available
        auto slotIt = findFreeSlot(_sizeBytes);

        // 3. If no space, try evict
        if (slotIt == blocks.end())
        {
            if (tryEvict(_sizeBytes))
            {
                slotIt = findFreeSlot(_sizeBytes);
            }
        }

        // 4. If still no space, eviction failed.
        if (slotIt == blocks.end())
        {
            throw std::runtime_error("Cannot allocate: Not enough contiguous space.");
        }

        // 5. Claim the free slot
        if (slotIt->sizeBytes > _sizeBytes)
        {
            // Split the block (leaves leftovers as free space naturally)
            MemBlock newAlloc;
            newAlloc.offset = slotIt->offset;
            newAlloc.sizeBytes = _sizeBytes;
            newAlloc.nodeId = nodeId;
            newAlloc.storageType = storageType;
            newAlloc.refCount = refCount;
            newAlloc.isLocked = true;
            newAlloc.cost = cost;

            auto insertedIt = blocks.insert(slotIt, newAlloc);
            allocationMap[nodeId] = insertedIt;

            // Shrink the leftover free block
            slotIt->offset += _sizeBytes;
            slotIt->sizeBytes -= _sizeBytes;
            return newAlloc.offset;
        }
        else
        {
            // Exact size match; overwrite free properties
            slotIt->nodeId = nodeId;
            slotIt->storageType = storageType;
            slotIt->refCount = refCount;
            slotIt->isLocked = true;
            slotIt->cost = cost;
            allocationMap[nodeId] = slotIt;
            return slotIt->offset;
        }
    }
};

struct MemoryManager
{
    std::unordered_map<Backend, DeviceBuffer> buffers;

    uint64_t allocate(Backend backend, uint32_t nodeId, uint64_t sizeBytes, StorageType storageType, int32_t refCount = 0, float cost = 0.0f)
    {
        auto it = buffers.find(backend);
        if (it == buffers.end())
        {
            throw std::runtime_error("Backend buffer not initialized in MemoryManager");
        }
        return it->second.allocate(nodeId, sizeBytes, storageType, refCount, cost);
    }
};

// ---------------------------------------------------------
// SAFETENSORS LOADER
// ---------------------------------------------------------

struct TensorMetadata
{
    DType dtype;
    std::vector<uint32_t> shape;
    uint64_t dataOffsetStart;
    uint64_t dataOffsetEnd;

    uint64_t sizeBytes() const
    {
        return dataOffsetEnd - dataOffsetStart;
    }
};

// TODO: make safetensors loader handle multiple files
class SafetensorsLoader
{
public:
    SafetensorsLoader(const std::string &filepath) : filename(filepath)
    {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open safetensors file: " + filepath);
        }

        // Read 8-byte header size (Safetensors spec relies on little-endian layout here)
        uint64_t headerSize = 0;
        if (!file.read(reinterpret_cast<char *>(&headerSize), sizeof(headerSize)))
        {
            throw std::runtime_error("Could not read safetensors header size.");
        }

        // Read JSON Header
        jsonHeader.resize(headerSize);
        if (!file.read(&jsonHeader[0], headerSize))
        {
            throw std::runtime_error("Could not read safetensors JSON header.");
        }

        dataStartOffset = 8 + headerSize;
        parseJson(jsonHeader);
    }

    const TensorMetadata &getMetadata(const std::string &name) const
    {
        auto it = metadata.find(name);
        if (it == metadata.end())
        {
            throw std::runtime_error("Tensor not found in safetensors: " + name);
        }
        return it->second;
    }

    bool hasTensor(const std::string &name) const
    {
        return metadata.find(name) != metadata.end();
    }

    void loadTensor(const std::string &name, void *dest, uint64_t destSize) const
    {
        const auto &meta = getMetadata(name);
        if (meta.sizeBytes() > destSize)
        {
            throw std::runtime_error("Destination buffer too small for tensor: " + name);
        }

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open safetensors file: " + filename);
        }

        file.seekg(dataStartOffset + meta.dataOffsetStart, std::ios::beg);
        file.read(reinterpret_cast<char *>(dest), meta.sizeBytes());
    }

private:
    std::string filename;
    uint64_t dataStartOffset;
    std::unordered_map<std::string, TensorMetadata> metadata;
    std::string jsonHeader;

    void parseJson(const std::string &json_str)
    {
        auto root = json::parse(json_str);

        for (const auto &[key, val] : root.items())
        {
            if (key == "__metadata__")
            {
                continue; // Skip global metadata
            }

            // Expecting tensor definition object: { "dtype": "...", "shape": [], "data_offsets": [] }
            if (!val.is_object())
                continue;

            TensorMetadata meta;
            bool valid = true;

            // 1. Parse DType
            std::string dtype_str = val.at("dtype").get<std::string>();
            try
            {
                meta.dtype = fromString(dtype_str);
            }
            catch (const std::runtime_error &)
            {
                valid = false;
            }

            // 2. Parse Shape
            auto shapeArr = val.at("shape");
            for (const auto &dim : shapeArr)
            {
                meta.shape.push_back(static_cast<uint32_t>(dim.get<int64_t>()));
            }

            // 3. Parse Data Offsets
            auto offsetArr = val.at("data_offsets");
            if (offsetArr.size() >= 2)
            {
                meta.dataOffsetStart = static_cast<uint64_t>(offsetArr[0].get<int64_t>());
                meta.dataOffsetEnd = static_cast<uint64_t>(offsetArr[1].get<int64_t>());
            }
            else
            {
                valid = false;
            }

            if (valid)
            {
                metadata[key] = std::move(meta);
            }
        }
    }
};

// ---------------------------------------------------------
// GRAPH AND SHAPE PROPAGATOR
// ---------------------------------------------------------

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

bool shapesMatch(const std::vector<uint32_t> &shape1, const std::vector<uint32_t> &shape2)
{
    if (shape1.size() != shape2.size())
        return false;
    for (size_t i = 0; i < shape1.size(); ++i)
    {
        if (shape1[i] != shape2[i])
            return false;
    }
    return true;
}

struct ShapePropagator
{
    std::vector<Region> forward(const TensorNode &node, const std::vector<TensorNode> &allNodes, const std::vector<std::vector<Region>> &parentRegions)
    {
        for (uint32_t pid : node.parentIds)
        {
            if (pid >= allNodes.size())
            {
                std::stringstream ss;
                ss << "[ShapePropagator.forward] Invalid parent ID " << pid << " for OpType " << node.opType;
                throw std::runtime_error(ss.str());
            }
        }
        switch (node.opType)
        {
        case OpType::ADD:
            return forwardAdd(node, allNodes, parentRegions);
        default:
            std::stringstream ss;
            ss << "[ShapePropagator.forward] Unsupported OpType for ShapePropagator.forward: " << node.opType;
            throw std::runtime_error(ss.str());
        }
    }

    // Output regions are the unique set of all parent regions
    std::vector<Region> forwardAdd(const TensorNode &node, const std::vector<TensorNode> &allNodes, const std::vector<std::vector<Region>> &parentRegions)
    {
        if (node.parentIds.size() != 2)
        {
            std::stringstream ss;
            ss << "[ShapePropagator.forwardAdd] ADD requires exactly 2 parents, got "
               << node.parentIds.size();
            throw std::runtime_error(ss.str());
        }

        uint32_t pid0 = node.parentIds[0];
        uint32_t pid1 = node.parentIds[1];
        const auto &parent0 = allNodes[pid0];
        const auto &parent1 = allNodes[pid1];

        if (!shapesMatch(parent0.shape, parent1.shape))
        {
            std::stringstream ss;
            ss << "[ShapePropagator.forwardAdd] Shape mismatch in ADD node: " << toString(parent0.shape) << ", " << toString(parent1.shape);
            throw std::runtime_error(ss.str());
        }

        std::vector<Region> outputRegions;

        auto regionExists = [&](const Region &r)
        {
            for (const auto &existing : outputRegions)
            {
                if (regionsMatch(existing, r))
                    return true;
            }
            return false;
        };

        for (const auto &region : parentRegions[0])
        {
            if (!regionExists(region))
            {
                outputRegions.push_back(region);
            }
        }
        for (const auto &region : parentRegions[1])
        {
            if (!regionExists(region))
            {
                outputRegions.push_back(region);
            }
        }

        return outputRegions;
    }

    // Dispatch to op-specific backward function
    std::vector<std::vector<Region>> backward(const TensorNode &node, const std::vector<TensorNode> &allNodes, const std::vector<Region> &outputRegions)
    {
        switch (node.opType)
        {
        case OpType::ADD:
            return backwardAdd(node, allNodes, outputRegions);
        default:
            std::stringstream ss;
            ss << "[ShapePropagator.backward] Unsupported OpType for ShapePropagator.backward: " << node.opType;
            throw std::runtime_error(ss.str());
        }
    }

    // For every dirty region in the output, BOTH corresponding inputs are also dirty
    std::vector<std::vector<Region>> backwardAdd(const TensorNode &node, const std::vector<TensorNode> &allNodes, const std::vector<Region> &outputRegions)
    {
        std::vector<std::vector<Region>> inputRegions(2);
        for (size_t i = 0; i < 2; ++i)
        {
            inputRegions[i] = outputRegions;
        }

        return inputRegions;
    }
};

// ---------------------------------------------------------
// KERNEL REGISTRY
// ---------------------------------------------------------

// A matching function checks the context of the requested operation to determine
// if the kernel supports the specific layout, rank, dimensions, or dtypes.
using MatchFunc = bool (*)(const std::vector<TensorNode> &inputs, const TensorNode &output);

// The execution function receives raw pointers dynamically mapped to the device buffer,
// alongside the TensorViews to access strides and shapes during execution.
using KernelFunc = void (*)(const std::vector<const void *> &inputs,
                            const std::vector<void *> &outputs,
                            const std::vector<TensorView> &inViews,
                            const std::vector<TensorView> &outViews);

struct KernelEntry
{
    OpType opType;
    Backend backend;
    MatchFunc match;
    KernelFunc run;
};

class KernelRegistry
{
public:
    static KernelRegistry &get()
    {
        static KernelRegistry instance;
        return instance;
    }

    void registerKernel(OpType op, Backend backend, MatchFunc match, KernelFunc run)
    {
        entries.push_back({op, backend, match, run});
    }

    KernelFunc findKernel(OpType op, Backend backend,
                          const std::vector<TensorNode> &inputs,
                          const TensorNode &output) const
    {
        for (const auto &entry : entries)
        {
            if (entry.opType == op && entry.backend == backend)
            {
                if (entry.match(inputs, output))
                {
                    return entry.run;
                }
            }
        }
        return nullptr; // Kernel not found
    }

private:
    std::vector<KernelEntry> entries;
};

// Helper struct and macro for clean static-time kernel registration
struct KernelRegistrar
{
    KernelRegistrar(OpType op, Backend backend, MatchFunc match, KernelFunc run)
    {
        KernelRegistry::get().registerKernel(op, backend, match, run);
    }
};

#define REGISTER_KERNEL(op, backend, match, run) static KernelRegistrar _registrar_##run(op, backend, match, run)