#pragma once
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/types.hpp"

struct TensorMetadata
{
    DType dtype = DType::FLOAT32;
    std::vector<uint32_t> shape;
    uint64_t dataOffsetStart = 0;
    uint64_t dataOffsetEnd = 0;
    std::string filePath;

    uint64_t sizeBytes() const
    {
        return dataOffsetEnd - dataOffsetStart;
    }
};

class ITensorStore
{
  public:
    virtual ~ITensorStore() = default;
    virtual bool isValid() const
    {
        return true;
    }
    virtual bool isWritable() const
    {
        return false;
    }
    virtual bool has(const std::string &name) const = 0;
    virtual TensorMetadata getMetadata(const std::string &name) const = 0;
    virtual std::vector<uint8_t> read(const std::string &name) const = 0;

    virtual void loadTensor(const std::string &name, void *dest, uint64_t dest_size) const
    {
        std::vector<uint8_t> data = read(name);
        if (data.size() > dest_size)
        {
            Error::throw_err("[ITensorStore.loadTensor] Destination buffer too small: " + std::to_string(dest_size) +
                             " < " + std::to_string(data.size()));
        }
        std::memcpy(dest, data.data(), data.size());
    }

    virtual void write(const std::string &name, const TensorView &view, const void *data, uint64_t size_bytes)
    {
        Error::throw_err("[ITensorStore.write] Store is read-only");
    }

    // Convenience overloads for LogicalId
    bool has(LogicalId id) const
    {
        return has(std::to_string(id.value));
    }

    std::vector<uint8_t> read(LogicalId id) const
    {
        return read(std::to_string(id.value));
    }

    void write(LogicalId id, const TensorView &view, const void *data, uint64_t size_bytes)
    {
        write(std::to_string(id.value), view, data, size_bytes);
    }

    TensorMetadata getMetadata(LogicalId id) const
    {
        return getMetadata(std::to_string(id.value));
    }
};

struct TensorLocation
{
    std::string storePath;
    std::string tensorName;
    std::shared_ptr<ITensorStore> store = nullptr;
};
