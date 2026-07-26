#pragma once
#include <string>
#include <vector>

#include "core/types.hpp"

struct TensorMetadata
{
    DType dtype;
    std::vector<uint32_t> shape;
    uint64_t dataOffsetStart;
    uint64_t dataOffsetEnd;
    std::string filePath;
};

class ModelLoader
{
  public:
    virtual ~ModelLoader() = default;
    virtual bool hasTensor(const std::string &name) const = 0;
    virtual TensorMetadata getMetadata(const std::string &name) const = 0;
    virtual void loadTensor(const std::string &name, void *dest, uint64_t destSize) const = 0;
};