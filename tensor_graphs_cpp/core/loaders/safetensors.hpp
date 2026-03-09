#pragma once
#include "core/types.hpp"

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
            throw std::runtime_error("[SafetensorsLoader.SafetensorsLoader] Could not open safetensors file: " + filepath);
        }

        // Read 8-byte header size (Safetensors spec relies on little-endian layout here)
        uint64_t headerSize = 0;
        if (!file.read(reinterpret_cast<char *>(&headerSize), sizeof(headerSize)))
        {
            throw std::runtime_error("[SafetensorsLoader.SafetensorsLoader] Could not read safetensors header size.");
        }

        // Read JSON Header
        jsonHeader.resize(headerSize);
        if (!file.read(&jsonHeader[0], headerSize))
        {
            throw std::runtime_error("[SafetensorsLoader.SafetensorsLoader] Could not read safetensors JSON header.");
        }

        dataStartOffset = 8 + headerSize;
        parseJson(jsonHeader);
    }

    const TensorMetadata &getMetadata(const std::string &name) const
    {
        auto it = metadata.find(name);
        if (it == metadata.end())
        {
            throw std::runtime_error("[SafetensorsLoader.getMetadata] Tensor not found in safetensors: " + name);
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
            throw std::runtime_error("[SafetensorsLoader.loadTensor] Destination buffer too small for tensor: " + name);
        }

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("[SafetensorsLoader.loadTensor] Could not open safetensors file: " + filename);
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