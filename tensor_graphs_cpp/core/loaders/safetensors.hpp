#pragma once
#include "core/types.hpp"
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

struct TensorMetadata
{
    DType dtype;
    std::vector<uint32_t> shape;
    uint64_t dataOffsetStart;
    uint64_t dataOffsetEnd;
    size_t fileIndex; // Index into the files vector

    uint64_t sizeBytes() const
    {
        return dataOffsetEnd - dataOffsetStart;
    }
};

class SafetensorsLoader
{
public:
    SafetensorsLoader(const std::string &path)
    {
        std::vector<std::string> filepaths;

        if (fs::is_directory(path))
        {
            for (const auto &entry : fs::directory_iterator(path))
            {
                if (entry.path().extension() == ".safetensors")
                {
                    filepaths.push_back(entry.path().string());
                }
            }
            if (filepaths.empty())
            {
                Error::throw_err("[SafetensorsLoader] No .safetensors files found in directory: " + path);
            }
            std::sort(filepaths.begin(), filepaths.end());
        }
        else
        {
            if (!fs::exists(path))
            {
                Error::throw_err("[SafetensorsLoader] Safetensors file not found: " + path);
            }
            filepaths.push_back(path);
        }

        for (size_t i = 0; i < filepaths.size(); ++i)
        {
            loadFile(filepaths[i], i);
        }
    }

    const TensorMetadata &getMetadata(const std::string &name) const
    {
        auto it = metadata.find(name);
        if (it == metadata.end())
        {
            Error::throw_err("[SafetensorsLoader.getMetadata] Tensor not found: " + name);
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
            Error::throw_err("[SafetensorsLoader.loadTensor] Destination buffer too small for tensor '" + name + "' (dst=" + std::to_string(destSize) + "), (tensor_size=" + std::to_string(meta.sizeBytes()) + ")");
        }

        const std::string &fname = files[meta.fileIndex].path;
        std::ifstream file(fname, std::ios::binary);
        if (!file.is_open())
        {
            Error::throw_err("[SafetensorsLoader.loadTensor] Could not open file: " + fname);
        }

        file.seekg(files[meta.fileIndex].dataStartOffset + meta.dataOffsetStart, std::ios::beg);
        file.read(reinterpret_cast<char *>(dest), meta.sizeBytes());
    }

private:
    struct FileInfo
    {
        std::string path;
        uint64_t dataStartOffset;
    };

    std::vector<FileInfo> files;
    std::unordered_map<std::string, TensorMetadata> metadata;

    void loadFile(const std::string &filepath, size_t fileIdx)
    {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            Error::throw_err("[SafetensorsLoader] Could not open: " + filepath);
        }

        uint64_t headerSize = 0;
        if (!file.read(reinterpret_cast<char *>(&headerSize), sizeof(headerSize)))
        {
            Error::throw_err("[SafetensorsLoader] Could not read header size from: " + filepath);
        }

        std::string jsonHeader(headerSize, '\0');
        if (!file.read(&jsonHeader[0], headerSize))
        {
            Error::throw_err("[SafetensorsLoader] Could not read JSON header from: " + filepath);
        }

        FileInfo info;
        info.path = filepath;
        info.dataStartOffset = 8 + headerSize;
        files.push_back(info);

        auto root = json::parse(jsonHeader);
        for (const auto &[key, val] : root.items())
        {
            if (key == "__metadata__" || !val.is_object())
                continue;

            TensorMetadata meta;
            meta.fileIndex = fileIdx;

            // 1. DType
            meta.dtype = fromString(val.at("dtype").get<std::string>());

            // 2. Shape
            for (const auto &dim : val.at("shape"))
            {
                meta.shape.push_back(static_cast<uint32_t>(dim.get<int64_t>()));
            }

            // 3. Offsets
            auto offsets = val.at("data_offsets");
            meta.dataOffsetStart = static_cast<uint64_t>(offsets[0].get<int64_t>());
            meta.dataOffsetEnd = static_cast<uint64_t>(offsets[1].get<int64_t>());

            if (metadata.count(key))
            {
                Error::throw_err("[SafetensorsLoader] Duplicate tensor '" + key + "' found in shards.");
            }
            metadata[key] = std::move(meta);
        }
    }
};