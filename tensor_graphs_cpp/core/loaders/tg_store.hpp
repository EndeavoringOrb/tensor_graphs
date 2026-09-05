// tensor_graphs_cpp/core/loaders/tg_store.hpp
#pragma once
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/graph.hpp"
#include "core/loaders/store.hpp"
#include "core/types.hpp"

struct RefMetaEntry
{
    LogicalId logicalId;
    uint64_t offset;
    uint64_t sizeBytes;
    DType dtype;
    std::vector<uint32_t> shape;
};

inline void tg_serialize(BinaryWriter &bw, const RefMetaEntry &val)
{
    bw.write(val.logicalId);
    bw.write(val.offset);
    bw.write(val.sizeBytes);
    bw.write(val.dtype);
    bw.write(val.shape);
}

inline void tg_deserialize(BinaryReader &br, RefMetaEntry &val)
{
    br.read(val.logicalId);
    br.read(val.offset);
    br.read(val.sizeBytes);
    br.read(val.dtype);
    br.read(val.shape);
}

class InMemoryTensorStore : public ITensorStore
{
    mutable std::mutex storeMtx;
    std::unordered_map<std::string, std::vector<uint8_t>> dataStore;
    std::unordered_map<std::string, TensorView> viewStore;

  public:
    InMemoryTensorStore() = default;
    explicit InMemoryTensorStore(std::unordered_map<LogicalId, std::vector<uint8_t>> initial_data)
    {
        for (auto &pair : initial_data)
        {
            dataStore[std::to_string(pair.first.value)] = std::move(pair.second);
        }
    }

    bool isValid() const override
    {
        return true;
    }

    bool isWritable() const override
    {
        return true;
    }

    bool has(const std::string &name) const override
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        return dataStore.count(name) > 0;
    }

    bool has(LogicalId id) const
    {
        return has(std::to_string(id.value));
    }

    std::vector<uint8_t> read(const std::string &name) const override
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        auto it = dataStore.find(name);
        if (it != dataStore.end())
        {
            return it->second;
        }
        return {};
    }

    std::vector<uint8_t> read(LogicalId id) const
    {
        return read(std::to_string(id.value));
    }

    TensorMetadata getMetadata(const std::string &name) const override
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        auto it = viewStore.find(name);
        if (it == viewStore.end())
        {
            Error::throw_err("[InMemoryTensorStore.getMetadata] Tensor not found: " + name);
        }
        TensorMetadata meta;
        meta.dtype = it->second.dtype;
        meta.shape = it->second.getShape();
        meta.dataOffsetStart = 0;
        meta.dataOffsetEnd = dataStore.at(name).size();
        meta.filePath = ":memory:";
        return meta;
    }

    TensorMetadata getMetadata(LogicalId id) const
    {
        return getMetadata(std::to_string(id.value));
    }

    void write(const std::string &name, const TensorView &view, const void *data, uint64_t size_bytes) override
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        const uint8_t *byte_ptr = static_cast<const uint8_t *>(data);
        dataStore[name].assign(byte_ptr, byte_ptr + size_bytes);
        viewStore[name] = view;
    }

    void write(LogicalId id, const TensorView &view, const void *data, uint64_t size_bytes)
    {
        write(std::to_string(id.value), view, data, size_bytes);
    }

    const TensorView &getView(LogicalId logical_id) const
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        return viewStore.at(std::to_string(logical_id.value));
    }

    const std::unordered_map<std::string, std::vector<uint8_t>> &getData() const
    {
        return dataStore;
    }
};

class TGStore : public ITensorStore
{
    std::string basePath;
    std::string metaPath;
    std::string dataPath;
    std::string graphHash;
    std::unordered_map<std::string, RefMetaEntry> entries;
    std::unordered_map<LogicalId, RefMetaEntry> idEntries;
    std::ofstream dataOut;
    std::ofstream metaOut;
    mutable std::ifstream dataIn;
    mutable std::mutex storeMtx;
    bool readOnly;
    bool valid = false;

  public:
    TGStore(const std::string &path, const std::string &gHash, bool ro = true)
        : basePath(path), graphHash(gHash), readOnly(ro)
    {
        metaPath = path + ".refmeta";
        dataPath = path + ".reftensors";

        std::ifstream metaIn(metaPath, std::ios::binary);
        if (metaIn.is_open())
        {
            BinaryReader br(metaIn);
            std::string saved_hash;
            br.read(saved_hash);
            if (saved_hash == graphHash)
            {
                valid = true;
                while (metaIn.peek() != EOF)
                {
                    RefMetaEntry e;
                    br.read(e);
                    idEntries[e.logicalId] = e;
                    entries[std::to_string(e.logicalId.value)] = e;
                }
            }
            else
            {
                std::cout << "[TGStore] Hash mismatch. Expected " << graphHash << ", got " << saved_hash << std::endl;
            }
        }

        if (!valid && !readOnly)
        {
            std::error_code ec;
            std::filesystem::remove(metaPath, ec);
            std::filesystem::remove(dataPath, ec);
            std::filesystem::path meta_parent = std::filesystem::path(metaPath).parent_path();
            if (!meta_parent.empty())
                std::filesystem::create_directories(meta_parent);
            metaOut.open(metaPath, std::ios::binary | std::ios::trunc);
            BinaryWriter bw(metaOut);
            bw.write(graphHash);
            metaOut.flush();
            dataOut.open(dataPath, std::ios::binary | std::ios::trunc);
            valid = true;
        }
        else if (valid && !readOnly)
        {
            metaOut.open(metaPath, std::ios::binary | std::ios::app);
            dataOut.open(dataPath, std::ios::binary | std::ios::app);
        }

        if (valid)
        {
            dataIn.open(dataPath, std::ios::binary);
        }
    }

    bool enableWriting()
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        if (!readOnly && valid)
            return true;
        readOnly = false;
        if (!valid)
        {
            std::error_code ec;
            std::filesystem::remove(metaPath, ec);
            std::filesystem::remove(dataPath, ec);
            std::filesystem::path meta_parent = std::filesystem::path(metaPath).parent_path();
            if (!meta_parent.empty())
                std::filesystem::create_directories(meta_parent);
            metaOut.open(metaPath, std::ios::binary | std::ios::trunc);
            BinaryWriter bw(metaOut);
            bw.write(graphHash);
            metaOut.flush();
            dataOut.open(dataPath, std::ios::binary | std::ios::trunc);
            valid = true;
        }
        else
        {
            if (!metaOut.is_open())
                metaOut.open(metaPath, std::ios::binary | std::ios::app);
            if (!dataOut.is_open())
                dataOut.open(dataPath, std::ios::binary | std::ios::app);
        }
        if (!dataIn.is_open())
        {
            dataIn.open(dataPath, std::ios::binary);
        }
        return valid;
    }

    bool isValid() const override
    {
        return valid;
    }

    bool isWritable() const override
    {
        return !readOnly && valid;
    }

    bool has(const std::string &name) const override
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        return entries.count(name) > 0;
    }

    bool has(LogicalId id) const
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        return idEntries.count(id) > 0 || entries.count(std::to_string(id.value)) > 0;
    }

    TensorMetadata getMetadata(const std::string &name) const override
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        auto it = entries.find(name);
        if (it == entries.end())
        {
            Error::throw_err("[TGStore.getMetadata] Tensor not found: " + name);
        }
        const auto &e = it->second;
        return TensorMetadata{e.dtype, e.shape, e.offset, e.offset + e.sizeBytes, dataPath};
    }

    TensorMetadata getMetadata(LogicalId id) const
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        auto it = idEntries.find(id);
        if (it != idEntries.end())
        {
            const auto &e = it->second;
            return TensorMetadata{e.dtype, e.shape, e.offset, e.offset + e.sizeBytes, dataPath};
        }
        auto sit = entries.find(std::to_string(id.value));
        if (sit != entries.end())
        {
            const auto &e = sit->second;
            return TensorMetadata{e.dtype, e.shape, e.offset, e.offset + e.sizeBytes, dataPath};
        }
        Error::throw_err("[TGStore.getMetadata] Tensor not found: " + toString(id));
    }

    std::vector<uint8_t> read(const std::string &name) const override
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        auto it = entries.find(name);
        if (it == entries.end())
        {
            return {};
        }
        const auto &e = it->second;
        std::vector<uint8_t> data(e.sizeBytes);
        if (!dataIn.is_open())
        {
            dataIn.open(dataPath, std::ios::binary);
        }
        dataIn.seekg(e.offset, std::ios::beg);
        dataIn.read(reinterpret_cast<char *>(data.data()), e.sizeBytes);
        return data;
    }

    std::vector<uint8_t> read(LogicalId id) const
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        auto it = idEntries.find(id);
        if (it != idEntries.end())
        {
            const auto &e = it->second;
            std::vector<uint8_t> data(e.sizeBytes);
            if (!dataIn.is_open())
            {
                dataIn.open(dataPath, std::ios::binary);
            }
            dataIn.seekg(e.offset, std::ios::beg);
            dataIn.read(reinterpret_cast<char *>(data.data()), e.sizeBytes);
            return data;
        }
        auto sit = entries.find(std::to_string(id.value));
        if (sit != entries.end())
        {
            const auto &e = sit->second;
            std::vector<uint8_t> data(e.sizeBytes);
            if (!dataIn.is_open())
            {
                dataIn.open(dataPath, std::ios::binary);
            }
            dataIn.seekg(e.offset, std::ios::beg);
            dataIn.read(reinterpret_cast<char *>(data.data()), e.sizeBytes);
            return data;
        }
        return {};
    }

    void loadTensor(const std::string &name, void *dest, uint64_t dest_size) const override
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        auto it = entries.find(name);
        if (it == entries.end())
        {
            Error::throw_err("[TGStore.loadTensor] Tensor not found: " + name);
        }
        const auto &e = it->second;
        if (e.sizeBytes > dest_size)
        {
            Error::throw_err("[TGStore.loadTensor] Destination buffer too small for '" + name +
                             "' (" + std::to_string(dest_size) + " < " + std::to_string(e.sizeBytes) + ")");
        }
        if (!dataIn.is_open())
        {
            dataIn.open(dataPath, std::ios::binary);
            if (!dataIn.is_open())
            {
                Error::throw_err("[TGStore.loadTensor] Could not open data file: " + dataPath);
            }
        }
        dataIn.seekg(e.offset, std::ios::beg);
        dataIn.read(reinterpret_cast<char *>(dest), e.sizeBytes);
    }

    void write(LogicalId logical_id, const TensorView &view, const void *data, uint64_t size_bytes)
    {
        std::lock_guard<std::mutex> lock(storeMtx);
        if (readOnly || !valid)
            return;
        if (idEntries.count(logical_id) > 0)
            return;

        uint64_t offset = dataOut.tellp();
        dataOut.write(reinterpret_cast<const char *>(data), size_bytes);
        dataOut.flush();

        RefMetaEntry e;
        e.logicalId = logical_id;
        e.offset = offset;
        e.sizeBytes = size_bytes;
        e.dtype = view.dtype;
        e.shape = view.getShape();
        idEntries[logical_id] = e;
        entries[std::to_string(logical_id.value)] = e;

        BinaryWriter bw(metaOut);
        bw.write(e);
        metaOut.flush();
    }

    void write(const std::string &name, const TensorView &view, const void *data, uint64_t size_bytes) override
    {
        LogicalId id;
        try
        {
            id = LogicalId{static_cast<uint32_t>(std::stoul(name))};
        }
        catch (...)
        {
            id = LogicalId{static_cast<uint32_t>(std::hash<std::string>{}(name))};
        }
        write(id, view, data, size_bytes);
    }

    const std::string &getDataPath() const
    {
        return dataPath;
    }

    const std::string &getMetaPath() const
    {
        return metaPath;
    }

    const std::string &getBasePath() const
    {
        return basePath;
    }
};
