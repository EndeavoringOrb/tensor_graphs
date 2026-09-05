// tensor_graphs_cpp/core/repo.hpp
#pragma once
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/graph.hpp"
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

class ITensorStore
{
  public:
    virtual ~ITensorStore() = default;
    virtual bool isValid() const = 0;
    virtual bool has(LogicalId logical_id) const = 0;
    virtual std::vector<uint8_t> read(LogicalId logical_id) const = 0;
    virtual void write(LogicalId logical_id, const TensorView &view, const void *data, uint64_t size_bytes) = 0;
};

class InMemoryTensorStore : public ITensorStore
{
    mutable std::mutex store_mtx;
    std::unordered_map<LogicalId, std::vector<uint8_t>> data_store;
    std::unordered_map<LogicalId, TensorView> view_store;

  public:
    InMemoryTensorStore() = default;
    explicit InMemoryTensorStore(std::unordered_map<LogicalId, std::vector<uint8_t>> initial_data)
        : data_store(std::move(initial_data))
    {
    }

    bool isValid() const override
    {
        return true;
    }

    bool has(LogicalId logical_id) const override
    {
        std::lock_guard<std::mutex> lock(store_mtx);
        return data_store.count(logical_id) > 0;
    }

    std::vector<uint8_t> read(LogicalId logical_id) const override
    {
        std::lock_guard<std::mutex> lock(store_mtx);
        auto it = data_store.find(logical_id);
        if (it != data_store.end())
        {
            return it->second;
        }
        return {};
    }

    void write(LogicalId logical_id, const TensorView &view, const void *data, uint64_t size_bytes) override
    {
        std::lock_guard<std::mutex> lock(store_mtx);
        const uint8_t *byte_ptr = static_cast<const uint8_t *>(data);
        data_store[logical_id].assign(byte_ptr, byte_ptr + size_bytes);
        view_store[logical_id] = view;
    }

    const TensorView &getView(LogicalId logical_id) const
    {
        std::lock_guard<std::mutex> lock(store_mtx);
        return view_store.at(logical_id);
    }

    const std::unordered_map<LogicalId, std::vector<uint8_t>> &getData() const
    {
        return data_store;
    }
};

class Repo : public ITensorStore
{
    std::string metaPath;
    std::string dataPath;
    std::string graphHash;
    std::unordered_map<LogicalId, RefMetaEntry> entries;
    std::ofstream dataOut;
    std::ofstream metaOut;
    mutable std::ifstream dataIn;
    mutable std::mutex repoMtx;
    bool readOnly;
    bool valid = false;

  public:
    Repo(const std::string &path, const std::string &gHash, bool ro = true) : graphHash(gHash), readOnly(ro)
    {
        metaPath = path + ".refmeta";
        dataPath = path + ".reftensors";

        std::ifstream metaIn(metaPath, std::ios::binary);
        if (metaIn.is_open())
        {
            BinaryReader br(metaIn);
            std::string savedHash;
            br.read(savedHash);
            if (savedHash == graphHash)
            {
                valid = true;
                while (metaIn.peek() != EOF)
                {
                    RefMetaEntry e;
                    br.read(e);
                    entries[e.logicalId] = e;
                }
            }
            else
            {
                std::cout << "[Repo] Hash mismatch. Expected " << graphHash << ", got " << savedHash << std::endl;
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
        std::lock_guard<std::mutex> lock(repoMtx);
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

    bool has(LogicalId logicalId) const override
    {
        return entries.count(logicalId) > 0;
    }

    std::vector<uint8_t> read(LogicalId logicalId) const override
    {
        std::lock_guard<std::mutex> lock(repoMtx);
        if (!has(logicalId))
            return {};
        const auto &e = entries.at(logicalId);
        std::vector<uint8_t> data(e.sizeBytes);
        dataIn.seekg(e.offset, std::ios::beg);
        dataIn.read(reinterpret_cast<char *>(data.data()), e.sizeBytes);
        return data;
    }

    void write(LogicalId logicalId, const TensorView &view, const void *data, uint64_t sizeBytes) override
    {
        std::lock_guard<std::mutex> lock(repoMtx);
        if (readOnly || !valid)
            return;
        if (has(logicalId))
            return;

        uint64_t offset = dataOut.tellp();
        dataOut.write(reinterpret_cast<const char *>(data), sizeBytes);
        dataOut.flush();

        RefMetaEntry e;
        e.logicalId = logicalId;
        e.offset = offset;
        e.sizeBytes = sizeBytes;
        e.dtype = view.dtype;
        e.shape = view.getShape();
        entries[logicalId] = e;

        BinaryWriter bw(metaOut);
        bw.write(e);
        metaOut.flush();
    }
};