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

class Repo
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

    bool isValid() const
    {
        return valid;
    }

    bool has(LogicalId logicalId) const
    {
        return entries.count(logicalId) > 0;
    }

    std::vector<uint8_t> read(LogicalId logicalId) const
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

    void write(LogicalId logicalId, const TensorView &view, const void *data, uint64_t sizeBytes)
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