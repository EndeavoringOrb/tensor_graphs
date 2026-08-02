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

inline std::string computeGraphHash(const Graph &graph, const std::vector<LogicalId> &rootIds)
{
    // Pass 1: Perform an iterative post-order DFS to establish evaluation order.
    std::vector<LogicalId> postOrder;
    std::unordered_set<LogicalId> visited;

    struct StackFrame
    {
        LogicalId id;
        size_t childIdx;
    };

    std::vector<StackFrame> stack;

    for (LogicalId r : rootIds)
    {
        if (visited.count(r))
            continue;

        visited.insert(r);
        stack.push_back({r, 0});

        while (!stack.empty())
        {
            LogicalId currId = stack.back().id;
            size_t &childIdx = stack.back().childIdx;
            const TensorNode &n = graph.getNode(currId);

            if (childIdx < n.child_ids.size())
            {
                LogicalId childId = n.child_ids[childIdx++];
                if (visited.insert(childId).second)
                {
                    stack.push_back({childId, 0});
                }
            }
            else
            {
                postOrder.push_back(currId);
                stack.pop_back();
            }
        }
    }

    // Pass 2: Iteratively compute hashes in post-order (leaves -> roots).
    std::unordered_map<LogicalId, std::string> memo;

    for (LogicalId id : postOrder)
    {
        const TensorNode &n = graph.getNode(id);
        SHA256 sha;
        sha.update(toString(n.opType));
        sha.update(":");
        for (auto s : n.getShape())
        {
            sha.update(std::to_string(s) + ",");
        }
        sha.update(":");
        sha.update(toString(n.dtype));
        if (graph.constantStaging.count(id))
        {
            sha.update(":");
            sha.update(n.contentHash);
        }
        for (LogicalId pid : n.child_ids)
        {
            sha.update(":");
            sha.update(memo.at(pid)); // Guaranteed to be computed already
        }
        memo[id] = sha.digest();
    }

    // Final Hash across all root nodes
    SHA256 finalSha;
    for (LogicalId r : rootIds)
    {
        finalSha.update(memo.at(r));
    }
    return finalSha.digest();
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