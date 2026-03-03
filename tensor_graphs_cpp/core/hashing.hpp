#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"

namespace Hashing
{

    // ---------------------------------------------------------
    // ZERO-DEPENDENCY SHA-256 IMPLEMENTATION
    // ---------------------------------------------------------
    class SHA256
    {
    private:
        uint32_t state[8];
        uint64_t bitlen;
        uint8_t data[64];
        uint32_t datalen;

        static constexpr uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

        static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
        static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
        static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
        static inline uint32_t ep0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
        static inline uint32_t ep1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
        static inline uint32_t sig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
        static inline uint32_t sig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

        void transform()
        {
            uint32_t a = state[0], b = state[1], c = state[2], d = state[3],
                     e = state[4], f = state[5], g = state[6], h = state[7];
            uint32_t w[64];

            for (int i = 0; i < 16; i++)
                w[i] = (static_cast<uint32_t>(data[i * 4]) << 24) |
                       (static_cast<uint32_t>(data[i * 4 + 1]) << 16) |
                       (static_cast<uint32_t>(data[i * 4 + 2]) << 8) |
                       (static_cast<uint32_t>(data[i * 4 + 3]));
            for (int i = 16; i < 64; i++)
                w[i] = sig1(w[i - 2]) + w[i - 7] + sig0(w[i - 15]) + w[i - 16];

            for (int i = 0; i < 64; i++)
            {
                uint32_t t1 = h + ep1(e) + ch(e, f, g) + K[i] + w[i];
                uint32_t t2 = ep0(a) + maj(a, b, c);
                h = g;
                g = f;
                f = e;
                e = d + t1;
                d = c;
                c = b;
                b = a;
                a = t1 + t2;
            }

            state[0] += a;
            state[1] += b;
            state[2] += c;
            state[3] += d;
            state[4] += e;
            state[5] += f;
            state[6] += g;
            state[7] += h;
        }

    public:
        SHA256()
        {
            state[0] = 0x6a09e667;
            state[1] = 0xbb67ae85;
            state[2] = 0x3c6ef372;
            state[3] = 0xa54ff53a;
            state[4] = 0x510e527f;
            state[5] = 0x9b05688c;
            state[6] = 0x1f83d9ab;
            state[7] = 0x5be0cd19;
            datalen = 0;
            bitlen = 0;
        }

        void update(const uint8_t *msg, size_t length)
        {
            for (size_t i = 0; i < length; i++)
            {
                data[datalen++] = msg[i];
                if (datalen == 64)
                {
                    transform();
                    bitlen += 512;
                    datalen = 0;
                }
            }
        }

        void update(const std::string &str)
        {
            update(reinterpret_cast<const uint8_t *>(str.data()), str.length());
        }

        std::string digest()
        {
            uint64_t i = datalen;
            if (datalen < 56)
            {
                data[i++] = 0x80;
                while (i < 56)
                    data[i++] = 0x00;
            }
            else
            {
                data[i++] = 0x80;
                while (i < 64)
                    data[i++] = 0x00;
                transform();
                std::fill(std::begin(data), std::end(data), 0);
            }

            bitlen += datalen * 8;
            for (int i = 0; i < 8; ++i)
            {
                data[63 - i] = static_cast<uint8_t>(bitlen >> (i * 8));
            }
            transform();

            std::stringstream ss;
            ss << std::hex << std::setfill('0');
            for (int j = 0; j < 8; j++)
            {
                ss << std::setw(8) << state[j];
            }
            return ss.str();
        }
    };

    // ---------------------------------------------------------
    // GRAPH HASHING ALGORITHMS
    // ---------------------------------------------------------
    namespace detail
    {
        inline std::string structuralHashImpl(uint32_t nodeId, const Graph &graph, MemoryManager &memManager, std::unordered_map<uint32_t, std::string> &memo)
        {
            if (memo.count(nodeId))
            {
                return memo[nodeId];
            }
            if (nodeId >= graph.nodes.size())
            {
                throw TensorGraphError("Invalid node ID encountered during hashing.");
            }

            const TensorNode &node = graph.nodes[nodeId];
            SHA256 sha;

            uint32_t opVal = static_cast<uint32_t>(node.opType);
            sha.update(reinterpret_cast<const uint8_t *>(&opVal), sizeof(opVal));

            if (node.opType == OpType::FUSED)
            {
                sha.update(node.opName);
                sha.update("|");
            }

            uint32_t dtypeVal = static_cast<uint32_t>(node.dtype);
            sha.update(reinterpret_cast<const uint8_t *>(&dtypeVal), sizeof(dtypeVal));

            uint32_t shapeRank = static_cast<uint32_t>(node.shape.size());
            sha.update(reinterpret_cast<const uint8_t *>(&shapeRank), sizeof(shapeRank));
            for (uint32_t dim : node.shape)
            {
                sha.update(reinterpret_cast<const uint8_t *>(&dim), sizeof(dim));
            }

            if (node.opType == OpType::INPUT)
            {
                if (node.storageType == StorageType::PERSISTENT)
                {
                    bool loadedInternally = false;
                    if (graph.weightSources.count(nodeId) && memManager.read(node.backend, nodeId) == nullptr)
                    {
                        const auto &source = graph.weightSources.at(nodeId);
                        auto &loader = graph.loaders.at(source.first);
                        const auto &meta = loader->getMetadata(source.second);

                        std::vector<uint8_t> tempBuffer(meta.sizeBytes());
                        loader->loadTensor(source.second, tempBuffer.data(), tempBuffer.size());
                        memManager.write(node.backend, nodeId, tempBuffer.data(), tempBuffer.size());
                        loadedInternally = true;
                    }

                    const uint8_t *data = memManager.read(node.backend, nodeId);
                    if (data)
                    {
                        uint64_t size = getSizeBytes(node.shape, node.dtype);
                        sha.update(data, size);
                    }
                    else
                    {
                        throw std::runtime_error("[Hashing::detail::structuralHashImpl no data returned from MemoryManager.read]");
                    }

                    // Unload if we were the ones to bring it into RAM
                    if (loadedInternally)
                    {
                        memManager.unload(node.backend, nodeId);
                    }
                }
                else
                {
                    uint32_t idVal = node.id;
                    sha.update(reinterpret_cast<const uint8_t *>(&idVal), sizeof(idVal));
                }
            }
            else
            {
                uint32_t numParents = static_cast<uint32_t>(node.parentIds.size());
                sha.update(reinterpret_cast<const uint8_t *>(&numParents), sizeof(numParents));

                const std::string delim = "|";
                for (uint32_t pid : node.parentIds)
                {
                    std::string ph = structuralHashImpl(pid, graph, memManager, memo);
                    sha.update(ph);
                    sha.update(delim);
                }
            }

            std::string result = sha.digest();
            memo[nodeId] = result;
            return result;
        }

        inline std::string patternHashImpl(uint32_t nodeId, const Graph &graph, std::unordered_map<uint32_t, std::string> &memo)
        {
            if (memo.count(nodeId))
            {
                return memo[nodeId];
            }
            if (nodeId >= graph.nodes.size())
            {
                throw TensorGraphError("Invalid node ID encountered during hashing.");
            }

            const TensorNode &node = graph.nodes[nodeId];
            SHA256 sha;

            // Hash OpType (Fixed size)
            uint32_t opVal = static_cast<uint32_t>(node.opType);
            sha.update(reinterpret_cast<const uint8_t *>(&opVal), sizeof(opVal));

            if (node.opType == OpType::FUSED)
            {
                sha.update(node.opName);
                sha.update("|");
            }

            if (node.opType == OpType::INPUT)
            {
                // Ignore shapes, dtypes, and exact identifiers.
                // Treats all inputs identically to enable flexible pattern matching for the Rewrite Engine.
                sha.update("*|"); // Delimited identifier
            }
            else
            {
                // Hash the fixed Parent Length
                uint32_t numParents = static_cast<uint32_t>(node.parentIds.size());
                sha.update(reinterpret_cast<const uint8_t *>(&numParents), sizeof(numParents));

                const std::string delim = "|";
                for (uint32_t pid : node.parentIds)
                {
                    std::string ph = patternHashImpl(pid, graph, memo);
                    sha.update(ph);
                    sha.update(delim);
                }
            }

            std::string result = sha.digest();
            memo[nodeId] = result;
            return result;
        }
    }

    inline std::string getStructuralHash(uint32_t nodeId, const Graph &graph, MemoryManager &memManager)
    {
        std::unordered_map<uint32_t, std::string> memo;
        return detail::structuralHashImpl(nodeId, graph, memManager, memo);
    }

    inline std::string getPatternHash(uint32_t nodeId, const Graph &graph)
    {
        std::unordered_map<uint32_t, std::string> memo;
        return detail::patternHashImpl(nodeId, graph, memo);
    }

} // namespace Hashing