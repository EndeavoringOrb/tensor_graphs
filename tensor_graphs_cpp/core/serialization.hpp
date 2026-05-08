// File: tensor_graphs_cpp/core/serialization.hpp
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <type_traits>
#include <sstream>

class BinaryWriter {
    std::ostream& os;
public:
    BinaryWriter(std::ostream& os) : os(os) {}

    template<typename T>
    void write(const T& val);

    std::ostream& getStream() { return os; }
};

class BinaryReader {
    std::istream& is;
public:
    BinaryReader(std::istream& is) : is(is) {}

    template<typename T>
    void read(T& val);

    std::istream& getStream() { return is; }
};

// Global generic serializers for fundamental and standard types
template<typename T, std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>, int> = 0>
void tg_serialize(BinaryWriter& bw, const T& val) {
    bw.getStream().write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template<typename T, std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>, int> = 0>
void tg_deserialize(BinaryReader& br, T& val) {
    br.getStream().read(reinterpret_cast<char*>(&val), sizeof(T));
}

inline void tg_serialize(BinaryWriter& bw, const std::string& val) {
    uint32_t size = static_cast<uint32_t>(val.size());
    bw.write(size);
    if (size > 0) bw.getStream().write(val.data(), size);
}

inline void tg_deserialize(BinaryReader& br, std::string& val) {
    uint32_t size;
    br.read(size);
    if (size > 0) {
        val.resize(size);
        br.getStream().read(val.data(), size);
    } else {
        val.clear();
    }
}

template<typename T>
void tg_serialize(BinaryWriter& bw, const std::vector<T>& val) {
    uint32_t size = static_cast<uint32_t>(val.size());
    bw.write(size);
    if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T>) {
        if (size > 0) bw.getStream().write(reinterpret_cast<const char*>(val.data()), size * sizeof(T));
    } else {
        for (const auto& item : val) {
            bw.write(item);
        }
    }
}

template<typename T>
void tg_deserialize(BinaryReader& br, std::vector<T>& val) {
    uint32_t size;
    br.read(size);
    val.resize(size);
    if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T>) {
        if (size > 0) br.getStream().read(reinterpret_cast<char*>(val.data()), size * sizeof(T));
    } else {
        for (uint32_t i = 0; i < size; ++i) {
            br.read(val[i]);
        }
    }
}

template<typename K, typename V>
void tg_serialize(BinaryWriter& bw, const std::unordered_map<K, V>& val) {
    uint32_t size = static_cast<uint32_t>(val.size());
    bw.write(size);
    for (const auto& pair : val) {
        bw.write(pair.first);
        bw.write(pair.second);
    }
}

template<typename K, typename V>
void tg_deserialize(BinaryReader& br, std::unordered_map<K, V>& val) {
    uint32_t size;
    br.read(size);
    val.clear();
    for (uint32_t i = 0; i < size; ++i) {
        K k;
        V v;
        br.read(k);
        br.read(v);
        val[k] = std::move(v);
    }
}

template<typename T>
void BinaryWriter::write(const T& val) {
    tg_serialize(*this, val);
}

template<typename T>
void BinaryReader::read(T& val) {
    tg_deserialize(*this, val);
}

// Utility function to uniquely hash records/strings in-memory
template<typename T>
std::string serializeToString(const T& val) {
    std::stringstream ss(std::ios::binary | std::ios::out);
    BinaryWriter bw(ss);
    bw.write(val);
    return ss.str();
}
