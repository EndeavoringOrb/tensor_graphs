#pragma once
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/loaders/safetensors.hpp"
#include "core/loaders/store.hpp"
#include "core/loaders/tg_store.hpp"
#include "core/types.hpp"

#ifdef TG_OS_WINDOWS
#include <fcntl.h>
#include <io.h>
#include <share.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

// ============================================================================
// Layer 1: OS File & Descriptor Manager
// ============================================================================
class FileDescriptorManager
{
    std::vector<int> openFiles; // Store OS file descriptors, not FILE*
    std::unordered_map<std::string, uint32_t> pathToId;
    std::mutex mtx;

  public:
    static FileDescriptorManager &get()
    {
        static FileDescriptorManager instance;
        return instance;
    }

    uint32_t getFileId(const std::string &path)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (pathToId.count(path))
            return pathToId[path];

        int fd = -1;
#ifdef TG_OS_WINDOWS
        _wsopen_s(&fd, std::filesystem::path(path).c_str(), _O_RDONLY | _O_BINARY, _SH_DENYNO, 0);
#else
        fd = open(path.c_str(), O_RDONLY);
#endif
        if (fd < 0)
            Error::throw_err("Failed to open " + path);
        uint32_t id = static_cast<uint32_t>(openFiles.size());
        openFiles.push_back(fd);
        pathToId[path] = id;
        return id;
    }

    int getFd(uint32_t id)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (id >= openFiles.size())
            Error::throw_err("Invalid file descriptor id: " + std::to_string(id));
        return openFiles[id];
    }

    int getFdForPath(const std::string &path)
    {
        return getFd(getFileId(path));
    }
};

// ============================================================================
// Resolver Layer: Resolves LogicalId & Paths to Stores and OS Descriptors
// ============================================================================
class TensorResolver
{
    std::mutex mtx;
    std::unordered_map<std::string, std::shared_ptr<ITensorStore>> stores; // Path -> Store instance
    std::unordered_map<LogicalId, TensorLocation> nodeLocations;           // NodeId -> Location

  public:
    static TensorResolver &get()
    {
        static TensorResolver tr;
        return tr;
    }

    uint32_t getFileId(const std::string &path)
    {
        return FileDescriptorManager::get().getFileId(path);
    }

    int getFd(uint32_t id)
    {
        return FileDescriptorManager::get().getFd(id);
    }

    int getFdForPath(const std::string &path)
    {
        return FileDescriptorManager::get().getFdForPath(path);
    }

    inline std::shared_ptr<ITensorStore> createStore(const std::string &path)
    {
        namespace fs = std::filesystem;
        if (fs::is_directory(path))
        {
            bool has_safetensors = false;
            for (const auto &entry : fs::directory_iterator(path))
            {
                if (entry.path().extension() == ".safetensors")
                    has_safetensors = true;
            }
            if (has_safetensors)
            {
                return std::make_shared<SafetensorsStore>(path);
            }
            else
            {
                Error::throw_err("[TensorResolver.createStore] No model files found in directory: " + path);
            }
        }
        else
        {
            fs::path p(path);
            if (p.extension() == ".safetensors")
            {
                return std::make_shared<SafetensorsStore>(path);
            }
            else if (p.extension() == ".refmeta" || p.extension() == ".reftensors")
            {
                std::string base_path = (p.parent_path() / p.stem()).string();
                return std::make_shared<TGStore>(base_path, "", true);
            }
            else
            {
                Error::throw_err("[TensorResolver.createStore] unrecognized file extension: " + p.extension().string());
            }
        }
    }

    void registerStore(const std::string &path, std::shared_ptr<ITensorStore> store)
    {
        std::lock_guard<std::mutex> lock(mtx);
        stores[path] = store;
    }

    void registerPath(const std::string &path)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (stores.find(path) == stores.end())
        {
            stores[path] = createStore(path);
        }
    }

    std::shared_ptr<ITensorStore> getStore(const std::string &path)
    {
        registerPath(path);
        std::lock_guard<std::mutex> lock(mtx);
        auto it = stores.find(path);
        if (it == stores.end())
        {
            Error::throw_err("[TensorResolver.getStore] Store not found: " + path);
        }
        return it->second;
    }

    bool hasTensor(const std::string &path, const std::string &name)
    {
        registerPath(path);
        std::lock_guard<std::mutex> lock(mtx);
        const auto &it = stores.find(path);
        if (it == stores.end())
        {
            return false;
        }
        return it->second->has(name);
    }

    TensorMetadata getMetadata(const std::string &path, const std::string &name)
    {
        registerPath(path);
        std::lock_guard<std::mutex> lock(mtx);
        const auto &it = stores.find(path);
        if (it == stores.end())
        {
            Error::throw_err("[TensorResolver.getMetadata] Path " + path + " is not registered");
        }
        if (it->second->has(name))
        {
            return it->second->getMetadata(name);
        }
        Error::throw_err("[TensorResolver.getMetadata] Tensor not found: " + name);
    }

    void registerNode(LogicalId nodeId, const std::string &path, const std::string &name)
    {
        registerPath(path);
        std::lock_guard<std::mutex> lock(mtx);
        TensorLocation loc;
        loc.storePath = path;
        loc.tensorName = name;
        loc.store = stores[path];
        nodeLocations[nodeId] = std::move(loc);
    }

    void registerNode(LogicalId nodeId, std::shared_ptr<ITensorStore> store, const std::string &name)
    {
        std::lock_guard<std::mutex> lock(mtx);
        TensorLocation loc;
        loc.storePath = "";
        loc.tensorName = name;
        loc.store = store;
        nodeLocations[nodeId] = std::move(loc);
    }

    bool hasNode(LogicalId nodeId)
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = nodeLocations.find(nodeId);
        if (it == nodeLocations.end() || !it->second.store)
            return false;
        return it->second.store->has(it->second.tensorName);
    }

    TensorMetadata getNodeMeta(LogicalId nodeId)
    {
        TensorLocation loc;
        {
            std::lock_guard<std::mutex> lock(mtx);
            const auto &it = nodeLocations.find(nodeId);
            if (it == nodeLocations.end())
            {
                Error::throw_err("[TensorResolver.getNodeMeta] node id " + toString(nodeId) + " is not registered");
            }
            loc = it->second;
        }
        if (loc.store)
        {
            return loc.store->getMetadata(loc.tensorName);
        }
        return getMetadata(loc.storePath, loc.tensorName);
    }

    int getNodeFd(LogicalId nodeId)
    {
        TensorMetadata meta = getNodeMeta(nodeId);
        return FileDescriptorManager::get().getFdForPath(meta.filePath);
    }

    std::vector<uint8_t> readNode(LogicalId nodeId)
    {
        TensorLocation loc;
        {
            std::lock_guard<std::mutex> lock(mtx);
            const auto &it = nodeLocations.find(nodeId);
            if (it == nodeLocations.end())
            {
                Error::throw_err("[TensorResolver.readNode] node id " + toString(nodeId) + " is not registered");
            }
            loc = it->second;
        }
        if (loc.store)
        {
            return loc.store->read(loc.tensorName);
        }
        auto store = getStore(loc.storePath);
        return store->read(loc.tensorName);
    }
};
