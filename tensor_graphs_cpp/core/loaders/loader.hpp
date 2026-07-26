#pragma once
#include <mutex>
#include <string>
#include <vector>

#include "core/loaders/safetensors.hpp"
#include "core/types.hpp"

#ifdef TG_OS_WINDOWS
#include <fcntl.h>
#include <io.h>
#include <share.h>
#endif

class FileRegistry
{
    std::vector<int> openFiles; // Store OS file descriptors, not FILE*
    std::unordered_map<std::string, uint32_t> pathToId;
    std::mutex mtx;
    std::unordered_map<std::string, std::shared_ptr<ModelLoader>>
        loaders; // Mapping of path -> polymorphic Loader instance
    std::unordered_map<LogicalId, std::pair<std::string, std::string>>
        weightSources; // Mapping of nodeId -> {path, tensor_name}

    uint32_t getFileId(const std::string &path)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (pathToId.count(path))
            return pathToId[path];

        int fd = -1;
        // Open file using low-level OS calls for stateless reading
#ifdef TG_OS_WINDOWS
        _wsopen_s(&fd, std::filesystem::path(path).c_str(), _O_RDONLY | _O_BINARY, _SH_DENYNO, 0);
#else
        fd = open(path.c_str(), O_RDONLY);
#endif
        if (fd < 0)
            Error::throw_err("Failed to open " + path);
        uint32_t id = openFiles.size();
        openFiles.push_back(fd);
        pathToId[path] = id;
        return id;
    }

    int getFd(uint32_t id)
    {
        return openFiles[id];
    }

  public:
    static FileRegistry &get()
    {
        static FileRegistry fr;
        return fr;
    }

    inline std::shared_ptr<ModelLoader> createLoader(const std::string &path)
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
                return std::make_shared<SafetensorsLoader>(path);
            }
            else
            {
                Error::throw_err("[LoaderFactory] No model files found in directory: " + path);
            }
        }
        else
        {
            fs::path p(path);
            if (p.extension() == ".safetensors")
            {
                return std::make_shared<SafetensorsLoader>(path);
            }
            else
            {
                Error::throw_err("[FileRegistry.createLoader] unrecognized file extension: " + p.extension().string());
            }
        }
    }

    void registerPath(const std::string &path)
    {
        if (loaders.find(path) == loaders.end())
        {
            loaders[path] = createLoader(path);
        }
    }

    bool hasTensor(const std::string &path, const std::string &name)
    {
        registerPath(path);
        const auto &it = loaders.find(path);
        if (it == loaders.end())
        {
            return false;
        }
        return it->second->hasTensor(name);
    }

    TensorMetadata getMetadata(const std::string &path, const std::string &name)
    {
        registerPath(path);
        const auto &it = loaders.find(path);
        if (it == loaders.end())
        {
            Error::throw_err("[FileRegistry.getMetadata] Path " + path + " is not registered");
        }
        if (it->second->hasTensor(name))
        {
            return it->second->getMetadata(name);
        }
        Error::throw_err("[FileRegistry.getMetadata] Tensor not found: " + name);
    }

    void registerNode(LogicalId nodeId, const std::string &path, const std::string &name)
    {
        weightSources[nodeId] = {path, name};
    }

    TensorMetadata getNodeMeta(LogicalId nodeId)
    {
        const auto &it = weightSources.find(nodeId);
        if (it == weightSources.end())
        {
            Error::throw_err("[FileRegistry.getNodeMeta] node id " + toString(nodeId) +
                             " is not registered"); // TODO: make build.py linter check if
                                                    // Error::throw_err calls inside a function
                                                    // start with [struct.func]. just use
                                                    // std::source_location
        }
        const auto &pair = weightSources.at(nodeId);
        return getMetadata(pair.first, pair.second);
    }

    int getNodeFd(LogicalId nodeId)
    {
        const auto &it = weightSources.find(nodeId);
        if (it == weightSources.end())
        {
            Error::throw_err("[FileRegistry.getNodeFd] node id " + toString(nodeId) + " is not registered");
        }
        const auto &pair = weightSources.at(nodeId);

        // Resolve the actual shard path using metadata
        TensorMetadata meta = getMetadata(pair.first, pair.second);
        return getFd(getFileId(meta.filePath));
    }
};