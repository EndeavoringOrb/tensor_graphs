#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>

// A matching function checks the context of the requested operation to determine
// if the kernel supports the specific layout, rank, dimensions, or dtypes.
using MatchFunc = bool (*)(const std::vector<TensorNode> &inputs, const TensorNode &output);

// The execution function receives raw pointers dynamically mapped to the device buffer,
// alongside the TensorViews to access strides and shapes during execution.
using KernelFunc = void (*)(const std::vector<const void *> &inputs,
                            const std::vector<void *> &outputs,
                            const std::vector<TensorView> &inViews,
                            const std::vector<TensorView> &outViews);

using ReferenceFactory = uint32_t (*)(const std::vector<uint32_t> &inputs, Graph &graph);

struct ReferenceGraphEntry
{
    size_t numInputs;
    ReferenceFactory factory;
};

class ReferenceGraphRegistry
{
public:
    static ReferenceGraphRegistry &get()
    {
        static ReferenceGraphRegistry instance;
        return instance;
    }

    void registerFactory(const std::string &name, size_t numInputs, ReferenceFactory factory)
    {
        factories[name] = {numInputs, factory};
    }

    const ReferenceGraphEntry *getFactory(const std::string &name) const
    {
        auto it = factories.find(name);
        if (it != factories.end())
            return &it->second;
        return nullptr;
    }

    const std::unordered_map<std::string, ReferenceGraphEntry> &getAll() const
    {
        return factories;
    }

private:
    std::unordered_map<std::string, ReferenceGraphEntry> factories;
};

// Implement the Graph builder method here to resolve dependencies smoothly
inline uint32_t Graph::tanh(uint32_t id0)
{
    auto *entry = ReferenceGraphRegistry::get().getFactory("Tanh");
    if (!entry)
        throw std::runtime_error("No reference factory registered for Tanh");
    return entry->factory({id0}, *this);
}

struct KernelEntry
{
    OpType opType;
    std::string opName;
    size_t numInputs;
    Backend backend;
    MatchFunc match;
    KernelFunc run;
    ReferenceFactory refFactory;
};

class KernelRegistry
{
public:
    static KernelRegistry &get()
    {
        static KernelRegistry instance;
        return instance;
    }

    void registerKernel(OpType op, const std::string &opName, size_t numInputs, Backend backend, MatchFunc match, KernelFunc run, ReferenceFactory refFactory)
    {
        entries.push_back({op, opName, numInputs, backend, match, run, refFactory});
        if (refFactory && op == OpType::FUSED)
        {
            ReferenceGraphRegistry::get().registerFactory(opName, numInputs, refFactory);
        }
    }

    std::vector<uint32_t> findMatchingKernels(OpType op, const std::string &opName, Backend backend,
                                              const std::vector<TensorNode> &inputs,
                                              const TensorNode &output) const
    {
        std::vector<uint32_t> matches;
        for (uint32_t i = 0; i < entries.size(); ++i)
        {
            if (entries[i].opType == op && entries[i].backend == backend)
            {
                if (op == OpType::FUSED && entries[i].opName != opName)
                    continue;
                if (entries[i].match(inputs, output))
                {
                    matches.push_back(i);
                }
            }
        }
        return matches;
    }

    const KernelEntry &getKernel(uint32_t id) const
    {
        if (id >= entries.size())
        {
            throw std::runtime_error("Invalid kernel ID");
        }
        return entries[id];
    }

private:
    std::vector<KernelEntry> entries;
};

struct KernelRegistrar
{
    KernelRegistrar(OpType op, const std::string &opName, size_t numInputs, Backend backend, MatchFunc match, KernelFunc run, ReferenceFactory refFactory)
    {
        KernelRegistry::get().registerKernel(op, opName, numInputs, backend, match, run, refFactory);
    }
};

#define REGISTER_KERNEL(op, backend, match, run) \
    static KernelRegistrar _registrar_##run(op, "", 0, backend, match, run, nullptr)

#define REGISTER_FUSED_KERNEL(opName, numInputs, backend, match, run, refFactory) \
    static KernelRegistrar _registrar_fused_##run(OpType::FUSED, opName, numInputs, backend, match, run, refFactory)
