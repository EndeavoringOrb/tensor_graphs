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
    bool inplace; // Strict flag: true for inplace-only, false for out-of-place-only
};

class KernelRegistry
{
public:
    static KernelRegistry &get()
    {
        static KernelRegistry instance;
        return instance;
    }

    void registerKernel(OpType op, const std::string &opName, size_t numInputs, Backend backend, MatchFunc match, KernelFunc run, ReferenceFactory refFactory, bool inplace)
    {
        entries.push_back({op, opName, numInputs, backend, match, run, refFactory, inplace});
        if (refFactory && op == OpType::FUSED)
        {
            ReferenceGraphRegistry::get().registerFactory(opName, numInputs, refFactory);
        }
    }

    // Updated signature to include refCounts for runtime inplace safety checks
    std::vector<uint32_t> findMatchingKernels(
        OpType op,
        const std::string &opName,
        Backend backend,
        const std::vector<TensorNode> &inputs,
        const TensorNode &output,
        const std::unordered_map<uint32_t, uint32_t> *refCounts = nullptr) const
    {
        std::vector<uint32_t> matches;
        for (uint32_t i = 0; i < entries.size(); ++i)
        {
            const auto &entry = entries[i];
            if (entry.opType != op || entry.backend != backend)
                continue;
            if (op == OpType::FUSED && entry.opName != opName)
                continue;

            // ---------------------------------------------------------
            // Inplace Kernel Logic Centralized Here
            // ---------------------------------------------------------
            if (entry.inplace)
            {
                if (inputs.empty())
                    continue;

                const TensorNode &input0 = inputs[0];

                // Constraint 1: Cannot perform inplace operation on PERSISTENT (read-only) inputs
                if (input0.storageType == StorageType::PERSISTENT)
                    continue;

                // Constraint 2: Shape and DType must be compatible for aliasing
                if (countElements(input0.shape) != countElements(output.shape))
                    continue;
                if (getDTypeSize(input0.dtype) != getDTypeSize(output.dtype))
                    continue;

                // Constraint 3: Runtime Reference Count Check
                // If refCounts are provided, we can only alias if the input is no longer needed
                if (refCounts)
                {
                    auto it = refCounts->find(input0.id);
                    if (it == refCounts->end() || it->second != 1)
                        continue;
                }
            }

            // Finally, check the specific kernel match function
            if (entry.match(inputs, output))
            {
                matches.push_back(i);
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
    KernelRegistrar(OpType op, const std::string &opName, size_t numInputs, Backend backend, MatchFunc match, KernelFunc run, ReferenceFactory refFactory, bool inplace)
    {
        KernelRegistry::get().registerKernel(op, opName, numInputs, backend, match, run, refFactory, inplace);
    }
};

// Standard kernel (out-of-place)
#define REGISTER_KERNEL(op, backend, match, run) \
    static KernelRegistrar _registrar_##run(op, "", 0, backend, match, run, nullptr, false)

// In-place kernel
#define REGISTER_KERNEL_INPLACE(op, backend, match, run) \
    static KernelRegistrar _registrar_##run(op, "", 0, backend, match, run, nullptr, true)

// Standard fused kernel (out-of-place)
#define REGISTER_FUSED_KERNEL(opName, numInputs, backend, match, run, refFactory) \
    static KernelRegistrar _registrar_fused_##run(OpType::FUSED, opName, numInputs, backend, match, run, refFactory, false)

// In-place fused kernel
#define REGISTER_FUSED_KERNEL_INPLACE(opName, numInputs, backend, match, run, refFactory) \
    static KernelRegistrar _registrar_fused_##run(OpType::FUSED, opName, numInputs, backend, match, run, refFactory, true)