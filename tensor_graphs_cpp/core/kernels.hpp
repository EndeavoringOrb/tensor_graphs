#pragma once
#include "core/types.hpp"
#include <vector>
#include <stdexcept>
#include <string>

// A matching function checks the context of the requested operation to determine
// if the kernel supports the specific layout, rank, dimensions, or dtypes.
using MatchFunc = bool (*)(const std::vector<TensorNode> &inputs, const TensorNode &output);

// The execution function receives raw pointers dynamically mapped to the device buffer,
// alongside the TensorViews to access strides and shapes during execution.
using KernelFunc = void (*)(const std::vector<const void *> &inputs,
                            const std::vector<void *> &outputs,
                            const std::vector<TensorView> &inViews,
                            const std::vector<TensorView> &outViews);

struct KernelEntry
{
    OpType opType;
    Backend backend;
    MatchFunc match;
    KernelFunc run;
};

class KernelRegistry
{
public:
    static KernelRegistry &get()
    {
        static KernelRegistry instance;
        return instance;
    }

    void registerKernel(OpType op, Backend backend, MatchFunc match, KernelFunc run)
    {
        entries.push_back({op, backend, match, run});
    }

    std::vector<uint32_t> findMatchingKernels(OpType op, Backend backend,
                                              const std::vector<TensorNode> &inputs,
                                              const TensorNode &output) const
    {
        std::vector<uint32_t> matches;
        for (uint32_t i = 0; i < entries.size(); ++i)
        {
            if (entries[i].opType == op && entries[i].backend == backend)
            {
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

// Helper struct and macro for clean static-time kernel registration
struct KernelRegistrar
{
    KernelRegistrar(OpType op, Backend backend, MatchFunc match, KernelFunc run)
    {
        KernelRegistry::get().registerKernel(op, backend, match, run);
    }
};

#define REGISTER_KERNEL(op, backend, match, run) static KernelRegistrar _registrar_##run(op, backend, match, run)

// Include kernels here so they can use the registry and macro
#include "kernels/add/F32_1D.hpp"