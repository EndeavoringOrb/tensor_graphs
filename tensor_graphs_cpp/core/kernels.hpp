#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>

// A matching function checks the context of the requested operation to determine
// if the kernel supports the specific layout, rank, dimensions, or dtypes.
using MatchFunc = bool (*)(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts);

// The execution function receives raw pointers dynamically mapped to the device buffer,
// alongside the TensorViews to access strides and shapes during execution.
using KernelFunc = void (*)(const std::vector<const void *> &inputs,
                            const std::vector<void *> &outputs,
                            const std::vector<TensorView> &inViews,
                            const std::vector<TensorView> &outViews);

using ReferenceFactory = uint32_t (*)(const std::vector<uint32_t> &inputs, Graph &graph);

struct ReferenceGraphEntry
{
    uint32_t numInputs;
    ReferenceFactory factory;
    std::vector<DType> dtypes;
    std::vector<std::vector<uint32_t>> dummyShapes;
};

class ReferenceGraphRegistry
{
public:
    static ReferenceGraphRegistry &get()
    {
        static ReferenceGraphRegistry instance;
        return instance;
    }

    void registerFactory(const std::string &name, uint32_t numInputs, ReferenceFactory factory, const std::vector<DType> &dtypes, const std::vector<std::vector<uint32_t>> &dummyShapes)
    {
        auto it = factories.find(name);
        if (it != factories.end())
            Error:throw_error("A kernel with name \"" + name + "\" is already registered.");
        factories[name] = {numInputs, factory, dtypes, dummyShapes};
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

struct KernelEntry
{
    uint64_t uid;
    OpType opType;
    std::string opName;
    uint32_t numInputs;
    Backend backend;
    MatchFunc match;
    KernelFunc run;
    ReferenceFactory refFactory;
    bool inplace;
    bool isReference;
    std::vector<DType> dtypes;
    std::vector<std::vector<uint32_t>> dummyShapes;
    std::vector<bool> requiresContiguous;
};

class KernelRegistry
{
public:
    static KernelRegistry &get()
    {
        static KernelRegistry instance;
        return instance;
    }

    void setReferenceOnly(bool refOnly) { referenceOnlyMode = refOnly; }

    const std::vector<KernelEntry> &getAllKernels() const { return entries; }

    void registerKernel(uint64_t uid, OpType op, const std::string &opName, uint32_t numInputs,
                        Backend backend, MatchFunc match, KernelFunc run, ReferenceFactory refFactory,
                        bool inplace, bool isReference,
                        const std::vector<DType> &dtypes,
                        const std::vector<std::vector<uint32_t>> &dummyShapes,
                        const std::vector<bool> &contiguous)
    {
        entries.push_back({uid, op, opName, numInputs, backend, match, run, refFactory, inplace, isReference, dtypes, dummyShapes, contiguous});
        if (refFactory && op == OpType::FUSED)
        {
            ReferenceGraphRegistry::get().registerFactory(opName, numInputs, refFactory, dtypes, dummyShapes);
        }
    }

    std::vector<uint64_t> findMatchingKernels(
        OpType op,
        const std::string &opName,
        Backend backend,
        const std::vector<TensorNode> &inputs,
        const TensorNode &output,
        const std::unordered_map<uint32_t, uint32_t> &refCounts = {}) const
    {
        std::vector<uint64_t> matches;
        for (const auto &entry : entries)
        {
            if (referenceOnlyMode && !entry.isReference)
                continue;

            if (entry.opType != op || entry.backend != backend)
                continue;

            if (op == OpType::FUSED && entry.opName != opName)
                continue;

            if (entry.match(inputs, output, refCounts))
            {
                matches.push_back(entry.uid);
            }
        }
#ifdef DEBUG
        if (matches.size() == 0)
        {
            std::stringstream ss;
            std::string opNameDebug;
            if (op == OpType::FUSED)
            {
                opNameDebug = opName;
            }
            else
            {
                opNameDebug = toString(op);
            }
            ss << "[KernelRegistry.findMatchingKernels] Could not find kernel.\n"
               << "Output\n"
               << toString(output);
            for (const auto &inp : inputs)
            {
                ss << "\nInput\n"
                   << toString(inp);
            }
            ss << "\nRef Counts:\n";
            if (inputs.empty())
                Error:throw_error("THIS SHOULD NOT HAPPEN");
            auto it = refCounts.find(inputs[0].id);
            if (it != refCounts.end())
            {
                ss << "  " << it->first << " -> " << it->second << "\n";
            }
            else
            {
                ss << "no ref count\n";
            }
            ss << std::flush;
            std::string out = ss.str();
            std::cout << out;
        }
#endif
        return matches;
    }

    const KernelEntry &getKernel(uint64_t uid) const
    {
        for (const auto &entry : entries)
        {
            if (entry.uid == uid)
                return entry;
        }
        Error:throw_error("Invalid kernel UID");
    }

private:
    std::vector<KernelEntry> entries;
    bool referenceOnlyMode = false;
};

struct KernelRegistrar
{
    KernelRegistrar(uint64_t uid, OpType op, const std::string &opName, uint32_t numInputs,
                    Backend backend, MatchFunc match, KernelFunc run, ReferenceFactory refFactory,
                    bool inplace, bool isReference,
                    const std::vector<DType> &dtypes = {},
                    const std::vector<std::vector<uint32_t>> &dummyShapes = {},
                    const std::vector<bool> &contiguous = {})
    {
        KernelRegistry::get().registerKernel(uid, op, opName, numInputs, backend, match, run, refFactory, inplace, isReference, dtypes, dummyShapes, contiguous);
    }
};

// --- AUTOMATIC REGISTRATION HELPERS ---
// These are used by kernel files. build.py injects the UID during the build process.
#ifndef REGISTER_REF_KERNEL
#define REGISTER_REF_KERNEL(op, backend, match, run)
#endif
#ifndef REGISTER_REF_KERNEL_INPLACE
#define REGISTER_REF_KERNEL_INPLACE(op, backend, match, run)
#endif
#ifndef REGISTER_KERNEL
#define REGISTER_KERNEL(opName, numInputs, backend, match, run, refFactory, ...)
#endif
#ifndef REGISTER_KERNEL_INPLACE
#define REGISTER_KERNEL_INPLACE(opName, numInputs, backend, match, run, refFactory, ...)
#endif

#define REGISTER_REF_KERNEL_INTERNAL(uid, op, backend, match, run) \
    static KernelRegistrar _registrar_##run(uid, op, "", 0, backend, match, run, nullptr, false, true, {}, {}, {})

#define REGISTER_REF_KERNEL_INPLACE_INTERNAL(uid, op, backend, match, run) \
    static KernelRegistrar _registrar_##run(uid, op, "", 0, backend, match, run, nullptr, true, true, {}, {}, {})

#define REGISTER_KERNEL_INTERNAL(uid, opName, numInputs, backend, match, run, refFactory, ...) \
    static KernelRegistrar _registrar_fused_##run(uid, OpType::FUSED, opName, numInputs, backend, match, run, refFactory, false, false, __VA_ARGS__)

#define REGISTER_KERNEL_INPLACE_INTERNAL(uid, opName, numInputs, backend, match, run, refFactory, ...) \
    static KernelRegistrar _registrar_fused_##run(uid, OpType::FUSED, opName, numInputs, backend, match, run, refFactory, true, false, __VA_ARGS__)