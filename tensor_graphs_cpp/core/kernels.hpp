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
            throw std::runtime_error("A kernel with name \"" + name + "\" is already registered.");
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
                        const std::vector<std::vector<uint32_t>> &dummyShapes)
    {
        entries.push_back({uid, op, opName, numInputs, backend, match, run, refFactory, inplace, isReference, dtypes, dummyShapes});
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
        const std::unordered_map<uint32_t, uint32_t> *refCounts = nullptr) const
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

            if (entry.inplace)
            {
                if (inputs.empty())
                    continue;
                const TensorNode &input0 = inputs[0];

                if (input0.storageType == StorageType::PERSISTENT)
                    continue;
                if (countElements(input0.shape) != countElements(output.shape))
                    continue;
                if (getDTypeSize(input0.dtype) != getDTypeSize(output.dtype))
                    continue;

                if (refCounts)
                {
                    auto it = refCounts->find(input0.id);
                    if (it == refCounts->end() || it->second != 1)
                        continue;
                }
            }

            if (entry.match(inputs, output))
            {
                matches.push_back(entry.uid);
            }
        }
        return matches;
    }

    const KernelEntry &getKernel(uint64_t uid) const
    {
        for (const auto &entry : entries)
        {
            if (entry.uid == uid)
                return entry;
        }
        throw std::runtime_error("Invalid kernel UID");
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
                    const std::vector<std::vector<uint32_t>> &dummyShapes = {})
    {
        KernelRegistry::get().registerKernel(uid, op, opName, numInputs, backend, match, run, refFactory, inplace, isReference, dtypes, dummyShapes);
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
    static KernelRegistrar _registrar_##run(uid, op, "", 0, backend, match, run, nullptr, false, true, {}, {})

#define REGISTER_REF_KERNEL_INPLACE_INTERNAL(uid, op, backend, match, run) \
    static KernelRegistrar _registrar_##run(uid, op, "", 0, backend, match, run, nullptr, true, true, {}, {})

#define REGISTER_KERNEL_INTERNAL(uid, opName, numInputs, backend, match, run, refFactory, ...) \
    static KernelRegistrar _registrar_fused_##run(uid, OpType::FUSED, opName, numInputs, backend, match, run, refFactory, false, false, __VA_ARGS__)

#define REGISTER_KERNEL_INPLACE_INTERNAL(uid, opName, numInputs, backend, match, run, refFactory, ...) \
    static KernelRegistrar _registrar_fused_##run(uid, OpType::FUSED, opName, numInputs, backend, match, run, refFactory, true, false, __VA_ARGS__)