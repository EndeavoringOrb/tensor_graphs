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

using InferViewFunc = void (*)(TensorNode &node, const std::vector<TensorNode> &inputs, const Graph &graph);

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
        {
            return; // TODO: somehow check that the reference graphs are the same
            // Error::throw_err("A kernel with name \"" + name + "\" is already registered.");
        }
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
    bool isVariadic;
    std::vector<Backend> backends;
    std::vector<std::vector<Backend>> inputBackends;
    MatchFunc match;
    KernelFunc run;
    ReferenceFactory refFactory;
    bool inplace;
    bool isView;
    bool isReference;
    InferViewFunc inferView;
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
                        const std::vector<Backend> &backends, MatchFunc match, KernelFunc run, ReferenceFactory refFactory,
                        bool inplace, bool isView, bool isReference, InferViewFunc inferView,
                        const std::vector<DType> &dtypes,
                        const std::vector<std::vector<uint32_t>> &dummyShapes,
                        const std::vector<bool> &contiguous,
                        const std::vector<std::vector<Backend>> &inputBackends)
    {
        bool isVariadic = (op == OpType::CONCAT);
        if (!isVariadic && inputBackends.size() != numInputs)
        {
            Error::throw_err("[KernelRegistry.registerKernel] expected inputBackends.size() == " + std::to_string(numInputs) + " but got " + std::to_string(inputBackends.size()) + ". Info:\n" +
                             "  UID: " + std::to_string(uid) + "\n" +
                             "  OpType: " + toString(op) + "\n" +
                             "  OpName: " + opName + "\n" +
                             "  # Inputs: " + std::to_string(numInputs) + "\n" +
                             "  # Backends: " + std::to_string(backends.size()) + "\n" +
                             "  # Input Backends: " + std::to_string(inputBackends.size()) + "\n" +
                             "  Inplace: " + std::to_string(inplace) + "\n" +
                             "  Is View: " + std::to_string(isView) + "\n" +
                             "  Is Reference: " + std::to_string(isReference) + "\n" +
                             "  # DTypes: " + std::to_string(dtypes.size()) + "\n" +
                             "  # Dummy Shapes: " + std::to_string(dummyShapes.size()) + "\n" +
                             "  # Contiguous: " + std::to_string(contiguous.size()) + "\n");
        }
        if (!isVariadic && contiguous.size() != numInputs)
        {
            Error::throw_err("[KernelRegistry.registerKernel] expected contiguous.size() == " + std::to_string(numInputs) + " but got " + std::to_string(contiguous.size()) + ". Info:\n" +
                             "  UID: " + std::to_string(uid) + "\n" +
                             "  OpType: " + toString(op) + "\n" +
                             "  OpName: " + opName + "\n" +
                             "  # Inputs: " + std::to_string(numInputs) + "\n" +
                             "  # Backends: " + std::to_string(backends.size()) + "\n" +
                             "  # Input Backends: " + std::to_string(inputBackends.size()) + "\n" +
                             "  Inplace: " + std::to_string(inplace) + "\n" +
                             "  Is View: " + std::to_string(isView) + "\n" +
                             "  Is Reference: " + std::to_string(isReference) + "\n" +
                             "  # DTypes: " + std::to_string(dtypes.size()) + "\n" +
                             "  # Dummy Shapes: " + std::to_string(dummyShapes.size()) + "\n" +
                             "  # Contiguous: " + std::to_string(contiguous.size()) + "\n");
        }
        entries.push_back({uid, op, opName, numInputs, isVariadic, backends, inputBackends, match, run, refFactory, inplace, isView, isReference, inferView, dtypes, dummyShapes, contiguous});
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
        const std::unordered_map<uint32_t, uint32_t> &refCounts = {},
        bool referenceOnly = false,
        bool ignoreInputBackends = false) const
    {
        std::vector<uint64_t> matches;
        for (const auto &entry : entries)
        {
            if ((referenceOnlyMode || referenceOnly) && !entry.isReference)
                continue;

            if (entry.opType != op)
                continue;

            bool backendFound = false;
            for (auto b : entry.backends)
            {
                if (b == backend)
                {
                    backendFound = true;
                    break;
                }
            }
            if (!backendFound)
                continue;

            if (op == OpType::FUSED && entry.opName != opName)
                continue;

            if (entry.isVariadic)
            {
                if (inputs.size() < 2)
                    continue; // Must have at least 1 tensor + 1 axis
            }
            else if (inputs.size() != entry.inputBackends.size())
            {
                Error::throw_err("[KernelRegistry.findMatchingKernels] expected # inputs to equal # input backends but got " + std::to_string(inputs.size()) + " inputs and " + std::to_string(entry.inputBackends.size()) + " input backends");
            }
            if (!ignoreInputBackends)
            {
                bool inputBackendsMatch = true;
                for (uint32_t i = 0; i < inputs.size(); ++i)
                {
                    size_t ruleIdx = entry.isVariadic ? (i == inputs.size() - 1 ? 1 : 0) : i;
                    bool currentInputMatch = false;
                    for (uint32_t j = 0; j < entry.inputBackends[ruleIdx].size(); j++)
                    {
                        if (inputs[i].backend == entry.inputBackends[ruleIdx][j])
                        {
                            currentInputMatch = true;
                            break;
                        }
                    }
                    inputBackendsMatch = inputBackendsMatch && currentInputMatch;
                    if (!inputBackendsMatch)
                        break;
                }
                if (!inputBackendsMatch)
                    continue;
            }

            if (entry.inplace && entry.numInputs > 0)
            {
                if (inputs[0].backend != backend)
                    continue;
            }
            if (entry.isView && entry.numInputs > 0)
            {
                if (inputs[0].backend != backend)
                    continue;
            }

            bool contigMatch = true;
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                size_t ruleIdx = entry.isVariadic ? (i == inputs.size() - 1 ? 1 : 0) : i;
                if (entry.requiresContiguous[ruleIdx] && !isContiguous(inputs[i]))
                {
                    contigMatch = false;
                    break;
                }
            }
            if (!contigMatch)
                continue;

            if (entry.match(inputs, output, refCounts))
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
        Error::throw_err("Invalid kernel UID " + std::to_string(uid));
    }

    bool hasKernel(uint64_t uid) const
    {
        for (const auto &entry : entries)
        {
            if (entry.uid == uid)
                return true;
        }
        return false;
    }

private:
    std::vector<KernelEntry> entries;
    bool referenceOnlyMode = false;
};

struct KernelRegistrar
{
    KernelRegistrar(uint64_t uid, OpType op, const std::string &opName, uint32_t numInputs,
                    MatchFunc match, KernelFunc run, ReferenceFactory refFactory,
                    bool inplace, bool isView, bool isReference, InferViewFunc inferView,
                    const std::vector<Backend> &backends,
                    const std::vector<DType> &dtypes = {},
                    const std::vector<std::vector<uint32_t>> &dummyShapes = {},
                    const std::vector<bool> &contiguous = {},
                    const std::vector<std::vector<Backend>> &inputBackends = {})
    {
        KernelRegistry::get().registerKernel(uid, op, opName, numInputs, backends, match, run, refFactory, inplace, isView, isReference, inferView, dtypes, dummyShapes, contiguous, inputBackends);
    }
};

// --- AUTOMATIC REGISTRATION HELPERS ---
// These are used by kernel files. build.py injects the UID during the build process.
#ifndef REGISTER_REF_KERNEL
#define REGISTER_REF_KERNEL(op, match, run, ...)
#endif
#ifndef REGISTER_REF_KERNEL_INPLACE
#define REGISTER_REF_KERNEL_INPLACE(op, match, run, ...)
#endif
#ifndef REGISTER_KERNEL
#define REGISTER_KERNEL(opName, numInputs, match, run, refFactory, ...)
#endif
#ifndef REGISTER_KERNEL_INPLACE
#define REGISTER_KERNEL_INPLACE(opName, numInputs, match, run, refFactory, ...)
#endif
#ifndef REGISTER_KERNEL_INPLACE_VIEW
#define REGISTER_KERNEL_INPLACE_VIEW(opName, numInputs, match, run, refFactory, inferView, ...)
#endif

#define REGISTER_REF_KERNEL_INTERNAL(uid, op, n, match, run, ...) \
    static KernelRegistrar _registrar_##run(uid, op, "", n, match, run, nullptr, false, false, true, nullptr, __VA_ARGS__)

#define REGISTER_REF_KERNEL_INPLACE_INTERNAL(uid, op, n, match, run, ...) \
    static KernelRegistrar _registrar_##run(uid, op, "", n, match, run, nullptr, true, false, true, nullptr, __VA_ARGS__)

#define REGISTER_REF_KERNEL_VIEW_INTERNAL(uid, op, n, match, inferView, ...) \
    static KernelRegistrar _registrar_##inferView(uid, op, "", n, match, nullptr, nullptr, false, true, true, inferView, __VA_ARGS__)

#define REGISTER_KERNEL_INTERNAL(uid, opName, numInputs, match, run, refFactory, ...) \
    static KernelRegistrar _registrar_fused_##run(uid, OpType::FUSED, opName, numInputs, match, run, refFactory, false, false, false, nullptr, __VA_ARGS__)

#define REGISTER_KERNEL_INPLACE_INTERNAL(uid, opName, numInputs, match, run, refFactory, ...) \
    static KernelRegistrar _registrar_fused_##run(uid, OpType::FUSED, opName, numInputs, match, run, refFactory, true, false, false, nullptr, __VA_ARGS__)

#define REGISTER_KERNEL_VIEW_INTERNAL(uid, opName, numInputs, match, refFactory, inferView, ...) \
    static KernelRegistrar _registrar_fused_##inferView(uid, OpType::FUSED, opName, numInputs, match, nullptr, refFactory, false, true, false, inferView, __VA_ARGS__)