#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>

using MatchFunc = bool (*)(const std::vector<TensorNode> &inputs, const TensorNode &output);
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
            // return; // TODO: somehow check that the reference graphs are the same
            Error::throw_err("A kernel with name \"" + name + "\" is already registered.");
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

    const std::unordered_map<std::string, ReferenceGraphEntry> &getAll() const { return factories; }

private:
    std::unordered_map<std::string, ReferenceGraphEntry> factories;
};

struct PatternCacheKey
{
    OpType pOpType;
    std::string pOpName;
    Backend backend;
    bool referenceOnly;
    bool ignoreInputBackends;
    bool ignoreInputContig;

    std::vector<TensorNode> inputs;
    TensorNode output;

    bool operator==(const PatternCacheKey &o) const
    {
        if (pOpType != o.pOpType || pOpName != o.pOpName || backend != o.backend ||
            referenceOnly != o.referenceOnly || ignoreInputBackends != o.ignoreInputBackends ||
            ignoreInputContig != o.ignoreInputContig)
            return false;
        if (inputs.size() != o.inputs.size())
            return false;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            if (inputs[i].dtype != o.inputs[i].dtype || inputs[i].backend != o.inputs[i].backend ||
                inputs[i].getShape() != o.inputs[i].getShape() || inputs[i].strides != o.inputs[i].strides ||
                inputs[i].viewOffset != o.inputs[i].viewOffset)
                return false;
        }
        if (output.dtype != o.output.dtype || output.backend != o.output.backend ||
            output.getShape() != o.output.getShape() || output.strides != o.output.strides ||
            output.viewOffset != o.output.viewOffset)
            return false;
        return true;
    }
};

struct PatternCacheKeyHash
{
    size_t operator()(const PatternCacheKey &k) const
    {
        size_t h = 0;
        auto combine = [&](size_t val)
        { h ^= val + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
        combine((size_t)k.pOpType);
        if (!k.pOpName.empty())
            combine(std::hash<std::string>()(k.pOpName));
        combine((size_t)k.backend);
        combine(k.referenceOnly);
        combine(k.ignoreInputBackends);
        combine(k.ignoreInputContig);
        for (const auto &in : k.inputs)
        {
            combine((size_t)in.dtype);
            combine((size_t)in.backend);
            for (auto s : in.getShape())
                combine(s);
            for (auto s : in.strides)
                combine(s);
            combine((size_t)in.viewOffset);
        }
        combine((size_t)k.output.dtype);
        combine((size_t)k.output.backend);
        for (auto s : k.output.getShape())
            combine(s);
        for (auto s : k.output.strides)
            combine(s);
        combine((size_t)k.output.viewOffset);
        return h;
    }
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

    // Abstracted validity check
    bool matches(const std::vector<TensorNode> &inputs, const TensorNode &output,
                 bool ignoreInputBackends = false, bool ignoreInputContig = false) const
    {
        // 1. Check number of inputs
        if (isVariadic)
        {
            if (inputs.size() < 2)
                return false;
        }
        else if (inputs.size() != numInputs)
        {
            return false;
        }

        // 2. Check input backends
        if (!ignoreInputBackends)
        {
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                size_t ruleIdx = isVariadic ? (i == inputs.size() - 1 ? 1 : 0) : i;
                bool found = false;
                for (Backend b : inputBackends[ruleIdx])
                {
                    if (inputs[i].backend == b)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    return false;
            }
        }

        // 3. Check input contiguity
        if (!ignoreInputContig)
        {
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                size_t ruleIdx = isVariadic ? (i == inputs.size() - 1 ? 1 : 0) : i;
                if (requiresContiguous[ruleIdx] && !isContiguous(inputs[i]))
                    return false;
            }
        }

        // 4. Check input dtypes if registered (skipping variadic operations like CONCAT)
        if (!dtypes.empty() && !isVariadic)
        {
            for (size_t i = 0; i < std::min((size_t)numInputs, dtypes.size()); ++i)
            {
                if (i < inputs.size() && inputs[i].dtype != dtypes[i])
                    return false;
            }
        }

        // 5. Output backend check
        bool backendFound = false;
        for (Backend b : backends)
        {
            if (b == output.backend)
            {
                backendFound = true;
                break;
            }
        }
        if (!backendFound)
            return false;

        // 6. Inplace and view checks assume input and output are on the same backend
        if (inplace && numInputs > 0 && !inputs.empty())
        {
            if (inputs[0].backend != output.backend)
                return false;
        }
        if (isView && numInputs > 0 && !inputs.empty())
        {
            if (inputs[0].backend != output.backend)
                return false;
        }

        // 7. Call custom match function
        if (match)
        {
            return match(inputs, output);
        }
        return true;
    }
};

class KernelRegistry
{
public:
    static KernelRegistry &get()
    {
        static KernelRegistry instance;
        return instance;
    }

    mutable std::unordered_map<PatternCacheKey, std::vector<uint64_t>, PatternCacheKeyHash> patternCache;

    void setReferenceOnly(bool refOnly) { referenceOnlyMode = refOnly; }
    const std::vector<KernelEntry> &getAllKernels() const { return entries; }

    std::vector<uint64_t> _findMatchingKernelsByPattern(
        const Graph &patternGraph, uint32_t patternRootId, Backend backend,
        const std::vector<TensorNode> &inputs, const TensorNode &output,
        bool referenceOnly = false, bool ignoreInputBackends = false, bool ignoreInputContig = false) const
    {
        std::vector<uint64_t> matches;
        for (const auto &entry : entries)
        {
            if ((referenceOnlyMode || referenceOnly) && !entry.isReference)
                continue;

            bool patternMatches = false;
            if (entry.opType == OpType::FUSED)
            {
                if (entry.refFactory)
                {
                    Graph kGraph;
                    std::vector<uint32_t> kInputs;
                    for (size_t i = 0; i < entry.numInputs; ++i)
                        kInputs.push_back(kGraph.input(entry.dummyShapes[i], entry.dtypes[i]));
                    uint32_t kRootId = entry.refFactory(kInputs, kGraph);
                    patternMatches = isIsomorphic(patternGraph, patternRootId, kGraph, kRootId);
                }
            }
            else
            {
                const TensorNode &pNode = patternGraph.getNode(patternRootId);
                if (pNode.opType == entry.opType)
                {
                    patternMatches = true;
                    for (uint32_t pid : pNode.parentIds)
                    {
                        if (patternGraph.getNode(pid).opType != OpType::INPUT &&
                            patternGraph.getNode(pid).opType != OpType::ARANGE &&
                            patternGraph.getNode(pid).opType != OpType::FILL)
                        {
                            patternMatches = false;
                            break;
                        }
                    }
                }
            }

            if (!patternMatches)
                continue;

            if (!entry.matches(inputs, output, ignoreInputBackends, ignoreInputContig))
                continue;

            matches.push_back(entry.uid);
        }
        return matches;
    }

    std::vector<uint64_t> findMatchingKernelsByPattern(
        const Graph &patternGraph, uint32_t patternRootId, Backend backend,
        const std::vector<TensorNode> &inputs, const TensorNode &output,
        bool referenceOnly = false, bool ignoreInputBackends = false, bool ignoreInputContig = false) const
    {
        const TensorNode &rootNode = patternGraph.getNode(patternRootId);
        PatternCacheKey key{
            rootNode.opType, rootNode.opName, backend,
            referenceOnly, ignoreInputBackends, ignoreInputContig,
            inputs, output};

        auto it = patternCache.find(key);
        if (it != patternCache.end())
        {
            return it->second;
        }

        std::vector<uint64_t> matches = _findMatchingKernelsByPattern(
            patternGraph, patternRootId, backend, inputs, output, referenceOnly, ignoreInputBackends, ignoreInputContig);

        patternCache[key] = matches;
        return matches;
    }

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
        OpType op, const std::string &opName, Backend backend,
        const std::vector<TensorNode> &inputs, const TensorNode &output,
        bool referenceOnly = false, bool ignoreInputBackends = false, bool ignoreInputContig = false) const
    {
        std::vector<uint64_t> matches;
        for (const auto &entry : entries)
        {
            if ((referenceOnlyMode || referenceOnly) && !entry.isReference)
                continue;
            if (entry.opType != op)
                continue;
            if (op == OpType::FUSED && entry.opName != opName)
                continue;

            if (!entry.matches(inputs, output, ignoreInputBackends, ignoreInputContig))
                continue;

            matches.push_back(entry.uid);
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