#pragma once
#include <CL/cl.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <vector>

#include "core/memory.hpp"
#include "core/misc.hpp"
#include "core/types.hpp"

using MatchFunc = bool (*)(const std::vector<TensorNode> &inputs, const TensorNode &output);
using KernelFunc = void (*)(const KernelContext &ctx);
using ReferenceFactory = LogicalId (*)(const std::vector<LogicalId> &inputs, Graph &graph);
using InferViewFunc = void (*)(const std::vector<TensorNode> &inputs, TensorView &output, const Graph &graph);

struct ReferenceGraphEntry
{
    uint32_t min_num_inputs;
    uint32_t max_num_inputs;
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

    void registerFactory(const std::string &name, uint32_t min_num_inputs, uint32_t max_num_inputs,
                         ReferenceFactory factory, const std::vector<DType> &dtypes,
                         const std::vector<std::vector<uint32_t>> &dummyShapes)
    {
        auto it = factories.find(name);
        if (it != factories.end())
        {
            Error::throw_err("A kernel with name \"" + name + "\" is already registered.");
        }
        factories[name] = {min_num_inputs, max_num_inputs, factory, dtypes, dummyShapes};
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
    KernelId uid;
    OpType opType;
    std::string opName;
    uint32_t min_num_inputs;
    uint32_t max_num_inputs;
    MatchFunc match;
    KernelFunc run;
    ReferenceFactory refFactory;
    std::vector<uint32_t> safe_inplace_idxs;
    bool is_view;
    bool isReference;
    InferViewFunc inferView;
    MemSpace output_mem_space;
    std::vector<Engine> engines;
    std::vector<DType> dtypes;
    std::vector<std::vector<uint32_t>> dummyShapes;
    std::vector<bool> requiresContiguous;
    std::vector<MemSpace> input_mem_spaces;

    std::string getName() const
    {
        if (opName != "")
        {
            return opName;
        }
        return toString(opType);
    }

    // Abstracted validity check
    bool matches(const std::vector<TensorNode> &inputs, const TensorNode &output, MemSpace output_mem_space = {},
                 const std::vector<MemSpace> &input_mem_spaces = {}, const std::vector<Engine> &engines = {},
                 bool ignore_output_mem_space = false, bool ignore_input_mem_spaces = false,
                 bool ignore_engines = false, bool ignore_input_contig = false) const
    {
        // 1. Check number of inputs
        if (inputs.size() < min_num_inputs || inputs.size() > max_num_inputs)
        {
            return false;
        }

        // 2. Check input contiguity
        if (!ignore_input_contig)
        {
            for (uint64_t i = 0; i < inputs.size(); ++i)
            {
                uint64_t ruleIdx = std::min(i, static_cast<uint64_t>(requiresContiguous.size() - 1));
                if (requiresContiguous[ruleIdx] && !isContiguous(inputs[i]))
                    return false;
            }
        }

        // 3 & 4. Check memory space topology.
        //
        // output_mem_space / input_mem_spaces on a KernelEntry are LOCAL to this
        // registration: their numeric value carries no meaning on its own. Two
        // slots that share a local idx must resolve to the SAME actual MemSpace;
        // two slots with different local idxs must resolve to DIFFERENT actual
        // MemSpaces. This lets a kernel registration pick any idxs it likes
        // without coordinating with other kernels or with however many real
        // devices of a given HandleType happen to exist at runtime.
        if (!ignore_output_mem_space || !ignore_input_mem_spaces)
        {
            std::unordered_map<MemSpace, MemSpace> localToActual;
            std::unordered_map<MemSpace, MemSpace> actualToLocal;

            auto reconcile = [&](const MemSpace &local, const MemSpace &actual)
            {
                if (local.type != actual.type)
                    return false;

                auto [fwdIt, fwdInserted] = localToActual.try_emplace(local, actual);
                if (!fwdInserted && !(fwdIt->second == actual))
                    return false; // same local idx resolved two different ways

                auto [bwdIt, bwdInserted] = actualToLocal.try_emplace(actual, local);
                if (!bwdInserted && !(bwdIt->second == local))
                    return false; // two different local idxs collapsed onto one actual space

                return true;
            };

            if (!ignore_output_mem_space && !reconcile(this->output_mem_space, output_mem_space))
            {
                return false;
            }

            if (!ignore_input_mem_spaces && !this->input_mem_spaces.empty())
            {
                for (uint64_t i = 0; i < inputs.size(); ++i)
                {
                    uint64_t ruleIdx = std::min(i, static_cast<uint64_t>(this->input_mem_spaces.size() - 1));
                    if (i >= input_mem_spaces.size())
                        return false;
                    if (!reconcile(this->input_mem_spaces[ruleIdx], input_mem_spaces[i]))
                        return false;
                }
            }
        }

        // 5. Check engines
        if (!ignore_engines && !this->engines.empty())
        {
            if (this->engines != engines)
            {
                return false;
            }
        }

        // 6. Check input dtypes if registered
        if (!dtypes.empty())
        {
            for (uint64_t i = 0; i < inputs.size(); ++i)
            {
                uint64_t ruleIdx = std::min(i, static_cast<uint64_t>(dtypes.size() - 1));
                if (inputs[i].dtype != dtypes[ruleIdx])
                    return false;
            }
        }

        // 7. Call custom match function
        if (match)
        {
            return match(inputs, output);
        }
        return true;
    }
};

inline std::string toString(const KernelEntry &entry)
{
    std::stringstream ss;
    ss << "KernelEntry {\n"
       << "  UID:                 " << entry.uid << "\n"
       << "  OpType:              " << toString(entry.opType) << "\n"
       << "  OpName:              " << (entry.opName.empty() ? "N/A" : entry.opName) << "\n"
       << "  Min Num Inputs:      " << entry.min_num_inputs << "\n"
       << "  Max Num Inputs:      " << entry.max_num_inputs << "\n"
       << "  Match Func:          " << (entry.match ? "present" : "nullptr") << "\n"
       << "  Run Func:            " << (entry.run ? "present" : "nullptr") << "\n"
       << "  Ref Factory:         " << (entry.refFactory ? "present" : "nullptr") << "\n"
       << "  Safe Inplace Idxs:   " << toString(entry.safe_inplace_idxs) << "\n"
       << "  Is View:             " << (entry.is_view ? "true" : "false") << "\n"
       << "  Is Reference:        " << (entry.isReference ? "true" : "false") << "\n"
       << "  Infer View Func:     " << (entry.inferView ? "present" : "nullptr") << "\n"
       << "  Output MemSpace:     " << entry.output_mem_space << "\n"
       << "  Engines:             " << toString(entry.engines) << "\n"
       << "  DTypes:              " << toString(entry.dtypes) << "\n"
       << "  Dummy Shapes:        [";
    for (uint64_t i = 0; i < entry.dummyShapes.size(); ++i)
    {
        if (i > 0)
            ss << ", ";
        ss << toString(entry.dummyShapes[i]);
    }
    ss << "]\n"
       << "  Requires Contiguous: [";
    for (uint64_t i = 0; i < entry.requiresContiguous.size(); ++i)
    {
        if (i > 0)
            ss << ", ";
        ss << (entry.requiresContiguous[i] ? "true" : "false");
    }
    ss << "]\n"
       << "  Input MemSpaces:     " << toString(entry.input_mem_spaces) << "\n"
       << "}";
    return ss.str();
}

inline std::ostream &operator<<(std::ostream &os, const KernelEntry &entry)
{
    return os << toString(entry);
}

class KernelRegistry
{
public:
    static KernelRegistry &get()
    {
        static KernelRegistry instance;
        return instance;
    }

    mutable std::unordered_map<GraphPatternCacheKey, std::vector<KernelId>> patternCache;

    void setReferenceOnly(bool refOnly)
    {
        reference_only_mode = refOnly;
    }
    const std::unordered_map<KernelId, KernelEntry> &getAllKernels() const
    {
        return entries;
    }

    std::vector<KernelId> _findMatchingKernelsByPattern(
        const Graph &patternGraph, LogicalId patternRootId, const std::vector<TensorNode> &inputs,
        const TensorNode &output, bool reference_only = false, MemSpace output_mem_space = {},
        const std::vector<MemSpace> &input_mem_spaces = {}, const std::vector<Engine> &engines = {},
        bool ignore_output_mem_space = false, bool ignore_input_mem_spaces = false, bool ignore_engines = false,
        bool ignore_input_contig = false) const
    {
        std::vector<KernelId> matches;
        for (const auto &[uid, entry] : entries)
        {
            if ((reference_only_mode || reference_only) && !entry.isReference)
                continue;

            bool patternMatches = false;
            if (entry.opType == OpType::FUSED)
            {
                if (entry.refFactory)
                {
                    Graph kGraph;
                    std::vector<LogicalId> kInputs;
                    for (uint64_t i = 0; i < entry.min_num_inputs; ++i)
                        kInputs.push_back(kGraph.input(entry.dummyShapes[i], entry.dtypes[i]));
                    LogicalId kRootId = entry.refFactory(kInputs, kGraph);
                    patternMatches = isIsomorphic(patternGraph, patternRootId, kGraph, kRootId);
                }
            }
            else
            {
                const TensorNode &pNode = patternGraph.getNode(patternRootId);
                if (pNode.opType == entry.opType)
                {
                    patternMatches = true;
                    for (LogicalId pid : pNode.child_ids)
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

            if (!entry.matches(inputs, output, output_mem_space, input_mem_spaces, engines, ignore_output_mem_space,
                               ignore_input_mem_spaces, ignore_engines, ignore_input_contig))
                continue;

            matches.push_back(entry.uid);
        }
        return matches;
    }

    std::vector<KernelId> findMatchingKernelsByPattern(
        const Graph &patternGraph, LogicalId patternRootId, const std::vector<TensorNode> &inputs,
        const TensorNode &output, bool reference_only = false, MemSpace output_mem_space = {},
        const std::vector<MemSpace> &input_mem_spaces = {}, const std::vector<Engine> &engines = {},
        bool ignore_output_mem_space = false, bool ignore_input_mem_spaces = false, bool ignore_engines = false,
        bool ignore_input_contig = false) const
    {
        const TensorNode &rootNode = patternGraph.getNode(patternRootId);
        GraphPatternCacheKey key{rootNode.opType,
                                 rootNode.opName,
                                 reference_only,
                                 ignore_output_mem_space,
                                 ignore_input_mem_spaces,
                                 ignore_engines,
                                 output_mem_space,
                                 input_mem_spaces,
                                 engines,
                                 inputs,
                                 output};

        auto it = patternCache.find(key);
        if (it != patternCache.end())
        {
            return it->second;
        }

        std::vector<KernelId> matches = _findMatchingKernelsByPattern(
            patternGraph, patternRootId, inputs, output, reference_only, output_mem_space, input_mem_spaces, engines,
            ignore_output_mem_space, ignore_input_mem_spaces, ignore_engines, ignore_input_contig);

        patternCache[key] = matches;
        return matches;
    }

    void registerKernel(KernelId uid, OpType op, const std::string &opName, uint32_t min_num_inputs,
                        uint32_t max_num_inputs, MatchFunc match, KernelFunc run, ReferenceFactory refFactory,
                        const std::vector<uint32_t> &safe_inplace_idxs, bool is_view, bool isReference,
                        InferViewFunc inferView, const MemSpace output_mem_space, const std::vector<Engine> &engines,
                        const std::vector<DType> &dtypes = {},
                        const std::vector<std::vector<uint32_t>> &dummyShapes = {},
                        const std::vector<bool> &contiguous = {}, const std::vector<MemSpace> &input_mem_spaces = {})
    {
        if (input_mem_spaces.size() != min_num_inputs)
        {
            Error::throw_err("input_mem_spaces.size() != min_num_inputs");
        }
        if (dtypes.size() != min_num_inputs)
        {
            Error::throw_err("dtypes.size() != min_num_inputs");
        }
        if (contiguous.size() != min_num_inputs)
        {
            Error::throw_err("contiguous.size() != min_num_inputs");
        }

        entries.emplace(uid, KernelEntry{uid, op, opName, min_num_inputs, max_num_inputs, match, run, refFactory,
                                         safe_inplace_idxs, is_view, isReference, inferView, output_mem_space, engines,
                                         dtypes, dummyShapes, contiguous, input_mem_spaces});
        if (refFactory && op == OpType::FUSED)
        {
            ReferenceGraphRegistry::get().registerFactory(opName, min_num_inputs, max_num_inputs, refFactory, dtypes,
                                                          dummyShapes);
        }
    }

    std::vector<KernelId> findMatchingKernels(
        OpType op, const std::string &opName, const std::vector<TensorNode> &inputs, const TensorNode &output,
        bool reference_only = false, MemSpace output_mem_space = {}, const std::vector<MemSpace> &input_mem_spaces = {},
        const std::vector<Engine> &engines = {}, bool ignore_output_mem_space = false,
        bool ignore_input_mem_spaces = false, bool ignore_engines = false, bool ignore_input_contig = false) const
    {
        std::vector<KernelId> matches;
        for (const auto &[uid, entry] : entries)
        {
            if ((reference_only_mode || reference_only) && !entry.isReference)
                continue;
            if (entry.opType != op)
                continue;
            if (entry.opName != opName)
                continue;

            if (!entry.matches(inputs, output, output_mem_space, input_mem_spaces, engines, ignore_output_mem_space,
                               ignore_input_mem_spaces, ignore_engines, ignore_input_contig))
                continue;

            matches.push_back(entry.uid);
        }
        return matches;
    }

    const KernelEntry &getKernel(KernelId uid) const
    {
        auto it = entries.find(uid);
        if (it != entries.end())
            return it->second;
        Error::throw_err("Invalid kernel UID " + toString(uid));
    }

    bool hasKernel(KernelId uid) const
    {
        return entries.find(uid) != entries.end();
    }

private:
    std::unordered_map<KernelId, KernelEntry> entries;
    bool reference_only_mode = false;
};

struct KernelRegistrar
{
    KernelRegistrar(KernelId uid, OpType op, const std::string &opName, uint32_t min_num_inputs,
                    uint32_t max_num_inputs, MatchFunc match, KernelFunc run, ReferenceFactory refFactory,
                    const std::vector<uint32_t> &safe_inplace_idxs, bool is_view, bool isReference,
                    InferViewFunc inferView, const MemSpace output_mem_space, const std::vector<Engine> &engines,
                    const std::vector<DType> &dtypes = {}, const std::vector<std::vector<uint32_t>> &dummyShapes = {},
                    const std::vector<bool> &contiguous = {}, const std::vector<MemSpace> &input_mem_spaces = {})
    {
        KernelRegistry::get().registerKernel(uid, op, opName, min_num_inputs, max_num_inputs, match, run, refFactory,
                                             safe_inplace_idxs, is_view, isReference, inferView, output_mem_space,
                                             engines, dtypes, dummyShapes, contiguous, input_mem_spaces);
    }
};

#ifndef REGISTER_REF_KERNEL
#define REGISTER_REF_KERNEL(op, n_min, n_max, match, run, ...)
#endif
#ifndef REGISTER_REF_KERNEL_VIEW
#define REGISTER_REF_KERNEL_VIEW(op, n_min, n_max, match, run, ...)
#endif
#ifndef REGISTER_KERNEL
#define REGISTER_KERNEL(opName, n_min, n_max, match, run, refFactory, ...)
#endif
#ifndef REGISTER_KERNEL_INPLACE
#define REGISTER_KERNEL_INPLACE(opName, n_min, n_max, match, run, refFactory, ...)
#endif
#ifndef REGISTER_KERNEL_VIEW
#define REGISTER_KERNEL_VIEW(opName, n_min, n_max, match, ref, inferView, ...)
#endif

#define REGISTER_REF_KERNEL_INTERNAL(uid, op, n_min, n_max, match, run, ...)                                          \
    static KernelRegistrar _registrar_##run(uid, op, "", n_min, n_max, match, run, nullptr, {}, false, true, nullptr, \
                                            __VA_ARGS__)

#define REGISTER_REF_KERNEL_VIEW_INTERNAL(uid, op, n_min, n_max, match, inferView, ...)                               \
    static KernelRegistrar _registrar_##inferView(uid, op, "", n_min, n_max, match, nullptr, nullptr, {}, true, true, \
                                                  inferView, __VA_ARGS__)

#define REGISTER_KERNEL_INTERNAL(uid, opName, n_min, n_max, match, run, refFactory, ...)                            \
    static KernelRegistrar _registrar_fused_##run(uid, OpType::FUSED, opName, n_min, n_max, match, run, refFactory, \
                                                  {}, false, false, nullptr, __VA_ARGS__)

#define REGISTER_KERNEL_INPLACE_INTERNAL(uid, opName, n_min, n_max, match, run, refFactory, ...)                    \
    static KernelRegistrar _registrar_fused_##run(uid, OpType::FUSED, opName, n_min, n_max, match, run, refFactory, \
                                                  {0}, false, false, nullptr, __VA_ARGS__)

#define REGISTER_KERNEL_VIEW_INTERNAL(uid, opName, n_min, n_max, match, refFactory, inferView, ...)               \
    static KernelRegistrar _registrar_fused_##inferView(uid, OpType::FUSED, opName, n_min, n_max, match, nullptr, \
                                                        refFactory, {}, true, false, inferView, __VA_ARGS__)