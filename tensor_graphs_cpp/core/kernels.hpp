#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <CL/cl.h>

struct KernelContext
{
    std::vector<const void *> inputs;
    std::vector<void *> outputs;
    std::vector<TensorView> inViews;
    std::vector<TensorView> outViews;
    std::vector<int> fd; // same number of elements as inputs/inViews, has -1 if not a file, positive number file descriptor if is a file. COPY_TO kernels that start from STORAGE should use fd + inViews baseOffset to read from file
    std::vector<cl_mem> cl_inputs;
    std::vector<cl_mem> cl_outputs;

    KernelContext() {}
    KernelContext(const std::vector<const void *> &_inputs,
                  const std::vector<void *> &_outputs,
                  const std::vector<TensorView> &_inViews,
                  const std::vector<TensorView> &_outViews) : inputs(_inputs), outputs(_outputs), inViews(_inViews), outViews(_outViews)
    {
        for (int i = 0; i < inputs.size(); i++)
        {
            fd.push_back(-1);
            cl_inputs.push_back(nullptr);
        }
        for (int i = 0; i < outputs.size(); i++)
        {
            cl_outputs.push_back(nullptr);
        }
    }
};

using MatchFunc = bool (*)(const std::vector<TensorNode> &inputs, const TensorNode &output);
using KernelFunc = void (*)(const KernelContext &ctx);
using ReferenceFactory = LogicalId (*)(const std::vector<LogicalId> &inputs, Graph &graph);
using InferViewFunc = void (*)(TensorNode &node, const std::vector<TensorNode> &inputs, const Graph &graph);

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

    void registerFactory(const std::string &name, uint32_t min_num_inputs, uint32_t max_num_inputs, ReferenceFactory factory, const std::vector<DType> &dtypes, const std::vector<std::vector<uint32_t>> &dummyShapes)
    {
        auto it = factories.find(name);
        if (it != factories.end())
        {
            // return; // TODO: somehow check that the reference graphs are the same
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

    const std::unordered_map<std::string, ReferenceGraphEntry> &getAll() const { return factories; }

private:
    std::unordered_map<std::string, ReferenceGraphEntry> factories;
};

struct GraphPatternCacheKey
{
    OpType pOpType;
    std::string pOpName;
    bool reference_only;
    bool ignore_output_mem_space;
    bool ignore_input_mem_spaces;
    MemSpace output_mem_space;
    std::vector<MemSpace> input_mem_spaces;

    std::vector<TensorNode> inputs;
    TensorNode output;

    bool operator==(const GraphPatternCacheKey &o) const
    {
        if (pOpType != o.pOpType || pOpName != o.pOpName ||
            reference_only != o.reference_only)
            return false;
        if (inputs.size() != o.inputs.size())
            return false;
        if (!ignore_output_mem_space && output_mem_space != o.output_mem_space)
            return false;
        for (uint64_t i = 0; i < inputs.size(); ++i)
        {
            if ((!ignore_input_mem_spaces && input_mem_spaces[i] != o.input_mem_spaces[i]) || inputs[i].dtype != o.inputs[i].dtype ||
                inputs[i].getShape() != o.inputs[i].getShape() || inputs[i].strides != o.inputs[i].strides)
                return false;
        }
        if (output.dtype != o.output.dtype ||
            output.getShape() != o.output.getShape() || output.strides != o.output.strides)
            return false;
        return true;
    }
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
    bool matches(const std::vector<TensorNode> &inputs, const TensorNode &output,
                 MemSpace output_mem_space, std::vector<MemSpace> input_mem_spaces,
                 bool ignore_output_mem_space = false, bool ignore_input_mem_spaces = false,
                 bool ignore_input_contig = false) const
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
                uint64_t ruleIdx = std::min(i, requiresContiguous.size() - 1);
                if (requiresContiguous[ruleIdx] && !isContiguous(inputs[i]))
                    return false;
            }
        }

        // 3. Check memory spaces
        if (!ignore_output_mem_space)
        {
            if (this->output_mem_space != output_mem_space)
            {
                return false;
            }
        }

        if (!ignore_input_mem_spaces && !this->input_mem_spaces.empty())
        {
            for (uint64_t i = 0; i < inputs.size(); ++i)
            {
                uint64_t ruleIdx = std::min(i, static_cast<uint64_t>(this->input_mem_spaces.size() - 1));
                if (input_mem_spaces[ruleIdx] != this->input_mem_spaces[ruleIdx])
                {
                    return false;
                }
            }
        }

        // 4. Check input dtypes if registered
        if (!dtypes.empty())
        {
            for (uint64_t i = 0; i < inputs.size(); ++i)
            {
                uint64_t ruleIdx = std::min(i, dtypes.size() - 1);
                if (inputs[i].dtype != dtypes[ruleIdx])
                    return false;
            }
        }

        // 5. Call custom match function
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

    mutable std::unordered_map<GraphPatternCacheKey, std::vector<KernelId>> patternCache;

    void setReferenceOnly(bool refOnly) { reference_only_mode = refOnly; }
    const std::unordered_map<KernelId, KernelEntry> &getAllKernels() const { return entries; }

    std::vector<KernelId> _findMatchingKernelsByPattern(
        const Graph &patternGraph, LogicalId patternRootId,
        const std::vector<TensorNode> &inputs, const TensorNode &output,
        bool reference_only = false, bool ignore_input_contig = false) const
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

            if (!entry.matches(inputs, output, output_mem_space, input_mem_spaces, ignore_output_mem_space, ignore_input_mem_spaces, ignore_input_contig))
                continue;

            matches.push_back(entry.uid);
        }
        return matches;
    }

    std::vector<KernelId> findMatchingKernelsByPattern(
        const Graph &patternGraph, LogicalId patternRootId,
        const std::vector<TensorNode> &inputs, const TensorNode &output,
        bool reference_only = false, MemSpace output_mem_space, std::vector<MemSpace> input_mem_spaces,
        bool ignore_output_mem_space = false, bool ignore_input_mem_spaces = false, bool ignore_input_contig = false) const
    {
        const TensorNode &rootNode = patternGraph.getNode(patternRootId);
        GraphPatternCacheKey key{
            rootNode.opType, rootNode.opName,
            reference_only, ignore_output_mem_space, ignore_input_mem_spaces,
            output_mem_space, input_mem_spaces, inputs, output};

        auto it = patternCache.find(key);
        if (it != patternCache.end())
        {
            return it->second;
        }

        std::vector<KernelId> matches = _findMatchingKernelsByPattern(
            patternGraph, patternRootId, inputs, output, reference_only, ignore_input_contig);

        patternCache[key] = matches;
        return matches;
    }

    void registerKernel(KernelId uid, OpType op, const std::string &opName,
                        uint32_t min_num_inputs, uint32_t max_num_inputs,
                        MatchFunc match, KernelFunc run, ReferenceFactory refFactory,
                        const std::vector<uint32_t> &safe_inplace_idxs, bool is_view, bool isReference, InferViewFunc inferView,
                        const MemSpace output_mem_space,    // mem space for output
                        const std::vector<Engine> &engines, // which engines this kernel blocks while running
                        const std::vector<DType> &dtypes = {},
                        const std::vector<std::vector<uint32_t>> &dummyShapes = {},
                        const std::vector<bool> &contiguous = {},
                        const std::vector<MemSpace> &input_mem_spaces = {} // mem space for each input
    )
    {
        // TODO: more checks here
        if (input_mem_spaces.size() != min_num_inputs)
        {
            Error::throw_err("input_mem_spaces.size() != min_num_inputs"); // TODO: use std::source_location to point to kernel file
        }
        if (dtypes.size() != min_num_inputs)
        {
            Error::throw_err("dtypes.size() != min_num_inputs");
        }
        if (contiguous.size() != min_num_inputs)
        {
            Error::throw_err("contiguous.size() != min_num_inputs");
        }

        entries.emplace(uid, KernelEntry{uid, op, opName, min_num_inputs, max_num_inputs, match, run, refFactory, safe_inplace_idxs, is_view, isReference, inferView, output_mem_space, engines, dtypes, dummyShapes, contiguous, input_mem_spaces});
        if (refFactory && op == OpType::FUSED)
        {
            ReferenceGraphRegistry::get().registerFactory(opName, min_num_inputs, max_num_inputs, refFactory, dtypes, dummyShapes);
        }
    }

    std::vector<KernelId> findMatchingKernels(
        OpType op, const std::string &opName,
        const std::vector<TensorNode> &inputs, const TensorNode &output,
        bool reference_only = false, MemSpace output_mem_space, std::vector<MemSpace> input_mem_spaces,
        bool ignore_output_mem_space = false, bool ignore_input_mem_spaces = false, bool ignore_input_contig = false) const
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

            if (!entry.matches(inputs, output, output_mem_space, input_mem_spaces, ignore_output_mem_space, ignore_input_mem_spaces, ignore_input_contig))
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
    KernelRegistrar(KernelId uid, OpType op, const std::string &opName,
                    uint32_t min_num_inputs, uint32_t max_num_inputs,
                    MatchFunc match, KernelFunc run, ReferenceFactory refFactory,
                    const std::vector<uint32_t> &safe_inplace_idxs, bool is_view, bool isReference, InferViewFunc inferView,
                    const MemSpace output_mem_space,    // mem space for output
                    const std::vector<Engine> &engines, // which engines this kernel blocks while running
                    const std::vector<DType> &dtypes = {},
                    const std::vector<std::vector<uint32_t>> &dummyShapes = {},
                    const std::vector<bool> &contiguous = {},
                    const std::vector<MemSpace> &input_mem_spaces = {} // mem space for each input
    )
    {
        KernelRegistry::get().registerKernel(uid, op, opName, min_num_inputs, max_num_inputs, match, run, refFactory, safe_inplace_idxs, is_view, isReference, inferView, output_mem_space, engines, dtypes, dummyShapes, contiguous, input_mem_spaces);
    }
};

// --- AUTOMATIC REGISTRATION HELPERS ---
// These are used by kernel files. build.py injects the UID during the build process.
#ifndef REGISTER_REF_KERNEL
#define REGISTER_REF_KERNEL(op, n_min, n_max, match, run, ...)
#endif
#ifndef REGISTER_REF_KERNEL_VIEW
#define REGISTER_REF_KERNEL_VIEW(op, n_min, n_max, match, run, ...)
#endif
#ifndef REGISTER_KERNEL
#define REGISTER_KERNEL(opName, n_min, n_max, match, run, refFactory, ...)
#endif
#ifndef REGISTER_KERNEL_VIEW
#define REGISTER_KERNEL_VIEW(opName, n_min, n_max, match, ref, inferView, ...)
#endif

#define REGISTER_REF_KERNEL_INTERNAL(uid, op, n_min, n_max, match, run, ...) \
    static KernelRegistrar _registrar_##run(uid, op, "", n_min, n_max, match, run, nullptr, false, false, true, nullptr, __VA_ARGS__)

#define REGISTER_REF_KERNEL_VIEW_INTERNAL(uid, op, n_min, n_max, match, inferView, ...) \
    static KernelRegistrar _registrar_##inferView(uid, op, "", n_min, n_max, match, nullptr, nullptr, false, true, true, inferView, __VA_ARGS__)

#define REGISTER_KERNEL_INTERNAL(uid, opName, n_min, n_max, match, run, refFactory, ...) \
    static KernelRegistrar _registrar_fused_##run(uid, OpType::FUSED, opName, n_min, n_max, match, run, refFactory, false, false, false, nullptr, __VA_ARGS__)

#define REGISTER_KERNEL_VIEW_INTERNAL(uid, opName, n_min, n_max, match, refFactory, inferView, ...) \
    static KernelRegistrar _registrar_fused_##inferView(uid, OpType::FUSED, opName, n_min, n_max, match, nullptr, refFactory, false, true, false, inferView, __VA_ARGS__)