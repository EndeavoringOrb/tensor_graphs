#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

// Concrete physical hardware assignment for a kernel instance.
struct HardwareBinding
{
    MemSpace output_mem_space;
    std::vector<MemSpace> input_mem_spaces;
    std::vector<Engine> engines;
};

// Resolves the abstract topology used by a registered kernel to physical hardware.
class TopologyMapper
{
  public:
    static bool resolve(const MemSpace &kernel_out_ms, const std::vector<MemSpace> &kernel_in_ms,
                        const std::vector<Engine> &kernel_engines, bool is_view, size_t num_inputs,
                        MemSpace target_out_ms, const std::vector<MemSpace> &target_in_ms,
                        const std::vector<Engine> &target_engines, bool ignore_out_ms, bool ignore_in_ms,
                        bool ignore_eng, HardwareBinding &out_binding)
    {
        const auto &sys_spaces = System::get().getAvailableMemSpaces();
        const auto &sys_engines = System::get().getAvailableEngines();
        const auto isPhysicalMem = [&](const MemSpace &ms) {
            return std::find(sys_spaces.begin(), sys_spaces.end(), ms) != sys_spaces.end();
        };
        const auto isPhysicalEngine = [&](const Engine &engine) {
            return std::find(sys_engines.begin(), sys_engines.end(), engine) != sys_engines.end();
        };

        if (is_view)
        {
            if (!ignore_out_ms && !ignore_in_ms && !target_in_ms.empty() &&
                target_out_ms.type != HandleType::STORAGE && target_in_ms[0].type != HandleType::STORAGE &&
                target_out_ms != target_in_ms[0])
                return false;

            MemSpace resolved_out{1, HandleType::CPP};
            if (!ignore_out_ms && target_out_ms.type != HandleType::STORAGE && isPhysicalMem(target_out_ms))
                resolved_out = target_out_ms;
            else if (!ignore_in_ms && !target_in_ms.empty() && isPhysicalMem(target_in_ms[0]))
                resolved_out = target_in_ms[0];

            out_binding.output_mem_space = resolved_out;
            out_binding.input_mem_spaces.assign(num_inputs, MemSpace{1, HandleType::CPP});
            if (num_inputs > 0)
                out_binding.input_mem_spaces[0] = resolved_out;

            out_binding.engines.clear();
            if (!ignore_eng && !target_engines.empty())
            {
                bool all_physical = true;
                for (const auto &engine : target_engines)
                {
                    if (!isPhysicalEngine(engine))
                    {
                        all_physical = false;
                        break;
                    }
                }
                if (all_physical)
                    out_binding.engines = target_engines;
            }
            if (out_binding.engines.empty())
            {
                for (const auto &engine : sys_engines)
                {
                    if ((resolved_out.type == HandleType::CUDA && engine.type == EngineType::CUDA_GPU &&
                         engine.idx == resolved_out.idx) ||
                        (resolved_out.type == HandleType::OPENCL && engine.type == EngineType::QUALCOMM_IGPU) ||
                        (resolved_out.type == HandleType::CPP && engine.type == EngineType::CPU))
                    {
                        out_binding.engines.push_back(engine);
                        break;
                    }
                }
            }
            return !out_binding.engines.empty();
        }

        // Storage is read-only. Only metadata views may retain a storage output;
        // executable kernels must never treat an explicit storage target as a wildcard.
        if (kernel_out_ms.type == HandleType::STORAGE ||
            (!ignore_out_ms && target_out_ms.type == HandleType::STORAGE))
            return false;

        std::unordered_map<MemSpace, MemSpace> local_to_actual_mem;
        std::unordered_map<MemSpace, MemSpace> actual_to_local_mem;
        const auto reconcileMem = [&](const MemSpace &local, const MemSpace &actual) {
            if (local.type != actual.type)
                return false;
            auto [forward, inserted_forward] = local_to_actual_mem.try_emplace(local, actual);
            if (!inserted_forward && forward->second != actual)
                return false;
            auto [backward, inserted_backward] = actual_to_local_mem.try_emplace(actual, local);
            return inserted_backward || backward->second == local;
        };

        std::unordered_map<Engine, Engine> local_to_actual_engine;
        std::unordered_map<Engine, Engine> actual_to_local_engine;
        const auto reconcileEngine = [&](const Engine &local, const Engine &actual) {
            if (local.type != actual.type)
                return false;
            auto [forward, inserted_forward] = local_to_actual_engine.try_emplace(local, actual);
            if (!inserted_forward && forward->second != actual)
                return false;
            auto [backward, inserted_backward] = actual_to_local_engine.try_emplace(actual, local);
            return inserted_backward || backward->second == local;
        };

        if (!ignore_out_ms && target_out_ms.type != HandleType::STORAGE && isPhysicalMem(target_out_ms) &&
            !reconcileMem(kernel_out_ms, target_out_ms))
            return false;
        if (!ignore_in_ms && !kernel_in_ms.empty())
        {
            for (size_t i = 0; i < num_inputs && i < target_in_ms.size(); ++i)
            {
                if (isPhysicalMem(target_in_ms[i]) &&
                    !reconcileMem(kernel_in_ms[std::min(i, kernel_in_ms.size() - 1)], target_in_ms[i]))
                    return false;
            }
        }
        if (!ignore_eng && !kernel_engines.empty() && !target_engines.empty())
        {
            if (kernel_engines.size() != target_engines.size())
                return false;
            for (size_t i = 0; i < kernel_engines.size(); ++i)
            {
                if (isPhysicalEngine(target_engines[i]) && !reconcileEngine(kernel_engines[i], target_engines[i]))
                    return false;
            }
        }

        std::vector<MemSpace> needed_spaces{kernel_out_ms};
        for (size_t i = 0; i < num_inputs && !kernel_in_ms.empty(); ++i)
            needed_spaces.push_back(kernel_in_ms[std::min(i, kernel_in_ms.size() - 1)]);
        for (const auto &local : needed_spaces)
        {
            if (local_to_actual_mem.find(local) != local_to_actual_mem.end())
                continue;
            bool mapped = false;
            for (const auto &available : sys_spaces)
            {
                if (available.type == local.type && actual_to_local_mem.find(available) == actual_to_local_mem.end() &&
                    reconcileMem(local, available))
                {
                    mapped = true;
                    break;
                }
            }
            if (!mapped)
                return false; // A bijection cannot collapse distinct local spaces.
        }
        for (const auto &local : kernel_engines)
        {
            if (local_to_actual_engine.find(local) != local_to_actual_engine.end())
                continue;
            bool mapped = false;
            for (const auto &available : sys_engines)
            {
                if (available.type == local.type &&
                    actual_to_local_engine.find(available) == actual_to_local_engine.end() &&
                    reconcileEngine(local, available))
                {
                    mapped = true;
                    break;
                }
            }
            if (!mapped)
                return false;
        }

        out_binding.output_mem_space = local_to_actual_mem.at(kernel_out_ms);
        out_binding.input_mem_spaces.clear();
        for (size_t i = 0; i < num_inputs; ++i)
        {
            out_binding.input_mem_spaces.push_back(
                kernel_in_ms.empty() ? out_binding.output_mem_space
                                      : local_to_actual_mem.at(kernel_in_ms[std::min(i, kernel_in_ms.size() - 1)]));
        }
        out_binding.engines.clear();
        for (const auto &local : kernel_engines)
            out_binding.engines.push_back(local_to_actual_engine.at(local));
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
    bool matches(const std::vector<TensorNode> &inputs, const TensorNode &output, MemSpace output_mem_space = {},
                 const std::vector<MemSpace> &input_mem_spaces = {}, const std::vector<Engine> &engines = {},
                 bool ignore_output_mem_space = false, bool ignore_input_mem_spaces = false,
                 bool ignore_engines = false, bool ignore_input_contig = false,
                 std::vector<Engine> *out_mapped_engines = nullptr,
                 std::vector<MemSpace> *out_mapped_input_mem_spaces = nullptr,
                 HardwareBinding *out_binding = nullptr) const
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
                if (ruleIdx < requiresContiguous.size() && requiresContiguous[ruleIdx] && !isContiguous(inputs[i]))
                    return false;
            }
        }

        HardwareBinding binding;
        if (!TopologyMapper::resolve(this->output_mem_space, this->input_mem_spaces, this->engines, this->is_view,
                                     inputs.size(), output_mem_space, input_mem_spaces, engines,
                                     ignore_output_mem_space, ignore_input_mem_spaces, ignore_engines, binding))
            return false;

        if (out_mapped_engines)
            *out_mapped_engines = binding.engines;
        if (out_mapped_input_mem_spaces)
            *out_mapped_input_mem_spaces = binding.input_mem_spaces;
        if (out_binding)
            *out_binding = binding;

        if (!dtypes.empty())
        {
            for (uint64_t i = 0; i < inputs.size(); ++i)
            {
                const uint64_t ruleIdx = std::min(i, static_cast<uint64_t>(dtypes.size() - 1));
                if (inputs[i].dtype != dtypes[ruleIdx])
                    return false;
            }
        }
        return !match || match(inputs, output);

#if 0 // Legacy topology reconciliation retained temporarily for reference.

        // View operations (SLICE, RESHAPE, PERMUTE, REPEAT, FILL, view CAST) perform compile-time
        // metadata calculations on the host and do not dispatch device kernels. They operate
        // agnostically on any backend, provided the output memory space matches data input #0.
        if (this->is_view)
        {
            if (!ignore_output_mem_space && !ignore_input_mem_spaces && !input_mem_spaces.empty())
            {
                if (output_mem_space.type != HandleType::STORAGE && input_mem_spaces[0].type != HandleType::STORAGE)
                {
                    if (!(output_mem_space == input_mem_spaces[0]))
                        return false;
                }
            }

            if (out_mapped_engines)
            {
                out_mapped_engines->clear();
                if (!engines.empty())
                {
                    *out_mapped_engines = engines;
                }
                else
                {
                    const auto &avail_engines = System::get().getAvailableEngines();
                    bool found = false;
                    for (const auto &eng : avail_engines)
                    {
                        if (output_mem_space.type == HandleType::CUDA && eng.type == EngineType::CUDA_GPU &&
                            eng.idx == output_mem_space.idx)
                        {
                            out_mapped_engines->push_back(eng);
                            found = true;
                            break;
                        }
                        else if (output_mem_space.type == HandleType::OPENCL && eng.type == EngineType::QUALCOMM_IGPU)
                        {
                            out_mapped_engines->push_back(eng);
                            found = true;
                            break;
                        }
                        else if (output_mem_space.type == HandleType::CPP && eng.type == EngineType::CPU)
                        {
                            out_mapped_engines->push_back(eng);
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        if (output_mem_space.type == HandleType::CUDA)
                            out_mapped_engines->push_back(
                                Engine{output_mem_space.idx, EngineType::CUDA_GPU, {output_mem_space}});
                        else if (output_mem_space.type == HandleType::OPENCL)
                            out_mapped_engines->push_back(Engine{0, EngineType::QUALCOMM_IGPU, {output_mem_space}});
                        else
                            out_mapped_engines->push_back(Engine{0, EngineType::CPU, {MemSpace{1, HandleType::CPP}}});
                    }
                }
            }

            if (out_mapped_input_mem_spaces)
            {
                *out_mapped_input_mem_spaces = input_mem_spaces;
                if (out_mapped_input_mem_spaces->empty())
                {
                    out_mapped_input_mem_spaces->assign(inputs.size(), output_mem_space);
                    for (size_t i = 1; i < inputs.size(); ++i)
                    {
                        (*out_mapped_input_mem_spaces)[i] = MemSpace{1, HandleType::CPP};
                    }
                }
            }

            if (!dtypes.empty())
            {
                for (uint64_t i = 0; i < inputs.size(); ++i)
                {
                    uint64_t ruleIdx = std::min(i, static_cast<uint64_t>(dtypes.size() - 1));
                    if (inputs[i].dtype != dtypes[ruleIdx])
                        return false;
                }
            }

            if (match)
            {
                return match(inputs, output);
            }
            return true;
        }

        // 3 & 4. Check memory space topology.
        std::unordered_map<MemSpace, MemSpace> localToActualMem;
        std::unordered_map<MemSpace, MemSpace> actualToLocalMem;

        auto reconcileMem = [&](const MemSpace &local, const MemSpace &actual) {
            if (local.type != actual.type)
                return false;

            auto [fwdIt, fwdInserted] = localToActualMem.try_emplace(local, actual);
            if (!fwdInserted && !(fwdIt->second == actual))
                return false; // same local idx resolved two different ways

            auto [bwdIt, bwdInserted] = actualToLocalMem.try_emplace(actual, local);
            if (!bwdInserted && !(bwdIt->second == local))
                return false; // two different local idxs collapsed onto one actual space

            return true;
        };

        if (!ignore_output_mem_space && output_mem_space.type != HandleType::STORAGE)
        {
            if (!reconcileMem(this->output_mem_space, output_mem_space))
                return false;
        }

        if (!ignore_input_mem_spaces && !this->input_mem_spaces.empty())
        {
            for (uint64_t i = 0; i < inputs.size(); ++i)
            {
                uint64_t ruleIdx = std::min(i, static_cast<uint64_t>(this->input_mem_spaces.size() - 1));
                if (i < input_mem_spaces.size())
                {
                    if (!reconcileMem(this->input_mem_spaces[ruleIdx], input_mem_spaces[i]))
                        return false;
                }
            }
        }

        // 5. Check engine topology and perform local-to-actual engine mapping
        std::unordered_map<Engine, Engine> localToActualEngine;
        std::unordered_map<Engine, Engine> actualToLocalEngine;

        auto reconcileEngine = [&](const Engine &local, const Engine &actual) {
            if (local.type != actual.type)
                return false;

            auto [fwdIt, fwdInserted] = localToActualEngine.try_emplace(local, actual);
            if (!fwdInserted && !(fwdIt->second == actual))
                return false;

            auto [bwdIt, bwdInserted] = actualToLocalEngine.try_emplace(actual, local);
            if (!bwdInserted && !(bwdIt->second == local))
                return false;

            return true;
        };

        if (!ignore_engines && !this->engines.empty() && !engines.empty())
        {
            if (this->engines.size() != engines.size())
            {
                return false;
            }

            for (uint64_t i = 0; i < this->engines.size(); ++i)
            {
                if (!reconcileEngine(this->engines[i], engines[i]))
                    return false;
            }
        }

        // Map any unmapped local engines to available system engines of matching EngineType
        const auto &available_engines = System::get().getAvailableEngines();
        for (const auto &local_eng : this->engines)
        {
            if (localToActualEngine.find(local_eng) == localToActualEngine.end())
            {
                bool mapped = false;
                // First pass: try an unused available engine of the matching EngineType
                for (const auto &avail_eng : available_engines)
                {
                    if (avail_eng.type == local_eng.type &&
                        actualToLocalEngine.find(avail_eng) == actualToLocalEngine.end())
                    {
                        reconcileEngine(local_eng, avail_eng);
                        mapped = true;
                        break;
                    }
                }
                // Second pass fallback: any available engine of matching EngineType
                if (!mapped)
                {
                    for (const auto &avail_eng : available_engines)
                    {
                        if (avail_eng.type == local_eng.type)
                        {
                            reconcileEngine(local_eng, avail_eng);
                            mapped = true;
                            break;
                        }
                    }
                }
                if (!mapped)
                {
                    reconcileEngine(local_eng, Engine{local_eng.idx, local_eng.type});
                }
            }
        }

        // Populate mapped engines output
        if (out_mapped_engines)
        {
            out_mapped_engines->clear();
            for (const auto &local_eng : this->engines)
            {
                auto it = localToActualEngine.find(local_eng);
                if (it != localToActualEngine.end())
                {
                    out_mapped_engines->push_back(it->second);
                }
                else
                {
                    out_mapped_engines->push_back(local_eng);
                }
            }
            if (out_mapped_engines->empty())
            {
                if (output_mem_space.type == HandleType::CUDA)
                    out_mapped_engines->push_back(
                        Engine{output_mem_space.idx, EngineType::CUDA_GPU, {output_mem_space}});
                else if (output_mem_space.type == HandleType::OPENCL)
                    out_mapped_engines->push_back(Engine{0, EngineType::QUALCOMM_IGPU, {output_mem_space}});
                else
                    out_mapped_engines->push_back(Engine{0, EngineType::CPU, {MemSpace{1, HandleType::CPP}}});
            }
        }

        // Populate mapped input memory spaces output
        if (out_mapped_input_mem_spaces)
        {
            out_mapped_input_mem_spaces->clear();
            for (uint64_t i = 0; i < inputs.size(); ++i)
            {
                if (this->input_mem_spaces.empty())
                {
                    out_mapped_input_mem_spaces->push_back(output_mem_space);
                }
                else
                {
                    uint64_t ruleIdx = std::min(i, static_cast<uint64_t>(this->input_mem_spaces.size() - 1));
                    MemSpace local_in = this->input_mem_spaces[ruleIdx];
                    auto it = localToActualMem.find(local_in);
                    if (it != localToActualMem.end())
                    {
                        out_mapped_input_mem_spaces->push_back(it->second);
                    }
                    else if (!ignore_output_mem_space && local_in.type == output_mem_space.type)
                    {
                        out_mapped_input_mem_spaces->push_back(output_mem_space);
                    }
                    else
                    {
                        out_mapped_input_mem_spaces->push_back(local_in);
                    }
                }
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
#endif
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

    mutable std::mutex patternCacheMtx;
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

    std::vector<KernelId> findMatchingKernelsByPatternHash(
        const Graph &patternGraph, LogicalId patternRootId, const std::string &patternHash,
        const std::vector<TensorNode> &inputs, const TensorNode &output, bool reference_only = false,
        MemSpace output_mem_space = {}, const std::vector<MemSpace> &input_mem_spaces = {},
        const std::vector<Engine> &engines = {}, bool ignore_output_mem_space = false,
        bool ignore_input_mem_spaces = false, bool ignore_engines = false, bool ignore_input_contig = false) const
    {
        const TensorNode &rootNode = patternGraph.getNode(patternRootId);
        GraphPatternCacheKey key{rootNode.opType,
                                 rootNode.opName,
                                 patternHash,
                                 reference_only,
                                 ignore_output_mem_space,
                                 ignore_input_mem_spaces,
                                 ignore_engines,
                                 output_mem_space,
                                 input_mem_spaces,
                                 engines,
                                 inputs,
                                 output};

        {
            std::lock_guard<std::mutex> lock(patternCacheMtx);
            auto it = patternCache.find(key);
            if (it != patternCache.end())
            {
                return it->second;
            }
        }

        std::vector<KernelId> matches = _findMatchingKernelsByPattern(
            patternGraph, patternRootId, inputs, output, reference_only, output_mem_space, input_mem_spaces, engines,
            ignore_output_mem_space, ignore_input_mem_spaces, ignore_engines, ignore_input_contig);

        {
            std::lock_guard<std::mutex> lock(patternCacheMtx);
            patternCache[key] = matches;
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
        std::string patternHash = computeGraphHash(patternGraph, {patternRootId});
        return findMatchingKernelsByPatternHash(patternGraph, patternRootId, patternHash, inputs, output,
                                                reference_only, output_mem_space, input_mem_spaces, engines,
                                                ignore_output_mem_space, ignore_input_mem_spaces, ignore_engines,
                                                ignore_input_contig);
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

    uint64_t nKernels() const
    {
        return entries.size();
    }

  private:
    std::unordered_map<KernelId, KernelEntry> entries;
    bool reference_only_mode = false;
};

struct KernelRegistrar
{
    KernelRegistrar(KernelId uid, OpType op, const std::string &opName, uint32_t min_num_inputs,
                    uint32_t max_num_inputs, MatchFunc match, KernelFunc run, ReferenceFactory refFactory,
                    const std::vector<uint32_t> &safe_inplace_idxs, const MemSpace output_mem_space,
                    const std::vector<Engine> &engines, const std::vector<DType> &dtypes = {},
                    const std::vector<std::vector<uint32_t>> &dummyShapes = {},
                    const std::vector<bool> &contiguous = {}, const std::vector<MemSpace> &input_mem_spaces = {},
                    bool is_view = false, bool isReference = false, InferViewFunc inferView = nullptr)
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
#ifndef REGISTER_KERNEL_VIEW
#define REGISTER_KERNEL_VIEW(opName, n_min, n_max, match, ref, inferView, ...)
#endif

#define REGISTER_REF_KERNEL_INTERNAL(uid, op, n_min, n_max, match, run, ...)                                           \
    static KernelRegistrar _registrar_##run(uid, op, "", n_min, n_max, match, run, nullptr, {}, __VA_ARGS__, false,    \
                                            true, nullptr)

#define REGISTER_REF_KERNEL_VIEW_INTERNAL(uid, op, n_min, n_max, match, inferView, ...)                                \
    static KernelRegistrar _registrar_##inferView(uid, op, "", n_min, n_max, match, nullptr, nullptr, {}, __VA_ARGS__, \
                                                  true, true, inferView)

#define REGISTER_KERNEL_INTERNAL(uid, opName, n_min, n_max, match, run, refFactory, ...)                               \
    static KernelRegistrar _registrar_fused_##run(uid, OpType::FUSED, opName, n_min, n_max, match, run, refFactory,    \
                                                  __VA_ARGS__)

#define REGISTER_KERNEL_VIEW_INTERNAL(uid, opName, n_min, n_max, match, refFactory, inferView, ...)                    \
    static KernelRegistrar _registrar_fused_##inferView(uid, OpType::FUSED, opName, n_min, n_max, match, nullptr,      \
                                                        refFactory, {}, __VA_ARGS__, true, false, inferView)
