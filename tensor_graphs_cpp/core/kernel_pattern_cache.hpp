#pragma once

// Shape-independent part of KernelRegistry pattern matching, cached.
//
// KernelRegistry::findMatchingKernelsByPatternHash caches per exact
// input/output configuration; on a miss it rebuilds every fused kernel's
// reference graph and tests isomorphism. Which entries are isomorphic to a
// pattern does not depend on shapes, so this caches that list per pattern hash
// and only runs the entries' match checks per call. The cache key holds
// exactly what isIsomorphic compares (op types, fused op names, structure and
// constant content hashes), so a shape seen for the first time is a hit.
//
// Kept outside core/kernels.hpp: build.py derives kernel UIDs from that file's
// bytes, and changing them invalidates benchmarks/records.bin.

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/graph.hpp"
#include "core/kernels.hpp"

class KernelPatternCache
{
  public:
    static KernelPatternCache &get()
    {
        static KernelPatternCache instance;
        return instance;
    }

    // Mirrors KernelRegistry's reference-only mode (set both through
    // setKernelReferenceOnly).
    void setReferenceOnly(bool refOnly)
    {
        reference_only_mode = refOnly;
    }

    std::vector<KernelId> getCandidates(const Graph &patternGraph, LogicalId patternRootId, bool refOnly = false)
    {
        const KernelRegistry &registry = KernelRegistry::get();
        const bool ref_only = reference_only_mode || refOnly;
        std::string key = ref_only ? "r" : "a";
        appendStructureKey(patternGraph, patternRootId, key);
        std::vector<KernelId> candidates;
        bool found = false;
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto it = cache.find(key);
            if (it != cache.end())
            {
                candidates = it->second;
                found = true;
            }
        }
        if (!found)
        {
            candidates = patternMatchingEntries(registry, patternGraph, patternRootId, ref_only);
            std::lock_guard<std::mutex> lock(mtx);
            cache.emplace(key, candidates);
        }
        return candidates;
    }

    // Same result as KernelRegistry::get().findMatchingKernelsByPatternHash
    // with the same arguments.
    std::vector<KernelId> find(const Graph &patternGraph, LogicalId patternRootId,
                               const std::vector<TensorNode> &inputs, const TensorNode &output, bool reference_only,
                               MemSpace output_mem_space, const std::vector<MemSpace> &input_mem_spaces,
                               const std::vector<Engine> &engines, bool ignore_output_mem_space,
                               bool ignore_input_mem_spaces, bool ignore_engines, bool ignore_input_contig)
    {
        const KernelRegistry &registry = KernelRegistry::get();
        std::vector<KernelId> candidates = getCandidates(patternGraph, patternRootId, reference_only);

        std::vector<KernelId> matches;
        for (KernelId uid : candidates)
        {
            if (registry.getKernel(uid).matches(inputs, output, output_mem_space, input_mem_spaces, engines,
                                                ignore_output_mem_space, ignore_input_mem_spaces, ignore_engines,
                                                ignore_input_contig))
                matches.push_back(uid);
        }
        return matches;
    }

  private:
    // Everything isIsomorphic looks at, in prefix order.
    static void appendStructureKey(const Graph &graph, LogicalId id, std::string &key)
    {
        const TensorNode &node = graph.getNode(id);
        key += "(";
        key += std::to_string(static_cast<uint32_t>(node.opType));
        if (node.opType == OpType::FUSED)
        {
            key += ":";
            key += std::to_string(node.opName.size());
            key += ":";
            key += node.opName;
        }
        if (node.opType == OpType::INPUT)
        {
            key += "#";
            key += node.contentHash;
        }
        else
        {
            for (LogicalId child : node.child_ids)
                appendStructureKey(graph, child, key);
        }
        key += ")";
    }

    // Entries whose pattern is isomorphic to the given one, in registry
    // iteration order (the order KernelRegistry returns matches in).
    static std::vector<KernelId> patternMatchingEntries(const KernelRegistry &registry, const Graph &patternGraph,
                                                        LogicalId patternRootId, bool refOnly)
    {
        std::vector<KernelId> result;
        const TensorNode &pNode = patternGraph.getNode(patternRootId);
        for (const auto &[uid, entry] : registry.getAllKernels())
        {
            if (refOnly && !entry.isReference)
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
            else if (pNode.opType == entry.opType)
            {
                patternMatches = true;
                for (LogicalId pid : pNode.child_ids)
                {
                    const OpType childOp = patternGraph.getNode(pid).opType;
                    if (childOp != OpType::INPUT && childOp != OpType::ARANGE && childOp != OpType::FILL)
                    {
                        patternMatches = false;
                        break;
                    }
                }
            }
            if (patternMatches)
                result.push_back(uid);
        }
        return result;
    }

    std::mutex mtx;
    std::unordered_map<std::string, std::vector<KernelId>> cache;
    std::atomic<bool> reference_only_mode{false};
};

inline void setKernelReferenceOnly(bool refOnly)
{
    KernelRegistry::get().setReferenceOnly(refOnly);
    KernelPatternCache::get().setReferenceOnly(refOnly);
}
