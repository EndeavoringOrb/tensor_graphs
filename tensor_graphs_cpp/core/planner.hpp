#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/cost_model.hpp"
#include "core/kernels.hpp"
#include "core/rewrite.hpp"
#include "core/shapes.hpp"
#include "core/misc.hpp"
#include "core/egraph.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <functional>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>

void propagateDirtyRegionsAtomic(
    const std::vector<uint32_t> &topo,
    const Graph &graph,
    std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
    std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &dirtyInputRegions)
{
    ShapePropagator propagator;

    for (uint32_t nodeId : topo)
    {
        if (!graph.hasNode(nodeId))
            continue;
        const TensorNode &node = graph.getNode(nodeId);
        if (node.opType == OpType::INPUT)
            continue;

        std::vector<std::vector<Region>> parentRegions;
        bool anyParentDirty = false;
        for (uint32_t pid : node.parentIds)
        {
            auto it = dirtyOutputRegions.find(pid);
            if (it != dirtyOutputRegions.end() && !it->second.empty())
            {
                parentRegions.push_back(it->second);
                anyParentDirty = true;
            }
            else
            {
                parentRegions.push_back({});
            }
        }

        std::vector<Region> propagated;
        if (anyParentDirty)
            propagated = propagator.forward(node, graph, parentRegions);

        auto existingIt = dirtyOutputRegions.find(nodeId);
        if (existingIt != dirtyOutputRegions.end() && !existingIt->second.empty())
        {
            if (!propagated.empty())
            {
                dirtyOutputRegions[nodeId] = mergeRegions(propagated, existingIt->second);
            }
        }
        else
        {
            dirtyOutputRegions[nodeId] = propagated;
        }

        dirtyInputRegions[nodeId] = propagator.backward(node, graph, dirtyOutputRegions[nodeId]);
    }
}

struct PeakMemoryResult
{
    std::unordered_map<Backend, uint64_t> peakMemory;
    uint32_t oomEClassId = UINT32_MAX;
};

class Planner
{
private:
    uint32_t egraph_dump_counter_ = 0;

    void dumpEGraphBinary(const EGraph &egraph, uint32_t rootEClassId)
    {
        const std::string dir = "egraph_viewer/egraphs";
        std::filesystem::create_directories(dir);

        std::string path;
        while (true)
        {
            path = dir + "/" + std::to_string(egraph_dump_counter_) + ".bin";
            egraph_dump_counter_++;
            if (!std::filesystem::exists(path))
                break;
        }

        std::ofstream out(path, std::ios::binary);
        if (!out)
        {
            std::cerr << "[Planner.dumpEGraphBinary] Failed to open " << path << " for writing." << std::endl;
            return;
        }

        const auto &classes = egraph.getClasses();
        const auto &enodes = egraph.getENodes();

        uint32_t num_classes = static_cast<uint32_t>(classes.size());
        uint32_t num_enodes = static_cast<uint32_t>(enodes.size());

        out.write(reinterpret_cast<const char *>(&num_classes), 4);
        out.write(reinterpret_cast<const char *>(&num_enodes), 4);
        out.write(reinterpret_cast<const char *>(&rootEClassId), 4);

        for (const auto &cls : classes)
        {
            out.write(reinterpret_cast<const char *>(&cls.id), 4);
            uint32_t s_size = static_cast<uint32_t>(cls.shape.size());
            out.write(reinterpret_cast<const char *>(&s_size), 4);
            if (s_size > 0)
                out.write(reinterpret_cast<const char *>(cls.shape.data()), s_size * 4);

            uint32_t st_size = static_cast<uint32_t>(cls.strides.size());
            out.write(reinterpret_cast<const char *>(&st_size), 4);
            if (st_size > 0)
                out.write(reinterpret_cast<const char *>(cls.strides.data()), st_size * 8);

            out.write(reinterpret_cast<const char *>(&cls.viewOffset), 8);
            out.write(reinterpret_cast<const char *>(&cls.dtype), 4);
            out.write(reinterpret_cast<const char *>(&cls.backend), 4);

            uint32_t e_size = static_cast<uint32_t>(cls.enodes.size());
            out.write(reinterpret_cast<const char *>(&e_size), 4);
            if (e_size > 0)
                out.write(reinterpret_cast<const char *>(cls.enodes.data()), e_size * 4);
        }

        for (const auto &enode : enodes)
        {
            out.write(reinterpret_cast<const char *>(&enode.kernelUid), 8);
            uint32_t op_type = static_cast<uint32_t>(enode.opType);
            out.write(reinterpret_cast<const char *>(&op_type), 4);
            uint32_t n_len = static_cast<uint32_t>(enode.opName.length());
            out.write(reinterpret_cast<const char *>(&n_len), 4);
            if (n_len > 0)
                out.write(enode.opName.c_str(), n_len);

            uint32_t c_size = static_cast<uint32_t>(enode.children.size());
            out.write(reinterpret_cast<const char *>(&c_size), 4);
            if (c_size > 0)
                out.write(reinterpret_cast<const char *>(enode.children.data()), c_size * 4);

            out.write(reinterpret_cast<const char *>(&enode.leafId), 4);

            uint32_t s_size = static_cast<uint32_t>(enode.shape.size());
            out.write(reinterpret_cast<const char *>(&s_size), 4);
            if (s_size > 0)
                out.write(reinterpret_cast<const char *>(enode.shape.data()), s_size * 4);

            uint32_t st_size = static_cast<uint32_t>(enode.strides.size());
            out.write(reinterpret_cast<const char *>(&st_size), 4);
            if (st_size > 0)
                out.write(reinterpret_cast<const char *>(enode.strides.data()), st_size * 8);

            out.write(reinterpret_cast<const char *>(&enode.viewOffset), 8);
            out.write(reinterpret_cast<const char *>(&enode.dtype), 4);
            out.write(reinterpret_cast<const char *>(&enode.backend), 4);
            out.write(reinterpret_cast<const char *>(&enode.sig), 8);
        }

        uint32_t num_constants = static_cast<uint32_t>(egraph.constantStaging.size());
        out.write(reinterpret_cast<const char *>(&num_constants), 4);
        for (const auto &[eclassId, data_ptr] : egraph.constantStaging)
        {
            uint32_t canonId = eclassId;
            out.write(reinterpret_cast<const char *>(&canonId), 4);
            const auto &data = *data_ptr;
            uint64_t data_size = static_cast<uint64_t>(data.size());
            out.write(reinterpret_cast<const char *>(&data_size), 8);
            out.write(reinterpret_cast<const char *>(data.data()), data_size);
        }

        out.close();
        std::cout << "[Planner.dumpEGraphBinary] Dumped EGraph to " << path << std::endl;
    }

    struct ENodeInfo
    {
        float cost;
        std::unordered_map<Backend, uint64_t> memSizes;
        bool inplace;
        int32_t inplace_idx;
        bool isScatter;
        bool isView;
    };

    struct ExtractChoice
    {
        uint32_t enodeId = 0;
        float cost = std::numeric_limits<float>::infinity();
        bool valid = false;
    };

    struct ExtractionResult
    {
        std::unordered_map<uint32_t, ExtractChoice> choiceByEClass;
        float totalCost = std::numeric_limits<float>::infinity();
    };

    CostModel &costModel;
    std::unordered_map<Backend, uint64_t> maxMemoryByBackend;

    uint64_t getMemoryLimit(Backend backend) const
    {
        auto it = maxMemoryByBackend.find(backend);
        if (it != maxMemoryByBackend.end())
            return it->second;
        return std::numeric_limits<uint64_t>::max();
    }

    void inferShapes(const std::vector<uint32_t> &topo, Graph &graph)
    {
        ShapePropagator propagator;
        for (uint32_t nodeId : topo)
        {
            propagator.inferShape(nodeId, graph);
        }
    }

    std::unordered_map<uint32_t, uint32_t> computeRefCounts(const std::vector<uint32_t> &topo, uint32_t rootId, const Graph &graph) const
    {
        std::unordered_map<uint32_t, uint32_t> refCounts;
        for (uint32_t nodeId : topo)
        {
            for (uint32_t pid : graph.getNode(nodeId).parentIds)
            {
                refCounts[pid]++;
            }
        }
        refCounts[rootId] = std::max<uint32_t>(1, refCounts[rootId]);
        return refCounts;
    }

    void saturate(EGraph &egraph, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical, bool injected, bool allowPushDownOnProtected = false, Repo *repo = nullptr)
    {
        RuleCtx ctx{egraph, protectedEClasses, eclassToLogical, repo};
        std::vector<std::unique_ptr<Rule>> rules;
        rules.emplace_back(std::make_unique<FusionRule>());
        if (injected)
        {
            rules.emplace_back(std::make_unique<InfinityDomination>());
            rules.emplace_back(std::make_unique<SlicePushDownElementwise>(allowPushDownOnProtected));
            rules.emplace_back(std::make_unique<SlicePushDownDot>(allowPushDownOnProtected));
        }

        std::map<std::string, uint32_t> ruleMatchCounts;
        size_t iterations = 0;
        bool changed = true;
        uint32_t nMatches = 0;
#ifdef DEBUG
        ProgressTimer timer(0, "saturating ");
#endif
        while (changed)
        {
            if (InterruptManager::isInterrupted())
            {
                std::cerr << "\n[Planner.saturate] Interrupt detected, aborting execution..." << std::endl;
                InterruptManager::cleanup();
                std::exit(SIGINT);
            }
            iterations++;
            uint32_t numENodes = egraph.getENodes().size();
            ProgressTimer timer2(0, "saturation round " + std::to_string(iterations - 1) + " ");
            for (uint32_t eNodeIdx = 0; eNodeIdx < egraph.getENodes().size(); eNodeIdx++)
            {
                for (const auto &rule : rules)
                {
                    if (!rule->match(eNodeIdx, ctx))
                        continue;

                    rule->apply(eNodeIdx, ctx);
                    changed = true;
                    ruleMatchCounts[rule->name()]++;
                    nMatches++;
                }
                timer2.tick();
            }
            egraph.rebuild();
            changed = egraph.getENodes().size() != numENodes;
            std::cout << "\n--- Saturation Summary (" << iterations << " iterations) ---" << std::endl;
            for (auto const &[name, count] : ruleMatchCounts)
            {
                std::cout << "  " << name << ": " << count << " matches" << std::endl;
            }
            std::cout << "Total Matches: " << nMatches << std::endl;
#ifdef DEBUG
            timer.tick();
            std::cout << "# New enodes: " << egraph.getENodes().size() - numENodes << std::endl;
#endif
        }
        std::cout << "Finished saturation in " << iterations << " iterations with " << nMatches << " matches\n"
                  << std::flush;
    }

    std::unordered_map<uint32_t, uint32_t> build_ref_counts(const EGraph &egraph, const std::unordered_map<uint32_t, uint32_t> &selection_map, uint32_t root) const
    {
        std::unordered_map<uint32_t, uint32_t> ref;
        for (const auto &kv : selection_map)
        {
            uint32_t eclass = kv.first;
            uint32_t sel = kv.second;
            const ENode &node = egraph.getENodes()[egraph.getEClass(eclass).enodes[sel]];
            for (uint32_t c : node.children)
            {
                ref[c]++;
            }
        }
        ref[root]++;
        return ref;
    }

    // --- Extracted Helper: Compute Static Memory Baseline ---
    std::unordered_map<Backend, uint64_t> computeStaticBaseline(
        const Graph &graph,
        const std::unordered_map<uint32_t, Backend> &cachedNodes) const
    {
        auto getAlignedSize = [](uint64_t size) -> uint64_t
        {
            return (size + 63) & ~63ULL;
        };

        std::unordered_map<Backend, uint64_t> baseline;

        // 1. Account for all persistent nodes in the logical graph (weights, persistent inputs)
        for (const auto &pair : graph.nodes)
        {
            const TensorNode &node = pair.second;
            if (node.storageType == StorageType::PERSISTENT)
            {
                if (node.backend != Backend::STORAGE)
                {
                    uint64_t size = getAlignedSize(countElements(node.getShape()) * getDTypeSize(node.dtype));
                    baseline[node.backend] += size;
                }
            }
        }

        // 2. Account for all pinned nodes in cachedNodes (protectedCachedNodes)
        for (const auto &kv : cachedNodes)
        {
            uint32_t logicalId = kv.first;
            Backend b = kv.second;
            if (graph.hasNode(logicalId))
            {
                const TensorNode &node = graph.getNode(logicalId);
                if (node.storageType != StorageType::PERSISTENT) // Avoid double counting
                {
                    uint64_t size = getAlignedSize(countElements(node.getShape()) * getDTypeSize(node.dtype));
                    baseline[b] += size;
                }
            }
        }

        return baseline;
    }

    // --- Extracted Helper: Validate Static Memory Baseline ---
    void validateStaticBaseline(const std::unordered_map<Backend, uint64_t> &baseline) const
    {
        for (const auto &kv : baseline)
        {
            auto limitIt = maxMemoryByBackend.find(kv.first);
            if (limitIt != maxMemoryByBackend.end() && kv.second > limitIt->second)
            {
                std::stringstream ss;
                ss << "[Static Memory Limit Exceeded] Static baseline memory of " << kv.second
                   << " bytes on backend " << toString(kv.first)
                   << " exceeds the memory limit of " << limitIt->second << " bytes. "
                   << "This is likely because too many intermediate nodes are being protected/pinned "
                   << "across scheduled buckets, exhausting the hardware limits.";
                Error::throw_err(ss.str());
            }
        }
    }

    PeakMemoryResult computePeakMemory(
        const EGraph &egraph,
        const std::unordered_map<uint32_t, uint32_t> &selection_map,
        const std::vector<ENodeInfo> &enodeInfos,
        uint32_t root,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        const std::unordered_map<uint32_t, uint32_t> &eclassToLogical,
        const Graph &graph,
        const std::vector<uint32_t> &path) const
    {
        auto ref = build_ref_counts(egraph, selection_map, root);

        // Alignment helper matching the allocator's 64-byte block boundary logic
        auto getAlignedSize = [](uint64_t size) -> uint64_t
        {
            return (size + 63) & ~63ULL;
        };

        // Initialize simulation memory with the calculated static baseline
        std::unordered_map<Backend, uint64_t> live_mem = computeStaticBaseline(graph, cachedNodes);

        std::unordered_set<uint32_t> visited;
        std::unordered_map<uint32_t, uint32_t> sim_aliasMap;

        uint32_t oomEClassId = UINT32_MAX;

        // Add selected eclasses that contain constant staging or are chosen as CACHE nodes.
        // Iterating in 'path' order guarantees we identify the earliest decision that causes OOM.
        std::unordered_set<uint32_t> processed;
        for (uint32_t eclassId : path)
        {
            uint32_t canon = egraph.findConst(eclassId);
            if (processed.count(canon))
                continue;
            processed.insert(canon);

            auto mapIt = selection_map.find(canon);
            if (mapIt == selection_map.end())
                continue;

            // Check the selected enode for this class
            uint32_t sel = mapIt->second;
            uint32_t enode_id = egraph.getEClass(canon).enodes[sel];
            const ENode &enode = egraph.getENodes()[enode_id];

            if (enode.opType == OpType::CACHE)
            {
                uint32_t logicalId = eclassToLogical.count(canon) ? eclassToLogical.at(canon) : UINT32_MAX;
                // Only add to baseline if it isn't already included in the static baseline via cachedNodes
                if (logicalId == UINT32_MAX || cachedNodes.count(logicalId) == 0)
                {
                    uint64_t size = getAlignedSize(getSizeBytes(enode.shape, enode.dtype));
                    live_mem[enode.backend] += size;
                    if (live_mem[enode.backend] > getMemoryLimit(enode.backend))
                    {
                        if (oomEClassId == UINT32_MAX)
                        {
                            oomEClassId = canon;
                        }
                    }
                }
            }
            else if (egraph.constantStaging.count(canon))
            {
                uint32_t logicalId = eclassToLogical.count(canon) ? eclassToLogical.at(canon) : UINT32_MAX;
                if (logicalId == UINT32_MAX || !graph.hasNode(logicalId) || graph.getNode(logicalId).storageType != StorageType::PERSISTENT)
                {
                    const EClass &cls = egraph.getEClass(canon);
                    if (cls.backend != Backend::STORAGE)
                    {
                        uint64_t size = getAlignedSize(countElements(cls.shape) * getDTypeSize(cls.dtype));
                        live_mem[cls.backend] += size;
                        if (live_mem[cls.backend] > getMemoryLimit(cls.backend))
                        {
                            if (oomEClassId == UINT32_MAX)
                            {
                                oomEClassId = canon;
                            }
                        }
                    }
                }
            }
        }

        // Start peak memory tracking from the calculated permanent memory baseline
        std::unordered_map<Backend, uint64_t> peak_mem = live_mem;

        auto isPersistent = [&](uint32_t eclass_id) -> bool
        {
            eclass_id = egraph.findConst(eclass_id);
            uint32_t logicalId = eclassToLogical.count(eclass_id) ? eclassToLogical.at(eclass_id) : UINT32_MAX;
            if (logicalId != UINT32_MAX && graph.hasNode(logicalId) && graph.getNode(logicalId).storageType == StorageType::PERSISTENT)
                return true;
            if (egraph.constantStaging.count(eclass_id))
                return true;

            // Treat newly chosen cache nodes as persistent
            auto selIt = selection_map.find(eclass_id);
            if (selIt != selection_map.end())
            {
                uint32_t sel = selIt->second;
                uint32_t enode_id = egraph.getEClass(eclass_id).enodes[sel];
                if (egraph.getENodes()[enode_id].opType == OpType::CACHE)
                {
                    return true;
                }
            }
            return false;
        };

        auto release = [&](auto &self, uint32_t id) -> void
        {
            auto aliasIt = sim_aliasMap.find(id);
            if (aliasIt != sim_aliasMap.end())
            {
                uint32_t targetId = aliasIt->second;
                sim_aliasMap.erase(aliasIt);

                if (ref.find(targetId) != ref.end())
                {
                    ref[targetId]--;
                    if (ref[targetId] == 0)
                    {
                        self(self, targetId);
                    }
                }
                return;
            }

            uint32_t logicalId = eclassToLogical.count(id) ? eclassToLogical.at(id) : UINT32_MAX;
            bool childIsCached = (logicalId != UINT32_MAX && cachedNodes.count(logicalId));
            bool childPersistent = isPersistent(id);

            // Only release transient nodes; permanent/pinned nodes stay allocated
            if (!childIsCached && !childPersistent)
            {
                auto selIt = selection_map.find(id);
                if (selIt != selection_map.end())
                {
                    uint32_t sel = selIt->second;
                    uint32_t enode_id = egraph.getEClass(id).enodes[sel];
                    const ENodeInfo &info = enodeInfos[enode_id];

                    if (!info.inplace && !info.isView)
                    {
                        for (const auto &kv : info.memSizes)
                        {
                            live_mem[kv.first] -= getAlignedSize(kv.second);
                        }
                    }
                }
            }
        };

        std::function<void(uint32_t)> visit = [&](uint32_t eclass)
        {
            eclass = egraph.findConst(eclass);
            if (visited.count(eclass))
                return;
            visited.insert(eclass);

            uint32_t sel = selection_map.at(eclass);
            uint32_t enode_id = egraph.getEClass(eclass).enodes[sel];
            const ENode &node = egraph.getENodes()[enode_id];
            const ENodeInfo &info = enodeInfos[enode_id];

            for (uint32_t c : node.children)
            {
                visit(c);
            }

            uint32_t logicalId = eclassToLogical.count(eclass) ? eclassToLogical.at(eclass) : UINT32_MAX;
            bool isCached = (logicalId != UINT32_MAX && cachedNodes.count(logicalId));
            bool is_perm = isCached || isPersistent(eclass);

            // Dynamically allocate only non-persistent, non-cached transient nodes
            if (!info.inplace && !info.isView && !is_perm)
            {
                for (const auto &kv : info.memSizes)
                {
                    uint64_t size = getAlignedSize(kv.second);
                    live_mem[kv.first] += size;
                    peak_mem[kv.first] = std::max(peak_mem[kv.first], live_mem[kv.first]);
                    if (live_mem[kv.first] > getMemoryLimit(kv.first))
                    {
                        if (oomEClassId == UINT32_MAX)
                        {
                            oomEClassId = eclass;
                        }
                    }
                }
            }

            for (size_t i = 0; i < node.children.size(); ++i)
            {
                uint32_t c = egraph.findConst(node.children[i]);

                if (info.inplace && (int)i == info.inplace_idx)
                {
                    sim_aliasMap[eclass] = c;
                    continue;
                }

                if (info.isView && i == 0)
                {
                    sim_aliasMap[eclass] = c;
                    continue;
                }

                ref[c]--;
                if (ref[c] == 0)
                {
                    release(release, c);
                }
            }
        };

        visit(root);
        return {peak_mem, oomEClassId};
    }

    std::vector<uint32_t> getEClassTopoOrder(const EGraph &egraph, const std::unordered_map<uint32_t, uint32_t> &selectionMap, uint32_t rootEClassId) const
    {
        std::vector<uint32_t> topo;
        std::unordered_set<uint32_t> visited_classes;
        std::function<void(uint32_t)> visit = [&](uint32_t eclassId)
        {
            eclassId = egraph.findConst(eclassId);
            if (visited_classes.count(eclassId))
                return;
            visited_classes.insert(eclassId);

            auto choiceIt = selectionMap.find(eclassId);
            if (choiceIt == selectionMap.end())
                return;

            const ENode &enode = egraph.getENodes()[egraph.getEClass(eclassId).enodes[choiceIt->second]];
            for (uint32_t child : enode.children)
                visit(child);
            topo.push_back(eclassId);
        };

        visit(rootEClassId);
        return topo;
    }

    bool validateInplaceSchedules(
        const EGraph &egraph,
        const std::unordered_map<uint32_t, uint32_t> &selection_map,
        const std::vector<ENodeInfo> &enodeInfos,
        uint32_t rootEClassId,
        const std::unordered_map<uint32_t, uint32_t> &eclassToLogical,
        std::string &outReason,
        uint32_t &conflictEClass1,
        uint32_t &conflictEClass2) const
    {
        std::vector<uint32_t> topo = getEClassTopoOrder(egraph, selection_map, rootEClassId);
        std::unordered_map<uint32_t, uint32_t> overwritten;         // overwritten child_root -> eclass
        std::unordered_map<uint32_t, uint32_t> overwritten_logical; // overwritten logicalId -> eclass
        std::unordered_map<uint32_t, uint32_t> mem_root;            // eclass -> root eclass

        for (size_t i = 0; i < topo.size(); i++)
        {
            auto choiceIt = selection_map.find(topo[i]);
            if (choiceIt == selection_map.end())
            {
                Error::throw_err("[Planner.validateInplaceSchedules] Selection not found for active eclass");
            }

            const uint32_t eclass = topo[i];
            const uint32_t enodeId = egraph.getEClass(eclass).enodes[choiceIt->second];
            const ENode &enode = egraph.getENodes()[enodeId];
            const ENodeInfo &info = enodeInfos[enodeId];

            if (info.isView && !enode.children.empty())
            {
                uint32_t child_eclass = egraph.findConst(enode.children[0]);
                mem_root[eclass] = mem_root.count(child_eclass) ? mem_root[child_eclass] : child_eclass;
            }
            else
            {
                mem_root[eclass] = eclass;
            }

            for (size_t j = 0; j < enode.children.size(); j++)
            {
                uint32_t child_eclass = egraph.findConst(enode.children[j]);
                uint32_t child_root = mem_root.count(child_eclass) ? mem_root[child_eclass] : child_eclass;

                if (overwritten.count(child_root))
                {
                    outReason = "inplace " + std::to_string(child_root);
                    conflictEClass1 = overwritten[child_root];
                    conflictEClass2 = eclass;
                    return false;
                }

                uint32_t logicalId = eclassToLogical.count(child_root) ? eclassToLogical.at(child_root) : UINT32_MAX;
                if (logicalId != UINT32_MAX && overwritten_logical.count(logicalId))
                {
                    // Only conflict if we're reading an old/different version of the logical node,
                    // not the eclass that actually performed the inplace update.
                    if (overwritten_logical[logicalId] != child_root)
                    {
                        outReason = "inplace_logical " + std::to_string(logicalId);
                        conflictEClass1 = overwritten_logical[logicalId];
                        conflictEClass2 = eclass;
                        return false;
                    }
                }
            }

            if (info.inplace)
            {
                uint32_t inplace_child = egraph.findConst(enode.children[info.inplace_idx]);
                uint32_t child_root = mem_root.count(inplace_child) ? mem_root[inplace_child] : inplace_child;
                overwritten[child_root] = eclass;

                uint32_t logicalId = eclassToLogical.count(child_root) ? eclassToLogical.at(child_root) : UINT32_MAX;
                if (logicalId != UINT32_MAX)
                {
                    overwritten_logical[logicalId] = eclass;
                }
            }
        }

        return true;
    }

    ExtractionResult extractBest(const uint32_t rootId, const Graph &graph, EGraph &egraph,
                                 const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                                 const std::unordered_map<Backend, uint64_t> &maxMemoryByBackend,
                                 const std::unordered_map<uint32_t, Backend> &cachedNodes,
                                 const std::unordered_map<uint32_t, uint32_t> &eclassToLogical,
                                 const std::unordered_set<uint32_t> &immutable_eclasses,
                                 bool stopOnFirstValid = true,
                                 bool strictCache = false)
    {
        constexpr float INF = std::numeric_limits<float>::infinity();
        constexpr float EPS = 1e-6f;

        auto isConstantNeeded = [](OpType op, size_t inputIdx, size_t numInputs) -> bool
        {
            if (op == OpType::REPEAT && (inputIdx == 1 || inputIdx == 2))
                return true;
            if (op == OpType::RESHAPE && inputIdx == 1)
                return true;
            if (op == OpType::PERMUTE && inputIdx == 1)
                return true;
            if (op == OpType::SLICE && (inputIdx == 1 || inputIdx == 2 || inputIdx == 3))
                return true;
            if (op == OpType::SCATTER && (inputIdx == 2 || inputIdx == 3 || inputIdx == 4))
                return true;
            if ((op == OpType::SUM || op == OpType::MAX) && inputIdx == 1)
                return true;
            if (op == OpType::CONCAT && inputIdx == numInputs - 1)
                return true;
            if (op == OpType::TRIU && inputIdx == 1)
                return true;
            if (op == OpType::FILL && inputIdx == 1)
                return true;
            if (op == OpType::IM2COL && (inputIdx == 1 || inputIdx == 2 || inputIdx == 3))
                return true;
            if (op == OpType::ARANGE && (inputIdx == 0 || inputIdx == 1 || inputIdx == 2))
                return true;
            if (op == OpType::ARGMAX && (inputIdx == 1 || inputIdx == 2))
                return true;
            return false;
        };

        ProgressTimer timer3(egraph.getENodes().size(), "calculating enode info ");
        std::vector<ENodeInfo> enodeInfos(egraph.getENodes().size());
        for (size_t i = 0; i < egraph.getENodes().size(); ++i)
        {
            const ENode &enode = egraph.getENodes()[i];
            ENodeInfo info;
            info.memSizes[enode.backend] = getSizeBytes(enode.shape, enode.dtype);
            info.inplace = false;
            info.inplace_idx = -1;
            info.isScatter = false;
            info.isView = false;

            if (enode.kernelUid != 0)
            {
                const auto &kernel = KernelRegistry::get().getKernel(enode.kernelUid);
                info.inplace = kernel.inplace;
                info.isView = kernel.isView;
                if (info.inplace && kernel.numInputs > 0)
                {
                    info.inplace_idx = 0;
                }

                if (enode.opType == OpType::SCATTER)
                {
                    info.isScatter = true;
                }
                else if (enode.opType == OpType::FUSED)
                {
                    const auto *refEntry = ReferenceGraphRegistry::get().getFactory(kernel.opName);
                    if (refEntry)
                    {
                        Graph pGraph;
                        std::vector<uint32_t> pInputs;
                        for (size_t k = 0; k < kernel.numInputs; ++k)
                        {
                            pInputs.push_back(pGraph.input(kernel.dummyShapes[k], kernel.dtypes[k]));
                        }
                        uint32_t pRoot = refEntry->factory(pInputs, pGraph);
                        if (pGraph.getNode(pRoot).opType == OpType::SCATTER)
                        {
                            info.isScatter = true;
                        }
                    }
                }
            }

            if (enode.opType == OpType::INPUT || enode.opType == OpType::CACHE)
            {
                info.cost = 0.0f;
                if (strictCache && (enode.leafId & 0x80000000))
                {
                    uint32_t eclassId = egraph.getENodeEClass(i);
                    uint32_t canonId = egraph.findConst(eclassId);
                    uint32_t logicalId = eclassToLogical.count(canonId) ? eclassToLogical.at(canonId) : UINT32_MAX;
                    if (logicalId == UINT32_MAX || cachedNodes.find(logicalId) == cachedNodes.end())
                    {
                        info.cost = INF;
                    }
                    else if (enode.backend != cachedNodes.at(logicalId))
                    {
                        info.cost = INF;
                    }
                }
            }
            else if (enode.kernelUid != 0)
            {

                std::vector<std::vector<uint32_t>> inShapes;
                std::vector<std::vector<uint64_t>> inStrides;
                std::vector<DType> inDTypes;
                std::vector<std::vector<uint8_t>> inConstants;

                inShapes.reserve(enode.children.size());
                inStrides.reserve(enode.children.size());
                inDTypes.reserve(enode.children.size());
                inConstants.reserve(enode.children.size());

                const ReferenceGraphEntry *refEntry = nullptr;
                std::unique_ptr<Graph> pGraph;
                std::vector<uint32_t> pInputs;

                const auto &kernel = KernelRegistry::get().getKernel(enode.kernelUid);
                if (enode.opType == OpType::FUSED)
                {
                    refEntry = ReferenceGraphRegistry::get().getFactory(kernel.opName);
                    if (refEntry)
                    {
                        pGraph = std::make_unique<Graph>();
                        for (size_t k = 0; k < kernel.numInputs; ++k)
                        {
                            pInputs.push_back(pGraph->input(kernel.dummyShapes[k], kernel.dtypes[k]));
                        }
                        refEntry->factory(pInputs, *pGraph);
                    }
                }

                for (size_t j = 0; j < enode.children.size(); j++)
                {
                    uint32_t childEClassId = enode.children[j];
                    const EClass &childCls = egraph.getEClass(egraph.find(childEClassId));
                    inShapes.push_back(childCls.shape);

                    std::vector<uint64_t> strides_cast;
                    strides_cast.reserve(childCls.strides.size());
                    for (uint64_t s : childCls.strides)
                        strides_cast.push_back(s);
                    inStrides.push_back(std::move(strides_cast));

                    inDTypes.push_back(childCls.dtype);

                    uint32_t canonChild = egraph.find(childEClassId);
                    bool needed = false;

                    if (enode.opType == OpType::FUSED)
                    {
                        if (refEntry && pGraph)
                        {
                            auto traceToInputIdx = [&](uint32_t pid) -> int
                            {
                                uint32_t curr = pid;
                                while (pGraph->hasNode(curr) &&
                                       (pGraph->getNode(curr).opType == OpType::CONTIGUOUS ||
                                        pGraph->getNode(curr).opType == OpType::CAST ||
                                        pGraph->getNode(curr).opType == OpType::COPY_TO ||
                                        pGraph->getNode(curr).opType == OpType::RESHAPE ||
                                        pGraph->getNode(curr).opType == OpType::PERMUTE))
                                {
                                    if (pGraph->getNode(curr).parentIds.empty())
                                        break;
                                    curr = pGraph->getNode(curr).parentIds[0];
                                }
                                for (size_t k = 0; k < pInputs.size(); ++k)
                                {
                                    if (pInputs[k] == curr)
                                        return (int)k;
                                }
                                return -1;
                            };

                            for (const auto &pair : pGraph->nodes)
                            {
                                const TensorNode &n = pair.second;
                                for (size_t p_idx = 0; p_idx < n.parentIds.size(); ++p_idx)
                                {
                                    if (isConstantNeeded(n.opType, p_idx, n.parentIds.size()))
                                    {
                                        int inputIdx = traceToInputIdx(n.parentIds[p_idx]);
                                        if (kernel.isVariadic)
                                        {
                                            if (inputIdx == (int)kernel.numInputs - 1 && j == enode.children.size() - 1)
                                            {
                                                needed = true;
                                                break;
                                            }
                                            else if (inputIdx >= 0 && inputIdx < (int)kernel.numInputs - 1 && j < enode.children.size() - 1)
                                            {
                                                needed = true;
                                                break;
                                            }
                                        }
                                        else if (inputIdx == (int)j)
                                        {
                                            needed = true;
                                            break;
                                        }
                                    }
                                }
                                if (needed)
                                    break;
                            }
                        }
                    }
                    else
                    {
                        needed = isConstantNeeded(enode.opType, j, enode.children.size());
                    }

                    if (!needed)
                    {
                        inConstants.push_back({});
                    }
                    else if (egraph.constantStaging.count(canonChild))
                    {
                        inConstants.push_back(*egraph.constantStaging.at(canonChild));
                    }
                    else
                    {
                        inConstants.push_back({});
                    }
                }

                info.cost = costModel.estimateCost(
                    enode.kernelUid,
                    enode.shape,
                    enode.strides,
                    enode.dtype,
                    inShapes, inStrides, inDTypes, inConstants);

                if (info.inplace && info.inplace_idx >= 0)
                {
                    uint32_t mutated_eclass = egraph.find(enode.children[info.inplace_idx]);
                    if (immutable_eclasses.count(mutated_eclass) && !info.isScatter)
                    {
                        info.cost = INF;
                    }
                }
            }
            else
            {
                Error::throw_err("[Planner.extractBest] enode.kernelUid != 0, but isn't OpType::INPUT or OpType::CACHE. this shouldn't happen");
            }

            enodeInfos[i] = std::move(info);
            timer3.tick();
        }

        bool droppedInf = false;
        for (size_t i = 0; i < egraph.getClasses().size(); ++i)
        {
            uint32_t eclassId = egraph.find(static_cast<uint32_t>(i));
            if (eclassId != i)
                continue;

            EClass &cls = egraph.getEClass(eclassId);
            std::vector<uint32_t> validEnodes;
            validEnodes.reserve(cls.enodes.size());

            for (uint32_t enodeId : cls.enodes)
            {
                if (enodeInfos[enodeId].cost == INF)
                {
                    droppedInf = true;
                }
                else
                {
                    validEnodes.push_back(enodeId);
                }
            }

            cls.enodes = std::move(validEnodes);
        }

        if (droppedInf)
        {
            std::cout << "[Planner.extractBest] Warning: Filtered out nodes with infinite cost. "
                      << "You may need to run 'bench' to gather missing kernel performance data." << std::endl;
        }

        auto rootIt = nodeToEClass.find(rootId);
        if (rootIt == nodeToEClass.end())
        {
            Error::throw_err("[Planner.extractBest] Root node missing from nodeToEClass.");
        }
        uint32_t rootEClassId = egraph.find(rootIt->second);

        const size_t numClasses = egraph.getClasses().size();

        std::vector<uint32_t> canonicalClasses;
        std::vector<uint32_t> classToBitIdx(numClasses, UINT32_MAX);
        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t eclassId = egraph.find(static_cast<uint32_t>(i));
            if (eclassId == i)
            {
                classToBitIdx[i] = static_cast<uint32_t>(canonicalClasses.size());
                canonicalClasses.push_back(i);
            }
        }

        const size_t numCanonical = canonicalClasses.size();
        const size_t bitWords = numCanonical == 0 ? 0 : (numCanonical + 63) >> 6;

        auto bitTest = [&](const std::vector<uint64_t> &bits, uint32_t eclassId) -> bool
        {
            uint32_t idx = classToBitIdx[eclassId];
            if (idx == UINT32_MAX || bits.empty())
                return false;
            return (bits[idx >> 6] >> (idx & 63)) & 1ULL;
        };

        auto bitSet = [&](std::vector<uint64_t> &bits, uint32_t eclassId)
        {
            uint32_t idx = classToBitIdx[eclassId];
            if (idx != UINT32_MAX && !bits.empty())
            {
                bits[idx >> 6] |= (1ULL << (idx & 63));
            }
        };

        struct OptSummary
        {
            float cost = INF;
            float intrinsic = INF;
            uint32_t chosenEnode = UINT32_MAX;
            std::vector<uint64_t> coveredBits;
            bool valid = false;
        };

        std::vector<OptSummary> opt(numClasses);
        for (uint32_t canonId : canonicalClasses)
        {
            opt[canonId].coveredBits.assign(bitWords, 0);
        }

        std::vector<std::vector<uint32_t>> parentMap(numClasses);
        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t eclassId = egraph.find(static_cast<uint32_t>(i));
            if (eclassId != i)
                continue;

            const EClass &cls = egraph.getEClass(eclassId);
            for (uint32_t enodeId : cls.enodes)
            {
                const ENode &enode = egraph.getENodes()[enodeId];
                for (uint32_t child : enode.children)
                {
                    uint32_t childEClass = egraph.find(child);
                    parentMap[childEClass].push_back(eclassId);
                }
            }
        }

        for (auto &parents : parentMap)
        {
            std::sort(parents.begin(), parents.end());
            parents.erase(std::unique(parents.begin(), parents.end()), parents.end());
        }

        std::vector<uint32_t> worklist;
        std::vector<uint32_t> next_worklist;
        std::vector<bool> inQueue(numClasses, false);

        worklist.reserve(numClasses);
        next_worklist.reserve(numClasses);

        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t eclassId = egraph.find(static_cast<uint32_t>(i));
            if (eclassId == i)
            {
                worklist.push_back(eclassId);
                inQueue[eclassId] = true;
            }
        }

        std::vector<uint64_t> candidateBits(bitWords, 0);
        std::vector<float> optimisticEnodeDagCost(egraph.getENodes().size(), INF);

        ProgressTimer optTimer(0, "calculating optimistic cost");
        while (!worklist.empty())
        {
            for (uint32_t eclassId : worklist)
            {
                inQueue[eclassId] = false;

                const EClass &cls = egraph.getEClass(eclassId);
                OptSummary best;
                best.coveredBits.assign(bitWords, 0);

                for (uint32_t enodeId : cls.enodes)
                {
                    const ENodeInfo &info = enodeInfos[enodeId];
                    if (info.cost == INF)
                        continue;

                    std::fill(candidateBits.begin(), candidateBits.end(), 0);
                    float candidateCost = info.cost;
                    bool candidateValid = true;

                    bitSet(candidateBits, eclassId);

                    const ENode &enode = egraph.getENodes()[enodeId];

                    std::vector<uint32_t> childEClasses;
                    childEClasses.reserve(enode.children.size());
                    for (uint32_t child : enode.children)
                    {
                        childEClasses.push_back(egraph.find(child));
                    }
                    std::sort(childEClasses.begin(), childEClasses.end());
                    childEClasses.erase(std::unique(childEClasses.begin(), childEClasses.end()), childEClasses.end());

                    for (uint32_t childEClass : childEClasses)
                    {
                        if (childEClass == eclassId)
                        {
                            candidateValid = false;
                            break;
                        }

                        const OptSummary &childOpt = opt[childEClass];
                        if (!childOpt.valid)
                        {
                            candidateValid = false;
                            break;
                        }

                        if (bitTest(childOpt.coveredBits, eclassId))
                        {
                            candidateValid = false;
                            break;
                        }

                        for (size_t w = 0; w < bitWords; ++w)
                        {
                            uint64_t newBits = childOpt.coveredBits[w] & ~candidateBits[w];
                            if (!newBits)
                                continue;

                            candidateBits[w] |= newBits;

                            while (newBits)
                            {
#if defined(__GNUG__) || defined(__clang__)
                                unsigned bit = static_cast<unsigned>(__builtin_ctzll(newBits));
#else
                                unsigned bit = 0;
                                uint64_t tmp = newBits;
                                while ((tmp & 1ULL) == 0)
                                {
                                    tmp >>= 1;
                                    ++bit;
                                }
#endif
                                uint32_t k_idx = static_cast<uint32_t>((w << 6) + bit);
                                if (k_idx < numCanonical)
                                {
                                    uint32_t k_eclass = canonicalClasses[k_idx];
                                    candidateCost += opt[k_eclass].intrinsic;
                                }
                                newBits &= (newBits - 1);
                            }
                        }
                    }

                    if (!candidateValid)
                        continue;

                    optimisticEnodeDagCost[enodeId] = candidateCost;

                    if (!best.valid ||
                        candidateCost < best.cost - EPS ||
                        (std::abs(candidateCost - best.cost) <= EPS && enodeId < best.chosenEnode))
                    {
                        best.valid = true;
                        best.cost = candidateCost;
                        best.intrinsic = info.cost;
                        best.chosenEnode = enodeId;
                        best.coveredBits = candidateBits;
                    }
                }

                if (!best.valid)
                    continue;

                if (!opt[eclassId].valid ||
                    best.cost < opt[eclassId].cost - EPS ||
                    (std::abs(best.cost - opt[eclassId].cost) <= EPS && best.chosenEnode < opt[eclassId].chosenEnode))
                {
                    opt[eclassId] = std::move(best);

                    for (uint32_t parentId : parentMap[eclassId])
                    {
                        if (!inQueue[parentId])
                        {
                            inQueue[parentId] = true;
                            next_worklist.push_back(parentId);
                        }
                    }
                }
            }

            worklist.clear();
            std::swap(worklist, next_worklist);
            optTimer.tick();
        }

        std::vector<float> eclassMinCost(numClasses, INF);
        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t eclassId = egraph.find(static_cast<uint32_t>(i));
            if (eclassId == i && opt[eclassId].valid)
            {
                eclassMinCost[eclassId] = opt[eclassId].cost;
            }
        }

        std::vector<uint64_t> tempBits(bitWords, 0);
        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t eclassId = egraph.find(static_cast<uint32_t>(i));
            if (eclassId != i)
                continue;

            const EClass &cls = egraph.getEClass(eclassId);
            for (uint32_t enodeId : cls.enodes)
            {
                const ENodeInfo &info = enodeInfos[enodeId];
                if (info.cost == INF)
                {
                    optimisticEnodeDagCost[enodeId] = INF;
                    continue;
                }

                std::fill(tempBits.begin(), tempBits.end(), 0);
                bitSet(tempBits, eclassId);

                float total = info.cost;
                bool valid = true;

                const ENode &enode = egraph.getENodes()[enodeId];

                std::vector<uint32_t> childEClasses;
                childEClasses.reserve(enode.children.size());
                for (uint32_t child : enode.children)
                {
                    childEClasses.push_back(egraph.find(child));
                }
                std::sort(childEClasses.begin(), childEClasses.end());
                childEClasses.erase(std::unique(childEClasses.begin(), childEClasses.end()), childEClasses.end());

                for (uint32_t childEClass : childEClasses)
                {
                    if (childEClass == eclassId)
                    {
                        valid = false;
                        break;
                    }
                    if (!opt[childEClass].valid)
                    {
                        valid = false;
                        break;
                    }
                    if (bitTest(opt[childEClass].coveredBits, eclassId))
                    {
                        valid = false;
                        break;
                    }

                    for (size_t w = 0; w < bitWords; ++w)
                    {
                        uint64_t newBits = opt[childEClass].coveredBits[w] & ~tempBits[w];
                        if (!newBits)
                            continue;

                        tempBits[w] |= newBits;

                        while (newBits)
                        {
#if defined(__GNUG__) || defined(__clang__)
                            unsigned bit = static_cast<unsigned>(__builtin_ctzll(newBits));
#else
                            unsigned bit = 0;
                            uint64_t tmp = newBits;
                            while ((tmp & 1ULL) == 0)
                            {
                                tmp >>= 1;
                                ++bit;
                            }
#endif
                            uint32_t k_idx = static_cast<uint32_t>((w << 6) + bit);
                            if (k_idx < numCanonical)
                            {
                                uint32_t k_eclass = canonicalClasses[k_idx];
                                total += opt[k_eclass].intrinsic;
                            }
                            newBits &= (newBits - 1);
                        }
                    }
                }

                optimisticEnodeDagCost[enodeId] = valid ? total : INF;
            }
        }

        for (size_t i = 0; i < numClasses; ++i)
        {
            uint32_t eclassId = egraph.find(static_cast<uint32_t>(i));
            if (eclassId != i)
                continue;

            EClass &cls = egraph.getEClass(eclassId);
            std::sort(cls.enodes.begin(), cls.enodes.end(),
                      [&](uint32_t a, uint32_t b)
                      {
                          float costA = optimisticEnodeDagCost[a];
                          float costB = optimisticEnodeDagCost[b];

                          if (costA < costB)
                              return true;
                          if (costA > costB)
                              return false;
                          return a < b;
                      });
        }

        if (egraph.getEClass(rootEClassId).enodes.size() == 0)
        {
            Error::throw_err("[Planner.extractBest] no valid extractions");
        }
        std::cout << "[Planner.extractBest] Optimistic root cost: "
                  << std::to_string(optimisticEnodeDagCost[egraph.getEClass(rootEClassId).enodes[0]]) << std::endl;

        // TODO: prune dominated enodes. if A.cost < B.cost and they have the same inputs, and have the same inplace&view status, then remove B. basically if there are two kernels implementing the same thing but one is faster on this specific hardware/shape, we don't need to bloat the search space with B. maybe we can do this even earlier, as we don't need dag cost as long as A and B have same inputs and inplace and view stuff.

        std::unordered_map<uint32_t, uint32_t> selection_map;
        std::vector<uint32_t> path;
        std::vector<uint32_t> to_process = {rootEClassId};
        std::vector<uint32_t> to_process_enode;
        std::unordered_map<uint32_t, uint32_t> next_sel;

        float best_cost = INF;
        std::unordered_map<uint32_t, uint32_t> best_selection_map;
        std::unordered_map<Backend, uint64_t> minPeakMemSeen;

        int max_iters = 100000;
        int remaining_iters = max_iters;
        ProgressTimer timer(max_iters, "extracting graphs ");

        while (remaining_iters-- > 0)
        {
            timer.tick();
            bool valid = true;
            std::string reason = "";
            float current_cost = 0.0f;

            for (const auto &kv : selection_map)
            {
                uint32_t eclass = kv.first;
                uint32_t sel = kv.second;
                current_cost += enodeInfos[egraph.getEClass(eclass).enodes[sel]].cost;
            }

            while (!to_process.empty())
            {
                uint32_t current = to_process.front();
                to_process.erase(to_process.begin());

                if (selection_map.find(current) != selection_map.end())
                {
                    continue;
                }

                path.push_back(current);

                uint32_t sel = 0;
                auto nextIt = next_sel.find(current);
                if (nextIt != next_sel.end())
                {
                    sel = nextIt->second;
                    next_sel.erase(nextIt);
                }

                const auto &enodes = egraph.getEClass(current).enodes;
                if (sel >= enodes.size())
                {
                    Error::throw_err("Invalid selection index in EGraph");
                }

                uint32_t enode_id = enodes[sel];
                const ENode &node = egraph.getENodes()[enode_id];
                const ENodeInfo &info = enodeInfos[enode_id];

                selection_map[current] = sel;
                current_cost += info.cost;

                if (info.cost == INF)
                {
                    valid = false;
                    reason = "cost=inf";
                }

                if (best_cost != INF && current_cost >= best_cost)
                {
                    valid = false;
                    reason = "cost=" + std::to_string(current_cost);
                }

                if (enodes.size() > sel + 1)
                {
                    if (std::find(to_process_enode.begin(), to_process_enode.end(), current) == to_process_enode.end())
                    {
                        to_process_enode.push_back(current);
                    }
                }

                if (!valid)
                    break;

                std::vector<uint32_t> new_to_process;
                new_to_process.reserve(node.children.size());
                for (uint32_t child : node.children)
                {
                    uint32_t childEClass = egraph.find(child);
                    if (selection_map.find(childEClass) == selection_map.end())
                    {
                        new_to_process.push_back(childEClass);
                    }
                }
                to_process.insert(to_process.begin(), new_to_process.begin(), new_to_process.end());
            }

            if (valid)
            {
                std::vector<uint32_t> indegree(numClasses, 0);
                for (const auto &kv : selection_map)
                {
                    uint32_t sel = kv.second;
                    const ENode &enode = egraph.getENodes()[egraph.getEClass(kv.first).enodes[sel]];
                    for (uint32_t child : enode.children)
                    {
                        indegree[egraph.find(child)]++;
                    }
                }

                std::vector<uint32_t> zero_indegree;
                for (const auto &kv : selection_map)
                {
                    if (indegree[kv.first] == 0)
                    {
                        zero_indegree.push_back(kv.first);
                    }
                }

                uint32_t processed = 0;
                while (!zero_indegree.empty())
                {
                    uint32_t curr = zero_indegree.back();
                    zero_indegree.pop_back();
                    processed++;

                    uint32_t sel = selection_map[curr];
                    const ENode &enode = egraph.getENodes()[egraph.getEClass(curr).enodes[sel]];
                    for (uint32_t child : enode.children)
                    {
                        uint32_t canonChild = egraph.find(child);
                        indegree[canonChild]--;
                        if (indegree[canonChild] == 0)
                        {
                            zero_indegree.push_back(canonChild);
                        }
                    }
                }

                if (processed < selection_map.size())
                {
                    valid = false;
                    reason = "cycle";
                }
            }

            PeakMemoryResult peakResult;
            std::unordered_map<Backend, uint64_t> peak;
            uint32_t oomEClassId = UINT32_MAX;
            if (valid)
            {
                peakResult = computePeakMemory(
                    egraph, selection_map, enodeInfos, rootEClassId, cachedNodes, eclassToLogical, graph, path);
                peak = peakResult.peakMemory;
                oomEClassId = peakResult.oomEClassId;

                bool newMinSeen = false;
                for (const auto &kv : peak)
                {
                    auto mit = minPeakMemSeen.find(kv.first);
                    if (mit == minPeakMemSeen.end() || kv.second < mit->second)
                    {
                        minPeakMemSeen[kv.first] = kv.second;
                        newMinSeen = true;
                    }
                }
                if (newMinSeen)
                {
                    std::cout << "[Planner.extractBest] New lowest peak mem candidate seen (iter " << (max_iters - remaining_iters) << "): ";
                    for (auto it = minPeakMemSeen.begin(); it != minPeakMemSeen.end(); ++it)
                    {
                        if (it != minPeakMemSeen.begin())
                            std::cout << ", ";
                        std::cout << toString(it->first) << ": " << it->second << " bytes";
                    }
                    std::cout << std::endl;
                }

                for (const auto &kv : maxMemoryByBackend)
                {
                    if (peak[kv.first] > kv.second)
                    {
                        valid = false;
                        reason = "OOM";
                        break;
                    }
                }
            }

            uint32_t conflictEClass1 = UINT32_MAX;
            uint32_t conflictEClass2 = UINT32_MAX;

            if (valid)
            {
                valid = validateInplaceSchedules(egraph, selection_map, enodeInfos, rootEClassId, eclassToLogical, reason, conflictEClass1, conflictEClass2);
            }

            if (valid)
            {
                if (current_cost < best_cost)
                {
                    float actual_current_cost = 0.0f;

                    for (const auto &kv : selection_map)
                    {
                        uint32_t eclass = kv.first;
                        uint32_t sel = kv.second;
                        actual_current_cost += enodeInfos[egraph.getEClass(eclass).enodes[sel]].cost;
                    }
                    if (actual_current_cost != current_cost)
                    {
                        std::cout << "WARNING actual cost (" + std::to_string(actual_current_cost) + ") != current cost (" + std::to_string(current_cost) + ")" << std::endl;
                    }
                    best_cost = current_cost;
                    best_selection_map = selection_map;
                    std::cout << "new best cost: " << std::to_string(best_cost) << std::endl;
                    std::cout << "peak mem: " << toString(peak) << std::endl;
                }

                if (stopOnFirstValid)
                {
                    break;
                }
            }

            if (!valid)
            {
                std::cout << "[Planner.extractBest] [iter " << std::to_string(max_iters - remaining_iters) << "] invalid reason: " << reason << std::endl;
            }

            if (to_process_enode.empty())
                break;

            uint32_t target_backtrack_eclass = UINT32_MAX;
            if (!valid && conflictEClass1 != UINT32_MAX && conflictEClass2 != UINT32_MAX)
            {
                int idx1 = -1;
                int idx2 = -1;
                for (int i = 0; i < (int)path.size(); ++i)
                {
                    if (path[i] == conflictEClass1)
                        idx1 = i;
                    if (path[i] == conflictEClass2)
                        idx2 = i;
                }
                if (idx1 >= 0 && idx2 >= 0)
                {
                    target_backtrack_eclass = path[std::max(idx1, idx2)];
                }
            }
            else if (!valid && reason == "OOM" && oomEClassId != UINT32_MAX)
            {
                for (int i = 0; i < (int)path.size(); ++i)
                {
                    if (path[i] == oomEClassId)
                    {
                        target_backtrack_eclass = path[i];
                        break;
                    }
                }
            }

            bool skip_increment = (target_backtrack_eclass != UINT32_MAX);
            std::cout << "path size " << std::to_string(path.size()) << std::endl;

            while (!path.empty())
            {
                uint32_t current = path.back();
                path.pop_back();

                if (selection_map.find(current) == selection_map.end())
                    continue;

                if (skip_increment && current == target_backtrack_eclass)
                {
                    std::cout << "skipped back to path size " << std::to_string(path.size()) << std::endl;
                    skip_increment = false;
                }

                uint32_t sel = selection_map[current];
                const auto &enodes = egraph.getEClass(current).enodes;
                uint32_t enode_id = enodes[sel];
                const ENode &node = egraph.getENodes()[enode_id];
                const ENodeInfo &info = enodeInfos[enode_id];

                if (!skip_increment && sel + 1 < enodes.size())
                {
                    next_sel[current] = sel + 1;

                    std::vector<uint32_t> keys_to_delete;
                    keys_to_delete.reserve(selection_map.size());
                    for (const auto &kv : selection_map)
                    {
                        if (std::find(path.begin(), path.end(), kv.first) == path.end() && kv.first != current)
                        {
                            keys_to_delete.push_back(kv.first);
                        }
                    }
                    for (uint32_t k : keys_to_delete)
                        selection_map.erase(k);

                    selection_map.erase(current);

                    auto it = std::remove(to_process_enode.begin(), to_process_enode.end(), current);
                    if (it != to_process_enode.end())
                        to_process_enode.erase(it, to_process_enode.end());

                    if (enodes.size() > sel + 2)
                    {
                        to_process_enode.push_back(current);
                    }

                    to_process.clear();
                    for (uint32_t eclass : path)
                    {
                        uint32_t n_id = egraph.getEClass(eclass).enodes[selection_map[eclass]];
                        const ENode &n = egraph.getENodes()[n_id];
                        std::vector<uint32_t> new_to_process;
                        new_to_process.reserve(n.children.size());
                        for (uint32_t child : n.children)
                        {
                            uint32_t childEClass = egraph.find(child);
                            if (selection_map.find(childEClass) == selection_map.end())
                            {
                                new_to_process.push_back(childEClass);
                            }
                        }
                        to_process.insert(to_process.begin(), new_to_process.begin(), new_to_process.end());
                    }
                    to_process.insert(to_process.begin(), current);
                    break;
                }
                else
                {
                    selection_map.erase(current);
                    auto it = std::remove(to_process_enode.begin(), to_process_enode.end(), current);
                    if (it != to_process_enode.end())
                        to_process_enode.erase(it, to_process_enode.end());
                }
            }
        }

        if (best_cost == INF)
        {
            Error::throw_err("[Planner.extractBest] no valid extraction found under given constraints. try running bench");
        }

        ExtractionResult result;
        result.totalCost = best_cost;

        for (auto const &kv : best_selection_map)
        {
            ExtractChoice c;
            c.enodeId = egraph.getEClass(kv.first).enodes[kv.second];
            c.cost = enodeInfos[c.enodeId].cost;
            c.valid = true;
            result.choiceByEClass[kv.first] = c;
        }

        return result;
    }

    CompiledGraph buildCompiledGraph(
        uint32_t rootId,
        const Graph &graph,
        EGraph &egraph,
        const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
        const ExtractionResult &extraction,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        const std::unordered_map<uint32_t, uint32_t> &eclassToLogical)
    {
        CompiledGraph compiled;

        std::vector<uint32_t> topo;
        std::unordered_set<uint32_t> visited_classes;
        std::function<void(uint32_t)> visit = [&](uint32_t eclassId)
        {
            eclassId = egraph.find(eclassId);
            if (visited_classes.count(eclassId))
                return;
            visited_classes.insert(eclassId);

            auto choiceIt = extraction.choiceByEClass.find(eclassId);
            if (choiceIt == extraction.choiceByEClass.end() || !choiceIt->second.valid)
                return;

            const ENode &enode = egraph.getENodes()[choiceIt->second.enodeId];
            for (uint32_t child : enode.children)
                visit(child);
            topo.push_back(eclassId);
        };

        uint32_t rootEClassId = egraph.find(nodeToEClass.at(rootId));
        visit(rootEClassId);

        std::unordered_map<uint32_t, uint32_t> eclassToPhys;
        for (uint32_t eclassId : topo)
        {
            eclassToPhys[eclassId] = GlobalNextPhysId++;
        }

        std::unordered_map<uint32_t, uint32_t> lastPhysIdForLogical;
        for (uint32_t eclassId : topo)
        {
            uint32_t logicalId = eclassToLogical.count(eclassId) ? eclassToLogical.at(eclassId) : UINT32_MAX;
            if (logicalId != UINT32_MAX)
            {
                lastPhysIdForLogical[logicalId] = eclassToPhys[eclassId];
            }
        }

        for (uint32_t eclassId : topo)
        {
            const ExtractChoice &choice = extraction.choiceByEClass.at(eclassId);
            const ENode &enode = egraph.getENodes()[choice.enodeId];
            uint32_t logicalId = eclassToLogical.count(eclassId) ? eclassToLogical.at(eclassId) : UINT32_MAX;

            uint32_t physId = eclassToPhys[eclassId];

            TensorNode tNode;
            tNode.id = physId;
            tNode.opType = enode.opType;
            tNode.opName = enode.opName;
            tNode.dtype = enode.dtype;
            tNode.setShape(enode.shape);
            tNode.strides = enode.strides;
            tNode.viewOffset = enode.viewOffset;
            tNode.backend = enode.backend;
            tNode.parentIds.reserve(enode.children.size());
            for (uint32_t c : enode.children)
                tNode.parentIds.push_back(eclassToPhys[egraph.find(c)]);

            OpInstruction inst;
            inst.nodeId = physId;
            inst.logicalNodeId = logicalId;
            inst.inputNodeIds = tNode.parentIds;
            inst.backend = enode.backend;
            inst.fullKernelId = enode.kernelUid;

            if (enode.kernelUid != 0)
            {
                const KernelEntry &kEntry = KernelRegistry::get().getKernel(enode.kernelUid);
                inst.inplaceInputIndex = kEntry.inplace ? 0 : -1;
                inst.viewInputIndex = kEntry.isView ? 0 : -1;
            }

            tNode.storageType = StorageType::TRANSIENT;
            if (logicalId != UINT32_MAX)
            {
                if (graph.hasNode(logicalId))
                {
                    tNode.storageType = graph.getNode(logicalId).storageType;
                }
                if (cachedNodes.count(logicalId) && (physId == lastPhysIdForLogical[logicalId] || tNode.opType == OpType::INPUT || tNode.opType == OpType::CACHE))
                {
                    tNode.storageType = StorageType::PINNED;
                }
            }

            if (egraph.constantStaging.count(eclassId))
            {
                tNode.storageType = StorageType::PERSISTENT;
                compiled.constantStaging[physId] = egraph.constantStaging.at(eclassId);
            }

            inst.outputStorageType = tNode.storageType;
            if (enode.opType != OpType::INPUT && enode.opType != OpType::CACHE)
            {
                compiled.instructions.push_back(inst);
            }

            compiled.nodesMap[physId] = tNode;
            compiled.nodeCosts[physId] = choice.cost;
            compiled.physicalToLogicalNodeMap[physId] = logicalId;
        }

        std::unordered_map<uint32_t, uint32_t> compiledRefCounts;
        for (const auto &inst : compiled.instructions)
        {
            for (uint32_t pid : inst.inputNodeIds)
            {
                compiledRefCounts[pid]++;
            }
        }
        uint32_t rootPhysId = eclassToPhys[rootEClassId];
        compiledRefCounts[rootPhysId] = std::max<uint32_t>(1, compiledRefCounts[rootPhysId]);
        compiled.refCounts = compiledRefCounts;

        for (const auto &pair : compiled.nodesMap)
        {
            const TensorNode &node = pair.second;
            if ((node.opType == OpType::INPUT || node.opType == OpType::CACHE) && node.storageType == StorageType::TRANSIENT)
            {
                uint32_t logicalId = compiled.getLogicalId(node.id);
                if (logicalId == UINT32_MAX)
                {
                    Error::throw_err("[buildCompiledGraph] Orphan cache INPUT/CACHE node " + std::to_string(node.id) + " has no logicalId mapping and is TRANSIENT. This will crash at runtime.");
                }
            }
        }

        return compiled;
    }

    struct BaseEGraphState
    {
        EGraph egraph;
        std::unordered_map<uint32_t, uint32_t> nodeToEClass;
        std::unordered_map<uint32_t, uint32_t> eclassToLogical;
    };

    BaseEGraphState baseState;
    bool baseStateInitialized = false;

    void initBaseEGraph(uint32_t rootId, const Graph &graph, bool doSaturate, Repo *repo = nullptr)
    {
        if (baseStateInitialized)
            return;

        std::vector<uint32_t> topo = topologicalSort({rootId}, graph);

        Graph tempGraph = graph;
        inferShapes(topo, tempGraph);

        auto refCounts = computeRefCounts(topo, rootId, tempGraph);
        baseState.nodeToEClass.reserve(tempGraph.nodes.size());

        for (uint32_t nodeId : topo)
        {
            TensorNode &node = tempGraph.getNode(nodeId);
            uint32_t eclassId = baseState.egraph.addEClass(node.getShape(), node.strides, node.viewOffset, node.dtype, node.backend);
            baseState.nodeToEClass[nodeId] = eclassId;
            if (tempGraph.constantStaging.count(nodeId))
            {
                baseState.egraph.constantStaging[eclassId] = tempGraph.constantStaging.at(nodeId);
            }
        }

        for (uint32_t nodeId : topo)
        {
            const TensorNode &node = tempGraph.getNode(nodeId);
            uint32_t eclassId = baseState.nodeToEClass[nodeId];

            if (node.opType == OpType::INPUT)
            {
                ENode enode;
                enode.kernelUid = 0;
                enode.opType = node.opType;
                enode.opName = node.opName;
                for (uint32_t pid : node.parentIds)
                    enode.children.push_back(baseState.nodeToEClass[pid]);
                enode.shape = node.getShape();
                enode.strides = node.strides;
                enode.viewOffset = node.viewOffset;
                enode.dtype = node.dtype;
                enode.backend = node.backend;
                enode.leafId = node.id;
                baseState.egraph.addENode(eclassId, enode);
                continue;
            }

            std::vector<TensorNode> inputs;
            for (uint32_t pid : node.parentIds)
                inputs.push_back(tempGraph.getNode(pid));

            std::vector<uint64_t> refs = KernelRegistry::get().findMatchingKernels(
                node.opType, node.opName, node.backend, inputs, node, true);

            if (refs.size() == 0)
            {
                Error::throw_err("[Planner.initBaseEGraph] couldn't find any kernels to init EClass " + std::to_string(eclassId) + " " + toString(baseState.egraph.getEClass(eclassId)) + "\nNode " + toString(node, tempGraph));
            }
            for (uint64_t uid : refs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                ENode enode;
                enode.kernelUid = uid;
                enode.opType = node.opType;
                enode.opName = node.opName;
                for (uint32_t pid : node.parentIds)
                    enode.children.push_back(baseState.nodeToEClass[pid]);
                enode.shape = node.getShape();
                enode.dtype = node.dtype;
                enode.backend = node.backend;

                if (kernel.isView)
                {
                    enode.strides = node.strides;
                    enode.viewOffset = node.viewOffset;
                }
                else
                {
                    enode.strides = calcContiguousStrides(node.getShape());
                    enode.viewOffset = 0;
                }

                baseState.egraph.addENode(eclassId, enode);
            }
        }

        for (const auto &kv : baseState.nodeToEClass)
        {
            uint32_t physId = kv.first;
            uint32_t ecl = baseState.egraph.find(kv.second);
            baseState.eclassToLogical[ecl] = physId;
        }

        baseStateInitialized = true;
    }

    bool injectPartialPath(
        EGraph &egraph,
        const Graph &graph,
        uint32_t logicalId,
        const std::vector<Region> &regions,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
        std::unordered_map<uint32_t, uint32_t> &eclassToLogical,
        bool strictCache = false)
    {
        bool injected = false;
        uint32_t E_L = egraph.find(nodeToEClass.at(logicalId));
        const TensorNode &sourceNode = graph.getNode(logicalId);

        if (sourceNode.opType == OpType::INPUT)
        {
            return injected;
        }

        bool isFullRegion = false;
        if (regions.size() == 1)
        {
            const Region &reg = regions[0];
            const auto &shape = sourceNode.getShape();
            if (reg.region.size() == shape.size())
            {
                isFullRegion = true;
                for (size_t d = 0; d < shape.size(); ++d)
                {
                    if (reg.region[d].start != 0 || reg.region[d].stop != shape[d])
                    {
                        isFullRegion = false;
                        break;
                    }
                }
            }
        }

        if (isFullRegion)
        {
            return injected;
        }

        Backend targetBackend = sourceNode.backend;
        auto it = cachedNodes.find(logicalId);
        if (it != cachedNodes.end())
        {
            targetBackend = it->second;
        }
        else if (strictCache)
        {
            return injected;
        }

        const EClass lClass = egraph.getEClass(E_L);

        uint32_t E_Cache = egraph.addEClass(lClass.shape, lClass.strides, lClass.viewOffset, lClass.dtype, targetBackend);
        ENode cacheNode;
        cacheNode.opType = OpType::CACHE;
        cacheNode.dtype = lClass.dtype;
        cacheNode.shape = lClass.shape;
        cacheNode.strides = lClass.strides;
        cacheNode.viewOffset = lClass.viewOffset;
        cacheNode.backend = targetBackend;
        cacheNode.leafId = logicalId | 0x80000000;
        egraph.addENode(E_Cache, cacheNode);

        eclassToLogical[E_Cache] = logicalId;
        uint32_t current_E = E_Cache;

        auto addConst = [&](const std::vector<int32_t> &vals)
        {
            return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, Backend::CPU, vals);
        };

        for (size_t r = 0; r < regions.size(); ++r)
        {
            const Region &recomputeRegion = regions[r];

            std::vector<uint32_t> partialShape;
            for (const Dim &d : recomputeRegion.region)
                partialShape.push_back(d.stop - d.start);

            ShapePropagator prop;
            std::vector<std::vector<Region>> dirtyInputRegions = prop.backward(sourceNode, graph, {recomputeRegion});

            std::vector<int32_t> starts, ends, steps;
            for (const Dim &d : recomputeRegion.region)
            {
                starts.push_back(d.start);
                ends.push_back(d.stop);
                steps.push_back(1);
            }

            uint32_t startsId = addConst(starts);
            uint32_t endsId = addConst(ends);
            uint32_t stepsId = addConst(steps);

            uint32_t slicedEClass = UINT32_MAX;

            if (sourceNode.opType == OpType::INPUT)
            {
                const EClass &lClass = egraph.getEClass(E_L);
                std::vector<uint64_t> sliceStrides = lClass.strides;
                uint64_t sliceViewOffset = lClass.viewOffset;

                for (size_t d = 0; d < starts.size(); ++d)
                {
                    int32_t start = starts[d];
                    if (start < 0)
                        start += lClass.shape[d];
                    sliceViewOffset += start * sliceStrides[d];
                    sliceStrides[d] *= steps[d];
                }

                slicedEClass = egraph.addEClass(partialShape, sliceStrides, sliceViewOffset, lClass.dtype, lClass.backend);
                ENode sliceNode;
                sliceNode.opType = OpType::SLICE;
                sliceNode.children = {E_L, startsId, endsId, stepsId};
                sliceNode.shape = partialShape;
                sliceNode.strides = sliceStrides;
                sliceNode.viewOffset = sliceViewOffset;
                sliceNode.dtype = lClass.dtype;
                sliceNode.backend = lClass.backend;

                TensorNode dOut;
                dOut.setShape(partialShape);
                dOut.dtype = sliceNode.dtype;
                dOut.backend = sliceNode.backend;
                std::vector<TensorNode> dIns(4);
                dIns[0].setShape(lClass.shape);
                dIns[0].dtype = sliceNode.dtype;
                dIns[0].backend = sliceNode.backend;
                dIns[1].setShape({(uint32_t)starts.size()});
                dIns[1].dtype = DType::INT32;
                dIns[1].backend = Backend::CPU;
                dIns[2].setShape({(uint32_t)ends.size()});
                dIns[2].dtype = DType::INT32;
                dIns[2].backend = Backend::CPU;
                dIns[3].setShape({(uint32_t)steps.size()});
                dIns[3].dtype = DType::INT32;
                dIns[3].backend = Backend::CPU;

                auto sliceRefs = KernelRegistry::get().findMatchingKernels(OpType::SLICE, "", sliceNode.backend, dIns, dOut, true);
                for (uint64_t uid : sliceRefs)
                {
                    const auto &kernel = KernelRegistry::get().getKernel(uid);
                    ENode sn = sliceNode;
                    sn.kernelUid = uid;
                    if (kernel.isView)
                    {
                        sn.strides = sliceStrides;
                        sn.viewOffset = sliceViewOffset;
                    }
                    else
                    {
                        sn.strides = calcContiguousStrides(partialShape);
                        sn.viewOffset = 0;
                    }
                    egraph.addENode(slicedEClass, sn);
                }
            }
            else
            {
                std::vector<uint32_t> slicedInputs;
                std::vector<TensorNode> dummyInputNodes;

                for (size_t p_idx = 0; p_idx < sourceNode.parentIds.size(); ++p_idx)
                {
                    uint32_t parentLogicalId = sourceNode.parentIds[p_idx];
                    uint32_t E_parent = egraph.find(nodeToEClass.at(parentLogicalId));
                    const EClass &pClass = egraph.getEClass(E_parent);

                    std::vector<Region> inputSliceRegions = dirtyInputRegions[p_idx];
                    if (inputSliceRegions.size() != 1)
                    {
                        Error::throw_err("[Planner.injectPartialPath] expected exactly 1 input slice region for parent " + std::to_string(p_idx) + " but got " + std::to_string(inputSliceRegions.size()));
                    }
                    Region inputSliceRegion = inputSliceRegions[0];

                    std::vector<uint32_t> pPartialShape;
                    for (const Dim &d : inputSliceRegion.region)
                        pPartialShape.push_back(d.stop - d.start);

                    std::vector<int32_t> pStarts, pEnds, pSteps;
                    for (const Dim &d : inputSliceRegion.region)
                    {
                        pStarts.push_back(d.start);
                        pEnds.push_back(d.stop);
                        pSteps.push_back(1);
                    }

                    uint32_t pStartsId = addConst(pStarts);
                    uint32_t pEndsId = addConst(pEnds);
                    uint32_t pStepsId = addConst(pSteps);

                    std::vector<uint64_t> pSliceStrides = pClass.strides;
                    uint64_t pSliceViewOffset = pClass.viewOffset;

                    for (size_t d = 0; d < pStarts.size(); ++d)
                    {
                        int32_t start = pStarts[d];
                        if (start < 0)
                            start += pClass.shape[d];
                        pSliceViewOffset += start * pSliceStrides[d];
                        pSliceStrides[d] *= pSteps[d];
                    }

                    uint32_t pSliceEClass = egraph.addEClass(pPartialShape, pSliceStrides, pSliceViewOffset, pClass.dtype, pClass.backend);
                    ENode pSliceNode;
                    pSliceNode.opType = OpType::SLICE;
                    pSliceNode.children = {E_parent, pStartsId, pEndsId, pStepsId};
                    pSliceNode.shape = pPartialShape;
                    pSliceNode.strides = pSliceStrides;
                    pSliceNode.viewOffset = pSliceViewOffset;
                    pSliceNode.dtype = pClass.dtype;
                    pSliceNode.backend = pClass.backend;

                    TensorNode pOut;
                    pOut.setShape(pPartialShape);
                    pOut.dtype = pSliceNode.dtype;
                    pOut.backend = pSliceNode.backend;
                    std::vector<TensorNode> pIns(4);
                    pIns[0].setShape(pClass.shape);
                    pIns[0].dtype = pSliceNode.dtype;
                    pIns[0].backend = pSliceNode.backend;
                    pIns[1].setShape({(uint32_t)pStarts.size()});
                    pIns[1].dtype = DType::INT32;
                    pIns[1].backend = Backend::CPU;
                    pIns[2].setShape({(uint32_t)pEnds.size()});
                    pIns[2].dtype = DType::INT32;
                    pIns[2].backend = Backend::CPU;
                    pIns[3].setShape({(uint32_t)pSteps.size()});
                    pIns[3].dtype = DType::INT32;
                    pIns[3].backend = Backend::CPU;

                    auto pSliceRefs = KernelRegistry::get().findMatchingKernels(OpType::SLICE, "", pSliceNode.backend, pIns, pOut, true);
                    for (uint64_t uid : pSliceRefs)
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(uid);
                        ENode sn = pSliceNode;
                        sn.kernelUid = uid;
                        if (kernel.isView)
                        {
                            sn.strides = pSliceStrides;
                            sn.viewOffset = pSliceViewOffset;
                        }
                        else
                        {
                            sn.strides = calcContiguousStrides(pPartialShape);
                            sn.viewOffset = 0;
                        }
                        egraph.addENode(pSliceEClass, sn);
                    }

                    uint32_t pContigEClass = egraph.addEClass(pPartialShape, calcContiguousStrides(pPartialShape), 0, pSliceNode.dtype, pSliceNode.backend);
                    ENode pContigNode;
                    pContigNode.opType = OpType::CONTIGUOUS;
                    pContigNode.children = {pSliceEClass};
                    pContigNode.shape = pPartialShape;
                    pContigNode.strides = calcContiguousStrides(pPartialShape);
                    pContigNode.dtype = pSliceNode.dtype;
                    pContigNode.backend = pSliceNode.backend;

                    TensorNode cOut;
                    cOut.setShape(pPartialShape);
                    cOut.dtype = pSliceNode.dtype;
                    cOut.backend = pSliceNode.backend;
                    cOut.strides = pContigNode.strides;
                    TensorNode cIn;
                    cIn.setShape(pPartialShape);
                    cIn.dtype = pSliceNode.dtype;
                    cIn.backend = pSliceNode.backend;
                    cIn.strides = pSliceStrides;

                    auto contigRefs = KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", pContigNode.backend, {cIn}, cOut, true);
                    for (uint64_t uid : contigRefs)
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(uid);
                        ENode cn = pContigNode;
                        cn.kernelUid = uid;
                        if (kernel.isView)
                        {
                            cn.strides = pSliceStrides;
                            cn.viewOffset = pSliceViewOffset;
                        }
                        else
                        {
                            cn.strides = calcContiguousStrides(pPartialShape);
                            cn.viewOffset = 0;
                        }
                        egraph.addENode(pContigEClass, cn);
                    }

                    slicedInputs.push_back(pContigEClass);

                    TensorNode dummyIn;
                    dummyIn.opType = OpType::INPUT;
                    dummyIn.setShape(pPartialShape);
                    dummyIn.dtype = pSliceNode.dtype;
                    dummyIn.backend = pSliceNode.backend;
                    dummyIn.strides = pContigNode.strides;
                    dummyIn.viewOffset = 0;
                    dummyInputNodes.push_back(dummyIn);
                }

                ENode opSlicedNode;
                opSlicedNode.opType = sourceNode.opType;
                opSlicedNode.opName = sourceNode.opName;
                opSlicedNode.children = slicedInputs;
                opSlicedNode.shape = partialShape;
                opSlicedNode.strides = calcContiguousStrides(partialShape);
                opSlicedNode.viewOffset = 0;
                opSlicedNode.dtype = sourceNode.dtype;
                opSlicedNode.backend = sourceNode.backend;

                TensorNode dummyOut;
                dummyOut.opType = sourceNode.opType;
                dummyOut.opName = sourceNode.opName;
                dummyOut.setShape(partialShape);
                dummyOut.dtype = sourceNode.dtype;
                dummyOut.backend = sourceNode.backend;
                dummyOut.strides = opSlicedNode.strides;
                dummyOut.viewOffset = 0;

                auto opRefs = KernelRegistry::get().findMatchingKernels(sourceNode.opType, sourceNode.opName, sourceNode.backend, dummyInputNodes, dummyOut, true);
                if (opRefs.size() == 0)
                {
                    Error::throw_err("[Planner.injectPartialPath] couldn't find any slice kernels");
                }
                slicedEClass = egraph.addEClass(partialShape, calcContiguousStrides(partialShape), 0, sourceNode.dtype, sourceNode.backend);
                for (uint64_t uid : opRefs)
                {
                    ENode sn = opSlicedNode;
                    sn.kernelUid = uid;
                    egraph.addENode(slicedEClass, sn);
                }
            }

            uint32_t contigEClass = egraph.addEClass(partialShape, calcContiguousStrides(partialShape), 0, sourceNode.dtype, targetBackend);
            ENode contigNode;
            contigNode.opType = OpType::CONTIGUOUS;
            contigNode.children = {slicedEClass};
            contigNode.shape = partialShape;
            contigNode.strides = calcContiguousStrides(partialShape);
            contigNode.dtype = sourceNode.dtype;
            contigNode.backend = targetBackend;

            TensorNode cOut;
            cOut.setShape(partialShape);
            cOut.dtype = sourceNode.dtype;
            cOut.backend = targetBackend;
            cOut.strides = contigNode.strides;
            TensorNode cIn;
            cIn.setShape(partialShape);
            cIn.dtype = sourceNode.dtype;
            cIn.backend = sourceNode.backend;
            cIn.strides = calcContiguousStrides(partialShape);

            auto contigRefs = KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", contigNode.backend, {cIn}, cOut, true);
            for (uint64_t uid : contigRefs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                ENode cn = contigNode;
                cn.kernelUid = uid;
                if (kernel.isView)
                {
                    cn.strides = cIn.strides;
                    cn.viewOffset = 0;
                }
                else
                {
                    cn.strides = calcContiguousStrides(partialShape);
                    cn.viewOffset = 0;
                }
                egraph.addENode(contigEClass, cn);
            }

            uint32_t scatterEClass = egraph.addEClass(lClass.shape, lClass.strides, lClass.viewOffset, lClass.dtype, targetBackend);
            ENode scatterNode;
            scatterNode.opType = OpType::SCATTER;
            scatterNode.children = {current_E, contigEClass, startsId, endsId, stepsId};
            scatterNode.shape = lClass.shape;
            scatterNode.strides = lClass.strides;
            scatterNode.viewOffset = lClass.viewOffset;
            scatterNode.dtype = lClass.dtype;
            scatterNode.backend = targetBackend;

            TensorNode sOut;
            sOut.setShape(scatterNode.shape);
            sOut.dtype = scatterNode.dtype;
            sOut.backend = scatterNode.backend;
            std::vector<TensorNode> sIns(5);
            sIns[0].setShape(egraph.getEClass(current_E).shape);
            sIns[0].dtype = scatterNode.dtype;
            sIns[0].backend = scatterNode.backend;
            sIns[1].setShape(partialShape);
            sIns[1].dtype = scatterNode.dtype;
            sIns[1].backend = scatterNode.backend;
            sIns[2].setShape({(uint32_t)starts.size()});
            sIns[2].dtype = DType::INT32;
            sIns[2].backend = Backend::CPU;
            sIns[3].setShape({(uint32_t)ends.size()});
            sIns[3].dtype = DType::INT32;
            sIns[3].backend = Backend::CPU;
            sIns[4].setShape({(uint32_t)steps.size()});
            sIns[4].dtype = DType::INT32;
            sIns[4].backend = Backend::CPU;

            auto scatterRefs = KernelRegistry::get().findMatchingKernels(OpType::SCATTER, "", scatterNode.backend, sIns, sOut, true);
            for (uint64_t uid : scatterRefs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                ENode sn = scatterNode;
                sn.kernelUid = uid;
                if (kernel.isView || kernel.inplace)
                {
                    sn.strides = lClass.strides;
                    sn.viewOffset = lClass.viewOffset;
                }
                else
                {
                    sn.strides = calcContiguousStrides(scatterNode.shape);
                    sn.viewOffset = 0;
                }
                egraph.addENode(scatterEClass, sn);
            }

            current_E = scatterEClass;
        }

        egraph.merge(E_L, current_E);
        eclassToLogical[egraph.find(E_L)] = logicalId;
        return true;
    }

    bool injectInputPartialPaths(
        EGraph &egraph,
        const Graph &graph,
        const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
        std::unordered_map<uint32_t, uint32_t> &eclassToLogical)
    {
        bool injected = false;
        for (const auto &kv : dirtyOutputRegions)
        {
            uint32_t nodeId = kv.first;
            if (!graph.hasNode(nodeId))
                continue;

            const TensorNode &node = graph.getNode(nodeId);
            if (node.opType == OpType::INPUT && graph.constantStaging.count(nodeId) == 0)
            {
                if (!kv.second.empty())
                {
                    injected = injected || injectPartialPath(egraph, graph, nodeId, kv.second, cachedNodes, nodeToEClass, eclassToLogical);
                }
            }
        }
        if (injected)
        {
            egraph.rebuild();
        }
        return injected;
    }

    bool injectOutputPartialPaths(
        EGraph &egraph,
        const Graph &graph,
        uint32_t rootId,
        const std::vector<Region> &outputNeeded,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
        std::unordered_map<uint32_t, uint32_t> &eclassToLogical)
    {
        bool injected = false;
        if (!outputNeeded.empty())
        {
            injected = injectPartialPath(egraph, graph, rootId, outputNeeded, cachedNodes, nodeToEClass, eclassToLogical);
        }
        if (injected)
        {
            egraph.rebuild();
        }
        return injected;
    }

public:
    Planner(CostModel &costModel, std::unordered_map<Backend, uint64_t> maxMemoryByBackend = {})
        : costModel(costModel), maxMemoryByBackend(std::move(maxMemoryByBackend)) {}

    CompiledGraph plan(
        uint32_t rootId,
        const Graph &graph,
        const Bucket &bucket,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        bool doSaturate = true,
        bool strictCache = false,
        Repo *repo = nullptr)
    {
        // Early static baseline validation
        auto baseline = computeStaticBaseline(graph, cachedNodes);
        validateStaticBaseline(baseline);

        initBaseEGraph(rootId, graph, doSaturate, repo);

        EGraph egraph = baseState.egraph;
        auto eclassToLogical = baseState.eclassToLogical;

        std::unordered_map<uint32_t, uint32_t> canonToLogical;
        canonToLogical.reserve(eclassToLogical.size());
        for (const auto &kv : eclassToLogical)
        {
            canonToLogical[egraph.find(kv.first)] = kv.second;
        }
        eclassToLogical = canonToLogical;

        std::unordered_map<uint32_t, bool> logicalDirty;
        std::vector<uint32_t> topo = topologicalSort({rootId}, graph);
        for (uint32_t nodeId : topo)
        {
            if (bucket.inputDirtyRegions.count(nodeId) && !bucket.inputDirtyRegions.at(nodeId).empty())
            {
                logicalDirty[nodeId] = true;
            }
            else if (graph.getNode(nodeId).opType == OpType::INPUT)
            {
                logicalDirty[nodeId] = false;
            }
            else
            {
                bool isDirty = false;
                for (uint32_t pid : graph.getNode(nodeId).parentIds)
                {
                    if (logicalDirty[pid])
                    {
                        isDirty = true;
                        break;
                    }
                }
                logicalDirty[nodeId] = isDirty;
            }
        }

        for (const auto &cls : egraph.getClasses())
        {
            uint32_t canonId = egraph.find(cls.id);
            if (canonId != cls.id)
                continue;
            if (strictCache)
            {
                if (eclassToLogical.count(canonId) == 0)
                    continue;
                if (cachedNodes.count(eclassToLogical.at(canonId)) == 0)
                    continue;
            }
            for (int i = 0; i < cls.enodes.size(); i++)
            {
                if (egraph.getENodes()[cls.enodes[i]].opType == OpType::CACHE)
                {
                    continue;
                }
            }

            uint32_t logicalId = UINT32_MAX;
            auto it = eclassToLogical.find(canonId);
            if (it != eclassToLogical.end())
            {
                logicalId = it->second;
            }

            if (logicalId != UINT32_MAX && !logicalDirty[logicalId])
            {
                ENode cacheNode;
                cacheNode.kernelUid = 0;
                cacheNode.opType = OpType::CACHE;
                cacheNode.shape = cls.shape;
                cacheNode.strides = cls.strides;
                cacheNode.viewOffset = cls.viewOffset;
                cacheNode.dtype = cls.dtype;
                cacheNode.backend = cls.backend;
                cacheNode.leafId = logicalId | 0x40000000;
                egraph.addENode(canonId, cacheNode);
            }
        }

        std::unordered_set<uint32_t> protectedEClasses;
        for (const auto &kv : cachedNodes)
        {
            uint32_t logicalId = kv.first;
            protectedEClasses.insert(egraph.find(baseState.nodeToEClass.at(logicalId)));
        }

        bool dirtyInjected = injectInputPartialPaths(egraph, graph, bucket.inputDirtyRegions, cachedNodes, baseState.nodeToEClass, eclassToLogical);

        bool neededInjected = injectOutputPartialPaths(egraph, graph, rootId, bucket.outputNeededRegion, cachedNodes, baseState.nodeToEClass, eclassToLogical);

        if (doSaturate)
        {
            saturate(egraph, protectedEClasses, eclassToLogical, true, false, repo);
        }

        bool injected = dirtyInjected || neededInjected;
        std::cout << "Injected: " << injected << std::endl;

#ifdef DEBUG
        auto rootIt = baseState.nodeToEClass.find(rootId);
        if (rootIt == baseState.nodeToEClass.end())
        {
            Error::throw_err("[Planner.plan] Root node missing from baseState.nodeToEClass.");
        }
        uint32_t rootEClassId = egraph.find(rootIt->second);
        dumpEGraphBinary(egraph, rootEClassId);
#endif

        std::unordered_map<uint32_t, uint32_t> updatedEClassToLogical;
        for (const auto &kv : eclassToLogical)
        {
            updatedEClassToLogical[egraph.find(kv.first)] = kv.second;
        }
        eclassToLogical = std::move(updatedEClassToLogical);

        std::cout << "[Planner.plan] initializing immutable classes" << std::endl;
        std::unordered_set<uint32_t> immutable_eclasses;
        for (const auto &kv : eclassToLogical)
        {
            uint32_t logicalId = kv.second;
            uint32_t ecl = egraph.find(kv.first);
            const TensorNode &node = graph.getNode(logicalId);
            if (node.storageType != StorageType::TRANSIENT || cachedNodes.count(logicalId))
            {
                immutable_eclasses.insert(ecl);
            }
        }

        bool changed = true;
        while (changed)
        {
            changed = false;
            for (const auto &cls : egraph.getClasses())
            {
                uint32_t eclassId = egraph.find(cls.id);
                if (immutable_eclasses.count(eclassId))
                    continue;

                for (uint32_t enodeId : cls.enodes)
                {
                    const ENode &enode = egraph.getENodes()[enodeId];
                    if (enode.kernelUid != 0)
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(enode.kernelUid);
                        if (kernel.isView && !enode.children.empty())
                        {
                            uint32_t parentEclass = egraph.find(enode.children[0]);
                            if (immutable_eclasses.count(parentEclass))
                            {
                                immutable_eclasses.insert(eclassId);
                                changed = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        auto extraction = extractBest(rootId, graph, egraph, baseState.nodeToEClass, maxMemoryByBackend, cachedNodes, eclassToLogical, immutable_eclasses, true, strictCache);
        return buildCompiledGraph(
            rootId, graph, egraph, baseState.nodeToEClass, extraction, cachedNodes, eclassToLogical);
    }
};