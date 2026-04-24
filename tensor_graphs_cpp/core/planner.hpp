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

inline bool updateRegionListIfChanged(std::vector<Region> &dst, const std::vector<Region> &src)
{
    std::vector<Region> merged = mergeRegions(dst, src);
    if (encodeRegionList(merged) == encodeRegionList(dst))
        return false;
    dst = std::move(merged);
    return true;
}

struct PlanningRegionState
{
    bool initialized = false;
    std::unordered_map<uint32_t, std::vector<Region>> recompute;
    std::unordered_map<uint32_t, std::vector<Region>> recomputeCached;
    std::unordered_map<uint32_t, std::vector<Region>> needed;

    const std::vector<Region> *getRecompute(
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        uint32_t nodeId) const
    {
        if (cachedNodes.count(nodeId) != 0)
        {
            auto it = recomputeCached.find(nodeId);
            if (it != recomputeCached.end())
            {
                return &it->second;
            }
        }

        auto it = recompute.find(nodeId);
        if (it != recompute.end())
        {
            return &it->second;
        }

        return nullptr;
    }
};

// Worklist backward pass to compute needed regions from the root.
static void updateNeeded(
    uint32_t rootId,
    const Graph &graph,
    ShapePropagator &prop,
    std::unordered_map<uint32_t, std::vector<Region>> &needed)
{
    if (!graph.hasNode(rootId))
        return;

    std::vector<uint32_t> worklist = {rootId};
    std::unordered_set<uint32_t> queued = {rootId};

    while (!worklist.empty())
    {
        uint32_t nodeId = worklist.back();
        worklist.pop_back();
        queued.erase(nodeId);

        if (!graph.hasNode(nodeId))
            continue;

        const TensorNode &node = graph.getNode(nodeId);
        if (node.opType == OpType::INPUT)
            continue;

        auto neededIt = needed.find(nodeId);
        if (neededIt == needed.end() || neededIt->second.empty())
            continue;

        // These operations rely on global coordinates or full context to compute correct values, so they cannot be partially evaluated.
        bool forceFull = (node.opType == OpType::ARANGE ||
                          node.opType == OpType::TRIU ||
                          node.opType == OpType::IM2COL ||
                          node.opType == OpType::FILL);

        std::vector<Region> currentNeeded = neededIt->second;
        if (forceFull)
        {
            currentNeeded = {node.fullRegion()};
            needed[nodeId] = currentNeeded;
        }

        std::vector<std::vector<Region>> parentNeeded = prop.backward(node, graph, neededIt->second);
        for (size_t i = 0; i < node.parentIds.size(); ++i)
        {
            if (i >= parentNeeded.size() || parentNeeded[i].empty())
                continue;

            uint32_t parentId = node.parentIds[i];
            if (!graph.hasNode(parentId))
                continue;

            const TensorNode &parentNode = graph.getNode(parentId);
            if (parentNode.opType == OpType::INPUT)
                continue;

            if (updateRegionListIfChanged(needed[parentId], parentNeeded[i]) && !queued.count(parentId))
            {
                worklist.push_back(parentId);
                queued.insert(parentId);
            }
        }
    }
}

static PlanningRegionState derivePlanningRegions(
    uint32_t rootId,
    const Graph &graph,
    const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
    const std::vector<Region> &outputNeeded)
{
    PlanningRegionState state;
    ShapePropagator prop;

    if (outputNeeded.empty())
    {
        Error::throw_err("[derivePlanningRegions] outputNeeded is empty");
    }
    state.needed[rootId] = mergeRegions(outputNeeded);

    updateNeeded(rootId, graph, prop, state.needed);

    for (const auto &pair : state.needed)
    {
        uint32_t nodeId = pair.first;
        const std::vector<Region> &neededRegions = pair.second;
        if (neededRegions.empty())
            continue;
        const TensorNode &node = graph.nodes.at(nodeId);
        bool forceFull = (node.opType == OpType::ARANGE ||
                          node.opType == OpType::TRIU ||
                          node.opType == OpType::IM2COL ||
                          node.opType == OpType::FILL);

        auto dirtyIt = dirtyOutputRegions.find(nodeId);
        state.recompute[nodeId] = normalizeRegions(mergeRegions(neededRegions));
        if (dirtyIt != dirtyOutputRegions.end() && !dirtyIt->second.empty() && !forceFull)
            state.recomputeCached[nodeId] = normalizeRegions(intersectRegionLists(dirtyIt->second, neededRegions));
    }

    state.initialized = true;
    return state;
}

class Planner
{
private:
    uint32_t egraph_dump_counter_ = 0;
    uint32_t nextPhysId = 0x80000000;

    void dumpEGraphBinary(const EGraph &egraph, uint32_t idx, uint32_t rootEClassId)
    {
        std::filesystem::create_directories("egraph_viewer/egraphs");
        std::string path = "egraphs_viewer/egraphs/" + std::to_string(idx) + ".bin";
        std::ofstream out(path, std::ios::binary);
        if (!out)
        {
            std::cerr << "[Planner] Failed to open " << path << " for writing." << std::endl;
            return;
        }

        const auto &classes = egraph.getClasses();
        const auto &enodes = egraph.getENodes();

        uint32_t num_classes = static_cast<uint32_t>(classes.size());
        uint32_t num_enodes = static_cast<uint32_t>(enodes.size());

        out.write(reinterpret_cast<const char *>(&num_classes), 4);
        out.write(reinterpret_cast<const char *>(&num_enodes), 4);
        out.write(reinterpret_cast<const char *>(&rootEClassId), 4);

        // Write EClasses
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

        // Write ENodes
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
        out.close();
        std::cout << "[Planner] Dumped EGraph to " << path << std::endl;
    }

    struct ENodeInfo
    {
        float cost;
        std::unordered_map<Backend, uint64_t> memSizes;
        bool inplace;
        int32_t inplace_idx;
        bool isScatter;
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

    std::vector<uint32_t> topologicalSort(const uint32_t rootId, const Graph &graph) const
    {
        std::vector<uint32_t> order;
        std::unordered_set<uint32_t> visited;
        auto visit = [&](auto &self, uint32_t node) -> void
        {
            if (visited.count(node))
                return;
            visited.insert(node);
            for (uint32_t pid : graph.getNode(node).parentIds)
            {
                self(self, pid);
            }
            order.push_back(node);
        };
        visit(visit, rootId);
        return order;
    }

    void saturate(EGraph &egraph, const std::unordered_set<uint32_t> &protectedEClasses)
    {
        std::vector<std::unique_ptr<Rule>> rules;
        rules.emplace_back(std::make_unique<FusionRule>());
        rules.emplace_back(std::make_unique<CopyToOfContiguous>());
        rules.emplace_back(std::make_unique<ContiguousOfCopyTo>());
        rules.emplace_back(std::make_unique<ContiguousElimination>());
        rules.emplace_back(std::make_unique<ConstantFolding>());
        rules.emplace_back(std::make_unique<InfinityDomination>());
        rules.emplace_back(std::make_unique<SlicePushBackward>());
        rules.emplace_back(std::make_unique<ScatterPushForward>());
        // rules.emplace_back(std::make_unique<DistributiveProperty>());

        std::vector<uint32_t> protectedVec(protectedEClasses.begin(), protectedEClasses.end());

        size_t iterations = 0;
        bool changed = true;
        uint32_t nMatches = 0;

        ProgressTimer timer(0, "", true);
        while (changed)
        {
            timer.reset();
            iterations++;
            uint32_t numENodes = egraph.getENodes().size();
            for (uint32_t eNodeIdx = 0; eNodeIdx < numENodes; eNodeIdx++)
            {
                for (const auto &rule : rules)
                {
                    if (!rule->match(egraph, eNodeIdx, protectedVec))
                        continue;

                    rule->apply(egraph, eNodeIdx, protectedVec);
                    changed = true;
                    nMatches++;
                }
            }
            egraph.rebuild();
            double elapsed = timer.elapsed();
            changed = egraph.getENodes().size() != numENodes;
            std::cout << "# New enodes: " << egraph.getENodes().size() - numENodes << ", took " << std::to_string(elapsed) << "s" << std::endl;
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

    std::unordered_map<Backend, uint64_t> computePeakMemory(
        const EGraph &egraph,
        const std::unordered_map<uint32_t, uint32_t> &selection_map,
        const std::vector<ENodeInfo> &enodeInfos,
        uint32_t root,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        const std::unordered_map<uint32_t, uint32_t> &eclassToLogical) const
    {
        auto ref = build_ref_counts(egraph, selection_map, root);
        std::unordered_map<Backend, uint64_t> live_mem;
        std::unordered_map<Backend, uint64_t> peak_mem;
        std::unordered_set<uint32_t> visited;

        std::function<void(uint32_t)> visit = [&](uint32_t eclass)
        {
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

            if (!info.inplace && !isCached)
            {
                for (const auto &kv : info.memSizes)
                {
                    live_mem[kv.first] += kv.second;
                    peak_mem[kv.first] = std::max(peak_mem[kv.first], live_mem[kv.first]);
                }
            }

            for (size_t i = 0; i < node.children.size(); ++i)
            {
                uint32_t c = node.children[i];
                ref[c]--;
                if (ref[c] == 0)
                {
                    uint32_t child_sel = selection_map.at(c);
                    uint32_t child_enode_id = egraph.getEClass(c).enodes[child_sel];
                    const ENodeInfo &child_info = enodeInfos[child_enode_id];
                    uint32_t child_logicalId = eclassToLogical.count(c) ? eclassToLogical.at(c) : UINT32_MAX;
                    bool childIsCached = (child_logicalId != UINT32_MAX && cachedNodes.count(child_logicalId));

                    if (info.inplace && (int)i == info.inplace_idx)
                    {
                        continue;
                    }

                    if (!child_info.inplace && !childIsCached)
                    {
                        for (const auto &kv : child_info.memSizes)
                        {
                            live_mem[kv.first] -= kv.second;
                        }
                    }
                }
            }
        };

        visit(root);
        return peak_mem;
    }

    ExtractionResult extractBest(const uint32_t rootId, const Graph &graph, EGraph &egraph,
                                 const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                                 const std::unordered_map<Backend, uint64_t> &maxMemoryByBackend,
                                 const std::unordered_map<uint32_t, Backend> &cachedNodes,
                                 const std::unordered_map<uint32_t, uint32_t> &eclassToLogical,
                                 const std::unordered_set<uint32_t> &immutable_eclasses,
                                 bool stopOnFirstValid = false,
                                 bool cheapInputCopy = false)
    {
        constexpr float INF = std::numeric_limits<float>::infinity();
        constexpr float EPS = 1e-6f;

        std::vector<ENodeInfo> enodeInfos(egraph.getENodes().size());
        for (size_t i = 0; i < egraph.getENodes().size(); ++i)
        {
            const ENode &enode = egraph.getENodes()[i];
            ENodeInfo info;
            info.memSizes[enode.backend] = getSizeBytes(enode.shape, enode.dtype);
            info.inplace = false;
            info.inplace_idx = -1;
            info.isScatter = false;

            if (enode.kernelUid != 0)
            {
                const auto &kernel = KernelRegistry::get().getKernel(enode.kernelUid);
                info.inplace = kernel.inplace;
                if (info.inplace && kernel.numInputs > 0)
                {
                    info.inplace_idx = 0;
                }

                if (enode.opType == OpType::SCATTER)
                {
                    info.isScatter = true;
                }
                else if (enode.opType == OpType::FUSED && kernel.numInputs == 5)
                {
                    Graph pGraph;
                    std::vector<TensorNode> dummyInputs;
                    for (size_t k = 0; k < 5; ++k)
                    {
                        pGraph.input(kernel.dummyShapes[k], kernel.dtypes[k]);
                        TensorNode inNode;
                        inNode.opType = OpType::INPUT;
                        inNode.dtype = kernel.dtypes[k];
                        inNode.setShape(kernel.dummyShapes[k]);
                        inNode.backend = enode.backend;
                        dummyInputs.push_back(inNode);
                    }

                    uint32_t pRoot = pGraph.scatter(0, 1, 2, 3, 4);

                    TensorNode dummyOut;
                    dummyOut.opType = enode.opType;
                    dummyOut.dtype = enode.dtype;
                    dummyOut.setShape(enode.shape);
                    dummyOut.backend = enode.backend;

                    auto matches = KernelRegistry::get().findMatchingKernelsByPattern(
                        pGraph, pRoot, enode.backend, dummyInputs, dummyOut, false, true, true);

                    if (std::find(matches.begin(), matches.end(), enode.kernelUid) != matches.end())
                    {
                        info.isScatter = true;
                    }
                }
            }

            if (enode.opType == OpType::INPUT)
            {
                info.cost = 0.0f;
            }
            else if (enode.kernelUid != 0)
            {
                bool isCheapCopy = false;
                if (cheapInputCopy && enode.opType == OpType::COPY_TO && enode.children.size() == 1)
                {
                    uint32_t childEClassId = egraph.find(enode.children[0]);
                    uint32_t logicalId = eclassToLogical.count(childEClassId) ? eclassToLogical.at(childEClassId) : UINT32_MAX;
                    if (logicalId != UINT32_MAX && graph.hasNode(logicalId))
                    {
                        const TensorNode &childLogicalNode = graph.getNode(logicalId);
                        if (childLogicalNode.opType == OpType::INPUT &&
                            childLogicalNode.storageType == StorageType::PERSISTENT)
                        {
                            isCheapCopy = true;
                        }
                    }
                }

                if (isCheapCopy)
                {
                    info.cost = 0.0f;
                }
                else
                {
                    std::vector<std::vector<uint32_t>> inShapes;
                    std::vector<std::vector<uint64_t>> inStrides;
                    std::vector<DType> inDTypes;
                    std::vector<std::vector<uint8_t>> inConstants;

                    inShapes.reserve(enode.children.size());
                    inStrides.reserve(enode.children.size());
                    inDTypes.reserve(enode.children.size());
                    inConstants.reserve(enode.children.size());

                    for (uint32_t childEClassId : enode.children)
                    {
                        const EClass &childCls = egraph.getEClass(egraph.find(childEClassId));
                        inShapes.push_back(childCls.shape);

                        std::vector<uint64_t> strides_cast;
                        strides_cast.reserve(childCls.strides.size());
                        for (uint64_t s : childCls.strides)
                            strides_cast.push_back(s);
                        inStrides.push_back(std::move(strides_cast));

                        inDTypes.push_back(childCls.dtype);

                        uint32_t canonChild = egraph.find(childEClassId);
                        if (egraph.constantStaging.count(canonChild))
                        {
                            inConstants.push_back(egraph.constantStaging.at(canonChild));
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
                }

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
                Error::throw_err("[Planner.extractBest] enode.kernelUid != 0, but isn't OpType::INPUT. this shouldn't happen");
            }

            enodeInfos[i] = std::move(info);
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

            if (validEnodes.empty())
            {
                std::cout << "[Planner.extractBest] Warning: EClass " << eclassId << " has NO valid enodes\n";
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
        const size_t bitWords = (numClasses + 63) >> 6;

        auto bitTest = [](const std::vector<uint64_t> &bits, uint32_t idx) -> bool
        {
            return (bits[idx >> 6] >> (idx & 63)) & 1ULL;
        };

        auto bitSet = [](std::vector<uint64_t> &bits, uint32_t idx)
        {
            bits[idx >> 6] |= (1ULL << (idx & 63));
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
        for (auto &o : opt)
        {
            o.coveredBits.assign(bitWords, 0);
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
                                uint32_t k = static_cast<uint32_t>((w << 6) + bit);
                                if (k < numClasses)
                                {
                                    candidateCost += opt[k].intrinsic;
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

        // Final one-pass cached optimistic DAG cost computation for sorting.
        // This avoids recomputing inside the std::sort comparator.
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
                            uint32_t k = static_cast<uint32_t>((w << 6) + bit);
                            if (k < numClasses)
                            {
                                total += opt[k].intrinsic;
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

        std::cout << "[Planner.extractBest] Optimistic root cost: "
                  << std::to_string(eclassMinCost[rootEClassId]) << std::endl;

        std::unordered_map<uint32_t, uint32_t> selection_map;
        std::vector<uint32_t> path;
        std::vector<uint32_t> to_process = {rootEClassId};
        std::vector<uint32_t> to_process_enode;
        std::unordered_map<uint32_t, uint32_t> ref_counts;
        std::unordered_map<uint32_t, uint32_t> need_single_ref;
        std::unordered_map<uint32_t, uint32_t> next_sel;

        float best_cost = INF;
        std::unordered_map<uint32_t, uint32_t> best_selection_map;

        int max_iters = 100;
        ProgressTimer timer(max_iters, "extracting graphs ", stopOnFirstValid);

        while (max_iters-- > 0)
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

                if (info.inplace && info.inplace_idx >= 0)
                {
                    uint32_t inplace_child = egraph.find(node.children[info.inplace_idx]);
                    if (need_single_ref.find(inplace_child) != need_single_ref.end())
                    {
                        valid = false;
                        reason = "inplace";
                    }
                    else if (immutable_eclasses.count(inplace_child) && !info.isScatter)
                    {
                        valid = false;
                        reason = "inplace_immutable";
                    }
                    else
                    {
                        need_single_ref[inplace_child] = current;
                    }
                }

                for (uint32_t child : node.children)
                {
                    uint32_t canonChild = egraph.find(child);
                    ref_counts[canonChild]++;
                    if (need_single_ref.find(canonChild) != need_single_ref.end() && ref_counts[canonChild] > 1)
                    {
                        valid = false;
                        reason = "inplace_ref";
                    }
                }

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
                std::unordered_map<Backend, uint64_t> peak = computePeakMemory(
                    egraph, selection_map, enodeInfos, rootEClassId, cachedNodes, eclassToLogical);

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

            if (valid)
            {
                if (current_cost < best_cost)
                {
                    // #ifdef DEBUG
                    //                     float selection_cost = 0.0f;
                    //                     for (auto const &kv : selection_map)
                    //                     {
                    //                         selection_cost += enodeInfos[egraph.getEClass(kv.first).enodes[kv.second]].cost;
                    //                     }
                    //                     if (std::abs(current_cost - selection_cost) > 1e-3f)
                    //                     {
                    //                         Error::throw_err("[Planner.extractBest] current_cost calculation went wrong somewhere. current_cost=" +
                    //                                          std::to_string(current_cost) + ", selection_cost=" + std::to_string(selection_cost));
                    //                     }
                    // #endif
                    best_cost = current_cost;
                    best_selection_map = selection_map;
                    if (!stopOnFirstValid)
                    {
                        std::cout << "new best cost: " << std::to_string(best_cost) << std::endl;
                    }
                }

                if (stopOnFirstValid)
                {
                    break;
                }
            }
            else
            {
#ifdef DEBUG
                // std::cout << "invalid graph (" << reason << ")" << std::endl;
#endif
            }

            if (to_process_enode.empty())
                break;

            while (!path.empty())
            {
                uint32_t current = path.back();
                path.pop_back();

                if (selection_map.find(current) == selection_map.end())
                    continue;

                uint32_t sel = selection_map[current];
                const auto &enodes = egraph.getEClass(current).enodes;
                uint32_t enode_id = enodes[sel];
                const ENode &node = egraph.getENodes()[enode_id];
                const ENodeInfo &info = enodeInfos[enode_id];

                for (uint32_t child : node.children)
                {
                    uint32_t canonChild = egraph.find(child);
                    ref_counts[canonChild]--;
                    if (ref_counts[canonChild] == 0)
                        ref_counts.erase(canonChild);
                }

                if (info.inplace && info.inplace_idx >= 0)
                {
                    uint32_t inplace_child = egraph.find(node.children[info.inplace_idx]);
                    auto it = need_single_ref.find(inplace_child);
                    if (it != need_single_ref.end() && it->second == current)
                    {
                        need_single_ref.erase(it);
                    }
                }

                if (sel + 1 < enodes.size())
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
        if (stopOnFirstValid)
        {
            return result;
        }

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
            eclassToPhys[eclassId] = nextPhysId++;
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

            // Offset physical IDs so they never collide with logical IDs from the original Graph
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
                if (cachedNodes.count(logicalId) && physId == lastPhysIdForLogical[logicalId])
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
            if (enode.opType != OpType::INPUT)
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

    void initBaseEGraph(uint32_t rootId, const Graph &graph, bool doSaturate)
    {
        if (baseStateInitialized)
            return;

        std::vector<uint32_t> topo = topologicalSort(rootId, graph);

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
                baseState.egraph.addENode(eclassId, enode);
                continue;
            }

            std::vector<TensorNode> inputs;
            for (uint32_t pid : node.parentIds)
                inputs.push_back(tempGraph.getNode(pid));

            std::vector<uint64_t> refs = KernelRegistry::get().findMatchingKernels(
                node.opType, node.opName, node.backend, inputs, node, true);

            for (uint64_t uid : refs)
            {
                ENode enode;
                enode.kernelUid = uid;
                enode.opType = node.opType;
                enode.opName = node.opName;
                for (uint32_t pid : node.parentIds)
                    enode.children.push_back(baseState.nodeToEClass[pid]);
                enode.shape = node.getShape();
                enode.strides = node.strides;
                enode.viewOffset = node.viewOffset;
                enode.dtype = node.dtype;
                enode.backend = node.backend;
                baseState.egraph.addENode(eclassId, enode);
            }
        }

        if (doSaturate)
        {
            std::unordered_set<uint32_t> emptyProtected;
            saturate(baseState.egraph, emptyProtected);
        }

        for (const auto &kv : baseState.nodeToEClass)
        {
            uint32_t physId = kv.first; // Here physId == logicalId
            uint32_t ecl = baseState.egraph.find(kv.second);
            baseState.eclassToLogical[ecl] = physId;
        }

        baseStateInitialized = true;
    }

    // Rewrite parents to point to Scatter and merge root for proper E-Graph evaluation
    void injectPartialCompute(EGraph &egraph, const Graph &graph,
                              const std::unordered_map<uint32_t, Backend> &cachedNodes,
                              const PlanningRegionState &regionState,
                              const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                              std::unordered_map<uint32_t, uint32_t> &eclassToLogical)
    {
        ShapePropagator prop;

        for (const auto &kv : cachedNodes)
        {
            uint32_t logicalId = kv.first;
            auto recomputePtr = regionState.getRecompute(cachedNodes, logicalId);
            uint32_t E_L = egraph.find(nodeToEClass.at(logicalId));

            if (!recomputePtr || recomputePtr->empty())
            {
                ENode cacheNode;
                cacheNode.opType = OpType::INPUT;
                cacheNode.dtype = graph.getNode(logicalId).dtype;
                cacheNode.shape = graph.getNode(logicalId).getShape();
                cacheNode.strides = calcContiguousStrides(cacheNode.shape);
                cacheNode.backend = kv.second;
                egraph.addENode(E_L, cacheNode);
                continue;
            }

            const std::vector<Region> &recomputeRegions = *recomputePtr;

            bool isFullRegion = false;
            if (recomputeRegions.size() == 1)
            {
                const Region &reg = recomputeRegions[0];
                const auto &shape = graph.getNode(logicalId).getShape();
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
                continue;
            }

            uint32_t E_Cache = egraph.addEClass(graph.getNode(logicalId).getShape(), calcContiguousStrides(graph.getNode(logicalId).getShape()), 0, graph.getNode(logicalId).dtype, kv.second);
            ENode cacheNode;
            cacheNode.opType = OpType::INPUT;
            cacheNode.dtype = graph.getNode(logicalId).dtype;
            cacheNode.shape = graph.getNode(logicalId).getShape();
            cacheNode.strides = calcContiguousStrides(cacheNode.shape);
            cacheNode.backend = kv.second;
            egraph.addENode(E_Cache, cacheNode);

            eclassToLogical[E_Cache] = logicalId;
            uint32_t current_E = E_Cache;

            for (const Region &recomputeRegion : recomputeRegions)
            {
                const TensorNode &sourceNode = graph.getNode(logicalId);
                std::vector<std::vector<Region>> parentRegions = prop.backward(sourceNode, graph, {recomputeRegion});

                std::vector<uint32_t> partialParents;
                for (size_t i = 0; i < sourceNode.parentIds.size(); ++i)
                {
                    uint32_t pLogicalId = sourceNode.parentIds[i];
                    uint32_t pEClass = egraph.find(nodeToEClass.at(pLogicalId));

                    bool shouldSlice = true;
                    switch (sourceNode.opType)
                    {
                    case OpType::SUM:
                    case OpType::MAX:
                    case OpType::PERMUTE:
                    case OpType::TRIU:
                    case OpType::RESHAPE:
                    case OpType::FILL:
                    case OpType::SLICE:
                        shouldSlice = (i == 0);
                        break;
                    case OpType::REPEAT:
                        shouldSlice = (i == 0);
                        break;
                    case OpType::CONCAT:
                        shouldSlice = (i + 1 < sourceNode.parentIds.size());
                        break;
                    case OpType::SCATTER:
                        shouldSlice = (i < 2);
                        break;
                    case OpType::GATHER:
                        shouldSlice = (i == 1);
                        break;
                    default:
                        shouldSlice = true;
                        break;
                    }

                    if (shouldSlice && !parentRegions[i].empty())
                    {
                        const Region &pReg = parentRegions[i].front();

                        bool isParentFull = false;
                        const auto &pShape = egraph.getEClass(pEClass).shape;
                        if (pReg.region.size() == pShape.size())
                        {
                            isParentFull = true;
                            for (size_t d = 0; d < pShape.size(); ++d)
                            {
                                if (pReg.region[d].start != 0 || pReg.region[d].stop != pShape[d])
                                {
                                    isParentFull = false;
                                    break;
                                }
                            }
                        }

                        if (isParentFull)
                        {
                            partialParents.push_back(pEClass);
                        }
                        else
                        {
                            auto addConst = [&](const std::vector<int32_t> &vals)
                            {
                                std::vector<uint8_t> bytes(vals.size() * sizeof(int32_t));
                                std::memcpy(bytes.data(), vals.data(), bytes.size());
                                std::vector<uint32_t> shape = {(uint32_t)vals.size()};
                                return getOrAddConstant(egraph, shape, DType::INT32, bytes);
                            };

                            std::vector<int32_t> starts, ends, steps;
                            for (const Dim &d : pReg.region)
                            {
                                starts.push_back(d.start);
                                ends.push_back(d.stop);
                                steps.push_back(1);
                            }

                            uint32_t startsId = addConst(starts);
                            uint32_t endsId = addConst(ends);
                            uint32_t stepsId = addConst(steps);

                            std::vector<uint32_t> sliceShape;
                            for (size_t d = 0; d < starts.size(); ++d)
                                sliceShape.push_back(ends[d] - starts[d]);

                            const EClass &pClass = egraph.getEClass(pEClass);
                            std::vector<uint64_t> sliceStrides(pClass.strides.size());
                            uint64_t sliceViewOffset = pClass.viewOffset;

                            for (size_t d = 0; d < pClass.strides.size(); ++d)
                            {
                                int32_t start = d < starts.size() ? starts[d] : 0;
                                int32_t step = d < steps.size() ? steps[d] : 1;
                                if (start < 0)
                                    start += pClass.shape[d];
                                sliceViewOffset += start * pClass.strides[d];
                                sliceStrides[d] = pClass.strides[d] * step;
                            }

                            uint32_t sliceEClass = egraph.addEClass(sliceShape, sliceStrides, sliceViewOffset, pClass.dtype, pClass.backend);
                            ENode sliceNode;
                            sliceNode.opType = OpType::SLICE;
                            sliceNode.children = {pEClass, startsId, endsId, stepsId};
                            sliceNode.shape = sliceShape;
                            sliceNode.strides = sliceStrides;
                            sliceNode.viewOffset = sliceViewOffset;
                            sliceNode.dtype = pClass.dtype;
                            sliceNode.backend = pClass.backend;

                            TensorNode dOut;
                            dOut.setShape(sliceShape);
                            dOut.dtype = sliceNode.dtype;
                            dOut.backend = sliceNode.backend;
                            std::vector<TensorNode> dIns(4);
                            dIns[0].setShape(egraph.getEClass(pEClass).shape);
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
                                ENode sn = sliceNode;
                                sn.kernelUid = uid;
                                egraph.addENode(sliceEClass, sn);
                            }

                            uint32_t contigEClass = egraph.addEClass(sliceShape, calcContiguousStrides(sliceShape), 0, sliceNode.dtype, sliceNode.backend);
                            ENode contigNode;
                            contigNode.opType = OpType::CONTIGUOUS;
                            contigNode.children = {sliceEClass};
                            contigNode.shape = sliceShape;
                            contigNode.strides = calcContiguousStrides(sliceShape);
                            contigNode.dtype = sliceNode.dtype;
                            contigNode.backend = sliceNode.backend;

                            auto contigRefs = KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", contigNode.backend, {dOut}, dOut, true);
                            for (uint64_t uid : contigRefs)
                            {
                                ENode cn = contigNode;
                                cn.kernelUid = uid;
                                egraph.addENode(contigEClass, cn);
                            }

                            partialParents.push_back(contigEClass);
                        }
                    }
                    else
                    {
                        partialParents.push_back(pEClass);
                    }
                }

                std::vector<uint32_t> partialShape;
                for (const Dim &d : recomputeRegion.region)
                    partialShape.push_back(d.stop - d.start);

                uint32_t partialEClass = egraph.addEClass(partialShape, calcContiguousStrides(partialShape), 0, sourceNode.dtype, sourceNode.backend);
                ENode partialNode;
                partialNode.opType = sourceNode.opType;
                partialNode.opName = sourceNode.opName;
                partialNode.children = partialParents;
                partialNode.shape = partialShape;
                partialNode.strides = calcContiguousStrides(partialShape);
                partialNode.dtype = sourceNode.dtype;
                partialNode.backend = sourceNode.backend;

                TensorNode dOut;
                dOut.setShape(partialShape);
                dOut.dtype = partialNode.dtype;
                dOut.backend = partialNode.backend;
                std::vector<TensorNode> dIns;
                for (size_t i = 0; i < partialParents.size(); ++i)
                {
                    TensorNode inN;
                    inN.setShape(egraph.getEClass(partialParents[i]).shape);
                    inN.dtype = egraph.getEClass(partialParents[i]).dtype;
                    inN.backend = egraph.getEClass(partialParents[i]).backend;
                    dIns.push_back(inN);
                }

                auto partialRefs = KernelRegistry::get().findMatchingKernels(partialNode.opType, partialNode.opName, partialNode.backend, dIns, dOut, true);
                for (uint64_t uid : partialRefs)
                {
                    ENode pn = partialNode;
                    pn.kernelUid = uid;
                    egraph.addENode(partialEClass, pn);
                }

                auto addConst = [&](const std::vector<int32_t> &vals)
                {
                    std::vector<uint8_t> bytes(vals.size() * sizeof(int32_t));
                    std::memcpy(bytes.data(), vals.data(), bytes.size());
                    std::vector<uint32_t> shape = {(uint32_t)vals.size()};
                    return getOrAddConstant(egraph, shape, DType::INT32, bytes);
                };

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

                uint32_t scatterEClass = egraph.addEClass(sourceNode.getShape(), calcContiguousStrides(sourceNode.getShape()), 0, sourceNode.dtype, sourceNode.backend);
                ENode scatterNode;
                scatterNode.opType = OpType::SCATTER;
                scatterNode.children = {current_E, partialEClass, startsId, endsId, stepsId};
                scatterNode.shape = sourceNode.getShape();
                scatterNode.strides = calcContiguousStrides(scatterNode.shape);
                scatterNode.dtype = sourceNode.dtype;
                scatterNode.backend = sourceNode.backend;

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
                    ENode sn = scatterNode;
                    sn.kernelUid = uid;
                    egraph.addENode(scatterEClass, sn);
                }

                current_E = scatterEClass;
            }

            // Finally, safely unify so roots can evaluate appropriately between choices
            egraph.merge(E_L, current_E);
            eclassToLogical[egraph.find(E_L)] = logicalId;
        }

        egraph.rebuild();
    }

public:
    Planner(CostModel &costModel, std::unordered_map<Backend, uint64_t> maxMemoryByBackend = {})
        : costModel(costModel), maxMemoryByBackend(std::move(maxMemoryByBackend)) {}

    CompiledGraph plan(
        uint32_t rootId,
        const Graph &graph,
        const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
        const std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &dirtyInputRegions,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        PlanningRegionState &regionState,
        const std::vector<Region> &outputNeeded,
        bool doSaturate = true,
        bool protectCachedNodes = true,
        bool cheapInputCopy = false)
    {
        initBaseEGraph(rootId, graph, doSaturate);

        if (!regionState.initialized)
        {
            regionState = derivePlanningRegions(rootId, graph, dirtyOutputRegions, outputNeeded);
        }

        EGraph egraph = baseState.egraph;
        auto eclassToLogical = baseState.eclassToLogical;

        injectPartialCompute(egraph, graph, cachedNodes, regionState, baseState.nodeToEClass, eclassToLogical);

        if (doSaturate)
        {
            std::unordered_set<uint32_t> protectedEClasses;
            if (protectCachedNodes)
            {
                for (const auto &kv : cachedNodes)
                {
                    protectedEClasses.insert(egraph.find(baseState.nodeToEClass.at(kv.first)));
                }
            }
            saturate(egraph, protectedEClasses);
        }

#ifdef DEBUG
        auto rootIt = baseState.nodeToEClass.find(rootId);
        if (rootIt == baseState.nodeToEClass.end())
        {
            Error::throw_err("[Planner.plan] Root node missing from baseState.nodeToEClass.");
        }
        uint32_t rootEClassId = egraph.find(rootIt->second);
        dumpEGraphBinary(egraph, egraph_dump_counter_++, rootEClassId);
#endif

        std::unordered_map<uint32_t, uint32_t> updatedEClassToLogical;
        for (const auto &kv : eclassToLogical)
        {
            updatedEClassToLogical[egraph.find(kv.first)] = kv.second;
        }
        eclassToLogical = std::move(updatedEClassToLogical);

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

        auto extraction = extractBest(rootId, graph, egraph, baseState.nodeToEClass, maxMemoryByBackend, cachedNodes, eclassToLogical, immutable_eclasses, false, cheapInputCopy);

        return buildCompiledGraph(
            rootId, graph, egraph, baseState.nodeToEClass, extraction, cachedNodes, eclassToLogical);
    }
};