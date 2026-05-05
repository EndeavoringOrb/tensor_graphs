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

class Planner
{
private:
    uint32_t egraph_dump_counter_ = 0;
    uint32_t nextPhysId = 0x80000000;

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

        // 3. Write Constants Section
        uint32_t num_constants = static_cast<uint32_t>(egraph.constantStaging.size());
        out.write(reinterpret_cast<const char *>(&num_constants), 4);
        for (const auto &[eclassId, data] : egraph.constantStaging)
        {
            uint32_t canonId = eclassId; // Parser will use this to map to the canonical class
            out.write(reinterpret_cast<const char *>(&canonId), 4);
            uint64_t data_size = static_cast<uint64_t>(data.size());
            out.write(reinterpret_cast<const char *>(&data_size), 8);
            out.write(reinterpret_cast<const char *>(data.data()), data_size);
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

    void saturate(EGraph &egraph, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical, bool injected, bool allowPushDownOnProtected = false)
    {
        std::vector<std::unique_ptr<Rule>> rules;
        rules.emplace_back(std::make_unique<FusionRule>());
        rules.emplace_back(std::make_unique<CopyToOfContiguous>());
        rules.emplace_back(std::make_unique<ContiguousOfCopyTo>());
        rules.emplace_back(std::make_unique<ContiguousElimination>());
        rules.emplace_back(std::make_unique<ConstantFolding>());
        if (injected)
        {
            rules.emplace_back(std::make_unique<InfinityDomination>());
            rules.emplace_back(std::make_unique<SlicePushDownElementwise>(allowPushDownOnProtected));
            rules.emplace_back(std::make_unique<SlicePushDownDot>(allowPushDownOnProtected));
            rules.emplace_back(std::make_unique<SlicePullUpDot>(allowPushDownOnProtected));
            rules.emplace_back(std::make_unique<ScatterSliceCancellation>(allowPushDownOnProtected));
            rules.emplace_back(std::make_unique<SlicePushDownContiguous>(allowPushDownOnProtected));
            rules.emplace_back(std::make_unique<SlicePushDownPermute>(allowPushDownOnProtected));
            rules.emplace_back(std::make_unique<SlicePushDownReshape>(allowPushDownOnProtected));
            rules.emplace_back(std::make_unique<SlicePushDownConcat>(allowPushDownOnProtected));
            rules.emplace_back(std::make_unique<SlicePushDownRepeat>(allowPushDownOnProtected));
        }
        // rules.emplace_back(std::make_unique<DistributiveProperty>());

        std::map<std::string, uint32_t> ruleMatchCounts;
        size_t iterations = 0;
        bool changed = true;
        uint32_t nMatches = 0;
#ifdef DEBUG
        ProgressTimer timer(0, "saturating ");
#endif
        while (changed)
        {
            iterations++;
            uint32_t numENodes = egraph.getENodes().size();
            // #ifdef DEBUG
            ProgressTimer timer2(0, "saturation round " + std::to_string(iterations - 1) + " ");
            // #endif
            for (uint32_t eNodeIdx = 0; eNodeIdx < egraph.getENodes().size(); eNodeIdx++)
            {
                for (const auto &rule : rules)
                {
                    if (!rule->match(egraph, eNodeIdx, protectedEClasses))
                        continue;

                    rule->apply(egraph, eNodeIdx, protectedEClasses, eclassToLogical);
                    changed = true;
                    ruleMatchCounts[rule->name()]++;
                    nMatches++;
                }
                // #ifdef DEBUG
                timer2.tick();
                // #endif
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
                uint32_t c = egraph.findConst(node.children[i]);
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
                                 bool stopOnFirstValid = true,
                                 bool cheapInputCopy = false,
                                 bool strictCache = false)
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

            if (enode.opType == OpType::INPUT)
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
                }
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

            // if (validEnodes.empty())
            // {
            //     std::cout << "[Planner.extractBest] Warning: EClass " << eclassId << " has NO valid enodes\n";
            // }

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
            // Only allocate and size vectors for canonical classes
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

        int max_iters = 10000;
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
                    best_cost = current_cost;
                    best_selection_map = selection_map;
                    std::cout << "new best cost: " << std::to_string(best_cost) << std::endl;
                }

                if (stopOnFirstValid)
                {
                    break;
                }
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
                if (cachedNodes.count(logicalId) && (physId == lastPhysIdForLogical[logicalId] || tNode.opType == OpType::INPUT))
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

        // Validate all INPUT nodes
        for (const auto &pair : compiled.nodesMap)
        {
            const TensorNode &node = pair.second;
            if (node.opType == OpType::INPUT && node.storageType == StorageType::TRANSIENT)
            {
                uint32_t logicalId = compiled.getLogicalId(node.id);
                if (logicalId == UINT32_MAX)
                {
                    Error::throw_err("[buildCompiledGraph] Orphan cache INPUT node " + std::to_string(node.id) + " has no logicalId mapping and is TRANSIENT. This will crash at runtime. A rewrite rule forgot to register its cache node in eclassToLogical.");
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
                enode.leafId = node.id;
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
            uint32_t physId = kv.first; // Here physId == logicalId
            uint32_t ecl = baseState.egraph.find(kv.second);
            baseState.eclassToLogical[ecl] = physId;
        }

        if (doSaturate)
        {
            std::unordered_set<uint32_t> emptyProtected;
            saturate(baseState.egraph, emptyProtected, baseState.eclassToLogical, false);
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
        std::unordered_map<uint32_t, uint32_t> &eclassToLogical)
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

        const EClass &lClass = egraph.getEClass(E_L);

        uint32_t E_Cache = egraph.addEClass(lClass.shape, lClass.strides, lClass.viewOffset, lClass.dtype, targetBackend);
        ENode cacheNode;
        cacheNode.opType = OpType::INPUT;
        cacheNode.dtype = lClass.dtype;
        cacheNode.shape = lClass.shape;
        cacheNode.strides = lClass.strides;
        cacheNode.viewOffset = lClass.viewOffset;
        cacheNode.backend = targetBackend;
        cacheNode.leafId = logicalId;
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
            if (node.opType == OpType::INPUT && graph.weightSources.count(nodeId) == 0 && graph.constantStaging.count(nodeId) == 0)
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
        const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
        const std::unordered_map<uint32_t, Backend> &cachedNodes,
        const std::vector<Region> &outputNeeded,
        bool doSaturate = true,
        bool cheapInputCopy = false,
        bool strictCache = false)
    {
        // 1. saturate base egraph
        initBaseEGraph(rootId, graph, doSaturate);

        EGraph egraph = baseState.egraph;
        auto eclassToLogical = baseState.eclassToLogical;

        std::unordered_set<uint32_t> protectedEClasses;
        for (const auto &kv : cachedNodes)
        {
            uint32_t logicalId = kv.first;
            protectedEClasses.insert(egraph.find(baseState.nodeToEClass.at(logicalId)));
        }

        // 2. inject input dirty slices
        bool dirtyInjected = injectInputPartialPaths(egraph, graph, dirtyOutputRegions, cachedNodes, baseState.nodeToEClass, eclassToLogical);

        // 3. saturate
        if (doSaturate && dirtyInjected)
        {
            saturate(egraph, protectedEClasses, eclassToLogical, true, true);
        }

        // 4. inject output needed regions
        bool neededInjected = injectOutputPartialPaths(egraph, graph, rootId, outputNeeded, cachedNodes, baseState.nodeToEClass, eclassToLogical);

        // 5. saturate again, but this time don't allow slice push down/pull up if the eclass is protected
        if (doSaturate && neededInjected)
        {
            saturate(egraph, protectedEClasses, eclassToLogical, true, false);
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

        auto extraction = extractBest(rootId, graph, egraph, baseState.nodeToEClass, maxMemoryByBackend, cachedNodes, eclassToLogical, immutable_eclasses, true, cheapInputCopy, strictCache);

        return buildCompiledGraph(
            rootId, graph, egraph, baseState.nodeToEClass, extraction, cachedNodes, eclassToLogical);
    }
};