#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/cost_model.hpp"
#include "core/kernels.hpp"
#include "core/rewrite.hpp"
#include "core/shapes.hpp"
#include "core/misc.hpp"
#include "core/egraph.hpp"
#include "core/plan/extractor.hpp"
#include "core/plan/validators/cycle.hpp"
#include "core/plan/validators/mem.hpp"
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
#include <queue>

struct ExtractChoice
{
    ENodeId enodeId;
    float cost = std::numeric_limits<float>::infinity();
    bool valid = false;
};

struct ExtractionResult
{
    std::unordered_map<EClassId, ExtractChoice> choiceByEClass;
    float totalCost = std::numeric_limits<float>::infinity();
};

inline std::vector<EClassId> topologicalSort(const EClassId &root, const EGraph &egraph, const std::unordered_map<EClassId, ExtractChoice> &choiceByEClass)
{
    std::vector<EClassId> topo;
    std::unordered_set<EClassId> visited_classes;
    std::function<void(EClassId)> visit = [&](EClassId e_class_id)
    {
        e_class_id = egraph.findConst(e_class_id);
        if (visited_classes.count(e_class_id))
            return;
        visited_classes.insert(e_class_id);

        auto choiceIt = choiceByEClass.find(e_class_id);
        if (choiceIt == choiceByEClass.end() || !choiceIt->second.valid)
            return;

        const ENode &enode = egraph.getENode(choiceIt->second.enodeId);
        for (EClassId child : enode.getChildren())
            visit(child);
        topo.push_back(e_class_id);
    };
    return topo;
}

struct Planner
{
    CostModel &costModel;
    std::unordered_map<uint32_t, uint64_t> mem_caps; // mem_idx -> max memory

    std::vector<MemSpace> findMemSpacePath(MemSpace src, MemSpace dst, const TensorNode &node, Engine engine)
    {
        if (src == dst)
            return {src};

        std::unordered_map<MemSpace, std::vector<MemSpace>> adj;
        for (const auto &[uid, k] : KernelRegistry::get().getAllKernels())
        {
            if (k.opType == OpType::COPY_TO && k.input_mem_spaces.size() == 1)
            {
                adj[k.input_mem_spaces[0]].push_back(k.output_mem_space);
            }
        }

        std::unordered_map<MemSpace, MemSpace> parent;
        std::queue<MemSpace> q;
        std::unordered_set<MemSpace> visited;

        q.push(src);
        visited.insert(src);

        bool found = false;
        std::cout << "findMemSpacePath" << std::endl;
        while (!q.empty())
        {
            MemSpace curr = q.front();
            q.pop();
            std::cout << curr << std::endl;

            if (curr == dst)
            {
                found = true;
                break;
            }

            for (MemSpace next : adj[curr])
            {
                if (visited.find(next) == visited.end())
                {
                    TensorNode dummyIn = node;
                    TensorNode dummyOut = node;
                    auto refs = KernelRegistry::get().findMatchingKernels(
                        OpType::COPY_TO, "", {dummyIn}, dummyOut, false, next, {curr}, {engine}, false, false, false, true);
                    if (!refs.empty())
                    {
                        visited.insert(next);
                        parent[next] = curr;
                        q.push(next);
                    }
                }
            }
        }

        if (!found)
            return {};

        std::vector<MemSpace> path;
        MemSpace curr = dst;
        while (!(curr == src))
        {
            path.push_back(curr);
            curr = parent[curr];
        }
        path.push_back(src);
        std::reverse(path.begin(), path.end());
        return path;
    }

    void inferShapes(const std::vector<LogicalId> &topo, Graph &graph)
    {
        ShapePropagator propagator;
        for (LogicalId nodeId : topo)
        {
            propagator.inferShape(nodeId, graph);
        }
    }

    void saturate(EGraph &egraph, const std::unordered_set<EClassId> &protectedEClasses, std::unordered_map<EClassId, LogicalId> &eclassToLogical, bool injected, bool allowPushDownOnProtected = false, Repo *repo = nullptr)
    {
        RuleCtx ctx{egraph, protectedEClasses, eclassToLogical, repo};
        std::vector<std::unique_ptr<Rule>> rules;
        rules.emplace_back(std::make_unique<FusionRule>());
        rules.emplace_back(std::make_unique<FlattenBatchDot>());
        rules.emplace_back(std::make_unique<FlattenElementwise>());
        if (injected)
        {
            rules.emplace_back(std::make_unique<InfinityDomination>());
            rules.emplace_back(std::make_unique<SlicePushDownElementwise>(allowPushDownOnProtected));
            rules.emplace_back(std::make_unique<SlicePushDownDot>(allowPushDownOnProtected));
        }

        std::map<std::string, uint32_t> ruleMatchCounts;
        uint64_t iterations = 0;
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

    ExtractionResult extractBest(const LogicalId rootId, const Graph &graph, EGraph &egraph,
                                 const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                                 const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
                                 const std::unordered_map<EClassId, LogicalId> &eclassToLogical,
                                 bool stopOnFirstValid = true,
                                 bool strictCache = false)
    {
        constexpr float EPS = 1e-6f;

        auto isConstantNeeded = [](OpType op, uint64_t inputIdx, uint64_t numInputs) -> bool
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
        for (uint32_t i = 0; i < egraph.getENodes().size(); ++i)
        {
            const ENode &enode = egraph.getENodes()[i];
            ENodeInfo info;
            info.inplace = false;
            info.inplace_idx = -1;
            info.is_view = false;

            if (enode.getKernelId() != KernelId{0})
            {
                const auto &kernel = KernelRegistry::get().getKernel(enode.getKernelId());
                info.is_view = kernel.is_view;
            }

            if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
            {
                info.cost = 0.0f;
                if (strictCache && enode.getOpType() == OpType::CACHE)
                {
                    EClassId e_class_id = egraph.getENodeEClass(ENodeId{i});
                    EClassId canonId = egraph.findConst(e_class_id);
                    LogicalId logicalId = eclassToLogical.count(canonId) ? eclassToLogical.at(canonId) : LogicalId{UINT32_MAX};
                    if (logicalId == LogicalId{UINT32_MAX} || cachedNodes.find(logicalId) == cachedNodes.end())
                    {
                        info.cost = TGConstants::INF;
                    }
                    else if (enode.getMemSpace() != cachedNodes.at(logicalId))
                    {
                        info.cost = TGConstants::INF;
                    }
                }
            }
            else if (enode.getKernelId() != KernelId{0})
            {
                std::vector<std::vector<uint32_t>> inShapes;
                std::vector<std::vector<uint64_t>> inStrides;
                std::vector<DType> inDTypes;
                std::vector<std::vector<uint8_t>> inConstants;

                inShapes.reserve(enode.getChildren().size());
                inStrides.reserve(enode.getChildren().size());
                inDTypes.reserve(enode.getChildren().size());
                inConstants.reserve(enode.getChildren().size());

                const ReferenceGraphEntry *refEntry = nullptr;
                std::unique_ptr<Graph> pGraph;
                std::vector<LogicalId> pInputs;

                const auto &kernel = KernelRegistry::get().getKernel(enode.getKernelId());
                if (enode.getOpType() == OpType::FUSED)
                {
                    refEntry = ReferenceGraphRegistry::get().getFactory(kernel.opName);
                    if (refEntry)
                    {
                        pGraph = std::make_unique<Graph>();
                        for (uint64_t k = 0; k < kernel.min_num_inputs; ++k)
                        {
                            pInputs.push_back(pGraph->input(kernel.dummyShapes[k], kernel.dtypes[k]));
                        }
                        refEntry->factory(pInputs, *pGraph);
                    }
                }

                for (uint64_t j = 0; j < enode.getChildren().size(); j++)
                {
                    EClassId childEClassId = enode.getChildren()[j];
                    const EClass &childCls = egraph.getEClass(egraph.find(childEClassId));
                    inShapes.push_back(childCls.shape);

                    std::vector<uint64_t> strides_cast;
                    strides_cast.reserve(childCls.strides.size());
                    for (uint64_t s : childCls.strides)
                        strides_cast.push_back(s);
                    inStrides.push_back(std::move(strides_cast));

                    inDTypes.push_back(childCls.dtype);

                    EClassId canonChild = egraph.find(childEClassId);
                    bool needed = false;

                    if (enode.getOpType() == OpType::FUSED)
                    {
                        if (refEntry && pGraph)
                        {
                            auto traceToInputIdx = [&](LogicalId pid) -> int
                            {
                                LogicalId curr = pid;
                                while (pGraph->hasNode(curr) &&
                                       (pGraph->getNode(curr).opType == OpType::CONTIGUOUS ||
                                        pGraph->getNode(curr).opType == OpType::CAST ||
                                        pGraph->getNode(curr).opType == OpType::COPY_TO ||
                                        pGraph->getNode(curr).opType == OpType::RESHAPE ||
                                        pGraph->getNode(curr).opType == OpType::PERMUTE))
                                {
                                    if (pGraph->getNode(curr).child_ids.empty())
                                        break;
                                    curr = pGraph->getNode(curr).child_ids[0];
                                }
                                for (uint64_t k = 0; k < pInputs.size(); ++k)
                                {
                                    if (pInputs[k] == curr)
                                        return (int)k;
                                }
                                return -1;
                            };

                            for (const auto &pair : pGraph->nodes)
                            {
                                const TensorNode &n = pair.second;
                                for (uint64_t p_idx = 0; p_idx < n.child_ids.size(); ++p_idx)
                                {
                                    if (isConstantNeeded(n.opType, p_idx, n.child_ids.size()))
                                    {
                                        int inputIdx = traceToInputIdx(n.child_ids[p_idx]);
                                        if (kernel.min_num_inputs != kernel.max_num_inputs)
                                        {
                                            if (inputIdx == (int)kernel.min_num_inputs - 1 && j == enode.getChildren().size() - 1)
                                            {
                                                needed = true;
                                                break;
                                            }
                                            else if (inputIdx >= 0 && inputIdx < (int)kernel.min_num_inputs - 1 && j < enode.getChildren().size() - 1)
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
                        needed = isConstantNeeded(enode.getOpType(), j, enode.getChildren().size());
                    }

                    if (needed && egraph.constantStaging.count(canonChild))
                    {
                        inConstants.push_back(*egraph.constantStaging.at(canonChild));
                    }
                    else
                    {
                        inConstants.push_back({});
                    }
                }

                info.cost = costModel.estimateCost(
                    enode.getKernelId(),
                    enode.getShape(),
                    enode.getStrides(),
                    enode.getDType(),
                    inShapes, inStrides, inDTypes, inConstants);
            }
            else
            {
                Error::throw_err("[Planner.extractBest] enode.kernelId != 0, but isn't OpType::INPUT or OpType::CACHE. this shouldn't happen");
            }

            enodeInfos[i] = std::move(info);
            timer3.tick();
        }

        bool droppedInf = false;
        uint32_t totalPruned = 0;
        for (uint32_t i = 0; i < egraph.getClasses().size(); ++i)
        {
            EClassId e_class_id = egraph.find(EClassId{i});
            if (e_class_id != EClassId{i})
                continue;

            EClass &cls = egraph.getEClass(e_class_id);
            std::vector<ENodeId> validEnodes;
            validEnodes.reserve(cls.enodes.size());

            // 1. Remove infinite-cost nodes
            for (ENodeId enodeId : cls.enodes)
            {
                if (enodeInfos[enodeId.value].cost == TGConstants::INF)
                {
                    droppedInf = true;
                }
                else
                {
                    validEnodes.push_back(enodeId);
                }
            }

            // 2. Prune duplicated nodes to minimize search space bloat
            std::vector<ENodeId> deduped;
            deduped.reserve(validEnodes.size());
            for (uint64_t idxA = 0; idxA < validEnodes.size(); ++idxA)
            {
                ENodeId idA = validEnodes[idxA];
                const ENode &a = egraph.getENode(idA);
                const ENodeInfo &ia = enodeInfos[idA.value];

                bool dominated = false;
                for (uint64_t idxB = 0; idxB < validEnodes.size(); ++idxB) // TODO: loop from idxA+1 and check both ways
                {
                    if (idxA == idxB)
                        continue;
                    ENodeId idB = validEnodes[idxB];
                    const ENode &b = egraph.getENode(idB);
                    const ENodeInfo &ib = enodeInfos[idB.value];

                    if (a != b)
                        continue;

                    // B dominates A only if strictly cheaper, OR equal-cost with smaller ID.
                    if (ib.cost < ia.cost - 1e-9f)
                    {
                        dominated = true;
                        break;
                    }
                    if (std::abs(ib.cost - ia.cost) <= 1e-9f && idB < idA)
                    {
                        dominated = true;
                        break;
                    }
                }
                if (!dominated)
                    deduped.push_back(idA);
            }
            totalPruned += (validEnodes.size() - deduped.size());
            cls.enodes = std::move(deduped);
        }

        if (totalPruned > 0)
        {
            std::cout << "[Planner.extractBest] Pruned " << totalPruned << " dominated enodes from the search space." << std::endl;
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
        EClassId rootEClassId = egraph.find(rootIt->second);

        const uint64_t numClasses = egraph.getClasses().size();

        std::vector<EClassId> canonicalClasses;
        std::vector<uint32_t> classToBitIdx(numClasses, UINT32_MAX);
        for (uint32_t i = 0; i < numClasses; ++i)
        {
            EClassId e_class_id = egraph.find(EClassId{i});
            if (e_class_id == EClassId{i})
            {
                classToBitIdx[i] = static_cast<uint32_t>(canonicalClasses.size());
                canonicalClasses.push_back(EClassId{i});
            }
        }

        const uint64_t numCanonical = canonicalClasses.size();
        const uint64_t bitWords = numCanonical == 0 ? 0 : (numCanonical + 63) >> 6;

        auto bitTest = [&](const std::vector<uint64_t> &bits, EClassId e_class_id) -> bool
        {
            uint32_t idx = classToBitIdx[e_class_id.value];
            if (idx == UINT32_MAX || bits.empty())
                return false;
            return (bits[idx >> 6] >> (idx & 63)) & 1ULL;
        };

        auto bitSet = [&](std::vector<uint64_t> &bits, EClassId e_class_id)
        {
            uint32_t idx = classToBitIdx[e_class_id.value];
            if (idx != UINT32_MAX && !bits.empty())
            {
                bits[idx >> 6] |= (1ULL << (idx & 63));
            }
        };

        struct OptSummary
        {
            float cost = TGConstants::INF;
            float intrinsic = TGConstants::INF;
            ENodeId chosenEnode = ENodeId{UINT32_MAX};
            std::vector<uint64_t> coveredBits;
            bool valid = false;
        };

        std::vector<OptSummary> opt(numClasses);
        for (EClassId canonId : canonicalClasses)
        {
            opt[canonId.value].coveredBits.assign(bitWords, 0);
        }

        std::vector<std::vector<EClassId>> parentMap(numClasses);
        for (EClassId e_class_id : canonicalClasses)
        {
            const EClass &cls = egraph.getEClass(e_class_id);
            for (ENodeId enodeId : cls.enodes)
            {
                const ENode &enode = egraph.getENode(enodeId);
                for (EClassId child : enode.getChildren())
                {
                    EClassId childEClass = egraph.find(child);
                    parentMap[childEClass.value].push_back(e_class_id);
                }
            }
        }

        for (auto &parents : parentMap)
        {
            std::sort(parents.begin(), parents.end());
            parents.erase(std::unique(parents.begin(), parents.end()), parents.end());
        }

        std::vector<EClassId> worklist;
        std::vector<EClassId> next_worklist;
        std::vector<bool> inQueue(numClasses, false);

        worklist.reserve(numClasses);
        next_worklist.reserve(numClasses);

        for (EClassId e_class_id : canonicalClasses)
        {
            worklist.push_back(e_class_id);
            inQueue[e_class_id.value] = true;
        }

        std::vector<uint64_t> candidateBits(bitWords, 0);
        std::vector<float> optimisticEnodeDagCost(egraph.getENodes().size(), TGConstants::INF);

        ProgressTimer optTimer(0, "calculating optimistic cost");
        while (!worklist.empty())
        {
            for (EClassId e_class_id : worklist)
            {
                inQueue[e_class_id.value] = false;

                const EClass &cls = egraph.getEClass(e_class_id);
                OptSummary best;
                best.coveredBits.assign(bitWords, 0);

                for (ENodeId enodeId : cls.enodes)
                {
                    const ENodeInfo &info = enodeInfos[enodeId.value];
                    if (info.cost == TGConstants::INF)
                        continue;

                    std::fill(candidateBits.begin(), candidateBits.end(), 0);
                    float candidateCost = info.cost;
                    bool candidateValid = true;

                    bitSet(candidateBits, e_class_id);

                    const ENode &enode = egraph.getENode(enodeId);

                    std::vector<EClassId> childEClasses;
                    childEClasses.reserve(enode.getChildren().size());
                    for (EClassId child : enode.getChildren())
                    {
                        childEClasses.push_back(egraph.find(child));
                    }
                    std::sort(childEClasses.begin(), childEClasses.end());
                    childEClasses.erase(std::unique(childEClasses.begin(), childEClasses.end()), childEClasses.end());

                    for (EClassId childEClass : childEClasses)
                    {
                        if (childEClass == e_class_id)
                        {
                            candidateValid = false;
                            break;
                        }

                        const OptSummary &childOpt = opt[childEClass.value];
                        if (!childOpt.valid)
                        {
                            candidateValid = false;
                            break;
                        }

                        if (bitTest(childOpt.coveredBits, e_class_id))
                        {
                            candidateValid = false;
                            break;
                        }

                        for (uint64_t w = 0; w < bitWords; ++w)
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
                                    EClassId k_eclass = canonicalClasses[k_idx];
                                    candidateCost += opt[k_eclass.value].intrinsic;
                                }
                                newBits &= (newBits - 1);
                            }
                        }
                    }

                    if (!candidateValid)
                        continue;

                    optimisticEnodeDagCost[enodeId.value] = candidateCost;

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

                if (!opt[e_class_id.value].valid ||
                    best.cost < opt[e_class_id.value].cost - EPS ||
                    (std::abs(best.cost - opt[e_class_id.value].cost) <= EPS && best.chosenEnode < opt[e_class_id.value].chosenEnode))
                {
                    opt[e_class_id.value] = std::move(best);

                    for (EClassId parentId : parentMap[e_class_id.value])
                    {
                        if (!inQueue[parentId.value])
                        {
                            inQueue[parentId.value] = true;
                            next_worklist.push_back(parentId);
                        }
                    }
                }
            }

            worklist.clear();
            std::swap(worklist, next_worklist);
            optTimer.tick();
        }

        std::vector<float> eclassMinCost(numClasses, TGConstants::INF);
        for (EClassId e_class_id : canonicalClasses)
        {
            if (opt[e_class_id.value].valid)
            {
                eclassMinCost[e_class_id.value] = opt[e_class_id.value].cost;
            }
        }

        std::vector<uint64_t> tempBits(bitWords, 0);
        for (EClassId e_class_id : canonicalClasses)
        {
            const EClass &cls = egraph.getEClass(e_class_id);
            for (ENodeId enodeId : cls.enodes)
            {
                const ENodeInfo &info = enodeInfos[enodeId.value];
                if (info.cost == TGConstants::INF)
                {
                    optimisticEnodeDagCost[enodeId.value] = TGConstants::INF;
                    continue;
                }

                std::fill(tempBits.begin(), tempBits.end(), 0);
                bitSet(tempBits, e_class_id);

                float total = info.cost;
                bool valid = true;

                const ENode &enode = egraph.getENode(enodeId);

                std::vector<EClassId> childEClasses;
                childEClasses.reserve(enode.getChildren().size());
                for (EClassId child : enode.getChildren())
                {
                    childEClasses.push_back(egraph.find(child));
                }
                std::sort(childEClasses.begin(), childEClasses.end());
                childEClasses.erase(std::unique(childEClasses.begin(), childEClasses.end()), childEClasses.end());

                for (EClassId childEClass : childEClasses)
                {
                    if (childEClass == e_class_id)
                    {
                        valid = false;
                        break;
                    }
                    if (!opt[childEClass.value].valid)
                    {
                        valid = false;
                        break;
                    }
                    if (bitTest(opt[childEClass.value].coveredBits, e_class_id))
                    {
                        valid = false;
                        break;
                    }

                    for (uint64_t w = 0; w < bitWords; ++w)
                    {
                        uint64_t newBits = opt[childEClass.value].coveredBits[w] & ~tempBits[w];
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
                                EClassId k_eclass = canonicalClasses[k_idx];
                                total += opt[k_eclass.value].intrinsic;
                            }
                            newBits &= (newBits - 1);
                        }
                    }
                }

                optimisticEnodeDagCost[enodeId.value] = valid ? total : TGConstants::INF;
            }
        }

        for (EClassId e_class_id : canonicalClasses)
        {
            EClass &cls = egraph.getEClass(e_class_id);
            std::sort(cls.enodes.begin(), cls.enodes.end(),
                      [&](ENodeId a, ENodeId b)
                      {
                          float costA = optimisticEnodeDagCost[a.value];
                          float costB = optimisticEnodeDagCost[b.value];

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
        float root_optimistic_cost = optimisticEnodeDagCost[egraph.getEClass(rootEClassId).enodes[0].value];
        std::cout << "[Planner.extractBest] Optimistic root cost: "
                  << std::to_string(root_optimistic_cost) << std::endl;
        if (std::isinf(root_optimistic_cost))
        {
            Error::throw_err("[Planner.extractBest] cannot have inf root cost");
        }

        std::vector<uint32_t> ref_counts(numClasses, 0);
        std::vector<bool> processed_mem(numClasses, false);
        std::vector<uint32_t> sim_aliasMap(numClasses, UINT32_MAX);
        std::vector<bool> visit_visited(numClasses, false);

        std::vector<uint32_t> topo_order;
        topo_order.reserve(numClasses);
        std::vector<bool> val_visited_classes(numClasses, false);
        std::vector<uint32_t> val_overwritten(numClasses, UINT32_MAX);
        std::vector<uint32_t> val_mem_root(numClasses);

        std::vector<uint32_t> indegree(numClasses, 0);
        std::vector<uint32_t> zero_indegree;
        zero_indegree.reserve(numClasses);

        Extractor extractor = Extractor(numClasses);
        extractor.registerValidator(std::make_unique<CycleValidator>(egraph));
        extractor.registerValidator(std::make_unique<MemValidator>(egraph, enodeInfos, mem_caps, stopOnFirstValid));

        float best_cost = TGConstants::INF;
        std::unordered_map<EClassId, uint32_t> best_selection_map;
        std::string reason = "";

        int max_iters = 100000;
        int remaining_iters = max_iters;
        ProgressTimer timer(max_iters, "extracting graphs ");
        ProgressTimer loopTimer(0, "", true);
        while (remaining_iters-- > 0)
        {
            std::cout << "loop " << std::to_string(loopTimer.getElapsed() * 1000) << "ms";
            if (max_iters - (remaining_iters + 1) > 0)
            {
                std::cout << ", avg loop " << std::to_string(timer.getElapsed() * 1000 / (max_iters - (remaining_iters + 1))) << "ms";
            }
            std::cout << std::endl;
            loopTimer.reset();
            timer.tick();

            const std::unordered_map<EClassId, uint32_t> &selection_map = extractor.getNextSelection();

            bool valid = extractor.validate(selection_map, reason);

            if (!valid)
            {
                std::cout << "[Planner.extractBest] [iter "
                          << std::to_string(max_iters - remaining_iters)
                          << "] invalid reason: " << reason << std::endl;
            }

            if (extractor.to_process_enode.empty())
                break; // Finished going through all graphs contained in egraph

            if (!valid)
            {
                extractor.backtrack(reason);
            }

            extractor.ascend(enodeInfos);
        }

        if (best_cost == TGConstants::INF)
        {
            Error::throw_err("[Planner.extractBest] no valid extraction found under given constraints. try running bench");
        }

        ExtractionResult result;
        result.totalCost = best_cost;

        for (auto const &kv : best_selection_map)
        {
            ExtractChoice c;
            c.enodeId = egraph.getEClass(kv.first).enodes[kv.second];
            c.cost = enodeInfos[c.enodeId.value].cost;
            c.valid = true;
            result.choiceByEClass[kv.first] = c;
        }

        return result;
    }

    CompiledGraph buildCompiledGraph(
        LogicalId rootId,
        const Graph &graph,
        EGraph &egraph,
        const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
        const ExtractionResult &extraction,
        const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
        const std::unordered_map<EClassId, LogicalId> &eclassToLogical)
    {
        CompiledGraph compiled;

        std::vector<EClassId> topo = topologicalSort(egraph.find(nodeToEClass.at(rootId)), egraph, extraction.choiceByEClass);

        std::unordered_map<EClassId, PhysicalId> eclassToPhys;
        for (EClassId e_class_id : topo)
        {
            eclassToPhys[e_class_id] = PhysicalIdAllocator::allocate();
        }

        std::vector<ENodeInfo> dummyInfos(egraph.getENodes().size());
        std::unordered_map<EClassId, uint32_t> selMap;
        for (const auto &kv : extraction.choiceByEClass)
        {
            dummyInfos[kv.second.enodeId.value].cost = kv.second.cost;
            const EClass &cls = egraph.getEClass(kv.first);
            for (uint32_t i = 0; i < cls.enodes.size(); ++i)
            {
                if (cls.enodes[i] == kv.second.enodeId)
                {
                    selMap[kv.first] = i;
                    break;
                }
            }
        }

        std::unordered_map<uint32_t, float> engine_finish;
        std::vector<ParallelBuffer> buffers = bufferize(topo, egraph, selMap, dummyInfos, engine_finish);

        std::unordered_map<uint32_t, std::vector<ParallelBuffer>> buf_by_mem_idx;
        for (auto &buf : buffers)
        {
            buf_by_mem_idx[buf.mem_space.idx].push_back(buf);
        }

        std::unordered_map<uint32_t, ParallelBuffer> final_allocs;
        for (auto &kv : buf_by_mem_idx)
        {
            std::vector<ParallelBuffer> allocated;
            uint64_t cap = mem_caps.count(kv.first) ? mem_caps.at(kv.first) : std::numeric_limits<uint64_t>::max();
            if (!malloc_recursive(cap, kv.second, allocated))
            {
                Error::throw_err("Failed to allocate memory in buildCompiledGraph!");
            }
            for (const auto &buf : allocated)
            {
                final_allocs[buf.eclass_val] = buf;
            }
        }

        for (EClassId e_class_id : topo)
        {
            const ExtractChoice &choice = extraction.choiceByEClass.at(e_class_id);
            const ENode &enode = egraph.getENode(choice.enodeId);
            LogicalId logicalId = eclassToLogical.count(e_class_id) ? eclassToLogical.at(e_class_id) : LogicalId{UINT32_MAX};
            PhysicalId physId = eclassToPhys[e_class_id];

            OpInstruction inst;
            inst.nodeId = physId;
            inst.logicalNodeId = logicalId;
            inst.fullKernelId = enode.getKernelId();
            inst.inputNodeIds.reserve(enode.getChildren().size());
            for (EClassId c : enode.getChildren())
            {
                inst.inputNodeIds.push_back(eclassToPhys[egraph.find(c)]);
            }

            if (final_allocs.count(e_class_id.value))
            {
                inst.outBuffer = final_allocs[e_class_id.value];
            }
            else
            {
                inst.outBuffer.offset = -1;
                inst.outBuffer.mem_space = enode.getMemSpace();
            }

            for (EClassId c : enode.getChildren())
            {
                EClassId cc = egraph.findConst(c);
                if (final_allocs.count(cc.value))
                {
                    inst.inBuffers.push_back(final_allocs[cc.value]);
                }
                else
                {
                    ParallelBuffer pb;
                    pb.offset = -1;
                    pb.mem_space = egraph.getEClass(cc).mem_space;
                    inst.inBuffers.push_back(pb);
                }
            }

            if (enode.getKernelId() != KernelId{0})
            {
                const KernelEntry &kEntry = KernelRegistry::get().getKernel(enode.getKernelId());
                inst.inplaceInputIndex = kEntry.is_view ? 0 : -1;
            }

            if (logicalId != LogicalId{UINT32_MAX} && graph.hasNode(logicalId))
            {
                inst.debugOrigin = graph.getNode(logicalId).debugOrigin;
            }

            if (egraph.constantStaging.count(e_class_id))
            {
                compiled.constantStaging[physId] = egraph.constantStaging.at(e_class_id);
            }

            if (enode.getOpType() != OpType::INPUT && enode.getOpType() != OpType::CACHE)
            {
                compiled.instructions.push_back(inst);
            }

            compiled.nodeCosts[physId] = choice.cost;
            compiled.physicalToLogicalNodeMap[physId] = logicalId;
        }

        return compiled;
    }

    struct BaseEGraphState
    {
        EGraph egraph;
        std::unordered_map<LogicalId, EClassId> nodeToEClass;
        std::unordered_map<EClassId, LogicalId> eclassToLogical;
    };

    BaseEGraphState baseState;
    bool baseStateInitialized = false;

    // Initialize baseState.egraph from graph
    void initBaseEGraph(LogicalId rootId, Graph &graph, const std::vector<LogicalId> &topo, Repo *repo = nullptr)
    {
        if (baseStateInitialized)
            return;

        inferShapes(topo, graph);

        baseState.nodeToEClass.reserve(graph.nodes.size());

        MemSpace storage = MemSpace{0, HandleType::STORAGE};
        MemSpace ram = MemSpace{1, HandleType::CPP};
        Engine cpu = Engine{0, EngineType::CPU};

        for (LogicalId nodeId : topo)
        {
            TensorNode &node = graph.getNode(nodeId);
            MemSpace mem_space = ram;
            if (node.opType == OpType::INPUT && graph.getInputDataType(nodeId) == InputDataType::STORAGE)
            {
                mem_space = storage;
            }
            EClassId e_class_id = baseState.egraph.addEClass(node.getShape(), node.strides, node.dtype, mem_space);
            baseState.nodeToEClass[nodeId] = e_class_id;
            if (graph.constantStaging.count(nodeId))
            {
                baseState.egraph.constantStaging[e_class_id] = graph.constantStaging.at(nodeId);
            }
        }

        for (LogicalId nodeId : topo)
        {
            const TensorNode &node = graph.getNode(nodeId);
            EClassId e_class_id = baseState.nodeToEClass[nodeId];

            if (node.opType == OpType::INPUT)
            {
                std::vector<EClassId> children;
                for (LogicalId pid : node.child_ids)
                    children.push_back(baseState.egraph.findConst(baseState.nodeToEClass[pid]));
                ENode enode = ENode(KernelId{0}, node.opType, node.opName, children, node.getShape(), node.strides, node.dtype, graph.getInputDataType(nodeId) == InputDataType::STORAGE ? storage : ram, {cpu});
                baseState.egraph.addENode(e_class_id, enode);
                continue;
            }

            std::vector<TensorNode> inputs;
            std::vector<MemSpace> input_mem_spaces;
            for (LogicalId pid : node.child_ids)
            {
                inputs.push_back(graph.getNode(pid));
                EClassId pid_eclass = baseState.egraph.findConst(baseState.nodeToEClass[pid]);
                input_mem_spaces.push_back(baseState.egraph.getEClass(pid_eclass).mem_space);
            }

            std::vector<KernelId> refs = KernelRegistry::get().findMatchingKernels(
                node.opType, node.opName, inputs, node, true, ram, input_mem_spaces, {cpu}, false, true, false, true);

            if (refs.size() == 0)
            {
                Error::throw_err("[Planner.initBaseEGraph] couldn't find any kernels to init EClass " + toString(e_class_id) + " " + toString(baseState.egraph.getEClass(e_class_id)) + "\nNode " + toString(node, graph));
            }

            bool any_success = false;
            for (KernelId uid : refs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);

                bool path_exists = true;
                std::vector<EClassId> children;

                for (uint64_t i = 0; i < node.child_ids.size(); ++i)
                {
                    LogicalId pid = node.child_ids[i];
                    EClassId p_eclass = baseState.egraph.findConst(baseState.nodeToEClass[pid]);
                    MemSpace src_ms = input_mem_spaces[i];

                    uint64_t ruleIdx = i;
                    if (kernel.min_num_inputs != kernel.max_num_inputs)
                    {
                        ruleIdx = (i == node.child_ids.size() - 1) ? (kernel.input_mem_spaces.empty() ? 0 : kernel.input_mem_spaces.size() - 1) : 0;
                    }
                    MemSpace dst_ms = ram;
                    if (!kernel.input_mem_spaces.empty() && ruleIdx < kernel.input_mem_spaces.size())
                    {
                        dst_ms = kernel.input_mem_spaces[ruleIdx];
                    }

                    bool requires_contig = false;
                    if (ruleIdx < kernel.requiresContiguous.size())
                    {
                        requires_contig = kernel.requiresContiguous[ruleIdx];
                    }

                    if (src_ms == dst_ms)
                    {
                        EClassId curr_eclass = p_eclass;
                        EClass curr_cls = baseState.egraph.getEClass(curr_eclass);
                        if (requires_contig && !isContiguous(curr_cls))
                        {
                            curr_eclass = addOpToEGraph(baseState.egraph, OpType::CONTIGUOUS, {curr_eclass}, curr_cls.shape, calcContiguousStrides(curr_cls.shape), curr_cls.dtype, curr_cls.mem_space);
                        }
                        children.push_back(curr_eclass);
                    }
                    else
                    {
                        std::vector<MemSpace> path = findMemSpacePath(src_ms, dst_ms, inputs[i], cpu);
                        if (path.empty())
                        {
                            path_exists = false;
                            break;
                        }

                        EClassId curr_eclass = p_eclass;
                        EClass curr_cls = baseState.egraph.getEClass(curr_eclass);

                        if (!isContiguous(curr_cls))
                        {
                            curr_eclass = addOpToEGraph(baseState.egraph, OpType::CONTIGUOUS, {curr_eclass}, curr_cls.shape, calcContiguousStrides(curr_cls.shape), curr_cls.dtype, curr_cls.mem_space);
                            curr_cls = baseState.egraph.getEClass(baseState.egraph.findConst(curr_eclass));
                        }

                        for (uint64_t p_idx = 1; p_idx < path.size(); ++p_idx)
                        {
                            MemSpace next_ms = path[p_idx];
                            curr_eclass = addOpToEGraph(baseState.egraph, OpType::COPY_TO, {curr_eclass}, curr_cls.shape, curr_cls.strides, curr_cls.dtype, next_ms);
                        }
                        children.push_back(curr_eclass);
                    }
                }

                if (!path_exists)
                    continue;
                any_success = true;

                std::vector<uint64_t> strides;
                if (kernel.is_view)
                {
                    strides = node.strides;
                }
                else
                {
                    strides = calcContiguousStrides(node.getShape());
                }
                ENode enode = ENode(uid, node.opType, node.opName, children, node.getShape(), strides, node.dtype, ram, {cpu});
                baseState.egraph.addENode(e_class_id, enode);
            }

            if (!any_success)
            {
                Error::throw_err("[Planner.initBaseEGraph] found kernels, but could not route memory spaces to satisfy input constraints for node " + toString(nodeId) + "\n" + toString(node, graph));
            }
        }

        for (const auto &kv : baseState.nodeToEClass)
        {
            baseState.eclassToLogical[baseState.egraph.findConst(kv.second)] = kv.first;
        }

        baseStateInitialized = true;
    }

    bool injectPartialPath(
        EGraph &egraph,
        const Graph &graph,
        LogicalId logicalId,
        const std::vector<Region> &regions,
        const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
        const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
        std::unordered_map<EClassId, LogicalId> &eclassToLogical,
        bool strictCache = false)
    {
        bool injected = false;
        EClassId E_L = egraph.find(nodeToEClass.at(logicalId));
        const TensorNode &sourceNode = graph.getNode(logicalId);

        bool isFullRegion = false;
        if (regions.size() == 1)
        {
            const Region &reg = regions[0];
            const auto &shape = sourceNode.getShape();
            if (reg.region.size() == shape.size())
            {
                isFullRegion = true;
                for (uint64_t d = 0; d < shape.size(); ++d)
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

        MemSpace ram = MemSpace{1, HandleType::CPP};
        Engine cpu = Engine{0, EngineType::CPU};

        MemSpace target_mem_space = ram;
        auto it = cachedNodes.find(logicalId);
        if (it != cachedNodes.end())
        {
            target_mem_space = it->second;
        }
        else if (strictCache)
        {
            return injected;
        }

        const EClass lClass = egraph.getEClass(E_L);

        EClassId E_Cache = egraph.addEClass(lClass.shape, lClass.strides, lClass.dtype, target_mem_space);
        ENode cacheNode(KernelId{0}, OpType::CACHE, "", {}, lClass.shape, lClass.strides, lClass.dtype, target_mem_space, {cpu});
        egraph.addENode(E_Cache, cacheNode);

        eclassToLogical[E_Cache] = logicalId;
        EClassId current_E = E_Cache;

        auto addConst = [&](const std::vector<int32_t> &vals)
        {
            return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, vals);
        };

        for (uint64_t r = 0; r < regions.size(); ++r)
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

            EClassId startsId = addConst(starts);
            EClassId endsId = addConst(ends);
            EClassId stepsId = addConst(steps);

            EClassId slicedEClass;

            if (sourceNode.opType == OpType::INPUT)
            {
                std::vector<uint64_t> sliceStrides = lClass.strides;

                for (uint64_t d = 0; d < starts.size(); ++d)
                {
                    int32_t start = starts[d];
                    if (start < 0)
                        start += lClass.shape[d];
                    sliceStrides[d] *= steps[d];
                }

                slicedEClass = egraph.addEClass(partialShape, sliceStrides, lClass.dtype, lClass.mem_space);

                TensorNode dOut;
                dOut.setShape(partialShape);
                dOut.dtype = lClass.dtype;
                std::vector<TensorNode> dIns(4);
                dIns[0].setShape(lClass.shape);
                dIns[0].dtype = lClass.dtype;
                dIns[1].setShape({(uint32_t)starts.size()});
                dIns[1].dtype = DType::INT32;
                dIns[2].setShape({(uint32_t)ends.size()});
                dIns[2].dtype = DType::INT32;
                dIns[3].setShape({(uint32_t)steps.size()});
                dIns[3].dtype = DType::INT32;

                std::vector<MemSpace> input_mem_spaces = {lClass.mem_space, ram, ram, ram};

                auto sliceRefs = KernelRegistry::get().findMatchingKernels(OpType::SLICE, "", dIns, dOut, true, lClass.mem_space, input_mem_spaces, {cpu});
                for (KernelId kid : sliceRefs)
                {
                    ENode sliceNode(kid, OpType::SLICE, "", {E_L, startsId, endsId, stepsId}, partialShape, sliceStrides, lClass.dtype, lClass.mem_space, {cpu});
                    egraph.addENode(slicedEClass, sliceNode);
                }
            }
            else
            {
                std::vector<EClassId> slicedInputs;
                std::vector<TensorNode> dummyInputNodes;
                std::vector<MemSpace> dummyInputMemSpaces;

                for (uint64_t p_idx = 0; p_idx < sourceNode.child_ids.size(); ++p_idx)
                {
                    LogicalId parentLogicalId = sourceNode.child_ids[p_idx];
                    EClassId E_parent = egraph.find(nodeToEClass.at(parentLogicalId));
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

                    EClassId pStartsId = addConst(pStarts);
                    EClassId pEndsId = addConst(pEnds);
                    EClassId pStepsId = addConst(pSteps);

                    std::vector<uint64_t> pSliceStrides = pClass.strides;
                    for (uint64_t d = 0; d < pStarts.size(); ++d)
                    {
                        int32_t start = pStarts[d];
                        if (start < 0)
                            start += pClass.shape[d];
                        pSliceStrides[d] *= pSteps[d];
                    }

                    EClassId pSliceEClass = egraph.addEClass(pPartialShape, pSliceStrides, pClass.dtype, pClass.mem_space);

                    TensorNode pOut;
                    pOut.setShape(pPartialShape);
                    pOut.dtype = pClass.dtype;

                    std::vector<TensorNode> pIns(4);
                    pIns[0].setShape(pClass.shape);
                    pIns[0].dtype = pClass.dtype;
                    pIns[1].setShape({(uint32_t)pStarts.size()});
                    pIns[1].dtype = DType::INT32;
                    pIns[2].setShape({(uint32_t)pEnds.size()});
                    pIns[2].dtype = DType::INT32;
                    pIns[3].setShape({(uint32_t)pSteps.size()});
                    pIns[3].dtype = DType::INT32;

                    std::vector<MemSpace> pSliceInputMemSpaces = {pClass.mem_space, ram, ram, ram};
                    auto pSliceRefs = KernelRegistry::get().findMatchingKernels(OpType::SLICE, "", pIns, pOut, true, pClass.mem_space, pSliceInputMemSpaces, {cpu});

                    for (KernelId uid : pSliceRefs)
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(uid);
                        std::vector<uint64_t> strides = kernel.is_view ? pSliceStrides : calcContiguousStrides(pPartialShape);
                        ENode sn(uid, OpType::SLICE, "", {E_parent, pStartsId, pEndsId, pStepsId}, pPartialShape, strides, pClass.dtype, pClass.mem_space, {cpu});
                        egraph.addENode(pSliceEClass, sn);
                    }

                    EClassId pContigEClass = egraph.addEClass(pPartialShape, calcContiguousStrides(pPartialShape), pClass.dtype, pClass.mem_space);

                    TensorNode cOut;
                    cOut.setShape(pPartialShape);
                    cOut.dtype = pClass.dtype;
                    cOut.strides = calcContiguousStrides(pPartialShape);

                    TensorNode cIn;
                    cIn.setShape(pPartialShape);
                    cIn.dtype = pClass.dtype;
                    cIn.strides = pSliceStrides;

                    auto contigRefs = KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", {cIn}, cOut, true, pClass.mem_space, {pClass.mem_space}, {cpu});
                    for (KernelId uid : contigRefs)
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(uid);
                        std::vector<uint64_t> strides = kernel.is_view ? pSliceStrides : calcContiguousStrides(pPartialShape);
                        ENode cn(uid, OpType::CONTIGUOUS, "", {pSliceEClass}, pPartialShape, strides, pClass.dtype, pClass.mem_space, {cpu});
                        egraph.addENode(pContigEClass, cn);
                    }

                    slicedInputs.push_back(pContigEClass);

                    TensorNode dummyIn;
                    dummyIn.opType = OpType::INPUT;
                    dummyIn.setShape(pPartialShape);
                    dummyIn.dtype = pClass.dtype;
                    dummyIn.strides = calcContiguousStrides(pPartialShape);
                    dummyInputNodes.push_back(dummyIn);
                    dummyInputMemSpaces.push_back(pClass.mem_space);
                }

                TensorNode dummyOut;
                dummyOut.opType = sourceNode.opType;
                dummyOut.opName = sourceNode.opName;
                dummyOut.setShape(partialShape);
                dummyOut.dtype = sourceNode.dtype;
                dummyOut.strides = calcContiguousStrides(partialShape);

                auto opRefs = KernelRegistry::get().findMatchingKernels(sourceNode.opType, sourceNode.opName, dummyInputNodes, dummyOut, true, target_mem_space, dummyInputMemSpaces, {cpu});
                if (opRefs.size() == 0)
                {
                    Error::throw_err("[Planner.injectPartialPath] couldn't find any slice kernels for op " + toString(sourceNode.opType));
                }

                slicedEClass = egraph.addEClass(partialShape, calcContiguousStrides(partialShape), sourceNode.dtype, target_mem_space);
                for (KernelId uid : opRefs)
                {
                    ENode sn(uid, sourceNode.opType, sourceNode.opName, slicedInputs, partialShape, calcContiguousStrides(partialShape), sourceNode.dtype, target_mem_space, {cpu});
                    egraph.addENode(slicedEClass, sn);
                }
            }

            EClassId contigEClass = egraph.addEClass(partialShape, calcContiguousStrides(partialShape), sourceNode.dtype, target_mem_space);

            TensorNode cOut;
            cOut.setShape(partialShape);
            cOut.dtype = sourceNode.dtype;
            cOut.strides = calcContiguousStrides(partialShape);

            TensorNode cIn;
            cIn.setShape(partialShape);
            cIn.dtype = sourceNode.dtype;
            cIn.strides = calcContiguousStrides(partialShape);

            auto contigRefs = KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", {cIn}, cOut, true, target_mem_space, {target_mem_space}, {cpu});
            for (KernelId uid : contigRefs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                std::vector<uint64_t> strides = kernel.is_view ? cIn.strides : calcContiguousStrides(partialShape);
                ENode cn(uid, OpType::CONTIGUOUS, "", {slicedEClass}, partialShape, strides, sourceNode.dtype, target_mem_space, {cpu});
                egraph.addENode(contigEClass, cn);
            }

            EClassId scatterEClass = egraph.addEClass(lClass.shape, lClass.strides, lClass.dtype, target_mem_space);

            TensorNode sOut;
            sOut.setShape(lClass.shape);
            sOut.dtype = lClass.dtype;

            std::vector<TensorNode> sIns(5);
            sIns[0].setShape(egraph.getEClass(current_E).shape);
            sIns[0].dtype = lClass.dtype;
            sIns[1].setShape(partialShape);
            sIns[1].dtype = lClass.dtype;
            sIns[2].setShape({(uint32_t)starts.size()});
            sIns[2].dtype = DType::INT32;
            sIns[3].setShape({(uint32_t)ends.size()});
            sIns[3].dtype = DType::INT32;
            sIns[4].setShape({(uint32_t)steps.size()});
            sIns[4].dtype = DType::INT32;

            std::vector<MemSpace> scatterInputSpaces = {target_mem_space, target_mem_space, ram, ram, ram};

            auto scatterRefs = KernelRegistry::get().findMatchingKernels(OpType::SCATTER, "", sIns, sOut, true, target_mem_space, scatterInputSpaces, {cpu});
            for (KernelId uid : scatterRefs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                std::vector<uint64_t> strides = (kernel.is_view) ? lClass.strides : calcContiguousStrides(lClass.shape); // TODO check inplace based on bufferization stuff
                ENode sn(uid, OpType::SCATTER, "", {current_E, contigEClass, startsId, endsId, stepsId}, lClass.shape, strides, lClass.dtype, target_mem_space, {cpu});
                egraph.addENode(scatterEClass, sn);
            }

            current_E = scatterEClass;
        }

        egraph.merge(E_L, current_E);
        eclassToLogical[egraph.find(E_L)] = logicalId;
        injected = true;
        return injected;
    }

    bool injectInputPartialPaths(
        EGraph &egraph,
        const Graph &graph,
        const std::unordered_map<LogicalId, std::vector<Region>> &dirtyOutputRegions,
        const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
        const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
        std::unordered_map<EClassId, LogicalId> &eclassToLogical)
    {
        bool injected = false;
        for (const auto &kv : dirtyOutputRegions)
        {
            LogicalId nodeId = kv.first;
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
        LogicalId rootId,
        const std::vector<Region> &outputNeeded,
        const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
        const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
        std::unordered_map<EClassId, LogicalId> &eclassToLogical)
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

    Planner(CostModel &costModel, const std::unordered_map<uint32_t, uint64_t> &mem_caps)
        : costModel(costModel), mem_caps(mem_caps) {}

    CompiledGraph plan(
        LogicalId rootId,
        const Graph &graph,
        const Bucket &bucket,
        const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
        bool doSaturate = true,
        bool strictCache = false,
        Repo *repo = nullptr)
    {
        std::vector<LogicalId> topo = topologicalSort({rootId}, graph);
        Graph tempGraph = graph;
        initBaseEGraph(rootId, tempGraph, topo, repo);

        EGraph egraph = baseState.egraph;
        auto eclassToLogical = baseState.eclassToLogical;

        std::unordered_map<LogicalId, bool> logicalDirty;
        for (LogicalId nodeId : topo)
        {
            if (bucket.inputDirtyRegions.count(nodeId) && !bucket.inputDirtyRegions.at(nodeId).empty())
            {
                logicalDirty[nodeId] = true;
            }
            else
            {
                bool isDirty = false;
                for (LogicalId pid : graph.getNode(nodeId).child_ids)
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

        // Add cache enodes
        Engine cpu = Engine{0, EngineType::CPU};
        for (const auto &cls : egraph.getClasses())
        {
            EClassId canonId = egraph.find(cls.id);
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
                if (egraph.getENode(cls.enodes[i]).getOpType() == OpType::CACHE)
                {
                    continue;
                }
            }

            LogicalId logicalId;
            auto it = eclassToLogical.find(canonId);
            if (it != eclassToLogical.end())
            {
                logicalId = it->second;
            }

            if (logicalId != LogicalId{UINT32_MAX} && !logicalDirty[logicalId])
            {
                ENode cacheNode = ENode(KernelId{0}, OpType::CACHE, "", {}, cls.shape, cls.strides, cls.dtype, cls.mem_space, {cpu});
                egraph.addENode(canonId, cacheNode);
            }
        }

        std::unordered_set<EClassId> protectedEClasses;
        for (const auto &kv : cachedNodes)
        {
            LogicalId logicalId = kv.first;
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

        std::unordered_map<EClassId, LogicalId> updatedEClassToLogical;
        for (const auto &kv : eclassToLogical)
        {
            updatedEClassToLogical[egraph.find(kv.first)] = kv.second;
        }
        eclassToLogical = std::move(updatedEClassToLogical);

        auto extraction = extractBest(rootId, graph, egraph, baseState.nodeToEClass, cachedNodes, eclassToLogical, true, strictCache);
        return buildCompiledGraph(
            rootId, graph, egraph, baseState.nodeToEClass, extraction, cachedNodes, eclassToLogical);
    }
};