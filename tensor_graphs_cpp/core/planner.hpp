#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/cost_model.hpp"
#include "core/kernels.hpp"
#include "core/rewrite.hpp"
#include "core/hashing.hpp"
#include "core/shapes.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cstring>

#define FUSE_OPS

class Planner
{
private:
    std::unordered_map<std::string, uint32_t> hashToId;
    uint32_t getHashId(const std::string &h)
    {
        auto it = hashToId.find(h);
        if (it != hashToId.end())
            return it->second;
        uint32_t id = hashToId.size() + 1;
        hashToId[h] = id;
        return id;
    }

    struct AdapterChain
    {
        std::vector<AdapterOp> ops;
        Backend finalBackend;
        bool finalContig;
        float cost;
    };

    std::vector<AdapterChain> getAdapterChains(const TensorNode &pNode, Backend pBackend, Backend targetBackend, const Graph &graph, const std::unordered_map<uint32_t, uint32_t> &refCounts, CostModel &costModel)
    {
        std::vector<AdapterChain> chains;
        bool isContig = pNode.view.isContiguous();

        auto evalKernel = [&](OpType op, Backend opBackend, const TensorNode &input, TensorNode &output, uint64_t &bestK, float &bestCost)
        {
            output = input; // copy properties
            // Keep input's original ID so costModel can look up its shape correctly.
            // (Both CONTIGUOUS and COPY_TO preserve shape/dtype)
            output.id = input.id;
            output.opType = op;
            output.opName = "";
            output.parentIds = {input.id};
            output.backend = opBackend;
            output.storageType = StorageType::TRANSIENT;
            if (op == OpType::CONTIGUOUS)
            {
                output.view.strides = TensorView::calcContiguousStrides(output.shape);
            }

            std::vector<TensorNode> inputsList = {input};
            std::vector<uint64_t> kernels = KernelRegistry::get().findMatchingKernels(op, "", opBackend, inputsList, output, &refCounts);
            bestCost = std::numeric_limits<float>::infinity();
            bestK = UINT64_MAX;
            for (uint64_t k : kernels)
            {
                float c = costModel.estimateCost(output, graph, k);
                if (c < bestCost)
                {
                    bestCost = c;
                    bestK = k;
                }
            }
        };

        if (pBackend == targetBackend)
        {
            // Chain 0: None
            chains.push_back({{}, targetBackend, isContig, 0.0f});

            // Chain 1: CONTIGUOUS
            if (!isContig)
            {
                TensorNode contigOut;
                uint64_t k;
                float c;
                TensorNode pNodeOverride = pNode;
                pNodeOverride.backend = pBackend;
                evalKernel(OpType::CONTIGUOUS, pBackend, pNodeOverride, contigOut, k, c);
                if (k != UINT64_MAX)
                {
                    chains.push_back({{{OpType::CONTIGUOUS, k, pBackend}}, targetBackend, true, c});
                }
            }
        }
        else
        {
            // Chain 2: COPY_TO
            TensorNode copyOut;
            uint64_t kCopy;
            float cCopy;
            TensorNode pNodeOverride = pNode;
            pNodeOverride.backend = pBackend;
            evalKernel(OpType::COPY_TO, targetBackend, pNodeOverride, copyOut, kCopy, cCopy);
            if (kCopy != UINT64_MAX)
            {
                chains.push_back({{{OpType::COPY_TO, kCopy, targetBackend}}, targetBackend, isContig, cCopy});
            }

            if (!isContig)
            {
                // Chain 3: CONTIGUOUS -> COPY_TO
                TensorNode contigOut;
                uint64_t kContig;
                float cContig;
                evalKernel(OpType::CONTIGUOUS, pBackend, pNodeOverride, contigOut, kContig, cContig);
                if (kContig != UINT64_MAX)
                {
                    TensorNode copyOut2;
                    uint64_t kCopy2;
                    float cCopy2;
                    evalKernel(OpType::COPY_TO, targetBackend, contigOut, copyOut2, kCopy2, cCopy2);
                    if (kCopy2 != UINT64_MAX)
                    {
                        chains.push_back({{{OpType::CONTIGUOUS, kContig, pBackend}, {OpType::COPY_TO, kCopy2, targetBackend}}, targetBackend, true, cContig + cCopy2});
                    }
                }

                // Chain 4: COPY_TO -> CONTIGUOUS
                if (kCopy != UINT64_MAX)
                {
                    TensorNode contigOut2;
                    uint64_t kContig2;
                    float cContig2;
                    evalKernel(OpType::CONTIGUOUS, targetBackend, copyOut, contigOut2, kContig2, cContig2);
                    if (kContig2 != UINT64_MAX)
                    {
                        chains.push_back({{{OpType::COPY_TO, kCopy, targetBackend}, {OpType::CONTIGUOUS, kContig2, targetBackend}}, targetBackend, true, cCopy + cContig2});
                    }
                }
            }
        }
        return chains;
    }

public:
    Planner(CostModel &costModel, uint64_t maxMemoryBytes = 4ULL * 1024 * 1024 * 1024)
        : costModel(costModel), maxMemoryBytes(maxMemoryBytes) {}

    CompiledGraph plan(uint32_t rootId, Graph &graph)
    {
        std::cout << "[Planner.plan] initial sort..." << std::endl;
        std::unordered_map<uint32_t, std::string> structHashMemo;
        std::unordered_map<uint32_t, std::string> patternHashMemo;
        std::vector<uint32_t> topo = topologicalSort(rootId, graph);

        std::cout << "[Planner.plan] inferring shapes..." << std::endl;
        inferShapes(topo, graph);

        // Build fused patterns from the Reference Graph Registry
        struct FusedPattern
        {
            std::string opName;
            std::vector<uint32_t> variables;
            uint32_t rootId;
            Graph graph;
        };
        std::vector<FusedPattern> fusedPatterns;

        std::cout << "[Planner.plan] generating fusedPatterns..." << std::endl;
        const std::unordered_map<std::string, ReferenceGraphEntry> &refGraphs = ReferenceGraphRegistry::get().getAll();
        uint32_t pairIdx = 0;
        for (const auto &pair : refGraphs)
        {
            pairIdx++;
            std::cout << pairIdx << "/" << refGraphs.size() << "\r";
            const std::string &opName = pair.first;
            size_t numInputs = pair.second.numInputs;
            ReferenceFactory factory = pair.second.factory;

            FusedPattern pattern;
            pattern.opName = opName;
            const auto &dummyShapes = pair.second.dummyShapes;

            for (size_t i = 0; i < numInputs; ++i)
            {
                uint32_t inId = pattern.graph.allocateId();
                TensorView view;
                view.shape = dummyShapes[i]; // TODO: maybe assert len
                view.strides = TensorView::calcContiguousStrides(view.shape);
                view.baseOffset = 0;
                DType dtype = pair.second.dtypes[i];
                view.dtype = dtype; // TODO: maybe assert len
                pattern.graph.inputWithId(inId, view.shape, dtype, view);
                pattern.variables.push_back(inId);
            }

            pattern.rootId = factory(pattern.variables, pattern.graph);

            std::vector<uint32_t> p_topo = topologicalSort(pattern.rootId, pattern.graph);
            // inferShapes(p_topo, pattern.graph);

            fusedPatterns.push_back(std::move(pattern));
        }
        std::unordered_map<OpType, std::vector<uint32_t>> patternsByRootOp;
        for (uint32_t i = 0; i < fusedPatterns.size(); i++)
        {
            OpType patternRootOp = fusedPatterns[i].graph.nodes[fusedPatterns[i].rootId].opType;
            patternsByRootOp[patternRootOp].push_back(i);
        }

        std::unordered_map<std::string, std::vector<uint32_t>> fusionMap;
#ifdef FUSE_OPS
        Rewrite::CommutativeRule cr;
        Rewrite::DistributiveRule dr;
        Rewrite::FactoringRule fr;
        Rewrite::AssociativeRule ar;
        Rewrite::DoubleNegationRule dnr;
        Rewrite::NegateAddRule nar;
        Rewrite::DivMulRule dmr;
        Rewrite::DivAddRule dar;

        std::vector<const Rewrite::RewriteRule *> rules = {&cr, &dr, &fr, &ar, &dnr, &nar, &dmr, &dar};
        // std::vector<const Rewrite::RewriteRule *> rules = {&cr, &dr};
        // std::vector<const Rewrite::RewriteRule *> rules = {}; // TODO: remove this line, this is just so the matching phase is fast while debugging planning phase

        std::cout << "[Planner.plan] matching fusion patterns..." << std::endl;
        uint32_t topoIdx = 0;
        uint32_t rewrites = 0;
        uint32_t fusionMatches = 0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            topoIdx++;
            std::cout << topoIdx << "/" << topo.size() << ", rewrites: " << rewrites << ", matches: " << fusionMatches << "\r";
            uint32_t nodeId = *it;
            // std::string patHash = Hashing::patternHash(nodeId, graph, patternHashMemo);
            std::string hash = Hashing::structuralHash(nodeId, graph, structHashMemo);

            std::vector<uint32_t> equivalents = Rewrite::generateAllEquivalents(nodeId, graph, rules, patternHashMemo);
            for (uint32_t eqId : equivalents)
            {
                if (eqId != nodeId)
                {
                    rewrites++;
                    fusionMap[hash].push_back(eqId);
                }

                OpType eqOpType = graph.nodes[eqId].opType;
                auto it = patternsByRootOp.find(eqOpType);
                if (it != patternsByRootOp.end())
                {
                    for (const uint32_t fpIdx : it->second)
                    {
                        const FusedPattern &fp = fusedPatterns[fpIdx];
                        std::unordered_map<uint32_t, uint32_t> binding;
                        if (matchPattern(eqId, graph, fp.rootId, fp.graph, fp.variables, binding))
                        {
                            bool allBound = true;
                            for (uint32_t var : fp.variables)
                            {
                                if (binding.find(var) == binding.end())
                                {
                                    allBound = false;
                                    break;
                                }
                            }
                            if (allBound)
                            {
                                std::vector<uint32_t> parentIds;
                                for (uint32_t var : fp.variables)
                                {
                                    parentIds.push_back(binding[var]);
                                }

                                TensorNode fusedNode;
                                fusedNode.id = graph.allocateId();
                                fusedNode.opType = OpType::FUSED;
                                fusedNode.opName = fp.opName;
                                fusedNode.dtype = graph.nodes[eqId].dtype;
                                fusedNode.shape = graph.nodes[eqId].shape;
                                fusedNode.parentIds = parentIds;
                                fusedNode.backend = graph.nodes[eqId].backend;

                                graph.nodes.push_back(fusedNode);

                                fusionMap[hash].push_back(fusedNode.id);
                                fusionMatches++;
                            }
                        }
                    }
                }
            }
        }
        std::cout << std::endl
                  << "# Rewrites: " << fusionMap.size() << std::endl;
        std::cout << "# Fusion matches: " << fusionMatches << std::endl;
#endif
        std::cout << "[Planner.plan] doing augmented topo sort..." << std::endl;
        std::vector<uint32_t> sortedNodes = getAugmentedTopologicalSort(topo, fusionMap, graph, structHashMemo);

        std::cout << "[Planner.plan] inferring shapes for augmented graph..." << std::endl;
        inferShapes(sortedNodes, graph);

        std::unordered_map<uint32_t, uint32_t> estimatedRefCounts;
        for (uint32_t id : topo)
        {
            for (uint32_t pid : graph.nodes[id].parentIds)
            {
                estimatedRefCounts[pid]++;
            }
        }
        estimatedRefCounts[rootId] = 1;

        std::unordered_set<uint32_t> topoSet(topo.begin(), topo.end());
        for (uint32_t id : sortedNodes)
        {
            if (topoSet.count(id) == 0)
            {
                for (uint32_t pid : graph.nodes[id].parentIds)
                {
                    if (topoSet.count(pid) == 0)
                    {
                        estimatedRefCounts[pid]++;
                    }
                }
            }
        }

        std::unordered_map<std::string, std::vector<std::shared_ptr<BeamStrategy>>> memo;

        std::cout << "[Planner.plan] planning nodes..." << std::endl;
        uint32_t nodeIdx = 0;
        for (uint32_t nodeId : sortedNodes)
        {
            nodeIdx++;
            std::cout << nodeIdx << "/" << sortedNodes.size() << "\r";
            planNodeIterative(nodeId, graph, fusionMap, memo, structHashMemo, estimatedRefCounts);
        }

        std::string rootHash = Hashing::structuralHash(rootId, graph, structHashMemo);
        if (memo.find(rootHash) == memo.end() || memo[rootHash].empty())
        {
            const auto &rootNode = graph.nodes[rootId];
            std::cout << "[Planner.plan] ERROR: Couldn't find any strategy for root node " << rootId
                      << " (OpType: " << toString(rootNode.opType)
                      << ", DType: " << toString(rootNode.dtype)
                      << ", Shape: " << toString(rootNode.shape)
                      << ", ParentCount: " << rootNode.parentIds.size()
                      << ")" << std::endl;

            // Print which parent IDs were not successfully planned
            for (uint32_t pid : rootNode.parentIds)
            {
                const auto &pNode = graph.nodes[pid];
                std::cout << "  Parent " << pid << ": OpType=" << toString(pNode.opType)
                          << ", DType=" << toString(pNode.dtype)
                          << ", Shape=" << toString(pNode.shape) << std::endl;
            }

            throw std::runtime_error("Planner failed to find any execution strategy for root node.");
        }

        auto bestRecipe = memo[rootHash][0];
        std::cout << "best recipe cost: " << bestRecipe->cost << std::endl;

        // Reconstruct the global maps from the winning path-tracking tree
        std::unordered_map<uint32_t, Backend> bestAssignments;
        std::unordered_map<uint32_t, uint64_t> bestKernelAssignments;
        std::unordered_map<uint32_t, uint32_t> bestSelectedNodes;
        std::unordered_map<uint64_t, std::vector<AdapterOp>> bestEdgeAdapters;

        std::unordered_set<uint32_t> visitedStrats; // track by original node ID to prevent duplicate DFS

        auto reconstruct = [&](auto &self, const std::shared_ptr<BeamStrategy> &strat) -> void
        {
            if (!strat)
                return;
            if (visitedStrats.count(strat->nodeId))
                return;
            visitedStrats.insert(strat->nodeId);

            // 1. Hash of the logical node (the node the user created, e.g., an ADD)
            std::string hLog = Hashing::structuralHash(strat->nodeId, graph, structHashMemo);
            uint32_t hLogId = getHashId(hLog);

            // 2. Hash of the physical node (the node we chose to run, e.g., FUSED_Add_3D_1D)
            std::string hPhys = Hashing::structuralHash(strat->selectedNodeId, graph, structHashMemo);
            uint32_t hPhysId = getHashId(hPhys);

            // Store physical assignments using the physical hash
            bestAssignments[hPhysId] = strat->backend;
            bestKernelAssignments[hPhysId] = strat->kernelId;

            // Store the mapping so that when we look at a logical node's children,
            // we know which physical node was actually chosen
            bestSelectedNodes[hLogId] = strat->selectedNodeId;

            for (size_t i = 0; i < strat->parentStrategies.size(); ++i)
            {
                const auto &pStrat = strat->parentStrategies[i];
                if (pStrat)
                {
                    // Edge adapters link physical parent to physical child
                    std::string phPhys = Hashing::structuralHash(pStrat->selectedNodeId, graph, structHashMemo);
                    uint32_t phPhysId = getHashId(phPhys);
                    uint64_t edgeHashId = ((uint64_t)phPhysId << 32) | hPhysId;

                    if (!strat->parentAdapters[i].empty())
                    {
                        bestEdgeAdapters[edgeHashId] = strat->parentAdapters[i];
                    }
                    self(self, pStrat);
                }
            }
        };
        reconstruct(reconstruct, bestRecipe);

        std::cout << "[Planner.plan] final topo sort..." << std::endl;
        std::vector<uint32_t> finalTopo;
        std::unordered_set<uint32_t> visited;

        auto visit = [&](auto &self, uint32_t currOriginalId) -> void
        {
            std::string h = Hashing::structuralHash(currOriginalId, graph, structHashMemo);
            uint32_t hId = getHashId(h);

            // Jump to the chosen fused node if one was selected
            uint32_t chosenId = bestSelectedNodes.count(hId) ? bestSelectedNodes.at(hId) : currOriginalId;

            if (visited.count(chosenId))
                return;
            visited.insert(chosenId);

            for (uint32_t pid : graph.nodes[chosenId].parentIds)
            {
                self(self, pid);
            }
            finalTopo.push_back(chosenId);
        };
        visit(visit, rootId);
        CompiledGraph compiled;

        uint32_t mapIdx = 0;
        std::unordered_map<std::string, uint32_t> insertedCopyNodes;
        std::vector<uint32_t> finalTopoWithCopies;
        for (uint32_t id : finalTopo)
        {
            mapIdx++;
            TensorNode mappedNode = graph.nodes[id];

            std::string h = Hashing::structuralHash(id, graph, structHashMemo);
            uint32_t hId = getHashId(h);
            if (bestAssignments.count(hId))
            {
                mappedNode.backend = bestAssignments.at(hId);
            }
            Backend assignedBackend = mappedNode.backend;
            std::vector<uint32_t> mappedParentIds;

            for (uint32_t pid : mappedNode.parentIds)
            {
                std::string phash = Hashing::structuralHash(pid, graph, structHashMemo);
                uint32_t phashId = getHashId(phash);
                uint32_t chosenPid = bestSelectedNodes.count(phashId) ? bestSelectedNodes.at(phashId) : pid;

                std::string chosenPhash = Hashing::structuralHash(chosenPid, graph, structHashMemo);
                uint32_t chosenPhashId = getHashId(chosenPhash);

                uint64_t edgeHashId = ((uint64_t)chosenPhashId << 32) | hId;

                if (bestEdgeAdapters.count(edgeHashId) && !bestEdgeAdapters.at(edgeHashId).empty())
                {
                    uint32_t currentPid = chosenPid;
                    for (const auto &adapter : bestEdgeAdapters.at(edgeHashId))
                    {
                        std::string adapterKey = std::to_string(currentPid) + "_" + toString(adapter.opType) + "_" + toString(adapter.backend);
                        if (insertedCopyNodes.count(adapterKey))
                        {
                            currentPid = insertedCopyNodes[adapterKey];
                            compiled.refCounts[currentPid]++;
                        }
                        else
                        {
                            TensorNode adapterNode;
                            adapterNode.id = graph.allocateId();
                            adapterNode.opType = adapter.opType;
                            adapterNode.opName = "";
                            adapterNode.dtype = graph.nodes[chosenPid].dtype;
                            adapterNode.shape = graph.nodes[chosenPid].shape;
                            adapterNode.parentIds = {currentPid};
                            adapterNode.backend = adapter.backend;
                            adapterNode.storageType = StorageType::TRANSIENT;

                            // Inherit view from currentPid, modify if CONTIGUOUS
                            adapterNode.view = graph.nodes[currentPid].view;
                            if (adapter.opType == OpType::CONTIGUOUS)
                            {
                                adapterNode.view.strides = TensorView::calcContiguousStrides(adapterNode.shape);
                                adapterNode.view.baseOffset = 0; // New allocation
                            }
                            else if (adapter.opType == OpType::COPY_TO)
                            {
                                adapterNode.view.baseOffset = 0; // New allocation
                            }

                            if (adapterNode.id >= graph.nodes.size())
                            {
                                graph.nodes.resize(adapterNode.id + 1);
                            }
                            graph.nodes[adapterNode.id] = adapterNode;

                            insertedCopyNodes[adapterKey] = adapterNode.id;
                            finalTopoWithCopies.push_back(adapterNode.id);

                            compiled.refCounts[adapterNode.id]++;
                            compiled.refCounts[currentPid]++;

                            compiled.nodesMap[adapterNode.id] = adapterNode;

                            std::string adapterHash = Hashing::structuralHash(adapterNode.id, graph, structHashMemo);
                            uint32_t adapterHashId = getHashId(adapterHash);
                            bestKernelAssignments[adapterHashId] = adapter.kernelId;
                            bestAssignments[adapterHashId] = adapter.backend;

                            currentPid = adapterNode.id;
                        }
                    }
                    mappedParentIds.push_back(currentPid);
                }
                else
                {
                    mappedParentIds.push_back(chosenPid);
                    compiled.refCounts[chosenPid]++;
                }
            }

            mappedNode.parentIds = mappedParentIds;
            compiled.nodesMap[id] = mappedNode;
            finalTopoWithCopies.push_back(id);
        }
        compiled.refCounts[bestRecipe->nodeId] = 1;

        uint32_t instIdx = 0;
        for (uint32_t id : finalTopoWithCopies)
        {
            instIdx++;
            std::cout << "Inst: " << instIdx << "/" << finalTopoWithCopies.size() << "\r";
            const auto &node = compiled.nodesMap.at(id);
            if (node.opType == OpType::INPUT)
                continue;

            std::string h = Hashing::structuralHash(id, graph, structHashMemo);
            uint32_t hId = getHashId(h);
            Backend assignedBackend = node.backend;
            if (bestAssignments.count(hId))
            {
                assignedBackend = bestAssignments.at(hId);
            }

            uint64_t finalKernelId = bestKernelAssignments.count(hId) ? bestKernelAssignments.at(hId) : UINT64_MAX;
            if (finalKernelId == UINT64_MAX)
            {
                throw std::runtime_error("[Planner.plan] CRITICAL: Missing kernel assignment for node " + std::to_string(id));
            }

            const KernelEntry &kEntry = KernelRegistry::get().getKernel(finalKernelId);

            if (kEntry.inplace)
            {
                uint32_t input0Id = compiled.nodesMap.at(id).parentIds[0];
                if (compiled.refCounts[input0Id] > 1)
                {
                    throw std::runtime_error("[Planner.plan] CRITICAL: Planned inplace kernel but refCount > 1 for node " + std::to_string(input0Id));
                }
            }

            OpInstruction inst;
            inst.nodeId = id;
            inst.kernelId = finalKernelId;
            inst.inputNodeIds = compiled.nodesMap.at(id).parentIds;
            inst.backend = assignedBackend;
            inst.inplaceInputIndex = kEntry.inplace ? 0 : -1; // TODO: this is simplified: inplace kernels assume input 0 is the target

            compiled.instructions.push_back(inst);
        }

        return compiled;
    }

private:
    CostModel &costModel;
    uint64_t maxMemoryBytes;
    size_t beamWidth = 3;

    void inferShapes(const std::vector<uint32_t> &topo, Graph &graph)
    {
        ShapePropagator propagator;
        for (uint32_t nodeId : topo)
        {
            propagator.inferShape(nodeId, graph);
            if (graph.nodes[nodeId].view.shape.empty() && !graph.nodes[nodeId].shape.empty())
            {
                graph.nodes[nodeId].view.shape = graph.nodes[nodeId].shape;
                graph.nodes[nodeId].view.strides = TensorView::calcContiguousStrides(graph.nodes[nodeId].shape);
            }
        }
    }

    bool matchPattern(uint32_t concreteId, const Graph &mainGraph,
                      uint32_t patternId, const Graph &patternGraph,
                      const std::vector<uint32_t> &patternVariables,
                      std::unordered_map<uint32_t, uint32_t> &binding)
    {
        if (std::find(patternVariables.begin(), patternVariables.end(), patternId) != patternVariables.end())
        {
            if (binding.count(patternId))
            {
                return binding[patternId] == concreteId;
            }
            else
            {
                binding[patternId] = concreteId;
                return true;
            }
        }

        const auto &cNode = mainGraph.nodes[concreteId];
        const auto &pNode = patternGraph.nodes[patternId];

        if (cNode.opType != pNode.opType)
            return false;
        if (cNode.opType == OpType::FUSED && cNode.opName != pNode.opName)
            return false;
        if (cNode.parentIds.size() != pNode.parentIds.size())
            return false;

        for (size_t i = 0; i < cNode.parentIds.size(); ++i)
        {
            if (!matchPattern(cNode.parentIds[i], mainGraph, pNode.parentIds[i], patternGraph, patternVariables, binding))
            {
                return false;
            }
        }

        return true;
    }

    std::vector<uint32_t> topologicalSort(uint32_t rootId, const Graph &graph)
    {
        std::vector<uint32_t> order;
        std::unordered_set<uint32_t> visited;
        auto visit = [&](auto &self, uint32_t node) -> void
        {
            if (visited.count(node))
                return;
            visited.insert(node);
            for (uint32_t pid : graph.nodes[node].parentIds)
            {
                self(self, pid);
            }
            order.push_back(node);
        };
        visit(visit, rootId);
        return order;
    }

    std::vector<uint32_t> getAugmentedTopologicalSort(const std::vector<uint32_t> &baseNodes,
                                                      std::unordered_map<std::string, std::vector<uint32_t>> &fusionMap,
                                                      const Graph &graph,
                                                      std::unordered_map<uint32_t, std::string> &structHashMemo)
    {
        std::vector<uint32_t> order;
        std::unordered_set<uint32_t> visited;
        auto visit = [&](auto &self, uint32_t node) -> void
        {
            if (visited.count(node))
                return;
            visited.insert(node);

            for (uint32_t pid : graph.nodes[node].parentIds)
            {
                self(self, pid);
            }

            std::string hash = Hashing::structuralHash(node, graph, structHashMemo);
            if (fusionMap.count(hash))
            {
                for (uint32_t altNode : fusionMap[hash])
                {
                    for (uint32_t pid : graph.nodes[altNode].parentIds)
                    {
                        self(self, pid);
                    }
                }
            }
            order.push_back(node);
        };

        for (uint32_t node : baseNodes)
        {
            visit(visit, node);
        }
        return order;
    }

    void planNodeIterative(uint32_t nodeId, const Graph &graph,
                           std::unordered_map<std::string, std::vector<uint32_t>> &fusionMap,
                           std::unordered_map<std::string, std::vector<std::shared_ptr<BeamStrategy>>> &memo,
                           std::unordered_map<uint32_t, std::string> &structHashMemo,
                           const std::unordered_map<uint32_t, uint32_t> &estimatedRefCounts)
    {
        std::string nodeHash = Hashing::structuralHash(nodeId, graph, structHashMemo);
        if (memo.count(nodeHash))
            return;

        const auto &node = graph.nodes[nodeId];
        uint32_t nodeHashId = getHashId(nodeHash);

        if (node.opType == OpType::INPUT)
        {
            auto strat = std::make_shared<BeamStrategy>();
            strat->cost = 0.0f;
            strat->nodeId = nodeId;
            strat->selectedNodeId = nodeId;
            strat->backend = node.backend;
            strat->kernelId = UINT64_MAX;
            memo[nodeHash].push_back(strat);
            return;
        }

        std::vector<uint32_t> targets = {nodeId};
        if (fusionMap.count(nodeHash))
        {
            targets.insert(targets.end(), fusionMap[nodeHash].begin(), fusionMap[nodeHash].end());
        }

        // TODO: make a handler or something to get available backends
#ifdef USE_CUDA
        std::vector<Backend> availableBackends = {Backend::CPU, Backend::CUDA};
#else
        std::vector<Backend> availableBackends = {Backend::CPU};
#endif
        std::vector<std::shared_ptr<BeamStrategy>> candidates;

        for (uint32_t targetId : targets)
        {
            const auto &target = graph.nodes[targetId];

            // Check parent planning status
            std::vector<std::vector<std::shared_ptr<BeamStrategy>>> parentBeamSets;
            bool anyParentMissing = false;
            for (uint32_t pid : target.parentIds)
            {
                std::string phash = Hashing::structuralHash(pid, graph, structHashMemo);
                if (memo.count(phash) == 0 || memo[phash].empty())
                {
                    anyParentMissing = true;
                    break;
                }
                parentBeamSets.push_back(memo[phash]);
            }
            if (anyParentMissing)
                continue;

            for (Backend backend : availableBackends)
            {
                if (target.parentIds.empty())
                {
                    std::vector<TensorNode> inputNodes;
                    std::vector<uint64_t> matchingKernels = KernelRegistry::get().findMatchingKernels(
                        target.opType, target.opName, backend, inputNodes, target, &estimatedRefCounts);

                    for (uint64_t kernelId : matchingKernels)
                    {
                        std::string targetHash = Hashing::structuralHash(targetId, graph, structHashMemo);
                        uint32_t targetHashId = getHashId(targetHash);
                        float cost = costModel.estimateCost(target, graph, kernelId);
                        auto strat = std::make_shared<BeamStrategy>();
                        strat->cost = cost;
                        strat->nodeId = nodeId;
                        strat->selectedNodeId = targetId;
                        strat->backend = backend;
                        strat->kernelId = kernelId;

                        candidates.push_back(strat);
                    }
                    continue;
                }

                std::vector<size_t> indices(parentBeamSets.size(), 0);
                while (true)
                {
                    std::vector<std::vector<AdapterChain>> parentAdapterChains(parentBeamSets.size());
                    bool possible = true;
                    for (size_t i = 0; i < parentBeamSets.size(); ++i)
                    {
                        const auto &pStrat = parentBeamSets[i][indices[i]];
                        Backend pBackend = pStrat->backend;

                        parentAdapterChains[i] = getAdapterChains(graph.nodes[pStrat->nodeId], pBackend, backend, graph, estimatedRefCounts, costModel);
                        if (parentAdapterChains[i].empty())
                        {
                            possible = false;
                            break;
                        }
                    }

                    if (possible)
                    {
                        std::vector<size_t> chainIndices(parentAdapterChains.size(), 0);
                        while (true)
                        {
                            std::vector<TensorNode> adaptedInputNodes;

                            for (size_t i = 0; i < parentAdapterChains.size(); ++i)
                            {
                                const auto &chain = parentAdapterChains[i][chainIndices[i]];
                                const auto &pStrat = parentBeamSets[i][indices[i]];

                                TensorNode adaptedNode = graph.nodes[pStrat->nodeId];
                                adaptedNode.backend = chain.finalBackend;
                                if (chain.finalContig)
                                {
                                    adaptedNode.view.strides = TensorView::calcContiguousStrides(adaptedNode.shape);
                                }
                                adaptedInputNodes.push_back(adaptedNode);
                            }

                            std::vector<uint64_t> matchingKernels = KernelRegistry::get().findMatchingKernels(
                                target.opType, target.opName, backend, adaptedInputNodes, target, &estimatedRefCounts);

                            for (uint64_t kernelId : matchingKernels)
                            {
                                float targetCost = costModel.estimateCost(target, graph, kernelId);
                                float totalCost = targetCost;

                                auto strat = std::make_shared<BeamStrategy>();
                                strat->nodeId = nodeId;
                                strat->selectedNodeId = targetId;
                                strat->backend = backend;
                                strat->kernelId = kernelId;

                                for (size_t i = 0; i < parentBeamSets.size(); ++i)
                                {
                                    const auto &pStrat = parentBeamSets[i][indices[i]];
                                    float chainCost = parentAdapterChains[i][chainIndices[i]].cost;

                                    totalCost += pStrat->cost + chainCost;

                                    strat->parentStrategies.push_back(pStrat);
                                    strat->parentAdapters.push_back(parentAdapterChains[i][chainIndices[i]].ops);
                                }

                                strat->cost = totalCost;
                                candidates.push_back(strat);
                            }

                            int c = static_cast<int>(chainIndices.size()) - 1;
                            while (c >= 0)
                            {
                                chainIndices[c]++;
                                if (chainIndices[c] < parentAdapterChains[c].size())
                                    break;
                                chainIndices[c] = 0;
                                c--;
                            }
                            if (c < 0)
                                break;
                        }
                    }

                    int p = static_cast<int>(indices.size()) - 1;
                    while (p >= 0)
                    {
                        indices[p]++;
                        if (indices[p] < parentBeamSets[p].size())
                            break;
                        indices[p] = 0;
                        p--;
                    }
                    if (p < 0)
                        break;
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const std::shared_ptr<BeamStrategy> &a, const std::shared_ptr<BeamStrategy> &b)
                  { return a->cost < b->cost; });
        if (candidates.size() > beamWidth)
        {
            candidates.resize(beamWidth);
        }

        for (int i = static_cast<int>(candidates.size()) - 1; i >= 0; --i)
        {
            if (std::isinf(candidates[i]->cost))
            {
                candidates.erase(candidates.begin() + i);
            }
        }

        if (candidates.empty())
        {
            const auto &node = graph.nodes[nodeId];
            std::stringstream ss;
            ss << "[Planner.planNodeIterative] ERROR: Node " << nodeId
               << " has NO valid strategies (OpType: " << node.opType
               << ", DType: " << node.dtype
               << ", Shape: " << toString(node.shape)
               << ", ParentCount: " << node.parentIds.size() << ")";
            for (int i = 0; i < node.parentIds.size(); i++)
            {
                uint32_t pid = node.parentIds[i];
                ss << ", p" << i << ": " << toString(graph.nodes[pid].shape) << " | " << graph.nodes[pid].dtype;
            }
            ss << std::endl;
            throw std::runtime_error(ss.str());
        }

        memo[nodeHash] = candidates;
    }
};

inline std::unordered_map<uint32_t, std::vector<Region>> propagateDirtyRegions(
    const CompiledGraph &compiled,
    const Graph &graph,
    const std::unordered_map<uint32_t, std::vector<Region>> &inputDirtyRegions)
{
    ShapePropagator propagator;
    std::unordered_map<uint32_t, std::vector<Region>> allRegions(inputDirtyRegions);

    for (const OpInstruction &inst : compiled.instructions)
    {
        const TensorNode &node = graph.nodes[inst.nodeId];

        // Gather parent regions
        std::vector<std::vector<Region>> parentRegions;
        bool anyParentDirty = false;
        for (uint32_t pid : node.parentIds)
        {
            auto it = allRegions.find(pid);
            if (it != allRegions.end() && !it->second.empty())
            {
                parentRegions.push_back(it->second);
                anyParentDirty = true;
            }
            else
            {
                parentRegions.push_back({});
            }
        }

        if (anyParentDirty)
        {
            allRegions[inst.nodeId] = propagator.forward(node, graph, parentRegions);
        }
        else
        {
            allRegions[inst.nodeId] = {};
        }
    }

    return allRegions;
}