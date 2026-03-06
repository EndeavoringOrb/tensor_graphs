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
        std::cout << std::endl << "# Rewrites: " << fusionMap.size() << std::endl;
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

        std::unordered_map<std::string, std::vector<BeamStrategy>> memo;

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
        std::cout << "best recipe cost: " << bestRecipe.cost << std::endl;

        std::cout << "[Planner.plan] final topo sort..." << std::endl;
        std::vector<uint32_t> finalTopo;
        std::unordered_set<uint32_t> visited;

        auto visit = [&](auto &self, uint32_t currOriginalId) -> void
        {
            std::string h = Hashing::structuralHash(currOriginalId, graph, structHashMemo);

            // Jump to the chosen fused node if one was selected
            uint32_t chosenId = bestRecipe.selectedNodes.count(h) ? bestRecipe.selectedNodes.at(h) : currOriginalId;

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
            if (bestRecipe.assignments.count(h))
            {
                mappedNode.backend = bestRecipe.assignments.at(h);
            }
            Backend assignedBackend = mappedNode.backend;
            std::vector<uint32_t> mappedParentIds;

            for (uint32_t pid : mappedNode.parentIds)
            {
                std::string phash = Hashing::structuralHash(pid, graph, structHashMemo);
                uint32_t chosenPid = bestRecipe.selectedNodes.count(phash) ? bestRecipe.selectedNodes.at(phash) : pid;

                Backend parentBackend = graph.nodes[chosenPid].backend;
                if (bestRecipe.assignments.count(phash))
                {
                    parentBackend = bestRecipe.assignments.at(phash);
                }

                if (parentBackend != assignedBackend)
                {
                    std::string copyKey = std::to_string(chosenPid) + "_" + std::to_string(static_cast<uint32_t>(assignedBackend));
                    if (insertedCopyNodes.count(copyKey))
                    {
                        uint32_t copyId = insertedCopyNodes[copyKey];
                        mappedParentIds.push_back(copyId);
                        compiled.refCounts[copyId]++;
                    }
                    else
                    {
                        TensorNode copyNode;
                        copyNode.id = graph.allocateId();
                        copyNode.opType = OpType::COPY_TO;
                        copyNode.opName = "";
                        copyNode.dtype = graph.nodes[chosenPid].dtype;
                        copyNode.shape = graph.nodes[chosenPid].shape;
                        copyNode.parentIds = {chosenPid};
                        copyNode.backend = assignedBackend;

                        if (copyNode.id >= graph.nodes.size())
                        {
                            graph.nodes.resize(copyNode.id + 1);
                        }
                        graph.nodes[copyNode.id] = copyNode;

                        insertedCopyNodes[copyKey] = copyNode.id;
                        finalTopoWithCopies.push_back(copyNode.id);

                        mappedParentIds.push_back(copyNode.id);
                        compiled.refCounts[copyNode.id]++;
                        compiled.refCounts[chosenPid]++;

                        compiled.nodesMap[copyNode.id] = copyNode;

                        std::string edgeHash = phash + "->" + h;
                        std::string copyHash = Hashing::structuralHash(copyNode.id, graph, structHashMemo);
                        if (bestRecipe.kernelAssignments.count(edgeHash))
                        {
                            bestRecipe.kernelAssignments[copyHash] = bestRecipe.kernelAssignments[edgeHash];
                            bestRecipe.assignments[copyHash] = assignedBackend;
                        }
                        else
                        {
                            throw std::runtime_error("Missing copy kernel assignment for edge " + edgeHash);
                        }
                    }
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
        compiled.refCounts[bestRecipe.nodeId] = 1;

        uint32_t instIdx = 0;
        for (uint32_t id : finalTopoWithCopies)
        {
            instIdx++;
            std::cout << "Inst: " << instIdx << "/" << finalTopoWithCopies.size() << "\r";
            const auto &node = compiled.nodesMap.at(id);
            if (node.opType == OpType::INPUT)
                continue;

            std::string h = Hashing::structuralHash(id, graph, structHashMemo);
            Backend assignedBackend = node.backend;
            if (bestRecipe.assignments.count(h))
            {
                assignedBackend = bestRecipe.assignments.at(h);
            }

            uint64_t finalKernelId = bestRecipe.kernelAssignments.count(h) ? bestRecipe.kernelAssignments.at(h) : UINT64_MAX;
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
                           std::unordered_map<std::string, std::vector<BeamStrategy>> &memo,
                           std::unordered_map<uint32_t, std::string> &structHashMemo,
                           const std::unordered_map<uint32_t, uint32_t> &estimatedRefCounts)
    {
        std::string nodeHash = Hashing::structuralHash(nodeId, graph, structHashMemo);
        if (memo.count(nodeHash))
            return;

        const auto &node = graph.nodes[nodeId];
        if (node.opType == OpType::INPUT)
        {
            BeamStrategy strat;
            strat.cost = 0.0f;
            strat.nodeId = nodeId;
            strat.assignments[nodeHash] = node.backend;
            strat.nodeCosts[nodeHash] = 0.0f;
            strat.selectedNodes[nodeHash] = nodeId;
            memo[nodeHash] = {strat};
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
        std::vector<BeamStrategy> candidates;

        for (uint32_t targetId : targets)
        {
            const auto &target = graph.nodes[targetId];
            if (target.opName != "")
            {
                int a = 5; // for debug breakpoint
            }

            // Check parent planning status
            std::vector<std::vector<BeamStrategy>> parentBeamSets;
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
                std::vector<TensorNode> inputNodes;
                for (uint32_t pid : target.parentIds)
                {
                    inputNodes.push_back(graph.nodes[pid]);
                }

                std::vector<uint64_t> matchingKernels = KernelRegistry::get().findMatchingKernels(
                    target.opType, target.opName, backend, inputNodes, target, &estimatedRefCounts);

                for (uint64_t kernelId : matchingKernels)
                {
                    std::string targetHash = Hashing::structuralHash(targetId, graph, structHashMemo);

                    if (parentBeamSets.empty())
                    {
                        float cost = costModel.estimateCost(target, graph, kernelId);
                        std::unordered_map<std::string, Backend> assigns;
                        assigns[targetHash] = backend;
                        std::unordered_map<std::string, uint64_t> kAssigns; // Changed
                        kAssigns[targetHash] = kernelId;
                        std::unordered_map<std::string, float> nCosts;
                        nCosts[targetHash] = cost;
                        std::unordered_map<std::string, uint32_t> sNodes;
                        sNodes[nodeHash] = targetId;

                        BeamStrategy strat{cost, targetId, assigns, kAssigns, nCosts, sNodes};
                        candidates.push_back(strat);
                        continue;
                    }

                    std::vector<size_t> indices(parentBeamSets.size(), 0);
                    while (true)
                    {
                        std::unordered_map<std::string, Backend> assigns;
                        assigns[targetHash] = backend;
                        std::unordered_map<std::string, uint64_t> kAssigns; // Changed
                        kAssigns[targetHash] = kernelId;
                        std::unordered_map<std::string, float> nCosts;
                        std::unordered_map<std::string, uint32_t> sNodes;
                        sNodes[nodeHash] = targetId;

                        for (size_t i = 0; i < parentBeamSets.size(); i++)
                        {
                            const auto &pStrat = parentBeamSets[i][indices[i]];
                            for (auto &pair : pStrat.assignments)
                            {
                                assigns[pair.first] = pair.second; // pair.first is now a hash string
                            }
                            for (auto &pair : pStrat.kernelAssignments)
                            {
                                kAssigns[pair.first] = pair.second;
                            }
                            for (auto &pair : pStrat.nodeCosts)
                            {
                                nCosts[pair.first] = pair.second;
                            }
                            for (auto &pair : pStrat.selectedNodes)
                            {
                                sNodes[pair.first] = pair.second;
                            }

                            // Check transfer cost using parent's hash
                            // TODO: make this actually good. might need to insert copy node into graph inbetween parent and current node
                            std::string phash = Hashing::structuralHash(pStrat.nodeId, graph, structHashMemo);
                            Backend parentBackend = pStrat.assignments.at(phash);
                            if (parentBackend != backend)
                            {
                                std::string edgeHash = phash + "->" + targetHash;

                                TensorNode copyInNode = graph.nodes[pStrat.nodeId];
                                copyInNode.backend = parentBackend;

                                TensorNode copyOutNode = copyInNode;
                                copyOutNode.id = 0; // dummy
                                copyOutNode.opType = OpType::COPY_TO;
                                copyOutNode.opName = "";
                                copyOutNode.parentIds = {pStrat.nodeId};
                                copyOutNode.backend = backend;

                                std::vector<TensorNode> copyInputs = {copyInNode};
                                std::vector<uint64_t> copyKernels = KernelRegistry::get().findMatchingKernels(
                                    OpType::COPY_TO, "", backend, copyInputs, copyOutNode, &estimatedRefCounts);

                                float bestCopyCost = std::numeric_limits<float>::infinity();
                                uint64_t bestCopyKernelId = UINT64_MAX;
                                for (uint64_t copyKId : copyKernels)
                                {
                                    float copyCost = costModel.estimateCost(copyOutNode, graph, copyKId);
                                    if (copyCost < bestCopyCost)
                                    {
                                        bestCopyCost = copyCost;
                                        bestCopyKernelId = copyKId;
                                    }
                                }
                                nCosts[edgeHash] = bestCopyCost;
                                kAssigns[edgeHash] = bestCopyKernelId;
                            }
                        }

                        nCosts[targetHash] = costModel.estimateCost(target, graph, kernelId);

                        float totalCost = 0.0f;
                        for (const auto &pair : nCosts)
                        {
                            totalCost += pair.second;
                        }

                        candidates.push_back({totalCost, targetId, assigns, kAssigns, nCosts, sNodes});

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
        }

        std::sort(candidates.begin(), candidates.end());
        if (candidates.size() > beamWidth)
        {
            candidates.resize(beamWidth);
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

        for (int i = static_cast<int>(candidates.size()) - 1; i > 0; --i)
        {
            if (std::isinf(candidates[i].cost))
            {
                std::cout << "Erasing inf cand cost at idx " << i << ": " << candidates[i].cost << std::endl;
                candidates.erase(candidates.begin() + i);
            }
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