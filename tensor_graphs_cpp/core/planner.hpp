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
                view.shape = (i < dummyShapes.size()) ? dummyShapes[i] : std::vector<uint32_t>{1};
                view.strides = TensorView::calcContiguousStrides(view.shape);
                view.baseOffset = 0;
                view.dtype = DType::FLOAT32;
                pattern.graph.inputWithId(inId, view.shape, DType::FLOAT32, view);
                pattern.variables.push_back(inId);
            }

            pattern.rootId = factory(pattern.variables, pattern.graph);

            std::vector<uint32_t> p_topo = topologicalSort(pattern.rootId, pattern.graph);
            inferShapes(p_topo, pattern.graph);

            fusedPatterns.push_back(std::move(pattern));
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

        // std::vector<const Rewrite::RewriteRule *> rules = {&cr, &dr, &fr, &ar, &dnr, &nar, &dmr, &dar};
        std::vector<const Rewrite::RewriteRule *> rules = {&cr, &dr};
        // std::vector<const Rewrite::RewriteRule *> rules = {}; // TODO: remove this line, this is just so the matching phase is fast while debugging planning phase

        std::cout << "[Planner.plan] matching fusion patterns..." << std::endl;
        uint32_t topoIdx = 0;
        uint32_t fusionMatches = 0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            topoIdx++;
            std::cout << topoIdx << "/" << topo.size() << ", rewrites: " << fusionMap.size() << ", matches: " << fusionMatches << "\r";
            uint32_t nodeId = *it;
            // std::string patHash = Hashing::detail::patternHashImpl(nodeId, graph, patternHashMemo);
            std::string hash = Hashing::detail::structuralHashImpl(nodeId, graph, structHashMemo);

            std::vector<uint32_t> equivalents = Rewrite::generateAllEquivalents(nodeId, graph, rules);
            for (uint32_t eqId : equivalents)
            {
                if (eqId != nodeId)
                {
                    fusionMap[hash].push_back(eqId);
                }

                for (const auto &fp : fusedPatterns)
                {
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
#endif
        std::cout << "[Planner.plan] doing augmented topo sort..." << std::endl;
        std::vector<uint32_t> sortedNodes = getAugmentedTopologicalSort(topo, fusionMap, graph, structHashMemo);

        std::cout << "[Planner.plan] inferring shapes for augmented graph..." << std::endl;
        inferShapes(sortedNodes, graph);

        std::unordered_map<std::string, std::vector<BeamStrategy>> memo;

        std::cout << "[Planner.plan] planning nodes..." << std::endl;
        uint32_t nodeIdx = 0;
        for (uint32_t nodeId : sortedNodes)
        {
            nodeIdx++;
            std::cout << nodeIdx << "/" << sortedNodes.size() << "\r";
            planNodeIterative(nodeId, graph, fusionMap, memo, structHashMemo);
        }

        std::string rootHash = Hashing::detail::structuralHashImpl(rootId, graph, structHashMemo);
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

        std::cout << "[Planner.plan] final topo sort..." << std::endl;
        std::vector<uint32_t> finalTopo = topologicalSort(bestRecipe.nodeId, graph);
        CompiledGraph compiled;

        uint32_t mapIdx = 0;
        for (uint32_t id : finalTopo)
        {
            mapIdx++;
            std::cout << "Map: " << mapIdx << "/" << finalTopo.size() << "\r";
            compiled.nodesMap[id] = graph.nodes[id];
            for (uint32_t pid : graph.nodes[id].parentIds)
            {
                compiled.refCounts[pid]++;
            }
            compiled.refCounts[bestRecipe.nodeId] = 1;
        }

        uint32_t instIdx = 0;
        for (uint32_t id : finalTopo)
        {
            instIdx++;
            std::cout << "Inst: " << instIdx << "/" << finalTopo.size() << "\r";
            const auto &node = graph.nodes[id];
            if (node.opType == OpType::INPUT)
                continue;

            std::string h = Hashing::detail::structuralHashImpl(id, graph, structHashMemo);
            Backend assignedBackend = node.backend;
            if (bestRecipe.assignments.count(h))
            {
                assignedBackend = bestRecipe.assignments.at(h);
            }

            // Prepare input nodes for querying
            std::vector<TensorNode> inputNodes;
            for (uint32_t pid : node.parentIds)
            {
                inputNodes.push_back(graph.nodes[pid]);
            }

            // Pass refCounts to findMatchingKernels to enforce inplace safety at runtime
            std::vector<uint64_t> validKernels = KernelRegistry::get().findMatchingKernels(
                node.opType, node.opName, assignedBackend, inputNodes, node, &compiled.refCounts);

            uint64_t finalKernelId; // Changed

            uint64_t plannedKernelId = bestRecipe.kernelAssignments.count(h) ? bestRecipe.kernelAssignments.at(h) : UINT64_MAX; // Changed
            bool isPlannedValid = false;
            for (uint64_t kid : validKernels) // Changed
            {
                if (kid == plannedKernelId)
                {
                    isPlannedValid = true;
                    break;
                }
            }

            if (isPlannedValid)
            {
                finalKernelId = plannedKernelId;
            }
            else
            {
                // Fallback: The planned kernel (e.g. inplace) is invalid under current refCounts.
                // Select the first available valid kernel (usually the out-of-place version).
                if (validKernels.empty())
                {
                    throw std::runtime_error("[Planner.plan] CRITICAL: No valid kernel found for node " + std::to_string(id));
                }
                finalKernelId = validKernels[0];
            }

            const KernelEntry &kEntry = KernelRegistry::get().getKernel(finalKernelId);

            OpInstruction inst;
            inst.nodeId = id;
            inst.kernelId = finalKernelId;
            inst.inputNodeIds = node.parentIds;
            inst.backend = assignedBackend;
            inst.inplaceInputIndex = kEntry.inplace ? 0 : -1; // Simplified: Inplace kernels assume input 0 is the target

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

            std::string hash = Hashing::detail::structuralHashImpl(node, graph, structHashMemo);
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
                           std::unordered_map<uint32_t, std::string> &structHashMemo)
    {
        std::string nodeHash = Hashing::detail::structuralHashImpl(nodeId, graph, structHashMemo);
        if (memo.count(nodeHash))
            return;

        const auto &node = graph.nodes[nodeId];
        if (node.opType == OpType::INPUT)
        {
            BeamStrategy strat;
            strat.cost = 0.0f;
            strat.nodeId = nodeId;
            strat.assignments[nodeHash] = node.backend;
            memo[nodeHash] = {strat};
            return;
        }

        std::vector<uint32_t> targets = {nodeId};
        if (fusionMap.count(nodeHash))
        {
            targets.insert(targets.end(), fusionMap[nodeHash].begin(), fusionMap[nodeHash].end());
        }

        std::vector<Backend> availableBackends = {Backend::CPU};
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
                std::string phash = Hashing::detail::structuralHashImpl(pid, graph, structHashMemo);
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

                // Pass nullptr for refCounts during planning phase.
                // We perform optimistic planning for inplace ops (assuming refcount=1).
                // Runtime safety checks will happen during the final instruction build.
                // TODO: pass actual ref counts, passing nullptr is bad.
                std::vector<uint64_t> matchingKernels = KernelRegistry::get().findMatchingKernels(
                    target.opType, target.opName, backend, inputNodes, target, nullptr);

                for (uint64_t kernelId : matchingKernels)
                {
                    const KernelEntry &kEntry = KernelRegistry::get().getKernel(kernelId);
                    std::string targetHash = Hashing::detail::structuralHashImpl(targetId, graph, structHashMemo);

                    if (parentBeamSets.empty())
                    {
                        float cost = costModel.estimateCost(target, graph, kernelId);
                        std::unordered_map<std::string, Backend> assigns;
                        assigns[targetHash] = backend;
                        std::unordered_map<std::string, uint64_t> kAssigns; // Changed
                        kAssigns[targetHash] = kernelId;

                        BeamStrategy strat{cost, targetId, assigns, kAssigns};
                        candidates.push_back(strat);
                        continue;
                    }

                    std::vector<size_t> indices(parentBeamSets.size(), 0);
                    while (true)
                    {
                        float cost = 0.0f;
                        std::unordered_map<std::string, Backend> assigns;
                        assigns[targetHash] = backend;
                        std::unordered_map<std::string, uint64_t> kAssigns; // Changed
                        kAssigns[targetHash] = kernelId;

                        for (size_t i = 0; i < parentBeamSets.size(); i++)
                        {
                            const auto &pStrat = parentBeamSets[i][indices[i]];
                            cost += pStrat.cost;
                            for (auto &pair : pStrat.assignments)
                            {
                                assigns[pair.first] = pair.second; // pair.first is now a hash string
                            }
                            for (auto &pair : pStrat.kernelAssignments)
                            {
                                kAssigns[pair.first] = pair.second;
                            }

                            // Check transfer cost using parent's hash
                            std::string phash = Hashing::detail::structuralHashImpl(pStrat.nodeId, graph, structHashMemo);
                            if (pStrat.assignments.at(phash) != backend)
                            {
                                cost += 0.05f; // TODO: use estimateCost with temp copyTo node
                            }
                        }

                        cost += costModel.estimateCost(target, graph, kernelId);

                        // std::string targetHash = Hashing::detail::structuralHashImpl(targetId, graph, structHashMemo);
                        assigns[targetHash] = backend;
                        kAssigns[targetHash] = kernelId;

                        candidates.push_back({cost, targetId, assigns, kAssigns});

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