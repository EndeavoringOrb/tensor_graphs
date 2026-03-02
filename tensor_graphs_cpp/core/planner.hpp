#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/cost_model.hpp"
#include "core/kernels.hpp"
#include "core/rewrite.hpp"
#include "core/hashing.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>

struct OpInstruction
{
    uint32_t nodeId;
    uint32_t kernelId;
    std::vector<uint32_t> inputNodeIds;
    int32_t inplaceInputIndex; // -1 if not inplace
    Backend backend;
};

struct CompiledGraph
{
    std::vector<OpInstruction> instructions;
    std::unordered_map<uint32_t, uint32_t> refCounts;
    std::unordered_map<uint32_t, TensorNode> nodesMap;
};

struct BeamStrategy
{
    float cost;
    uint32_t nodeId;
    std::unordered_map<uint32_t, Backend> assignments;
    std::unordered_map<uint32_t, uint32_t> kernelAssignments;
    MemoryManager memManager; // Enforce distinct independent copy tracking global simulation

    bool operator<(const BeamStrategy &other) const
    {
        return cost < other.cost;
    }
};

class Planner
{
public:
    Planner(CostModel &costModel, uint64_t maxMemoryBytes = 4ULL * 1024 * 1024 * 1024)
        : costModel(costModel), maxMemoryBytes(maxMemoryBytes) {}

    CompiledGraph plan(uint32_t rootId, Graph &graph, const MemoryManager &rootMemManager)
    {
        std::unordered_map<uint32_t, std::string> structHashMemo;
        std::vector<uint32_t> topo = topologicalSort(rootId, graph);

        // Build fused patterns from the Reference Graph Registry
        struct FusedPattern
        {
            std::string opName;
            std::vector<uint32_t> variables;
            uint32_t rootId;
            Graph graph;
            MemoryManager memManager;
        };
        std::vector<FusedPattern> fusedPatterns;

        for (const auto &pair : ReferenceGraphRegistry::get().getAll())
        {
            const std::string &opName = pair.first;
            size_t numInputs = pair.second.numInputs;
            ReferenceFactory factory = pair.second.factory;

            FusedPattern pattern;
            pattern.opName = opName;
            pattern.memManager.buffers.emplace(Backend::CPU, DeviceBuffer(1024 * 1024));

            for (size_t i = 0; i < numInputs; ++i)
            {
                uint32_t inId = pattern.graph.allocateId();
                pattern.memManager.allocate(Backend::CPU, inId, 4, StorageType::TRANSIENT);
                TensorView view; // TODO: implement MemoryManager.getView
                pattern.graph.inputWithId(inId, {1}, DType::FLOAT32, view);
                pattern.variables.push_back(inId);
            }

            pattern.rootId = factory(pattern.variables, pattern.graph, pattern.memManager);
            fusedPatterns.push_back(std::move(pattern));
        }

        std::unordered_map<std::string, std::vector<uint32_t>> fusionMap;

        Rewrite::CommutativeRule cr;
        Rewrite::DistributiveRule dr;
        Rewrite::FactoringRule fr;
        Rewrite::AssociativeRule ar;
        Rewrite::DoubleNegationRule dnr;
        Rewrite::NegateAddRule nar;
        Rewrite::DivMulRule dmr;
        Rewrite::DivAddRule dar;

        std::vector<const Rewrite::RewriteRule *> rules = {&cr, &dr, &fr, &ar, &dnr, &nar, &dmr, &dar};

        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            uint32_t nodeId = *it;
            std::string hash = Hashing::detail::structuralHashImpl(nodeId, graph, rootMemManager, structHashMemo);

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
                        }
                    }
                }
            }
        }

        std::vector<uint32_t> sortedNodes = getAugmentedTopologicalSort(topo, fusionMap, graph, rootMemManager, structHashMemo);

        std::unordered_map<std::string, std::vector<BeamStrategy>> memo;

        for (uint32_t nodeId : sortedNodes)
        {
            planNodeIterative(nodeId, graph, fusionMap, memo, rootMemManager, structHashMemo);
        }

        std::string rootHash = Hashing::detail::structuralHashImpl(rootId, graph, rootMemManager, structHashMemo);
        if (memo.find(rootHash) == memo.end() || memo[rootHash].empty())
        {
            throw std::runtime_error("Planner failed to find any execution strategy for root node.");
        }

        auto bestRecipe = memo[rootHash][0];

        std::vector<uint32_t> finalTopo = topologicalSort(bestRecipe.nodeId, graph);
        CompiledGraph compiled;

        for (uint32_t id : finalTopo)
        {
            compiled.nodesMap[id] = graph.nodes[id];
            for (uint32_t pid : graph.nodes[id].parentIds)
            {
                compiled.refCounts[pid]++;
            }
            compiled.refCounts[bestRecipe.nodeId] = 1;
        }

        for (uint32_t id : finalTopo)
        {
            const auto &node = graph.nodes[id];
            if (node.opType == OpType::INPUT)
                continue;

            Backend assignedBackend = bestRecipe.assignments.count(id) ? bestRecipe.assignments[id] : node.backend;

            uint32_t assignedKernelId = 0;
            if (bestRecipe.kernelAssignments.count(id))
            {
                assignedKernelId = bestRecipe.kernelAssignments[id];
            }
            else
            {
                std::string opStr = (node.opType == OpType::FUSED) ? node.opName : toString(node.opType);
                throw std::runtime_error("No kernel assigned for node with OpType " + opStr);
            }

            bool is_inplace_safe = false;
            if (!node.parentIds.empty())
            {
                uint32_t p0 = node.parentIds[0];
                if (compiled.refCounts[p0] == 1 && countElements(graph.nodes[p0].shape) == countElements(node.shape))
                {
                    is_inplace_safe = true;
                }
            }

            OpInstruction inst;
            inst.nodeId = id;
            inst.kernelId = assignedKernelId;
            inst.inputNodeIds = node.parentIds;
            inst.backend = assignedBackend;
            inst.inplaceInputIndex = is_inplace_safe ? 0 : -1;

            compiled.instructions.push_back(inst);
        }

        return compiled;
    }

private:
    CostModel &costModel;
    uint64_t maxMemoryBytes;
    size_t beamWidth = 3;

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
                                                      const MemoryManager &rootMemManager,
                                                      std::unordered_map<uint32_t, std::string> &structHashMemo)
    {
        std::vector<uint32_t> order;
        std::unordered_set<std::string> visited;
        auto visit = [&](auto &self, uint32_t node) -> void
        {
            std::string hash = Hashing::detail::structuralHashImpl(node, graph, rootMemManager, structHashMemo);
            if (visited.count(hash))
                return;
            visited.insert(hash);

            for (uint32_t pid : graph.nodes[node].parentIds)
            {
                self(self, pid);
            }

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
                           const MemoryManager &rootMemManager,
                           std::unordered_map<uint32_t, std::string> &structHashMemo)
    {
        std::string nodeHash = Hashing::detail::structuralHashImpl(nodeId, graph, rootMemManager, structHashMemo);
        if (memo.count(nodeHash))
            return;

        const auto &node = graph.nodes[nodeId];
        if (node.opType == OpType::INPUT)
        {
            BeamStrategy strat;
            strat.cost = 0.0f;
            strat.nodeId = nodeId;
            strat.assignments[nodeId] = node.backend;
            strat.memManager = rootMemManager; // Initialize copy explicitly
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

            std::vector<std::vector<BeamStrategy>> parentBeamSets;
            bool missingParents = false;
            for (uint32_t pid : target.parentIds)
            {
                std::string phash = Hashing::detail::structuralHashImpl(pid, graph, rootMemManager, structHashMemo);
                if (memo.count(phash) == 0)
                {
                    missingParents = true;
                    break;
                }
                parentBeamSets.push_back(memo[phash]);
            }
            if (missingParents)
                continue;

            for (Backend backend : availableBackends)
            {
                std::vector<TensorNode> inputNodes;
                for (uint32_t pid : target.parentIds)
                {
                    inputNodes.push_back(graph.nodes[pid]);
                }

                std::vector<uint32_t> matchingKernels = KernelRegistry::get().findMatchingKernels(target.opType, target.opName, backend, inputNodes, target);
                if (matchingKernels.empty())
                    continue;

                for (uint32_t kernelId : matchingKernels)
                {
                    if (parentBeamSets.empty())
                    {
                        float cost = costModel.estimateCost(target, graph, kernelId);
                        std::unordered_map<uint32_t, Backend> assigns;
                        assigns[targetId] = backend;
                        std::unordered_map<uint32_t, uint32_t> kAssigns;
                        kAssigns[targetId] = kernelId;

                        BeamStrategy strat{cost, targetId, assigns, kAssigns, rootMemManager};
                        try
                        {
                            strat.memManager.allocate(backend, targetId, getSizeBytes(target.shape, target.dtype), StorageType::TRANSIENT, 1, cost);
                            candidates.push_back(strat);
                        }
                        catch (MemoryAllocationError)
                        {
                            continue;
                        }

                        continue;
                    }

                    std::vector<size_t> indices(parentBeamSets.size(), 0);
                    while (true)
                    {
                        float cost = 0.0f;
                        std::unordered_map<uint32_t, Backend> assigns;
                        assigns[targetId] = backend;
                        std::unordered_map<uint32_t, uint32_t> kAssigns;
                        kAssigns[targetId] = kernelId;

                        for (size_t i = 0; i < parentBeamSets.size(); i++)
                        {
                            const auto &pStrat = parentBeamSets[i][indices[i]];
                            cost += pStrat.cost;
                            for (auto &pair : pStrat.assignments)
                            {
                                assigns[pair.first] = pair.second;
                            }
                            for (auto &pair : pStrat.kernelAssignments)
                            {
                                kAssigns[pair.first] = pair.second;
                            }

                            if (pStrat.assignments.at(pStrat.nodeId) != backend)
                            {
                                cost += 0.05f;
                            }
                        }

                        cost += costModel.estimateCost(target, graph, kernelId);

                        MemoryManager newMem = parentBeamSets[0][indices[0]].memManager;
                        try
                        {
                            newMem.allocate(backend, targetId, getSizeBytes(target.shape, target.dtype), StorageType::TRANSIENT, 1, cost);
                            candidates.push_back({cost, targetId, assigns, kAssigns, newMem});
                        }
                        catch (MemoryAllocationError)
                        {
                            // TODO: If OOM simulation fails, we legitimately discard this sequence path.
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
        }

        std::sort(candidates.begin(), candidates.end());
        if (candidates.size() > beamWidth)
        {
            candidates.resize(beamWidth);
        }

        memo[nodeHash] = candidates;
    }
};