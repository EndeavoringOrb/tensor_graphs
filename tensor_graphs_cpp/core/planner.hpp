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
#include <iostream>
#include <sstream>
#include <cstring>

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
    std::unordered_map<std::string, Backend> assignments;
    std::unordered_map<std::string, uint32_t> kernelAssignments;

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

    CompiledGraph plan(uint32_t rootId, Graph &graph)
    {
        std::cout << "[Planner.plan] initial sort..." << std::endl;
        std::unordered_map<uint32_t, std::string> structHashMemo;
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

            for (size_t i = 0; i < numInputs; ++i)
            {
                uint32_t inId = pattern.graph.allocateId();
                TensorView view;
                view.shape = {1};
                view.strides = TensorView::calcContiguousStrides({1});
                view.baseOffset = 0;
                pattern.graph.inputWithId(inId, {1}, DType::FLOAT32, view);
                pattern.variables.push_back(inId);
            }

            pattern.rootId = factory(pattern.variables, pattern.graph);

            std::vector<uint32_t> p_topo = topologicalSort(pattern.rootId, pattern.graph);
            inferShapes(p_topo, pattern.graph);

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

        // std::vector<const Rewrite::RewriteRule *> rules = {&cr, &dr, &fr, &ar, &dnr, &nar, &dmr, &dar};
        std::vector<const Rewrite::RewriteRule *> rules = {}; // TODO: remove this line, this is just so the matching phase is fast while debugging planning phase

        std::cout << "[Planner.plan] matching fusion patterns..." << std::endl;
        uint32_t topoIdx = 0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            topoIdx++;
            std::cout << topoIdx << "/" << topo.size() << "\r";
            uint32_t nodeId = *it;
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
                        }
                    }
                }
            }
        }

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

            uint32_t assignedKernelId = 0;
            if (bestRecipe.kernelAssignments.count(h))
            {
                assignedKernelId = bestRecipe.kernelAssignments.at(h);
            }
            else
            {
                std::string opStr = (node.opType == OpType::FUSED) ? node.opName : toString(node.opType);
                throw std::runtime_error("No kernel assigned for node with OpType " + opStr + " (ID " + std::to_string(id) + ")");
            }

            bool is_inplace_safe = false;
            if (!node.parentIds.empty())
            {
                uint32_t p0 = node.parentIds[0];
                if (compiled.refCounts[p0] == 1 &&
                    countElements(graph.nodes[p0].shape) == countElements(node.shape) &&
                    getDTypeSize(graph.nodes[p0].dtype) == getDTypeSize(node.dtype))
                {
                    is_inplace_safe = true;
                }
            }

            OpInstruction inst;
            inst.nodeId = id;
            inst.kernelId = assignedKernelId;
            inst.inputNodeIds = node.parentIds;
            inst.backend = assignedBackend;

            const KernelEntry &kEntry = KernelRegistry::get().getKernel(assignedKernelId);
            inst.inplaceInputIndex = (is_inplace_safe && kEntry.supportsInplace) ? 0 : -1;

            compiled.instructions.push_back(inst);
        }

        return compiled;
    }

private:
    CostModel &costModel;
    uint64_t maxMemoryBytes;
    size_t beamWidth = 3;

    std::vector<uint32_t> broadcastShapes(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b)
    {
        int rankA = a.size();
        int rankB = b.size();
        int outRank = std::max(rankA, rankB);
        std::vector<uint32_t> out(outRank);
        for (int i = 0; i < outRank; ++i)
        {
            uint32_t dimA = (i < outRank - rankA) ? 1 : a[i - (outRank - rankA)];
            uint32_t dimB = (i < outRank - rankB) ? 1 : b[i - (outRank - rankB)];
            if (dimA == 1)
                out[i] = dimB;
            else if (dimB == 1)
                out[i] = dimA;
            else if (dimA == dimB)
                out[i] = dimA;
            else
            {
                std::stringstream ss;
                ss << "Cannot broadcast shapes " << toString(a) << " and " << toString(b);
                throw std::runtime_error(ss.str());
            }
        }
        return out;
    }

    std::vector<int32_t> getConstantInt32(uint32_t id, const Graph &graph)
    {
        if (graph.constantStaging.count(id))
        {
            const auto &data = graph.constantStaging.at(id);
            std::vector<int32_t> res(data.size() / sizeof(int32_t));
            std::memcpy(res.data(), data.data(), data.size());
            return res;
        }
        std::stringstream ss;
        ss << "Expected constant for shape inference but not found in staging. Node ID: " << id;
        throw std::runtime_error(ss.str());
    }

    // TODO: move this to ShapePropagator and use ShapePropagator.forward/ShapePropagator.backward
    void inferShapes(const std::vector<uint32_t> &topo, Graph &graph)
    {
        for (uint32_t nodeId : topo)
        {
            TensorNode &node = graph.nodes[nodeId];

            if (!node.shape.empty() && node.opType != OpType::RESHAPE)
            {
                continue;
            }
            if (node.opType == OpType::INPUT)
            { // handles both INPUT and CONSTANT mappings
                continue;
            }

            switch (node.opType)
            {
            case OpType::ADD:
            case OpType::MUL:
            case OpType::DIVIDE:
            case OpType::POWER:
            {
                auto s0 = graph.nodes[node.parentIds[0]].shape;
                auto s1 = graph.nodes[node.parentIds[1]].shape;

                if (s0 != s1)
                {
                    std::stringstream ss;
                    ss << "[Planner.inferShapes] Atomic " << toString(node.opType)
                       << " requires exact shape match. Got " << toString(s0)
                       << " and " << toString(s1) << ". Use explicit repeat/reshape. (Node " << nodeId << ")";
                    throw std::runtime_error(ss.str());
                }
                node.shape = s0;
                break;
            }
            case OpType::DOT:
            {
                const auto &s0 = graph.nodes[node.parentIds[0]].shape;
                const auto &s1 = graph.nodes[node.parentIds[1]].shape;
                size_t r0 = s0.size();
                size_t r1 = s1.size();

                if (r0 != r1)
                {
                    std::stringstream ss;
                    ss << "[Planner.inferShapes] DOT requires equal ranks. Got " << r0 << " and " << r1
                       << ". Implicit broadcasting is disabled; use explicit reshape to align ranks.";
                    throw std::runtime_error(ss.str());
                }

                if (r0 == 2)
                {
                    if (s0[1] != s1[0])
                        throw std::runtime_error("DOT: K-dim mismatch [M,K] @ [K,N]");
                    node.shape = {s0[0], s1[1]};
                }
                else if (r0 == 3)
                {
                    if (s0[0] != s1[0])
                        throw std::runtime_error("DOT: Batch dim mismatch [B,M,K] @ [B,K,N]");
                    if (s0[2] != s1[1])
                        throw std::runtime_error("DOT: K-dim mismatch [B,M,K] @ [B,K,N]");
                    node.shape = {s0[0], s0[1], s1[2]};
                }
                else
                {
                    throw std::runtime_error("DOT: Only Rank 2 and Rank 3 are currently supported in this framework.");
                }
                break;
            }
            case OpType::SIN:
            case OpType::COS:
            case OpType::NEGATE:
            case OpType::CAST:
            case OpType::TRIU:
            case OpType::COPY_TO:
            {
                node.shape = graph.nodes[node.parentIds[0]].shape;
                break;
            }
            case OpType::SUM:
            case OpType::MAX:
            {
                auto s0 = graph.nodes[node.parentIds[0]].shape;
                auto axis_vec = getConstantInt32(node.parentIds[1], graph);
                int32_t axis = axis_vec[0];

                if (axis < 0)
                    axis += s0.size();

                std::vector<uint32_t> new_shape;
                for (size_t i = 0; i < s0.size(); ++i)
                {
                    if (i == (size_t)axis)
                    {
                        new_shape.push_back(1);
                    }
                    else
                    {
                        new_shape.push_back(s0[i]);
                    }
                }
                node.shape = new_shape;
                break;
            }
            case OpType::RESHAPE:
            {
                auto s0 = graph.nodes[node.parentIds[0]].shape;
                auto target_dims = getConstantInt32(node.parentIds[1], graph);
                uint64_t total_vol = countElements(s0);
                uint64_t known_vol = 1;
                for (size_t i = 0; i < target_dims.size(); ++i)
                {
                    if (target_dims[i] != -1)
                    {
                        known_vol *= target_dims[i];
                    }
                }
                std::vector<uint32_t> out_shape(target_dims.size());
                for (size_t i = 0; i < target_dims.size(); ++i)
                {
                    if (target_dims[i] == -1)
                    {
                        out_shape[i] = total_vol / known_vol;
                    }
                    else
                    {
                        out_shape[i] = target_dims[i];
                    }
                }
                node.shape = out_shape;
                break;
            }
            case OpType::PERMUTE:
            {
                auto s0 = graph.nodes[node.parentIds[0]].shape;
                auto dims = getConstantInt32(node.parentIds[1], graph);
                std::vector<uint32_t> out_shape(dims.size());
                for (size_t i = 0; i < dims.size(); ++i)
                {
                    out_shape[i] = s0[dims[i]];
                }
                node.shape = out_shape;
                break;
            }
            case OpType::GATHER:
            {
                auto data_shape = graph.nodes[node.parentIds[0]].shape;
                auto idx_shape = graph.nodes[node.parentIds[1]].shape;
                std::vector<uint32_t> out_shape = idx_shape;
                for (size_t i = 1; i < data_shape.size(); ++i)
                {
                    out_shape.push_back(data_shape[i]);
                }
                node.shape = out_shape;
                break;
            }
            case OpType::CONCAT:
            {
                uint32_t axis_id = node.parentIds.back();
                auto axis_vec = getConstantInt32(axis_id, graph);
                int32_t axis = axis_vec[0];
                auto s0 = graph.nodes[node.parentIds[0]].shape;
                if (axis < 0)
                    axis += s0.size();

                std::vector<uint32_t> out_shape = s0;
                uint32_t total_dim = s0[axis];
                for (size_t i = 1; i < node.parentIds.size() - 1; ++i)
                {
                    auto si = graph.nodes[node.parentIds[i]].shape;
                    total_dim += si[axis];
                }
                out_shape[axis] = total_dim;
                node.shape = out_shape;
                break;
            }
            case OpType::REPEAT:
            {
                auto s0 = graph.nodes[node.parentIds[0]].shape;
                auto repeats = getConstantInt32(node.parentIds[1], graph)[0];
                auto axis = getConstantInt32(node.parentIds[2], graph)[0];
                if (axis < 0)
                    axis += s0.size();
                std::vector<uint32_t> out_shape = s0;
                out_shape[axis] *= repeats;
                node.shape = out_shape;
                break;
            }
            case OpType::FILL:
            {
                auto target_dims = getConstantInt32(node.parentIds[1], graph);
                std::vector<uint32_t> out_shape(target_dims.size());
                for (size_t i = 0; i < target_dims.size(); ++i)
                {
                    out_shape[i] = target_dims[i];
                }
                node.shape = out_shape;
                break;
            }
            case OpType::IM2COL:
            {
                auto s0 = graph.nodes[node.parentIds[0]].shape; // N, C, H, W
                uint32_t k = getConstantInt32(node.parentIds[1], graph)[0];
                uint32_t s = getConstantInt32(node.parentIds[2], graph)[0];
                uint32_t p = getConstantInt32(node.parentIds[3], graph)[0];
                uint32_t H = s0[2];
                uint32_t W = s0[3];
                uint32_t H_out = (H + 2 * p - k) / s + 1;
                uint32_t W_out = (W + 2 * p - k) / s + 1;
                node.shape = {s0[0], s0[1] * k * k, H_out * W_out};
                break;
            }
            case OpType::SLICE:
            {
                auto s0 = graph.nodes[node.parentIds[0]].shape;
                auto starts = getConstantInt32(node.parentIds[1], graph);
                auto ends = getConstantInt32(node.parentIds[2], graph);
                auto steps = getConstantInt32(node.parentIds[3], graph);
                std::vector<uint32_t> out_shape(s0.size());
                for (size_t i = 0; i < s0.size(); ++i)
                {
                    int32_t start = i < starts.size() ? starts[i] : 0;
                    int32_t end = i < ends.size() ? ends[i] : s0[i];
                    int32_t step = i < steps.size() ? steps[i] : 1;
                    if (start < 0)
                        start += s0[i];
                    if (end < 0)
                        end += s0[i];
                    out_shape[i] = std::max(0, (end - start + step - 1) / step);
                }
                node.shape = out_shape;
                break;
            }
            case OpType::ARANGE:
            {
                int32_t start = getConstantInt32(node.parentIds[0], graph)[0];
                int32_t stop = getConstantInt32(node.parentIds[1], graph)[0];
                int32_t step = getConstantInt32(node.parentIds[2], graph)[0];
                node.shape = {(uint32_t)std::max(0, (stop - start + step - 1) / step)};
                break;
            }
            case OpType::FUSED:
            {
                break;
            }
            default:
                break;
            }

            if (node.view.shape.empty() && !node.shape.empty())
            {
                node.view.shape = node.shape;
                node.view.strides = TensorView::calcContiguousStrides(node.shape);
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
        std::unordered_set<std::string> visited;
        auto visit = [&](auto &self, uint32_t node) -> void
        {
            std::string hash = Hashing::detail::structuralHashImpl(node, graph, structHashMemo);
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

            // Check if all parents have been successfully planned
            std::vector<std::vector<BeamStrategy>> parentBeamSets;
            bool anyParentMissing = false;
            for (uint32_t pid : target.parentIds)
            {
                std::string phash = Hashing::detail::structuralHashImpl(pid, graph, structHashMemo);
                if (memo.count(phash) == 0)
                {
                    anyParentMissing = true;
                    const auto &pNode = graph.nodes[pid];
                    break;
                }
                parentBeamSets.push_back(memo[phash]);
            }
            if (anyParentMissing)
                continue;

            // Check if any parent beam sets are empty
            bool anyParentEmpty = false;
            for (size_t i = 0; i < parentBeamSets.size(); ++i)
            {
                if (parentBeamSets[i].empty())
                {
                    const auto &pNode = graph.nodes[target.parentIds[i]];
                    anyParentEmpty = true;
                    break;
                }
            }
            if (anyParentEmpty)
                continue;

            for (Backend backend : availableBackends)
            {
                std::vector<TensorNode> inputNodes;
                for (uint32_t pid : target.parentIds)
                {
                    inputNodes.push_back(graph.nodes[pid]);
                }

                std::string opName = (target.opType == OpType::FUSED) ? target.opName : toString(target.opType);
                std::vector<uint32_t> matchingKernels = KernelRegistry::get().findMatchingKernels(target.opType, target.opName, backend, inputNodes, target);

                if (matchingKernels.empty())
                {
                    continue;
                }

                for (uint32_t kernelId : matchingKernels)
                {
                    std::string targetHash = Hashing::detail::structuralHashImpl(targetId, graph, structHashMemo);

                    if (parentBeamSets.empty())
                    {
                        float cost = costModel.estimateCost(target, graph, kernelId);
                        std::unordered_map<std::string, Backend> assigns;
                        assigns[targetHash] = backend;
                        std::unordered_map<std::string, uint32_t> kAssigns;
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
                        std::unordered_map<std::string, uint32_t> kAssigns;
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
                                cost += 0.05f;
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
