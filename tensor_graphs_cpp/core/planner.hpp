#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/cost_model.hpp"
#include "core/kernels.hpp"
#include "core/rewrite.hpp"
#include "core/hashing.hpp"
#include "core/shapes.hpp"
#include "core/misc.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cstring>

#define FUSE_OPS

void propagateDirtyRegionsAtomic(
    const std::vector<uint32_t> &topo,
    const Graph &graph,
    std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
    std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &dirtyInputRegions // dirtyInputRegions[parentId][outputRegionId]
)
{
    ShapePropagator propagator;

    for (uint32_t nodeId : topo)
    {
        if (nodeId >= graph.nodes.size())
            continue;
        const TensorNode &node = graph.nodes[nodeId];
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

        if (anyParentDirty)
            dirtyOutputRegions[nodeId] = propagator.forward(node, graph, parentRegions);
        else
            dirtyOutputRegions[nodeId] = {};

        dirtyInputRegions[nodeId] = propagator.backward(node, graph, dirtyOutputRegions[nodeId]);
    }
}

class Planner
{
private:
    uint32_t currentGeneration = 0;

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

    std::vector<AdapterChain> getSliceAdapterChains(const TensorNode &pNode, Backend pBackend, Backend targetBackend, const Region &region, const Graph &graph, const std::unordered_map<uint32_t, uint32_t> &refCounts, CostModel &costModel)
    {
        std::vector<AdapterChain> chains;

        std::vector<uint32_t> starts, ends, steps, sliceShape;
        for (size_t i = 0; i < pNode.shape.size(); ++i)
        {
            if (i < region.region.size())
            {
                starts.push_back(region.region[i].start);
                ends.push_back(region.region[i].stop);
                steps.push_back(1);
                sliceShape.push_back(region.region[i].stop - region.region[i].start);
            }
            else
            {
                starts.push_back(0);
                ends.push_back(pNode.shape[i]);
                steps.push_back(1);
                sliceShape.push_back(pNode.shape[i]);
            }
        }

        auto evalKernelMulti = [&](OpType op, Backend opBackend, const std::vector<TensorNode> &inputsList, TensorNode &output, uint64_t &bestK, float &bestCost)
        {
            output.id = 999999;
            output.opType = op;
            output.opName = "";
            output.backend = opBackend;
            output.storageType = StorageType::TRANSIENT;
            if (op == OpType::CONTIGUOUS || op == OpType::SLICE || op == OpType::SCATTER)
            {
                output.view.strides = TensorView::calcContiguousStrides(output.shape);
            }

            std::vector<uint64_t> kernels = KernelRegistry::get().findMatchingKernels(op, "", opBackend, inputsList, output, refCounts);
            bestCost = std::numeric_limits<float>::infinity();
            bestK = UINT64_MAX;
            for (uint64_t k : kernels)
            {
                float c = costModel.estimateCost(output, inputsList, graph, k);
                if (c < bestCost || bestK == UINT64_MAX)
                {
                    bestCost = c;
                    bestK = k;
                }
            }
        };

        TensorNode sliceOut = pNode;
        sliceOut.shape = sliceShape;
        sliceOut.dtype = pNode.dtype;

        TensorNode dStarts, dEnds, dSteps;
        dStarts.dtype = DType::INT32;
        dStarts.shape = {(uint32_t)starts.size()};
        dEnds.dtype = DType::INT32;
        dEnds.shape = {(uint32_t)ends.size()};
        dSteps.dtype = DType::INT32;
        dSteps.shape = {(uint32_t)steps.size()};

        std::vector<TensorNode> sliceInputs = {pNode, dStarts, dEnds, dSteps};

        uint64_t kSlice;
        float cSlice;
        evalKernelMulti(OpType::SLICE, pBackend, sliceInputs, sliceOut, kSlice, cSlice);

        if (kSlice == UINT64_MAX)
            return chains;

        AdapterOp sliceOp = {OpType::SLICE, kSlice, pBackend, true, starts, ends, steps, sliceShape, 0};

        TensorNode contigOut = sliceOut;
        uint64_t kContig;
        float cContig;
        evalKernelMulti(OpType::CONTIGUOUS, pBackend, {sliceOut}, contigOut, kContig, cContig);
        if (kContig == UINT64_MAX)
            return chains;

        AdapterOp contigOp = {OpType::CONTIGUOUS, kContig, pBackend, false, {}, {}, {}, sliceShape, 0};

        if (pBackend == targetBackend)
        {
            chains.push_back({{sliceOp, contigOp}, targetBackend, true, cSlice + cContig});
        }
        else
        {
            TensorNode copyOut = contigOut;
            uint64_t kCopy;
            float cCopy;
            evalKernelMulti(OpType::COPY_TO, targetBackend, {contigOut}, copyOut, kCopy, cCopy);
            if (kCopy != UINT64_MAX)
            {
                AdapterOp copyOp = {OpType::COPY_TO, kCopy, targetBackend, false, {}, {}, {}, sliceShape, 0};
                chains.push_back({{sliceOp, contigOp, copyOp}, targetBackend, true, cSlice + cContig + cCopy});
            }
        }

        return chains;
    }

    std::vector<AdapterChain> getScatterAdapterChains(const TensorNode &updatesNode, Backend updatesBackend, const TensorNode &targetNode, Backend targetBackend, const Region &region, const Graph &graph, const std::unordered_map<uint32_t, uint32_t> &refCounts, CostModel &costModel)
    {
        std::vector<AdapterChain> chains;

        std::vector<uint32_t> starts, ends, steps;
        for (size_t i = 0; i < targetNode.shape.size(); ++i)
        {
            if (i < region.region.size())
            {
                starts.push_back(region.region[i].start);
                ends.push_back(region.region[i].stop);
                steps.push_back(1);
            }
            else
            {
                starts.push_back(0);
                ends.push_back(targetNode.shape[i]);
                steps.push_back(1);
            }
        }

        auto evalKernelMulti = [&](OpType op, Backend opBackend, const std::vector<TensorNode> &inputsList, TensorNode &output, uint64_t &bestK, float &bestCost)
        {
            output.id = 999999;
            output.opType = op;
            output.opName = "";
            output.backend = opBackend;
            output.storageType = StorageType::TRANSIENT;
            if (op == OpType::CONTIGUOUS || op == OpType::SCATTER)
            {
                output.view.strides = TensorView::calcContiguousStrides(output.shape);
            }

            std::vector<uint64_t> kernels = KernelRegistry::get().findMatchingKernels(op, "", opBackend, inputsList, output, refCounts);
            bestCost = std::numeric_limits<float>::infinity();
            bestK = UINT64_MAX;
            for (uint64_t k : kernels)
            {
                float c = costModel.estimateCost(output, inputsList, graph, k);
                if (c < bestCost || bestK == UINT64_MAX)
                {
                    bestCost = c;
                    bestK = k;
                }
            }
        };

        std::vector<AdapterOp> baseOps;
        float baseCost = 0.0f;
        TensorNode currentUpdates = updatesNode;
        currentUpdates.backend = updatesBackend;

        if (updatesBackend != targetBackend)
        {
            TensorNode copyOut = currentUpdates;
            uint64_t kCopy;
            float cCopy;
            evalKernelMulti(OpType::COPY_TO, targetBackend, {currentUpdates}, copyOut, kCopy, cCopy);
            if (kCopy != UINT64_MAX)
            {
                AdapterOp copyOp = {OpType::COPY_TO, kCopy, targetBackend, false, {}, {}, {}, currentUpdates.shape, 0};
                baseOps.push_back(copyOp);
                baseCost += cCopy;
                currentUpdates.backend = targetBackend;
            }
            else
            {
                return chains;
            }
        }

        TensorNode dStarts, dEnds, dSteps;
        dStarts.dtype = DType::INT32;
        dStarts.shape = {(uint32_t)starts.size()};
        dEnds.dtype = DType::INT32;
        dEnds.shape = {(uint32_t)ends.size()};
        dSteps.dtype = DType::INT32;
        dSteps.shape = {(uint32_t)steps.size()};

        TensorNode targetNodeBackend = targetNode;
        targetNodeBackend.backend = targetBackend;

        std::vector<TensorNode> scatterInputs = {targetNodeBackend, currentUpdates, dStarts, dEnds, dSteps};

        TensorNode scatterOut = targetNode;
        scatterOut.backend = targetBackend;

        uint64_t kScatter;
        float cScatter;
        evalKernelMulti(OpType::SCATTER, targetBackend, scatterInputs, scatterOut, kScatter, cScatter);

        if (kScatter != UINT64_MAX)
        {
            AdapterOp scatterOp = {OpType::SCATTER, kScatter, targetBackend, true, starts, ends, steps, targetNode.shape, targetNode.id};
            baseOps.push_back(scatterOp);
            chains.push_back({baseOps, targetBackend, true, baseCost + cScatter});
        }

        return chains;
    }

    std::vector<AdapterChain> getAdapterChains(const TensorNode &pNode, Backend pBackend, Backend targetBackend, const Graph &graph, const std::unordered_map<uint32_t, uint32_t> &refCounts, CostModel &costModel)
    {
        std::vector<AdapterChain> chains;
        bool isContig = pNode.view.isContiguous();

        auto evalKernel = [&](OpType op, Backend opBackend, const TensorNode &input, TensorNode &output, uint64_t &bestK, float &bestCost)
        {
            output = input;
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
            std::vector<uint64_t> kernels = KernelRegistry::get().findMatchingKernels(op, "", opBackend, inputsList, output, refCounts);
            bestCost = std::numeric_limits<float>::infinity();
            bestK = UINT64_MAX;
            for (uint64_t k : kernels)
            {
                float c = costModel.estimateCost(output, inputsList, graph, k);
                if (c < bestCost || bestK == UINT64_MAX)
                {
                    bestCost = c;
                    bestK = k;
                }
            }
        };

        if (pBackend == targetBackend)
        {
            chains.push_back({{}, targetBackend, isContig, 0.0f});

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

    float dfsCost(const BeamStrategy *root)
    {
        if (!root || root->visitedGen == currentGeneration)
            return 0.0f;
        root->visitedGen = currentGeneration;

        float total = root->nodeCost + root->edgeCost;
        for (const auto &p : root->parentStrategies)
        {
            total += dfsCost(p.get());
        }
        return total;
    }

    float getDeduplicatedCost(const BeamStrategy *root)
    {
        currentGeneration++;
        return dfsCost(root);
    }

public:
    Planner(CostModel &costModel, uint64_t maxMemoryBytes = 4ULL * 1024 * 1024 * 1024)
        : costModel(costModel), maxMemoryBytes(maxMemoryBytes) {}

    LogicalGraph planLogical(uint32_t rootId, Graph &graph)
    {
        LogicalGraph lg;
        std::cout << "[Planner.plan] initial sort..." << std::endl;
        std::unordered_map<uint32_t, std::string> structHashMemo;
        std::unordered_map<uint32_t, std::string> patternHashMemo;
        std::vector<uint32_t> topo = topologicalSort(rootId, graph);

        std::cout << "[Planner.plan] inferring shapes..." << std::endl;
        inferShapes(topo, graph);

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
                view.shape = dummyShapes[i];
                view.strides = TensorView::calcContiguousStrides(view.shape);
                view.baseOffset = 0;
                DType dtype = pair.second.dtypes[i];
                view.dtype = dtype;
                pattern.graph.inputWithId(inId, view.shape, dtype, view);
                pattern.variables.push_back(inId);
            }

            pattern.rootId = factory(pattern.variables, pattern.graph);

            std::vector<uint32_t> p_topo = topologicalSort(pattern.rootId, pattern.graph);

            fusedPatterns.push_back(std::move(pattern));
        }
        std::unordered_map<OpType, std::vector<uint32_t>> patternsByRootOp;
        for (uint32_t i = 0; i < fusedPatterns.size(); i++)
        {
            OpType patternRootOp = fusedPatterns[i].graph.nodes[fusedPatterns[i].rootId].opType;
            patternsByRootOp[patternRootOp].push_back(i);
        }

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

        std::cout << "[Planner.plan] matching fusion patterns..." << std::endl;
        uint32_t topoIdx = 0;
        uint32_t rewrites = 0;
        uint32_t fusionMatches = 0;
        ShapePropagator prop;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            topoIdx++;
            std::cout << topoIdx << "/" << topo.size() << ", rewrites: " << rewrites << ", matches: " << fusionMatches << "\r";
            uint32_t nodeId = *it;
            std::string hash = Hashing::structuralHash(nodeId, graph, structHashMemo);

            std::vector<uint32_t> equivalents = Rewrite::generateAllEquivalents(nodeId, graph, rules, patternHashMemo);
            for (uint32_t eqId : equivalents)
            {
                if (eqId != nodeId)
                {
                    rewrites++;
                    lg.fusionMap[hash].push_back(eqId);
                }

                prop.inferShapeRecursive(eqId, graph);

                OpType eqOpType = graph.nodes[eqId].opType;
                auto patIt = patternsByRootOp.find(eqOpType);
                if (patIt != patternsByRootOp.end())
                {
                    for (const uint32_t fpIdx : patIt->second)
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

                                lg.fusionMap[hash].push_back(fusedNode.id);
                                fusionMatches++;
                            }
                        }
                    }
                }
            }
        }
        std::cout << std::endl
                  << "# Rewrites: " << lg.fusionMap.size() << std::endl;
        std::cout << "# Fusion matches: " << fusionMatches << std::endl;
#endif
        std::cout << "[Planner.plan] doing augmented topo sort..." << std::endl;
        std::vector<uint32_t> sortedNodes = getAugmentedTopologicalSort(topo, lg.fusionMap, graph, structHashMemo);

        std::cout << "[Planner.plan] inferring shapes for augmented graph..." << std::endl;
        inferShapes(sortedNodes, graph);

        std::unordered_map<std::string, uint32_t> canonicalNode;
        std::unordered_map<uint32_t, uint32_t> nodeToCanonical;

        std::unordered_set<uint32_t> topoSet(topo.begin(), topo.end());

        for (uint32_t id : sortedNodes)
        {
            std::string hash = Hashing::structuralHash(id, graph, structHashMemo);
            if (canonicalNode.find(hash) == canonicalNode.end())
            {
                canonicalNode[hash] = id;
            }
            nodeToCanonical[id] = canonicalNode[hash];
        }

        std::unordered_map<uint32_t, uint32_t> canonicalRefCounts;

        for (uint32_t id : topo)
        {
            if (nodeToCanonical[id] == id)
            {
                for (uint32_t pid : graph.nodes[id].parentIds)
                {
                    canonicalRefCounts[nodeToCanonical[pid]]++;
                }
            }
        }

        for (uint32_t id : sortedNodes)
        {
            if (topoSet.count(id) == 0)
            {
                if (nodeToCanonical[id] == id)
                {
                    for (uint32_t pid : graph.nodes[id].parentIds)
                    {
                        if (topoSet.count(pid) == 0)
                        {
                            canonicalRefCounts[nodeToCanonical[pid]]++;
                        }
                    }
                }
            }
        }

        canonicalRefCounts[nodeToCanonical[rootId]]++;

        std::unordered_map<uint32_t, uint32_t> estimatedRefCounts;
        for (uint32_t id : sortedNodes)
        {
            estimatedRefCounts[id] = canonicalRefCounts[nodeToCanonical[id]];
        }

        // TODO: don't copy all these structs, build them on the LogicalGraph

        lg.graph = graph;
        lg.estimatedRefCounts = estimatedRefCounts;
        return lg;
    }

    CompiledGraph planPhysical(uint32_t rootId, LogicalGraph &lg, const std::unordered_map<uint32_t, std::vector<Region>> &atomicDirtyOutputRegions, const std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &atomicDirtyInputRegions)
    {
        Graph &graph = lg.graph;
        std::vector<uint32_t> topo = topologicalSort(rootId, graph);

        std::unordered_map<uint32_t, std::vector<Region>> dirtyOutputRegions = atomicDirtyOutputRegions;
        std::unordered_map<uint32_t, std::vector<std::vector<Region>>> dirtyInputRegions = atomicDirtyInputRegions;
        propagateDirtyRegionsAtomic(topo, graph, dirtyOutputRegions, dirtyInputRegions);
        std::unordered_map<std::string, std::vector<std::shared_ptr<BeamStrategy>>> memo;
        std::unordered_map<uint32_t, std::string> structHashMemo;

        std::cout << "[Planner.plan] planning nodes..." << std::endl;
        std::vector<uint32_t> sortedNodes = getAugmentedTopologicalSort(topo, lg.fusionMap, graph, structHashMemo);

        uint32_t nodeIdx = 0;
        for (uint32_t nodeId : sortedNodes)
        {
            nodeIdx++;
            std::cout << nodeIdx << "/" << sortedNodes.size() << "\r";
            planNodeIterative(nodeId, graph, lg.fusionMap, memo, structHashMemo, lg.estimatedRefCounts, dirtyOutputRegions, dirtyInputRegions);
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

            for (uint32_t pid : rootNode.parentIds)
            {
                const auto &pNode = graph.nodes[pid];
                std::cout << "  Parent " << pid << ": OpType=" << toString(pNode.opType)
                          << ", DType=" << toString(pNode.dtype)
                          << ", Shape=" << toString(pNode.shape) << std::endl;
            }

            Error::throw_err("Planner failed to find any execution strategy for root node.");
        }

        auto bestRecipe = memo[rootHash][0];
        std::cout << "\nbest recipe cost: " << bestRecipe->cost << " ms" << std::endl;

        std::unordered_map<uint32_t, Backend> bestAssignments;
        std::unordered_map<uint32_t, uint64_t> bestKernelAssignments;
        std::unordered_map<uint32_t, uint32_t> bestSelectedNodes;
        std::unordered_map<uint64_t, std::vector<AdapterOp>> bestEdgeAdapters;

        std::unordered_set<uint32_t> visitedStrats;
        CompiledGraph compiled;

        auto reconstruct = [&](auto &self, const std::shared_ptr<BeamStrategy> &strat) -> void
        {
            if (!strat)
                return;
            if (visitedStrats.count(strat->nodeId))
                return;
            visitedStrats.insert(strat->nodeId);

            std::string hLog = Hashing::structuralHash(strat->nodeId, graph, structHashMemo);
            uint32_t hLogId = getHashId(hLog);

            std::string hPhys = Hashing::structuralHash(strat->selectedNodeId, graph, structHashMemo);
            uint32_t hPhysId = getHashId(hPhys);

            bestAssignments[hPhysId] = strat->backend;
            bestKernelAssignments[hPhysId] = strat->kernelId;
            bestSelectedNodes[hLogId] = strat->selectedNodeId;

            compiled.logicalNodeMap[strat->selectedNodeId] = strat->nodeId;
            compiled.nodeCosts[strat->selectedNodeId] = strat->nodeCost;

            for (size_t i = 0; i < strat->parentStrategies.size(); ++i)
            {
                const auto &pStrat = strat->parentStrategies[i];
                if (pStrat)
                {
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

                            adapterNode.view = graph.nodes[currentPid].view;
                            if (adapter.opType == OpType::CONTIGUOUS)
                            {
                                adapterNode.view.strides = TensorView::calcContiguousStrides(adapterNode.shape);
                                adapterNode.view.baseOffset = 0;
                            }
                            else if (adapter.opType == OpType::COPY_TO)
                            {
                                adapterNode.view.baseOffset = 0;
                            }

                            if (adapter.isSliceOrScatter)
                            {
                                adapterNode.shape = adapter.outShape;
                                adapterNode.view.shape = adapter.outShape;
                                adapterNode.view.strides = TensorView::calcContiguousStrides(adapter.outShape);
                                adapterNode.view.baseOffset = 0;

                                uint32_t startsId = graph.allocateId();
                                uint32_t endsId = graph.allocateId();
                                uint32_t stepsId = graph.allocateId();

                                auto createConst = [&](uint32_t id, const std::vector<uint32_t> &vec)
                                {
                                    TensorNode cNode;
                                    cNode.id = id;
                                    cNode.opType = OpType::INPUT;
                                    cNode.storageType = StorageType::PERSISTENT;
                                    cNode.dtype = DType::INT32;
                                    cNode.shape = {(uint32_t)vec.size()};
                                    cNode.backend = adapter.backend;
                                    cNode.view.shape = cNode.shape;
                                    cNode.view.strides = {1};
                                    cNode.view.baseOffset = 0;

                                    if (id >= graph.nodes.size())
                                        graph.nodes.resize(id + 1);
                                    graph.nodes[id] = cNode;
                                    compiled.nodesMap[id] = cNode;

                                    std::vector<uint8_t> bytes(vec.size() * sizeof(uint32_t));
                                    std::memcpy(bytes.data(), vec.data(), bytes.size());
                                    graph.constantStaging[id] = bytes;
                                };

                                createConst(startsId, adapter.sliceStarts);
                                createConst(endsId, adapter.sliceEnds);
                                createConst(stepsId, adapter.sliceSteps);

                                if (adapter.opType == OpType::SLICE)
                                {
                                    adapterNode.parentIds = {currentPid, startsId, endsId, stepsId};
                                }
                                else if (adapter.opType == OpType::SCATTER)
                                {
                                    adapterNode.parentIds = {adapter.scatterTargetId, currentPid, startsId, endsId, stepsId};
                                    compiled.refCounts[adapter.scatterTargetId]++;
                                }
                                compiled.refCounts[startsId]++;
                                compiled.refCounts[endsId]++;
                                compiled.refCounts[stepsId]++;
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
                            compiled.logicalNodeMap[adapterNode.id] = compiled.logicalNodeMap.count(currentPid) ? compiled.logicalNodeMap[currentPid] : currentPid;

                            std::string adapterHash = Hashing::structuralHash(adapterNode.id, graph, structHashMemo);
                            uint32_t adapterHashId = getHashId(adapterHash);
                            bestKernelAssignments[adapterHashId] = adapter.kernelId;
                            bestAssignments[adapterHashId] = adapter.backend;
                            std::vector<TensorNode> adapterInputs = {graph.nodes[currentPid]};
                            compiled.nodeCosts[adapterNode.id] = costModel.estimateCost(adapterNode, adapterInputs, graph, adapter.kernelId);

                            currentPid = adapterNode.id;
                        }
                    }
                    mappedParentIds.push_back(currentPid);
                }
                else
                {
                    uint32_t currentPid = chosenPid;
                    Backend parentBackend = compiled.nodesMap.count(chosenPid) ? compiled.nodesMap.at(chosenPid).backend : graph.nodes[chosenPid].backend;

                    if (parentBackend != mappedNode.backend)
                    {
                        auto fallbackChains = getAdapterChains(compiled.nodesMap.count(chosenPid) ? compiled.nodesMap.at(chosenPid) : graph.nodes[chosenPid], parentBackend, mappedNode.backend, graph, lg.estimatedRefCounts, costModel);
                        if (fallbackChains.empty())
                        {
                            Error::throw_err("No valid adapter chain found during DAG safety fallback");
                        }
                        std::sort(fallbackChains.begin(), fallbackChains.end(), [](const auto &a, const auto &b)
                                  { return a.cost < b.cost; });
                        const auto &bestChain = fallbackChains[0];

                        for (const auto &adapter : bestChain.ops)
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

                                adapterNode.view = compiled.nodesMap.count(currentPid) ? compiled.nodesMap.at(currentPid).view : graph.nodes[currentPid].view;
                                if (adapter.opType == OpType::CONTIGUOUS)
                                {
                                    adapterNode.view.strides = TensorView::calcContiguousStrides(adapterNode.shape);
                                    adapterNode.view.baseOffset = 0;
                                }
                                else if (adapter.opType == OpType::COPY_TO)
                                {
                                    adapterNode.view.baseOffset = 0;
                                }

                                if (adapterNode.id >= graph.nodes.size())
                                    graph.nodes.resize(adapterNode.id + 1);
                                graph.nodes[adapterNode.id] = adapterNode;

                                insertedCopyNodes[adapterKey] = adapterNode.id;
                                finalTopoWithCopies.push_back(adapterNode.id);

                                compiled.refCounts[adapterNode.id]++;
                                compiled.refCounts[currentPid]++;

                                compiled.nodesMap[adapterNode.id] = adapterNode;
                                compiled.logicalNodeMap[adapterNode.id] = compiled.logicalNodeMap.count(currentPid) ? compiled.logicalNodeMap[currentPid] : currentPid;

                                std::string adapterHash = Hashing::structuralHash(adapterNode.id, graph, structHashMemo);
                                uint32_t adapterHashId = getHashId(adapterHash);
                                bestKernelAssignments[adapterHashId] = adapter.kernelId;
                                bestAssignments[adapterHashId] = adapter.backend;

                                std::vector<TensorNode> adapterInputs = {compiled.nodesMap.count(currentPid) ? compiled.nodesMap.at(currentPid) : graph.nodes[currentPid]};
                                compiled.nodeCosts[adapterNode.id] = costModel.estimateCost(adapterNode, adapterInputs, graph, adapter.kernelId);

                                currentPid = adapterNode.id;
                            }
                        }
                    }

                    mappedParentIds.push_back(currentPid);
                    if (currentPid == chosenPid)
                    {
                        compiled.refCounts[chosenPid]++;
                    }
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
                Error::throw_err("[Planner.plan] CRITICAL: Missing kernel assignment for node " + std::to_string(id));
            }

            const KernelEntry &kEntry = KernelRegistry::get().getKernel(finalKernelId);

            // TODO: Long-term fix — defer kernel selection until after adapter
            // insertion so actual refCounts are known, or re-run beam search
            // with corrected refCounts for affected nodes.
            if (kEntry.inplace)
            {
                uint32_t input0Id = compiled.nodesMap.at(id).parentIds[0];
                if (compiled.refCounts[input0Id] > 1)
                {
                    std::vector<TensorNode> fallbackInputs;
                    for (uint32_t pid : compiled.nodesMap.at(id).parentIds)
                    {
                        fallbackInputs.push_back(compiled.nodesMap.at(pid));
                    }
                    std::vector<uint64_t> fallbackKernels = KernelRegistry::get().findMatchingKernels(
                        node.opType, node.opName, assignedBackend, fallbackInputs, node, compiled.refCounts);

                    uint64_t fallbackId = UINT64_MAX;
                    for (uint64_t fk : fallbackKernels)
                    {
                        if (!KernelRegistry::get().getKernel(fk).inplace)
                        {
                            fallbackId = fk;
                            break;
                        }
                    }
                    if (fallbackId == UINT64_MAX)
                    {
                        Error::throw_err("[Planner.plan] CRITICAL: Planned inplace kernel but refCount > 1 for node " + std::to_string(input0Id) + ", and no non-inplace fallback kernel found.\n" + toString(node));
                    }
                    finalKernelId = fallbackId;
                }
            }

            const KernelEntry &kEntryFinal = KernelRegistry::get().getKernel(finalKernelId);

            OpInstruction inst;
            inst.nodeId = id;
            inst.kernelId = finalKernelId;
            inst.inputNodeIds = compiled.nodesMap.at(id).parentIds;
            inst.backend = assignedBackend;
            inst.inplaceInputIndex = kEntryFinal.inplace ? 0 : -1;

            if (kEntryFinal.inplace)
            {
                if (kEntryFinal.inferView)
                {
                    std::vector<TensorNode> inputs;
                    for (uint32_t pid : inst.inputNodeIds)
                        inputs.push_back(compiled.nodesMap.at(pid));
                    compiled.nodesMap[id].view = kEntryFinal.inferView(compiled.nodesMap.at(id), inputs);
                }
                else
                {
                    compiled.nodesMap[id].view = compiled.nodesMap.at(inst.inputNodeIds[inst.inplaceInputIndex]).view;
                }
            }

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
                graph.nodes[nodeId].view.dtype = graph.nodes[nodeId].dtype;
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
                    self(self, altNode);
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
                           const std::unordered_map<uint32_t, uint32_t> &estimatedRefCounts,
                           const std::unordered_map<uint32_t, std::vector<Region>> &atomicDirtyOutputRegions,
                           const std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &atomicDirtyInputRegions)
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
            strat->nodeCost = 0.0f;
            strat->edgeCost = 0.0f;
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

#ifdef USE_CUDA
        std::vector<Backend> availableBackends = {Backend::CPU, Backend::CUDA};
#else
        std::vector<Backend> availableBackends = {Backend::CPU};
#endif
        std::vector<std::shared_ptr<BeamStrategy>> candidates;

        for (uint32_t targetId : targets)
        {
            const auto &target = graph.nodes[targetId];

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
                        target.opType, target.opName, backend, inputNodes, target, estimatedRefCounts);

                    for (uint64_t kernelId : matchingKernels)
                    {
                        float cost = costModel.estimateCost(target, inputNodes, graph, kernelId);

                        auto strat = std::make_shared<BeamStrategy>();
                        strat->cost = cost;
                        strat->nodeCost = cost;
                        strat->edgeCost = 0.0f;
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
                                target.opType, target.opName, backend, adaptedInputNodes, target, estimatedRefCounts);

                            for (uint64_t kernelId : matchingKernels)
                            {
                                float targetCost = costModel.estimateCost(target, adaptedInputNodes, graph, kernelId);

                                auto strat = std::make_shared<BeamStrategy>();
                                strat->nodeId = nodeId;
                                strat->selectedNodeId = targetId;
                                strat->backend = backend;
                                strat->kernelId = kernelId;
                                strat->nodeCost = targetCost;

                                float totalEdgeCost = 0.0f;
                                for (size_t i = 0; i < parentBeamSets.size(); ++i)
                                {
                                    const auto &pStrat = parentBeamSets[i][indices[i]];
                                    float chainCost = parentAdapterChains[i][chainIndices[i]].cost;

                                    totalEdgeCost += chainCost;

                                    strat->parentStrategies.push_back(pStrat);
                                    strat->parentAdapters.push_back(parentAdapterChains[i][chainIndices[i]].ops);
                                }

                                strat->edgeCost = totalEdgeCost;
                                strat->cost = getDeduplicatedCost(strat.get());

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

        for (int i = static_cast<int>(candidates.size()) - 1; i > 0; --i)
        {
            if (std::isinf(candidates[i]->cost))
            {
                candidates.erase(candidates.begin() + i);
            }
        }

        if (candidates.empty())
        {
            return;
        }

        memo[nodeHash] = candidates;
    }
};