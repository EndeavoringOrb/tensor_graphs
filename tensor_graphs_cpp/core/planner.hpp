#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/cost_model.hpp"
#include "core/kernels.hpp"
#include "core/rewrite.hpp"
#include "core/hashing.hpp"
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

void propagateDirtyRegionsAtomic(
    const std::vector<uint32_t> &topo,
    const Graph &graph,
    std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
    std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &dirtyInputRegions)
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
public:
    Planner(CostModel &costModel, uint64_t maxMemoryBytes = 4ULL * 1024 * 1024 * 1024)
        : costModel(costModel), maxMemoryBytes(maxMemoryBytes) {}

    CompiledGraph plan(
        uint32_t rootId,
        Graph &graph,
        const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
        const std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &dirtyInputRegions)
    {
        std::unordered_map<uint32_t, std::vector<uint32_t>> partialNodesMap;
        applyDirtyPrepass(rootId, graph, dirtyOutputRegions, dirtyInputRegions, partialNodesMap);

        std::vector<uint32_t> roots = {rootId};
        for (const auto &pair : partialNodesMap)
        {
            for (uint32_t pn : pair.second)
            {
                roots.push_back(pn);
            }
        }
        std::vector<uint32_t> topo = topologicalSort(roots, graph);
        inferShapes(topo, graph);

        auto refCounts = computeRefCounts(topo, rootId, graph);

        EGraph egraph;
        std::unordered_map<uint32_t, uint32_t> nodeToEClass;
        nodeToEClass.reserve(graph.nodes.size());

        for (uint32_t nodeId : topo)
        {
            TensorNode &node = graph.nodes[nodeId];
            ensureContiguousView(node);
            uint32_t refCount = 0;
            auto rcIt = refCounts.find(nodeId);
            if (rcIt != refCounts.end())
                refCount = rcIt->second;
            uint32_t eclassId = egraph.addEClass(node.shape, node.dtype, refCount, node.view.isContiguous());
            egraph.getEClass(eclassId).backends.insert(node.backend);
            nodeToEClass[nodeId] = eclassId;
        }

        // Seed with reference kernels only
        for (uint32_t nodeId : topo)
        {
            const TensorNode &node = graph.nodes[nodeId];
            uint32_t eclassId = nodeToEClass[nodeId];
            if (node.opType == OpType::INPUT || node.opType == OpType::CONTIGUOUS || node.opType == OpType::SLICE)
            {
                if (node.opType != OpType::INPUT)
                {
                    std::vector<TensorNode> inputs;
                    for (uint32_t pid : node.parentIds)
                        inputs.push_back(graph.nodes[pid]);
                    std::vector<uint64_t> refs = KernelRegistry::get().findMatchingKernels(node.opType, node.opName, node.backend, inputs, node, refCounts, true);
                    if (refs.empty())
                    {
                        Error::throw_err("No reference kernel found for SLICE/CONTIGUOUS node.\n" + toString(node, graph));
                    }
                    for (uint64_t uid : refs)
                    {
                        ENode enode;
                        enode.nodeId = nodeId;
                        enode.kernelUid = uid;
                        enode.opType = node.opType;
                        enode.backend = node.backend;
                        for (uint32_t pid : node.parentIds)
                            enode.children.push_back(nodeToEClass[pid]);
                        egraph.addENode(eclassId, enode);
                    }
                }
                else
                {
                    ENode enode;
                    enode.nodeId = nodeId;
                    enode.kernelUid = 0;
                    enode.opType = node.opType;
                    enode.backend = node.backend;
                    for (uint32_t pid : node.parentIds)
                        enode.children.push_back(nodeToEClass[pid]);
                    egraph.addENode(eclassId, enode);
                }
                continue;
            }

            std::vector<TensorNode> inputs;
            for (uint32_t pid : node.parentIds)
                inputs.push_back(graph.nodes[pid]);

            std::vector<uint64_t> refs = KernelRegistry::get().findMatchingKernels(
                node.opType, node.opName, node.backend, inputs, node, refCounts, true);
            if (refs.empty())
            {
                Error::throw_err("No reference kernel found for node " + toString(node));
            }

            for (uint64_t uid : refs)
            {
                ENode enode;
                enode.nodeId = nodeId;
                enode.kernelUid = uid;
                enode.opType = node.opType;
                enode.opName = node.opName;
                enode.backend = node.backend;
                for (uint32_t pid : node.parentIds)
                    enode.children.push_back(nodeToEClass[pid]);

                egraph.addENode(eclassId, enode);
            }
        }

        saturate(topo, graph, egraph, nodeToEClass, refCounts);

        auto extraction = extractBest(roots, graph, egraph, nodeToEClass, refCounts);

        return buildCompiledGraph(rootId, graph, egraph, nodeToEClass, refCounts,
                                  extraction, dirtyOutputRegions, dirtyInputRegions, partialNodesMap);
    }

private:
    struct ExtractChoice
    {
        uint32_t enodeId = 0;
        float cost = std::numeric_limits<float>::infinity();
        bool valid = false;
        TensorView view;
    };

    struct ExtractionResult
    {
        std::unordered_map<uint32_t, ExtractChoice> choiceByEClass;
        std::unordered_map<uint32_t, uint32_t> eclassToNodeId;
    };

    CostModel &costModel;
    uint64_t maxMemoryBytes;

    static void ensureContiguousView(TensorNode &node)
    {
        if (node.view.shape.empty() && !node.shape.empty())
        {
            node.view.shape = node.shape;
            node.view.strides = TensorView::calcContiguousStrides(node.shape);
            node.view.dtype = node.dtype;
            node.view.baseOffset = 0;
        }
    }

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

    std::unordered_map<uint32_t, uint32_t> computeRefCounts(const std::vector<uint32_t> &topo, uint32_t rootId, const Graph &graph) const
    {
        std::unordered_map<uint32_t, uint32_t> refCounts;
        for (uint32_t nodeId : topo)
        {
            for (uint32_t pid : graph.nodes[nodeId].parentIds)
            {
                refCounts[pid]++;
            }
        }
        refCounts[rootId] = std::max<uint32_t>(1, refCounts[rootId]);
        return refCounts;
    }

    std::vector<uint32_t> topologicalSort(const std::vector<uint32_t> &roots, const Graph &graph) const
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
        for (uint32_t root : roots)
        {
            visit(visit, root);
        }
        return order;
    }

    struct RuleMatch
    {
        uint32_t nodeId = 0;
    };

    struct Rule
    {
        virtual ~Rule() = default;
        virtual bool match(uint32_t nodeId, const Graph &graph, RuleMatch &out) const = 0;
        virtual bool predicate(const RuleMatch &m, const Graph &graph, const EGraph &egraph,
                               const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                               const std::unordered_map<uint32_t, uint32_t> &refCounts) const
        {
            (void)m;
            (void)graph;
            (void)egraph;
            (void)nodeToEClass;
            (void)refCounts;
            return true;
        }
        virtual std::vector<uint32_t> apply(const RuleMatch &m, Graph &graph) const = 0;
    };

    struct GraphRewriteRuleAdapter : public Rule
    {
        const Rewrite::RewriteRule *rule;
        explicit GraphRewriteRuleAdapter(const Rewrite::RewriteRule *r) : rule(r) {}

        bool match(uint32_t nodeId, const Graph &graph, RuleMatch &out) const override
        {
            (void)graph;
            out.nodeId = nodeId;
            return true;
        }

        std::vector<uint32_t> apply(const RuleMatch &m, Graph &graph) const override
        {
            return rule->apply(m.nodeId, graph);
        }
    };

    struct CopyElimRule : public Rule
    {
        bool match(uint32_t nodeId, const Graph &graph, RuleMatch &out) const override
        {
            if (nodeId >= graph.nodes.size())
                return false;
            const auto &node = graph.nodes[nodeId];
            if (node.opType != OpType::COPY_TO || node.parentIds.empty())
                return false;
            uint32_t parentId = node.parentIds[0];
            if (parentId >= graph.nodes.size())
                return false;
            const auto &parent = graph.nodes[parentId];
            if (parent.opType != OpType::COPY_TO || parent.parentIds.empty())
                return false;
            out.nodeId = nodeId;
            return true;
        }

        std::vector<uint32_t> apply(const RuleMatch &m, Graph &graph) const override
        {
            const TensorNode &node = graph.nodes[m.nodeId];
            const TensorNode &parent = graph.nodes[node.parentIds[0]];
            uint32_t grandparentId = parent.parentIds[0];

            std::vector<uint32_t> results;
            // E.g. copyto(copyto(X, GPU), CPU) => X
            if (node.backend == graph.nodes[grandparentId].backend)
            {
                results.push_back(grandparentId);
            }
            // E.g. copyto(copyto(X, GPU), GPU) => copyto(X, GPU)
            if (node.backend == parent.backend)
            {
                results.push_back(node.parentIds[0]);
            }
            return results;
        }
    };

    struct FusionRule : public Rule
    {
        struct Pattern
        {
            std::string opName;
            uint32_t rootId;
            std::vector<uint32_t> variables;
            std::vector<DType> dtypes;
            std::vector<std::vector<uint32_t>> dummyShapes;
            Graph graph;
        };

        std::vector<Pattern> patterns;

        FusionRule()
        {
            const auto &refGraphs = ReferenceGraphRegistry::get().getAll();
            for (const auto &pair : refGraphs)
            {
                Pattern pattern;
                pattern.opName = pair.first;
                const auto &entry = pair.second;

                for (size_t i = 0; i < entry.numInputs; ++i)
                {
                    TensorNode &node = pattern.graph.allocateNode();
                    uint32_t inId = node.id;
                    TensorView view;
                    view.shape = entry.dummyShapes[i];
                    view.strides = TensorView::calcContiguousStrides(view.shape);
                    view.baseOffset = 0;
                    view.dtype = entry.dtypes[i];
                    pattern.graph.inputWithId(inId, view.shape, view.dtype, view);
                    pattern.variables.push_back(inId);
                }
                pattern.rootId = entry.factory(pattern.variables, pattern.graph);
                pattern.dtypes = entry.dtypes;
                pattern.dummyShapes = entry.dummyShapes;
                patterns.push_back(std::move(pattern));
            }
        }

        bool match(uint32_t nodeId, const Graph &graph, RuleMatch &out) const override
        {
            if (nodeId >= graph.nodes.size())
                return false;
            if (graph.nodes[nodeId].opType == OpType::INPUT)
                return false;
            out.nodeId = nodeId;
            return true;
        }

        std::vector<uint32_t> apply(const RuleMatch &m, Graph &graph) const override
        {
            std::vector<uint32_t> results;
            const TensorNode &refNode = graph.nodes[m.nodeId];

            for (const auto &pattern : patterns)
            {
                std::unordered_map<uint32_t, uint32_t> binding;
                if (matchPattern(m.nodeId, graph, pattern.rootId, pattern.graph, pattern.variables, binding, pattern.dtypes))
                {
                    std::vector<uint32_t> inputs;
                    inputs.reserve(pattern.variables.size());
                    for (uint32_t var : pattern.variables)
                        inputs.push_back(binding[var]);

                    for (const auto &kernel : KernelRegistry::get().getAllKernels())
                    {
                        if (kernel.opType == OpType::FUSED && kernel.opName == pattern.opName)
                        {
                            uint32_t newNode = addFusedNode(graph, kernel, inputs, refNode);
                            if (newNode != UINT32_MAX)
                                results.push_back(newNode);
                        }
                    }
                }
            }
            return results;
        }

        uint32_t addFusedNode(Graph &graph, const KernelEntry &kernel, const std::vector<uint32_t> &parentIds, const TensorNode &refNode) const
        {
            std::vector<uint32_t> adaptedParents;
            for (size_t i = 0; i < parentIds.size(); ++i)
            {
                uint32_t pid = parentIds[i];
                const TensorNode &parent = graph.nodes[pid];

                bool needCopy = (parent.backend != kernel.backend); // TODO: update once KernelEntry stores expected per-input backend
                bool needContig = false;
                if (i < kernel.requiresContiguous.size())
                {
                    needContig = kernel.requiresContiguous[i] && !parent.view.isContiguous();
                }

                if (!needCopy && !needContig)
                {
                    adaptedParents.push_back(pid);
                    continue;
                }

                uint32_t currentId = pid;
                if (needCopy && needContig)
                {
                    TensorNode dummyCopyOut = parent;
                    dummyCopyOut.backend = kernel.backend;
                    bool copyWorks = !KernelRegistry::get().findMatchingKernels(OpType::COPY_TO, "", kernel.backend, {parent}, dummyCopyOut, {}, false).empty();

                    TensorNode dummyContigOut = dummyCopyOut;
                    dummyContigOut.view.strides = TensorView::calcContiguousStrides(dummyContigOut.shape);
                    bool contigWorksAfterCopy = !KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", kernel.backend, {dummyCopyOut}, dummyContigOut, {}, false).empty();

                    if (copyWorks && contigWorksAfterCopy)
                    {
                        currentId = graph.copyto(currentId, kernel.backend);
                        currentId = graph.contiguous(currentId);
                    }
                    else
                    {
                        TensorNode dummyContigOut2 = parent;
                        dummyContigOut2.view.strides = TensorView::calcContiguousStrides(dummyContigOut2.shape);
                        bool contigWorks = !KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", parent.backend, {parent}, dummyContigOut2, {}, false).empty();

                        TensorNode dummyCopyOut2 = dummyContigOut2;
                        dummyCopyOut2.backend = kernel.backend;
                        bool copyWorksAfterContig = !KernelRegistry::get().findMatchingKernels(OpType::COPY_TO, "", kernel.backend, {dummyContigOut2}, dummyCopyOut2, {}, false).empty();

                        if (contigWorks && copyWorksAfterContig)
                        {
                            currentId = graph.contiguous(currentId);
                            currentId = graph.copyto(currentId, kernel.backend);
                        }
                        else
                        {
                            Error::throw_err("No valid adapter chain for copyto and contiguous");
                        }
                    }
                }
                else if (needCopy)
                {
                    currentId = graph.copyto(currentId, kernel.backend);
                }
                else if (needContig)
                {
                    currentId = graph.contiguous(currentId);
                }
                adaptedParents.push_back(currentId);
            }

            TensorNode &node = graph.allocateNode();
            uint32_t id = node.id;
            node.opType = kernel.opType;
            node.opName = kernel.opName;
            node.dtype = refNode.dtype;
            node.shape = refNode.shape;
            node.parentIds = adaptedParents;
            node.backend = kernel.backend;

            if (node.backend != refNode.backend)
            {
                id = graph.copyto(id, refNode.backend);
            }

            return id;
        }

        static bool matchPattern(uint32_t concreteId, const Graph &mainGraph,
                                 uint32_t patternId, const Graph &patternGraph,
                                 const std::vector<uint32_t> &patternVariables,
                                 std::unordered_map<uint32_t, uint32_t> &binding,
                                 const std::vector<DType> &patternDtypes)
        {
            auto itVar = std::find(patternVariables.begin(), patternVariables.end(), patternId);
            if (itVar != patternVariables.end())
            {
                size_t varIdx = static_cast<size_t>(std::distance(patternVariables.begin(), itVar));
                const TensorNode &cNode = mainGraph.nodes[concreteId];
                if (varIdx < patternDtypes.size() && cNode.dtype != patternDtypes[varIdx])
                    return false;

                if (binding.count(patternId))
                {
                    return binding[patternId] == concreteId;
                }
                binding[patternId] = concreteId;
                return true;
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
                if (!matchPattern(cNode.parentIds[i], mainGraph, pNode.parentIds[i], patternGraph, patternVariables, binding, patternDtypes))
                {
                    return false;
                }
            }

            return true;
        }
    };

    // TODO: insert scatter or concat on outgoing side so we don't need to do any slicing in executor
    void applyDirtyPrepass(
        uint32_t rootId,
        Graph &graph,
        const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
        const std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &dirtyInputRegions,
        std::unordered_map<uint32_t, std::vector<uint32_t>> &partialNodesMap)
    {
        std::vector<uint32_t> topo = topologicalSort({rootId}, graph);
        for (uint32_t nodeId : topo)
        {
            if (graph.nodes[nodeId].opType == OpType::INPUT)
                continue;

            auto dirtyIt = dirtyOutputRegions.find(nodeId);
            if (dirtyIt != dirtyOutputRegions.end() && !dirtyIt->second.empty())
            {
                const std::vector<Region> &regions = dirtyIt->second;
                std::vector<uint32_t> partialNodes;
                for (size_t rIdx = 0; rIdx < regions.size(); ++rIdx)
                {
                    const Region &outReg = regions[rIdx];
                    bool isFullRegion = true;
                    for (size_t d = 0; d < outReg.region.size(); ++d)
                    {
                        if (outReg.region[d].start != 0 || outReg.region[d].stop != graph.nodes[nodeId].shape[d])
                        {
                            isFullRegion = false;
                            break;
                        }
                    }
                    if (isFullRegion)
                    {
                        partialNodes.push_back(nodeId);
                        continue;
                    }

                    std::vector<uint32_t> partialInputs;
                    auto slicesIt = dirtyInputRegions.find(nodeId);
                    for (size_t pIdx = 0; pIdx < graph.nodes[nodeId].parentIds.size(); ++pIdx)
                    {
                        uint32_t inId = graph.nodes[nodeId].parentIds[pIdx];
                        if (slicesIt != dirtyInputRegions.end() && pIdx < slicesIt->second.size() &&
                            rIdx < slicesIt->second[pIdx].size() && !slicesIt->second[pIdx][rIdx].empty())
                        {
                            const Region &inReg = slicesIt->second[pIdx][rIdx];
                            bool isFullInput = true;
                            if (inReg.region.size() == graph.nodes[inId].shape.size())
                            {
                                for (size_t d = 0; d < inReg.region.size(); ++d)
                                {
                                    if (inReg.region[d].start != 0 || inReg.region[d].stop != graph.nodes[inId].shape[d])
                                    {
                                        isFullInput = false;
                                        break;
                                    }
                                }
                            }
                            else
                            {
                                isFullInput = false;
                            }

                            if (isFullInput)
                            {
                                partialInputs.push_back(graph.contiguous(inId));
                            }
                            else
                            {
                                std::vector<int32_t> starts, ends, steps;
                                for (size_t d = 0; d < inReg.region.size(); ++d)
                                {
                                    starts.push_back(inReg.region[d].start);
                                    ends.push_back(inReg.region[d].stop);
                                    steps.push_back(1);
                                }
                                uint32_t startsId = graph.constant({(uint32_t)starts.size()}, starts.data(), DType::INT32);
                                uint32_t endsId = graph.constant({(uint32_t)ends.size()}, ends.data(), DType::INT32);
                                uint32_t stepsId = graph.constant({(uint32_t)steps.size()}, steps.data(), DType::INT32);
                                uint32_t safeInId = graph.contiguous(inId);
                                uint32_t sliceId = graph.slice(safeInId, startsId, endsId, stepsId);
                                uint32_t contigId = graph.contiguous(sliceId);
                                partialInputs.push_back(contigId);
                            }
                        }
                        else
                        {
                            partialInputs.push_back(inId);
                        }
                    }

                    TensorNode &newNode = graph.allocateNode();
                    uint32_t newId = newNode.id;
                    newNode = graph.nodes[nodeId];
                    newNode.id = newId;
                    newNode.parentIds = partialInputs;
                    for (size_t d = 0; d < outReg.region.size(); ++d)
                    {
                        newNode.shape[d] = outReg.region[d].stop - outReg.region[d].start;
                    }

                    // Decouple shape constants for operators that define output shape via an input node
                    if (newNode.opType == OpType::RESHAPE || newNode.opType == OpType::FILL)
                    {
                        if (newNode.parentIds.size() > 1)
                        {
                            std::vector<int32_t> newDims;
                            for (uint32_t d : newNode.shape)
                                newDims.push_back(static_cast<int32_t>(d));
                            uint32_t newShapeId = graph.constant({static_cast<uint32_t>(newDims.size())}, newDims.data(), DType::INT32);
                            newNode.parentIds[1] = newShapeId;
                        }
                    }

                    newNode.view.shape.clear();
                    partialNodes.push_back(newId);
                }
                partialNodesMap[nodeId] = partialNodes;
            }
        }
    }

    void saturate(const std::vector<uint32_t> &topo, Graph &graph, EGraph &egraph,
                  std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                  const std::unordered_map<uint32_t, uint32_t> &refCounts)
    {
        Rewrite::CommutativeRule cr;
        Rewrite::DistributiveRule dr;
        Rewrite::FactoringRule fr;
        Rewrite::AssociativeRule ar;
        Rewrite::DoubleNegationRule dnr;
        Rewrite::NegateAddRule nar;
        Rewrite::DivMulRule dmr;
        Rewrite::DivAddRule dar;

        // TODO: make some sort of rule registry so I don't have to update this every time I add/remove a rule
        std::vector<std::unique_ptr<Rule>> rules;
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&cr));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&dr));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&fr));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&ar));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&dnr));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&nar));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&dmr));
        rules.emplace_back(std::make_unique<GraphRewriteRuleAdapter>(&dar));
        rules.emplace_back(std::make_unique<FusionRule>());
        rules.emplace_back(std::make_unique<CopyElimRule>());

        std::vector<uint32_t> worklist = topo;

        size_t iterations = 0;
        const size_t maxIterations = 3;
        while (!worklist.empty() && iterations < maxIterations)
        {
            iterations++;
            std::vector<uint32_t> nextWorklist;
            nextWorklist.reserve(worklist.size());
            for (uint32_t nodeId : worklist)
            {
                for (const auto &rule : rules)
                {
                    RuleMatch match;
                    if (!rule->match(nodeId, graph, match))
                        continue;
                    if (!rule->predicate(match, graph, egraph, nodeToEClass, refCounts))
                        continue;

                    size_t oldSize = graph.nodes.size();
                    std::vector<uint32_t> newNodes = rule->apply(match, graph);
                    if (newNodes.empty())
                    {
                        continue;
                    }
                    for (size_t i = oldSize; i < graph.nodes.size(); ++i)
                    {
                        uint32_t newId = static_cast<uint32_t>(i);
                        ShapePropagator prop;
                        prop.inferShapeRecursive(newId, graph);
                        ensureContiguousView(graph.nodes[newId]);

                        if (!nodeToEClass.count(newId))
                        {
                            uint32_t refCount = 0;
                            auto rcIt = refCounts.find(newId);
                            if (rcIt != refCounts.end())
                                refCount = rcIt->second;
                            uint32_t eclassId = egraph.addEClass(graph.nodes[newId].shape, graph.nodes[newId].dtype, refCount, graph.nodes[newId].view.isContiguous());
                            nodeToEClass[newId] = eclassId;
                        }

                        addBasicEnode(graph, egraph, nodeToEClass, refCounts, newId);
                        nextWorklist.push_back(newId);
                    }
                    for (uint32_t newId : newNodes)
                    {
                        if (nodeToEClass.count(nodeId) && nodeToEClass.count(newId))
                        {
                            egraph.merge(nodeToEClass[nodeId], nodeToEClass[newId]);
                            nodeToEClass[newId] = egraph.find(nodeToEClass[nodeId]);
                        }
                    }
                }
            }
            egraph.rebuild();
            worklist = std::move(nextWorklist);
        }
    }

    bool addBasicEnode(const Graph &graph, EGraph &egraph,
                       const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                       const std::unordered_map<uint32_t, uint32_t> &refCounts,
                       uint32_t nodeId)
    {
        const TensorNode &node = graph.nodes[nodeId];
        if (node.opType == OpType::INPUT)
            return false;

        std::vector<TensorNode> inputs;
        for (uint32_t pid : node.parentIds)
            inputs.push_back(graph.nodes[pid]);

        std::vector<uint64_t> refs = KernelRegistry::get().findMatchingKernels(
            node.opType, node.opName, node.backend, inputs, node, refCounts, false);
        if (refs.empty())
            return false;

        for (uint64_t uid : refs)
        {
            ENode enode;
            enode.nodeId = nodeId;
            enode.kernelUid = uid;
            enode.opType = node.opType;
            enode.opName = node.opName;
            enode.backend = node.backend;
            for (uint32_t pid : node.parentIds)
                enode.children.push_back(nodeToEClass.at(pid));
            egraph.addENode(nodeToEClass.at(nodeId), enode);
        }
        return true;
    }

    ExtractionResult extractBest(const std::vector<uint32_t> &rootIds, const Graph &graph, EGraph &egraph,
                                 const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
                                 const std::unordered_map<uint32_t, uint32_t> &refCounts)
    {
        ExtractionResult result;

        std::unordered_map<uint32_t, ExtractChoice> &choice = result.choiceByEClass;

        std::function<ExtractChoice(uint32_t)> solve = [&](uint32_t eclassId) -> ExtractChoice
        {
            eclassId = egraph.find(eclassId);
            auto it = choice.find(eclassId);
            if (it != choice.end())
                return it->second;

            const EClass &cls = egraph.getEClass(eclassId);
            ExtractChoice best;

            for (uint32_t enodeId : cls.enodes)
            {
                const ENode &enode = egraph.getENodes()[enodeId]; // TODO: make EGraph::getENode(uint32_t enodeId) function
                if (enode.opType == OpType::INPUT)
                {
                    ExtractChoice c;
                    c.enodeId = enodeId;
                    c.cost = 0.0f;
                    c.valid = true;
                    TensorNode inNode = graph.nodes[enode.nodeId];
                    ensureContiguousView(inNode);
                    c.view = inNode.view;
                    if (c.cost < best.cost)
                        best = c;
                    continue;
                }

                float childrenCost = 0.0f;
                std::vector<TensorNode> inputs;
                inputs.reserve(enode.children.size());
                bool childValid = true;
                for (uint32_t childEClass : enode.children)
                {
                    ExtractChoice childChoice = solve(childEClass);
                    if (!childChoice.valid)
                    {
                        childValid = false;
                        break;
                    }
                    childrenCost += childChoice.cost;
                    const ENode &childEnode = egraph.getENodes()[childChoice.enodeId];
                    TensorNode inNode = graph.nodes[childEnode.nodeId];
                    inNode.backend = childEnode.backend;
                    if (childChoice.view.shape.size() > 0)
                    {
                        inNode.view = childChoice.view;
                    }
                    else
                    {
                        ensureContiguousView(inNode);
                    }
                    inputs.push_back(inNode);
                }
                if (!childValid)
                {
                    continue;
                }

                if (enode.opType != OpType::COPY_TO)
                {
                    bool backendMatch = true;
                    for (const auto &inNode : inputs)
                    {
                        if (inNode.backend != enode.backend)
                        {
                            backendMatch = false;
                            break;
                        }
                    }
                    if (!backendMatch)
                    {
                        continue;
                    }
                }

                TensorNode outNode = graph.nodes[enode.nodeId];
                outNode.backend = enode.backend;
                ensureContiguousView(outNode);

                if (enode.kernelUid == 0)
                    continue;

                const KernelEntry &entry = KernelRegistry::get().getKernel(enode.kernelUid);

                if (entry.inplace)
                {
                    if (entry.inferView)
                    {
                        outNode.view = entry.inferView(outNode, inputs);
                    }
                    else if (!inputs.empty())
                    {
                        outNode.view = inputs[0].view;
                    }
                }

                if (!entry.match(inputs, outNode, refCounts))
                {
                    continue;
                }

                float kernelCost = costModel.estimateCost(outNode, inputs, graph, enode.kernelUid);

                ExtractChoice c;
                c.enodeId = enodeId;
                c.cost = childrenCost + kernelCost;
                c.valid = true;
                c.view = outNode.view;
                if (!best.valid || c.cost < best.cost)
                    best = c;
            }

            if (!best.valid)
            {
                best.cost = std::numeric_limits<float>::infinity();
            }
            choice[eclassId] = best;
            return best;
        };

        for (uint32_t rootId : rootIds)
        {
            auto it = nodeToEClass.find(rootId);
            if (it == nodeToEClass.end())
            {
                Error::throw_err("Root node missing from egraph.");
            }

            solve(it->second);
        }

        for (const auto &kv : choice)
        {
            const ExtractChoice &c = kv.second;
            if (c.valid && c.enodeId < egraph.getENodes().size())
            {
                result.eclassToNodeId[kv.first] = egraph.getENodes()[c.enodeId].nodeId;
            }
        }

        return result;
    }

    CompiledGraph buildCompiledGraph(
        uint32_t rootId,
        Graph &graph,
        EGraph &egraph,
        const std::unordered_map<uint32_t, uint32_t> &nodeToEClass,
        const std::unordered_map<uint32_t, uint32_t> &refCounts,
        const ExtractionResult &extraction,
        const std::unordered_map<uint32_t, std::vector<Region>> &dirtyOutputRegions,
        const std::unordered_map<uint32_t, std::vector<std::vector<Region>>> &dirtyInputRegions,
        const std::unordered_map<uint32_t, std::vector<uint32_t>> &partialNodesMap)
    {
        CompiledGraph compiled;

        std::unordered_map<uint32_t, uint32_t> eclassLogicalId;
        for (const auto &pair : nodeToEClass)
        {
            uint32_t nodeId = pair.first;
            uint32_t eclassId = egraph.find(pair.second);
            auto it = eclassLogicalId.find(eclassId);
            if (it == eclassLogicalId.end())
            {
                eclassLogicalId[eclassId] = nodeId;
            }
            else
            {
                it->second = std::min(it->second, nodeId);
            }
        }

        std::unordered_map<uint32_t, uint32_t> selectedNodeForEClass = extraction.eclassToNodeId;

        auto mapToSelected = [&](uint32_t nodeId)
        {
            auto it = nodeToEClass.find(nodeId);
            if (it == nodeToEClass.end())
                return nodeId;
            uint32_t eclassId = egraph.find(it->second);
            auto sit = selectedNodeForEClass.find(eclassId);
            if (sit != selectedNodeForEClass.end())
                return sit->second;
            return nodeId;
        };

        uint32_t selectedRoot = mapToSelected(rootId);

        std::vector<uint32_t> topo;
        std::unordered_set<uint32_t> visited;
        std::function<void(uint32_t)> visit = [&](uint32_t nodeId)
        {
            if (visited.count(nodeId))
                return;
            visited.insert(nodeId);
            for (uint32_t pid : graph.nodes[nodeId].parentIds)
            {
                visit(mapToSelected(pid));
            }
            topo.push_back(nodeId);
        };
        visit(selectedRoot);

        std::unordered_map<uint32_t, uint32_t> compiledRefCounts;
        for (uint32_t nodeId : topo)
        {
            for (uint32_t pid : graph.nodes[nodeId].parentIds)
            {
                compiledRefCounts[mapToSelected(pid)]++;
            }
        }
        compiledRefCounts[selectedRoot] = std::max<uint32_t>(1, compiledRefCounts[selectedRoot]);

        compiled.refCounts = compiledRefCounts;

        for (uint32_t nodeId : topo)
        {
            compiled.nodesMap[nodeId] = graph.nodes[nodeId];
        }

        for (uint32_t nodeId : topo)
        {
            const TensorNode &node = graph.nodes[nodeId];
            if (node.opType == OpType::INPUT)
                continue;

            uint32_t eclassId = nodeToEClass.at(nodeId);
            auto choiceIt = extraction.choiceByEClass.find(egraph.find(eclassId));
            if (choiceIt == extraction.choiceByEClass.end() || !choiceIt->second.valid)
            {
                Error::throw_err("Missing or invalid extraction choice for node " + std::to_string(nodeId));
            }

            const ExtractChoice &choice = choiceIt->second;
            const ENode &enode = egraph.getENodes()[choice.enodeId];
            const KernelEntry &kEntry = KernelRegistry::get().getKernel(enode.kernelUid);

            OpInstruction inst;
            inst.nodeId = nodeId;
            inst.fullKernelId = enode.kernelUid;
            inst.inputNodeIds.reserve(node.parentIds.size());
            for (uint32_t pid : node.parentIds)
                inst.inputNodeIds.push_back(mapToSelected(pid));
            inst.backend = enode.backend;
            inst.inplaceInputIndex = kEntry.inplace ? 0 : -1;

            compiled.nodesMap[nodeId].backend = enode.backend;
            if (kEntry.inplace)
            {
                std::vector<TensorNode> inplaceInputs;
                inplaceInputs.reserve(inst.inputNodeIds.size());
                for (uint32_t pid : inst.inputNodeIds)
                    inplaceInputs.push_back(compiled.nodesMap.at(pid));

                if (kEntry.inferView)
                {
                    compiled.nodesMap[nodeId].view = kEntry.inferView(compiled.nodesMap[nodeId], inplaceInputs);
                }
                else if (!inplaceInputs.empty())
                {
                    compiled.nodesMap[nodeId].view = inplaceInputs[static_cast<size_t>(inst.inplaceInputIndex)].view;
                }
            }

            uint32_t logicalId = nodeId;
            auto logicalIt = eclassLogicalId.find(egraph.find(nodeToEClass.at(nodeId)));
            if (logicalIt != eclassLogicalId.end())
            {
                logicalId = logicalIt->second;
            }

            auto dirtyIt = dirtyOutputRegions.find(logicalId);
            if (dirtyIt != dirtyOutputRegions.end() && !dirtyIt->second.empty())
            {
                const std::vector<Region> &regions = dirtyIt->second;
                inst.cachedKernelIds.reserve(regions.size());
                for (size_t rIdx = 0; rIdx < regions.size(); ++rIdx)
                {
                    uint32_t partialNodeId = partialNodesMap.at(logicalId)[rIdx];
                    if (partialNodeId == nodeId)
                    {
                        inst.cachedKernelIds.push_back(inst.fullKernelId);
                    }
                    else
                    {
                        uint32_t eclassId = nodeToEClass.at(partialNodeId);
                        auto cIt = extraction.choiceByEClass.find(egraph.find(eclassId));
                        if (cIt != extraction.choiceByEClass.end() && cIt->second.valid)
                        {
                            uint64_t uid = egraph.getENodes()[cIt->second.enodeId].kernelUid;
                            if (uid == 0)
                            {
                                std::stringstream ss;
                                ss << "\n[Planner Warning] Assigning kernel UID 0 to partial node " << partialNodeId
                                   << " (" << toString(graph.nodes[partialNodeId].opType) << ") for parent node " << nodeId;
                                Error::throw_err(ss.str());
                            }
                            inst.cachedKernelIds.push_back(uid);
                        }
                        else
                        {
                            Error::throw_err("No valid extraction for partial node " + toString(graph.nodes[partialNodeId], graph));
                        }
                    }
                }
            }

            compiled.instructions.push_back(inst);

            std::vector<TensorNode> costInputs;
            costInputs.reserve(inst.inputNodeIds.size());
            for (uint32_t pid : inst.inputNodeIds)
            {
                costInputs.push_back(compiled.nodesMap.at(pid));
            }
            TensorNode costOut = compiled.nodesMap.at(nodeId);
            compiled.nodeCosts[nodeId] = costModel.estimateCost(costOut, costInputs, graph, enode.kernelUid);

            if (logicalIt != eclassLogicalId.end())
            {
                compiled.logicalNodeMap[nodeId] = logicalIt->second;
            }
        }

        return compiled;
    }
};
