#pragma once
#include "core/graph.hpp"
#include "core/hashing.hpp"
#include "core/kernels.hpp"
#include "core/shapes.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <string>

namespace Rewrite
{
    inline TensorNode makeBackendAdjustedNode(const TensorNode &source, Backend backend, bool makeContiguous = false)
    {
        TensorNode node = source;
        node.backend = backend;
        if (makeContiguous)
        {
            node.strides = calcContiguousStrides(source.getShape());
        }
        return node;
    }

    inline bool hasKernelMatch(OpType opType, const std::vector<TensorNode> &inputs, const TensorNode &output)
    {
        return !KernelRegistry::get().findMatchingKernels(opType, "", output.backend, inputs, output, {}, false).empty();
    }

    struct RewriteRule
    {
        virtual ~RewriteRule() = default;
        // Applies a rule to the node `id` in `graph`. Returns a list of newly created equivalent node IDs.
        virtual std::vector<uint32_t> apply(uint32_t id, Graph &graph) const = 0;
    };

    struct CommutativeRule : public RewriteRule
    {
        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            if ((graph.getNode(id).opType == OpType::ADD || graph.getNode(id).opType == OpType::MUL) &&
                graph.getNode(id).parentIds.size() == 2)
            {
                uint32_t p0 = graph.getNode(id).parentIds[0];
                uint32_t p1 = graph.getNode(id).parentIds[1];
                if (p0 != p1)
                {
                    uint32_t newId = (graph.getNode(id).opType == OpType::ADD) ? graph.add(p1, p0) : graph.mul(p1, p0);
                    return {newId};
                }
            }
            return {};
        }
    };

    struct DistributiveRule : public RewriteRule
    {
        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            // Matches: a * (b + c) -> (a * b) + (a * c)
            if (graph.getNode(id).opType == OpType::MUL && graph.getNode(id).parentIds.size() == 2)
            {
                for (int i = 0; i < 2; ++i)
                {
                    uint32_t a_id = graph.getNode(id).parentIds[i];
                    uint32_t add_id = graph.getNode(id).parentIds[1 - i];

                    if (graph.getNode(add_id).opType == OpType::ADD && graph.getNode(add_id).parentIds.size() == 2)
                    {
                        uint32_t b = graph.getNode(add_id).parentIds[0];
                        uint32_t c = graph.getNode(add_id).parentIds[1];

                        uint32_t mul1 = graph.mul(a_id, b);
                        uint32_t mul2 = graph.mul(a_id, c);
                        return {graph.add(mul1, mul2)};
                    }
                }
            }
            return {};
        }
    };

    struct FactoringRule : public RewriteRule
    {
        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            // Matches: (a * b) + (a * c) -> a * (b + c)
            if (graph.getNode(id).opType == OpType::ADD && graph.getNode(id).parentIds.size() == 2)
            {
                uint32_t m1_id = graph.getNode(id).parentIds[0];
                uint32_t m2_id = graph.getNode(id).parentIds[1];

                if (graph.getNode(m1_id).opType == OpType::MUL && graph.getNode(m2_id).opType == OpType::MUL &&
                    graph.getNode(m1_id).parentIds.size() == 2 && graph.getNode(m2_id).parentIds.size() == 2)
                {
                    for (int i = 0; i < 2; i++)
                    {
                        for (int j = 0; j < 2; j++)
                        {
                            if (graph.getNode(m1_id).parentIds[i] == graph.getNode(m2_id).parentIds[j])
                            {
                                uint32_t a = graph.getNode(m1_id).parentIds[i];
                                uint32_t b = graph.getNode(m1_id).parentIds[1 - i];
                                uint32_t c = graph.getNode(m2_id).parentIds[1 - j];

                                uint32_t add_bc = graph.add(b, c);
                                return {graph.mul(a, add_bc)};
                            }
                        }
                    }
                }
            }
            return {};
        }
    };

    struct AssociativeRule : public RewriteRule
    {
        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            std::vector<uint32_t> results;
            if ((graph.getNode(id).opType == OpType::ADD || graph.getNode(id).opType == OpType::MUL) &&
                graph.getNode(id).parentIds.size() == 2)
            {
                OpType op = graph.getNode(id).opType;
                uint32_t a_id = graph.getNode(id).parentIds[0];
                uint32_t b_id = graph.getNode(id).parentIds[1];

                // (x op y) op z -> x op (y op z)
                if (graph.getNode(a_id).opType == op && graph.getNode(a_id).parentIds.size() == 2)
                {
                    uint32_t x = graph.getNode(a_id).parentIds[0];
                    uint32_t y = graph.getNode(a_id).parentIds[1];
                    uint32_t z = b_id;

                    uint32_t new_inner = (op == OpType::ADD) ? graph.add(y, z) : graph.mul(y, z);
                    uint32_t new_outer = (op == OpType::ADD) ? graph.add(x, new_inner) : graph.mul(x, new_inner);
                    results.push_back(new_outer);
                }

                // x op (y op z) -> (x op y) op z
                if (graph.getNode(b_id).opType == op && graph.getNode(b_id).parentIds.size() == 2)
                {
                    uint32_t x = a_id;
                    uint32_t y = graph.getNode(b_id).parentIds[0];
                    uint32_t z = graph.getNode(b_id).parentIds[1];

                    uint32_t new_inner = (op == OpType::ADD) ? graph.add(x, y) : graph.mul(x, y);
                    uint32_t new_outer = (op == OpType::ADD) ? graph.add(new_inner, z) : graph.mul(new_inner, z);
                    results.push_back(new_outer);
                }
            }
            return results;
        }
    };

    struct DoubleNegationRule : public RewriteRule
    {
        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            if (graph.getNode(id).opType == OpType::NEGATE && graph.getNode(id).parentIds.size() == 1)
            {
                uint32_t inner_id = graph.getNode(id).parentIds[0];
                if (graph.getNode(inner_id).opType == OpType::NEGATE && graph.getNode(inner_id).parentIds.size() == 1)
                {
                    return {graph.getNode(inner_id).parentIds[0]};
                }
            }
            return {};
        }
    };

    struct NegateAddRule : public RewriteRule
    {
        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            if (graph.getNode(id).opType == OpType::NEGATE && graph.getNode(id).parentIds.size() == 1)
            {
                uint32_t inner_id = graph.getNode(id).parentIds[0];
                if (graph.getNode(inner_id).opType == OpType::ADD && graph.getNode(inner_id).parentIds.size() == 2)
                {
                    uint32_t a = graph.getNode(inner_id).parentIds[0];
                    uint32_t b = graph.getNode(inner_id).parentIds[1];
                    uint32_t neg_a = graph.neg(a);
                    uint32_t neg_b = graph.neg(b);
                    return {graph.add(neg_a, neg_b)};
                }
            }
            return {};
        }
    };

    struct DivMulRule : public RewriteRule
    {
        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            if (graph.getNode(id).opType == OpType::MUL && graph.getNode(id).parentIds.size() == 2)
            {
                uint32_t a_id = graph.getNode(id).parentIds[0];
                uint32_t b_id = graph.getNode(id).parentIds[1];

                if (graph.getNode(a_id).opType == OpType::DIVIDE &&
                    graph.getNode(a_id).parentIds.size() == 2 &&
                    graph.getNode(a_id).parentIds[1] == b_id)
                {
                    return {graph.getNode(a_id).parentIds[0]};
                }
                if (graph.getNode(b_id).opType == OpType::DIVIDE &&
                    graph.getNode(b_id).parentIds.size() == 2 &&
                    graph.getNode(b_id).parentIds[1] == a_id)
                {
                    return {graph.getNode(b_id).parentIds[0]};
                }
            }
            return {};
        }
    };

    struct DivAddRule : public RewriteRule
    {
        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            // (a / c) + (b / c) -> (a + b) / c
            if (graph.getNode(id).opType == OpType::ADD && graph.getNode(id).parentIds.size() == 2)
            {
                uint32_t d1_id = graph.getNode(id).parentIds[0];
                uint32_t d2_id = graph.getNode(id).parentIds[1];

                if (graph.getNode(d1_id).opType == OpType::DIVIDE && graph.getNode(d2_id).opType == OpType::DIVIDE &&
                    graph.getNode(d1_id).parentIds.size() == 2 && graph.getNode(d2_id).parentIds.size() == 2)
                {
                    if (graph.getNode(d1_id).parentIds[1] == graph.getNode(d2_id).parentIds[1])
                    {
                        uint32_t a = graph.getNode(d1_id).parentIds[0];
                        uint32_t b = graph.getNode(d2_id).parentIds[0];
                        uint32_t c = graph.getNode(d1_id).parentIds[1];
                        uint32_t add_ab = graph.add(a, b);
                        return {graph.div(add_ab, c)};
                    }
                }
            }
            return {};
        }
    };

    // Reorder copyto with contiguous when the matching kernels exist.
    struct CopyToContiguousReorderRule : public RewriteRule
    {
        bool skipKernelChecks = false;

        explicit CopyToContiguousReorderRule(bool skipKernelChecks = false)
            : skipKernelChecks(skipKernelChecks) {}

        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            if (!graph.hasNode(id))
                return {};

            const TensorNode &node = graph.getNode(id);

            // contiguous(copyto(x, B)) -> copyto(contiguous(x), B)
            if (node.opType == OpType::CONTIGUOUS && !node.parentIds.empty())
            {
                uint32_t copyId = node.parentIds[0];
                if (!graph.hasNode(copyId))
                    return {};
                const TensorNode &copyNode = graph.getNode(copyId);
                if (copyNode.opType != OpType::COPY_TO || copyNode.parentIds.empty())
                    return {};

                uint32_t sourceId = copyNode.parentIds[0];
                if (!graph.hasNode(sourceId))
                    return {};

                const TensorNode &sourceNode = graph.getNode(sourceId);
                if (!skipKernelChecks)
                {
                    TensorNode copyOutput = makeBackendAdjustedNode(sourceNode, copyNode.backend);
                    if (!hasKernelMatch(OpType::COPY_TO, {sourceNode}, copyOutput))
                        return {};

                    TensorNode contigInput = makeBackendAdjustedNode(sourceNode, sourceNode.backend);
                    TensorNode contigOutput = makeBackendAdjustedNode(sourceNode, sourceNode.backend, true);
                    if (!hasKernelMatch(OpType::CONTIGUOUS, {contigInput}, contigOutput))
                        return {};
                }

                uint32_t newContigId = graph.contiguous(sourceId);
                graph.getNode(newContigId).backend = sourceNode.backend;
                uint32_t newCopyId = graph.copyto(newContigId, copyNode.backend);
                graph.getNode(newCopyId).backend = copyNode.backend;
                return {newCopyId};
            }

            // copyto(contiguous(x), B) -> contiguous(copyto(x, B))
            if (node.opType == OpType::COPY_TO && !node.parentIds.empty())
            {
                uint32_t contigId = node.parentIds[0];
                if (!graph.hasNode(contigId))
                    return {};
                const TensorNode &contigNode = graph.getNode(contigId);
                if (contigNode.opType != OpType::CONTIGUOUS || contigNode.parentIds.empty())
                    return {};

                uint32_t sourceId = contigNode.parentIds[0];
                if (!graph.hasNode(sourceId))
                    return {};

                const TensorNode &sourceNode = graph.getNode(sourceId);
                if (!skipKernelChecks)
                {
                    TensorNode copyOutput = makeBackendAdjustedNode(sourceNode, node.backend);
                    if (!hasKernelMatch(OpType::COPY_TO, {sourceNode}, copyOutput))
                        return {};

                    TensorNode contigInput = makeBackendAdjustedNode(sourceNode, node.backend);
                    TensorNode contigOutput = makeBackendAdjustedNode(sourceNode, node.backend, true);
                    if (!hasKernelMatch(OpType::CONTIGUOUS, {contigInput}, contigOutput))
                        return {};
                }

                uint32_t newCopyId = graph.copyto(sourceId, node.backend);
                graph.getNode(newCopyId).backend = node.backend;
                uint32_t newContigId = graph.contiguous(newCopyId);
                graph.getNode(newContigId).backend = node.backend;
                return {newContigId};
            }

            return {};
        }
    };

    // Reorder copyto with scatter when the matching kernels exist.
    struct CopyToScatterReorderRule : public RewriteRule
    {
        bool skipKernelChecks = false;

        explicit CopyToScatterReorderRule(bool skipKernelChecks = false)
            : skipKernelChecks(skipKernelChecks) {}

        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            if (!graph.hasNode(id))
                return {};

            const TensorNode &node = graph.getNode(id);

            // scatter(copyto(target, B), updates, ...) -> copyto(scatter(target, updates, ...), B)
            if (node.opType == OpType::SCATTER && !node.parentIds.empty())
            {
                uint32_t copyId = node.parentIds[0];
                if (!graph.hasNode(copyId))
                    return {};
                const TensorNode &copyNode = graph.getNode(copyId);
                if (copyNode.opType != OpType::COPY_TO || copyNode.parentIds.empty())
                    return {};

                uint32_t sourceTargetId = copyNode.parentIds[0];
                if (!graph.hasNode(sourceTargetId))
                    return {};

                const TensorNode &sourceTargetNode = graph.getNode(sourceTargetId);
                if (!skipKernelChecks)
                {
                    std::vector<TensorNode> scatterInputs = {
                        sourceTargetNode,
                        graph.getNode(node.parentIds[1]),
                        graph.getNode(node.parentIds[2]),
                        graph.getNode(node.parentIds[3]),
                        graph.getNode(node.parentIds[4]),
                    };
                    TensorNode scatterOutput = makeBackendAdjustedNode(sourceTargetNode, sourceTargetNode.backend);
                    if (!hasKernelMatch(OpType::SCATTER, scatterInputs, scatterOutput))
                        return {};

                    TensorNode copyOutput = makeBackendAdjustedNode(sourceTargetNode, copyNode.backend);
                    if (!hasKernelMatch(OpType::COPY_TO, {sourceTargetNode}, copyOutput))
                        return {};
                }

                uint32_t newScatterId = graph.scatter(sourceTargetId, node.parentIds[1], node.parentIds[2], node.parentIds[3], node.parentIds[4]);
                graph.getNode(newScatterId).backend = sourceTargetNode.backend;
                uint32_t newCopyId = graph.copyto(newScatterId, copyNode.backend);
                graph.getNode(newCopyId).backend = copyNode.backend;
                return {newCopyId};
            }

            // copyto(scatter(target, updates, ...), B) -> scatter(copyto(target, B), updates, ...)
            if (node.opType == OpType::COPY_TO && !node.parentIds.empty())
            {
                uint32_t scatterId = node.parentIds[0];
                if (!graph.hasNode(scatterId))
                    return {};
                const TensorNode &scatterNode = graph.getNode(scatterId);
                if (scatterNode.opType != OpType::SCATTER || scatterNode.parentIds.empty())
                    return {};

                uint32_t sourceTargetId = scatterNode.parentIds[0];
                if (!graph.hasNode(sourceTargetId))
                    return {};

                const TensorNode &sourceTargetNode = graph.getNode(sourceTargetId);
                if (!skipKernelChecks)
                {
                    std::vector<TensorNode> scatterInputs = {
                        sourceTargetNode,
                        graph.getNode(scatterNode.parentIds[1]),
                        graph.getNode(scatterNode.parentIds[2]),
                        graph.getNode(scatterNode.parentIds[3]),
                        graph.getNode(scatterNode.parentIds[4]),
                    };
                    TensorNode scatterOutput = makeBackendAdjustedNode(sourceTargetNode, node.backend);
                    if (!hasKernelMatch(OpType::SCATTER, scatterInputs, scatterOutput))
                        return {};

                    TensorNode copyOutput = makeBackendAdjustedNode(sourceTargetNode, node.backend);
                    if (!hasKernelMatch(OpType::COPY_TO, {sourceTargetNode}, copyOutput))
                        return {};
                }

                uint32_t newCopyId = graph.copyto(sourceTargetId, node.backend);
                graph.getNode(newCopyId).backend = node.backend;
                uint32_t newScatterId = graph.scatter(newCopyId, scatterNode.parentIds[1], scatterNode.parentIds[2], scatterNode.parentIds[3], scatterNode.parentIds[4]);
                graph.getNode(newScatterId).backend = node.backend;
                return {newScatterId};
            }

            return {};
        }
    };

    struct RemoveContiguousRule : public RewriteRule
    {
        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            if (!graph.hasNode(id))
                return {};

            const TensorNode &node = graph.getNode(id);

            std::vector<uint32_t> results;
            const auto &parents = node.parentIds;

            for (size_t i = 0; i < parents.size(); ++i)
            {
                uint32_t parentId = parents[i];
                if (!graph.hasNode(parentId))
                    continue;

                const TensorNode &parentNode = graph.getNode(parentId);

                // Check if this parent is a 'contiguous' operation
                if (parentNode.opType == OpType::CONTIGUOUS && !parentNode.parentIds.empty())
                {
                    uint32_t sourceId = parentNode.parentIds[0];

                    // Construct hypothetical inputs: replace the CONTIGUOUS node with its source
                    std::vector<TensorNode> hypoInputs;
                    hypoInputs.reserve(parents.size());
                    for (size_t j = 0; j < parents.size(); ++j)
                    {
                        uint32_t actualId = (i == j) ? sourceId : parents[j];
                        hypoInputs.push_back(graph.getNode(actualId));
                    }

                    // Ask the registry: "Is there any kernel for this Op that supports these inputs?"
                    auto matches = KernelRegistry::get().findMatchingKernels(
                        node.opType,
                        node.opName,
                        node.backend,
                        hypoInputs,
                        node,
                        {});

                    if (!matches.empty())
                    {
                        // Redundancy found! Create a new version of the node using the source directly.
                        std::vector<uint32_t> newParents = parents;
                        newParents[i] = sourceId;

                        // We use allocateNode to create a node with identical properties but different parents.
                        uint32_t newNodeId = graph.allocateNode(
                                                      node.opType,
                                                      node.opName,
                                                      node.dtype,
                                                      newParents,
                                                      node.getShape(),
                                                      node.strides,
                                                      node.backend,
                                                      node.storageType,
                                                      node.contentHash)
                                                 .id;

                        results.push_back(newNodeId);
                    }
                }
            }

            return results;
        }
    };

    inline std::vector<uint32_t> generateAllEquivalents(uint32_t rootId, Graph &graph, const std::vector<const RewriteRule *> &rules, std::unordered_map<uint32_t, std::string> &memo)
    {
        std::vector<uint32_t> equivalents;
        std::unordered_set<std::string> seenHashes;
        std::queue<uint32_t> worklist;

        equivalents.push_back(rootId);
        worklist.push(rootId);
        seenHashes.insert(Hashing::patternHash(rootId, graph, memo));

        while (!worklist.empty())
        {
            uint32_t current = worklist.front();
            worklist.pop();

            for (const auto *rule : rules)
            {
                std::vector<uint32_t> newNodes = rule->apply(current, graph);
                for (uint32_t newNode : newNodes)
                {
                    std::string newHash = Hashing::patternHash(newNode, graph, memo);
                    if (seenHashes.find(newHash) == seenHashes.end())
                    {
                        seenHashes.insert(newHash);
                        equivalents.push_back(newNode);
                        worklist.push(newNode);
                    }
                }
            }
        }
        return equivalents;
    }

} // namespace Rewrite
