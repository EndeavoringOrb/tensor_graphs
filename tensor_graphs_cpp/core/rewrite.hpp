#pragma once
#include "core/graph.hpp"
#include "core/hashing.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <string>

namespace Rewrite
{
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
            const auto &node = graph.nodes[id];
            if ((node.opType == OpType::ADD || node.opType == OpType::MUL) && node.parentIds.size() == 2)
            {
                uint32_t p0 = node.parentIds[0];
                uint32_t p1 = node.parentIds[1];
                if (p0 != p1)
                {
                    uint32_t newId = (node.opType == OpType::ADD) ? graph.add(p1, p0) : graph.mul(p1, p0);
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
            const auto &node = graph.nodes[id];
            // Matches: a * (b + c) -> (a * b) + (a * c)
            if (node.opType == OpType::MUL && node.parentIds.size() == 2)
            {
                uint32_t a = node.parentIds[0];
                uint32_t add_node_id = node.parentIds[1];

                auto match = [&](uint32_t a_id, uint32_t add_id) -> std::vector<uint32_t>
                {
                    const auto &add_node = graph.nodes[add_id];
                    if (add_node.opType == OpType::ADD && add_node.parentIds.size() == 2)
                    {
                        uint32_t b = add_node.parentIds[0];
                        uint32_t c = add_node.parentIds[1];
                        uint32_t mul1 = graph.mul(a_id, b);
                        uint32_t mul2 = graph.mul(a_id, c);
                        return {graph.add(mul1, mul2)};
                    }
                    return {};
                };

                auto res1 = match(a, add_node_id);
                if (!res1.empty())
                    return res1;

                auto res2 = match(add_node_id, a);
                if (!res2.empty())
                    return res2;
            }
            return {};
        }
    };

    struct FactoringRule : public RewriteRule
    {
        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            const auto &node = graph.nodes[id];
            // Matches: (a * b) + (a * c) -> a * (b + c)
            if (node.opType == OpType::ADD && node.parentIds.size() == 2)
            {
                const auto &mul1 = graph.nodes[node.parentIds[0]];
                const auto &mul2 = graph.nodes[node.parentIds[1]];

                if (mul1.opType == OpType::MUL && mul2.opType == OpType::MUL &&
                    mul1.parentIds.size() == 2 && mul2.parentIds.size() == 2)
                {
                    for (int i = 0; i < 2; i++)
                    {
                        for (int j = 0; j < 2; j++)
                        {
                            if (mul1.parentIds[i] == mul2.parentIds[j])
                            {
                                uint32_t a = mul1.parentIds[i];
                                uint32_t b = mul1.parentIds[1 - i];
                                uint32_t c = mul2.parentIds[1 - j];
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
            const auto &node = graph.nodes[id];
            if ((node.opType == OpType::ADD || node.opType == OpType::MUL) && node.parentIds.size() == 2)
            {
                uint32_t a_id = node.parentIds[0];
                uint32_t b_id = node.parentIds[1];

                const auto &a = graph.nodes[a_id];
                const auto &b = graph.nodes[b_id];

                // (x op y) op z -> x op (y op z)
                if (a.opType == node.opType && a.parentIds.size() == 2)
                {
                    uint32_t x = a.parentIds[0];
                    uint32_t y = a.parentIds[1];
                    uint32_t z = b_id;

                    uint32_t new_inner = (node.opType == OpType::ADD) ? graph.add(y, z) : graph.mul(y, z);
                    uint32_t new_outer = (node.opType == OpType::ADD) ? graph.add(x, new_inner) : graph.mul(x, new_inner);
                    results.push_back(new_outer);
                }

                // x op (y op z) -> (x op y) op z
                if (b.opType == node.opType && b.parentIds.size() == 2)
                {
                    uint32_t x = a_id;
                    uint32_t y = b.parentIds[0];
                    uint32_t z = b.parentIds[1];

                    uint32_t new_inner = (node.opType == OpType::ADD) ? graph.add(x, y) : graph.mul(x, y);
                    uint32_t new_outer = (node.opType == OpType::ADD) ? graph.add(new_inner, z) : graph.mul(new_inner, z);
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
            const auto &node = graph.nodes[id];
            if (node.opType == OpType::NEGATE && node.parentIds.size() == 1)
            {
                const auto &inner = graph.nodes[node.parentIds[0]];
                if (inner.opType == OpType::NEGATE && inner.parentIds.size() == 1)
                {
                    return {inner.parentIds[0]};
                }
            }
            return {};
        }
    };

    struct NegateAddRule : public RewriteRule
    {
        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            const auto &node = graph.nodes[id];
            if (node.opType == OpType::NEGATE && node.parentIds.size() == 1)
            {
                const auto &inner = graph.nodes[node.parentIds[0]];
                if (inner.opType == OpType::ADD && inner.parentIds.size() == 2)
                {
                    uint32_t a = inner.parentIds[0];
                    uint32_t b = inner.parentIds[1];
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
            const auto &node = graph.nodes[id];
            if (node.opType == OpType::MUL && node.parentIds.size() == 2)
            {
                uint32_t a_id = node.parentIds[0];
                uint32_t b_id = node.parentIds[1];
                const auto &a = graph.nodes[a_id];
                const auto &b = graph.nodes[b_id];

                if (a.opType == OpType::DIVIDE && a.parentIds.size() == 2 && a.parentIds[1] == b_id)
                {
                    return {a.parentIds[0]};
                }
                if (b.opType == OpType::DIVIDE && b.parentIds.size() == 2 && b.parentIds[1] == a_id)
                {
                    return {b.parentIds[0]};
                }
            }
            return {};
        }
    };

    struct DivAddRule : public RewriteRule
    {
        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            const auto &node = graph.nodes[id];
            // (a / c) + (b / c) -> (a + b) / c
            if (node.opType == OpType::ADD && node.parentIds.size() == 2)
            {
                const auto &div1 = graph.nodes[node.parentIds[0]];
                const auto &div2 = graph.nodes[node.parentIds[1]];
                if (div1.opType == OpType::DIVIDE && div2.opType == OpType::DIVIDE &&
                    div1.parentIds.size() == 2 && div2.parentIds.size() == 2)
                {
                    if (div1.parentIds[1] == div2.parentIds[1])
                    {
                        uint32_t a = div1.parentIds[0];
                        uint32_t b = div2.parentIds[0];
                        uint32_t c = div1.parentIds[1];
                        uint32_t add_ab = graph.add(a, b);
                        return {graph.div(add_ab, c)};
                    }
                }
            }
            return {};
        }
    };

    // TODO: make generateAllEquivalents have a memo arg, pass memo to getPatternHash
    inline std::vector<uint32_t> generateAllEquivalents(uint32_t rootId, Graph &graph, const std::vector<const RewriteRule *> &rules)
    {
        std::vector<uint32_t> equivalents;
        std::unordered_set<std::string> seenHashes;
        std::queue<uint32_t> worklist;

        equivalents.push_back(rootId);
        worklist.push(rootId);
        seenHashes.insert(Hashing::getPatternHash(rootId, graph));

        while (!worklist.empty())
        {
            uint32_t current = worklist.front();
            worklist.pop();

            for (const auto *rule : rules)
            {
                std::vector<uint32_t> newNodes = rule->apply(current, graph);
                for (uint32_t newNode : newNodes)
                {
                    std::string newHash = Hashing::getPatternHash(newNode, graph);
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