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
            if ((graph.nodes[id].opType == OpType::ADD || graph.nodes[id].opType == OpType::MUL) &&
                graph.nodes[id].parentIds.size() == 2)
            {
                uint32_t p0 = graph.nodes[id].parentIds[0];
                uint32_t p1 = graph.nodes[id].parentIds[1];
                if (p0 != p1)
                {
                    uint32_t newId = (graph.nodes[id].opType == OpType::ADD) ? graph.add(p1, p0) : graph.mul(p1, p0);
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
            if (graph.nodes[id].opType == OpType::MUL && graph.nodes[id].parentIds.size() == 2)
            {
                for (int i = 0; i < 2; ++i)
                {
                    uint32_t a_id = graph.nodes[id].parentIds[i];
                    uint32_t add_id = graph.nodes[id].parentIds[1 - i];

                    if (graph.nodes[add_id].opType == OpType::ADD && graph.nodes[add_id].parentIds.size() == 2)
                    {
                        uint32_t b = graph.nodes[add_id].parentIds[0];
                        uint32_t c = graph.nodes[add_id].parentIds[1];

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
            if (graph.nodes[id].opType == OpType::ADD && graph.nodes[id].parentIds.size() == 2)
            {
                uint32_t m1_id = graph.nodes[id].parentIds[0];
                uint32_t m2_id = graph.nodes[id].parentIds[1];

                if (graph.nodes[m1_id].opType == OpType::MUL && graph.nodes[m2_id].opType == OpType::MUL &&
                    graph.nodes[m1_id].parentIds.size() == 2 && graph.nodes[m2_id].parentIds.size() == 2)
                {
                    for (int i = 0; i < 2; i++)
                    {
                        for (int j = 0; j < 2; j++)
                        {
                            if (graph.nodes[m1_id].parentIds[i] == graph.nodes[m2_id].parentIds[j])
                            {
                                uint32_t a = graph.nodes[m1_id].parentIds[i];
                                uint32_t b = graph.nodes[m1_id].parentIds[1 - i];
                                uint32_t c = graph.nodes[m2_id].parentIds[1 - j];

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
            if ((graph.nodes[id].opType == OpType::ADD || graph.nodes[id].opType == OpType::MUL) &&
                graph.nodes[id].parentIds.size() == 2)
            {
                OpType op = graph.nodes[id].opType;
                uint32_t a_id = graph.nodes[id].parentIds[0];
                uint32_t b_id = graph.nodes[id].parentIds[1];

                // (x op y) op z -> x op (y op z)
                if (graph.nodes[a_id].opType == op && graph.nodes[a_id].parentIds.size() == 2)
                {
                    uint32_t x = graph.nodes[a_id].parentIds[0];
                    uint32_t y = graph.nodes[a_id].parentIds[1];
                    uint32_t z = b_id;

                    uint32_t new_inner = (op == OpType::ADD) ? graph.add(y, z) : graph.mul(y, z);
                    uint32_t new_outer = (op == OpType::ADD) ? graph.add(x, new_inner) : graph.mul(x, new_inner);
                    results.push_back(new_outer);
                }

                // x op (y op z) -> (x op y) op z
                if (graph.nodes[b_id].opType == op && graph.nodes[b_id].parentIds.size() == 2)
                {
                    uint32_t x = a_id;
                    uint32_t y = graph.nodes[b_id].parentIds[0];
                    uint32_t z = graph.nodes[b_id].parentIds[1];

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
            if (graph.nodes[id].opType == OpType::NEGATE && graph.nodes[id].parentIds.size() == 1)
            {
                uint32_t inner_id = graph.nodes[id].parentIds[0];
                if (graph.nodes[inner_id].opType == OpType::NEGATE && graph.nodes[inner_id].parentIds.size() == 1)
                {
                    return {graph.nodes[inner_id].parentIds[0]};
                }
            }
            return {};
        }
    };

    struct NegateAddRule : public RewriteRule
    {
        std::vector<uint32_t> apply(uint32_t id, Graph &graph) const override
        {
            if (graph.nodes[id].opType == OpType::NEGATE && graph.nodes[id].parentIds.size() == 1)
            {
                uint32_t inner_id = graph.nodes[id].parentIds[0];
                if (graph.nodes[inner_id].opType == OpType::ADD && graph.nodes[inner_id].parentIds.size() == 2)
                {
                    uint32_t a = graph.nodes[inner_id].parentIds[0];
                    uint32_t b = graph.nodes[inner_id].parentIds[1];
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
            if (graph.nodes[id].opType == OpType::MUL && graph.nodes[id].parentIds.size() == 2)
            {
                uint32_t a_id = graph.nodes[id].parentIds[0];
                uint32_t b_id = graph.nodes[id].parentIds[1];

                if (graph.nodes[a_id].opType == OpType::DIVIDE &&
                    graph.nodes[a_id].parentIds.size() == 2 &&
                    graph.nodes[a_id].parentIds[1] == b_id)
                {
                    return {graph.nodes[a_id].parentIds[0]};
                }
                if (graph.nodes[b_id].opType == OpType::DIVIDE &&
                    graph.nodes[b_id].parentIds.size() == 2 &&
                    graph.nodes[b_id].parentIds[1] == a_id)
                {
                    return {graph.nodes[b_id].parentIds[0]};
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
            if (graph.nodes[id].opType == OpType::ADD && graph.nodes[id].parentIds.size() == 2)
            {
                uint32_t d1_id = graph.nodes[id].parentIds[0];
                uint32_t d2_id = graph.nodes[id].parentIds[1];

                if (graph.nodes[d1_id].opType == OpType::DIVIDE && graph.nodes[d2_id].opType == OpType::DIVIDE &&
                    graph.nodes[d1_id].parentIds.size() == 2 && graph.nodes[d2_id].parentIds.size() == 2)
                {
                    if (graph.nodes[d1_id].parentIds[1] == graph.nodes[d2_id].parentIds[1])
                    {
                        uint32_t a = graph.nodes[d1_id].parentIds[0];
                        uint32_t b = graph.nodes[d2_id].parentIds[0];
                        uint32_t c = graph.nodes[d1_id].parentIds[1];
                        uint32_t add_ab = graph.add(a, b);
                        return {graph.div(add_ab, c)};
                    }
                }
            }
            return {};
        }
    };

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