#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryFlashAttentionGeneric4D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId Q = inputs[0], K = inputs[1], V = inputs[2];
    auto sQ = g.getNode(Q).getShape();
    uint32_t B = sQ[0];
    uint32_t num_heads = sQ[1];
    uint32_t S = sQ[2];

    int32_t perm[] = {0, 1, 3, 2};
    LogicalId K_t = g.contiguous(g.permute(K, g.constant({4}, perm, DType::INT32)));
    LogicalId scores = g.dot(Q, K_t);

    LogicalId max_s = g.repeat(g.max(scores, -1), S, 3);
    LogicalId shifted = g.add(scores, g.neg(max_s));
    LogicalId exps = g.pow(g.fill(TGConstants::E, {B, num_heads, S, S}), shifted);
    LogicalId sums = g.repeat(g.sum(exps, -1), S, 3);
    LogicalId probs = g.div(exps, sums);

    return g.dot(probs, V);
}
