#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactorySoftmax(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0]; // The scores tensor [Heads, Seq, Seq]
    auto shape = g.getNode(x).getShape();
    uint32_t H = shape[0];
    uint32_t S = shape[1];

    // --- Part 1: Safe Softmax Shift (Max reduction) ---
    int32_t axis_val = -1;
    LogicalId axis_node = g.constant({1}, &axis_val, DType::INT32);
    LogicalId max_scores = g.max(x, axis_node);

    // repeat_3d_axis(max_scores, seq_len, 2)
    int32_t s_rep_val = (int32_t)S;
    LogicalId s_rep_node = g.constant({1}, &s_rep_val, DType::INT32);
    int32_t ax2_val = 2;
    LogicalId ax2_node = g.constant({1}, &ax2_val, DType::INT32);
    LogicalId max_expanded = g.repeat(max_scores, s_rep_node, ax2_node);

    LogicalId shifted_scores = g.add(x, g.neg(max_expanded));

    // --- Part 2: Exponentiate (expand_scalar_to_3d for e_node) ---
    float e_val = 2.718281828459045f;
    LogicalId e_scalar = g.constant({1}, &e_val, DType::FLOAT32);
    int32_t shape_3d_const[] = {1, 1, 1};
    LogicalId e_reshaped = g.reshape(e_scalar, g.constant({3}, shape_3d_const, DType::INT32));

    LogicalId e_node = e_reshaped;
    if (H > 1)
    {
        int32_t h_rep = (int32_t)H;
        int32_t ax0 = 0;
        e_node = g.repeat(e_node, g.constant({1}, &h_rep, DType::INT32), g.constant({1}, &ax0, DType::INT32));
    }
    if (S > 1)
    {
        int32_t s_rep = (int32_t)S;
        int32_t ax1 = 1;
        e_node = g.repeat(e_node, g.constant({1}, &s_rep, DType::INT32), g.constant({1}, &ax1, DType::INT32));
    }
    if (S > 1)
    { // Third dimension expansion
        int32_t s_rep = (int32_t)S;
        int32_t ax2 = 2;
        e_node = g.repeat(e_node, g.constant({1}, &s_rep, DType::INT32), g.constant({1}, &ax2, DType::INT32));
    }

    LogicalId exp_scores = g.pow(e_node, shifted_scores);

    // --- Part 3: Normalize (Sum reduction) ---
    LogicalId sum_exp = g.sum(exp_scores, g.constant({1}, &axis_val, DType::INT32));

    // repeat_3d_axis(sum_exp, seq_len, 2)
    LogicalId sum_exp_expanded = g.repeat(sum_exp, s_rep_node, ax2_node);

    return g.div(exp_scores, sum_exp_expanded);
}
