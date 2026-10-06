#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryMoETopKFusedGEMM_StreamingStorage(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // inputs[0]: X            [1, S, H]      fp32 CPU
    // inputs[1]: W_gu         [E, 2I, H]     bf16 STORAGE  (the raw g.weight
    // INPUT node) inputs[2]: W_dn         [E, H, I]      bf16 STORAGE  (the raw
    // g.weight INPUT node) inputs[3]: router_probs [1, S, E]      fp32 CPU
    // inputs[4]: sel          [1, S, K]      int32 CPU

    const LogicalId X_id = inputs[0];
    const LogicalId W_gu_id = inputs[1];
    const LogicalId W_dn_id = inputs[2];
    const LogicalId RP_id = inputs[3];
    const LogicalId sel_id = inputs[4];

    // Derive dimensions from input shapes
    const auto sX = graph.getNode(X_id).getShape();      // [1, S, H]
    const auto sWgu = graph.getNode(W_gu_id).getShape(); // [E, 2I, H]
    const auto sSel = graph.getNode(sel_id).getShape();  // [1, S, K]

    const uint32_t S = sX[1];
    const uint32_t H = sX[2];
    const uint32_t E = sWgu[0];
    const uint32_t I2 = sWgu[1]; // 2 * I
    const uint32_t I = I2 / 2;
    const uint32_t K = sSel[2];

    // -- Local helpers that mirror the model's repeat_ax / repeat_3d_axis --
    auto rep_axis = [&](LogicalId id, uint32_t repeats, uint32_t axis) -> LogicalId {
        if (repeats <= 1)
            return id;
        int32_t r = static_cast<int32_t>(repeats);
        int32_t a = static_cast<int32_t>(axis);
        return graph.repeat(id, graph.constant({1}, &r, DType::INT32), graph.constant({1}, &a, DType::INT32));
    };

    // Mirrors model's expand_scalar_to_3d(scalar_id, d0, d1, d2)
    auto expand_scalar_3d = [&](LogicalId sid, uint32_t d0, uint32_t d1, uint32_t d2) -> LogicalId {
        int32_t sh3[] = {1, 1, 1};
        LogicalId out = graph.reshape(sid, graph.constant({3}, sh3, DType::INT32));
        if (d0 > 1)
            out = rep_axis(out, d0, 0);
        if (d1 > 1)
            out = rep_axis(out, d1, 1);
        if (d2 > 1)
            out = rep_axis(out, d2, 2);
        return out;
    };

    // Convenience: create a float scalar constant and expand to 3D
    auto expand_float_3d = [&](float val, uint32_t d0, uint32_t d1, uint32_t d2) -> LogicalId {
        return expand_scalar_3d(graph.constant({1}, &val, DType::FLOAT32), d0, d1, d2);
    };

    // =====================================================================
    // STEP 1: Build router_mask [1, S, E] from sel [1, S, K]
    //
    //   sel_expanded [1, S, K, E] = repeat(reshape(sel, [1,S,K,1]), E, axis=3)
    //   range_expanded [1, S, K, E] = repeat(repeat(reshape(arange(0,E),
    //   [1,1,1,E]), S, axis=1), K, axis=2) mask_bool = eq(sel_expanded,
    //   range_expanded) mask_float = cast(mask_bool, F32) mask_reduced =
    //   sum(mask_float, axis=2)            -> [1, S, 1, E] router_mask =
    //   reshape(mask_reduced, [1, S, E])
    // =====================================================================

    // 1a. sel_expanded: [1, S, K, 1] -> repeat axis 3 by E -> [1, S, K, E]
    int32_t sh4_sel[] = {1, static_cast<int32_t>(S), static_cast<int32_t>(K), 1};
    LogicalId sel_reshaped = graph.reshape(sel_id, graph.constant({4}, sh4_sel, DType::INT32));
    LogicalId sel_expanded = graph.contiguous(rep_axis(sel_reshaped, E, 3));

    // 1b. range_expanded: arange(0, E) -> reshape [1,1,1,E] -> repeat axis 1 by S
    // -> repeat axis 2 by K
    int32_t arange_start = 0;
    int32_t arange_stop = static_cast<int32_t>(E);
    int32_t arange_step = 1;
    LogicalId range_1d =
        graph.arange(graph.constant({1}, &arange_start, DType::INT32), graph.constant({1}, &arange_stop, DType::INT32),
                     graph.constant({1}, &arange_step, DType::INT32));
    int32_t sh4_range[] = {1, 1, 1, static_cast<int32_t>(E)};
    LogicalId range_reshaped = graph.reshape(range_1d, graph.constant({4}, sh4_range, DType::INT32));
    LogicalId range_expanded = graph.contiguous(rep_axis(rep_axis(range_reshaped, S, 1), K, 2));

    // 1c. mask_bool = eq(sel_expanded, range_expanded)  -> [1, S, K, E] BOOL
    LogicalId mask_bool = graph.eq(sel_expanded, range_expanded);

    // 1d. mask_float = cast(mask_bool, F32)  -> [1, S, K, E] F32
    LogicalId mask_float = graph.cast(mask_bool, DType::FLOAT32);

    // 1e. mask_reduced = sum(mask_float, axis=2)  -> [1, S, 1, E]
    int32_t ax2_4d = 2;
    LogicalId mask_reduced = graph.sum(mask_float, graph.constant({1}, &ax2_4d, DType::INT32));

    // 1f. router_mask = reshape(mask_reduced, [1, S, E])
    int32_t sh3_mask[] = {1, static_cast<int32_t>(S), static_cast<int32_t>(E)};
    LogicalId router_mask = graph.reshape(mask_reduced, graph.constant({3}, sh3_mask, DType::INT32));

    // =====================================================================
    // STEP 2-4: Normalize probs
    //
    //   gated_probs = mul(router_probs, router_mask)          [1, S, E]
    //   row_sum = sum(gated_probs, axis=-1)                   [1, S, 1]
    //   row_sum = contiguous(repeat(row_sum, E, axis=2))      [1, S, E]
    //   normalized_probs = div(gated_probs, row_sum)          [1, S, E]
    // =====================================================================
    LogicalId gated_probs = graph.mul(RP_id, router_mask);

    int32_t axis_neg1 = -1;
    LogicalId row_sum = graph.sum(gated_probs, graph.constant({1}, &axis_neg1, DType::INT32));
    row_sum = graph.contiguous(rep_axis(row_sum, E, 2));

    LogicalId normalized_probs = graph.div(gated_probs, row_sum);

    // =====================================================================
    // STEP 5: Expand X to [E, S, H]
    //
    //   x_reshaped = reshape(X, [1, S, H])
    //   x_expanded = contiguous(repeat(x_reshaped, E, axis=0))  [E, S, H]
    // =====================================================================
    int32_t sh3_x[] = {1, static_cast<int32_t>(S), static_cast<int32_t>(H)};
    LogicalId x_reshaped = graph.reshape(X_id, graph.constant({3}, sh3_x, DType::INT32));
    LogicalId x_expanded = graph.contiguous(rep_axis(x_reshaped, E, 0));

    // =====================================================================
    // STEP 6: fused_gate_up_t = contiguous(permute(cast(copyto(W_gu, CPU), F32),
    // [0,2,1]))
    //
    //   copyto: STORAGE bf16 -> CPU bf16   (this is what g.weight does
    //   internally) cast:   CPU bf16 -> CPU fp32       (this is what the model's
    //   weight() helper does) permute [0,2,1]: [E, 2I, H] -> [E, H, 2I]
    //   contiguous: materialize the permuted view
    // =====================================================================
    LogicalId w_gu_cpu = graph._copyto(W_gu_id);
    LogicalId w_gu_f32 = graph.cast(w_gu_cpu, DType::FLOAT32);
    int32_t perm_w_3d[] = {0, 2, 1};
    LogicalId fused_gate_up_t = graph.permute(w_gu_f32, graph.constant({3}, perm_w_3d, DType::INT32));
    fused_gate_up_t = graph.contiguous(fused_gate_up_t);

    // =====================================================================
    // STEP 7: gate_up_proj = dot(x_expanded, fused_gate_up_t)  [E, S, 2I]
    // =====================================================================
    LogicalId gate_up_proj = graph.dot(x_expanded, fused_gate_up_t);

    // =====================================================================
    // STEP 8-9: Slice gate and up
    //
    //   exp_gate = contiguous(slice(gate_up_proj, [0,0,0], [E,S,I], [1,1,1]))
    //   exp_up   = contiguous(slice(gate_up_proj, [0,0,I], [E,S,2I], [1,1,1]))
    // =====================================================================
    int32_t steps_3d[] = {1, 1, 1};

    int32_t starts_gate[] = {0, 0, 0};
    int32_t ends_gate[] = {static_cast<int32_t>(E), static_cast<int32_t>(S), static_cast<int32_t>(I)};
    LogicalId exp_gate =
        graph.slice(gate_up_proj, graph.constant({3}, starts_gate, DType::INT32),
                    graph.constant({3}, ends_gate, DType::INT32), graph.constant({3}, steps_3d, DType::INT32));
    exp_gate = graph.contiguous(exp_gate);

    int32_t starts_up[] = {0, 0, static_cast<int32_t>(I)};
    int32_t ends_up[] = {static_cast<int32_t>(E), static_cast<int32_t>(S), static_cast<int32_t>(I * 2)};
    LogicalId exp_up =
        graph.slice(gate_up_proj, graph.constant({3}, starts_up, DType::INT32),
                    graph.constant({3}, ends_up, DType::INT32), graph.constant({3}, steps_3d, DType::INT32));
    exp_up = graph.contiguous(exp_up);

    // =====================================================================
    // STEP 10: exp_gate_silu = silu_atomic(exp_gate, E, S, I)
    //
    // Reproduces the model's silu_atomic EXACTLY:
    //   neg_one = expand_scalar_to_3d(constant(-1.0), E, S, I)
    //   neg_x = mul(exp_gate, neg_one)
    //   e_node = expand_scalar_to_3d(constant(2.71828...), E, S, I)
    //   exp_neg_x = pow(e_node, neg_x)
    //   one_node = expand_scalar_to_3d(constant(1.0), E, S, I)
    //   den = add(one_node, exp_neg_x)
    //   sigmoid = div(one_node, den)
    //   exp_gate_silu = mul(exp_gate, sigmoid)
    //
    // Note: the model uses one_fp32 (a pre-created member constant) for 1.0.
    // We create a fresh constant here; the e-graph should deduplicate
    // identical constants during saturation.
    // =====================================================================
    float neg_one_val = -1.0f;
    LogicalId neg_one = expand_float_3d(neg_one_val, E, S, I);
    LogicalId neg_x = graph.mul(exp_gate, neg_one);

    float e_val = 2.718281828459045f;
    LogicalId e_node = expand_float_3d(e_val, E, S, I);
    LogicalId exp_neg_x = graph.pow(e_node, neg_x);

    float one_val = 1.0f;
    LogicalId one_node = expand_float_3d(one_val, E, S, I);
    LogicalId den = graph.add(one_node, exp_neg_x);
    LogicalId sigmoid_val = graph.div(one_node, den);
    LogicalId exp_gate_silu = graph.mul(exp_gate, sigmoid_val);

    // =====================================================================
    // STEP 11: exp_gate_up = mul(exp_gate_silu, exp_up)  [E, S, I]
    // =====================================================================
    LogicalId exp_gate_up = graph.mul(exp_gate_silu, exp_up);

    // =====================================================================
    // STEP 12: fused_down_t = contiguous(permute(cast(copyto(W_dn, CPU), F32),
    // [0,2,1]))
    //
    //   [E, H, I] -> permute [0,2,1] -> [E, I, H] -> contiguous
    // =====================================================================
    LogicalId w_dn_cpu = graph._copyto(W_dn_id);
    LogicalId w_dn_f32 = graph.cast(w_dn_cpu, DType::FLOAT32);
    LogicalId fused_down_t = graph.permute(w_dn_f32, graph.constant({3}, perm_w_3d, DType::INT32));
    fused_down_t = graph.contiguous(fused_down_t);

    // =====================================================================
    // STEP 13: exp_down = dot(exp_gate_up, fused_down_t)  [E, S, H]
    // =====================================================================
    LogicalId exp_down = graph.dot(exp_gate_up, fused_down_t);

    // =====================================================================
    // STEP 14: exp_down_perm = contiguous(permute(exp_down, [1,0,2]))  [S, E, H]
    // =====================================================================
    int32_t perm_esh[] = {1, 0, 2};
    LogicalId exp_down_perm = graph.permute(exp_down, graph.constant({3}, perm_esh, DType::INT32));
    exp_down_perm = graph.contiguous(exp_down_perm);

    // =====================================================================
    // STEP 15: normalized_probs_perm = contiguous(permute(normalized_probs,
    // [1,2,0]))  [S, E, 1]
    // =====================================================================
    int32_t perm_1se[] = {1, 2, 0};
    LogicalId normalized_probs_perm = graph.permute(normalized_probs, graph.constant({3}, perm_1se, DType::INT32));
    normalized_probs_perm = graph.contiguous(normalized_probs_perm);

    // =====================================================================
    // STEP 16: normalized_probs_exp = contiguous(repeat(normalized_probs_perm, H,
    // axis=2))  [S, E, H]
    // =====================================================================
    LogicalId normalized_probs_exp = rep_axis(normalized_probs_perm, H, 2);
    normalized_probs_exp = graph.contiguous(normalized_probs_exp);

    // =====================================================================
    // STEP 17: weighted_outputs = mul(exp_down_perm, normalized_probs_exp)  [S,
    // E, H]
    // =====================================================================
    LogicalId weighted_outputs = graph.mul(exp_down_perm, normalized_probs_exp);

    // =====================================================================
    // STEP 18: routed_out_sum = sum(weighted_outputs, axis=1)  [S, 1, H]
    // =====================================================================
    int32_t sum_ax1[] = {1};
    LogicalId routed_out_sum = graph.sum(weighted_outputs, graph.constant({1}, sum_ax1, DType::INT32));

    // =====================================================================
    // STEP 19: routed_out = reshape(routed_out_sum, [1, S, H])
    // =====================================================================
    int32_t final_shape[] = {1, static_cast<int32_t>(S), static_cast<int32_t>(H)};
    LogicalId routed_out = graph.reshape(routed_out_sum, graph.constant({3}, final_shape, DType::INT32));

    return routed_out;
}
