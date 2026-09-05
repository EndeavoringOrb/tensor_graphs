// ref:
// https://github.com/huggingface/transformers/blob/1048e9af78a6045444244412dfe216ba5810e7fb/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
#pragma once
#include <cmath>
#include <string>
#include <tuple>

#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/types.hpp"

struct Qwen3_6_35B_A3B_Config
{
    // Token Embedding
    uint32_t vocab_size = 248320;
    uint32_t num_hidden_layers = 40;
    uint32_t hidden_size = 2048;

    // Gated DeltaNet (Linear Attention)
    uint32_t linear_n_v_heads = 32;
    uint32_t linear_n_qk_heads = 16;
    uint32_t linear_head_dim = 128;
    uint32_t linear_conv_kernel = 4;

    // Gated Attention (Full Attention)
    uint32_t attn_n_q_heads = 16;
    uint32_t attn_n_kv_heads = 2;
    uint32_t head_dim = 256;
    uint32_t rope_dim = 64; // head_dim * partial_rotary_factor = 256 * 0.25 = 64
    float rope_theta = 10'000'000.0f;

    // Mixture Of Experts (MoE)
    uint32_t n_experts = 256;
    uint32_t n_active_experts = 8;
    uint32_t shared_expert_dim = 512;

    uint32_t query_pre_attn_scalar = 256;

    uint32_t mrope_section_t = 11;
    uint32_t mrope_section_h = 11;
    uint32_t mrope_section_w = 10;
};

class Qwen3_6_35B_A3B_Model
{
  private:
    Qwen3_6_35B_A3B_Config cfg;
    Graph &g;
    MemoryManager &mem;
    const std::string w_path;
    float eps;
    uint32_t seq_len;
    LogicalId one_fp32;
    LogicalId eps_fp32;
    LogicalId half_fp32;

  public:
    Qwen3_6_35B_A3B_Model(Qwen3_6_35B_A3B_Config config, uint32_t sequence_length, Graph &graph, MemoryManager &memory,
                          const std::string &weight_path)
        : cfg(config), g(graph), mem(memory), w_path(weight_path), eps(1e-6f), seq_len(sequence_length)
    {
        float one_val = 1.0f;
        one_fp32 = g.constant({1}, &one_val, DType::FLOAT32);
        eps_fp32 = g.constant({1}, &eps, DType::FLOAT32);
        float half_val = 0.5f;
        half_fp32 = g.constant({1}, &half_val, DType::FLOAT32);
    }

    LogicalId weight(const std::string &path, const std::string &name)
    {
        LogicalId raw_weight = g.weight(path, name);
        return g.cast(raw_weight, DType::FLOAT32);
    }

    LogicalId expand_scalar_to_3d(LogicalId scalar_id, uint32_t dim0, uint32_t dim1, uint32_t dim2)
    {
        int32_t shape_3d[] = {1, 1, 1};
        LogicalId shape_3d_node = g.constant({3}, shape_3d, DType::INT32);
        LogicalId out = g.reshape(scalar_id, shape_3d_node);
        if (dim0 > 1)
            out = g.repeat(out, dim0, 0);
        if (dim1 > 1)
            out = g.repeat(out, dim1, 1);
        if (dim2 > 1)
            out = g.repeat(out, dim2, 2);
        return out;
    }

    LogicalId expand_scalar_to_3d(float val, uint32_t dim0, uint32_t dim1, uint32_t dim2)
    {
        LogicalId node = g.constant({1}, &val, DType::FLOAT32);
        return expand_scalar_to_3d(node, dim0, dim1, dim2);
    }

    LogicalId expand_scalar_to_4d(float val, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3)
    {
        LogicalId node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh4[] = {1, 1, 1, 1};
        LogicalId out = g.reshape(node, g.constant({4}, sh4, DType::INT32));
        if (d0 > 1)
            out = g.repeat(out, d0, 0);
        if (d1 > 1)
            out = g.repeat(out, d1, 1);
        if (d2 > 1)
            out = g.repeat(out, d2, 2);
        if (d3 > 1)
            out = g.repeat(out, d3, 3);
        return out;
    }

    LogicalId expand_scalar_to_1d(float val, uint32_t d0)
    {
        LogicalId node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh1[] = {1};
        return g.repeat(g.reshape(node, g.constant({1}, sh1, DType::INT32)), d0, 0);
    }

    LogicalId expand_1d_to_3d(LogicalId vec_id, uint32_t vec_len, uint32_t dim0, uint32_t dim1)
    {
        int32_t shape_3d[] = {1, 1, (int32_t)vec_len};
        LogicalId shape_3d_node = g.constant({3}, shape_3d, DType::INT32);
        LogicalId out = g.reshape(vec_id, shape_3d_node);
        if (dim0 > 1)
            out = g.repeat(out, dim0, 0);
        if (dim1 > 1)
            out = g.repeat(out, dim1, 1);
        return out;
    }

    LogicalId sigmoid(LogicalId x_id, uint32_t last_dim)
    {
        float neg_one_val = -1.0f;
        LogicalId neg_one = expand_scalar_to_3d(g.constant({1}, &neg_one_val, DType::FLOAT32), 1, seq_len, last_dim);
        LogicalId neg_x = g.mul(x_id, neg_one);
        float e_val = 2.718281828459045f;
        LogicalId e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), 1, seq_len, last_dim);
        LogicalId exp_neg_x = g.pow(e_node, neg_x);
        LogicalId one_node = expand_scalar_to_3d(one_fp32, 1, seq_len, last_dim);
        LogicalId den = g.add(one_node, exp_neg_x);
        return g.div(one_node, den);
    }

    LogicalId silu_atomic(LogicalId x_id, uint32_t last_dim)
    {
        return g.mul(x_id, sigmoid(x_id, last_dim));
    }

    LogicalId silu_atomic(LogicalId x_id, uint32_t N, uint32_t L, uint32_t D)
    {
        float neg_one_val = -1.0f;
        LogicalId neg_one = expand_scalar_to_3d(g.constant({1}, &neg_one_val, DType::FLOAT32), N, L, D);
        LogicalId neg_x = g.mul(x_id, neg_one);
        float e_val = 2.718281828459045f;
        LogicalId e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), N, L, D);
        LogicalId exp_neg_x = g.pow(e_node, neg_x);
        LogicalId one_node = expand_scalar_to_3d(one_fp32, N, L, D);
        LogicalId den = g.add(one_node, exp_neg_x);
        LogicalId sigmoid = g.div(one_node, den);
        return g.mul(x_id, sigmoid);
    }

    LogicalId softmax(LogicalId scores, uint32_t dim_size)
    {
        int32_t axis_val = -1;
        LogicalId axis_node = g.constant({1}, &axis_val, DType::INT32);
        LogicalId max_scores = g.max(scores, axis_node);
        LogicalId max_expanded = g.repeat(max_scores, dim_size, 2);
        LogicalId shifted_scores = g.add(scores, g.neg(max_expanded));

        float e_val = 2.718281828459045f;
        LogicalId e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), 1, seq_len, dim_size);
        LogicalId exp_scores = g.pow(e_node, shifted_scores);

        LogicalId sum_exp = g.sum(exp_scores, axis_node);
        LogicalId sum_exp_expanded = g.repeat(sum_exp, dim_size, 2);

        return g.div(exp_scores, sum_exp_expanded);
    }

    LogicalId rms_norm_atomic(LogicalId x_id, LogicalId weight_id, uint32_t dim0, uint32_t dim_size)
    {
        LogicalId x_sq = g.mul(x_id, x_id);
        int32_t axis_val = -1;
        LogicalId axis_node = g.constant({1}, &axis_val, DType::INT32);
        LogicalId sum_sq = g.sum(x_sq, axis_node);
        float n_val = (float)dim_size;
        LogicalId n_node = g.constant({1}, &n_val, DType::FLOAT32);
        n_node = expand_scalar_to_3d(n_node, dim0, seq_len, 1);
        LogicalId mean_sq = g.div(sum_sq, n_node);
        LogicalId eps_expanded = expand_scalar_to_3d(eps_fp32, dim0, seq_len, 1);
        LogicalId mean_sq_plus_eps = g.add(mean_sq, eps_expanded);
        float half_val = 0.5f;
        LogicalId sqrt_node = g.constant({1}, &half_val, DType::FLOAT32);
        sqrt_node = expand_scalar_to_3d(sqrt_node, dim0, seq_len, 1);
        LogicalId std = g.pow(mean_sq_plus_eps, sqrt_node);
        LogicalId one_node = expand_scalar_to_3d(one_fp32, dim0, seq_len, 1);
        LogicalId inv_std = g.div(one_node, std);
        LogicalId inv_std_expanded = g.repeat(inv_std, dim_size, 2);
        LogicalId x_norm = g.mul(x_id, inv_std_expanded);
        LogicalId weight_expanded = expand_1d_to_3d(weight_id, dim_size, dim0, seq_len);

        // Qwen3.5 MoE / Qwen2 RMSNorm adds 1.0 to the weights (which are
        // initialized to 0)
        LogicalId one_full = expand_scalar_to_3d(one_fp32, dim0, seq_len, dim_size);
        LogicalId weight_plus_one = g.add(weight_expanded, one_full);

        return g.mul(x_norm, weight_plus_one);
    }

    LogicalId gated_rms_norm(LogicalId x, LogicalId z, const std::string &w_name, uint32_t dims, uint32_t cur_seq_len)
    {
        LogicalId sq = g.mul(x, x);
        int32_t ax = -1;
        LogicalId sum_sq = g.sum(sq, g.constant({1}, &ax, DType::INT32));
        LogicalId mean_sq = g.div(sum_sq, expand_scalar_to_3d((float)dims, 1, cur_seq_len, 1));
        LogicalId var = g.add(mean_sq, expand_scalar_to_3d(1e-6f, 1, cur_seq_len, 1));
        LogicalId std = g.pow(var, expand_scalar_to_3d(0.5f, 1, cur_seq_len, 1));
        LogicalId inv_std = g.repeat(g.div(expand_scalar_to_3d(1.0f, 1, cur_seq_len, 1), std), dims, 2);
        LogicalId x_norm = g.mul(x, inv_std);

        LogicalId w = weight(w_path, w_name);
        int32_t sh3[] = {1, 1, (int32_t)dims};
        LogicalId w_exp = g.repeat(g.reshape(w, g.constant({3}, sh3, DType::INT32)), cur_seq_len, 1);

        LogicalId x_norm_scaled = g.mul(x_norm, w_exp);

        // Apply SiLU on gate z
        LogicalId z_silu = silu_atomic(z, 1, cur_seq_len, dims);

        return g.mul(x_norm_scaled, z_silu);
    }

    std::tuple<LogicalId, LogicalId> compute_rope()
    {
        const int32_t zero = 0;
        const int32_t one_i = 1;
        const int32_t two_i = 2;
        const int32_t rope_dim_i = (int32_t)cfg.rope_dim;       // 64
        const int32_t half_dim_i = (int32_t)(cfg.rope_dim / 2); // 32
        const int32_t seq_len_i = (int32_t)seq_len;

        LogicalId zero_node = g.constant({1}, &zero, DType::INT32);
        LogicalId one_int_node = g.constant({1}, &one_i, DType::INT32);
        LogicalId two_node = g.constant({1}, &two_i, DType::INT32);
        LogicalId rope_dim_int = g.constant({1}, &rope_dim_i, DType::INT32);

        // -------- 1. inv_freq[i] = 1 / theta^(2i/rope_dim), shape [32] --------
        LogicalId indices_int = g.arange(zero_node, rope_dim_int, two_node);
        LogicalId indices = g.cast(indices_int, DType::FLOAT32);

        float rope_dim_f = (float)cfg.rope_dim;
        LogicalId rope_dim_fp = g.constant({1}, &rope_dim_f, DType::FLOAT32);
        int32_t rep32[] = {half_dim_i};
        LogicalId rope_dim_fp_1d =
            g.repeat(rope_dim_fp, g.constant({1}, rep32, DType::INT32), g.constant({1}, &zero, DType::INT32));

        LogicalId exponent = g.div(indices, rope_dim_fp_1d);

        float theta_val = cfg.rope_theta;
        LogicalId theta = g.constant({1}, &theta_val, DType::FLOAT32);
        LogicalId theta_1d =
            g.repeat(theta, g.constant({1}, rep32, DType::INT32), g.constant({1}, &zero, DType::INT32));
        LogicalId base_to_exponent = g.pow(theta_1d, exponent);

        float one_val = 1.0f;
        LogicalId one_node = g.constant({1}, &one_val, DType::FLOAT32);
        LogicalId one_1d =
            g.repeat(one_node, g.constant({1}, rep32, DType::INT32), g.constant({1}, &zero, DType::INT32));
        LogicalId inv_freq = g.div(one_1d, base_to_exponent); // [32]

        // -------- 2. Build the three M-RoPE position axes --------
        LogicalId seq_len_int = g.constant({1}, &seq_len_i, DType::INT32);
        LogicalId t_pos_int = g.arange(zero_node, seq_len_int, one_int_node);
        LogicalId t_pos = g.cast(t_pos_int, DType::FLOAT32);

        // For 1D text generation, the spatial features track the temporal sequence identically.
        LogicalId h_pos = t_pos;
        LogicalId w_pos = t_pos;

        // -------- 3. Compute full 32-element angles for T, H, W --------
        auto outer_full = [&](LogicalId pos_1d) -> LogicalId {
            // [seq_len] → [1, seq_len, 1], repeat on axis 2 → [1, seq_len, half_dim]
            int32_t pos_shape[] = {1, seq_len_i, 1};
            LogicalId pos_col = g.reshape(pos_1d, g.constant({3}, pos_shape, DType::INT32));
            LogicalId pos_expanded = g.repeat(pos_col, half_dim_i, 2);

            // [half_dim] → [1, 1, half_dim], repeat on axis 1 → [1, seq_len,
            // half_dim]
            int32_t freq_shape[] = {1, 1, half_dim_i};
            LogicalId freq_row = g.reshape(inv_freq, g.constant({3}, freq_shape, DType::INT32));
            LogicalId freq_expanded = g.repeat(freq_row, seq_len_i, 1);

            return g.mul(pos_expanded, freq_expanded); // [1, seq_len, half_dim]
        };

        LogicalId angles_t_full = outer_full(t_pos);
        LogicalId angles_h_full = outer_full(h_pos);
        LogicalId angles_w_full = outer_full(w_pos);

        // -------- 4. Interleave index-by-index to match index % 3 logic --------
        std::vector<LogicalId> interleaved_slices;
        int32_t steps_slice[] = {1, 1, 1}; // Slice from 3D tensor
        LogicalId steps_slice_node = g.constant({3}, steps_slice, DType::INT32);

        for (int32_t i = 0; i < half_dim_i; ++i)
        {
            int32_t starts[] = {0, 0, i};
            int32_t ends[] = {1, seq_len_i, i + 1};
            LogicalId starts_node = g.constant({3}, starts, DType::INT32);
            LogicalId ends_node = g.constant({3}, ends, DType::INT32);

            LogicalId source_angles = angles_t_full;
            if (i % 3 == 1)
            {
                source_angles = angles_h_full;
            }
            else if (i % 3 == 2)
            {
                source_angles = angles_w_full;
            }

            LogicalId slice_i = g.contiguous(g.slice(source_angles, starts_node, ends_node, steps_slice_node));
            interleaved_slices.push_back(slice_i);
        }

        // -------- 5. Concatenate interleaved elements along the channel axis
        // --------
        int32_t ax2_concat = 2;
        LogicalId angles_half = g.concat(interleaved_slices, g.constant({1}, &ax2_concat, DType::INT32));

        // -------- 6. cos/sin on the interleaved half-angles --------
        LogicalId cos_half = g.cos(angles_half);
        LogicalId sin_half = g.sin(angles_half);

        // -------- 7. Concat halves to match rotate_half --------
        LogicalId cos_out = g.concat({cos_half, cos_half}, g.constant({1}, &ax2_concat, DType::INT32));
        LogicalId sin_out = g.concat({sin_half, sin_half}, g.constant({1}, &ax2_concat, DType::INT32));

        return {cos_out, sin_out};
    }

    LogicalId apply_rope(LogicalId x_id, LogicalId cos_id, LogicalId sin_id, uint32_t n_groups, uint32_t head_dim)
    {
        uint32_t rope_dim = cfg.rope_dim; // 64
        uint32_t half_dim = rope_dim / 2; // 32
        int32_t seq_len_i = (int32_t)seq_len;
        int32_t n_groups_i = (int32_t)n_groups;
        int32_t rope_dim_i = (int32_t)rope_dim;
        int32_t half_dim_i = (int32_t)half_dim;
        int32_t head_dim_i = (int32_t)head_dim;

        int32_t steps_111[] = {1, 1, 1};
        LogicalId steps_111_node = g.constant({3}, steps_111, DType::INT32);

        // -------- 1. Slice out the rotary portion of x --------
        int32_t starts_rope[] = {0, 0, 0};
        int32_t ends_rope[] = {n_groups_i, seq_len_i, rope_dim_i};
        LogicalId x_rope = g.contiguous(g.slice(x_id, g.constant({3}, starts_rope, DType::INT32),
                                                g.constant({3}, ends_rope, DType::INT32), steps_111_node));

        // -------- 2. Slice x_rope into first and second half elements --------
        // x_first = x_rope[..., :32]
        int32_t starts_first[] = {0, 0, 0};
        int32_t ends_first[] = {n_groups_i, seq_len_i, half_dim_i};
        LogicalId x_first = g.contiguous(g.slice(x_rope, g.constant({3}, starts_first, DType::INT32),
                                                 g.constant({3}, ends_first, DType::INT32), steps_111_node));

        // x_second = x_rope[..., 32:64]
        int32_t starts_second[] = {0, 0, half_dim_i};
        int32_t ends_second[] = {n_groups_i, seq_len_i, rope_dim_i};
        LogicalId x_second = g.contiguous(g.slice(x_rope, g.constant({3}, starts_second, DType::INT32),
                                                  g.constant({3}, ends_second, DType::INT32), steps_111_node));

        // -------- 3. Build the standard rotated tensor: [-x_second, x_first]
        // --------
        LogicalId neg_x_second = g.neg(x_second);
        int32_t ax2 = 2;
        LogicalId rotated = g.concat({neg_x_second, x_first}, g.constant({1}, &ax2, DType::INT32));

        // -------- 4. Broadcast cos/sin over the head axis --------
        LogicalId cos_expanded = g.repeat(cos_id, n_groups, 0);
        LogicalId sin_expanded = g.repeat(sin_id, n_groups, 0);

        // -------- 5. Apply rotation: out = x_rope * cos + rotated * sin --------
        LogicalId term1 = g.mul(x_rope, cos_expanded);
        LogicalId term2 = g.mul(rotated, sin_expanded);
        LogicalId x_rope_applied = g.add(term1, term2);

        // -------- 6. Pass through the non-rotary portion (rope_dim < head_dim)
        // --------
        if (rope_dim < head_dim)
        {
            int32_t starts_pass[] = {0, 0, rope_dim_i};
            int32_t ends_pass[] = {n_groups_i, seq_len_i, head_dim_i};
            LogicalId x_pass = g.contiguous(g.slice(x_id, g.constant({3}, starts_pass, DType::INT32),
                                                    g.constant({3}, ends_pass, DType::INT32), steps_111_node));
            int32_t ax2_pass = 2;
            return g.concat({x_rope_applied, x_pass}, g.constant({1}, &ax2_pass, DType::INT32));
        }
        return x_rope_applied;
    }

    LogicalId compute_causal_mask()
    {
        int32_t mask_shape[] = {(int32_t)seq_len, (int32_t)seq_len};
        LogicalId mask_shape_node = g.constant({2}, mask_shape, DType::INT32);
        float one_val = 1.0f;
        LogicalId ones_matrix = g.fill(g.constant({1}, &one_val, DType::FLOAT32), mask_shape_node);
        int32_t k_val = 1;
        LogicalId triu_mask = g.triu(ones_matrix, g.constant({1}, &k_val, DType::INT32));
        float neg_inf_val = -1e9f;
        LogicalId neg_inf_node = g.constant({1}, &neg_inf_val, DType::FLOAT32);
        int32_t neg_inf_shape[] = {1, 1};
        LogicalId neg_inf_reshaped = g.reshape(neg_inf_node, g.constant({2}, neg_inf_shape, DType::INT32));
        LogicalId neg_inf_expanded = g.repeat(neg_inf_reshaped, seq_len, 0);
        neg_inf_expanded = g.repeat(neg_inf_expanded, seq_len, 1);
        LogicalId scaled_mask = g.mul(triu_mask, neg_inf_expanded);
        int32_t final_shape[] = {1, (int32_t)seq_len, (int32_t)seq_len};
        return g.reshape(scaled_mask, g.constant({3}, final_shape, DType::INT32));
    }

    // --- GATED ATTENTION (FULL ATTENTION LAYER) ---
    std::tuple<LogicalId, LogicalId, LogicalId, LogicalId> gated_attention_qkv_atomic(LogicalId x,
                                                                                      const std::string &prefix,
                                                                                      LogicalId rope_cos,
                                                                                      LogicalId rope_sin)
    {
        int32_t perm_dims[] = {1, 0};
        LogicalId dims_node = g.constant({2}, perm_dims, DType::INT32);

        auto project = [&](const std::string &suffix, uint32_t in_d, uint32_t out_d) {
            LogicalId w = weight(w_path, prefix + suffix);
            LogicalId w_t = g.permute(w, dims_node);
            w_t = g.contiguous(w_t);
            int32_t s3[] = {1, (int32_t)in_d, (int32_t)out_d};
            return g.dot(x, g.reshape(w_t, g.constant({3}, s3, DType::INT32)));
        };

        // q_proj provides Q and Gate
        LogicalId q_and_gate =
            project(".self_attn.q_proj.weight", cfg.hidden_size, cfg.attn_n_q_heads * cfg.head_dim * 2);

        // Reshape to 4D to isolate the head_dim * 2 structure [1, seq_len,
        // num_heads, head_dim * 2]
        int32_t q_and_gate_shape4[] = {1, (int32_t)seq_len, (int32_t)cfg.attn_n_q_heads, (int32_t)cfg.head_dim * 2};
        LogicalId q_and_gate_4d = g.reshape(q_and_gate, g.constant({4}, q_and_gate_shape4, DType::INT32));

        // Slice Q and Gate from the last axis (axis 3) of the 4D tensor
        int32_t s_q[] = {0, 0, 0, 0};
        int32_t e_q[] = {1, (int32_t)seq_len, (int32_t)cfg.attn_n_q_heads, (int32_t)cfg.head_dim};
        int32_t steps[] = {1, 1, 1, 1};
        LogicalId q_4d =
            g.contiguous(g.slice(q_and_gate_4d, g.constant({4}, s_q, DType::INT32), g.constant({4}, e_q, DType::INT32),
                                 g.constant({4}, steps, DType::INT32)));

        int32_t s_g[] = {0, 0, 0, (int32_t)cfg.head_dim};
        int32_t e_g[] = {1, (int32_t)seq_len, (int32_t)cfg.attn_n_q_heads, (int32_t)cfg.head_dim * 2};
        LogicalId gate_4d =
            g.contiguous(g.slice(q_and_gate_4d, g.constant({4}, s_g, DType::INT32), g.constant({4}, e_g, DType::INT32),
                                 g.constant({4}, steps, DType::INT32)));

        // Reshape Gate back to 3D: [1, seq_len, num_heads * head_dim]
        int32_t gate_shape3[] = {1, (int32_t)seq_len, (int32_t)(cfg.attn_n_q_heads * cfg.head_dim)};
        LogicalId gate = g.reshape(gate_4d, g.constant({3}, gate_shape3, DType::INT32));

        int32_t perm4[] = {0, 2, 1, 3};
        LogicalId perm4_node = g.constant({4}, perm4, DType::INT32);

        // Permute Q and reshape to 3D
        LogicalId q_perm = g.permute(q_4d, perm4_node);
        q_perm = g.contiguous(q_perm);
        int32_t shape3_q[] = {(int32_t)cfg.attn_n_q_heads, (int32_t)seq_len, (int32_t)cfg.head_dim};
        LogicalId q = g.reshape(q_perm, g.constant({3}, shape3_q, DType::INT32));

        LogicalId k = project(".self_attn.k_proj.weight", cfg.hidden_size, cfg.attn_n_kv_heads * cfg.head_dim);
        LogicalId v = project(".self_attn.v_proj.weight", cfg.hidden_size, cfg.attn_n_kv_heads * cfg.head_dim);

        int32_t k_shape4[] = {1, (int32_t)seq_len, (int32_t)cfg.attn_n_kv_heads, (int32_t)cfg.head_dim};
        LogicalId k_4d = g.reshape(k, g.constant({4}, k_shape4, DType::INT32));
        LogicalId k_perm = g.permute(k_4d, perm4_node);
        k_perm = g.contiguous(k_perm);
        int32_t shape3_k[] = {(int32_t)cfg.attn_n_kv_heads, (int32_t)seq_len, (int32_t)cfg.head_dim};
        k = g.reshape(k_perm, g.constant({3}, shape3_k, DType::INT32));

        LogicalId v_4d = g.reshape(v, g.constant({4}, k_shape4, DType::INT32));
        LogicalId v_perm = g.permute(v_4d, perm4_node);
        v_perm = g.contiguous(v_perm);
        v = g.reshape(v_perm, g.constant({3}, shape3_k, DType::INT32));

        LogicalId q_norm_w = weight(w_path, prefix + ".self_attn.q_norm.weight");
        q = rms_norm_atomic(q, q_norm_w, cfg.attn_n_q_heads, cfg.head_dim);
        LogicalId k_norm_w = weight(w_path, prefix + ".self_attn.k_norm.weight");
        k = rms_norm_atomic(k, k_norm_w, cfg.attn_n_kv_heads, cfg.head_dim);

        q = apply_rope(q, rope_cos, rope_sin, cfg.attn_n_q_heads, cfg.head_dim);
        k = apply_rope(k, rope_cos, rope_sin, cfg.attn_n_kv_heads, cfg.head_dim);

        if (cfg.attn_n_q_heads != cfg.attn_n_kv_heads)
        {
            uint32_t repeats = cfg.attn_n_q_heads / cfg.attn_n_kv_heads;

            // 1. Reshape [n_kv_heads, seq_len, head_dim] -> [n_kv_heads, 1, seq_len,
            // head_dim]
            int32_t sh4[] = {(int32_t)cfg.attn_n_kv_heads, 1, (int32_t)seq_len, (int32_t)cfg.head_dim};
            LogicalId sh4_node = g.constant({4}, sh4, DType::INT32);
            k = g.reshape(k, sh4_node);
            v = g.reshape(v, sh4_node);

            // 2. Repeat the newly inserted axis (axis 1) of size 1 by 'repeats'
            int32_t rep[] = {(int32_t)repeats};
            LogicalId rep_node = g.constant({1}, rep, DType::INT32);
            int32_t ax[] = {1};
            LogicalId ax_node = g.constant({1}, ax, DType::INT32);
            k = g.repeat(k, rep_node, ax_node);
            v = g.repeat(v, rep_node, ax_node);

            // 3. Materialize the zero-stride view into contiguous memory
            k = g.contiguous(k);
            v = g.contiguous(v);

            // 4. Reshape back to 3D: [n_q_heads, seq_len, head_dim]
            int32_t sh3[] = {(int32_t)cfg.attn_n_q_heads, (int32_t)seq_len, (int32_t)cfg.head_dim};
            LogicalId sh3_node = g.constant({3}, sh3, DType::INT32);
            k = g.reshape(k, sh3_node);
            v = g.reshape(v, sh3_node);
        }
        return std::make_tuple(g.contiguous(q), g.contiguous(k), g.contiguous(v), gate);
    }

    LogicalId gated_attention_output_atomic(std::tuple<LogicalId, LogicalId, LogicalId, LogicalId> qkvg,
                                            const std::string &prefix, LogicalId mask_id)
    {
        LogicalId q = std::get<0>(qkvg);
        LogicalId k = std::get<1>(qkvg);
        LogicalId v = std::get<2>(qkvg);
        LogicalId gate = std::get<3>(qkvg);

        float scale_val = 1.0f / std::sqrt((float)cfg.head_dim);
        LogicalId scale_node =
            expand_scalar_to_3d(g.constant({1}, &scale_val, DType::FLOAT32), cfg.attn_n_q_heads, seq_len, cfg.head_dim);
        LogicalId scaled_q = g.mul(q, scale_node);

        int32_t perm_k[] = {0, 2, 1};
        LogicalId k_t = g.permute(k, g.constant({3}, perm_k, DType::INT32));
        k_t = g.contiguous(k_t);

        LogicalId scores = g.dot(scaled_q, k_t);
        LogicalId mask_expanded = g.repeat(mask_id, cfg.attn_n_q_heads, 0);
        scores = g.add(scores, mask_expanded);

        int32_t axis_val = -1;
        LogicalId max_scores = g.max(scores, g.constant({1}, &axis_val, DType::INT32));
        max_scores = g.repeat(max_scores, seq_len, 2);
        LogicalId shifted_scores = g.add(scores, g.neg(max_scores));

        float e_val = 2.718281828459045f;
        LogicalId e_node =
            expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), cfg.attn_n_q_heads, seq_len, seq_len);
        LogicalId exp_scores = g.pow(e_node, shifted_scores);

        LogicalId sum_exp = g.sum(exp_scores, g.constant({1}, &axis_val, DType::INT32));
        sum_exp = g.repeat(sum_exp, seq_len, 2);

        LogicalId probs = g.div(exp_scores, sum_exp);
        LogicalId context = g.dot(probs, v);

        int32_t ctx_shape4[] = {1, (int32_t)cfg.attn_n_q_heads, (int32_t)seq_len, (int32_t)cfg.head_dim};
        LogicalId ctx_4d = g.reshape(context, g.constant({4}, ctx_shape4, DType::INT32));

        int32_t perm_ctx[] = {0, 2, 1, 3};
        LogicalId ctx_perm = g.permute(ctx_4d, g.constant({4}, perm_ctx, DType::INT32));
        ctx_perm = g.contiguous(ctx_perm);

        int32_t ctx_shape3[] = {1, (int32_t)seq_len, (int32_t)(cfg.attn_n_q_heads * cfg.head_dim)};
        LogicalId ctx_flat = g.reshape(ctx_perm, g.constant({3}, ctx_shape3, DType::INT32));

        // multiply by sigmoid(gate)
        float neg_one_val = -1.0f;
        LogicalId neg_one = expand_scalar_to_3d(g.constant({1}, &neg_one_val, DType::FLOAT32), 1, seq_len,
                                                cfg.attn_n_q_heads * cfg.head_dim);
        LogicalId neg_gate = g.mul(gate, neg_one);
        LogicalId e_node_gate =
            expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), 1, seq_len, cfg.attn_n_q_heads * cfg.head_dim);
        LogicalId exp_neg_gate = g.pow(e_node_gate, neg_gate);
        LogicalId one_node = expand_scalar_to_3d(one_fp32, 1, seq_len, cfg.attn_n_q_heads * cfg.head_dim);
        LogicalId den = g.add(one_node, exp_neg_gate);
        LogicalId sigmoid_gate = g.div(one_node, den);

        ctx_flat = g.mul(ctx_flat, sigmoid_gate);

        LogicalId w_o = weight(w_path, prefix + ".self_attn.o_proj.weight");
        int32_t perm_dims[] = {1, 0};
        LogicalId w_o_t = g.permute(w_o, g.constant({2}, perm_dims, DType::INT32));
        w_o_t = g.contiguous(w_o_t);

        int32_t s3[] = {1, (int32_t)(cfg.attn_n_q_heads * cfg.head_dim), (int32_t)cfg.hidden_size};
        LogicalId w_o_3d = g.reshape(w_o_t, g.constant({3}, s3, DType::INT32));
        return g.dot(ctx_flat, w_o_3d);
    }

    /*
    --- GATED DELTANET (LINEAR ATTENTION LAYER) ---
    x [B, seq_len, hidden_size]
    */
    LogicalId linear_attention_atomic(LogicalId x, const std::string &prefix)
    {
        // no mask of padding states L443
        int32_t perm_dims[] = {1, 0};
        LogicalId dims_node = g.constant({2}, perm_dims, DType::INT32);

        auto project_tensor = [&](LogicalId input_tensor, const std::string &suffix, uint32_t in_d, uint32_t out_d) {
            LogicalId w = weight(w_path, prefix + suffix);
            LogicalId w_t = g.permute(w, dims_node);
            w_t = g.contiguous(w_t);
            int32_t s3[] = {1, (int32_t)in_d, (int32_t)out_d};
            return g.dot(input_tensor, g.reshape(w_t, g.constant({3}, s3, DType::INT32)));
        };

        auto project = [&](const std::string &suffix, uint32_t in_d, uint32_t out_d) {
            return project_tensor(x, suffix, in_d, out_d);
        };

        // 1. Projections
        uint32_t key_dim = cfg.linear_n_qk_heads * cfg.linear_head_dim;
        uint32_t value_dim = cfg.linear_n_v_heads * cfg.linear_head_dim;
        uint32_t qkv_dim = key_dim * 2 + value_dim;

        LogicalId mixed_qkv = project(".linear_attn.in_proj_qkv.weight", cfg.hidden_size, qkv_dim);
        LogicalId z = project(".linear_attn.in_proj_z.weight", cfg.hidden_size, value_dim);
        LogicalId b = project(".linear_attn.in_proj_b.weight", cfg.hidden_size, cfg.linear_n_v_heads);
        LogicalId a = project(".linear_attn.in_proj_a.weight", cfg.hidden_size, cfg.linear_n_v_heads);

        // 2. Causal 1D Convolution
        int32_t perm_conv[] = {0, 2, 1};
        LogicalId mixed_qkv_tr = g.contiguous(g.permute(mixed_qkv, g.constant({3}, perm_conv, DType::INT32)));

        float zero_val = 0.0f;
        int32_t pad_shape[] = {1, (int32_t)qkv_dim, 3};
        LogicalId pad_zeros =
            g.fill(g.constant({1}, &zero_val, DType::FLOAT32), g.constant({3}, pad_shape, DType::INT32));

        int32_t ax2 = 2;
        LogicalId padded = g.concat({pad_zeros, mixed_qkv_tr}, g.constant({1}, &ax2, DType::INT32));

        int32_t steps[] = {1, 1, 1};

        int32_t starts0[] = {0, 0, 3};
        int32_t ends0[] = {1, (int32_t)qkv_dim, (int32_t)(seq_len + 3)};
        LogicalId padded_t0 = g.slice(padded, g.constant({3}, starts0, DType::INT32),
                                      g.constant({3}, ends0, DType::INT32), g.constant({3}, steps, DType::INT32));

        int32_t starts1[] = {0, 0, 2};
        int32_t ends1[] = {1, (int32_t)qkv_dim, (int32_t)(seq_len + 2)};
        LogicalId padded_t1 = g.slice(padded, g.constant({3}, starts1, DType::INT32),
                                      g.constant({3}, ends1, DType::INT32), g.constant({3}, steps, DType::INT32));

        int32_t starts2[] = {0, 0, 1};
        int32_t ends2[] = {1, (int32_t)qkv_dim, (int32_t)(seq_len + 1)};
        LogicalId padded_t2 = g.slice(padded, g.constant({3}, starts2, DType::INT32),
                                      g.constant({3}, ends2, DType::INT32), g.constant({3}, steps, DType::INT32));

        int32_t starts3[] = {0, 0, 0};
        int32_t ends3[] = {1, (int32_t)qkv_dim, (int32_t)seq_len};
        LogicalId padded_t3 = g.slice(padded, g.constant({3}, starts3, DType::INT32),
                                      g.constant({3}, ends3, DType::INT32), g.constant({3}, steps, DType::INT32));

        LogicalId conv_w = weight(w_path, prefix + ".linear_attn.conv1d.weight");

        int32_t w_reshape_shape[] = {1, (int32_t)qkv_dim, 1};
        LogicalId w_reshape = g.constant({3}, w_reshape_shape, DType::INT32);

        int32_t w_steps[] = {1, 1, 1};

        int32_t ws0[] = {0, 0, 0};
        int32_t we0[] = {(int32_t)qkv_dim, 1, 1};
        LogicalId w0 =
            g.reshape(g.contiguous(g.slice(conv_w, g.constant({3}, ws0, DType::INT32),
                                           g.constant({3}, we0, DType::INT32), g.constant({3}, w_steps, DType::INT32))),
                      w_reshape);

        int32_t ws1[] = {0, 0, 1};
        int32_t we1[] = {(int32_t)qkv_dim, 1, 2};
        LogicalId w1 =
            g.reshape(g.contiguous(g.slice(conv_w, g.constant({3}, ws1, DType::INT32),
                                           g.constant({3}, we1, DType::INT32), g.constant({3}, w_steps, DType::INT32))),
                      w_reshape);

        int32_t ws2[] = {0, 0, 2};
        int32_t we2[] = {(int32_t)qkv_dim, 1, 3};
        LogicalId w2 =
            g.reshape(g.contiguous(g.slice(conv_w, g.constant({3}, ws2, DType::INT32),
                                           g.constant({3}, we2, DType::INT32), g.constant({3}, w_steps, DType::INT32))),
                      w_reshape);

        int32_t ws3[] = {0, 0, 3};
        int32_t we3[] = {(int32_t)qkv_dim, 1, 4};
        LogicalId w3 =
            g.reshape(g.contiguous(g.slice(conv_w, g.constant({3}, ws3, DType::INT32),
                                           g.constant({3}, we3, DType::INT32), g.constant({3}, w_steps, DType::INT32))),
                      w_reshape);

        LogicalId w0_exp = g.repeat(w0, seq_len, 2);
        LogicalId w1_exp = g.repeat(w1, seq_len, 2);
        LogicalId w2_exp = g.repeat(w2, seq_len, 2);
        LogicalId w3_exp = g.repeat(w3, seq_len, 2);

        LogicalId term0 = g.mul(padded_t3, w0_exp);
        LogicalId term1 = g.mul(padded_t2, w1_exp);
        LogicalId term2 = g.mul(padded_t1, w2_exp);
        LogicalId term3 = g.mul(padded_t0, w3_exp);

        LogicalId conv_combined = g.add(g.add(g.add(term0, term1), term2), term3);

        int32_t perm_back[] = {0, 2, 1};
        LogicalId conv_out_tr = g.contiguous(g.permute(conv_combined, g.constant({3}, perm_back, DType::INT32)));

        conv_out_tr = silu_atomic(conv_out_tr, 1, seq_len, qkv_dim);

        int32_t starts_q[] = {0, 0, 0};
        int32_t ends_q[] = {1, (int32_t)seq_len, (int32_t)key_dim};
        LogicalId q =
            g.contiguous(g.slice(conv_out_tr, g.constant({3}, starts_q, DType::INT32),
                                 g.constant({3}, ends_q, DType::INT32), g.constant({3}, steps, DType::INT32)));

        int32_t starts_k[] = {0, 0, (int32_t)key_dim};
        int32_t ends_k[] = {1, (int32_t)seq_len, (int32_t)(key_dim * 2)};
        LogicalId k =
            g.contiguous(g.slice(conv_out_tr, g.constant({3}, starts_k, DType::INT32),
                                 g.constant({3}, ends_k, DType::INT32), g.constant({3}, steps, DType::INT32)));

        int32_t starts_v[] = {0, 0, (int32_t)(key_dim * 2)};
        int32_t ends_v[] = {1, (int32_t)seq_len, (int32_t)qkv_dim};
        LogicalId v =
            g.contiguous(g.slice(conv_out_tr, g.constant({3}, starts_v, DType::INT32),
                                 g.constant({3}, ends_v, DType::INT32), g.constant({3}, steps, DType::INT32)));

        // 3. Compute beta and g (decay alpha)
        LogicalId beta = sigmoid(b, cfg.linear_n_v_heads);

        LogicalId A_log = weight(w_path, prefix + ".linear_attn.A_log");
        LogicalId dt_bias = weight(w_path, prefix + ".linear_attn.dt_bias");

        int32_t ba_reshape_shape[] = {1, 1, (int32_t)cfg.linear_n_v_heads};
        LogicalId dt_bias_3d =
            g.repeat(g.reshape(dt_bias, g.constant({3}, ba_reshape_shape, DType::INT32)), seq_len, 1);
        LogicalId a_plus_dt_bias = g.add(a, dt_bias_3d);

        LogicalId exp_x = g.pow(expand_scalar_to_3d(2.7182818f, 1, seq_len, cfg.linear_n_v_heads), a_plus_dt_bias);
        LogicalId one_plus_exp = g.add(expand_scalar_to_3d(1.0f, 1, seq_len, cfg.linear_n_v_heads), exp_x);
        LogicalId softplus_x = g.log(one_plus_exp);

        LogicalId A_log_exp = g.pow(expand_scalar_to_1d(2.7182818f, cfg.linear_n_v_heads), A_log);
        LogicalId A_log_exp_3d =
            g.repeat(g.reshape(A_log_exp, g.constant({3}, ba_reshape_shape, DType::INT32)), seq_len, 1);
        LogicalId decay_g = g.mul(g.neg(A_log_exp_3d), softplus_x);
        LogicalId decay_alpha = g.pow(expand_scalar_to_3d(2.7182818f, 1, seq_len, cfg.linear_n_v_heads), decay_g);

        // 4. Reshape Q, K, V to head-based layout and L2 Norm
        int32_t perm_heads[] = {0, 2, 1, 3};
        int32_t q_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.linear_n_qk_heads, (int32_t)cfg.linear_head_dim};
        LogicalId q_heads = g.contiguous(
            g.permute(g.reshape(q, g.constant({4}, q_shape, DType::INT32)), g.constant({4}, perm_heads, DType::INT32)));

        int32_t k_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.linear_n_qk_heads, (int32_t)cfg.linear_head_dim};
        LogicalId k_heads = g.contiguous(
            g.permute(g.reshape(k, g.constant({4}, k_shape, DType::INT32)), g.constant({4}, perm_heads, DType::INT32)));

        int32_t v_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.linear_n_v_heads, (int32_t)cfg.linear_head_dim};
        LogicalId v_heads = g.contiguous(
            g.permute(g.reshape(v, g.constant({4}, v_shape, DType::INT32)), g.constant({4}, perm_heads, DType::INT32)));

        int32_t b_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.linear_n_v_heads, 1};
        LogicalId b_heads = g.contiguous(g.permute(g.reshape(beta, g.constant({4}, b_shape, DType::INT32)),
                                                   g.constant({4}, perm_heads, DType::INT32)));
        LogicalId a_heads = g.contiguous(g.permute(g.reshape(decay_alpha, g.constant({4}, b_shape, DType::INT32)),
                                                   g.constant({4}, perm_heads, DType::INT32)));

        // 1. Reshape [1, 16, 8, 128] -> [1, 16, 1, 8, 128] to insert a unit
        // dimension at axis 2
        int32_t sh5[] = {1, (int32_t)cfg.linear_n_qk_heads, 1, (int32_t)seq_len, (int32_t)cfg.linear_head_dim};
        LogicalId sh5_node = g.constant({5}, sh5, DType::INT32);
        LogicalId q_heads_5d = g.reshape(q_heads, sh5_node);
        LogicalId k_heads_5d = g.reshape(k_heads, sh5_node);

        // 2. Repeat along the newly inserted axis (axis 2) of size 1 by r_heads (2)
        int32_t r_heads = cfg.linear_n_v_heads / cfg.linear_n_qk_heads;
        int32_t rep[] = {(int32_t)r_heads};
        LogicalId rep_node = g.constant({1}, rep, DType::INT32);
        int32_t ax[] = {2};
        LogicalId ax_node = g.constant({1}, ax, DType::INT32);

        LogicalId q_heads_rep = g.repeat(q_heads_5d, rep_node, ax_node);
        LogicalId k_heads_rep = g.repeat(k_heads_5d, rep_node, ax_node);

        // 3. Materialize the zero-stride view into contiguous memory
        LogicalId q_heads_contig = g.contiguous(q_heads_rep);
        LogicalId k_heads_contig = g.contiguous(k_heads_rep);

        // 4. Reshape back to 4D: [1, 32, 8, 128], collapsing the [16, 2] dimensions
        // into 32
        int32_t sh4_target[] = {1, (int32_t)cfg.linear_n_v_heads, (int32_t)seq_len, (int32_t)cfg.linear_head_dim};
        LogicalId sh4_target_node = g.constant({4}, sh4_target, DType::INT32);

        LogicalId q_heads_exp = g.reshape(q_heads_contig, sh4_target_node);
        LogicalId k_heads_exp = g.reshape(k_heads_contig, sh4_target_node);

        int32_t ax_neg1 = -1;
        LogicalId q_sq = g.mul(q_heads_exp, q_heads_exp);
        LogicalId q_sum = g.sum(q_sq, g.constant({1}, &ax_neg1, DType::INT32));
        LogicalId q_std = g.pow(g.add(q_sum, expand_scalar_to_4d(1e-6f, 1, (int32_t)cfg.linear_n_v_heads, seq_len, 1)),
                                expand_scalar_to_4d(0.5f, 1, (int32_t)cfg.linear_n_v_heads, seq_len, 1));
        LogicalId q_norm = g.mul(
            q_heads_exp, g.repeat(g.div(expand_scalar_to_4d(1.0f, 1, (int32_t)cfg.linear_n_v_heads, seq_len, 1), q_std),
                                  (int32_t)cfg.linear_head_dim, 3));

        float scale_factor = 1.0f / std::sqrt((float)cfg.linear_head_dim);
        q_norm = g.mul(q_norm, expand_scalar_to_4d(scale_factor, 1, (int32_t)cfg.linear_n_v_heads, seq_len,
                                                   (int32_t)cfg.linear_head_dim));

        LogicalId k_sq = g.mul(k_heads_exp, k_heads_exp);
        LogicalId k_sum = g.sum(k_sq, g.constant({1}, &ax_neg1, DType::INT32));
        LogicalId k_std = g.pow(g.add(k_sum, expand_scalar_to_4d(1e-6f, 1, (int32_t)cfg.linear_n_v_heads, seq_len, 1)),
                                expand_scalar_to_4d(0.5f, 1, (int32_t)cfg.linear_n_v_heads, seq_len, 1));
        LogicalId k_norm = g.mul(
            k_heads_exp, g.repeat(g.div(expand_scalar_to_4d(1.0f, 1, (int32_t)cfg.linear_n_v_heads, seq_len, 1), k_std),
                                  (int32_t)cfg.linear_head_dim, 3));

        // 5. Gated Delta Rule Recurrence Loop
        int32_t s_shape[] = {(int32_t)cfg.linear_n_v_heads, (int32_t)cfg.linear_head_dim, (int32_t)cfg.linear_head_dim};
        LogicalId S = g.fill(g.constant({1}, &zero_val, DType::FLOAT32), g.constant({3}, s_shape, DType::INT32));

        std::vector<LogicalId> outs;
        for (uint32_t t = 0; t < seq_len; ++t)
        {
            int32_t starts_t_ab[] = {0, 0, (int32_t)t, 0};
            int32_t ends_t_b[] = {1, (int32_t)cfg.linear_n_v_heads, (int32_t)(t + 1), 1};
            int32_t steps_t[] = {1, 1, 1, 1};
            LogicalId a_t =
                g.contiguous(g.slice(a_heads, g.constant({4}, starts_t_ab, DType::INT32),
                                     g.constant({4}, ends_t_b, DType::INT32), g.constant({4}, steps_t, DType::INT32)));

            int32_t flat_scalar_shape[] = {(int32_t)cfg.linear_n_v_heads, 1, 1};
            LogicalId a_t_flat = g.reshape(a_t, g.constant({3}, flat_scalar_shape, DType::INT32));

            int32_t starts_t_k[] = {0, 0, (int32_t)t, 0};
            int32_t ends_t_k[] = {1, (int32_t)cfg.linear_n_v_heads, (int32_t)(t + 1), (int32_t)cfg.linear_head_dim};
            LogicalId k_t =
                g.contiguous(g.slice(k_norm, g.constant({4}, starts_t_k, DType::INT32),
                                     g.constant({4}, ends_t_k, DType::INT32), g.constant({4}, steps_t, DType::INT32)));

            int32_t flat_vector_shape[] = {(int32_t)cfg.linear_n_v_heads, 1, (int32_t)cfg.linear_head_dim};
            LogicalId k_t_flat = g.reshape(k_t, g.constant({3}, flat_vector_shape, DType::INT32));

            LogicalId v_t =
                g.contiguous(g.slice(v_heads, g.constant({4}, starts_t_k, DType::INT32),
                                     g.constant({4}, ends_t_k, DType::INT32), g.constant({4}, steps_t, DType::INT32)));
            LogicalId v_t_flat = g.reshape(v_t, g.constant({3}, flat_vector_shape, DType::INT32));

            LogicalId b_t =
                g.contiguous(g.slice(b_heads, g.constant({4}, starts_t_ab, DType::INT32),
                                     g.constant({4}, ends_t_b, DType::INT32), g.constant({4}, steps_t, DType::INT32)));
            LogicalId b_t_flat = g.reshape(b_t, g.constant({3}, flat_scalar_shape, DType::INT32));

            LogicalId q_t =
                g.contiguous(g.slice(q_norm, g.constant({4}, starts_t_k, DType::INT32),
                                     g.constant({4}, ends_t_k, DType::INT32), g.constant({4}, steps_t, DType::INT32)));
            LogicalId q_t_flat = g.reshape(q_t, g.constant({3}, flat_vector_shape, DType::INT32));

            // 1. Recall from the CURRENT (pre-decay) state
            LogicalId kv_mem = g.contiguous(g.dot(k_t_flat, S));

            // 2. Compute error and delta correction
            LogicalId err = g.add(v_t_flat, g.neg(kv_mem));
            LogicalId b_t_exp = g.repeat(b_t_flat, (int32_t)cfg.linear_head_dim, 2);
            LogicalId delta = g.mul(err, b_t_exp);

            // 3. Compute outer product k^T ⊗ delta
            int32_t perm_t[] = {0, 2, 1};
            LogicalId k_t_t = g.contiguous(g.permute(k_t_flat, g.constant({3}, perm_t, DType::INT32)));
            LogicalId outer_prod = g.contiguous(g.dot(k_t_t, delta));

            // 4. Apply decay AND write in one expression: S = g*S + outer
            LogicalId a_t_exp =
                g.repeat(g.repeat(a_t_flat, (int32_t)cfg.linear_head_dim, 1), (int32_t)cfg.linear_head_dim, 2);
            S = g.add(g.mul(S, a_t_exp), outer_prod);

            // 5. Read output from the fully updated state
            LogicalId y_t = g.contiguous(g.dot(q_t_flat, S));

            int32_t head_y_shape[] = {1, (int32_t)cfg.linear_n_v_heads, 1, (int32_t)cfg.linear_head_dim};
            LogicalId y_t_head = g.reshape(y_t, g.constant({4}, head_y_shape, DType::INT32));
            outs.push_back(y_t_head);
        }

        LogicalId context_heads;
        if (seq_len > 1)
        {
            int32_t ax2_concat = 2;
            context_heads = g.concat(outs, g.constant({1}, &ax2_concat, DType::INT32));
        }
        else
        {
            context_heads = outs[0];
        }

        int32_t perm_heads_back[] = {0, 2, 1, 3};
        LogicalId context_perm = g.contiguous(g.permute(context_heads, g.constant({4}, perm_heads_back, DType::INT32)));

        int32_t final_context_shape[] = {1, (int32_t)seq_len, (int32_t)value_dim};

        // 6. Gated RMSNorm and Final Output Projection
        // Reshape context and z to head-wise format: [1, seq_len *
        // linear_n_v_heads, linear_head_dim]
        int32_t head_format_shape[] = {1, (int32_t)(seq_len * cfg.linear_n_v_heads), (int32_t)cfg.linear_head_dim};
        LogicalId head_format_shape_node = g.constant({3}, head_format_shape, DType::INT32);

        LogicalId context_flat = g.reshape(context_perm, head_format_shape_node);
        LogicalId z_flat = g.reshape(z, head_format_shape_node);

        // Apply Gated RMSNorm on the head-wise format
        LogicalId norm_flat = gated_rms_norm(context_flat, z_flat, prefix + ".linear_attn.norm.weight",
                                             cfg.linear_head_dim, seq_len * cfg.linear_n_v_heads);

        // Reshape back to value_dim [1, seq_len, value_dim]
        LogicalId output = g.reshape(norm_flat, g.constant({3}, final_context_shape, DType::INT32));

        return project_tensor(output, ".linear_attn.out_proj.weight", value_dim, cfg.hidden_size);
    }

    // --- MIXTURE OF EXPERTS (MOE) MLP LAYER ---
    LogicalId mlp_moe_atomic(LogicalId x, const std::string &prefix)
    {
        int32_t perm_dims[] = {1, 0};
        LogicalId p_node = g.constant({2}, perm_dims, DType::INT32);

        auto project_tensor = [&](LogicalId input_tensor, const std::string &suffix, uint32_t in_d, uint32_t out_d) {
            LogicalId w = weight(w_path, prefix + suffix);
            LogicalId w_t = g.permute(w, p_node);
            w_t = g.contiguous(w_t);
            int32_t s3[] = {1, (int32_t)in_d, (int32_t)out_d};
            return g.dot(input_tensor, g.reshape(w_t, g.constant({3}, s3, DType::INT32)));
        };

        auto project = [&](const std::string &suffix, uint32_t in_d, uint32_t out_d) {
            return project_tensor(x, suffix, in_d, out_d);
        };

        // Load the full fused tensors from the safetensors file
        LogicalId fused_gate_up = weight(w_path, prefix + ".mlp.experts.gate_up_proj");
        LogicalId fused_down = weight(w_path, prefix + ".mlp.experts.down_proj");

        uint32_t expert_inter_dim = cfg.shared_expert_dim;
        {
            auto meta = TensorResolver::get().getMetadata(w_path, prefix + ".mlp.experts.gate_up_proj");
            // gate_up_proj shape: [num_experts, 2 * ffn_dim, hidden_size]
            expert_inter_dim = meta.shape[1] / 2;
        }

        // Project input to router logits and get probabilities
        LogicalId router_logits = project(".mlp.gate.weight", cfg.hidden_size, cfg.n_experts);
        LogicalId router_probs = softmax(router_logits, cfg.n_experts);

        // --- Gating Top-K Selection & Normalization ---
        uint32_t S = seq_len;
        uint32_t E = cfg.n_experts;
        uint32_t K = cfg.n_active_experts; // 8

        int32_t ax2_val = 2;
        LogicalId ax2_node = g.constant({1}, &ax2_val, DType::INT32);
        int32_t k_val = (int32_t)K;
        LogicalId k_node = g.constant({1}, &k_val, DType::INT32);

        // 1. Get Top-K Indices: [1, S, K]
        LogicalId selected_experts = g.argmax(router_probs, ax2_node, k_node);

        // 2. Expand selected_experts to [1, S, K, E]
        int32_t sh4_sel[] = {1, (int32_t)S, (int32_t)K, 1};
        LogicalId sel_reshaped = g.reshape(selected_experts, g.constant({4}, sh4_sel, DType::INT32));
        LogicalId sel_expanded = g.contiguous(g.repeat(sel_reshaped, E, 3));

        // 3. Generate Expert Range: [1, S, K, E]
        int32_t arange_start = 0, arange_stop = (int32_t)E, arange_step = 1;
        LogicalId range_1d =
            g.arange(g.constant({1}, &arange_start, DType::INT32), g.constant({1}, &arange_stop, DType::INT32),
                     g.constant({1}, &arange_step, DType::INT32));
        int32_t sh4_range[] = {1, 1, 1, (int32_t)E};
        LogicalId range_reshaped = g.reshape(range_1d, g.constant({4}, sh4_range, DType::INT32));
        LogicalId range_expanded = g.contiguous(g.repeat(g.repeat(range_reshaped, S, 1), K, 2));

        // 4. Compare elementwise and cast to float
        LogicalId mask_bool = g.eq(sel_expanded, range_expanded);
        LogicalId mask_float = g.cast(mask_bool, DType::FLOAT32); // [1, S, K, E]

        // 5. Reduce sum on K axis to yield the final [1, S, E] mask
        int32_t ax2_4d = 2;
        LogicalId mask_reduced = g.sum(mask_float, g.constant({1}, &ax2_4d, DType::INT32)); // [1, S, 1, E]
        int32_t sh3_final[] = {1, (int32_t)S, (int32_t)E};
        LogicalId router_mask = g.reshape(mask_reduced, g.constant({3}, sh3_final, DType::INT32)); // [1, S, E]

        // 6. Apply mask
        LogicalId gated_probs = g.mul(router_probs, router_mask); // [1, S, E]
        int32_t axis = -1;
        LogicalId row_sum = g.sum(gated_probs, g.constant({1}, &axis, DType::INT32)); // [1, S, 1]
        row_sum = g.contiguous(g.repeat(row_sum, cfg.n_experts, 2));
        LogicalId normalized_probs = g.div(gated_probs, row_sum);

        // --- Step 1: Expand Input X to [E, S, H] ---
        int32_t shape_3d_x[] = {1, (int32_t)seq_len, (int32_t)cfg.hidden_size};
        LogicalId x_reshaped = g.reshape(x, g.constant({3}, shape_3d_x, DType::INT32));

        int32_t rep_e[] = {(int32_t)cfg.n_experts};
        int32_t ax_e[] = {0};
        LogicalId x_expanded =
            g.repeat(x_reshaped, g.constant({1}, rep_e, DType::INT32), g.constant({1}, ax_e, DType::INT32));
        x_expanded = g.contiguous(x_expanded); // [E, S, H]

        // --- Step 2: Batched Gate/Up Projection ---
        // Permute fused_gate_up [E, 2*I, H] to [E, H, 2*I]
        int32_t perm_w_3d[] = {0, 2, 1};
        LogicalId fused_gate_up_t = g.permute(fused_gate_up, g.constant({3}, perm_w_3d, DType::INT32));
        fused_gate_up_t = g.contiguous(fused_gate_up_t); // [E, H, 2*I]

        LogicalId gate_up_proj = g.dot(x_expanded, fused_gate_up_t); // [E, S, 2*I]

        // --- Step 3: Slice and Activate ---
        int32_t steps_3d[] = {1, 1, 1};

        int32_t starts_gate[] = {0, 0, 0};
        int32_t ends_gate[] = {(int32_t)cfg.n_experts, (int32_t)seq_len, (int32_t)expert_inter_dim};
        LogicalId exp_gate = g.slice(gate_up_proj, g.constant({3}, starts_gate, DType::INT32),
                                     g.constant({3}, ends_gate, DType::INT32), g.constant({3}, steps_3d, DType::INT32));
        exp_gate = g.contiguous(exp_gate);

        int32_t starts_up[] = {0, 0, (int32_t)expert_inter_dim};
        int32_t ends_up[] = {(int32_t)cfg.n_experts, (int32_t)seq_len, (int32_t)(expert_inter_dim * 2)};
        LogicalId exp_up = g.slice(gate_up_proj, g.constant({3}, starts_up, DType::INT32),
                                   g.constant({3}, ends_up, DType::INT32), g.constant({3}, steps_3d, DType::INT32));
        exp_up = g.contiguous(exp_up);

        LogicalId exp_gate_silu = silu_atomic(exp_gate, cfg.n_experts, seq_len, expert_inter_dim);
        LogicalId exp_gate_up = g.mul(exp_gate_silu, exp_up); // [E, S, I]

        // --- Step 4: Batched Down Projection ---
        // Permute fused_down [E, H, I] to [E, I, H]
        LogicalId fused_down_t = g.permute(fused_down, g.constant({3}, perm_w_3d, DType::INT32));
        fused_down_t = g.contiguous(fused_down_t); // [E, I, H]

        LogicalId exp_down = g.dot(exp_gate_up, fused_down_t); // [E, S, H]

        // --- Step 5: Probabilistic Weighting and Reduction ---
        // 1. Permute exp_down [E, S, H] -> [S, E, H]
        int32_t perm_esh[] = {1, 0, 2};
        LogicalId exp_down_perm = g.permute(exp_down, g.constant({3}, perm_esh, DType::INT32));
        exp_down_perm = g.contiguous(exp_down_perm); // [S, E, H]

        // 2. Permute normalized_probs [1, S, E] -> [S, E, 1]
        int32_t perm_1se[] = {1, 2, 0};
        LogicalId normalized_probs_perm = g.permute(normalized_probs, g.constant({3}, perm_1se, DType::INT32));
        normalized_probs_perm = g.contiguous(normalized_probs_perm); // [S, E, 1]

        // 3. Repeat normalized_probs_perm [S, E, 1] -> [S, E, H]
        int32_t rep_h[] = {(int32_t)cfg.hidden_size};
        int32_t ax_h[] = {2};
        LogicalId normalized_probs_exp =
            g.repeat(normalized_probs_perm, g.constant({1}, rep_h, DType::INT32), g.constant({1}, ax_h, DType::INT32));
        normalized_probs_exp = g.contiguous(normalized_probs_exp); // [S, E, H]

        // 4. Multiply element-wise
        LogicalId weighted_outputs = g.mul(exp_down_perm, normalized_probs_exp); // [S, E, H]

        // 5. Sum along the expert dimension (axis 1)
        int32_t sum_ax[] = {1};
        LogicalId routed_out_sum = g.sum(weighted_outputs, g.constant({1}, sum_ax, DType::INT32)); // [S, 1, H]

        // 6. Reshape [S, 1, H] -> [1, S, H]
        int32_t final_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.hidden_size};
        LogicalId routed_out = g.reshape(routed_out_sum, g.constant({3}, final_shape, DType::INT32));

        // Shared Expert Component (remains unchanged)
        LogicalId shared_gate = project(".mlp.shared_expert.gate_proj.weight", cfg.hidden_size, cfg.shared_expert_dim);
        LogicalId shared_up = project(".mlp.shared_expert.up_proj.weight", cfg.hidden_size, cfg.shared_expert_dim);
        LogicalId shared_gate_silu = silu_atomic(shared_gate, cfg.shared_expert_dim);
        LogicalId shared_gate_up = g.mul(shared_gate_silu, shared_up);

        LogicalId w_shared_down = weight(w_path, prefix + ".mlp.shared_expert.down_proj.weight");
        LogicalId w_shared_down_t = g.permute(w_shared_down, p_node);
        w_shared_down_t = g.contiguous(w_shared_down_t);
        int32_t s3_shared[] = {1, (int32_t)cfg.shared_expert_dim, (int32_t)cfg.hidden_size};
        LogicalId shared_out =
            g.dot(shared_gate_up, g.reshape(w_shared_down_t, g.constant({3}, s3_shared, DType::INT32)));

        LogicalId shared_expert_gate = project(".mlp.shared_expert_gate.weight", cfg.hidden_size, 1);

        // Sigmoid mapping for shared_expert_gate
        float neg_one_val = -1.0f;
        LogicalId neg_one = expand_scalar_to_3d(g.constant({1}, &neg_one_val, DType::FLOAT32), 1, seq_len, 1);
        LogicalId neg_seg = g.mul(shared_expert_gate, neg_one);
        float e_val = 2.718281828459045f;
        LogicalId e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), 1, seq_len, 1);
        LogicalId exp_neg_seg = g.pow(e_node, neg_seg);
        LogicalId one_node = expand_scalar_to_3d(one_fp32, 1, seq_len, 1);
        LogicalId den = g.add(one_node, exp_neg_seg);
        LogicalId seg_sigmoid = g.div(one_node, den);

        LogicalId seg_expanded = g.repeat(seg_sigmoid, cfg.hidden_size, 2);
        LogicalId shared_out_gated = g.mul(shared_out, seg_expanded);

        return g.add(routed_out, shared_out_gated);
    }

    LogicalId build_graph(LogicalId input_ids_id)
    {
        // #L1275
        LogicalId w_emb = weight(w_path, "model.language_model.embed_tokens.weight"); // [vocab_size,
                                                                                      // hidden_size]
        LogicalId x = g.gather(w_emb, input_ids_id);                                  // [B, seq_len, hidden_size]

        // #L1283 position_ids [B, seq_len]
        // #L1284 position_ids [1, 1, seq_len] -> [4, B, seq_len]

        // #L1289 text_position_ids [1, B, seq_len]
        // #L1290 position_ids [3, B, seq_len]

        LogicalId mask_id = compute_causal_mask(); // #L1294

        auto rope = compute_rope(); // #L1304
        LogicalId rope_cos = std::get<0>(rope);
        LogicalId rope_sin = std::get<1>(rope);

        for (uint32_t i = 0; i < cfg.num_hidden_layers; ++i)
        {
            std::string prefix = "model.language_model.layers." + std::to_string(i);
            LogicalId residual = x; // [B, seq_len, hidden_size]
            LogicalId w_ln1 = weight(w_path, prefix + ".input_layernorm.weight");
            x = rms_norm_atomic(x, w_ln1, 1, cfg.hidden_size); // #L862

            // 3:1 Hybrid Attention Stack Routing
            // 3 Linear Attention Layers followed by 1 Full Attention Layer
            if ((i + 1) % 4 != 0)
            {
                // Linear Attention (Gated DeltaNet token mixer)
                x = linear_attention_atomic(x, prefix);
            }
            else
            {
                // Full Attention (Gated Attention token mixer)
                auto qkvg = gated_attention_qkv_atomic(x, prefix, rope_cos, rope_sin);
                x = gated_attention_output_atomic(qkvg, prefix, mask_id);
            }

            x = g.add(residual, x);
            residual = x;
            LogicalId w_ln2 = weight(w_path, prefix + ".post_attention_layernorm.weight");
            x = rms_norm_atomic(x, w_ln2, 1, cfg.hidden_size);

            // Sparse Mixture of Experts (MoE) Base FeedForward
            x = mlp_moe_atomic(x, prefix);

            x = g.add(residual, x);
        }

        LogicalId w_final_ln = weight(w_path, "model.language_model.norm.weight");
        x = rms_norm_atomic(x, w_final_ln, 1, cfg.hidden_size);

        LogicalId w_lm = weight(w_path, "lm_head.weight");

        int32_t perm_dims[] = {1, 0};
        LogicalId dims_node = g.constant({2}, perm_dims, DType::INT32);
        LogicalId w_lm_t = g.permute(w_lm, dims_node);
        w_lm_t = g.contiguous(w_lm_t);
        int32_t s3[] = {1, (int32_t)cfg.hidden_size, (int32_t)cfg.vocab_size};
        LogicalId w_lm_3d = g.reshape(w_lm_t, g.constant({3}, s3, DType::INT32));

        return g.dot(x, w_lm_3d);
    }
};