#pragma once
#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include <string>
#include <tuple>
#include <cmath>

struct Qwen3_6_35B_A3B_Config
{
    // Token Embedding
    uint32_t vocab_size = 248320;
    uint32_t n_layers = 40;
    uint32_t emb_dim = 2048;

    // Gated DeltaNet (Linear Attention)
    uint32_t linear_n_v_heads = 32;
    uint32_t linear_n_qk_heads = 16;
    uint32_t linear_head_dim = 128;
    uint32_t linear_conv_kernel = 4;

    // Gated Attention (Full Attention)
    uint32_t attn_n_q_heads = 16;
    uint32_t attn_n_kv_heads = 2;
    uint32_t attn_head_dim = 256;
    uint32_t rope_dim = 64;

    // Mixture Of Experts (MoE)
    uint32_t n_experts = 256;
    uint32_t n_active_experts = 8;
    uint32_t shared_expert_dim = 512;

    uint32_t query_pre_attn_scalar = 256;
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
    uint32_t one_fp32;
    uint32_t eps_fp32;
    uint32_t half_fp32;

    uint32_t repeat_ax(uint32_t id, uint32_t repeats, uint32_t axis)
    {
        if (repeats <= 1)
            return id;
        int32_t r = repeats, a = axis;
        return g.repeat(id, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }

public:
    Qwen3_6_35B_A3B_Model(Qwen3_6_35B_A3B_Config config, uint32_t sequence_length, Graph &graph, MemoryManager &memory, const std::string &weight_path)
        : cfg(config), g(graph), mem(memory), w_path(weight_path), eps(1e-6f), seq_len(sequence_length)
    {
        float one_val = 1.0f;
        one_fp32 = g.constant({1}, &one_val, DType::FLOAT32);
        eps_fp32 = g.constant({1}, &eps, DType::FLOAT32);
        float half_val = 0.5f;
        half_fp32 = g.constant({1}, &half_val, DType::FLOAT32);
    }

    uint32_t weight(const std::string &path, const std::string &name)
    {
        uint32_t raw_weight = g.weight(path, name);
        return g.cast(raw_weight, DType::FLOAT32);
    }

    uint32_t repeat_3d_axis(uint32_t tensor_id, uint32_t repeats, uint32_t axis)
    {
        if (repeats <= 1)
            return tensor_id;
        int32_t rep[] = {(int32_t)repeats};
        uint32_t rep_node = g.constant({1}, rep, DType::INT32);
        int32_t ax[] = {(int32_t)axis};
        uint32_t ax_node = g.constant({1}, ax, DType::INT32);
        return g.repeat(tensor_id, rep_node, ax_node);
    }

    uint32_t repeat_4d_axis(uint32_t tensor_id, uint32_t repeats, uint32_t axis)
    {
        if (repeats <= 1)
            return tensor_id;
        int32_t rep[] = {(int32_t)repeats};
        uint32_t rep_node = g.constant({1}, rep, DType::INT32);
        int32_t ax[] = {(int32_t)axis};
        uint32_t ax_node = g.constant({1}, ax, DType::INT32);
        return g.repeat(tensor_id, rep_node, ax_node);
    }

    uint32_t expand_scalar_to_3d(uint32_t scalar_id, uint32_t dim0, uint32_t dim1, uint32_t dim2)
    {
        int32_t shape_3d[] = {1, 1, 1};
        uint32_t shape_3d_node = g.constant({3}, shape_3d, DType::INT32);
        uint32_t out = g.reshape(scalar_id, shape_3d_node);
        if (dim0 > 1)
            out = repeat_3d_axis(out, dim0, 0);
        if (dim1 > 1)
            out = repeat_3d_axis(out, dim1, 1);
        if (dim2 > 1)
            out = repeat_3d_axis(out, dim2, 2);
        return out;
    }

    uint32_t expand_scalar_to_3d(float val, uint32_t dim0, uint32_t dim1, uint32_t dim2)
    {
        uint32_t node = g.constant({1}, &val, DType::FLOAT32);
        return expand_scalar_to_3d(node, dim0, dim1, dim2);
    }

    uint32_t expand_scalar_to_4d(float val, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3)
    {
        uint32_t node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh4[] = {1, 1, 1, 1};
        uint32_t out = g.reshape(node, g.constant({4}, sh4, DType::INT32));
        if (d0 > 1)
            out = repeat_4d_axis(out, d0, 0);
        if (d1 > 1)
            out = repeat_4d_axis(out, d1, 1);
        if (d2 > 1)
            out = repeat_4d_axis(out, d2, 2);
        if (d3 > 1)
            out = repeat_4d_axis(out, d3, 3);
        return out;
    }

    uint32_t expand_scalar_to_1d(float val, uint32_t d0)
    {
        uint32_t node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh1[] = {1};
        return repeat_3d_axis(g.reshape(node, g.constant({1}, sh1, DType::INT32)), d0, 0);
    }

    uint32_t expand_1d_to_3d(uint32_t vec_id, uint32_t vec_len, uint32_t dim0, uint32_t dim1)
    {
        int32_t shape_3d[] = {1, 1, (int32_t)vec_len};
        uint32_t shape_3d_node = g.constant({3}, shape_3d, DType::INT32);
        uint32_t out = g.reshape(vec_id, shape_3d_node);
        if (dim0 > 1)
            out = repeat_3d_axis(out, dim0, 0);
        if (dim1 > 1)
            out = repeat_3d_axis(out, dim1, 1);
        return out;
    }

    uint32_t sigmoid(uint32_t x_id, uint32_t last_dim)
    {
        float neg_one_val = -1.0f;
        uint32_t neg_one = expand_scalar_to_3d(g.constant({1}, &neg_one_val, DType::FLOAT32), 1, seq_len, last_dim);
        uint32_t neg_x = g.mul(x_id, neg_one);
        float e_val = 2.718281828459045f;
        uint32_t e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), 1, seq_len, last_dim);
        uint32_t exp_neg_x = g.pow(e_node, neg_x);
        uint32_t one_node = expand_scalar_to_3d(one_fp32, 1, seq_len, last_dim);
        uint32_t den = g.add(one_node, exp_neg_x);
        return g.div(one_node, den);
    }

    uint32_t silu_atomic(uint32_t x_id, uint32_t last_dim)
    {
        return g.mul(x_id, sigmoid(x_id, last_dim));
    }

    uint32_t silu_atomic(uint32_t x_id, uint32_t N, uint32_t L, uint32_t D)
    {
        float neg_one_val = -1.0f;
        uint32_t neg_one = expand_scalar_to_3d(g.constant({1}, &neg_one_val, DType::FLOAT32), N, L, D);
        uint32_t neg_x = g.mul(x_id, neg_one);
        float e_val = 2.718281828459045f;
        uint32_t e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), N, L, D);
        uint32_t exp_neg_x = g.pow(e_node, neg_x);
        uint32_t one_node = expand_scalar_to_3d(one_fp32, N, L, D);
        uint32_t den = g.add(one_node, exp_neg_x);
        uint32_t sigmoid = g.div(one_node, den);
        return g.mul(x_id, sigmoid);
    }

    uint32_t softmax(uint32_t scores, uint32_t dim_size)
    {
        int32_t axis_val = -1;
        uint32_t axis_node = g.constant({1}, &axis_val, DType::INT32);
        uint32_t max_scores = g.max(scores, axis_node);
        uint32_t max_expanded = repeat_3d_axis(max_scores, dim_size, 2);
        uint32_t shifted_scores = g.add(scores, g.neg(max_expanded));

        float e_val = 2.718281828459045f;
        uint32_t e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), 1, seq_len, dim_size);
        uint32_t exp_scores = g.pow(e_node, shifted_scores);

        uint32_t sum_exp = g.sum(exp_scores, axis_node);
        uint32_t sum_exp_expanded = repeat_3d_axis(sum_exp, dim_size, 2);

        return g.div(exp_scores, sum_exp_expanded);
    }

    uint32_t rms_norm_gemma_atomic(uint32_t x_id, uint32_t weight_id, uint32_t dim0, uint32_t dim_size)
    {
        uint32_t x_sq = g.mul(x_id, x_id);
        int32_t axis_val = -1;
        uint32_t axis_node = g.constant({1}, &axis_val, DType::INT32);
        uint32_t sum_sq = g.sum(x_sq, axis_node);
        float n_val = (float)dim_size;
        uint32_t n_node = g.constant({1}, &n_val, DType::FLOAT32);
        n_node = expand_scalar_to_3d(n_node, dim0, seq_len, 1);
        uint32_t mean_sq = g.div(sum_sq, n_node);
        uint32_t eps_expanded = expand_scalar_to_3d(eps_fp32, dim0, seq_len, 1);
        uint32_t mean_sq_plus_eps = g.add(mean_sq, eps_expanded);
        float half_val = 0.5f;
        uint32_t sqrt_node = g.constant({1}, &half_val, DType::FLOAT32);
        sqrt_node = expand_scalar_to_3d(sqrt_node, dim0, seq_len, 1);
        uint32_t std = g.pow(mean_sq_plus_eps, sqrt_node);
        uint32_t one_node = expand_scalar_to_3d(one_fp32, dim0, seq_len, 1);
        uint32_t inv_std = g.div(one_node, std);
        uint32_t inv_std_expanded = repeat_3d_axis(inv_std, dim_size, 2);
        uint32_t x_norm = g.mul(x_id, inv_std_expanded);
        uint32_t weight_expanded = expand_1d_to_3d(weight_id, dim_size, dim0, seq_len);
        uint32_t one_node_full = expand_scalar_to_3d(one_fp32, dim0, seq_len, dim_size);
        uint32_t scale = g.add(weight_expanded, one_node_full);
        return g.mul(x_norm, scale);
    }

    uint32_t gated_rms_norm(uint32_t x, uint32_t z, const std::string &w_name, uint32_t dims, uint32_t cur_seq_len)
    {
        uint32_t sq = g.mul(x, x);
        int32_t ax = -1;
        uint32_t sum_sq = g.sum(sq, g.constant({1}, &ax, DType::INT32));
        uint32_t mean_sq = g.div(sum_sq, expand_scalar_to_3d((float)dims, 1, cur_seq_len, 1));
        uint32_t var = g.add(mean_sq, expand_scalar_to_3d(1e-6f, 1, cur_seq_len, 1));
        uint32_t std = g.pow(var, expand_scalar_to_3d(0.5f, 1, cur_seq_len, 1));
        uint32_t inv_std = repeat_ax(g.div(expand_scalar_to_3d(1.0f, 1, cur_seq_len, 1), std), dims, 2);
        uint32_t x_norm = g.mul(x, inv_std);

        uint32_t w = weight(w_path, w_name);
        int32_t sh3[] = {1, 1, (int32_t)dims};
        uint32_t w_exp = repeat_ax(g.reshape(w, g.constant({3}, sh3, DType::INT32)), cur_seq_len, 1);

        uint32_t x_norm_scaled = g.mul(x_norm, w_exp);

        // Apply SiLU on gate z
        uint32_t z_silu = silu_atomic(z, 1, cur_seq_len, dims);

        return g.mul(x_norm_scaled, z_silu);
    }

    std::tuple<uint32_t, uint32_t> compute_rope()
    {
        int32_t start_val = 0, stop_val = cfg.rope_dim, step_val = 2;
        uint32_t start = g.constant({1}, &start_val, DType::INT32);
        uint32_t stop = g.constant({1}, &stop_val, DType::INT32);
        uint32_t step = g.constant({1}, &step_val, DType::INT32);
        uint32_t indices_int = g.arange(start, stop, step);
        uint32_t indices = g.cast(indices_int, DType::FLOAT32);
        float h_dim_val = (float)cfg.rope_dim;
        uint32_t h_dim_fp = g.constant({1}, &h_dim_val, DType::FLOAT32);
        int32_t shape_1d[] = {(int32_t)(cfg.rope_dim / 2)};
        uint32_t h_dim_fp_1d = g.repeat(h_dim_fp, g.constant({1}, shape_1d, DType::INT32), g.constant({1}, &start_val, DType::INT32));
        uint32_t exponent = g.div(indices, h_dim_fp_1d);
        float theta_val = 10000.0f;
        uint32_t theta = g.constant({1}, &theta_val, DType::FLOAT32);
        uint32_t theta_1d = g.repeat(theta, g.constant({1}, shape_1d, DType::INT32), g.constant({1}, &start_val, DType::INT32));
        uint32_t base_to_exponent = g.pow(theta_1d, exponent);
        uint32_t one_1d = g.repeat(one_fp32, g.constant({1}, shape_1d, DType::INT32), g.constant({1}, &start_val, DType::INT32));
        uint32_t inv_freq = g.div(one_1d, base_to_exponent);

        int32_t pos_stop_val = seq_len;
        int32_t pos_step_val = 1;
        uint32_t pos_stop = g.constant({1}, &pos_stop_val, DType::INT32);
        uint32_t pos_step = g.constant({1}, &pos_step_val, DType::INT32);
        uint32_t pos_int = g.arange(start, pos_stop, pos_step);
        uint32_t pos = g.cast(pos_int, DType::FLOAT32);
        int32_t pos_col_shape[] = {(int32_t)seq_len, 1};
        uint32_t pos_col = g.reshape(pos, g.constant({2}, pos_col_shape, DType::INT32));
        int32_t freq_row_shape[] = {1, (int32_t)cfg.rope_dim / 2};
        uint32_t freq_row = g.reshape(inv_freq, g.constant({2}, freq_row_shape, DType::INT32));
        uint32_t pos_col_expanded = repeat_3d_axis(pos_col, cfg.rope_dim / 2, 1);
        uint32_t freq_row_expanded = repeat_3d_axis(freq_row, seq_len, 0);
        uint32_t angles_half = g.mul(pos_col_expanded, freq_row_expanded);
        int32_t axis_val = 1;
        uint32_t axis_node = g.constant({1}, &axis_val, DType::INT32);
        uint32_t angles = g.concat({angles_half, angles_half}, axis_node);

        int32_t final_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.rope_dim};
        uint32_t final_shape_node = g.constant({3}, final_shape, DType::INT32);
        uint32_t cos_out = g.reshape(g.cos(angles), final_shape_node);
        uint32_t sin_out = g.reshape(g.sin(angles), final_shape_node);
        return {cos_out, sin_out};
    }

    uint32_t apply_rope(uint32_t x_id, uint32_t cos_id, uint32_t sin_id, uint32_t n_groups, uint32_t head_dim)
    {
        uint32_t rope_dim = cfg.rope_dim;

        int32_t starts1[] = {0, 0, 0};
        int32_t ends1[] = {(int32_t)n_groups, (int32_t)seq_len, (int32_t)rope_dim / 2};
        int32_t steps1[] = {1, 1, 1};
        uint32_t x1 = g.slice(x_id, g.constant({3}, starts1, DType::INT32), g.constant({3}, ends1, DType::INT32), g.constant({3}, steps1, DType::INT32));
        x1 = g.contiguous(x1);

        int32_t starts2[] = {0, 0, (int32_t)rope_dim / 2};
        int32_t ends2[] = {(int32_t)n_groups, (int32_t)seq_len, (int32_t)rope_dim};
        uint32_t x2 = g.slice(x_id, g.constant({3}, starts2, DType::INT32), g.constant({3}, ends2, DType::INT32), g.constant({3}, steps1, DType::INT32));
        uint32_t neg_x2 = g.neg(x2);
        int32_t axis = 2;
        uint32_t rotated = g.concat({neg_x2, x1}, g.constant({1}, &axis, DType::INT32));
        uint32_t cos_expanded = repeat_3d_axis(cos_id, n_groups, 0);
        uint32_t sin_expanded = repeat_3d_axis(sin_id, n_groups, 0);

        int32_t starts_rope[] = {0, 0, 0};
        int32_t ends_rope[] = {(int32_t)n_groups, (int32_t)seq_len, (int32_t)rope_dim};
        uint32_t x_rope = g.slice(x_id, g.constant({3}, starts_rope, DType::INT32), g.constant({3}, ends_rope, DType::INT32), g.constant({3}, steps1, DType::INT32));

        uint32_t term1 = g.mul(x_rope, cos_expanded);
        uint32_t term2 = g.mul(rotated, sin_expanded);
        uint32_t x_rope_applied = g.add(term1, term2);

        if (rope_dim < head_dim)
        {
            int32_t starts_pass[] = {0, 0, (int32_t)rope_dim};
            int32_t ends_pass[] = {(int32_t)n_groups, (int32_t)seq_len, (int32_t)head_dim};
            uint32_t x_pass = g.slice(x_id, g.constant({3}, starts_pass, DType::INT32), g.constant({3}, ends_pass, DType::INT32), g.constant({3}, steps1, DType::INT32));
            return g.concat({x_rope_applied, x_pass}, g.constant({1}, &axis, DType::INT32));
        }
        return x_rope_applied;
    }

    uint32_t compute_causal_mask()
    {
        int32_t mask_shape[] = {(int32_t)seq_len, (int32_t)seq_len};
        uint32_t mask_shape_node = g.constant({2}, mask_shape, DType::INT32);
        float one_val = 1.0f;
        uint32_t ones_matrix = g.fill(g.constant({1}, &one_val, DType::FLOAT32), mask_shape_node);
        int32_t k_val = 1;
        uint32_t triu_mask = g.triu(ones_matrix, g.constant({1}, &k_val, DType::INT32));
        float neg_inf_val = -1e9f;
        uint32_t neg_inf_node = g.constant({1}, &neg_inf_val, DType::FLOAT32);
        int32_t neg_inf_shape[] = {1, 1};
        uint32_t neg_inf_reshaped = g.reshape(neg_inf_node, g.constant({2}, neg_inf_shape, DType::INT32));
        uint32_t neg_inf_expanded = repeat_3d_axis(neg_inf_reshaped, seq_len, 0);
        neg_inf_expanded = repeat_3d_axis(neg_inf_expanded, seq_len, 1);
        uint32_t scaled_mask = g.mul(triu_mask, neg_inf_expanded);
        int32_t final_shape[] = {1, (int32_t)seq_len, (int32_t)seq_len};
        return g.reshape(scaled_mask, g.constant({3}, final_shape, DType::INT32));
    }

    // --- GATED ATTENTION (FULL ATTENTION LAYER) ---
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> gated_attention_qkv_atomic(uint32_t x, const std::string &prefix, uint32_t rope_cos, uint32_t rope_sin)
    {
        int32_t perm_dims[] = {1, 0};
        uint32_t dims_node = g.constant({2}, perm_dims, DType::INT32);

        auto project = [&](const std::string &suffix, uint32_t in_d, uint32_t out_d)
        {
            uint32_t w = weight(w_path, prefix + suffix);
            uint32_t w_t = g.permute(w, dims_node);
            w_t = g.contiguous(w_t);
            int32_t s3[] = {1, (int32_t)in_d, (int32_t)out_d};
            return g.dot(x, g.reshape(w_t, g.constant({3}, s3, DType::INT32)));
        };

        // q_proj provides Q and Gate
        uint32_t q_and_gate = project(".self_attn.q_proj.weight", cfg.emb_dim, cfg.attn_n_q_heads * cfg.attn_head_dim * 2);

        int32_t s_q[] = {0, 0, 0};
        int32_t e_q[] = {1, (int32_t)seq_len, (int32_t)(cfg.attn_n_q_heads * cfg.attn_head_dim)};
        int32_t steps[] = {1, 1, 1};
        uint32_t q = g.contiguous(g.slice(q_and_gate, g.constant({3}, s_q, DType::INT32), g.constant({3}, e_q, DType::INT32), g.constant({3}, steps, DType::INT32)));

        int32_t s_g[] = {0, 0, (int32_t)(cfg.attn_n_q_heads * cfg.attn_head_dim)};
        int32_t e_g[] = {1, (int32_t)seq_len, (int32_t)(cfg.attn_n_q_heads * cfg.attn_head_dim * 2)};
        uint32_t gate = g.slice(q_and_gate, g.constant({3}, s_g, DType::INT32), g.constant({3}, e_g, DType::INT32), g.constant({3}, steps, DType::INT32));

        uint32_t k = project(".self_attn.k_proj.weight", cfg.emb_dim, cfg.attn_n_kv_heads * cfg.attn_head_dim);
        uint32_t v = project(".self_attn.v_proj.weight", cfg.emb_dim, cfg.attn_n_kv_heads * cfg.attn_head_dim);

        int32_t perm4[] = {0, 2, 1, 3};
        uint32_t perm4_node = g.constant({4}, perm4, DType::INT32);

        int32_t q_shape4[] = {1, (int32_t)seq_len, (int32_t)cfg.attn_n_q_heads, (int32_t)cfg.attn_head_dim};
        uint32_t q_4d = g.reshape(q, g.constant({4}, q_shape4, DType::INT32));
        uint32_t q_perm = g.permute(q_4d, perm4_node);
        q_perm = g.contiguous(q_perm);
        int32_t shape3_q[] = {(int32_t)cfg.attn_n_q_heads, (int32_t)seq_len, (int32_t)cfg.attn_head_dim};
        q = g.reshape(q_perm, g.constant({3}, shape3_q, DType::INT32));

        int32_t k_shape4[] = {1, (int32_t)seq_len, (int32_t)cfg.attn_n_kv_heads, (int32_t)cfg.attn_head_dim};
        uint32_t k_4d = g.reshape(k, g.constant({4}, k_shape4, DType::INT32));
        uint32_t k_perm = g.permute(k_4d, perm4_node);
        k_perm = g.contiguous(k_perm);
        int32_t shape3_k[] = {(int32_t)cfg.attn_n_kv_heads, (int32_t)seq_len, (int32_t)cfg.attn_head_dim};
        k = g.reshape(k_perm, g.constant({3}, shape3_k, DType::INT32));

        uint32_t v_4d = g.reshape(v, g.constant({4}, k_shape4, DType::INT32));
        uint32_t v_perm = g.permute(v_4d, perm4_node);
        v_perm = g.contiguous(v_perm);
        v = g.reshape(v_perm, g.constant({3}, shape3_k, DType::INT32));

        uint32_t q_norm_w = weight(w_path, prefix + ".self_attn.q_norm.weight");
        q = rms_norm_gemma_atomic(q, q_norm_w, cfg.attn_n_q_heads, cfg.attn_head_dim);
        uint32_t k_norm_w = weight(w_path, prefix + ".self_attn.k_norm.weight");
        k = rms_norm_gemma_atomic(k, k_norm_w, cfg.attn_n_kv_heads, cfg.attn_head_dim);

        q = apply_rope(q, rope_cos, rope_sin, cfg.attn_n_q_heads, cfg.attn_head_dim);
        k = apply_rope(k, rope_cos, rope_sin, cfg.attn_n_kv_heads, cfg.attn_head_dim);

        if (cfg.attn_n_q_heads != cfg.attn_n_kv_heads)
        {
            uint32_t repeats = cfg.attn_n_q_heads / cfg.attn_n_kv_heads;

            // 1. Reshape [n_kv_heads, seq_len, head_dim] -> [n_kv_heads, 1, seq_len, head_dim]
            int32_t sh4[] = {(int32_t)cfg.attn_n_kv_heads, 1, (int32_t)seq_len, (int32_t)cfg.attn_head_dim};
            uint32_t sh4_node = g.constant({4}, sh4, DType::INT32);
            k = g.reshape(k, sh4_node);
            v = g.reshape(v, sh4_node);

            // 2. Repeat the newly inserted axis (axis 1) of size 1 by 'repeats'
            int32_t rep[] = {(int32_t)repeats};
            uint32_t rep_node = g.constant({1}, rep, DType::INT32);
            int32_t ax[] = {1};
            uint32_t ax_node = g.constant({1}, ax, DType::INT32);
            k = g.repeat(k, rep_node, ax_node);
            v = g.repeat(v, rep_node, ax_node);

            // 3. Materialize the zero-stride view into contiguous memory
            k = g.contiguous(k);
            v = g.contiguous(v);

            // 4. Reshape back to 3D: [n_q_heads, seq_len, head_dim]
            int32_t sh3[] = {(int32_t)cfg.attn_n_q_heads, (int32_t)seq_len, (int32_t)cfg.attn_head_dim};
            uint32_t sh3_node = g.constant({3}, sh3, DType::INT32);
            k = g.reshape(k, sh3_node);
            v = g.reshape(v, sh3_node);
        }
        return std::make_tuple(g.contiguous(q), g.contiguous(k), g.contiguous(v), gate);
    }

    uint32_t gated_attention_output_atomic(std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> qkvg, const std::string &prefix, uint32_t mask_id)
    {
        uint32_t q = std::get<0>(qkvg);
        uint32_t k = std::get<1>(qkvg);
        uint32_t v = std::get<2>(qkvg);
        uint32_t gate = std::get<3>(qkvg);

        float scale_val = 1.0f / std::sqrt((float)cfg.attn_head_dim);
        uint32_t scale_node = expand_scalar_to_3d(g.constant({1}, &scale_val, DType::FLOAT32), cfg.attn_n_q_heads, seq_len, cfg.attn_head_dim);
        uint32_t scaled_q = g.mul(q, scale_node);

        int32_t perm_k[] = {0, 2, 1};
        uint32_t k_t = g.permute(k, g.constant({3}, perm_k, DType::INT32));
        k_t = g.contiguous(k_t);

        uint32_t scores = g.dot(scaled_q, k_t);
        uint32_t mask_expanded = repeat_3d_axis(mask_id, cfg.attn_n_q_heads, 0);
        scores = g.add(scores, mask_expanded);

        int32_t axis_val = -1;
        uint32_t max_scores = g.max(scores, g.constant({1}, &axis_val, DType::INT32));
        max_scores = repeat_3d_axis(max_scores, seq_len, 2);
        uint32_t shifted_scores = g.add(scores, g.neg(max_scores));

        float e_val = 2.718281828459045f;
        uint32_t e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), cfg.attn_n_q_heads, seq_len, seq_len);
        uint32_t exp_scores = g.pow(e_node, shifted_scores);

        uint32_t sum_exp = g.sum(exp_scores, g.constant({1}, &axis_val, DType::INT32));
        sum_exp = repeat_3d_axis(sum_exp, seq_len, 2);

        uint32_t probs = g.div(exp_scores, sum_exp);
        uint32_t context = g.dot(probs, v);

        int32_t ctx_shape4[] = {1, (int32_t)cfg.attn_n_q_heads, (int32_t)seq_len, (int32_t)cfg.attn_head_dim};
        uint32_t ctx_4d = g.reshape(context, g.constant({4}, ctx_shape4, DType::INT32));

        int32_t perm_ctx[] = {0, 2, 1, 3};
        uint32_t ctx_perm = g.permute(ctx_4d, g.constant({4}, perm_ctx, DType::INT32));
        ctx_perm = g.contiguous(ctx_perm);

        int32_t ctx_shape3[] = {1, (int32_t)seq_len, (int32_t)(cfg.attn_n_q_heads * cfg.attn_head_dim)};
        uint32_t ctx_flat = g.reshape(ctx_perm, g.constant({3}, ctx_shape3, DType::INT32));

        // multiply by sigmoid(gate)
        float neg_one_val = -1.0f;
        uint32_t neg_one = expand_scalar_to_3d(g.constant({1}, &neg_one_val, DType::FLOAT32), 1, seq_len, cfg.attn_n_q_heads * cfg.attn_head_dim);
        uint32_t neg_gate = g.mul(gate, neg_one);
        uint32_t e_node_gate = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), 1, seq_len, cfg.attn_n_q_heads * cfg.attn_head_dim);
        uint32_t exp_neg_gate = g.pow(e_node_gate, neg_gate);
        uint32_t one_node = expand_scalar_to_3d(one_fp32, 1, seq_len, cfg.attn_n_q_heads * cfg.attn_head_dim);
        uint32_t den = g.add(one_node, exp_neg_gate);
        uint32_t sigmoid_gate = g.div(one_node, den);

        ctx_flat = g.mul(ctx_flat, sigmoid_gate);

        uint32_t w_o = weight(w_path, prefix + ".self_attn.o_proj.weight");
        int32_t perm_dims[] = {1, 0};
        uint32_t w_o_t = g.permute(w_o, g.constant({2}, perm_dims, DType::INT32));
        w_o_t = g.contiguous(w_o_t);

        int32_t s3[] = {1, (int32_t)(cfg.attn_n_q_heads * cfg.attn_head_dim), (int32_t)cfg.emb_dim};
        uint32_t w_o_3d = g.reshape(w_o_t, g.constant({3}, s3, DType::INT32));
        return g.dot(ctx_flat, w_o_3d);
    }

    // --- GATED DELTANET (LINEAR ATTENTION LAYER) ---
    uint32_t linear_attention_atomic(uint32_t x, const std::string &prefix)
    {
        int32_t perm_dims[] = {1, 0};
        uint32_t dims_node = g.constant({2}, perm_dims, DType::INT32);

        auto project_tensor = [&](uint32_t input_tensor, const std::string &suffix, uint32_t in_d, uint32_t out_d)
        {
            uint32_t w = weight(w_path, prefix + suffix);
            uint32_t w_t = g.permute(w, dims_node);
            w_t = g.contiguous(w_t);
            int32_t s3[] = {1, (int32_t)in_d, (int32_t)out_d};
            return g.dot(input_tensor, g.reshape(w_t, g.constant({3}, s3, DType::INT32)));
        };

        auto project = [&](const std::string &suffix, uint32_t in_d, uint32_t out_d)
        {
            return project_tensor(x, suffix, in_d, out_d);
        };

        // 1. Projections
        uint32_t key_dim = cfg.linear_n_qk_heads * cfg.linear_head_dim;
        uint32_t value_dim = cfg.linear_n_v_heads * cfg.linear_head_dim;
        uint32_t qkv_dim = key_dim * 2 + value_dim;

        uint32_t mixed_qkv = project(".linear_attn.in_proj_qkv.weight", cfg.emb_dim, qkv_dim);
        uint32_t z = project(".linear_attn.in_proj_z.weight", cfg.emb_dim, value_dim);
        uint32_t b = project(".linear_attn.in_proj_b.weight", cfg.emb_dim, cfg.linear_n_v_heads);
        uint32_t a = project(".linear_attn.in_proj_a.weight", cfg.emb_dim, cfg.linear_n_v_heads);

        // 2. Causal 1D Convolution
        int32_t perm_conv[] = {0, 2, 1};
        uint32_t mixed_qkv_tr = g.contiguous(g.permute(mixed_qkv, g.constant({3}, perm_conv, DType::INT32)));

        float zero_val = 0.0f;
        int32_t pad_shape[] = {1, (int32_t)qkv_dim, 3};
        uint32_t pad_zeros = g.fill(g.constant({1}, &zero_val, DType::FLOAT32), g.constant({3}, pad_shape, DType::INT32));

        int32_t ax2 = 2;
        uint32_t padded = g.concat({pad_zeros, mixed_qkv_tr}, g.constant({1}, &ax2, DType::INT32));

        int32_t steps[] = {1, 1, 1};

        int32_t starts0[] = {0, 0, 3};
        int32_t ends0[] = {1, (int32_t)qkv_dim, (int32_t)(seq_len + 3)};
        uint32_t padded_t0 = g.slice(padded, g.constant({3}, starts0, DType::INT32), g.constant({3}, ends0, DType::INT32), g.constant({3}, steps, DType::INT32));

        int32_t starts1[] = {0, 0, 2};
        int32_t ends1[] = {1, (int32_t)qkv_dim, (int32_t)(seq_len + 2)};
        uint32_t padded_t1 = g.slice(padded, g.constant({3}, starts1, DType::INT32), g.constant({3}, ends1, DType::INT32), g.constant({3}, steps, DType::INT32));

        int32_t starts2[] = {0, 0, 1};
        int32_t ends2[] = {1, (int32_t)qkv_dim, (int32_t)(seq_len + 1)};
        uint32_t padded_t2 = g.slice(padded, g.constant({3}, starts2, DType::INT32), g.constant({3}, ends2, DType::INT32), g.constant({3}, steps, DType::INT32));

        int32_t starts3[] = {0, 0, 0};
        int32_t ends3[] = {1, (int32_t)qkv_dim, (int32_t)seq_len};
        uint32_t padded_t3 = g.slice(padded, g.constant({3}, starts3, DType::INT32), g.constant({3}, ends3, DType::INT32), g.constant({3}, steps, DType::INT32));

        uint32_t conv_w = weight(w_path, prefix + ".linear_attn.conv1d.weight");

        int32_t w_reshape_shape[] = {1, (int32_t)qkv_dim, 1};
        uint32_t w_reshape = g.constant({3}, w_reshape_shape, DType::INT32);

        int32_t w_steps[] = {1, 1, 1};

        int32_t ws0[] = {0, 0, 0};
        int32_t we0[] = {(int32_t)qkv_dim, 1, 1};
        uint32_t w0 = g.reshape(g.contiguous(g.slice(conv_w, g.constant({3}, ws0, DType::INT32), g.constant({3}, we0, DType::INT32), g.constant({3}, w_steps, DType::INT32))), w_reshape);

        int32_t ws1[] = {0, 0, 1};
        int32_t we1[] = {(int32_t)qkv_dim, 1, 2};
        uint32_t w1 = g.reshape(g.contiguous(g.slice(conv_w, g.constant({3}, ws1, DType::INT32), g.constant({3}, we1, DType::INT32), g.constant({3}, w_steps, DType::INT32))), w_reshape);

        int32_t ws2[] = {0, 0, 2};
        int32_t we2[] = {(int32_t)qkv_dim, 1, 3};
        uint32_t w2 = g.reshape(g.contiguous(g.slice(conv_w, g.constant({3}, ws2, DType::INT32), g.constant({3}, we2, DType::INT32), g.constant({3}, w_steps, DType::INT32))), w_reshape);

        int32_t ws3[] = {0, 0, 3};
        int32_t we3[] = {(int32_t)qkv_dim, 1, 4};
        uint32_t w3 = g.reshape(g.contiguous(g.slice(conv_w, g.constant({3}, ws3, DType::INT32), g.constant({3}, we3, DType::INT32), g.constant({3}, w_steps, DType::INT32))), w_reshape);

        uint32_t w0_exp = repeat_3d_axis(w0, seq_len, 2);
        uint32_t w1_exp = repeat_3d_axis(w1, seq_len, 2);
        uint32_t w2_exp = repeat_3d_axis(w2, seq_len, 2);
        uint32_t w3_exp = repeat_3d_axis(w3, seq_len, 2);

        uint32_t term0 = g.mul(padded_t3, w0_exp);
        uint32_t term1 = g.mul(padded_t2, w1_exp);
        uint32_t term2 = g.mul(padded_t1, w2_exp);
        uint32_t term3 = g.mul(padded_t0, w3_exp);

        uint32_t conv_combined = g.add(g.add(g.add(term0, term1), term2), term3);

        int32_t perm_back[] = {0, 2, 1};
        uint32_t conv_out_tr = g.contiguous(g.permute(conv_combined, g.constant({3}, perm_back, DType::INT32)));

        int32_t starts_q[] = {0, 0, 0};
        int32_t ends_q[] = {1, (int32_t)seq_len, (int32_t)key_dim};
        uint32_t q = g.contiguous(g.slice(conv_out_tr, g.constant({3}, starts_q, DType::INT32), g.constant({3}, ends_q, DType::INT32), g.constant({3}, steps, DType::INT32)));

        int32_t starts_k[] = {0, 0, (int32_t)key_dim};
        int32_t ends_k[] = {1, (int32_t)seq_len, (int32_t)(key_dim * 2)};
        uint32_t k = g.contiguous(g.slice(conv_out_tr, g.constant({3}, starts_k, DType::INT32), g.constant({3}, ends_k, DType::INT32), g.constant({3}, steps, DType::INT32)));

        int32_t starts_v[] = {0, 0, (int32_t)(key_dim * 2)};
        int32_t ends_v[] = {1, (int32_t)seq_len, (int32_t)qkv_dim};
        uint32_t v = g.contiguous(g.slice(conv_out_tr, g.constant({3}, starts_v, DType::INT32), g.constant({3}, ends_v, DType::INT32), g.constant({3}, steps, DType::INT32)));

        // 3. Compute beta and g (decay alpha)
        uint32_t beta = sigmoid(b, cfg.linear_n_v_heads);

        uint32_t A_log = weight(w_path, prefix + ".linear_attn.A_log");
        uint32_t dt_bias = weight(w_path, prefix + ".linear_attn.dt_bias");

        int32_t ba_reshape_shape[] = {1, 1, (int32_t)cfg.linear_n_v_heads};
        uint32_t dt_bias_3d = repeat_3d_axis(g.reshape(dt_bias, g.constant({3}, ba_reshape_shape, DType::INT32)), seq_len, 1);
        uint32_t a_plus_dt_bias = g.add(a, dt_bias_3d);

        uint32_t exp_x = g.pow(expand_scalar_to_3d(2.7182818f, 1, seq_len, cfg.linear_n_v_heads), a_plus_dt_bias);
        uint32_t one_plus_exp = g.add(expand_scalar_to_3d(1.0f, 1, seq_len, cfg.linear_n_v_heads), exp_x);
        uint32_t softplus_x = g.log(one_plus_exp);

        uint32_t A_log_exp = g.pow(expand_scalar_to_1d(2.7182818f, cfg.linear_n_v_heads), A_log);
        uint32_t A_log_exp_3d = repeat_3d_axis(g.reshape(A_log_exp, g.constant({3}, ba_reshape_shape, DType::INT32)), seq_len, 1);
        uint32_t decay_g = g.mul(g.neg(A_log_exp_3d), softplus_x);
        uint32_t decay_alpha = g.pow(expand_scalar_to_3d(2.7182818f, 1, seq_len, cfg.linear_n_v_heads), decay_g);

        // 4. Reshape Q, K, V to head-based layout and L2 Norm
        int32_t perm_heads[] = {0, 2, 1, 3};
        int32_t q_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.linear_n_qk_heads, (int32_t)cfg.linear_head_dim};
        uint32_t q_heads = g.contiguous(g.permute(g.reshape(q, g.constant({4}, q_shape, DType::INT32)), g.constant({4}, perm_heads, DType::INT32)));

        int32_t k_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.linear_n_qk_heads, (int32_t)cfg.linear_head_dim};
        uint32_t k_heads = g.contiguous(g.permute(g.reshape(k, g.constant({4}, k_shape, DType::INT32)), g.constant({4}, perm_heads, DType::INT32)));

        int32_t v_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.linear_n_v_heads, (int32_t)cfg.linear_head_dim};
        uint32_t v_heads = g.contiguous(g.permute(g.reshape(v, g.constant({4}, v_shape, DType::INT32)), g.constant({4}, perm_heads, DType::INT32)));

        int32_t b_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.linear_n_v_heads, 1};
        uint32_t b_heads = g.contiguous(g.permute(g.reshape(beta, g.constant({4}, b_shape, DType::INT32)), g.constant({4}, perm_heads, DType::INT32)));
        uint32_t a_heads = g.contiguous(g.permute(g.reshape(decay_alpha, g.constant({4}, b_shape, DType::INT32)), g.constant({4}, perm_heads, DType::INT32)));

        // 1. Reshape [1, 16, 8, 128] -> [1, 16, 1, 8, 128] to insert a unit dimension at axis 2
        int32_t sh5[] = {1, (int32_t)cfg.linear_n_qk_heads, 1, (int32_t)seq_len, (int32_t)cfg.linear_head_dim};
        uint32_t sh5_node = g.constant({5}, sh5, DType::INT32);
        uint32_t q_heads_5d = g.reshape(q_heads, sh5_node);
        uint32_t k_heads_5d = g.reshape(k_heads, sh5_node);

        // 2. Repeat along the newly inserted axis (axis 2) of size 1 by r_heads (2)
        int32_t r_heads = 2;
        int32_t rep[] = {(int32_t)r_heads};
        uint32_t rep_node = g.constant({1}, rep, DType::INT32);
        int32_t ax[] = {2};
        uint32_t ax_node = g.constant({1}, ax, DType::INT32);

        uint32_t q_heads_rep = g.repeat(q_heads_5d, rep_node, ax_node);
        uint32_t k_heads_rep = g.repeat(k_heads_5d, rep_node, ax_node);

        // 3. Materialize the zero-stride view into contiguous memory
        uint32_t q_heads_contig = g.contiguous(q_heads_rep);
        uint32_t k_heads_contig = g.contiguous(k_heads_rep);

        // 4. Reshape back to 4D: [1, 32, 8, 128], collapsing the [16, 2] dimensions into 32
        int32_t sh4_target[] = {1, (int32_t)cfg.linear_n_v_heads, (int32_t)seq_len, (int32_t)cfg.linear_head_dim};
        uint32_t sh4_target_node = g.constant({4}, sh4_target, DType::INT32);

        uint32_t q_heads_exp = g.reshape(q_heads_contig, sh4_target_node);
        uint32_t k_heads_exp = g.reshape(k_heads_contig, sh4_target_node);

        int32_t ax_neg1 = -1;
        uint32_t q_sq = g.mul(q_heads_exp, q_heads_exp);
        uint32_t q_sum = g.sum(q_sq, g.constant({1}, &ax_neg1, DType::INT32));
        uint32_t q_std = g.pow(g.add(q_sum, expand_scalar_to_4d(1e-6f, 1, (int32_t)cfg.linear_n_v_heads, seq_len, 1)), expand_scalar_to_4d(0.5f, 1, (int32_t)cfg.linear_n_v_heads, seq_len, 1));
        uint32_t q_norm = g.mul(q_heads_exp, repeat_3d_axis(g.div(expand_scalar_to_4d(1.0f, 1, (int32_t)cfg.linear_n_v_heads, seq_len, 1), q_std), (int32_t)cfg.linear_head_dim, 3));

        float scale_factor = 1.0f / std::sqrt((float)cfg.linear_head_dim);
        q_norm = g.mul(q_norm, expand_scalar_to_4d(scale_factor, 1, (int32_t)cfg.linear_n_v_heads, seq_len, (int32_t)cfg.linear_head_dim));

        uint32_t k_sq = g.mul(k_heads_exp, k_heads_exp);
        uint32_t k_sum = g.sum(k_sq, g.constant({1}, &ax_neg1, DType::INT32));
        uint32_t k_std = g.pow(g.add(k_sum, expand_scalar_to_4d(1e-6f, 1, (int32_t)cfg.linear_n_v_heads, seq_len, 1)), expand_scalar_to_4d(0.5f, 1, (int32_t)cfg.linear_n_v_heads, seq_len, 1));
        uint32_t k_norm = g.mul(k_heads_exp, repeat_3d_axis(g.div(expand_scalar_to_4d(1.0f, 1, (int32_t)cfg.linear_n_v_heads, seq_len, 1), k_std), (int32_t)cfg.linear_head_dim, 3));

        // 5. Gated Delta Rule Recurrence Loop
        int32_t s_shape[] = {(int32_t)cfg.linear_n_v_heads, (int32_t)cfg.linear_head_dim, (int32_t)cfg.linear_head_dim};
        uint32_t S = g.fill(g.constant({1}, &zero_val, DType::FLOAT32), g.constant({3}, s_shape, DType::INT32));

        std::vector<uint32_t> outs;
        for (uint32_t t = 0; t < seq_len; ++t)
        {
            int32_t starts_t_ab[] = {0, 0, (int32_t)t, 0};
            int32_t ends_t_b[] = {1, (int32_t)cfg.linear_n_v_heads, (int32_t)(t + 1), 1};
            int32_t steps_t[] = {1, 1, 1, 1};
            uint32_t a_t = g.contiguous(g.slice(a_heads, g.constant({4}, starts_t_ab, DType::INT32), g.constant({4}, ends_t_b, DType::INT32), g.constant({4}, steps_t, DType::INT32)));

            int32_t flat_scalar_shape[] = {(int32_t)cfg.linear_n_v_heads, 1, 1};
            uint32_t a_t_flat = g.reshape(a_t, g.constant({3}, flat_scalar_shape, DType::INT32));

            uint32_t a_t_exp = repeat_3d_axis(repeat_3d_axis(a_t_flat, (int32_t)cfg.linear_head_dim, 1), (int32_t)cfg.linear_head_dim, 2);
            S = g.mul(S, a_t_exp);

            int32_t starts_t_k[] = {0, 0, (int32_t)t, 0};
            int32_t ends_t_k[] = {1, (int32_t)cfg.linear_n_v_heads, (int32_t)(t + 1), (int32_t)cfg.linear_head_dim};
            uint32_t k_t = g.contiguous(g.slice(k_norm, g.constant({4}, starts_t_k, DType::INT32), g.constant({4}, ends_t_k, DType::INT32), g.constant({4}, steps_t, DType::INT32)));

            int32_t flat_vector_shape[] = {(int32_t)cfg.linear_n_v_heads, 1, (int32_t)cfg.linear_head_dim};
            uint32_t k_t_flat = g.reshape(k_t, g.constant({3}, flat_vector_shape, DType::INT32));

            uint32_t v_t = g.contiguous(g.slice(v_heads, g.constant({4}, starts_t_k, DType::INT32), g.constant({4}, ends_t_k, DType::INT32), g.constant({4}, steps_t, DType::INT32)));
            uint32_t v_t_flat = g.reshape(v_t, g.constant({3}, flat_vector_shape, DType::INT32));

            uint32_t b_t = g.contiguous(g.slice(b_heads, g.constant({4}, starts_t_ab, DType::INT32), g.constant({4}, ends_t_b, DType::INT32), g.constant({4}, steps_t, DType::INT32)));
            uint32_t b_t_flat = g.reshape(b_t, g.constant({3}, flat_scalar_shape, DType::INT32));

            uint32_t kv_mem = g.contiguous(g.dot(k_t_flat, S));

            uint32_t err = g.add(v_t_flat, g.neg(kv_mem));
            uint32_t b_t_exp = repeat_3d_axis(b_t_flat, (int32_t)cfg.linear_head_dim, 2);
            uint32_t delta = g.mul(err, b_t_exp);

            int32_t perm_t[] = {0, 2, 1};
            uint32_t k_t_t = g.contiguous(g.permute(k_t_flat, g.constant({3}, perm_t, DType::INT32)));
            uint32_t outer_prod = g.contiguous(g.dot(k_t_t, delta));
            S = g.add(S, outer_prod);

            uint32_t q_t = g.contiguous(g.slice(q_norm, g.constant({4}, starts_t_k, DType::INT32), g.constant({4}, ends_t_k, DType::INT32), g.constant({4}, steps_t, DType::INT32)));
            uint32_t q_t_flat = g.reshape(q_t, g.constant({3}, flat_vector_shape, DType::INT32));

            uint32_t y_t = g.contiguous(g.dot(q_t_flat, S));

            int32_t head_y_shape[] = {1, (int32_t)cfg.linear_n_v_heads, 1, (int32_t)cfg.linear_head_dim};
            uint32_t y_t_head = g.reshape(y_t, g.constant({4}, head_y_shape, DType::INT32));
            outs.push_back(y_t_head);
        }

        uint32_t context_heads;
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
        uint32_t context_perm = g.contiguous(g.permute(context_heads, g.constant({4}, perm_heads_back, DType::INT32)));

        int32_t final_context_shape[] = {1, (int32_t)seq_len, (int32_t)value_dim};

        // 6. Gated RMSNorm and Final Output Projection
        // Reshape context and z to head-wise format: [1, seq_len * linear_n_v_heads, linear_head_dim]
        int32_t head_format_shape[] = {1, (int32_t)(seq_len * cfg.linear_n_v_heads), (int32_t)cfg.linear_head_dim};
        uint32_t head_format_shape_node = g.constant({3}, head_format_shape, DType::INT32);

        uint32_t context_flat = g.reshape(context_perm, head_format_shape_node);
        uint32_t z_flat = g.reshape(z, head_format_shape_node);

        // Apply Gated RMSNorm on the head-wise format
        uint32_t norm_flat = gated_rms_norm(context_flat, z_flat, prefix + ".linear_attn.norm.weight", cfg.linear_head_dim, seq_len * cfg.linear_n_v_heads);

        // Reshape back to value_dim [1, seq_len, value_dim]
        uint32_t output = g.reshape(norm_flat, g.constant({3}, final_context_shape, DType::INT32));

        return project_tensor(output, ".linear_attn.out_proj.weight", value_dim, cfg.emb_dim);
    }

    // --- MIXTURE OF EXPERTS (MOE) MLP LAYER ---
    uint32_t mlp_moe_atomic(uint32_t x, const std::string &prefix)
    {
        int32_t perm_dims[] = {1, 0};
        uint32_t p_node = g.constant({2}, perm_dims, DType::INT32);

        auto project_tensor = [&](uint32_t input_tensor, const std::string &suffix, uint32_t in_d, uint32_t out_d)
        {
            uint32_t w = weight(w_path, prefix + suffix);
            uint32_t w_t = g.permute(w, p_node);
            w_t = g.contiguous(w_t);
            int32_t s3[] = {1, (int32_t)in_d, (int32_t)out_d};
            return g.dot(input_tensor, g.reshape(w_t, g.constant({3}, s3, DType::INT32)));
        };

        auto project = [&](const std::string &suffix, uint32_t in_d, uint32_t out_d)
        {
            return project_tensor(x, suffix, in_d, out_d);
        };

        // Load the full fused tensors from the safetensors file
        uint32_t fused_gate_up = weight(w_path, prefix + ".mlp.experts.gate_up_proj");
        uint32_t fused_down = weight(w_path, prefix + ".mlp.experts.down_proj");

        uint32_t expert_inter_dim = cfg.shared_expert_dim;
        {
            auto meta = FileRegistry::get().getMetadata(w_path, prefix + ".mlp.experts.gate_up_proj");
            // gate_up_proj shape: [num_experts, 2 * ffn_dim, hidden_size]
            expert_inter_dim = meta.shape[1] / 2;
        }

        // Project input to router logits and get probabilities
        uint32_t router_logits = project(".mlp.gate.weight", cfg.emb_dim, cfg.n_experts);
        uint32_t router_probs = softmax(router_logits, cfg.n_experts);

        // --- Step 1: Expand Input X to [E, S, H] ---
        int32_t shape_3d_x[] = {1, (int32_t)seq_len, (int32_t)cfg.emb_dim};
        uint32_t x_reshaped = g.reshape(x, g.constant({3}, shape_3d_x, DType::INT32));

        int32_t rep_e[] = {(int32_t)cfg.n_experts};
        int32_t ax_e[] = {0};
        uint32_t x_expanded = g.repeat(x_reshaped, g.constant({1}, rep_e, DType::INT32), g.constant({1}, ax_e, DType::INT32));
        x_expanded = g.contiguous(x_expanded); // [E, S, H]

        // --- Step 2: Batched Gate/Up Projection ---
        // Permute fused_gate_up [E, 2*I, H] to [E, H, 2*I]
        int32_t perm_w_3d[] = {0, 2, 1};
        uint32_t fused_gate_up_t = g.permute(fused_gate_up, g.constant({3}, perm_w_3d, DType::INT32));
        fused_gate_up_t = g.contiguous(fused_gate_up_t); // [E, H, 2*I]

        uint32_t gate_up_proj = g.dot(x_expanded, fused_gate_up_t); // [E, S, 2*I]

        // --- Step 3: Slice and Activate ---
        int32_t steps_3d[] = {1, 1, 1};

        int32_t starts_gate[] = {0, 0, 0};
        int32_t ends_gate[] = {(int32_t)cfg.n_experts, (int32_t)seq_len, (int32_t)expert_inter_dim};
        uint32_t exp_gate = g.slice(gate_up_proj, g.constant({3}, starts_gate, DType::INT32), g.constant({3}, ends_gate, DType::INT32), g.constant({3}, steps_3d, DType::INT32));
        exp_gate = g.contiguous(exp_gate);

        int32_t starts_up[] = {0, 0, (int32_t)expert_inter_dim};
        int32_t ends_up[] = {(int32_t)cfg.n_experts, (int32_t)seq_len, (int32_t)(expert_inter_dim * 2)};
        uint32_t exp_up = g.slice(gate_up_proj, g.constant({3}, starts_up, DType::INT32), g.constant({3}, ends_up, DType::INT32), g.constant({3}, steps_3d, DType::INT32));
        exp_up = g.contiguous(exp_up);

        uint32_t exp_gate_silu = silu_atomic(exp_gate, cfg.n_experts, seq_len, expert_inter_dim);
        uint32_t exp_gate_up = g.mul(exp_gate_silu, exp_up); // [E, S, I]

        // --- Step 4: Batched Down Projection ---
        // Permute fused_down [E, H, I] to [E, I, H]
        uint32_t fused_down_t = g.permute(fused_down, g.constant({3}, perm_w_3d, DType::INT32));
        fused_down_t = g.contiguous(fused_down_t); // [E, I, H]

        uint32_t exp_down = g.dot(exp_gate_up, fused_down_t); // [E, S, H]

        // --- Step 5: Probabilistic Weighting and Reduction ---
        // 1. Permute exp_down [E, S, H] -> [S, E, H]
        int32_t perm_esh[] = {1, 0, 2};
        uint32_t exp_down_perm = g.permute(exp_down, g.constant({3}, perm_esh, DType::INT32));
        exp_down_perm = g.contiguous(exp_down_perm); // [S, E, H]

        // 2. Permute router_probs [1, S, E] -> [S, E, 1]
        int32_t perm_1se[] = {1, 2, 0};
        uint32_t router_probs_perm = g.permute(router_probs, g.constant({3}, perm_1se, DType::INT32));
        router_probs_perm = g.contiguous(router_probs_perm); // [S, E, 1]

        // 3. Repeat router_probs_perm [S, E, 1] -> [S, E, H]
        int32_t rep_h[] = {(int32_t)cfg.emb_dim};
        int32_t ax_h[] = {2};
        uint32_t router_probs_exp = g.repeat(router_probs_perm, g.constant({1}, rep_h, DType::INT32), g.constant({1}, ax_h, DType::INT32));
        router_probs_exp = g.contiguous(router_probs_exp); // [S, E, H]

        // 4. Multiply element-wise
        uint32_t weighted_outputs = g.mul(exp_down_perm, router_probs_exp); // [S, E, H]

        // 5. Sum along the expert dimension (axis 1)
        int32_t sum_ax[] = {1};
        uint32_t routed_out_sum = g.sum(weighted_outputs, g.constant({1}, sum_ax, DType::INT32)); // [S, 1, H]

        // 6. Reshape [S, 1, H] -> [1, S, H]
        int32_t final_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.emb_dim};
        uint32_t routed_out = g.reshape(routed_out_sum, g.constant({3}, final_shape, DType::INT32));

        // Shared Expert Component (remains unchanged)
        uint32_t shared_gate = project(".mlp.shared_expert.gate_proj.weight", cfg.emb_dim, cfg.shared_expert_dim);
        uint32_t shared_up = project(".mlp.shared_expert.up_proj.weight", cfg.emb_dim, cfg.shared_expert_dim);
        uint32_t shared_gate_silu = silu_atomic(shared_gate, cfg.shared_expert_dim);
        uint32_t shared_gate_up = g.mul(shared_gate_silu, shared_up);

        uint32_t w_shared_down = weight(w_path, prefix + ".mlp.shared_expert.down_proj.weight");
        uint32_t w_shared_down_t = g.permute(w_shared_down, p_node);
        w_shared_down_t = g.contiguous(w_shared_down_t);
        int32_t s3_shared[] = {1, (int32_t)cfg.shared_expert_dim, (int32_t)cfg.emb_dim};
        uint32_t shared_out = g.dot(shared_gate_up, g.reshape(w_shared_down_t, g.constant({3}, s3_shared, DType::INT32)));

        uint32_t shared_expert_gate = project(".mlp.shared_expert_gate.weight", cfg.emb_dim, 1);

        // Sigmoid mapping for shared_expert_gate
        float neg_one_val = -1.0f;
        uint32_t neg_one = expand_scalar_to_3d(g.constant({1}, &neg_one_val, DType::FLOAT32), 1, seq_len, 1);
        uint32_t neg_seg = g.mul(shared_expert_gate, neg_one);
        float e_val = 2.718281828459045f;
        uint32_t e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), 1, seq_len, 1);
        uint32_t exp_neg_seg = g.pow(e_node, neg_seg);
        uint32_t one_node = expand_scalar_to_3d(one_fp32, 1, seq_len, 1);
        uint32_t den = g.add(one_node, exp_neg_seg);
        uint32_t seg_sigmoid = g.div(one_node, den);

        uint32_t seg_expanded = repeat_3d_axis(seg_sigmoid, cfg.emb_dim, 2);
        uint32_t shared_out_gated = g.mul(shared_out, seg_expanded);

        return g.add(routed_out, shared_out_gated);
    }

    uint32_t build_graph(uint32_t input_ids_id)
    {
        uint32_t w_emb = weight(w_path, "model.language_model.embed_tokens.weight");
        uint32_t x = g.gather(w_emb, input_ids_id);

        auto rope = compute_rope();
        uint32_t rope_cos = std::get<0>(rope);
        uint32_t rope_sin = std::get<1>(rope);
        uint32_t mask_id = compute_causal_mask();

        for (uint32_t i = 0; i < cfg.n_layers; ++i)
        {
            std::string prefix = "model.language_model.layers." + std::to_string(i);
            uint32_t residual = x;
            uint32_t w_ln1 = weight(w_path, prefix + ".input_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_ln1, 1, cfg.emb_dim);

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
            uint32_t w_ln2 = weight(w_path, prefix + ".post_attention_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_ln2, 1, cfg.emb_dim);

            // Sparse Mixture of Experts (MoE) Base FeedForward
            x = mlp_moe_atomic(x, prefix);

            x = g.add(residual, x);
        }

        uint32_t w_final_ln = weight(w_path, "model.language_model.norm.weight");
        x = rms_norm_gemma_atomic(x, w_final_ln, 1, cfg.emb_dim);

        int32_t perm_dims[] = {1, 0};
        uint32_t dims_node = g.constant({2}, perm_dims, DType::INT32);
        uint32_t w_emb_t = g.permute(w_emb, dims_node);
        w_emb_t = g.contiguous(w_emb_t);
        int32_t s3[] = {1, (int32_t)cfg.emb_dim, (int32_t)cfg.vocab_size};
        uint32_t w_emb_3d = g.reshape(w_emb_t, g.constant({3}, s3, DType::INT32));

        return g.dot(x, w_emb_3d);
    }
};