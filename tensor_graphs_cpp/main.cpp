#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <tuple>
#if defined(_WIN32)
#include <float.h>
#endif

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/session.hpp"
#include "core/kernels.hpp"

#include "generated/kernels_all.gen.hpp"
#include "generated/build_context.gen.hpp"

struct ModelConfig
{
    uint32_t vocab_size = 262144;
    uint32_t n_layers = 18;
    uint32_t emb_dim = 640;
    uint32_t hidden_dim = 2048;
    uint32_t n_heads = 4;
    uint32_t head_dim = 256;
    uint32_t n_kv_groups = 1;
    uint32_t query_pre_attn_scalar = 256;
};

class Gemma3Model
{
private:
    ModelConfig cfg;
    Graph &g;
    MemoryManager &mem;
    const std::string w_path;
    float eps;
    uint32_t seq_len;

    uint32_t one_fp32;
    uint32_t eps_fp32;
    uint32_t half_fp32;

public:
    Gemma3Model(ModelConfig config, uint32_t sequence_length, Graph &graph, MemoryManager &memory, const std::string &weight_path)
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

    uint32_t gelu_atomic(uint32_t x_id, uint32_t last_dim)
    {
        float c1_val = 0.044715f;
        uint32_t c1_node = expand_scalar_to_3d(g.constant({1}, &c1_val, DType::FLOAT32), 1, seq_len, last_dim);

        float c2_val = 0.79788456f;
        uint32_t c2_node = expand_scalar_to_3d(g.constant({1}, &c2_val, DType::FLOAT32), 1, seq_len, last_dim);

        uint32_t x_sq = g.mul(x_id, x_id);
        uint32_t x_cube = g.mul(x_sq, x_id);

        uint32_t term1 = g.mul(x_cube, c1_node);
        uint32_t term2 = g.add(x_id, term1);
        uint32_t term3 = g.mul(term2, c2_node);

        uint32_t tanh_result = tanh_atomic(term3, last_dim);

        uint32_t one_node = expand_scalar_to_3d(one_fp32, 1, seq_len, last_dim);
        uint32_t term4 = g.add(one_node, tanh_result);

        uint32_t half_node = expand_scalar_to_3d(half_fp32, 1, seq_len, last_dim);
        uint32_t term5 = g.mul(x_id, half_node);

        return g.mul(term5, term4);
    }

    uint32_t tanh_atomic(uint32_t x_id, uint32_t last_dim)
    {
        float neg_two_val = -2.0f;
        uint32_t neg_two = expand_scalar_to_3d(g.constant({1}, &neg_two_val, DType::FLOAT32), 1, seq_len, last_dim);

        float two_val = 2.0f;
        uint32_t two = expand_scalar_to_3d(g.constant({1}, &two_val, DType::FLOAT32), 1, seq_len, last_dim);

        float e_val = 2.718281828459045f;
        uint32_t e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), 1, seq_len, last_dim);

        uint32_t one_node = expand_scalar_to_3d(one_fp32, 1, seq_len, last_dim);

        uint32_t neg_2x = g.mul(x_id, neg_two);
        uint32_t exp_neg_2x = g.pow(e_node, neg_2x);

        uint32_t den = g.add(one_node, exp_neg_2x);
        uint32_t quotient = g.div(two, den);

        uint32_t neg_one = g.neg(one_node);
        return g.add(quotient, neg_one);
    }

    // Generate RoPE parameters natively via graph
    std::tuple<uint32_t, uint32_t> compute_rope()
    {
        int32_t start_val = 0, stop_val = cfg.head_dim, step_val = 2;
        uint32_t start = g.constant({1}, &start_val, DType::INT32);
        uint32_t stop = g.constant({1}, &stop_val, DType::INT32);
        uint32_t step = g.constant({1}, &step_val, DType::INT32);

        uint32_t indices_int = g.arange(start, stop, step);
        uint32_t indices = g.cast(indices_int, DType::FLOAT32);

        float h_dim_val = (float)cfg.head_dim;
        uint32_t h_dim_fp = g.constant({1}, &h_dim_val, DType::FLOAT32);

        int32_t shape_1d[] = {(int32_t)(cfg.head_dim / 2)};
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

        int32_t freq_row_shape[] = {1, (int32_t)cfg.head_dim / 2};
        uint32_t freq_row = g.reshape(inv_freq, g.constant({2}, freq_row_shape, DType::INT32));

        uint32_t pos_col_expanded = repeat_3d_axis(pos_col, cfg.head_dim / 2, 1);
        uint32_t freq_row_expanded = repeat_3d_axis(freq_row, seq_len, 0);
        uint32_t angles_half = g.mul(pos_col_expanded, freq_row_expanded);

        int32_t axis_val = 1;
        uint32_t axis_node = g.constant({1}, &axis_val, DType::INT32);
        uint32_t angles = g.concat({angles_half, angles_half}, axis_node); // [seq_len, head_dim]

        int32_t final_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.head_dim};
        uint32_t final_shape_node = g.constant({3}, final_shape, DType::INT32);

        uint32_t cos_out = g.reshape(g.cos(angles), final_shape_node);
        uint32_t sin_out = g.reshape(g.sin(angles), final_shape_node);

        return {cos_out, sin_out};
    }

    uint32_t apply_rope(uint32_t x_id, uint32_t cos_id, uint32_t sin_id, uint32_t n_groups)
    {
        int32_t starts1[] = {0, 0, 0};
        int32_t ends1[] = {(int32_t)n_groups, (int32_t)seq_len, (int32_t)cfg.head_dim / 2};
        int32_t steps1[] = {1, 1, 1};

        uint32_t x1 = g.slice(x_id,
                              g.constant({3}, starts1, DType::INT32),
                              g.constant({3}, ends1, DType::INT32),
                              g.constant({3}, steps1, DType::INT32));

        int32_t starts2[] = {0, 0, (int32_t)cfg.head_dim / 2};
        int32_t ends2[] = {(int32_t)n_groups, (int32_t)seq_len, (int32_t)cfg.head_dim};

        uint32_t x2 = g.slice(x_id,
                              g.constant({3}, starts2, DType::INT32),
                              g.constant({3}, ends2, DType::INT32),
                              g.constant({3}, steps1, DType::INT32));

        uint32_t neg_x2 = g.neg(x2);

        int32_t axis = 2; // concat along head_dim
        uint32_t rotated = g.concat({neg_x2, x1}, g.constant({1}, &axis, DType::INT32));

        // Ensure cos and sin match x_id explicitly
        uint32_t cos_expanded = repeat_3d_axis(cos_id, n_groups, 0);
        uint32_t sin_expanded = repeat_3d_axis(sin_id, n_groups, 0);

        uint32_t term1 = g.mul(x_id, cos_expanded);
        uint32_t term2 = g.mul(rotated, sin_expanded);

        return g.add(term1, term2);
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

        // Exact shape matching prior to multiplication
        int32_t neg_inf_shape[] = {1, 1};
        uint32_t neg_inf_reshaped = g.reshape(neg_inf_node, g.constant({2}, neg_inf_shape, DType::INT32));
        uint32_t neg_inf_expanded = repeat_3d_axis(neg_inf_reshaped, seq_len, 0);
        neg_inf_expanded = repeat_3d_axis(neg_inf_expanded, seq_len, 1);

        uint32_t scaled_mask = g.mul(triu_mask, neg_inf_expanded);

        int32_t final_shape[] = {1, (int32_t)seq_len, (int32_t)seq_len};
        return g.reshape(scaled_mask, g.constant({3}, final_shape, DType::INT32));
    }

    std::tuple<uint32_t, uint32_t, uint32_t> attention_qkv_atomic(uint32_t x, const std::string &prefix, uint32_t rope_cos, uint32_t rope_sin)
    {
        int32_t perm_dims[] = {1, 0};
        uint32_t dims_node = g.constant({2}, perm_dims, DType::INT32);

        auto project = [&](const std::string &suffix, uint32_t in_d, uint32_t out_d)
        {
            uint32_t w = weight(w_path, prefix + suffix);
            uint32_t w_t = g.permute(w, dims_node);
            int32_t s3[] = {1, (int32_t)in_d, (int32_t)out_d};
            return g.dot(x, g.reshape(w_t, g.constant({3}, s3, DType::INT32)));
        };

        uint32_t q = project(".self_attn.q_proj.weight", cfg.emb_dim, cfg.n_heads * cfg.head_dim);
        uint32_t k = project(".self_attn.k_proj.weight", cfg.emb_dim, cfg.n_kv_groups * cfg.head_dim);
        uint32_t v = project(".self_attn.v_proj.weight", cfg.emb_dim, cfg.n_kv_groups * cfg.head_dim);

        int32_t perm4[] = {0, 2, 1, 3};
        uint32_t perm4_node = g.constant({4}, perm4, DType::INT32);

        // Q reshaping -> [n_heads, seq_len, head_dim]
        int32_t q_shape4[] = {1, (int32_t)seq_len, (int32_t)cfg.n_heads, (int32_t)cfg.head_dim};
        uint32_t q_4d = g.reshape(q, g.constant({4}, q_shape4, DType::INT32));
        uint32_t q_perm = g.permute(q_4d, perm4_node);
        int32_t shape3_q[] = {(int32_t)cfg.n_heads, (int32_t)seq_len, (int32_t)cfg.head_dim};
        q = g.reshape(q_perm, g.constant({3}, shape3_q, DType::INT32));

        // K reshaping -> [n_kv_groups, seq_len, head_dim]
        int32_t k_shape4[] = {1, (int32_t)seq_len, (int32_t)cfg.n_kv_groups, (int32_t)cfg.head_dim};
        uint32_t k_4d = g.reshape(k, g.constant({4}, k_shape4, DType::INT32));
        uint32_t k_perm = g.permute(k_4d, perm4_node);
        int32_t shape3_k[] = {(int32_t)cfg.n_kv_groups, (int32_t)seq_len, (int32_t)cfg.head_dim};
        k = g.reshape(k_perm, g.constant({3}, shape3_k, DType::INT32));

        // V reshaping -> [n_kv_groups, seq_len, head_dim]
        uint32_t v_4d = g.reshape(v, g.constant({4}, k_shape4, DType::INT32));
        uint32_t v_perm = g.permute(v_4d, perm4_node);
        v = g.reshape(v_perm, g.constant({3}, shape3_k, DType::INT32));

        // RMSNorm pre-attention
        uint32_t q_norm_w = weight(w_path, prefix + ".self_attn.q_norm.weight");
        q = rms_norm_gemma_atomic(q, q_norm_w, cfg.n_heads, cfg.head_dim);

        uint32_t k_norm_w = weight(w_path, prefix + ".self_attn.k_norm.weight");
        k = rms_norm_gemma_atomic(k, k_norm_w, cfg.n_kv_groups, cfg.head_dim);

        // Apply dynamic RoPE
        q = apply_rope(q, rope_cos, rope_sin, cfg.n_heads);
        k = apply_rope(k, rope_cos, rope_sin, cfg.n_kv_groups);

        if (cfg.n_heads != cfg.n_kv_groups)
        {
            uint32_t repeats = cfg.n_heads / cfg.n_kv_groups;
            int32_t rep[] = {(int32_t)repeats};
            uint32_t rep_node = g.constant({1}, rep, DType::INT32);
            int32_t ax[] = {0};
            uint32_t ax_node = g.constant({1}, ax, DType::INT32);
            k = g.repeat(k, rep_node, ax_node);
            v = g.repeat(v, rep_node, ax_node);
        }

        return std::make_tuple(q, k, v);
    }

    uint32_t attention_output_atomic(std::tuple<uint32_t, uint32_t, uint32_t> qkv, const std::string &prefix, uint32_t mask_id)
    {
        uint32_t q = std::get<0>(qkv);
        uint32_t k = std::get<1>(qkv);
        uint32_t v = std::get<2>(qkv);

        float scale_val = 1.0f / std::sqrt((float)cfg.query_pre_attn_scalar);
        uint32_t scale_node = expand_scalar_to_3d(g.constant({1}, &scale_val, DType::FLOAT32), cfg.n_heads, seq_len, cfg.head_dim);
        uint32_t scaled_q = g.mul(q, scale_node);

        int32_t perm_k[] = {0, 2, 1};
        uint32_t k_t = g.permute(k, g.constant({3}, perm_k, DType::INT32));

        uint32_t scores = g.dot(scaled_q, k_t); // [n_heads, seq_len, seq_len]

        uint32_t mask_expanded = repeat_3d_axis(mask_id, cfg.n_heads, 0); // [n_heads, seq_len, seq_len]
        scores = g.add(scores, mask_expanded);

        // Safe Softmax
        int32_t axis_val = -1;
        uint32_t max_scores = g.max(scores, g.constant({1}, &axis_val, DType::INT32));
        max_scores = repeat_3d_axis(max_scores, seq_len, 2);
        uint32_t shifted_scores = g.add(scores, g.neg(max_scores));

        float e_val = 2.718281828459045f;
        uint32_t e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), cfg.n_heads, seq_len, seq_len);
        uint32_t exp_scores = g.pow(e_node, shifted_scores);

        uint32_t sum_exp = g.sum(exp_scores, g.constant({1}, &axis_val, DType::INT32));
        sum_exp = repeat_3d_axis(sum_exp, seq_len, 2);

        uint32_t probs = g.div(exp_scores, sum_exp);

        uint32_t context = g.dot(probs, v); // [n_heads, seq_len, head_dim]

        int32_t ctx_shape4[] = {1, (int32_t)cfg.n_heads, (int32_t)seq_len, (int32_t)cfg.head_dim};
        uint32_t ctx_4d = g.reshape(context, g.constant({4}, ctx_shape4, DType::INT32));

        int32_t perm_ctx[] = {0, 2, 1, 3};
        uint32_t ctx_perm = g.permute(ctx_4d, g.constant({4}, perm_ctx, DType::INT32));

        int32_t ctx_shape3[] = {1, (int32_t)seq_len, (int32_t)(cfg.n_heads * cfg.head_dim)};
        uint32_t ctx_flat = g.reshape(ctx_perm, g.constant({3}, ctx_shape3, DType::INT32));

        uint32_t w_o = weight(w_path, prefix + ".self_attn.o_proj.weight");
        int32_t perm_dims[] = {1, 0};
        uint32_t w_o_t = g.permute(w_o, g.constant({2}, perm_dims, DType::INT32));

        int32_t s3[] = {1, (int32_t)(cfg.n_heads * cfg.head_dim), (int32_t)cfg.emb_dim};
        uint32_t w_o_3d = g.reshape(w_o_t, g.constant({3}, s3, DType::INT32));

        return g.dot(ctx_flat, w_o_3d);
    }

    uint32_t mlp_atomic(uint32_t x, const std::string &prefix)
    {
        int32_t perm_dims[] = {1, 0};
        uint32_t p_node = g.constant({2}, perm_dims, DType::INT32);

        auto project = [&](const std::string &suffix, uint32_t in_d, uint32_t out_d)
        {
            uint32_t w = weight(w_path, prefix + suffix);
            uint32_t w_t = g.permute(w, p_node);
            int32_t s3[] = {1, (int32_t)in_d, (int32_t)out_d};
            return g.dot(x, g.reshape(w_t, g.constant({3}, s3, DType::INT32)));
        };

        uint32_t gate = project(".mlp.gate_proj.weight", cfg.emb_dim, cfg.hidden_dim);
        gate = gelu_atomic(gate, cfg.hidden_dim);

        uint32_t up = project(".mlp.up_proj.weight", cfg.emb_dim, cfg.hidden_dim);
        uint32_t gate_up = g.mul(gate, up);

        uint32_t w_down = weight(w_path, prefix + ".mlp.down_proj.weight");
        uint32_t w_down_t = g.permute(w_down, p_node);
        int32_t s3[] = {1, (int32_t)cfg.hidden_dim, (int32_t)cfg.emb_dim};

        return g.dot(gate_up, g.reshape(w_down_t, g.constant({3}, s3, DType::INT32)));
    }

    uint32_t build_graph(uint32_t input_ids_id)
    {
        uint32_t w_emb = weight(w_path, "model.embed_tokens.weight");
        uint32_t x = g.gather(w_emb, input_ids_id);

        float scale_val = std::sqrt((float)cfg.emb_dim);
        uint32_t scale_node = expand_scalar_to_3d(g.constant({1}, &scale_val, DType::FLOAT32), 1, seq_len, cfg.emb_dim);
        x = g.mul(x, scale_node);

        auto rope = compute_rope();
        uint32_t rope_cos = std::get<0>(rope);
        uint32_t rope_sin = std::get<1>(rope);
        uint32_t mask_id = compute_causal_mask();

        for (uint32_t i = 0; i < cfg.n_layers; ++i)
        {
            std::string prefix = "model.layers." + std::to_string(i);

            uint32_t residual = x;

            uint32_t w_ln1 = weight(w_path, prefix + ".input_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_ln1, 1, cfg.emb_dim);

            auto qkv = attention_qkv_atomic(x, prefix, rope_cos, rope_sin);
            x = attention_output_atomic(qkv, prefix, mask_id);

            uint32_t w_post_attn = weight(w_path, prefix + ".post_attention_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_post_attn, 1, cfg.emb_dim);

            x = g.add(residual, x);

            residual = x;

            uint32_t w_ln2 = weight(w_path, prefix + ".pre_feedforward_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_ln2, 1, cfg.emb_dim);
            x = mlp_atomic(x, prefix);

            uint32_t w_post_ff = weight(w_path, prefix + ".post_feedforward_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_post_ff, 1, cfg.emb_dim);

            x = g.add(residual, x);
        }

        uint32_t w_final_ln = weight(w_path, "model.norm.weight");
        x = rms_norm_gemma_atomic(x, w_final_ln, 1, cfg.emb_dim);

        int32_t perm_dims[] = {1, 0};
        uint32_t dims_node = g.constant({2}, perm_dims, DType::INT32);
        uint32_t w_emb_t = g.permute(w_emb, dims_node);

        int32_t s3[] = {1, (int32_t)cfg.emb_dim, (int32_t)cfg.vocab_size};
        uint32_t w_emb_3d = g.reshape(w_emb_t, g.constant({3}, s3, DType::INT32));

        uint32_t logits = g.dot(x, w_emb_3d);

        return logits;
    }
};

int main()
{
#if defined(_WIN32)
    _controlfp_s(nullptr, 0, 0);
    _controlfp_s(nullptr, _EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW, _MCW_EM);
#endif
    std::vector<uint32_t> tokens = {2, 105, 2364, 107, 155122, 27825, 49087, 531, 496, 236743, 236810, 1051, 2255, 236761, 106, 107, 105, 4368, 107};
    uint32_t maxSeqLen = 128;
    std::string modelPath = "resources/model.safetensors";

    ModelConfig cfg;
#if USE_CUDA
    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 2ULL * 1024 * 1024 * 1024},{Backend::CUDA, 2ULL * 1024 * 1024 * 1024}};
#else
    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 2ULL * 1024 * 1024 * 1024}};
#endif
    MemoryManager mem = MemoryManager({{Backend::CPU, 2ULL * 1024 * 1024 * 1024}});

    Graph g;

    uint32_t inputIdsId = g.allocateId();
    uint64_t sizeBytes = maxSeqLen * getDTypeSize(DType::INT32);
    mem.allocate(Backend::CPU, inputIdsId, sizeBytes, StorageType::PERSISTENT);

    TensorView inputView;
    inputView.shape = {1, maxSeqLen};
    inputView.strides = TensorView::calcContiguousStrides(inputView.shape);
    inputView.dtype = DType::INT32;

    g.inputWithId(inputIdsId, {1, maxSeqLen}, DType::INT32, inputView, StorageType::PERSISTENT);

    std::cout << "Building Graph..." << std::endl;

    Gemma3Model model(cfg, maxSeqLen, g, mem, modelPath);
    uint32_t logits_id = model.build_graph(inputIdsId);

    std::cout << "Initializing Session..." << std::endl;
    Session session(g, mem, logits_id, "dirty_region_caches/gemma-3-270m-cpp.jsonl");

    mem.buffers.at(Backend::CPU).init();

    std::cout << "Running Inference..." << std::endl;

    std::vector<int32_t> input_data(maxSeqLen, 0);
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        input_data[i] = (int32_t)tokens[i];
    }

    std::unordered_map<uint32_t, const void *> inputs;
    inputs[inputIdsId] = input_data.data();

    session.run(inputs);

    const float *output_ptr = static_cast<const float *>(session.getOutput(logits_id));

    if (output_ptr)
    {
        uint32_t next_token_pos = (uint32_t)tokens.size() - 1;
        uint64_t offset = (uint64_t)next_token_pos * cfg.vocab_size;
        const float *next_token_logits = output_ptr + offset;

        float max_val = -FLT_MAX;
        int32_t argmax_idx = -1;

        for (uint32_t i = 0; i < cfg.vocab_size; ++i)
        {
            if (next_token_logits[i] > max_val)
            {
                max_val = next_token_logits[i];
                argmax_idx = i;
            }
        }

        std::cout << "\nInference Results:" << std::endl;
        std::cout << "------------------" << std::endl;
        std::cout << "Next Token Position: " << next_token_pos << std::endl;
        std::cout << "Predicted Token ID (Argmax): " << argmax_idx << std::endl;
        std::cout << "Logit Value: " << max_val << std::endl;

        std::cout << "\nFirst 5 logits for this token: " << std::endl;
        for (int i = 0; i < 5; ++i)
        {
            std::cout << "  [" << i << "]: " << next_token_logits[i] << std::endl;
        }
    }
    else
    {
        std::cerr << "Failed to retrieve output." << std::endl;
    }

    return 0;
}