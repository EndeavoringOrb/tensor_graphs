#pragma once
#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include <string>

struct Gemma3ModelConfig
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
    Gemma3ModelConfig cfg;
    Graph &g;
    MemoryManager &mem;
    const std::string w_path;
    float eps;
    uint32_t seq_len;
    LogicalId one_fp32;
    LogicalId eps_fp32;
    LogicalId half_fp32;

public:
    Gemma3Model(Gemma3ModelConfig config, uint32_t sequence_length, Graph &graph, MemoryManager &memory, const std::string &weight_path)
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

    LogicalId repeat_3d_axis(LogicalId tensor_id, uint32_t repeats, uint32_t axis)
    {
        if (repeats <= 1)
            return tensor_id;
        int32_t rep[] = {(int32_t)repeats};
        LogicalId rep_node = g.constant({1}, rep, DType::INT32);
        int32_t ax[] = {(int32_t)axis};
        LogicalId ax_node = g.constant({1}, ax, DType::INT32);
        return g.repeat(tensor_id, rep_node, ax_node);
    }

    LogicalId expand_scalar_to_3d(LogicalId scalar_id, uint32_t dim0, uint32_t dim1, uint32_t dim2)
    {
        int32_t shape_3d[] = {1, 1, 1};
        LogicalId shape_3d_node = g.constant({3}, shape_3d, DType::INT32);
        LogicalId out = g.reshape(scalar_id, shape_3d_node);
        if (dim0 > 1)
            out = repeat_3d_axis(out, dim0, 0);
        if (dim1 > 1)
            out = repeat_3d_axis(out, dim1, 1);
        if (dim2 > 1)
            out = repeat_3d_axis(out, dim2, 2);
        return out;
    }

    LogicalId expand_1d_to_3d(LogicalId vec_id, uint32_t vec_len, uint32_t dim0, uint32_t dim1)
    {
        int32_t shape_3d[] = {1, 1, (int32_t)vec_len};
        LogicalId shape_3d_node = g.constant({3}, shape_3d, DType::INT32);
        LogicalId out = g.reshape(vec_id, shape_3d_node);
        if (dim0 > 1)
            out = repeat_3d_axis(out, dim0, 0);
        if (dim1 > 1)
            out = repeat_3d_axis(out, dim1, 1);
        return out;
    }

    LogicalId rms_norm_gemma_atomic(LogicalId x_id, LogicalId weight_id, uint32_t dim0, uint32_t dim_size)
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
        LogicalId inv_std_expanded = repeat_3d_axis(inv_std, dim_size, 2);
        LogicalId x_norm = g.mul(x_id, inv_std_expanded);
        LogicalId weight_expanded = expand_1d_to_3d(weight_id, dim_size, dim0, seq_len);
        LogicalId one_node_full = expand_scalar_to_3d(one_fp32, dim0, seq_len, dim_size);
        LogicalId scale = g.add(weight_expanded, one_node_full);
        return g.mul(x_norm, scale);
    }

    LogicalId tanh_atomic(LogicalId x_id, uint32_t last_dim)
    {
        float neg_two_val = -2.0f;
        LogicalId neg_two = expand_scalar_to_3d(g.constant({1}, &neg_two_val, DType::FLOAT32), 1, seq_len, last_dim);
        float two_val = 2.0f;
        LogicalId two = expand_scalar_to_3d(g.constant({1}, &two_val, DType::FLOAT32), 1, seq_len, last_dim);
        float e_val = 2.718281828459045f;
        LogicalId e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), 1, seq_len, last_dim);
        LogicalId one_node = expand_scalar_to_3d(one_fp32, 1, seq_len, last_dim);
        LogicalId neg_2x = g.mul(x_id, neg_two);
        LogicalId exp_neg_2x = g.pow(e_node, neg_2x);
        LogicalId den = g.add(one_node, exp_neg_2x);
        LogicalId quotient = g.div(two, den);
        LogicalId neg_one = g.neg(one_node);
        return g.add(quotient, neg_one);
    }

    LogicalId gelu_atomic(LogicalId x_id, uint32_t last_dim)
    {
        float c1_val = 0.044715f;
        LogicalId c1_node = expand_scalar_to_3d(g.constant({1}, &c1_val, DType::FLOAT32), 1, seq_len, last_dim);
        float c2_val = 0.79788456f;
        LogicalId c2_node = expand_scalar_to_3d(g.constant({1}, &c2_val, DType::FLOAT32), 1, seq_len, last_dim);
        LogicalId x_sq = g.mul(x_id, x_id);
        LogicalId x_cube = g.mul(x_sq, x_id);
        LogicalId term1 = g.mul(x_cube, c1_node);
        LogicalId term2 = g.add(x_id, term1);
        LogicalId term3 = g.mul(term2, c2_node);
        LogicalId tanh_result = tanh_atomic(term3, last_dim);
        LogicalId one_node = expand_scalar_to_3d(one_fp32, 1, seq_len, last_dim);
        LogicalId term4 = g.add(one_node, tanh_result);
        LogicalId half_node = expand_scalar_to_3d(half_fp32, 1, seq_len, last_dim);
        LogicalId term5 = g.mul(x_id, half_node);
        return g.mul(term5, term4);
    }

    std::tuple<LogicalId, LogicalId> compute_rope()
    {
        int32_t start_val = 0, stop_val = cfg.head_dim, step_val = 2;
        LogicalId start = g.constant({1}, &start_val, DType::INT32);
        LogicalId stop = g.constant({1}, &stop_val, DType::INT32);
        LogicalId step = g.constant({1}, &step_val, DType::INT32);
        LogicalId indices_int = g.arange(start, stop, step);
        LogicalId indices = g.cast(indices_int, DType::FLOAT32);
        float h_dim_val = (float)cfg.head_dim;
        LogicalId h_dim_fp = g.constant({1}, &h_dim_val, DType::FLOAT32);
        int32_t shape_1d[] = {(int32_t)(cfg.head_dim / 2)};
        LogicalId h_dim_fp_1d = g.repeat(h_dim_fp, g.constant({1}, shape_1d, DType::INT32), g.constant({1}, &start_val, DType::INT32));
        LogicalId exponent = g.div(indices, h_dim_fp_1d);
        float theta_val = 10000.0f;
        LogicalId theta = g.constant({1}, &theta_val, DType::FLOAT32);
        LogicalId theta_1d = g.repeat(theta, g.constant({1}, shape_1d, DType::INT32), g.constant({1}, &start_val, DType::INT32));
        LogicalId base_to_exponent = g.pow(theta_1d, exponent);
        LogicalId one_1d = g.repeat(one_fp32, g.constant({1}, shape_1d, DType::INT32), g.constant({1}, &start_val, DType::INT32));
        LogicalId inv_freq = g.div(one_1d, base_to_exponent);

        int32_t pos_stop_val = seq_len;
        int32_t pos_step_val = 1;
        LogicalId pos_stop = g.constant({1}, &pos_stop_val, DType::INT32);
        LogicalId pos_step = g.constant({1}, &pos_step_val, DType::INT32);
        LogicalId pos_int = g.arange(start, pos_stop, pos_step);
        LogicalId pos = g.cast(pos_int, DType::FLOAT32);
        int32_t pos_col_shape[] = {(int32_t)seq_len, 1};
        LogicalId pos_col = g.reshape(pos, g.constant({2}, pos_col_shape, DType::INT32));
        int32_t freq_row_shape[] = {1, (int32_t)cfg.head_dim / 2};
        LogicalId freq_row = g.reshape(inv_freq, g.constant({2}, freq_row_shape, DType::INT32));
        LogicalId pos_col_expanded = repeat_3d_axis(pos_col, cfg.head_dim / 2, 1);
        LogicalId freq_row_expanded = repeat_3d_axis(freq_row, seq_len, 0);
        LogicalId angles_half = g.mul(pos_col_expanded, freq_row_expanded);
        int32_t axis_val = 1;
        LogicalId axis_node = g.constant({1}, &axis_val, DType::INT32);
        LogicalId angles = g.concat({angles_half, angles_half}, axis_node);

        int32_t final_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.head_dim};
        LogicalId final_shape_node = g.constant({3}, final_shape, DType::INT32);
        LogicalId cos_out = g.reshape(g.cos(angles), final_shape_node);
        LogicalId sin_out = g.reshape(g.sin(angles), final_shape_node);
        return {cos_out, sin_out};
    }

    LogicalId apply_rope(LogicalId x_id, LogicalId cos_id, LogicalId sin_id, uint32_t n_groups)
    {
        int32_t starts1[] = {0, 0, 0};
        int32_t ends1[] = {(int32_t)n_groups, (int32_t)seq_len, (int32_t)cfg.head_dim / 2};
        int32_t steps1[] = {1, 1, 1};
        LogicalId x1 = g.slice(x_id, g.constant({3}, starts1, DType::INT32), g.constant({3}, ends1, DType::INT32), g.constant({3}, steps1, DType::INT32));
        x1 = g.contiguous(x1);

        int32_t starts2[] = {0, 0, (int32_t)cfg.head_dim / 2};
        int32_t ends2[] = {(int32_t)n_groups, (int32_t)seq_len, (int32_t)cfg.head_dim};
        LogicalId x2 = g.slice(x_id, g.constant({3}, starts2, DType::INT32), g.constant({3}, ends2, DType::INT32), g.constant({3}, steps1, DType::INT32));
        LogicalId neg_x2 = g.neg(x2);
        int32_t axis = 2;
        LogicalId rotated = g.concat({neg_x2, x1}, g.constant({1}, &axis, DType::INT32));
        LogicalId cos_expanded = repeat_3d_axis(cos_id, n_groups, 0);
        LogicalId sin_expanded = repeat_3d_axis(sin_id, n_groups, 0);
        LogicalId term1 = g.mul(x_id, cos_expanded);
        LogicalId term2 = g.mul(rotated, sin_expanded);
        return g.add(term1, term2);
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
        LogicalId neg_inf_expanded = repeat_3d_axis(neg_inf_reshaped, seq_len, 0);
        neg_inf_expanded = repeat_3d_axis(neg_inf_expanded, seq_len, 1);
        LogicalId scaled_mask = g.mul(triu_mask, neg_inf_expanded);
        int32_t final_shape[] = {1, (int32_t)seq_len, (int32_t)seq_len};
        return g.reshape(scaled_mask, g.constant({3}, final_shape, DType::INT32));
    }

    std::tuple<LogicalId, LogicalId, LogicalId> attention_qkv_atomic(LogicalId x, const std::string &prefix, LogicalId rope_cos, LogicalId rope_sin)
    {
        int32_t perm_dims[] = {1, 0};
        LogicalId dims_node = g.constant({2}, perm_dims, DType::INT32);

        auto project = [&](const std::string &suffix, uint32_t in_d, uint32_t out_d)
        {
            LogicalId w = weight(w_path, prefix + suffix);
            LogicalId w_t = g.permute(w, dims_node);
            w_t = g.contiguous(w_t);
            int32_t s3[] = {1, (int32_t)in_d, (int32_t)out_d};
            return g.dot(x, g.reshape(w_t, g.constant({3}, s3, DType::INT32)));
        };

        LogicalId q = project(".self_attn.q_proj.weight", cfg.emb_dim, cfg.n_heads * cfg.head_dim);
        LogicalId k = project(".self_attn.k_proj.weight", cfg.emb_dim, cfg.n_kv_groups * cfg.head_dim);
        LogicalId v = project(".self_attn.v_proj.weight", cfg.emb_dim, cfg.n_kv_groups * cfg.head_dim);

        int32_t perm4[] = {0, 2, 1, 3};
        LogicalId perm4_node = g.constant({4}, perm4, DType::INT32);

        int32_t q_shape4[] = {1, (int32_t)seq_len, (int32_t)cfg.n_heads, (int32_t)cfg.head_dim};
        LogicalId q_4d = g.reshape(q, g.constant({4}, q_shape4, DType::INT32));
        LogicalId q_perm = g.permute(q_4d, perm4_node);
        q_perm = g.contiguous(q_perm);
        int32_t shape3_q[] = {(int32_t)cfg.n_heads, (int32_t)seq_len, (int32_t)cfg.head_dim};
        q = g.reshape(q_perm, g.constant({3}, shape3_q, DType::INT32));

        int32_t k_shape4[] = {1, (int32_t)seq_len, (int32_t)cfg.n_kv_groups, (int32_t)cfg.head_dim};
        LogicalId k_4d = g.reshape(k, g.constant({4}, k_shape4, DType::INT32));
        LogicalId k_perm = g.permute(k_4d, perm4_node);
        k_perm = g.contiguous(k_perm);
        int32_t shape3_k[] = {(int32_t)cfg.n_kv_groups, (int32_t)seq_len, (int32_t)cfg.head_dim};
        k = g.reshape(k_perm, g.constant({3}, shape3_k, DType::INT32));

        LogicalId v_4d = g.reshape(v, g.constant({4}, k_shape4, DType::INT32));
        LogicalId v_perm = g.permute(v_4d, perm4_node);
        v_perm = g.contiguous(v_perm);
        v = g.reshape(v_perm, g.constant({3}, shape3_k, DType::INT32));

        LogicalId q_norm_w = weight(w_path, prefix + ".self_attn.q_norm.weight");
        q = rms_norm_gemma_atomic(q, q_norm_w, cfg.n_heads, cfg.head_dim);
        LogicalId k_norm_w = weight(w_path, prefix + ".self_attn.k_norm.weight");
        k = rms_norm_gemma_atomic(k, k_norm_w, cfg.n_kv_groups, cfg.head_dim);

        q = apply_rope(q, rope_cos, rope_sin, cfg.n_heads);
        k = apply_rope(k, rope_cos, rope_sin, cfg.n_kv_groups);

        if (cfg.n_heads != cfg.n_kv_groups)
        {
            uint32_t repeats = cfg.n_heads / cfg.n_kv_groups;
            int32_t rep[] = {(int32_t)repeats};
            LogicalId rep_node = g.constant({1}, rep, DType::INT32);
            int32_t ax[] = {0};
            LogicalId ax_node = g.constant({1}, ax, DType::INT32);
            k = g.repeat(k, rep_node, ax_node);
            v = g.repeat(v, rep_node, ax_node);
        }
        return std::make_tuple(g.contiguous(q), g.contiguous(k), g.contiguous(v));
    }

    LogicalId attention_output_atomic(std::tuple<LogicalId, LogicalId, LogicalId> qkv, const std::string &prefix, LogicalId mask_id)
    {
        LogicalId q = std::get<0>(qkv);
        LogicalId k = std::get<1>(qkv);
        LogicalId v = std::get<2>(qkv);

        float scale_val = 1.0f / std::sqrt((float)cfg.query_pre_attn_scalar);
        LogicalId scale_node = expand_scalar_to_3d(g.constant({1}, &scale_val, DType::FLOAT32), cfg.n_heads, seq_len, cfg.head_dim);
        LogicalId scaled_q = g.mul(q, scale_node);

        int32_t perm_k[] = {0, 2, 1};
        LogicalId k_t = g.permute(k, g.constant({3}, perm_k, DType::INT32));
        k_t = g.contiguous(k_t);

        LogicalId scores = g.dot(scaled_q, k_t);
        LogicalId mask_expanded = repeat_3d_axis(mask_id, cfg.n_heads, 0);
        scores = g.add(scores, mask_expanded);

        int32_t axis_val = -1;
        LogicalId max_scores = g.max(scores, g.constant({1}, &axis_val, DType::INT32));
        max_scores = repeat_3d_axis(max_scores, seq_len, 2);
        LogicalId shifted_scores = g.add(scores, g.neg(max_scores));

        float e_val = 2.718281828459045f;
        LogicalId e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), cfg.n_heads, seq_len, seq_len);
        LogicalId exp_scores = g.pow(e_node, shifted_scores);

        LogicalId sum_exp = g.sum(exp_scores, g.constant({1}, &axis_val, DType::INT32));
        sum_exp = repeat_3d_axis(sum_exp, seq_len, 2);

        LogicalId probs = g.div(exp_scores, sum_exp);
        LogicalId context = g.dot(probs, v);

        int32_t ctx_shape4[] = {1, (int32_t)cfg.n_heads, (int32_t)seq_len, (int32_t)cfg.head_dim};
        LogicalId ctx_4d = g.reshape(context, g.constant({4}, ctx_shape4, DType::INT32));

        int32_t perm_ctx[] = {0, 2, 1, 3};
        LogicalId ctx_perm = g.permute(ctx_4d, g.constant({4}, perm_ctx, DType::INT32));
        ctx_perm = g.contiguous(ctx_perm);

        int32_t ctx_shape3[] = {1, (int32_t)seq_len, (int32_t)(cfg.n_heads * cfg.head_dim)};
        LogicalId ctx_flat = g.reshape(ctx_perm, g.constant({3}, ctx_shape3, DType::INT32));

        LogicalId w_o = weight(w_path, prefix + ".self_attn.o_proj.weight");
        int32_t perm_dims[] = {1, 0};
        LogicalId w_o_t = g.permute(w_o, g.constant({2}, perm_dims, DType::INT32));
        w_o_t = g.contiguous(w_o_t);

        int32_t s3[] = {1, (int32_t)(cfg.n_heads * cfg.head_dim), (int32_t)cfg.emb_dim};
        LogicalId w_o_3d = g.reshape(w_o_t, g.constant({3}, s3, DType::INT32));
        return g.dot(ctx_flat, w_o_3d);
    }

    LogicalId mlp_atomic(LogicalId x, const std::string &prefix)
    {
        int32_t perm_dims[] = {1, 0};
        LogicalId p_node = g.constant({2}, perm_dims, DType::INT32);

        auto project = [&](const std::string &suffix, uint32_t in_d, uint32_t out_d)
        {
            LogicalId w = weight(w_path, prefix + suffix);
            LogicalId w_t = g.permute(w, p_node);
            w_t = g.contiguous(w_t);
            int32_t s3[] = {1, (int32_t)in_d, (int32_t)out_d};
            return g.dot(x, g.reshape(w_t, g.constant({3}, s3, DType::INT32)));
        };

        LogicalId gate = project(".mlp.gate_proj.weight", cfg.emb_dim, cfg.hidden_dim);
        gate = gelu_atomic(gate, cfg.hidden_dim);
        LogicalId up = project(".mlp.up_proj.weight", cfg.emb_dim, cfg.hidden_dim);
        LogicalId gate_up = g.mul(gate, up);
        LogicalId w_down = weight(w_path, prefix + ".mlp.down_proj.weight");
        LogicalId w_down_t = g.permute(w_down, p_node);
        w_down_t = g.contiguous(w_down_t);
        int32_t s3[] = {1, (int32_t)cfg.hidden_dim, (int32_t)cfg.emb_dim};
        return g.dot(gate_up, g.reshape(w_down_t, g.constant({3}, s3, DType::INT32)));
    }

    LogicalId build_graph(LogicalId input_ids_id)
    {
        LogicalId w_emb = weight(w_path, "model.embed_tokens.weight");
        LogicalId x = g.gather(w_emb, input_ids_id);
        float scale_val = std::sqrt((float)cfg.emb_dim);
        LogicalId scale_node = expand_scalar_to_3d(g.constant({1}, &scale_val, DType::FLOAT32), 1, seq_len, cfg.emb_dim);
        x = g.mul(x, scale_node);

        auto rope = compute_rope();
        LogicalId rope_cos = std::get<0>(rope);
        LogicalId rope_sin = std::get<1>(rope);
        LogicalId mask_id = compute_causal_mask();

        for (uint32_t i = 0; i < cfg.n_layers; ++i)
        {
            std::string prefix = "model.layers." + std::to_string(i);
            LogicalId residual = x;
            LogicalId w_ln1 = weight(w_path, prefix + ".input_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_ln1, 1, cfg.emb_dim);
            auto qkv = attention_qkv_atomic(x, prefix, rope_cos, rope_sin);
            x = attention_output_atomic(qkv, prefix, mask_id);
            LogicalId w_post_attn = weight(w_path, prefix + ".post_attention_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_post_attn, 1, cfg.emb_dim);
            x = g.add(residual, x);
            residual = x;
            LogicalId w_ln2 = weight(w_path, prefix + ".pre_feedforward_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_ln2, 1, cfg.emb_dim);
            x = mlp_atomic(x, prefix);
            LogicalId w_post_ff = weight(w_path, prefix + ".post_feedforward_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_post_ff, 1, cfg.emb_dim);
            x = g.add(residual, x);
        }

        LogicalId w_final_ln = weight(w_path, "model.norm.weight");
        x = rms_norm_gemma_atomic(x, w_final_ln, 1, cfg.emb_dim);
        int32_t perm_dims[] = {1, 0};
        LogicalId dims_node = g.constant({2}, perm_dims, DType::INT32);
        LogicalId w_emb_t = g.permute(w_emb, dims_node);
        w_emb_t = g.contiguous(w_emb_t);
        int32_t s3[] = {1, (int32_t)cfg.emb_dim, (int32_t)cfg.vocab_size};
        LogicalId w_emb_3d = g.reshape(w_emb_t, g.constant({3}, s3, DType::INT32));
        return g.dot(x, w_emb_3d);
    }
};