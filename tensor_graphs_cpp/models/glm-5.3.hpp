#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "core/common/constants.hpp"
#include "core/graph.hpp"
#include "core/loaders/resolver.hpp"
#include "core/memory.hpp"
#include "core/types.hpp"

struct GLM5_3_Config
{
    uint32_t vocab_size = 154880;
    uint32_t hidden_size = 6144;
    uint32_t intermediate_size = 12288;
    uint32_t moe_intermediate_size = 2048;
    uint32_t num_hidden_layers = 78;
    uint32_t num_attention_heads = 64;
    uint32_t num_key_value_heads = 64;
    uint32_t q_lora_rank = 2048;
    uint32_t kv_lora_rank = 512;
    uint32_t qk_head_dim = 256;
    uint32_t qk_nope_head_dim = 192;
    uint32_t qk_rope_head_dim = 64;
    uint32_t v_head_dim = 256;
    float rope_theta = 8000000.0f;
    float rms_norm_eps = 1e-5f;
    uint32_t n_routed_experts = 256;
    uint32_t num_experts_per_tok = 8;
    uint32_t n_shared_experts = 1;
    float routed_scaling_factor = 2.5f;
    uint32_t first_k_dense_replace = 3;
    uint32_t index_n_heads = 32;
    uint32_t index_head_dim = 128;
    uint32_t index_topk = 2048;

    bool isIndexerFull(uint32_t layer_idx) const
    {
        return layer_idx < 3 || (layer_idx - 2) % 4 == 0;
    }

    bool isMlpDense(uint32_t layer_idx) const
    {
        return layer_idx < first_k_dense_replace;
    }
};

using GLM5_3ModelConfig = GLM5_3_Config;

class GLM5_3_Model
{
  private:
    GLM5_3_Config cfg;
    Graph &g;
    MemoryManager &mem;
    const std::string w_path;
    uint32_t seq_len;
    LogicalId one_fp32;
    LogicalId eps_fp32;
    LogicalId half_fp32;
    LogicalId rope_cos;
    LogicalId rope_sin;

  public:
    GLM5_3_Model(GLM5_3_Config config, uint32_t sequence_length, Graph &graph, MemoryManager &memory,
                 const std::string &weight_path)
        : cfg(config), g(graph), mem(memory), w_path(weight_path), seq_len(sequence_length)
    {
        float one_val = 1.0f;
        one_fp32 = g.constant({1}, &one_val, DType::FLOAT32);
        float eps_val = cfg.rms_norm_eps;
        eps_fp32 = g.constant({1}, &eps_val, DType::FLOAT32);
        float half_val = 0.5f;
        half_fp32 = g.constant({1}, &half_val, DType::FLOAT32);

        // Precompute RoPE cos/sin tables for qk_rope_head_dim (64)
        // 32 frequency pairs for d in 0, 2, ..., 62
        uint32_t half_rope = cfg.qk_rope_head_dim / 2;
        std::vector<float> cos_table(seq_len * half_rope);
        std::vector<float> sin_table(seq_len * half_rope);
        for (uint32_t s = 0; s < seq_len; ++s)
        {
            for (uint32_t j = 0; j < half_rope; ++j)
            {
                uint32_t d = j * 2;
                float freq = 1.0f / std::pow(cfg.rope_theta, static_cast<float>(d) / static_cast<float>(cfg.qk_rope_head_dim));
                float angle = static_cast<float>(s) * freq;
                cos_table[s * half_rope + j] = std::cos(angle);
                sin_table[s * half_rope + j] = std::sin(angle);
            }
        }
        rope_cos = g.constant({1, seq_len, 1, half_rope}, cos_table.data(), DType::FLOAT32);
        rope_sin = g.constant({1, seq_len, 1, half_rope}, sin_table.data(), DType::FLOAT32);
    }

    LogicalId loadFp8Weight(const std::string &w_name, const std::string &scale_name, uint32_t out_d, uint32_t in_d)
    {
        LogicalId raw_weight = g.weight(w_path, w_name);
        LogicalId unpacked_f32 = g.cast(raw_weight, DType::FLOAT32);

        LogicalId raw_scale = g.weight(w_path, scale_name);
        LogicalId scale_f32 = g.cast(raw_scale, DType::FLOAT32);

        uint32_t scale_h = (out_d + 127) / 128;
        uint32_t scale_w = (in_d + 127) / 128;
        int32_t sh4_scale[] = {(int32_t)scale_h, 1, (int32_t)scale_w, 1};
        LogicalId scale_reshaped = g.reshape(scale_f32, g.constant({4}, sh4_scale, DType::INT32));
        LogicalId scale_rep1 = g.repeat(scale_reshaped, 128, 1);
        LogicalId scale_rep2 = g.repeat(scale_rep1, 128, 3);
        int32_t sh2_rep[] = {(int32_t)(scale_h * 128), (int32_t)(scale_w * 128)};
        LogicalId scale_grid = g.reshape(scale_rep2, g.constant({2}, sh2_rep, DType::INT32));

        if (scale_h * 128 != out_d || scale_w * 128 != in_d)
        {
            int32_t starts[] = {0, 0};
            int32_t ends[] = {(int32_t)out_d, (int32_t)in_d};
            int32_t steps[] = {1, 1};
            scale_grid = g.slice(scale_grid, g.constant({2}, starts, DType::INT32),
                                 g.constant({2}, ends, DType::INT32), g.constant({2}, steps, DType::INT32));
        }

        return g.mul(unpacked_f32, scale_grid);
    }

    LogicalId weight(const std::string &name, uint32_t in_d = 0, uint32_t out_d = 0)
    {
        std::string scale_name = "";
        if (name.length() >= 7 && name.substr(name.length() - 7) == ".weight")
        {
            scale_name = name.substr(0, name.length() - 7) + ".weight_scale_inv";
        }
        else
        {
            scale_name = name + "_scale_inv";
        }

        if (TensorResolver::get().hasTensor(w_path, scale_name))
        {
            if (in_d == 0 || out_d == 0)
            {
                const auto &meta = TensorResolver::get().getMetadata(w_path, name);
                if (meta.shape.size() == 2)
                {
                    out_d = meta.shape[0];
                    in_d = meta.shape[1];
                }
            }
            return loadFp8Weight(name, scale_name, out_d, in_d);
        }

        LogicalId raw_weight = g.weight(w_path, name);
        return g.cast(raw_weight, DType::FLOAT32);
    }

    LogicalId linear(LogicalId x, const std::string &w_name, uint32_t in_d, uint32_t out_d)
    {
        LogicalId w = weight(w_name, in_d, out_d);
        int32_t p[] = {1, 0};
        LogicalId w_t = g.contiguous(g.permute(w, g.constant({2}, p, DType::INT32)));
        int32_t sh3[] = {1, (int32_t)in_d, (int32_t)out_d};
        return g.dot(x, g.reshape(w_t, g.constant({3}, sh3, DType::INT32)));
    }

    LogicalId rmsNorm(LogicalId x_id, const std::string &w_name, uint32_t dim_size)
    {
        LogicalId weight_id = weight(w_name);
        LogicalId x_sq = g.mul(x_id, x_id);
        int32_t axis_val = -1;
        LogicalId sum_sq = g.sum(x_sq, g.constant({1}, &axis_val, DType::INT32));
        LogicalId mean_sq = g.div(sum_sq, g.fill((float)dim_size, {1, seq_len, 1}));
        LogicalId half_node = g.fill(0.5f, {1, seq_len, 1});
        LogicalId std = g.pow(g.add(mean_sq, g.fill(cfg.rms_norm_eps, {1, seq_len, 1})), half_node);
        LogicalId one_node = g.fill(1.0f, {1, seq_len, 1});
        LogicalId inv_std = g.repeat(g.div(one_node, std), dim_size, 2);
        int32_t w_shape[] = {1, 1, (int32_t)dim_size};
        LogicalId w_exp = g.repeat(g.reshape(weight_id, g.constant({3}, w_shape, DType::INT32)), seq_len, 1);
        return g.mul(g.mul(x_id, inv_std), w_exp);
    }

    LogicalId layerNorm(LogicalId x_id, const std::string &w_name, const std::string &b_name, uint32_t dim_size)
    {
        LogicalId w = weight(w_name);
        LogicalId b = weight(b_name);

        int32_t axis_val = -1;
        LogicalId sum_x = g.sum(x_id, g.constant({1}, &axis_val, DType::INT32));
        LogicalId mean = g.div(sum_x, g.fill((float)dim_size, {1, seq_len, 1}));
        mean = g.repeat(mean, dim_size, 2);
        LogicalId x_sub = g.add(x_id, g.neg(mean));

        LogicalId x_sq = g.mul(x_sub, x_sub);
        LogicalId sum_sq = g.sum(x_sq, g.constant({1}, &axis_val, DType::INT32));
        LogicalId var = g.div(sum_sq, g.fill((float)dim_size, {1, seq_len, 1}));
        LogicalId half_node = g.fill(0.5f, {1, seq_len, 1});
        LogicalId std = g.pow(g.add(var, g.fill(1e-6f, {1, seq_len, 1})), half_node);
        LogicalId one_node = g.fill(1.0f, {1, seq_len, 1});
        LogicalId inv_std = g.repeat(g.div(one_node, std), dim_size, 2);

        int32_t wb_shape[] = {1, 1, (int32_t)dim_size};
        LogicalId w_exp = g.repeat(g.reshape(w, g.constant({3}, wb_shape, DType::INT32)), seq_len, 1);
        LogicalId b_exp = g.repeat(g.reshape(b, g.constant({3}, wb_shape, DType::INT32)), seq_len, 1);

        return g.add(g.mul(g.mul(x_sub, inv_std), w_exp), b_exp);
    }

    LogicalId silu(LogicalId x, const std::vector<uint32_t> &shape)
    {
        LogicalId neg_one = g.fill(-1.0f, shape);
        LogicalId neg_x = g.mul(x, neg_one);
        LogicalId e_node = g.fill(TGConstants::E, shape);
        LogicalId exp_neg_x = g.pow(e_node, neg_x);
        LogicalId one_node = g.fill(1.0f, shape);
        LogicalId den = g.add(one_node, exp_neg_x);
        LogicalId sig = g.div(one_node, den);
        return g.mul(x, sig);
    }

    LogicalId sigmoid(LogicalId x, uint32_t last_dim)
    {
        LogicalId neg_one = g.fill(-1.0f, {1, seq_len, last_dim});
        LogicalId neg_x = g.mul(x, neg_one);
        LogicalId e_node = g.fill(TGConstants::E, {1, seq_len, last_dim});
        LogicalId exp_neg_x = g.pow(e_node, neg_x);
        LogicalId one_node = g.fill(1.0f, {1, seq_len, last_dim});
        return g.div(one_node, g.add(one_node, exp_neg_x));
    }

    LogicalId applyRopeInterleave(LogicalId x, uint32_t n_heads, uint32_t rope_dim)
    {
        int32_t starts_even[] = {0, 0, 0, 0};
        int32_t ends_even[] = {1, (int32_t)seq_len, (int32_t)n_heads, (int32_t)rope_dim};
        int32_t steps_even[] = {1, 1, 1, 2};
        LogicalId x1 = g.contiguous(g.slice(x, g.constant({4}, starts_even, DType::INT32),
                                            g.constant({4}, ends_even, DType::INT32),
                                            g.constant({4}, steps_even, DType::INT32)));

        int32_t starts_odd[] = {0, 0, 0, 1};
        int32_t ends_odd[] = {1, (int32_t)seq_len, (int32_t)n_heads, (int32_t)rope_dim};
        int32_t steps_odd[] = {1, 1, 1, 2};
        LogicalId x2 = g.contiguous(g.slice(x, g.constant({4}, starts_odd, DType::INT32),
                                            g.constant({4}, ends_odd, DType::INT32),
                                            g.constant({4}, steps_odd, DType::INT32)));

        LogicalId cos_exp = (n_heads > 1) ? g.repeat(rope_cos, n_heads, 2) : rope_cos;
        LogicalId sin_exp = (n_heads > 1) ? g.repeat(rope_sin, n_heads, 2) : rope_sin;

        LogicalId out1 = g.add(g.mul(x1, cos_exp), g.neg(g.mul(x2, sin_exp)));
        LogicalId out2 = g.add(g.mul(x2, cos_exp), g.mul(x1, sin_exp));

        int32_t ax_3 = 3;
        return g.concat({out1, out2}, g.constant({1}, &ax_3, DType::INT32));
    }

    LogicalId createCausalMask()
    {
        int32_t m_shape[] = {(int32_t)seq_len, (int32_t)seq_len};
        float one_val = 1.0f;
        LogicalId ones = g.fill(g.constant({1}, &one_val, DType::FLOAT32), g.constant({2}, m_shape, DType::INT32));
        int32_t k_val = 1;
        LogicalId triu_mask = g.triu(ones, g.constant({1}, &k_val, DType::INT32));
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

    LogicalId indexerAtomic(LogicalId x, LogicalId q_resid, const std::string &prefix, LogicalId mask_3d)
    {
        LogicalId q_idx = linear(q_resid, prefix + "self_attn.indexer.wq_b.weight", cfg.q_lora_rank,
                                 cfg.index_n_heads * cfg.index_head_dim);
        int32_t sh4_q[] = {1, (int32_t)seq_len, (int32_t)cfg.index_n_heads, (int32_t)cfg.index_head_dim};
        q_idx = g.reshape(q_idx, g.constant({4}, sh4_q, DType::INT32));

        int32_t steps4[] = {1, 1, 1, 1};
        int32_t st_rot[] = {0, 0, 0, 0}, en_rot[] = {1, (int32_t)seq_len, (int32_t)cfg.index_n_heads, (int32_t)cfg.qk_rope_head_dim};
        LogicalId q_rot = g.contiguous(g.slice(q_idx, g.constant({4}, st_rot, DType::INT32),
                                               g.constant({4}, en_rot, DType::INT32), g.constant({4}, steps4, DType::INT32)));

        int32_t st_pass[] = {0, 0, 0, (int32_t)cfg.qk_rope_head_dim},
                en_pass[] = {1, (int32_t)seq_len, (int32_t)cfg.index_n_heads, (int32_t)cfg.index_head_dim};
        LogicalId q_pass = g.contiguous(g.slice(q_idx, g.constant({4}, st_pass, DType::INT32),
                                                g.constant({4}, en_pass, DType::INT32), g.constant({4}, steps4, DType::INT32)));

        q_rot = applyRopeInterleave(q_rot, cfg.index_n_heads, cfg.qk_rope_head_dim);
        int32_t ax_3 = 3;
        q_idx = g.concat({q_rot, q_pass}, g.constant({1}, &ax_3, DType::INT32));

        LogicalId k_proj = linear(x, prefix + "self_attn.indexer.wk.weight", cfg.hidden_size, cfg.index_head_dim);
        LogicalId k_normed = layerNorm(k_proj, prefix + "self_attn.indexer.k_norm.weight",
                                       prefix + "self_attn.indexer.k_norm.bias", cfg.index_head_dim);
        int32_t sh4_k[] = {1, (int32_t)seq_len, 1, (int32_t)cfg.index_head_dim};
        k_normed = g.reshape(k_normed, g.constant({4}, sh4_k, DType::INT32));

        int32_t en_k_rot[] = {1, (int32_t)seq_len, 1, (int32_t)cfg.qk_rope_head_dim};
        LogicalId k_rot = g.contiguous(g.slice(k_normed, g.constant({4}, st_rot, DType::INT32),
                                               g.constant({4}, en_k_rot, DType::INT32), g.constant({4}, steps4, DType::INT32)));
        int32_t st_k_pass[] = {0, 0, 0, (int32_t)cfg.qk_rope_head_dim},
                en_k_pass[] = {1, (int32_t)seq_len, 1, (int32_t)cfg.index_head_dim};
        LogicalId k_pass = g.contiguous(g.slice(k_normed, g.constant({4}, st_k_pass, DType::INT32),
                                                g.constant({4}, en_k_pass, DType::INT32), g.constant({4}, steps4, DType::INT32)));

        k_rot = applyRopeInterleave(k_rot, 1, cfg.qk_rope_head_dim);
        k_normed = g.concat({k_rot, k_pass}, g.constant({1}, &ax_3, DType::INT32));
        int32_t sh3_k[] = {1, (int32_t)seq_len, (int32_t)cfg.index_head_dim};
        LogicalId k_3d = g.reshape(k_normed, g.constant({3}, sh3_k, DType::INT32));

        // Compute scores: [index_n_heads, seq_len, seq_len]
        int32_t sh4_p_q[] = {0, 2, 1, 3};
        LogicalId q_perm = g.contiguous(g.permute(q_idx, g.constant({4}, sh4_p_q, DType::INT32)));
        int32_t sh3_qp[] = {(int32_t)cfg.index_n_heads, (int32_t)seq_len, (int32_t)cfg.index_head_dim};
        LogicalId q_heads = g.reshape(q_perm, g.constant({3}, sh3_qp, DType::INT32));

        int32_t p_k[] = {0, 2, 1};
        LogicalId k_t = g.contiguous(g.permute(k_3d, g.constant({3}, p_k, DType::INT32)));
        LogicalId k_t_rep = g.repeat(k_t, cfg.index_n_heads, 0);

        LogicalId raw_scores = g.dot(q_heads, k_t_rep);
        float scale_idx = 1.0f / std::sqrt(static_cast<float>(cfg.index_head_dim));
        LogicalId scale_idx_node = g.fill(g.constant({1}, &scale_idx, DType::FLOAT32),
                                          {cfg.index_n_heads, seq_len, seq_len});
        LogicalId scaled_scores = g.mul(raw_scores, scale_idx_node);
        LogicalId relu_scores = g.relu(scaled_scores, {cfg.index_n_heads, seq_len, seq_len});

        // Weights projection
        LogicalId weights_proj = linear(x, prefix + "self_attn.indexer.weights_proj.weight",
                                        cfg.hidden_size, cfg.index_n_heads);
        float w_scale = 1.0f / std::sqrt(static_cast<float>(cfg.index_n_heads));
        LogicalId w_scale_node = g.fill(g.constant({1}, &w_scale, DType::FLOAT32),
                                        {1, seq_len, cfg.index_n_heads});
        LogicalId weights = g.mul(weights_proj, w_scale_node);

        int32_t p_s[] = {1, 2, 0};
        LogicalId scores_sth = g.contiguous(g.permute(relu_scores, g.constant({3}, p_s, DType::INT32)));
        int32_t sh4_sth[] = {1, (int32_t)seq_len, (int32_t)seq_len, (int32_t)cfg.index_n_heads};
        LogicalId scores_4d = g.reshape(scores_sth, g.constant({4}, sh4_sth, DType::INT32));

        int32_t sh4_w[] = {1, (int32_t)seq_len, 1, (int32_t)cfg.index_n_heads};
        LogicalId weights_4d = g.reshape(weights, g.constant({4}, sh4_w, DType::INT32));
        LogicalId weights_rep = g.repeat(weights_4d, seq_len, 2);

        LogicalId weighted_scores = g.mul(scores_4d, weights_rep);
        LogicalId sum_scores_4d = g.sum(weighted_scores, g.constant({1}, &ax_3, DType::INT32));
        int32_t sh3_ss[] = {1, (int32_t)seq_len, (int32_t)seq_len};
        LogicalId index_scores = g.reshape(sum_scores_4d, g.constant({3}, sh3_ss, DType::INT32));

        index_scores = g.add(index_scores, mask_3d);

        int32_t ax_last = -1;
        int32_t topk = std::min(cfg.index_topk, seq_len);
        return g.argmax(index_scores, g.constant({1}, &ax_last, DType::INT32),
                        g.constant({1}, &topk, DType::INT32));
    }

    LogicalId attentionAtomic(LogicalId x, const std::string &prefix, LogicalId mask_3d, LogicalId q_resid)
    {
        // 1. Q projection
        LogicalId q_b = linear(q_resid, prefix + "self_attn.q_b_proj.weight", cfg.q_lora_rank,
                               cfg.num_attention_heads * cfg.qk_head_dim);
        int32_t sh4_qb[] = {1, (int32_t)seq_len, (int32_t)cfg.num_attention_heads, (int32_t)cfg.qk_head_dim};
        LogicalId q_states = g.reshape(q_b, g.constant({4}, sh4_qb, DType::INT32));

        int32_t steps4[] = {1, 1, 1, 1};
        int32_t st_qpass[] = {0, 0, 0, 0}, en_qpass[] = {1, (int32_t)seq_len, (int32_t)cfg.num_attention_heads, (int32_t)cfg.qk_nope_head_dim};
        LogicalId q_pass = g.contiguous(g.slice(q_states, g.constant({4}, st_qpass, DType::INT32),
                                                g.constant({4}, en_qpass, DType::INT32), g.constant({4}, steps4, DType::INT32)));

        int32_t st_qrot[] = {0, 0, 0, (int32_t)cfg.qk_nope_head_dim},
                en_qrot[] = {1, (int32_t)seq_len, (int32_t)cfg.num_attention_heads, (int32_t)cfg.qk_head_dim};
        LogicalId q_rot = g.contiguous(g.slice(q_states, g.constant({4}, st_qrot, DType::INT32),
                                               g.constant({4}, en_qrot, DType::INT32), g.constant({4}, steps4, DType::INT32)));

        q_rot = applyRopeInterleave(q_rot, cfg.num_attention_heads, cfg.qk_rope_head_dim);
        int32_t ax_3 = 3;
        LogicalId query_states = g.concat({q_pass, q_rot}, g.constant({1}, &ax_3, DType::INT32));

        // 2. KV projection
        LogicalId comp_kv = linear(x, prefix + "self_attn.kv_a_proj_with_mqa.weight", cfg.hidden_size,
                                   cfg.kv_lora_rank + cfg.qk_rope_head_dim);
        int32_t sh3_ckv[] = {1, (int32_t)seq_len, (int32_t)(cfg.kv_lora_rank + cfg.qk_rope_head_dim)};
        comp_kv = g.reshape(comp_kv, g.constant({3}, sh3_ckv, DType::INT32));

        int32_t steps3[] = {1, 1, 1};
        int32_t st_kpl[] = {0, 0, 0}, en_kpl[] = {1, (int32_t)seq_len, (int32_t)cfg.kv_lora_rank};
        LogicalId k_pass_lora = g.contiguous(g.slice(comp_kv, g.constant({3}, st_kpl, DType::INT32),
                                                     g.constant({3}, en_kpl, DType::INT32), g.constant({3}, steps3, DType::INT32)));

        int32_t st_krot[] = {0, 0, (int32_t)cfg.kv_lora_rank}, en_krot[] = {1, (int32_t)seq_len, (int32_t)(cfg.kv_lora_rank + cfg.qk_rope_head_dim)};
        LogicalId k_rot = g.contiguous(g.slice(comp_kv, g.constant({3}, st_krot, DType::INT32),
                                               g.constant({3}, en_krot, DType::INT32), g.constant({3}, steps3, DType::INT32)));

        LogicalId k_pass_normed = rmsNorm(k_pass_lora, prefix + "self_attn.kv_a_layernorm.weight", cfg.kv_lora_rank);

        LogicalId kv_b = linear(k_pass_normed, prefix + "self_attn.kv_b_proj.weight", cfg.kv_lora_rank,
                                cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim));
        int32_t sh4_kvb[] = {1, (int32_t)seq_len, (int32_t)cfg.num_attention_heads,
                             (int32_t)(cfg.qk_nope_head_dim + cfg.v_head_dim)};
        LogicalId kv_states = g.reshape(kv_b, g.constant({4}, sh4_kvb, DType::INT32));

        int32_t st_kpass[] = {0, 0, 0, 0}, en_kpass[] = {1, (int32_t)seq_len, (int32_t)cfg.num_attention_heads, (int32_t)cfg.qk_nope_head_dim};
        LogicalId k_pass = g.contiguous(g.slice(kv_states, g.constant({4}, st_kpass, DType::INT32),
                                                g.constant({4}, en_kpass, DType::INT32), g.constant({4}, steps4, DType::INT32)));

        int32_t st_v[] = {0, 0, 0, (int32_t)cfg.qk_nope_head_dim},
                en_v[] = {1, (int32_t)seq_len, (int32_t)cfg.num_attention_heads, (int32_t)(cfg.qk_nope_head_dim + cfg.v_head_dim)};
        LogicalId value_states = g.contiguous(g.slice(kv_states, g.constant({4}, st_v, DType::INT32),
                                                      g.constant({4}, en_v, DType::INT32), g.constant({4}, steps4, DType::INT32)));

        int32_t sh4_krot[] = {1, (int32_t)seq_len, 1, (int32_t)cfg.qk_rope_head_dim};
        k_rot = g.reshape(k_rot, g.constant({4}, sh4_krot, DType::INT32));
        k_rot = applyRopeInterleave(k_rot, 1, cfg.qk_rope_head_dim);
        LogicalId k_rot_rep = g.repeat(k_rot, cfg.num_attention_heads, 2);

        LogicalId key_states = g.concat({k_pass, k_rot_rep}, g.constant({1}, &ax_3, DType::INT32));

        // 3. Dot-Product Attention: [H, S, D]
        int32_t perm4[] = {0, 2, 1, 3};
        LogicalId p4_node = g.constant({4}, perm4, DType::INT32);

        LogicalId q_perm = g.contiguous(g.permute(query_states, p4_node));
        LogicalId k_perm = g.contiguous(g.permute(key_states, p4_node));
        LogicalId v_perm = g.contiguous(g.permute(value_states, p4_node));

        int32_t sh3_hsd[] = {(int32_t)cfg.num_attention_heads, (int32_t)seq_len, (int32_t)cfg.qk_head_dim};
        LogicalId q_3d = g.reshape(q_perm, g.constant({3}, sh3_hsd, DType::INT32));
        LogicalId k_3d = g.reshape(k_perm, g.constant({3}, sh3_hsd, DType::INT32));
        int32_t sh3_v[] = {(int32_t)cfg.num_attention_heads, (int32_t)seq_len, (int32_t)cfg.v_head_dim};
        LogicalId v_3d = g.reshape(v_perm, g.constant({3}, sh3_v, DType::INT32));

        float scale_val = 1.0f / std::sqrt(static_cast<float>(cfg.qk_head_dim));
        LogicalId scale_node = g.fill(g.constant({1}, &scale_val, DType::FLOAT32),
                                      {cfg.num_attention_heads, seq_len, cfg.qk_head_dim});
        LogicalId scaled_q = g.mul(q_3d, scale_node);

        int32_t perm_k[] = {0, 2, 1};
        LogicalId k_t = g.contiguous(g.permute(k_3d, g.constant({3}, perm_k, DType::INT32)));

        LogicalId scores = g.dot(scaled_q, k_t);
        LogicalId mask_rep = g.repeat(mask_3d, cfg.num_attention_heads, 0);
        scores = g.add(scores, mask_rep);

        int32_t ax_last = -1;
        LogicalId max_scores = g.max(scores, g.constant({1}, &ax_last, DType::INT32));
        max_scores = g.repeat(max_scores, seq_len, 2);
        LogicalId shifted_scores = g.add(scores, g.neg(max_scores));

        float e_val = TGConstants::E;
        LogicalId e_node = g.fill(g.constant({1}, &e_val, DType::FLOAT32),
                                  {cfg.num_attention_heads, seq_len, seq_len});
        LogicalId exp_scores = g.pow(e_node, shifted_scores);

        LogicalId sum_exp = g.sum(exp_scores, g.constant({1}, &ax_last, DType::INT32));
        sum_exp = g.repeat(sum_exp, seq_len, 2);

        LogicalId probs = g.div(exp_scores, sum_exp);
        LogicalId context = g.dot(probs, v_3d);

        int32_t sh4_ctx[] = {1, (int32_t)cfg.num_attention_heads, (int32_t)seq_len, (int32_t)cfg.v_head_dim};
        LogicalId ctx_4d = g.reshape(context, g.constant({4}, sh4_ctx, DType::INT32));
        LogicalId ctx_perm = g.contiguous(g.permute(ctx_4d, p4_node));

        int32_t sh3_out[] = {1, (int32_t)seq_len, (int32_t)(cfg.num_attention_heads * cfg.v_head_dim)};
        LogicalId ctx_flat = g.reshape(ctx_perm, g.constant({3}, sh3_out, DType::INT32));

        return linear(ctx_flat, prefix + "self_attn.o_proj.weight",
                      cfg.num_attention_heads * cfg.v_head_dim, cfg.hidden_size);
    }

    LogicalId moeRoutedAtomic(LogicalId x_ffn, const std::string &prefix)
    {
        LogicalId router_logits = linear(x_ffn, prefix + "mlp.gate.weight", cfg.hidden_size, cfg.n_routed_experts);
        LogicalId scores = sigmoid(router_logits, cfg.n_routed_experts);

        LogicalId e_bias = weight(prefix + "mlp.gate.e_score_correction_bias");
        int32_t sh_bias[] = {1, 1, (int32_t)cfg.n_routed_experts};
        LogicalId bias_3d = g.reshape(e_bias, g.constant({3}, sh_bias, DType::INT32));
        LogicalId bias_exp = g.repeat(bias_3d, seq_len, 1);
        LogicalId scores_for_choice = g.add(scores, bias_exp);

        int32_t ax_last = -1;
        int32_t topk_exp = cfg.num_experts_per_tok;
        LogicalId top_k_idxs = g.argmax(scores_for_choice, g.constant({1}, &ax_last, DType::INT32),
                                        g.constant({1}, &topk_exp, DType::INT32));

        // Gather top-k weights
        int32_t sh_flat[] = {(int32_t)(seq_len * cfg.n_routed_experts)};
        LogicalId probs_flat = g.reshape(scores, g.constant({1}, sh_flat, DType::INT32));

        int32_t start_val = 0, stop_val = (int32_t)seq_len, step_val = 1;
        LogicalId row_idx = g.arange(g.constant({1}, &start_val, DType::INT32),
                                     g.constant({1}, &stop_val, DType::INT32),
                                     g.constant({1}, &step_val, DType::INT32));
        int32_t n_exp_i = cfg.n_routed_experts;
        int32_t seq_len_sh[] = {(int32_t)seq_len};
        LogicalId n_exp_i_node = g.fill(g.constant({1}, &n_exp_i, DType::INT32),
                                        g.constant({1}, seq_len_sh, DType::INT32));
        LogicalId row_offset = g.mul(row_idx, n_exp_i_node);

        int32_t sh_ro[] = {1, (int32_t)seq_len, 1};
        LogicalId row_offset_3d = g.reshape(row_offset, g.constant({3}, sh_ro, DType::INT32));

        int32_t rep_k_ro[] = {topk_exp};
        int32_t ax_2_ro[] = {2};
        LogicalId row_offset_rep = g.repeat(row_offset_3d, g.constant({1}, rep_k_ro, DType::INT32),
                                            g.constant({1}, ax_2_ro, DType::INT32));

        LogicalId flat_idxs = g.add(row_offset_rep, top_k_idxs);
        LogicalId topk_weights = g.gather(probs_flat, flat_idxs);

        int32_t ax_2_val = 2;
        LogicalId sum_weights = g.sum(topk_weights, g.constant({1}, &ax_2_val, DType::INT32));
        LogicalId sum_weights_3d = g.reshape(sum_weights, g.constant({3}, sh_ro, DType::INT32));
        LogicalId sum_weights_rep = g.repeat(sum_weights_3d, g.constant({1}, rep_k_ro, DType::INT32),
                                             g.constant({1}, ax_2_ro, DType::INT32));

        LogicalId norm_weights = g.div(topk_weights, sum_weights_rep);
        int32_t sh_topk[] = {1, (int32_t)seq_len, (int32_t)topk_exp};
        LogicalId route_scale_node =
            g.fill(g.constant({1}, &cfg.routed_scaling_factor, DType::FLOAT32), g.constant({3}, sh_topk, DType::INT32));
        norm_weights = g.mul(norm_weights, route_scale_node);

        // Fetch and stack expert weights
        std::vector<LogicalId> w1_list, w2_list, w3_list;
        int32_t p[] = {1, 0};
        LogicalId p_node = g.constant({2}, p, DType::INT32);
        int32_t sh_1_dim_inter[] = {1, (int32_t)cfg.hidden_size, (int32_t)cfg.moe_intermediate_size};
        LogicalId sh_1_dim_inter_node = g.constant({3}, sh_1_dim_inter, DType::INT32);
        int32_t sh_1_inter_dim[] = {1, (int32_t)cfg.moe_intermediate_size, (int32_t)cfg.hidden_size};
        LogicalId sh_1_inter_dim_node = g.constant({3}, sh_1_inter_dim, DType::INT32);

        for (uint32_t e = 0; e < cfg.n_routed_experts; ++e)
        {
            std::string e_str = std::to_string(e);
            LogicalId w1 = weight(prefix + "mlp.experts." + e_str + ".gate_proj.weight", cfg.hidden_size, cfg.moe_intermediate_size);
            LogicalId w1_t = g.contiguous(g.permute(w1, p_node));
            w1_list.push_back(g.reshape(w1_t, sh_1_dim_inter_node));

            LogicalId w2 = weight(prefix + "mlp.experts." + e_str + ".down_proj.weight", cfg.moe_intermediate_size, cfg.hidden_size);
            LogicalId w2_t = g.contiguous(g.permute(w2, p_node));
            w2_list.push_back(g.reshape(w2_t, sh_1_inter_dim_node));

            LogicalId w3 = weight(prefix + "mlp.experts." + e_str + ".up_proj.weight", cfg.hidden_size, cfg.moe_intermediate_size);
            LogicalId w3_t = g.contiguous(g.permute(w3, p_node));
            w3_list.push_back(g.reshape(w3_t, sh_1_dim_inter_node));
        }

        int32_t ax_0_val = 0;
        LogicalId ax_0_node = g.constant({1}, &ax_0_val, DType::INT32);
        LogicalId W1_stacked = g.concat(w1_list, ax_0_node);
        LogicalId W2_stacked = g.concat(w2_list, ax_0_node);
        LogicalId W3_stacked = g.concat(w3_list, ax_0_node);

        // Gather active weights
        LogicalId W1_active = g.gather(W1_stacked, top_k_idxs);
        LogicalId W2_active = g.gather(W2_stacked, top_k_idxs);
        LogicalId W3_active = g.gather(W3_stacked, top_k_idxs);

        int32_t sh_active_w13[] = {1, (int32_t)(seq_len * topk_exp), (int32_t)cfg.hidden_size, (int32_t)cfg.moe_intermediate_size};
        int32_t sh_active_w2[] = {1, (int32_t)(seq_len * topk_exp), (int32_t)cfg.moe_intermediate_size, (int32_t)cfg.hidden_size};

        LogicalId W1_active_r = g.reshape(W1_active, g.constant({4}, sh_active_w13, DType::INT32));
        LogicalId W2_active_r = g.reshape(W2_active, g.constant({4}, sh_active_w2, DType::INT32));
        LogicalId W3_active_r = g.reshape(W3_active, g.constant({4}, sh_active_w13, DType::INT32));

        // Prepare x_ffn
        int32_t sh_x_1[] = {1, (int32_t)seq_len, 1, (int32_t)cfg.hidden_size};
        LogicalId x_ffn_1 = g.reshape(x_ffn, g.constant({4}, sh_x_1, DType::INT32));

        LogicalId x_ffn_rep =
            g.repeat(x_ffn_1, g.constant({1}, rep_k_ro, DType::INT32), g.constant({1}, ax_2_ro, DType::INT32));

        int32_t sh_x_2[] = {1, (int32_t)(seq_len * topk_exp), 1, (int32_t)cfg.hidden_size};
        LogicalId x_ffn_ready = g.reshape(x_ffn_rep, g.constant({4}, sh_x_2, DType::INT32));

        // Apply experts
        LogicalId w1_out = g.dot(x_ffn_ready, W1_active_r);
        LogicalId w3_out = g.dot(x_ffn_ready, W3_active_r);

        LogicalId gate_silu = silu(w1_out, {1, static_cast<uint32_t>(seq_len * topk_exp), 1, cfg.moe_intermediate_size});
        LogicalId gate_up = g.mul(gate_silu, w3_out);

        LogicalId expert_out = g.dot(gate_up, W2_active_r);

        // Apply weights
        int32_t sh_nw_1[] = {1, (int32_t)(seq_len * topk_exp), 1, 1};
        LogicalId norm_weights_reshaped = g.reshape(norm_weights, g.constant({4}, sh_nw_1, DType::INT32));

        int32_t rep_dim[] = {(int32_t)cfg.hidden_size};
        int32_t ax_3[] = {3};
        LogicalId norm_weights_exp = g.repeat(norm_weights_reshaped, g.constant({1}, rep_dim, DType::INT32),
                                             g.constant({1}, ax_3, DType::INT32));

        LogicalId weighted_experts = g.mul(expert_out, norm_weights_exp);

        int32_t sh_we[] = {1, (int32_t)seq_len, (int32_t)topk_exp, (int32_t)cfg.hidden_size};
        LogicalId weighted_experts_4d = g.reshape(weighted_experts, g.constant({4}, sh_we, DType::INT32));

        LogicalId routed_out = g.sum(weighted_experts_4d, g.constant({1}, &ax_2_val, DType::INT32));
        int32_t sh_final[] = {1, (int32_t)seq_len, (int32_t)cfg.hidden_size};
        return g.reshape(routed_out, g.constant({3}, sh_final, DType::INT32));
    }

    LogicalId buildGraph(LogicalId input_ids_id)
    {
        LogicalId w_emb = weight("model.embed_tokens.weight", cfg.hidden_size, cfg.vocab_size);
        LogicalId h = g.gather(w_emb, input_ids_id);

        LogicalId mask_3d = createCausalMask();
        LogicalId prev_topk_indices;

        for (uint32_t i = 0; i < cfg.num_hidden_layers; ++i)
        {
            std::string prefix = "model.layers." + std::to_string(i) + ".";
            LogicalId residual = h;

            LogicalId x_norm = rmsNorm(h, prefix + "input_layernorm.weight", cfg.hidden_size);

            LogicalId q_a = linear(x_norm, prefix + "self_attn.q_a_proj.weight", cfg.hidden_size, cfg.q_lora_rank);
            LogicalId q_resid = rmsNorm(q_a, prefix + "self_attn.q_a_layernorm.weight", cfg.q_lora_rank);

            if (cfg.isIndexerFull(i))
            {
                prev_topk_indices = indexerAtomic(x_norm, q_resid, prefix, mask_3d);
            }

            LogicalId attn_out = attentionAtomic(x_norm, prefix, mask_3d, q_resid);
            h = g.add(residual, attn_out);

            residual = h;
            LogicalId x_ffn = rmsNorm(h, prefix + "post_attention_layernorm.weight", cfg.hidden_size);

            if (cfg.isMlpDense(i))
            {
                LogicalId gate = linear(x_ffn, prefix + "mlp.gate_proj.weight", cfg.hidden_size, cfg.intermediate_size);
                LogicalId up = linear(x_ffn, prefix + "mlp.up_proj.weight", cfg.hidden_size, cfg.intermediate_size);
                LogicalId act = g.mul(silu(gate, {1, seq_len, cfg.intermediate_size}), up);
                LogicalId mlp_out = linear(act, prefix + "mlp.down_proj.weight", cfg.intermediate_size, cfg.hidden_size);
                h = g.add(residual, mlp_out);
            }
            else
            {
                LogicalId s_gate = linear(x_ffn, prefix + "mlp.shared_experts.gate_proj.weight", cfg.hidden_size,
                                          cfg.moe_intermediate_size);
                LogicalId s_up = linear(x_ffn, prefix + "mlp.shared_experts.up_proj.weight", cfg.hidden_size,
                                        cfg.moe_intermediate_size);
                LogicalId s_act = g.mul(silu(s_gate, {1, seq_len, cfg.moe_intermediate_size}), s_up);
                LogicalId shared_out = linear(s_act, prefix + "mlp.shared_experts.down_proj.weight",
                                              cfg.moe_intermediate_size, cfg.hidden_size);

                LogicalId routed_out = moeRoutedAtomic(x_ffn, prefix);
                LogicalId total_moe = g.add(routed_out, shared_out);
                h = g.add(residual, total_moe);
            }
        }

        LogicalId norm_out = rmsNorm(h, "model.norm.weight", cfg.hidden_size);
        return linear(norm_out, "lm_head.weight", cfg.hidden_size, cfg.vocab_size);
    }

    LogicalId build_graph(LogicalId input_ids_id)
    {
        return buildGraph(input_ids_id);
    }
};

using GLM5_3Model = GLM5_3_Model;
