#pragma once
#include <cmath>
#include <string>
#include <vector>

#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/types.hpp"
#include "kernels/fused/gemma_rms_norm/ref.hpp"
#include "kernels/fused/gemma_rope/ref.hpp"
#include "kernels/fused/gemma_attention/ref.hpp"
#include "kernels/fused/gelu_tanh_fill/ref.hpp"

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
    Gemma3Model(Gemma3ModelConfig config, uint32_t sequence_length, Graph &graph, MemoryManager &memory,
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

    LogicalId rms_norm_gemma_atomic(LogicalId x_id, LogicalId weight_id, uint32_t dim0, uint32_t dim_size)
    {
        return refFactoryGemmaRMSNormEx(x_id, weight_id, dim0, seq_len, dim_size, g);
    }

    LogicalId tanh_atomic(LogicalId x_id, uint32_t last_dim)
    {
        float neg_two_val = -2.0f;
        LogicalId neg_two = g.fill(g.constant({1}, &neg_two_val, DType::FLOAT32), {1, seq_len, last_dim});
        float two_val = 2.0f;
        LogicalId two = g.fill(g.constant({1}, &two_val, DType::FLOAT32), {1, seq_len, last_dim});
        float e_val = 2.718281828459045f;
        LogicalId e_node = g.fill(g.constant({1}, &e_val, DType::FLOAT32), {1, seq_len, last_dim});
        LogicalId one_node = g.fill(one_fp32, {1, seq_len, last_dim});
        LogicalId neg_2x = g.mul(x_id, neg_two);
        LogicalId exp_neg_2x = g.pow(e_node, neg_2x);
        LogicalId den = g.add(one_node, exp_neg_2x);
        LogicalId quotient = g.div(two, den);
        LogicalId neg_one = g.neg(one_node);
        return g.add(quotient, neg_one);
    }

    LogicalId gelu_atomic(LogicalId x_id, uint32_t last_dim)
    {
        return refFactoryGeluTanhFillEx(x_id, {1, seq_len, last_dim}, g);
    }

    std::tuple<LogicalId, LogicalId> compute_rope()
    {
        std::vector<float> cos_data(seq_len * cfg.head_dim);
        std::vector<float> sin_data(seq_len * cfg.head_dim);
        uint32_t half_dim = cfg.head_dim / 2;

        for (uint32_t p = 0; p < seq_len; ++p)
        {
            for (uint32_t i = 0; i < half_dim; ++i)
            {
                float exponent = (2.0f * static_cast<float>(i)) / static_cast<float>(cfg.head_dim);
                float inv_freq = 1.0f / std::pow(10000.0f, exponent);
                float angle = static_cast<float>(p) * inv_freq;
                float c = std::cos(angle);
                float s = std::sin(angle);
                cos_data[p * cfg.head_dim + i] = c;
                cos_data[p * cfg.head_dim + i + half_dim] = c;
                sin_data[p * cfg.head_dim + i] = s;
                sin_data[p * cfg.head_dim + i + half_dim] = s;
            }
        }

        LogicalId cos_out = g.constant({1, seq_len, cfg.head_dim}, cos_data.data(), DType::FLOAT32);
        LogicalId sin_out = g.constant({1, seq_len, cfg.head_dim}, sin_data.data(), DType::FLOAT32);
        return {cos_out, sin_out};
    }

    LogicalId apply_rope(LogicalId x_id, LogicalId cos_id, LogicalId sin_id, uint32_t n_groups)
    {
        return refFactoryGemmaRoPEEx(x_id, cos_id, sin_id, n_groups, seq_len, cfg.head_dim, g);
    }

    LogicalId compute_causal_mask()
    {
        std::vector<float> mask_data(seq_len * seq_len, 0.0f);
        for (uint32_t r = 0; r < seq_len; ++r)
        {
            for (uint32_t c = 0; c < seq_len; ++c)
            {
                if (c > r)
                    mask_data[r * seq_len + c] = -1e9f;
            }
        }
        return g.constant({1, seq_len, seq_len}, mask_data.data(), DType::FLOAT32);
    }

    std::tuple<LogicalId, LogicalId, LogicalId> attention_qkv_atomic(LogicalId x, const std::string &prefix,
                                                                     LogicalId rope_cos, LogicalId rope_sin)
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
            k = g.contiguous(g.repeat(k, rep_node, ax_node));
            v = g.contiguous(g.repeat(v, rep_node, ax_node));
        }
        return std::make_tuple(g.contiguous(q), k, v);
    }

    LogicalId attention_output_atomic(std::tuple<LogicalId, LogicalId, LogicalId> qkv, const std::string &prefix,
                                      LogicalId mask_id)
    {
        LogicalId q = std::get<0>(qkv);
        LogicalId k = std::get<1>(qkv);
        LogicalId v = std::get<2>(qkv);

        LogicalId ctx_flat = refFactoryGemmaAttentionEx(q, k, v, mask_id, cfg.n_heads, seq_len, cfg.head_dim, g);

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

        auto project = [&](const std::string &suffix, uint32_t in_d, uint32_t out_d) {
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
        std::vector<float> scale_data(seq_len * cfg.emb_dim, scale_val);
        LogicalId scale_node = g.constant({1, seq_len, cfg.emb_dim}, scale_data.data(), DType::FLOAT32);
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