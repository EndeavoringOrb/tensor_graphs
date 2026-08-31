#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/common/constants.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/types.hpp"

struct Qwen3VLConfig
{
    uint32_t vocab_size = 151936;
    uint32_t hidden_size = 2560;
    uint32_t intermediate_size = 9728;
    uint32_t num_hidden_layers = 36;
    uint32_t num_attention_heads = 32;
    uint32_t num_key_value_heads = 8;
    uint32_t head_dim = 128;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 5000000.0f;

    std::vector<uint32_t> select_layers = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35};
};

class Qwen3VLModel
{
  private:
    Qwen3VLConfig cfg;
    uint32_t seq_len;
    Graph &g;
    MemoryManager &mem;
    const std::string w_path;

    std::string resolve_weight_name(const std::string &name)
    {
        std::vector<std::string> candidate_prefixes = {
            "", "model.", "language_model.", "text_encoder.model.", "model.language_model.", "text_model."};
        for (const auto &prefix : candidate_prefixes)
        {
            std::string full_name = prefix + name;
            if (FileRegistry::get().hasTensor(w_path, full_name))
            {
                return full_name;
            }
        }
        return name;
    }

    LogicalId weight(const std::string &name)
    {
        std::string resolved = resolve_weight_name(name);
        TensorMetadata meta = FileRegistry::get().getMetadata(w_path, resolved);
        LogicalId raw_weight = g.weight(w_path, resolved);
        LogicalId weight_f32 = g.cast(raw_weight, DType::FLOAT32);

        std::string scale_name = resolved + "_scale";
        if (FileRegistry::get().hasTensor(w_path, scale_name))
        {
            LogicalId raw_scale = g.weight(w_path, scale_name);
            LogicalId scale_f32 = g.cast(raw_scale, DType::FLOAT32);
            LogicalId scale_expanded = g.fill(scale_f32, meta.shape);
            weight_f32 = g.mul(weight_f32, scale_expanded);
        }

        return weight_f32;
    }

    LogicalId linear(LogicalId x, const std::string &w_name, const std::string &b_name, uint32_t in_d, uint32_t out_d,
                     uint32_t S)
    {
        LogicalId w = weight(w_name);
        LogicalId w_t = g.contiguous(g.permute(w, {1, 0}));
        LogicalId out = g.dot(x, g.reshape(w_t, {1, (int32_t)in_d, (int32_t)out_d}));
        if (!b_name.empty())
        {
            std::string resolved_b = resolve_weight_name(b_name);
            if (FileRegistry::get().hasTensor(w_path, resolved_b))
            {
                LogicalId b = weight(b_name);
                LogicalId b_exp = g.repeat(g.reshape(b, {1, 1, (int32_t)out_d}), S, 1);
                out = g.add(out, b_exp);
            }
        }
        return out;
    }

    LogicalId rms_norm(LogicalId x, const std::string &w_name, uint32_t S, uint32_t D, float eps = 1e-6f)
    {
        LogicalId x_sq = g.mul(x, x);
        LogicalId sum_sq = g.sum(x_sq, -1);
        LogicalId mean_sq = g.div(sum_sq, g.fill((float)D, {1, S, 1}));
        LogicalId std = g.pow(g.add(mean_sq, g.fill(eps, {1, S, 1})), g.fill(0.5f, {1, S, 1}));
        LogicalId inv_std = g.repeat(g.div(g.fill(1.0f, {1, S, 1}), std), D, 2);
        LogicalId x_norm = g.mul(x, inv_std);

        if (!w_name.empty())
        {
            LogicalId w = weight(w_name);
            LogicalId w_exp = g.repeat(g.reshape(w, {1, 1, (int32_t)D}), S, 1);
            x_norm = g.mul(x_norm, w_exp);
        }
        return x_norm;
    }

    LogicalId per_head_rms_norm(LogicalId x, const std::string &w_name, uint32_t num_heads, uint32_t S,
                                uint32_t head_dim, float eps = 1e-6f)
    {
        LogicalId x_sq = g.mul(x, x);
        LogicalId sum_sq = g.sum(x_sq, -1);
        LogicalId mean_sq = g.div(sum_sq, g.fill((float)head_dim, {1, num_heads, S, 1}));
        LogicalId std = g.pow(g.add(mean_sq, g.fill(eps, {1, num_heads, S, 1})), g.fill(0.5f, {1, num_heads, S, 1}));
        LogicalId inv_std = g.repeat(g.div(g.fill(1.0f, {1, num_heads, S, 1}), std), head_dim, 3);
        LogicalId x_norm = g.mul(x, inv_std);

        if (!w_name.empty())
        {
            LogicalId w = weight(w_name);
            LogicalId w_4d = g.reshape(w, {1, 1, 1, (int32_t)head_dim});
            LogicalId w_exp = g.repeat(g.repeat(w_4d, num_heads, 1), S, 2);
            x_norm = g.mul(x_norm, w_exp);
        }
        return x_norm;
    }

    LogicalId sigmoid(LogicalId x, const std::vector<uint32_t> &shape)
    {
        LogicalId neg_x = g.mul(x, g.fill(-1.0f, shape));
        LogicalId exp_neg_x = g.pow(g.fill(TGConstants::E, shape), neg_x);
        LogicalId one = g.fill(1.0f, shape);
        return g.div(one, g.add(one, exp_neg_x));
    }

    LogicalId silu(LogicalId x, const std::vector<uint32_t> &shape)
    {
        return g.mul(x, sigmoid(x, shape));
    }

    std::tuple<LogicalId, LogicalId> compute_rope_1d(uint32_t S, uint32_t head_dim, float theta)
    {
        std::vector<float> freqs_cos(S * head_dim);
        std::vector<float> freqs_sin(S * head_dim);

        uint32_t half_dim = head_dim / 2;
        for (uint32_t s = 0; s < S; ++s)
        {
            for (uint32_t d = 0; d < half_dim; ++d)
            {
                float freq = 1.0f / std::pow(theta, static_cast<float>(2 * d) / static_cast<float>(head_dim));
                float val = static_cast<float>(s) * freq;
                float cos_val = std::cos(val);
                float sin_val = std::sin(val);
                freqs_cos[s * head_dim + d] = cos_val;
                freqs_cos[s * head_dim + d + half_dim] = cos_val;
                freqs_sin[s * head_dim + d] = sin_val;
                freqs_sin[s * head_dim + d + half_dim] = sin_val;
            }
        }

        LogicalId cos_node = g.constant({1, 1, S, head_dim}, freqs_cos.data(), DType::FLOAT32);
        LogicalId sin_node = g.constant({1, 1, S, head_dim}, freqs_sin.data(), DType::FLOAT32);
        return {cos_node, sin_node};
    }

    LogicalId apply_rope(LogicalId x, LogicalId cos_node, LogicalId sin_node, uint32_t num_heads, uint32_t S,
                         uint32_t head_dim)
    {
        int32_t half_dim = static_cast<int32_t>(head_dim / 2);
        LogicalId x1 = g.slice(x, {0, 0, 0, 0}, {1, (int32_t)num_heads, (int32_t)S, half_dim});
        LogicalId x2 = g.slice(x, {0, 0, 0, half_dim}, {1, (int32_t)num_heads, (int32_t)S, (int32_t)head_dim});

        LogicalId rotated = g.concat({g.neg(x2), x1}, 3);
        LogicalId cos_exp = g.repeat(cos_node, num_heads, 1);
        LogicalId sin_exp = g.repeat(sin_node, num_heads, 1);
        return g.add(g.mul(x, cos_exp), g.mul(rotated, sin_exp));
    }

    LogicalId compute_causal_mask(uint32_t S)
    {
        float one_val = 1.0f;
        LogicalId ones_matrix = g.fill(one_val, {S, S});
        int32_t k_val = 1;
        LogicalId triu_mask = g.triu(ones_matrix, k_val);
        float neg_inf_val = -1e9f;
        LogicalId neg_inf_node = g.fill(neg_inf_val, {1, 1, S, S});
        LogicalId triu_4d = g.reshape(triu_mask, {1, 1, (int32_t)S, (int32_t)S});
        return g.mul(triu_4d, neg_inf_node);
    }

    LogicalId softmax_4d(LogicalId scores, uint32_t S, uint32_t num_heads)
    {
        LogicalId max_s = g.repeat(g.max(scores, -1), S, 3);
        LogicalId shifted = g.add(scores, g.neg(max_s));
        LogicalId exps = g.pow(g.fill(TGConstants::E, {1, num_heads, S, S}), shifted);
        LogicalId sums = g.repeat(g.sum(exps, -1), S, 3);
        return g.div(exps, sums);
    }

    LogicalId attention(LogicalId x, uint32_t layer_idx, LogicalId cos_node, LogicalId sin_node, LogicalId mask,
                        uint32_t S)
    {
        std::string prefix = "layers." + std::to_string(layer_idx) + ".self_attn.";

        uint32_t q_dim = cfg.num_attention_heads * cfg.head_dim;
        uint32_t kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        LogicalId q = linear(x, prefix + "q_proj.weight", "", cfg.hidden_size, q_dim, S);
        LogicalId k = linear(x, prefix + "k_proj.weight", "", cfg.hidden_size, kv_dim, S);
        LogicalId v = linear(x, prefix + "v_proj.weight", "", cfg.hidden_size, kv_dim, S);

        q = g.contiguous(g.permute(
            g.reshape(q, {1, (int32_t)S, (int32_t)cfg.num_attention_heads, (int32_t)cfg.head_dim}), {0, 2, 1, 3}));
        k = g.contiguous(g.permute(
            g.reshape(k, {1, (int32_t)S, (int32_t)cfg.num_key_value_heads, (int32_t)cfg.head_dim}), {0, 2, 1, 3}));
        v = g.contiguous(g.permute(
            g.reshape(v, {1, (int32_t)S, (int32_t)cfg.num_key_value_heads, (int32_t)cfg.head_dim}), {0, 2, 1, 3}));

        q = per_head_rms_norm(q, prefix + "q_norm.weight", cfg.num_attention_heads, S, cfg.head_dim, cfg.rms_norm_eps);
        k = per_head_rms_norm(k, prefix + "k_norm.weight", cfg.num_key_value_heads, S, cfg.head_dim, cfg.rms_norm_eps);

        q = apply_rope(q, cos_node, sin_node, cfg.num_attention_heads, S, cfg.head_dim);
        k = apply_rope(k, cos_node, sin_node, cfg.num_key_value_heads, S, cfg.head_dim);

        float scale_val = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
        q = g.mul(q, g.fill(scale_val, {1, cfg.num_attention_heads, S, cfg.head_dim}));

        uint32_t rep_factor = cfg.num_attention_heads / cfg.num_key_value_heads;
        LogicalId k_5d = g.repeat(
            g.reshape(k, {1, (int32_t)cfg.num_key_value_heads, 1, (int32_t)S, (int32_t)cfg.head_dim}), rep_factor, 2);
        LogicalId v_5d = g.repeat(
            g.reshape(v, {1, (int32_t)cfg.num_key_value_heads, 1, (int32_t)S, (int32_t)cfg.head_dim}), rep_factor, 2);
        k = g.reshape(g.contiguous(k_5d), {1, (int32_t)cfg.num_attention_heads, (int32_t)S, (int32_t)cfg.head_dim});
        v = g.reshape(g.contiguous(v_5d), {1, (int32_t)cfg.num_attention_heads, (int32_t)S, (int32_t)cfg.head_dim});

        LogicalId k_t = g.contiguous(g.permute(k, {0, 1, 3, 2}));
        LogicalId scores = g.dot(q, k_t);

        scores = g.add(scores, g.repeat(mask, cfg.num_attention_heads, 1));
        LogicalId probs = softmax_4d(scores, S, cfg.num_attention_heads);

        LogicalId attn_out = g.dot(probs, v);
        LogicalId ctx_perm = g.contiguous(g.permute(attn_out, {0, 2, 1, 3}));
        LogicalId ctx_flat = g.reshape(ctx_perm, {1, (int32_t)S, (int32_t)q_dim});

        return linear(ctx_flat, prefix + "o_proj.weight", "", q_dim, cfg.hidden_size, S);
    }

    LogicalId mlp(LogicalId x, uint32_t layer_idx, uint32_t S)
    {
        std::string prefix = "layers." + std::to_string(layer_idx) + ".mlp.";
        LogicalId gate = linear(x, prefix + "gate_proj.weight", "", cfg.hidden_size, cfg.intermediate_size, S);
        LogicalId up = linear(x, prefix + "up_proj.weight", "", cfg.hidden_size, cfg.intermediate_size, S);
        LogicalId gate_act = silu(gate, {1, S, cfg.intermediate_size});
        LogicalId swiglu = g.mul(gate_act, up);
        return linear(swiglu, prefix + "down_proj.weight", "", cfg.intermediate_size, cfg.hidden_size, S);
    }

    LogicalId decoder_layer(LogicalId x, uint32_t layer_idx, LogicalId cos_node, LogicalId sin_node, LogicalId mask,
                            uint32_t S)
    {
        std::string prefix = "layers." + std::to_string(layer_idx) + ".";
        LogicalId residual = x;

        LogicalId norm1 = rms_norm(x, prefix + "input_layernorm.weight", S, cfg.hidden_size, cfg.rms_norm_eps);
        LogicalId attn_out = attention(norm1, layer_idx, cos_node, sin_node, mask, S);
        LogicalId h = g.add(residual, attn_out);
        residual = h;

        LogicalId norm2 = rms_norm(h, prefix + "post_attention_layernorm.weight", S, cfg.hidden_size, cfg.rms_norm_eps);
        LogicalId mlp_out = mlp(norm2, layer_idx, S);
        return g.add(residual, mlp_out);
    }

  public:
    Qwen3VLModel(Qwen3VLConfig config, uint32_t sequence_length, Graph &graph, MemoryManager &memory,
                 std::string weight_path)
        : cfg(config), seq_len(sequence_length), g(graph), mem(memory), w_path(std::move(weight_path))
    {
    }

    LogicalId build_graph(LogicalId input_ids_id)
    {
        LogicalId w_emb = weight("embed_tokens.weight");
        LogicalId h = g.gather(w_emb, input_ids_id);

        auto [cos_node, sin_node] = compute_rope_1d(seq_len, cfg.head_dim, cfg.rope_theta);
        LogicalId mask = compute_causal_mask(seq_len);

        std::unordered_set<uint32_t> select_set(cfg.select_layers.begin(), cfg.select_layers.end());
        std::vector<LogicalId> selected_hiddens;

        for (uint32_t i = 0; i < cfg.num_hidden_layers; ++i)
        {
            h = decoder_layer(h, i, cos_node, sin_node, mask, seq_len);
            if (select_set.count(i))
            {
                int32_t sh4[] = {1, (int32_t)seq_len, 1, (int32_t)cfg.hidden_size};
                LogicalId h_4d = g.reshape(h, g.constant({4}, sh4, DType::INT32));
                selected_hiddens.push_back(h_4d);
            }
        }

        if (selected_hiddens.empty())
        {
            Error::throw_err("[Qwen3VLModel] No layers selected for hidden states output!");
        }

        int32_t ax_layer = 2;
        return g.concat(selected_hiddens, g.constant({1}, &ax_layer, DType::INT32));
    }
};