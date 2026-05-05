// File: tensor_graphs_cpp/models/flux-klein-4b.hpp
#pragma once
#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include <string>
#include <cmath>

struct FluxKlein4BModelConfig
{
    uint32_t hidden_size = 3072;
    uint32_t num_heads = 24;
    uint32_t head_dim = 128;
    uint32_t image_h = 256;
    uint32_t image_w = 256;
    uint32_t depth = 19;        // Double stream blocks
    uint32_t single_depth = 38; // Single stream blocks
    float theta = 10000.0f;
};

class FluxKlein4BModel
{
private:
    FluxKlein4BModelConfig cfg;
    Graph &g;
    MemoryManager &mem;
    const std::string w_path;

    uint32_t weight(const std::string &prefix)
    {
        return g.cast(g.weight(w_path, prefix), DType::FLOAT32);
    }

    uint32_t repeat_ax(uint32_t id, uint32_t repeats, uint32_t axis)
    {
        if (repeats <= 1)
            return id;
        int32_t r = repeats, a = axis;
        return g.repeat(id, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }

    uint32_t expand_scalar(uint32_t s_id, const std::vector<uint32_t> &dims)
    {
        int32_t rank = dims.size();
        std::vector<int32_t> ones(rank, 1);
        uint32_t out = g.reshape(s_id, g.constant({(uint32_t)rank}, ones.data(), DType::INT32));
        for (size_t i = 0; i < dims.size(); i++)
        {
            out = repeat_ax(out, dims[i], i);
        }
        return out;
    }

    uint32_t silu(uint32_t x, const std::vector<uint32_t> &target_shape)
    {
        float neg_one = -1.0f, one = 1.0f, e = 2.7182818f;
        uint32_t neg_x = g.mul(x, expand_scalar(g.constant({1}, &neg_one, DType::FLOAT32), target_shape));
        uint32_t exp_n = g.pow(expand_scalar(g.constant({1}, &e, DType::FLOAT32), target_shape), neg_x);
        uint32_t den = g.add(expand_scalar(g.constant({1}, &one, DType::FLOAT32), target_shape), exp_n);
        uint32_t sig = g.div(expand_scalar(g.constant({1}, &one, DType::FLOAT32), target_shape), den);
        return g.mul(x, sig);
    }

    uint32_t rms_norm(uint32_t x, uint32_t w, float eps_val, uint32_t dim_size, const std::vector<uint32_t> &expand_dims)
    {
        uint32_t sq = g.mul(x, x);
        int32_t ax = -1;
        uint32_t s_sq = g.sum(sq, g.constant({1}, &ax, DType::INT32));
        float d_val = (float)dim_size;
        uint32_t m_sq = g.div(s_sq, expand_scalar(g.constant({1}, &d_val, DType::FLOAT32), expand_dims));
        uint32_t eps = expand_scalar(g.constant({1}, &eps_val, DType::FLOAT32), expand_dims);
        uint32_t var = g.add(m_sq, eps);
        float half = 0.5f, one = 1.0f;
        uint32_t std = g.pow(var, expand_scalar(g.constant({1}, &half, DType::FLOAT32), expand_dims));
        uint32_t inv_std = g.div(expand_scalar(g.constant({1}, &one, DType::FLOAT32), expand_dims), std);
        uint32_t x_norm = g.mul(x, repeat_ax(inv_std, dim_size, expand_dims.size() - 1));

        int32_t w_shape[] = {1, 1, (int32_t)dim_size};
        uint32_t w_3d = g.reshape(w, g.constant({3}, w_shape, DType::INT32));
        w_3d = repeat_ax(repeat_ax(w_3d, expand_dims[0], 0), expand_dims[1], 1);

        return g.mul(x_norm, g.add(w_3d, expand_scalar(g.constant({1}, &one, DType::FLOAT32), expand_dims)));
    }

    std::tuple<uint32_t, uint32_t> compute_rope()
    {
        // Build 2D Positional Embeddings internally
        uint32_t seq_len = (cfg.image_h * cfg.image_w) / 4;

        int32_t s = 0, end = cfg.head_dim / 2, step = 2;
        uint32_t idx = g.arange(g.constant({1}, &s, DType::INT32), g.constant({1}, &end, DType::INT32), g.constant({1}, &step, DType::INT32));
        uint32_t f_idx = g.cast(idx, DType::FLOAT32);

        float h_dim = (float)cfg.head_dim / 2;
        uint32_t f_dim = repeat_ax(g.constant({1}, &h_dim, DType::FLOAT32), cfg.head_dim / 4, 0);
        uint32_t exp = g.div(f_idx, f_dim);
        uint32_t theta = repeat_ax(g.constant({1}, &cfg.theta, DType::FLOAT32), cfg.head_dim / 4, 0);
        uint32_t base = g.pow(theta, exp);

        float one = 1.0f;
        uint32_t inv_freq = g.div(repeat_ax(g.constant({1}, &one, DType::FLOAT32), cfg.head_dim / 4, 0), base);

        int32_t p_end = seq_len;
        uint32_t pos = g.cast(g.arange(g.constant({1}, &s, DType::INT32), g.constant({1}, &p_end, DType::INT32), g.constant({1}, &s, DType::INT32)), DType::FLOAT32);

        int32_t p_s[] = {(int32_t)seq_len, 1};
        uint32_t p_col = g.reshape(pos, g.constant({2}, p_s, DType::INT32));
        p_col = repeat_ax(p_col, cfg.head_dim / 4, 1);

        int32_t i_s[] = {1, (int32_t)cfg.head_dim / 4};
        uint32_t i_row = repeat_ax(g.reshape(inv_freq, g.constant({2}, i_s, DType::INT32)), seq_len, 0);

        uint32_t freqs = g.mul(p_col, i_row);
        int32_t ax = 1;
        uint32_t freqs_cat = g.concat({freqs, freqs}, g.constant({1}, &ax, DType::INT32));

        int32_t f_shape[] = {1, (int32_t)seq_len, (int32_t)cfg.head_dim / 2};
        uint32_t f_sh = g.constant({3}, f_shape, DType::INT32);

        return {g.reshape(g.cos(freqs_cat), f_sh), g.reshape(g.sin(freqs_cat), f_sh)};
    }

    uint32_t linear(uint32_t x, const std::string &prefix, uint32_t in_d, uint32_t out_d)
    {
        uint32_t w = weight(prefix + ".weight");
        int32_t p[] = {1, 0};
        uint32_t w_t = g.contiguous(g.permute(w, g.constant({2}, p, DType::INT32)));
        int32_t s3[] = {1, (int32_t)in_d, (int32_t)out_d};
        return g.dot(x, g.reshape(w_t, g.constant({3}, s3, DType::INT32)));
    }

public:
    FluxKlein4BModel(FluxKlein4BModelConfig config, Graph &graph, MemoryManager &memory, const std::string &weight_path)
        : cfg(config), g(graph), mem(memory), w_path(weight_path) {}

    uint32_t build_graph(uint32_t img_in, uint32_t vec_in)
    {
        uint32_t seq_len = (cfg.image_h * cfg.image_w) / 4;

        // Input projections
        uint32_t x = linear(img_in, "img_in", cfg.hidden_size, cfg.hidden_size);
        uint32_t v = silu(vec_in, {1, cfg.hidden_size});
        v = linear(v, "time_in", cfg.hidden_size, cfg.hidden_size);

        auto rope = compute_rope();

        // Pass through double blocks
        for (uint32_t i = 0; i < cfg.depth; ++i)
        {
            std::string prefix = "double_blocks." + std::to_string(i);

            // Simplified modulation (Normally ADA-LN)
            uint32_t res = x;
            x = rms_norm(x, weight(prefix + ".img_mod.lin.weight"), 1e-6f, cfg.hidden_size, {1, seq_len, 1});

            // Dummy QKV for demonstration
            uint32_t q = linear(x, prefix + ".img_attn.qkv", cfg.hidden_size, cfg.hidden_size);

            // Minimal Attention + MLP fallback mapping
            uint32_t attn_out = linear(q, prefix + ".img_attn.proj", cfg.hidden_size, cfg.hidden_size);
            x = g.add(res, attn_out);

            uint32_t res_mlp = x;
            x = rms_norm(x, weight(prefix + ".img_mod.lin.weight"), 1e-6f, cfg.hidden_size, {1, seq_len, 1});
            uint32_t mlp_out = linear(x, prefix + ".img_mlp.0", cfg.hidden_size, cfg.hidden_size * 4);
            mlp_out = silu(mlp_out, {1, seq_len, cfg.hidden_size * 4});
            mlp_out = linear(mlp_out, prefix + ".img_mlp.2", cfg.hidden_size * 4, cfg.hidden_size);
            x = g.add(res_mlp, mlp_out);
        }

        // Pass through single blocks
        for (uint32_t i = 0; i < cfg.single_depth; ++i)
        {
            std::string prefix = "single_blocks." + std::to_string(i);
            uint32_t res = x;
            x = rms_norm(x, weight(prefix + ".norm.weight"), 1e-6f, cfg.hidden_size, {1, seq_len, 1});
            uint32_t qkv = linear(x, prefix + ".linear1", cfg.hidden_size, cfg.hidden_size * 4);
            x = g.add(res, linear(qkv, prefix + ".linear2", cfg.hidden_size * 4, cfg.hidden_size));
        }

        x = rms_norm(x, weight("final_layer.norm_out.weight"), 1e-6f, cfg.hidden_size, {1, seq_len, 1});
        x = linear(x, "final_layer.linear", cfg.hidden_size, cfg.hidden_size);

        return x;
    }
};