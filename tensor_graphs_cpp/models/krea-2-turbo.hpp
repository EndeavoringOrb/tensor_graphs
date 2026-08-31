#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/constants.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/types.hpp"

struct Krea2TurboConfig
{
    uint32_t height = 1024;
    uint32_t width = 1024;
    uint32_t text_seq_len = 128;

    uint32_t latent_channels = 16;
    uint32_t vae_scale_factor = 8;
    uint32_t patch_size = 2;

    uint32_t hidden_size = 6144;
    uint32_t num_layers = 28;
    uint32_t num_heads = 48;
    uint32_t num_kv_heads = 12;
    uint32_t head_dim = 128;
    uint32_t mlp_hidden_dim = 16384;

    uint32_t text_dim = 2560;
    uint32_t text_num_layers = 12;
    uint32_t text_fusion_heads = 20;
    uint32_t text_fusion_head_dim = 128;
    uint32_t text_fusion_intermediate = 6912;
    uint32_t num_layerwise_blocks = 2;
    uint32_t num_refiner_blocks = 2;

    uint32_t time_dim = 256;
    uint32_t time_mlp_dim = 6144;

    float rope_theta = 1000.0f;
    float rms_eps = 1e-6f;
    float mu = 1.15f;
    uint32_t num_inference_steps = 8;

    // Derived geometry
    uint32_t latent_h = 128;
    uint32_t latent_w = 128;
    uint32_t grid_h = 64;
    uint32_t grid_w = 64;
    uint32_t num_patches = 4096;
    uint32_t patch_dim = 64;
    uint32_t total_seq_len = 4224;

    Krea2TurboConfig(uint32_t h = 1024, uint32_t w = 1024, uint32_t txt_len = 128)
        : height(h), width(w), text_seq_len(txt_len)
    {
        init();
    }

    void init()
    {
        latent_h = height / vae_scale_factor;
        latent_w = width / vae_scale_factor;
        grid_h = latent_h / patch_size;
        grid_w = latent_w / patch_size;
        num_patches = grid_h * grid_w;
        patch_dim = latent_channels * patch_size * patch_size;
        total_seq_len = text_seq_len + num_patches;
    }
};

class Krea2TurboModel
{
  private:
    Krea2TurboConfig cfg;
    Graph &g;
    MemoryManager &mem;
    const std::string w_path;

    LogicalId weight(const std::string &name, SourceLocation loc = SourceLocation::current())
    {
        LogicalId raw_weight = g.weight(w_path, name, loc);
        LogicalId cast_w = g.cast(raw_weight, DType::FLOAT32, loc);
        return cast_w;
    }

    LogicalId linear(LogicalId x, const std::string &w_name, const std::string &b_name, uint32_t in_d, uint32_t out_d,
                     uint32_t B, uint32_t S, SourceLocation loc = SourceLocation::current())
    {
        LogicalId w = weight(w_name, loc);
        LogicalId w_t = g.contiguous(g.permute(w, {1, 0}));
        LogicalId x_flat = g.reshape(x, {1, static_cast<int32_t>(B * S), static_cast<int32_t>(in_d)});
        LogicalId out_flat =
            g.dot(x_flat, g.reshape(w_t, {1, static_cast<int32_t>(in_d), static_cast<int32_t>(out_d)}));
        LogicalId out =
            g.reshape(out_flat, {static_cast<int32_t>(B), static_cast<int32_t>(S), static_cast<int32_t>(out_d)});

        if (!b_name.empty() && FileRegistry::get().hasTensor(w_path, b_name))
        {
            LogicalId b = weight(b_name, loc);
            LogicalId b_3d = g.reshape(b, {1, 1, static_cast<int32_t>(out_d)});
            LogicalId b_exp = g.repeat(g.repeat(b_3d, B, 0), S, 1);
            out = g.add(out, b_exp);
        }
        return out;
    }

    LogicalId linear(LogicalId x, const std::string &w_name, const std::string &b_name, uint32_t in_d, uint32_t out_d,
                     uint32_t S, SourceLocation loc = SourceLocation::current())
    {
        return linear(x, w_name, b_name, in_d, out_d, 1, S, loc);
    }

    LogicalId rms_norm(LogicalId x, const std::string &w_name, uint32_t B, uint32_t S, uint32_t D, float eps = 1e-6f)
    {
        LogicalId x_sq = g.mul(x, x);
        int32_t axis_val = -1;
        LogicalId sum_sq = g.sum(x_sq, g.constant({1}, &axis_val, DType::INT32));
        LogicalId mean_sq = g.div(sum_sq, g.fill(static_cast<float>(D), {B, S, 1}));
        LogicalId std = g.pow(g.add(mean_sq, g.fill(eps, {B, S, 1})), g.fill(0.5f, {B, S, 1}));
        LogicalId inv_std = g.repeat(g.div(g.fill(1.0f, {B, S, 1}), std), D, 2);
        LogicalId x_norm = g.mul(x, inv_std);

        if (!w_name.empty() && FileRegistry::get().hasTensor(w_path, w_name))
        {
            LogicalId w = weight(w_name);
            LogicalId w_3d = g.reshape(w, {1, 1, static_cast<int32_t>(D)});
            LogicalId w_exp = g.repeat(g.repeat(w_3d, B, 0), S, 1);
            LogicalId one_full = g.fill(1.0f, {B, S, D});
            LogicalId scale = g.add(w_exp, one_full);
            x_norm = g.mul(x_norm, scale);
        }
        return x_norm;
    }

    LogicalId rms_norm(LogicalId x, const std::string &w_name, uint32_t S, uint32_t D, float eps = 1e-6f)
    {
        return rms_norm(x, w_name, 1, S, D, eps);
    }

    LogicalId per_head_rms_norm(LogicalId x, const std::string &w_name, uint32_t B, uint32_t num_heads, uint32_t S,
                                uint32_t head_dim, float eps = 1e-6f)
    {
        LogicalId x_sq = g.mul(x, x);
        int32_t axis_val = -1;
        LogicalId sum_sq = g.sum(x_sq, g.constant({1}, &axis_val, DType::INT32));
        LogicalId mean_sq = g.div(sum_sq, g.fill(static_cast<float>(head_dim), {B, num_heads, S, 1}));
        LogicalId std = g.pow(g.add(mean_sq, g.fill(eps, {B, num_heads, S, 1})), g.fill(0.5f, {B, num_heads, S, 1}));
        LogicalId inv_std = g.repeat(g.div(g.fill(1.0f, {B, num_heads, S, 1}), std), head_dim, 3);
        LogicalId x_norm = g.mul(x, inv_std);

        if (!w_name.empty() && FileRegistry::get().hasTensor(w_path, w_name))
        {
            LogicalId w = weight(w_name);
            LogicalId w_4d = g.reshape(w, {1, 1, 1, static_cast<int32_t>(head_dim)});
            LogicalId w_exp = g.repeat(g.repeat(g.repeat(w_4d, B, 0), num_heads, 1), S, 2);
            LogicalId one_full = g.fill(1.0f, {B, num_heads, S, head_dim});
            LogicalId scale = g.add(w_exp, one_full);
            x_norm = g.mul(x_norm, scale);
        }
        return x_norm;
    }

    LogicalId per_head_rms_norm(LogicalId x, const std::string &w_name, uint32_t num_heads, uint32_t S,
                                uint32_t head_dim, float eps = 1e-6f)
    {
        return per_head_rms_norm(x, w_name, 1, num_heads, S, head_dim, eps);
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

    LogicalId gelu_tanh(LogicalId x, const std::vector<uint32_t> &shape)
    {
        // 0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x^3)))
        LogicalId x_sq = g.mul(x, x);
        LogicalId x_cube = g.mul(x_sq, x);
        LogicalId c1 = g.fill(0.044715f, shape);
        LogicalId term1 = g.mul(x_cube, c1);
        LogicalId term2 = g.add(x, term1);
        LogicalId c2 = g.fill(0.79788456f, shape);
        LogicalId inner = g.mul(term2, c2);

        // tanh(u) = 2 / (1 + exp(-2u)) - 1
        LogicalId neg_two = g.fill(-2.0f, shape);
        LogicalId neg_2u = g.mul(inner, neg_two);
        LogicalId exp_neg_2u = g.pow(g.fill(TGConstants::E, shape), neg_2u);
        LogicalId one = g.fill(1.0f, shape);
        LogicalId two = g.fill(2.0f, shape);
        LogicalId den = g.add(one, exp_neg_2u);
        LogicalId tanh_val = g.add(g.div(two, den), g.neg(one));

        LogicalId one_plus_tanh = g.add(one, tanh_val);
        LogicalId half_x = g.mul(x, g.fill(0.5f, shape));
        return g.mul(half_x, one_plus_tanh);
    }

    LogicalId softmax_4d(LogicalId scores, uint32_t S, uint32_t num_heads, uint32_t B = 1)
    {
        LogicalId max_s = g.repeat(g.max(scores, -1), S, 3);
        LogicalId shifted = g.add(scores, g.neg(max_s));
        LogicalId exps = g.pow(g.fill(TGConstants::E, {B, num_heads, S, S}), shifted);
        LogicalId sums = g.repeat(g.sum(exps, -1), S, 3);
        return g.div(exps, sums);
    }

    // Interleaved 2D pair RoPE rotation matching mmdit.py (ropeapply)
    LogicalId apply_rope(LogicalId x, LogicalId cos_node, LogicalId sin_node, uint32_t num_heads, uint32_t S,
                         uint32_t head_dim)
    {
        uint32_t half_dim = head_dim / 2; // 64
        LogicalId x_5d = g.reshape(
            x, {1, static_cast<int32_t>(num_heads), static_cast<int32_t>(S), static_cast<int32_t>(half_dim), 2});
        LogicalId x_even = g.contiguous(
            g.slice(x_5d, {0, 0, 0, 0, 0},
                    {1, static_cast<int32_t>(num_heads), static_cast<int32_t>(S), static_cast<int32_t>(half_dim), 1}));
        x_even = g.reshape(
            x_even, {1, static_cast<int32_t>(num_heads), static_cast<int32_t>(S), static_cast<int32_t>(half_dim)});
        LogicalId x_odd = g.contiguous(
            g.slice(x_5d, {0, 0, 0, 0, 1},
                    {1, static_cast<int32_t>(num_heads), static_cast<int32_t>(S), static_cast<int32_t>(half_dim), 2}));
        x_odd = g.reshape(
            x_odd, {1, static_cast<int32_t>(num_heads), static_cast<int32_t>(S), static_cast<int32_t>(half_dim)});

        LogicalId cos_exp = g.repeat(cos_node, num_heads, 1);
        LogicalId sin_exp = g.repeat(sin_node, num_heads, 1);

        LogicalId x_rot_even = g.add(g.mul(x_even, cos_exp), g.neg(g.mul(x_odd, sin_exp)));
        LogicalId x_rot_odd = g.add(g.mul(x_even, sin_exp), g.mul(x_odd, cos_exp));

        LogicalId e_5d = g.reshape(x_rot_even, {1, static_cast<int32_t>(num_heads), static_cast<int32_t>(S),
                                                static_cast<int32_t>(half_dim), 1});
        LogicalId o_5d = g.reshape(x_rot_odd, {1, static_cast<int32_t>(num_heads), static_cast<int32_t>(S),
                                               static_cast<int32_t>(half_dim), 1});
        LogicalId pair_5d = g.concat({e_5d, o_5d}, 4);
        return g.reshape(pair_5d,
                         {1, static_cast<int32_t>(num_heads), static_cast<int32_t>(S), static_cast<int32_t>(head_dim)});
    }

    LogicalId compute_timestep_embedding(LogicalId t)
    {
        uint32_t half = cfg.time_dim / 2; // 128
        std::vector<float> freqs_data(half);
        for (uint32_t i = 0; i < half; ++i)
        {
            freqs_data[i] = std::exp(-std::log(10000.0f) * static_cast<float>(i) / static_cast<float>(half));
        }

        LogicalId freqs_node = g.constant({1, 1, half}, freqs_data.data(), DType::FLOAT32);
        // t * 1000.0f
        LogicalId t_scaled = g.mul(t, g.constant({1}, std::vector<float>{1000.0f}.data(), DType::FLOAT32));
        LogicalId t_3d = g.repeat(g.reshape(t_scaled, {1, 1, 1}), half, 2);
        LogicalId angles = g.mul(t_3d, freqs_node);

        int32_t ax_2 = 2;
        LogicalId t_emb = g.concat({g.cos(angles), g.sin(angles)}, g.constant({1}, &ax_2, DType::INT32));

        LogicalId h = linear(t_emb, "tmlp.0.weight", "tmlp.0.bias", cfg.time_dim, cfg.time_mlp_dim, 1);
        h = gelu_tanh(h, {1, 1, cfg.time_mlp_dim});
        return linear(h, "tmlp.2.weight", "tmlp.2.bias", cfg.time_mlp_dim, cfg.time_mlp_dim, 1);
    }

    LogicalId text_fusion_block(LogicalId x, const std::string &prefix, uint32_t B, uint32_t S)
    {
        LogicalId residual = x;
        LogicalId h = rms_norm(x, prefix + "prenorm.scale", B, S, cfg.text_dim, cfg.rms_eps);

        LogicalId q = linear(h, prefix + "attn.wq.weight", "", cfg.text_dim, cfg.text_dim, B, S);
        LogicalId k = linear(h, prefix + "attn.wk.weight", "", cfg.text_dim, cfg.text_dim, B, S);
        LogicalId v = linear(h, prefix + "attn.wv.weight", "", cfg.text_dim, cfg.text_dim, B, S);

        q = g.contiguous(g.permute(
            g.reshape(q, {static_cast<int32_t>(B), static_cast<int32_t>(S), static_cast<int32_t>(cfg.text_fusion_heads),
                          static_cast<int32_t>(cfg.text_fusion_head_dim)}),
            {0, 2, 1, 3}));
        k = g.contiguous(g.permute(
            g.reshape(k, {static_cast<int32_t>(B), static_cast<int32_t>(S), static_cast<int32_t>(cfg.text_fusion_heads),
                          static_cast<int32_t>(cfg.text_fusion_head_dim)}),
            {0, 2, 1, 3}));
        v = g.contiguous(g.permute(
            g.reshape(v, {static_cast<int32_t>(B), static_cast<int32_t>(S), static_cast<int32_t>(cfg.text_fusion_heads),
                          static_cast<int32_t>(cfg.text_fusion_head_dim)}),
            {0, 2, 1, 3}));

        q = per_head_rms_norm(q, prefix + "attn.qknorm.qnorm.scale", B, cfg.text_fusion_heads, S,
                              cfg.text_fusion_head_dim, cfg.rms_eps);
        k = per_head_rms_norm(k, prefix + "attn.qknorm.knorm.scale", B, cfg.text_fusion_heads, S,
                              cfg.text_fusion_head_dim, cfg.rms_eps);

        float scale_val = 1.0f / std::sqrt(static_cast<float>(cfg.text_fusion_head_dim));
        q = g.mul(q, g.fill(scale_val, {B, cfg.text_fusion_heads, S, cfg.text_fusion_head_dim}));

        LogicalId k_t = g.contiguous(g.permute(k, {0, 1, 3, 2}));
        LogicalId scores = g.dot(q, k_t);
        LogicalId probs = softmax_4d(scores, S, cfg.text_fusion_heads, B);

        LogicalId attn_out = g.dot(probs, v);
        LogicalId ctx_perm = g.contiguous(g.permute(attn_out, {0, 2, 1, 3}));
        LogicalId ctx_flat =
            g.reshape(ctx_perm, {static_cast<int32_t>(B), static_cast<int32_t>(S), static_cast<int32_t>(cfg.text_dim)});

        LogicalId gate =
            sigmoid(linear(h, prefix + "attn.gate.weight", "", cfg.text_dim, cfg.text_dim, B, S), {B, S, cfg.text_dim});
        LogicalId gated_attn = g.mul(ctx_flat, gate);
        LogicalId attn_proj = linear(gated_attn, prefix + "attn.wo.weight", "", cfg.text_dim, cfg.text_dim, B, S);
        x = g.add(residual, attn_proj);
        residual = x;

        h = rms_norm(x, prefix + "postnorm.scale", B, S, cfg.text_dim, cfg.rms_eps);
        LogicalId gate_mlp =
            linear(h, prefix + "mlp.gate.weight", "", cfg.text_dim, cfg.text_fusion_intermediate, B, S);
        LogicalId up_mlp = linear(h, prefix + "mlp.up.weight", "", cfg.text_dim, cfg.text_fusion_intermediate, B, S);
        LogicalId swiglu_mlp = g.mul(silu(gate_mlp, {B, S, cfg.text_fusion_intermediate}), up_mlp);
        LogicalId mlp_out =
            linear(swiglu_mlp, prefix + "mlp.down.weight", "", cfg.text_fusion_intermediate, cfg.text_dim, B, S);

        return g.add(residual, mlp_out);
    }

    LogicalId patchify_latents(LogicalId latents)
    {
        // latents: [1, 16, H_lat, W_lat] -> [1, 16, Gh, 2, Gw, 2]
        LogicalId split = g.reshape(latents, {1, static_cast<int32_t>(cfg.latent_channels),
                                              static_cast<int32_t>(cfg.grid_h), static_cast<int32_t>(cfg.patch_size),
                                              static_cast<int32_t>(cfg.grid_w), static_cast<int32_t>(cfg.patch_size)});
        // Permute to [1, Gh, Gw, 16, 2, 2]
        LogicalId perm = g.contiguous(g.permute(split, {0, 2, 4, 1, 3, 5}));
        LogicalId patches =
            g.reshape(perm, {1, static_cast<int32_t>(cfg.num_patches), static_cast<int32_t>(cfg.patch_dim)});
        return linear(patches, "first.weight", "first.bias", cfg.patch_dim, cfg.hidden_size, cfg.num_patches);
    }

    LogicalId unpatchify_latents(LogicalId x_img)
    {
        // x_img: [1, num_patches, 64]
        LogicalId split =
            g.reshape(x_img, {1, static_cast<int32_t>(cfg.grid_h), static_cast<int32_t>(cfg.grid_w),
                              static_cast<int32_t>(cfg.latent_channels), static_cast<int32_t>(cfg.patch_size),
                              static_cast<int32_t>(cfg.patch_size)});
        // Permute to [1, 16, Gh, 2, Gw, 2]
        LogicalId perm = g.contiguous(g.permute(split, {0, 3, 1, 4, 2, 5}));
        return g.reshape(perm, {1, static_cast<int32_t>(cfg.latent_channels), static_cast<int32_t>(cfg.latent_h),
                                static_cast<int32_t>(cfg.latent_w)});
    }

    LogicalId single_stream_block(LogicalId x, uint32_t layer_idx, LogicalId t_mod, LogicalId cos_node,
                                  LogicalId sin_node)
    {
        std::string prefix = "blocks." + std::to_string(layer_idx) + ".";
        uint32_t S = cfg.total_seq_len;
        LogicalId residual = x;

        // DoubleSharedModulation: out = t_mod + mod.lin (ADDITIVE!)
        LogicalId mod_lin = weight(prefix + "mod.lin");
        LogicalId mod_lin_3d = g.reshape(mod_lin, {1, 1, 36864});
        LogicalId mod = g.add(t_mod, mod_lin_3d);

        auto get_chunk = [&](uint32_t chunk_idx) -> LogicalId {
            LogicalId chunk = g.slice(mod, {0, 0, static_cast<int32_t>(chunk_idx * cfg.hidden_size)},
                                      {1, 1, static_cast<int32_t>((chunk_idx + 1) * cfg.hidden_size)});
            return g.repeat(chunk, S, 1);
        };

        LogicalId prescale = get_chunk(0);
        LogicalId preshift = get_chunk(1);
        LogicalId pregate = get_chunk(2);
        LogicalId postscale = get_chunk(3);
        LogicalId postshift = get_chunk(4);
        LogicalId postgate = get_chunk(5);

        LogicalId h = rms_norm(x, prefix + "prenorm.scale", S, cfg.hidden_size, cfg.rms_eps);
        LogicalId one = g.fill(1.0f, {1, S, cfg.hidden_size});
        h = g.add(g.mul(g.add(one, prescale), h), preshift);

        LogicalId q = linear(h, prefix + "attn.wq.weight", "", cfg.hidden_size, cfg.num_heads * cfg.head_dim, S);
        LogicalId k = linear(h, prefix + "attn.wk.weight", "", cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, S);
        LogicalId v = linear(h, prefix + "attn.wv.weight", "", cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, S);

        q = g.contiguous(g.permute(g.reshape(q, {1, static_cast<int32_t>(S), static_cast<int32_t>(cfg.num_heads),
                                                 static_cast<int32_t>(cfg.head_dim)}),
                                   {0, 2, 1, 3}));
        k = g.contiguous(g.permute(g.reshape(k, {1, static_cast<int32_t>(S), static_cast<int32_t>(cfg.num_kv_heads),
                                                 static_cast<int32_t>(cfg.head_dim)}),
                                   {0, 2, 1, 3}));
        v = g.contiguous(g.permute(g.reshape(v, {1, static_cast<int32_t>(S), static_cast<int32_t>(cfg.num_kv_heads),
                                                 static_cast<int32_t>(cfg.head_dim)}),
                                   {0, 2, 1, 3}));

        q = per_head_rms_norm(q, prefix + "attn.qknorm.qnorm.scale", cfg.num_heads, S, cfg.head_dim, cfg.rms_eps);
        k = per_head_rms_norm(k, prefix + "attn.qknorm.knorm.scale", cfg.num_kv_heads, S, cfg.head_dim, cfg.rms_eps);

        q = apply_rope(q, cos_node, sin_node, cfg.num_heads, S, cfg.head_dim);
        k = apply_rope(k, cos_node, sin_node, cfg.num_kv_heads, S, cfg.head_dim);

        float scale_val = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
        q = g.mul(q, g.fill(scale_val, {1, cfg.num_heads, S, cfg.head_dim}));

        uint32_t rep_factor = cfg.num_heads / cfg.num_kv_heads; // 4
        LogicalId k_5d = g.repeat(g.reshape(k, {1, static_cast<int32_t>(cfg.num_kv_heads), 1, static_cast<int32_t>(S),
                                                static_cast<int32_t>(cfg.head_dim)}),
                                  rep_factor, 2);
        LogicalId v_5d = g.repeat(g.reshape(v, {1, static_cast<int32_t>(cfg.num_kv_heads), 1, static_cast<int32_t>(S),
                                                static_cast<int32_t>(cfg.head_dim)}),
                                  rep_factor, 2);
        k = g.reshape(g.contiguous(k_5d), {1, static_cast<int32_t>(cfg.num_heads), static_cast<int32_t>(S),
                                           static_cast<int32_t>(cfg.head_dim)});
        v = g.reshape(g.contiguous(v_5d), {1, static_cast<int32_t>(cfg.num_heads), static_cast<int32_t>(S),
                                           static_cast<int32_t>(cfg.head_dim)});

        LogicalId scores = g.dot(q, g.contiguous(g.permute(k, {0, 1, 3, 2})));
        LogicalId probs = softmax_4d(scores, S, cfg.num_heads);

        LogicalId attn_out = g.dot(probs, v);
        LogicalId ctx_perm = g.contiguous(g.permute(attn_out, {0, 2, 1, 3}));
        LogicalId ctx_flat = g.reshape(ctx_perm, {1, static_cast<int32_t>(S), static_cast<int32_t>(cfg.hidden_size)});

        LogicalId gate = sigmoid(linear(h, prefix + "attn.gate.weight", "", cfg.hidden_size, cfg.hidden_size, S),
                                 {1, S, cfg.hidden_size});
        LogicalId gated_attn = g.mul(ctx_flat, gate);
        LogicalId attn_proj = linear(gated_attn, prefix + "attn.wo.weight", "", cfg.hidden_size, cfg.hidden_size, S);

        x = g.add(residual, g.mul(pregate, attn_proj));
        residual = x;

        LogicalId h2 = rms_norm(x, prefix + "postnorm.scale", S, cfg.hidden_size, cfg.rms_eps);
        h2 = g.add(g.mul(g.add(one, postscale), h2), postshift);

        LogicalId gate_mlp = linear(h2, prefix + "mlp.gate.weight", "", cfg.hidden_size, cfg.mlp_hidden_dim, S);
        LogicalId up_mlp = linear(h2, prefix + "mlp.up.weight", "", cfg.hidden_size, cfg.mlp_hidden_dim, S);
        LogicalId swiglu = g.mul(silu(gate_mlp, {1, S, cfg.mlp_hidden_dim}), up_mlp);
        LogicalId mlp_out = linear(swiglu, prefix + "mlp.down.weight", "", cfg.mlp_hidden_dim, cfg.hidden_size, S);

        return g.add(residual, g.mul(postgate, mlp_out));
    }

  public:
    Krea2TurboModel(Krea2TurboConfig config, Graph &graph, MemoryManager &memory, std::string weight_path)
        : cfg(config), g(graph), mem(memory), w_path(std::move(weight_path))
    {
    }

    std::tuple<LogicalId, LogicalId> compute_rope_3d(uint32_t S_text, uint32_t grid_h, uint32_t grid_w,
                                                     uint32_t head_dim)
    {
        uint32_t S_total = S_text + grid_h * grid_w;
        uint32_t half_head_dim = head_dim / 2; // 64
        std::vector<float> freqs_cos(S_total * half_head_dim);
        std::vector<float> freqs_sin(S_total * half_head_dim);

        // axes = [32, 48, 48], sum = 128
        // half axes = [16, 24, 24], sum = 64
        uint32_t dim_t = 32, dim_h = 48, dim_w = 48;
        uint32_t half_t = dim_t / 2; // 16
        uint32_t half_h = dim_h / 2; // 24
        uint32_t half_w = dim_w / 2; // 24

        for (uint32_t s = 0; s < S_total; ++s)
        {
            float pos_t = 0.0f, pos_h = 0.0f, pos_w = 0.0f;
            if (s >= S_text)
            {
                uint32_t p = s - S_text;
                pos_h = static_cast<float>(p / grid_w);
                pos_w = static_cast<float>(p % grid_w);
            }

            uint32_t offset = 0;
            for (uint32_t i = 0; i < half_t; ++i)
            {
                float freq = 1.0f / std::pow(cfg.rope_theta, static_cast<float>(2 * i) / static_cast<float>(dim_t));
                float theta = pos_t * freq;
                freqs_cos[s * half_head_dim + offset + i] = std::cos(theta);
                freqs_sin[s * half_head_dim + offset + i] = std::sin(theta);
            }
            offset += half_t;

            for (uint32_t i = 0; i < half_h; ++i)
            {
                float freq = 1.0f / std::pow(cfg.rope_theta, static_cast<float>(2 * i) / static_cast<float>(dim_h));
                float theta = pos_h * freq;
                freqs_cos[s * half_head_dim + offset + i] = std::cos(theta);
                freqs_sin[s * half_head_dim + offset + i] = std::sin(theta);
            }
            offset += half_h;

            for (uint32_t i = 0; i < half_w; ++i)
            {
                float freq = 1.0f / std::pow(cfg.rope_theta, static_cast<float>(2 * i) / static_cast<float>(dim_w));
                float theta = pos_w * freq;
                freqs_cos[s * half_head_dim + offset + i] = std::cos(theta);
                freqs_sin[s * half_head_dim + offset + i] = std::sin(theta);
            }
        }

        LogicalId cos_node = g.constant({1, 1, S_total, half_head_dim}, freqs_cos.data(), DType::FLOAT32);
        LogicalId sin_node = g.constant({1, 1, S_total, half_head_dim}, freqs_sin.data(), DType::FLOAT32);
        return {cos_node, sin_node};
    }

    LogicalId text_fusion(LogicalId text_raw)
    {
        // text_raw: [1, text_seq_len, text_num_layers, text_dim] -> [128, 12, 2560]
        uint32_t B_tok = cfg.text_seq_len;      // 128
        uint32_t S_layer = cfg.text_num_layers; // 12
        LogicalId h = g.reshape(
            text_raw, {static_cast<int32_t>(B_tok), static_cast<int32_t>(S_layer), static_cast<int32_t>(cfg.text_dim)});

        for (uint32_t i = 0; i < cfg.num_layerwise_blocks; ++i)
        {
            std::string prefix = "txtfusion.layerwise_blocks." + std::to_string(i) + ".";
            h = text_fusion_block(h, prefix, B_tok, S_layer);
        }

        // Projector: rearrange "(b l) n d -> b l d n", project n=12 -> 1
        LogicalId h_4d = g.reshape(
            h, {1, static_cast<int32_t>(B_tok), static_cast<int32_t>(S_layer), static_cast<int32_t>(cfg.text_dim)});
        LogicalId h_perm = g.contiguous(g.permute(h_4d, {0, 1, 3, 2}));
        LogicalId h_flat =
            g.reshape(h_perm, {1, static_cast<int32_t>(B_tok * cfg.text_dim), static_cast<int32_t>(S_layer)});

        LogicalId proj_w = weight("txtfusion.projector.weight");
        LogicalId proj_w_t = g.contiguous(g.permute(proj_w, {1, 0}));
        LogicalId proj_w_3d = g.reshape(proj_w_t, {1, static_cast<int32_t>(S_layer), 1});
        LogicalId h_collapsed = g.dot(h_flat, proj_w_3d);

        LogicalId fused = g.reshape(h_collapsed, {1, static_cast<int32_t>(B_tok), static_cast<int32_t>(cfg.text_dim)});

        for (uint32_t i = 0; i < cfg.num_refiner_blocks; ++i)
        {
            std::string prefix = "txtfusion.refiner_blocks." + std::to_string(i) + ".";
            fused = text_fusion_block(fused, prefix, 1, B_tok);
        }

        LogicalId h_norm = rms_norm(fused, "txtmlp.0.scale", 1, B_tok, cfg.text_dim, cfg.rms_eps);
        LogicalId h_mid = linear(h_norm, "txtmlp.1.weight", "txtmlp.1.bias", cfg.text_dim, cfg.hidden_size, B_tok);
        h_mid = gelu_tanh(h_mid, {1, B_tok, cfg.hidden_size});
        return linear(h_mid, "txtmlp.3.weight", "txtmlp.3.bias", cfg.hidden_size, cfg.hidden_size, B_tok);
    }

    LogicalId predict_velocity_step(LogicalId latent_id, LogicalId timestep_id, LogicalId txt_tokens,
                                    LogicalId cos_node, LogicalId sin_node)
    {
        LogicalId t = compute_timestep_embedding(timestep_id); // [1, 1, 6144]
        LogicalId t_gelu = gelu_tanh(t, {1, 1, cfg.time_mlp_dim});
        LogicalId t_mod = linear(t_gelu, "tproj.1.weight", "tproj.1.bias", cfg.time_mlp_dim, cfg.time_mlp_dim * 6, 1);

        LogicalId img_tokens = patchify_latents(latent_id);
        LogicalId x = g.concat({txt_tokens, img_tokens}, 1);

        for (uint32_t i = 0; i < cfg.num_layers; ++i)
        {
            x = single_stream_block(x, i, t_mod, cos_node, sin_node);
        }

        LogicalId x_img = g.slice(x, {0, static_cast<int32_t>(cfg.text_seq_len), 0},
                                  {1, static_cast<int32_t>(cfg.total_seq_len), static_cast<int32_t>(cfg.hidden_size)});
        x_img = g.contiguous(x_img);

        LogicalId x_norm = rms_norm(x_img, "last.norm.scale", cfg.num_patches, cfg.hidden_size, cfg.rms_eps);

        // SimpleModulation: scale = t + last.modulation.lin[0], shift = t + last.modulation.lin[1]
        LogicalId last_mod_lin = weight("last.modulation.lin");
        LogicalId last_scale = g.slice(last_mod_lin, {0, 0}, {1, static_cast<int32_t>(cfg.hidden_size)});
        LogicalId last_shift = g.slice(last_mod_lin, {1, 0}, {2, static_cast<int32_t>(cfg.hidden_size)});
        last_scale = g.reshape(last_scale, {1, 1, static_cast<int32_t>(cfg.hidden_size)});
        last_shift = g.reshape(last_shift, {1, 1, static_cast<int32_t>(cfg.hidden_size)});

        LogicalId scale_vec = g.add(t, last_scale);
        LogicalId shift_vec = g.add(t, last_shift);

        LogicalId scale_exp = g.repeat(scale_vec, cfg.num_patches, 1);
        LogicalId shift_exp = g.repeat(shift_vec, cfg.num_patches, 1);
        LogicalId one = g.fill(1.0f, {1, cfg.num_patches, cfg.hidden_size});

        LogicalId x_mod = g.add(g.mul(g.add(one, scale_exp), x_norm), shift_exp);

        LogicalId v_patches =
            linear(x_mod, "last.linear.weight", "last.linear.bias", cfg.hidden_size, cfg.patch_dim, cfg.num_patches);

        return unpatchify_latents(v_patches);
    }

    LogicalId build_graph(LogicalId latent_id, LogicalId timestep_id, LogicalId text_id)
    {
        LogicalId txt_tokens = text_fusion(text_id);
        auto [cos_node, sin_node] = compute_rope_3d(cfg.text_seq_len, cfg.grid_h, cfg.grid_w, cfg.head_dim);
        return predict_velocity_step(latent_id, timestep_id, txt_tokens, cos_node, sin_node);
    }

    LogicalId build_unrolled_dit(LogicalId initial_latent, LogicalId text_embeddings, uint32_t steps = 8,
                                 float mu = 1.15f)
    {
        LogicalId txt_tokens = text_fusion(text_embeddings);
        auto [cos_node, sin_node] = compute_rope_3d(cfg.text_seq_len, cfg.grid_h, cfg.grid_w, cfg.head_dim);

        std::vector<float> timesteps(steps + 1);
        float exp_mu = std::exp(mu);
        for (uint32_t i = 0; i <= steps; ++i)
        {
            float s = 1.0f - static_cast<float>(i) / static_cast<float>(steps);
            timesteps[i] = (exp_mu * s) / (1.0f + (exp_mu - 1.0f) * s);
        }

        LogicalId cur_latent = initial_latent;
        for (uint32_t step = 0; step < steps; ++step)
        {
            float t_cur = timesteps[step];
            float t_nxt = timesteps[step + 1];
            float dt = t_nxt - t_cur;

            LogicalId t_node = g.constant({1}, &t_cur, DType::FLOAT32);
            LogicalId v = predict_velocity_step(cur_latent, t_node, txt_tokens, cos_node, sin_node);

            LogicalId dt_node = g.fill(dt, {1, cfg.latent_channels, cfg.latent_h, cfg.latent_w});
            LogicalId delta = g.mul(v, dt_node);
            cur_latent = g.add(cur_latent, delta);
        }

        return cur_latent;
    }
};