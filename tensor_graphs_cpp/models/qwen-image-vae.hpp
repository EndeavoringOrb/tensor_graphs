#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/common/constants.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/types.hpp"

// Qwen-Image / Wan VAE Latent Normalization Constants
static const float QWEN_IMAGE_LATENTS_MEAN[16] = {-0.7571f, -0.7089f, -0.9113f, 0.1075f,  -0.1745f, 0.9653f,
                                                  -0.1517f, 1.5508f,  0.4134f,  -0.0715f, 0.5517f,  -0.3632f,
                                                  -0.1922f, -0.9497f, 0.2503f,  -0.2921f};

static const float QWEN_IMAGE_LATENTS_STD[16] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f,
                                                 2.6052f, 2.0743f, 3.2687f, 2.1526f, 2.8652f, 1.5579f,
                                                 1.6382f, 1.1253f, 2.8251f, 1.9160f};

struct Krea2TurboVAEConfig
{
    uint32_t height = 1024;
    uint32_t width = 1024;
    uint32_t in_channels = 3;
    uint32_t latent_channels = 16;
    uint32_t vae_scale_factor = 8;

    uint32_t base_dim = 96;
    std::vector<uint32_t> dim_mult = {1, 2, 4, 4};
    uint32_t num_res_blocks = 2;

    uint32_t latent_h = 128;
    uint32_t latent_w = 128;

    Krea2TurboVAEConfig(uint32_t h = 1024, uint32_t w = 1024) : height(h), width(w)
    {
        latent_h = height / vae_scale_factor;
        latent_w = width / vae_scale_factor;
    }
};

class Krea2TurboVAEModel
{
  private:
    Krea2TurboVAEConfig cfg;
    Graph &g;
    MemoryManager &mem;
    const std::string w_path;

    std::string resolve_weight_name(const std::string &name)
    {
        std::vector<std::string> candidate_prefixes = {
            "",
            "model.",
            "vae.",
            "first_stage_model.",
            "model.vae."
        };
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
        LogicalId raw_weight = g.weight(w_path, resolved);
        return g.cast(raw_weight, DType::FLOAT32);
    }

    LogicalId load_conv_weight(const std::string &name, uint32_t out_c, uint32_t in_c, uint32_t k)
    {
        std::string resolved = resolve_weight_name(name);
        TensorMetadata meta = FileRegistry::get().getMetadata(w_path, resolved);
        LogicalId raw = g.weight(w_path, resolved);
        LogicalId raw_f32 = g.cast(raw, DType::FLOAT32);

        // If weight is stored as 3D causal conv [out_c, in_c, kT, kH, kW], slice the last temporal frame
        if (meta.shape.size() == 5)
        {
            uint32_t kT = meta.shape[2];
            uint32_t kH = meta.shape[3];
            uint32_t kW = meta.shape[4];
            LogicalId sliced = g.slice(raw_f32, {0, 0, (int32_t)(kT - 1), 0, 0},
                                       {(int32_t)out_c, (int32_t)in_c, (int32_t)kT, (int32_t)kH, (int32_t)kW});
            sliced = g.contiguous(sliced);
            return g.reshape(sliced, {(int32_t)out_c, (int32_t)(in_c * k * k)});
        }
        else
        {
            return g.reshape(raw_f32, {(int32_t)out_c, (int32_t)(in_c * k * k)});
        }
    }

    LogicalId conv2d(LogicalId x, const std::string &w_name, const std::string &b_name, uint32_t in_c, uint32_t out_c,
                     uint32_t H, uint32_t W, uint32_t k = 3, uint32_t stride = 1, uint32_t pad = 1)
    {
        LogicalId w_2d = load_conv_weight(w_name, out_c, in_c, k);
        LogicalId w_3d = g.reshape(w_2d, {1, (int32_t)out_c, (int32_t)(in_c * k * k)});

        LogicalId col;
        uint32_t H_out = (H + 2 * pad - k) / stride + 1;
        uint32_t W_out = (W + 2 * pad - k) / stride + 1;

        if (k == 1 && stride == 1 && pad == 0)
        {
            col = g.reshape(x, {1, (int32_t)in_c, (int32_t)(H * W)});
        }
        else
        {
            col = g.im2col(x, (int32_t)k, (int32_t)stride, (int32_t)pad);
        }

        LogicalId out_flat = g.dot(w_3d, col);
        LogicalId out = g.reshape(out_flat, {1, (int32_t)out_c, (int32_t)H_out, (int32_t)W_out});

        if (!b_name.empty())
        {
            std::string resolved_b = resolve_weight_name(b_name);
            if (FileRegistry::get().hasTensor(w_path, resolved_b))
            {
                LogicalId b = weight(b_name);
                LogicalId b_4d = g.reshape(b, {1, (int32_t)out_c, 1, 1});
                LogicalId b_exp = g.repeat(g.repeat(b_4d, H_out, 2), W_out, 3);
                out = g.add(out, b_exp);
            }
        }
        return out;
    }

    LogicalId rms_norm_2d(LogicalId x, const std::string &gamma_name, uint32_t C, uint32_t H, uint32_t W,
                          float eps = 1e-6f)
    {
        LogicalId x_sq = g.mul(x, x);
        LogicalId sum_sq = g.sum(x_sq, 1);
        LogicalId mean_sq = g.div(sum_sq, g.fill((float)C, {1, 1, H, W}));
        LogicalId std = g.pow(g.add(mean_sq, g.fill(eps, {1, 1, H, W})), g.fill(0.5f, {1, 1, H, W}));
        LogicalId inv_std = g.repeat(g.div(g.fill(1.0f, {1, 1, H, W}), std), C, 1);
        LogicalId x_norm = g.mul(x, inv_std);

        if (!gamma_name.empty())
        {
            std::string resolved_gamma = resolve_weight_name(gamma_name);
            if (FileRegistry::get().hasTensor(w_path, resolved_gamma))
            {
                LogicalId gamma = weight(gamma_name);
                LogicalId gamma_4d = g.reshape(gamma, {1, (int32_t)C, 1, 1});
                LogicalId gamma_exp = g.repeat(g.repeat(gamma_4d, H, 2), W, 3);
                x_norm = g.mul(x_norm, gamma_exp);
            }
        }
        return x_norm;
    }

    LogicalId silu_2d(LogicalId x, uint32_t C, uint32_t H, uint32_t W)
    {
        LogicalId neg_x = g.mul(x, g.fill(-1.0f, {1, C, H, W}));
        LogicalId exp_neg_x = g.pow(g.fill(TGConstants::E, {1, C, H, W}), neg_x);
        LogicalId one = g.fill(1.0f, {1, C, H, W});
        LogicalId sig = g.div(one, g.add(one, exp_neg_x));
        return g.mul(x, sig);
    }

    LogicalId residual_block(LogicalId x, const std::string &prefix, uint32_t in_c, uint32_t out_c, uint32_t H,
                             uint32_t W)
    {
        LogicalId h = x;
        if (in_c != out_c)
        {
            h = conv2d(x, prefix + "shortcut.weight", prefix + "shortcut.bias", in_c, out_c, H, W, 1, 1, 0);
        }

        LogicalId out = rms_norm_2d(x, prefix + "residual.0.gamma", in_c, H, W);
        out = silu_2d(out, in_c, H, W);
        out = conv2d(out, prefix + "residual.2.weight", prefix + "residual.2.bias", in_c, out_c, H, W, 3, 1, 1);

        out = rms_norm_2d(out, prefix + "residual.3.gamma", out_c, H, W);
        out = silu_2d(out, out_c, H, W);
        out = conv2d(out, prefix + "residual.6.weight", prefix + "residual.6.bias", out_c, out_c, H, W, 3, 1, 1);

        return g.add(out, h);
    }

    LogicalId attention_block(LogicalId x, const std::string &prefix, uint32_t dim, uint32_t H, uint32_t W)
    {
        LogicalId identity = x;
        LogicalId norm_x = rms_norm_2d(x, prefix + "norm.gamma", dim, H, W);

        LogicalId qkv = conv2d(norm_x, prefix + "to_qkv.weight", prefix + "to_qkv.bias", dim, 3 * dim, H, W, 1, 1, 0);

        LogicalId qkv_flat = g.reshape(qkv, {1, (int32_t)(3 * dim), (int32_t)(H * W)});
        LogicalId qkv_t = g.contiguous(g.permute(qkv_flat, {0, 2, 1})); // [1, H*W, 3*dim]

        int32_t HW = H * W;
        LogicalId q = g.contiguous(g.slice(qkv_t, {0, 0, 0}, {1, HW, (int32_t)dim}));
        LogicalId k = g.contiguous(g.slice(qkv_t, {0, 0, (int32_t)dim}, {1, HW, (int32_t)(2 * dim)}));
        LogicalId v = g.contiguous(g.slice(qkv_t, {0, 0, (int32_t)(2 * dim)}, {1, HW, (int32_t)(3 * dim)}));

        float scale = 1.0f / std::sqrt((float)dim);
        LogicalId q_scaled = g.mul(q, g.fill(scale, {1, (uint32_t)HW, dim}));

        LogicalId k_t = g.contiguous(g.permute(k, {0, 2, 1})); // [1, dim, H*W]
        LogicalId scores = g.dot(q_scaled, k_t);               // [1, H*W, H*W]

        LogicalId max_s = g.repeat(g.max(scores, 2), (uint32_t)HW, 2);
        LogicalId shifted = g.add(scores, g.neg(max_s));
        LogicalId exps = g.pow(g.fill(TGConstants::E, {1, (uint32_t)HW, (uint32_t)HW}), shifted);
        LogicalId sum_exps = g.repeat(g.sum(exps, 2), (uint32_t)HW, 2);
        LogicalId probs = g.div(exps, sum_exps);

        LogicalId attn_out = g.dot(probs, v); // [1, H*W, dim]

        LogicalId attn_t = g.contiguous(g.permute(attn_out, {0, 2, 1})); // [1, dim, H*W]
        LogicalId attn_2d = g.reshape(attn_t, {1, (int32_t)dim, (int32_t)H, (int32_t)W});

        LogicalId proj = conv2d(attn_2d, prefix + "proj.weight", prefix + "proj.bias", dim, dim, H, W, 1, 1, 0);
        return g.add(identity, proj);
    }

    LogicalId upsample_2d(LogicalId x, uint32_t C, uint32_t H, uint32_t W)
    {
        LogicalId r1 = g.reshape(x, {1, (int32_t)C, (int32_t)H, 1, (int32_t)W, 1});
        LogicalId rep_h = g.repeat(r1, 2, 3);
        LogicalId rep_hw = g.repeat(rep_h, 2, 5);
        LogicalId contig = g.contiguous(rep_hw);
        return g.reshape(contig, {1, (int32_t)C, (int32_t)(2 * H), (int32_t)(2 * W)});
    }

  public:
    Krea2TurboVAEModel(Krea2TurboVAEConfig config, Graph &graph, MemoryManager &memory, std::string weight_path)
        : cfg(config), g(graph), mem(memory), w_path(std::move(weight_path))
    {
    }

    LogicalId build_graph(LogicalId latent_id)
    {
        // 1. Unscale input latents using pre-trained mean and standard deviation: z = z_norm * std + mean
        std::vector<float> std_data(16);
        std::vector<float> mean_data(16);
        for (int i = 0; i < 16; ++i)
        {
            std_data[i] = QWEN_IMAGE_LATENTS_STD[i];
            mean_data[i] = QWEN_IMAGE_LATENTS_MEAN[i];
        }

        LogicalId std_node = g.constant({1, 16, 1, 1}, std_data.data(), DType::FLOAT32);
        LogicalId mean_node = g.constant({1, 16, 1, 1}, mean_data.data(), DType::FLOAT32);
        LogicalId std_exp = g.repeat(g.repeat(std_node, cfg.latent_h, 2), cfg.latent_w, 3);
        LogicalId mean_exp = g.repeat(g.repeat(mean_node, cfg.latent_h, 2), cfg.latent_w, 3);

        LogicalId x = g.add(g.mul(latent_id, std_exp), mean_exp);

        // 2. Post-Quant 1x1 Convolution (conv2): 16 -> 16 [1, 16, H_lat, W_lat]
        uint32_t cur_h = cfg.latent_h;
        uint32_t cur_w = cfg.latent_w;
        x = conv2d(x, "conv2.weight", "conv2.bias", 16, 16, cur_h, cur_w, 1, 1, 0);

        // 3. Conv In (decoder.conv1): 16 -> 384 [1, 384, H_lat, W_lat]
        x = conv2d(x, "decoder.conv1.weight", "decoder.conv1.bias", 16, 384, cur_h, cur_w, 3, 1, 1);

        // 4. Middle Block: Resnet -> Attention -> Resnet
        x = residual_block(x, "decoder.middle.0.", 384, 384, cur_h, cur_w);
        x = attention_block(x, "decoder.middle.1.", 384, cur_h, cur_w);
        x = residual_block(x, "decoder.middle.2.", 384, 384, cur_h, cur_w);

        // 5. Up Blocks (4 stages)
        // Stage 0: 384 -> 384, Upsample 2x -> 192 (cur_h, cur_w: 128 -> 256)
        x = residual_block(x, "decoder.upsamples.0.", 384, 384, cur_h, cur_w);
        x = residual_block(x, "decoder.upsamples.1.", 384, 384, cur_h, cur_w);
        x = residual_block(x, "decoder.upsamples.2.", 384, 384, cur_h, cur_w);
        x = upsample_2d(x, 384, cur_h, cur_w);
        cur_h *= 2;
        cur_w *= 2;
        x = conv2d(x, "decoder.upsamples.3.resample.1.weight", "decoder.upsamples.3.resample.1.bias", 384, 192, cur_h, cur_w, 3, 1, 1);

        // Stage 1: 192 -> 384, Upsample 2x -> 192 (cur_h, cur_w: 256 -> 512)
        x = residual_block(x, "decoder.upsamples.4.", 192, 384, cur_h, cur_w);
        x = residual_block(x, "decoder.upsamples.5.", 384, 384, cur_h, cur_w);
        x = residual_block(x, "decoder.upsamples.6.", 384, 384, cur_h, cur_w);
        x = upsample_2d(x, 384, cur_h, cur_w);
        cur_h *= 2;
        cur_w *= 2;
        x = conv2d(x, "decoder.upsamples.7.resample.1.weight", "decoder.upsamples.7.resample.1.bias", 384, 192, cur_h, cur_w, 3, 1, 1);

        // Stage 2: 192 -> 192, Upsample 2x -> 96 (cur_h, cur_w: 512 -> 1024)
        x = residual_block(x, "decoder.upsamples.8.", 192, 192, cur_h, cur_w);
        x = residual_block(x, "decoder.upsamples.9.", 192, 192, cur_h, cur_w);
        x = residual_block(x, "decoder.upsamples.10.", 192, 192, cur_h, cur_w);
        x = upsample_2d(x, 192, cur_h, cur_w);
        cur_h *= 2;
        cur_w *= 2;
        x = conv2d(x, "decoder.upsamples.11.resample.1.weight", "decoder.upsamples.11.resample.1.bias", 192, 96, cur_h, cur_w, 3, 1, 1);

        // Stage 3: 96 -> 96 (no upsample) (cur_h, cur_w: 1024)
        x = residual_block(x, "decoder.upsamples.12.", 96, 96, cur_h, cur_w);
        x = residual_block(x, "decoder.upsamples.13.", 96, 96, cur_h, cur_w);
        x = residual_block(x, "decoder.upsamples.14.", 96, 96, cur_h, cur_w);

        // 6. Out Norm, SiLU, Conv Out: 96 -> 3 [1, 3, height, width]
        x = rms_norm_2d(x, "decoder.head.0.gamma", 96, cur_h, cur_w);
        x = silu_2d(x, 96, cur_h, cur_w);
        return conv2d(x, "decoder.head.2.weight", "decoder.head.2.bias", 96, 3, cur_h, cur_w, 3, 1, 1);
    }
};