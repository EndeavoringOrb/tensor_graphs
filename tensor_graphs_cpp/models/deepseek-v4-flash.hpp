#pragma once
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "core/common/constants.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/types.hpp"

struct DeepSeekV4FlashConfig
{
    uint32_t vocab_size = 129280;
    uint32_t dim = 4096;
    uint32_t moe_inter_dim = 2048;
    uint32_t n_layers = 43;
    uint32_t n_heads = 64;
    uint32_t n_routed_experts = 256;
    uint32_t n_activated_experts = 6;
    uint32_t q_lora_rank = 1024;
    uint32_t head_dim = 512;
    uint32_t v_head_dim = 64;
    uint32_t rope_head_dim = 64;
    uint32_t o_groups = 8;
    uint32_t o_lora_rank = 1024;
    uint32_t window_size = 128;
    uint32_t original_seq_len = 65536;
    float rope_theta = 10000.0f;
    float rope_factor = 16.0f;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
    uint32_t index_n_heads = 64;
    uint32_t index_head_dim = 128;
    uint32_t index_topk = 512;
    uint32_t hc_mult = 4;
    uint32_t hc_sinkhorn_iters = 20;
    float hc_eps = 1e-6f;
    float norm_eps = 1e-6f;
    std::vector<uint32_t> compress_ratios = {0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
                                             128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                                             4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4};
};

class DeepSeekV4FlashModel
{
private:
    DeepSeekV4FlashConfig cfg;
    Graph &g;
    MemoryManager &mem;
    const std::string w_path;
    uint32_t seq_len;
    LogicalId one_fp32;
    LogicalId eps_fp32;

public:
    DeepSeekV4FlashModel(DeepSeekV4FlashConfig config, uint32_t sequence_length, Graph &graph, MemoryManager &memory,
                         const std::string &weight_path)
        : cfg(config), g(graph), mem(memory), w_path(weight_path), seq_len(sequence_length)
    {
        float one_val = 1.0f;
        one_fp32 = g.constant({1}, &one_val, DType::FLOAT32);
        float eps_val = 1e-6f;
        eps_fp32 = g.constant({1}, &eps_val, DType::FLOAT32);
    }

    LogicalId load_mxfp4_weight(const std::string &w_name, const std::string &scale_name, uint32_t out_d, uint32_t in_d)
    {
        LogicalId raw_packed = g.weight(w_path, w_name);
        LogicalId packed = g.cast(raw_packed, DType::E2M1_PACKED_INT8);
        LogicalId unpacked = g.unpack(packed, DType::E2M1);
        LogicalId unpacked_f32 = g.cast(unpacked, DType::FLOAT32);

        LogicalId raw_scale = g.weight(w_path, scale_name);
        LogicalId scale_f32 = g.cast(raw_scale, DType::FLOAT32);

        uint32_t scale_w = in_d / 32;
        int32_t sh3_scale[] = {(int32_t)out_d, (int32_t)scale_w, 1};
        LogicalId scale_reshaped = g.reshape(scale_f32, g.constant({3}, sh3_scale, DType::INT32));
        LogicalId scale_repeated = g.repeat(scale_reshaped, 32, 2);
        int32_t sh2_final[] = {(int32_t)out_d, (int32_t)in_d};
        LogicalId scale_final = g.reshape(scale_repeated, g.constant({2}, sh2_final, DType::INT32));

        return g.mul(unpacked_f32, scale_final);
    }

    LogicalId load_mxfp8_weight(const std::string &w_name, const std::string &scale_name, uint32_t out_d, uint32_t in_d)
    {
        LogicalId raw_weight = g.weight(w_path, w_name);
        LogicalId unpacked_f32 = g.cast(raw_weight, DType::FLOAT32);

        LogicalId raw_scale = g.weight(w_path, scale_name);
        LogicalId scale_f32 = g.cast(raw_scale, DType::FLOAT32);

        uint32_t scale_h = out_d / 128;
        uint32_t scale_w = in_d / 128;
        int32_t sh4_scale[] = {(int32_t)scale_h, 1, (int32_t)scale_w, 1};
        LogicalId scale_reshaped = g.reshape(scale_f32, g.constant({4}, sh4_scale, DType::INT32));
        LogicalId scale_rep1 = g.repeat(scale_reshaped, 128, 1);
        LogicalId scale_rep2 = g.repeat(scale_rep1, 128, 3);
        int32_t sh2_final[] = {(int32_t)out_d, (int32_t)in_d};
        LogicalId scale_final = g.reshape(scale_rep2, g.constant({2}, sh2_final, DType::INT32));

        return g.mul(unpacked_f32, scale_final);
    }

    LogicalId weight(const std::string &name, uint32_t in_d = 0, uint32_t out_d = 0)
    {
        if (name.find("experts.") != std::string::npos && name.find("shared_experts") == std::string::npos)
        {
            std::string scale_name = name;
            size_t pos = scale_name.rfind(".weight");
            if (pos != std::string::npos)
                scale_name.replace(pos, 7, ".scale");
            return load_mxfp4_weight(name, scale_name, out_d, in_d);
        }
        else if (name.find("shared_experts") != std::string::npos || name.find("attn.wq_a") != std::string::npos ||
                 name.find("attn.wq_b") != std::string::npos || name.find("attn.wkv") != std::string::npos ||
                 name.find("attn.wo_a") != std::string::npos || name.find("attn.wo_b") != std::string::npos ||
                 name.find("indexer.wq_b") != std::string::npos)
        {
            std::string scale_name = name;
            size_t pos = scale_name.rfind(".weight");
            if (pos != std::string::npos)
                scale_name.replace(pos, 7, ".scale");
            return load_mxfp8_weight(name, scale_name, out_d, in_d);
        }

        LogicalId raw_weight = g.weight(w_path, name);
        return g.cast(raw_weight, DType::FLOAT32);
    }

    LogicalId rms_norm(LogicalId x_id, LogicalId weight_id, uint32_t dim0, uint32_t dim_size, uint32_t curr_seq_len = 0)
    {
        if (curr_seq_len == 0)
        {
            curr_seq_len = seq_len;
        }
        LogicalId x_sq = g.mul(x_id, x_id);
        int32_t axis_val = -1;
        LogicalId sum_sq = g.sum(x_sq, g.constant({1}, &axis_val, DType::INT32));
        LogicalId n_node = g.fill((float)dim_size, {dim0, curr_seq_len, 1});
        LogicalId mean_sq = g.div(sum_sq, n_node);
        LogicalId eps_node = g.fill(eps_fp32, {dim0, curr_seq_len, 1});
        float half_val = 0.5f;
        LogicalId std = g.pow(g.add(mean_sq, eps_node),
                              g.fill(half_val, {dim0, curr_seq_len, 1}));
        LogicalId inv_std = g.repeat(g.div(g.fill(one_fp32, {dim0, curr_seq_len, 1}), std), dim_size, 2);

        int32_t w_shape[] = {1, 1, (int32_t)dim_size};
        LogicalId w_exp = g.repeat(g.reshape(weight_id, g.constant({3}, w_shape, DType::INT32)), dim0, 0);
        w_exp = g.repeat(w_exp, curr_seq_len, 1);

        return g.mul(g.mul(x_id, inv_std), w_exp);
    }

    LogicalId linear(LogicalId x, const std::string &w_name, uint32_t in_d, uint32_t out_d)
    {
        LogicalId w = weight(w_name, in_d, out_d);
        int32_t p[] = {1, 0};
        LogicalId w_t = g.contiguous(g.permute(w, g.constant({2}, p, DType::INT32)));
        int32_t sh3[] = {1, (int32_t)in_d, (int32_t)out_d};
        return g.dot(x, g.reshape(w_t, g.constant({3}, sh3, DType::INT32)));
    }

    std::tuple<LogicalId, LogicalId, LogicalId> hc_split_sinkhorn(LogicalId mixes, const std::string &prefix)
    {
        uint32_t hc = cfg.hc_mult;
        LogicalId hc_scale = weight(prefix + "scale");
        LogicalId hc_base = weight(prefix + "base");

        auto slice_last_dim = [&](LogicalId t, int32_t st, int32_t en)
        {
            int32_t starts[] = {0, 0, st};
            int32_t ends[] = {1, (int32_t)seq_len, en};
            int32_t steps[] = {1, 1, 1};
            return g.slice(t, g.constant({3}, starts, DType::INT32), g.constant({3}, ends, DType::INT32),
                           g.constant({3}, steps, DType::INT32));
        };
        auto slice_1d = [&](LogicalId t, int32_t st, int32_t en)
        {
            int32_t starts[] = {st}, ends[] = {en}, steps[] = {1};
            return g.slice(t, g.constant({1}, starts, DType::INT32), g.constant({1}, ends, DType::INT32),
                           g.constant({1}, steps, DType::INT32));
        };

        LogicalId mixes_pre = slice_last_dim(mixes, 0, hc);
        LogicalId mixes_post = slice_last_dim(mixes, hc, 2 * hc);
        LogicalId mixes_comb = slice_last_dim(mixes, 2 * hc, 2 * hc + hc * hc);

        LogicalId scale0 = g.fill(slice_1d(hc_scale, 0, 1), {1, seq_len, hc});
        LogicalId scale1 = g.fill(slice_1d(hc_scale, 1, 2), {1, seq_len, hc});
        LogicalId scale2 = g.fill(slice_1d(hc_scale, 2, 3), {1, seq_len, hc * hc});

        auto expand_base = [&](LogicalId b, uint32_t dim)
        {
            int32_t sh[] = {1, 1, (int32_t)dim};
            return g.repeat(g.reshape(b, g.constant({3}, sh, DType::INT32)), seq_len, 1);
        };
        LogicalId base0 = expand_base(slice_1d(hc_base, 0, hc), hc);
        LogicalId base1 = expand_base(slice_1d(hc_base, hc, 2 * hc), hc);
        LogicalId base2 = expand_base(slice_1d(hc_base, 2 * hc, 2 * hc + hc * hc), hc * hc);

        auto sigmoid = [&](LogicalId t, uint32_t last_dim)
        {
            LogicalId neg_one =
                g.fill(-1.0f, {1, seq_len, last_dim});
            LogicalId neg_t = g.mul(t, neg_one);
            float e_val = TGConstants::E;
            LogicalId e_node = g.fill(TGConstants::E, {1, seq_len, last_dim});
            LogicalId exp_neg_t = g.pow(e_node, neg_t);
            LogicalId one_node = g.fill(one_fp32, {1, seq_len, last_dim});
            return g.div(one_node, g.add(one_node, exp_neg_t));
        };

        LogicalId pre =
            g.add(sigmoid(g.add(g.mul(mixes_pre, scale0), base0), hc), g.fill(eps_fp32, {1, seq_len, hc}));
        LogicalId two = g.fill(2.0f, {1, seq_len, hc});
        LogicalId post = g.mul(two, sigmoid(g.add(g.mul(mixes_post, scale1), base1), hc));
        LogicalId comb = g.add(g.mul(mixes_comb, scale2), base2);

        int32_t sh4[] = {1, (int32_t)seq_len, (int32_t)hc, (int32_t)hc};
        comb = g.reshape(comb, g.constant({4}, sh4, DType::INT32));

        // Softmax & Sinkhorn
        int32_t ax_last = -1;
        LogicalId max_c = g.repeat(g.max(comb, g.constant({1}, &ax_last, DType::INT32)), hc, 3);
        comb = g.pow(g.fill(TGConstants::E, {1, seq_len, hc, hc}), g.add(comb, g.neg(max_c)));
        LogicalId sum_c = g.repeat(g.sum(comb, g.constant({1}, &ax_last, DType::INT32)), hc, 3);
        comb = g.add(g.div(comb, sum_c), g.fill(cfg.hc_eps, {1, seq_len, hc, hc}));

        int32_t ax_2 = 2;
        LogicalId col_sum = g.repeat(g.sum(comb, g.constant({1}, &ax_2, DType::INT32)), hc, 2);
        comb = g.add(g.div(comb, g.add(col_sum, g.fill(cfg.hc_eps, {1, seq_len, hc, hc}))),
                     g.fill(cfg.hc_eps, {1, seq_len, hc, hc}));

        for (uint32_t i = 0; i < cfg.hc_sinkhorn_iters - 1; ++i)
        {
            LogicalId r_sum = g.repeat(g.sum(comb, g.constant({1}, &ax_last, DType::INT32)), hc, 3);
            comb = g.add(g.div(comb, g.add(r_sum, g.fill(cfg.hc_eps, {1, seq_len, hc, hc}))),
                         g.fill(cfg.hc_eps, {1, seq_len, hc, hc}));

            LogicalId c_sum = g.repeat(g.sum(comb, g.constant({1}, &ax_2, DType::INT32)), hc, 2);
            comb = g.add(g.div(comb, g.add(c_sum, g.fill(cfg.hc_eps, {1, seq_len, hc, hc}))),
                         g.fill(cfg.hc_eps, {1, seq_len, hc, hc}));
        }

        return std::make_tuple(pre, post, comb);
    }

    std::tuple<LogicalId, LogicalId, LogicalId> hc_pre(LogicalId x, const std::string &prefix)
    {
        uint32_t hc_dim = cfg.hc_mult * cfg.dim;
        uint32_t mix_hc = cfg.hc_mult * (2 + cfg.hc_mult);

        // 1. Reshape x to flat feature space [1, seq_len, hc_mult * dim]
        int32_t sh3[] = {1, (int32_t)seq_len, (int32_t)hc_dim};
        LogicalId x_flat = g.reshape(x, g.constant({3}, sh3, DType::INT32));

        // 2. Compute rsqrt across all hc_dim features -> [1, seq_len, 1]
        LogicalId x_sq = g.mul(x_flat, x_flat);
        int32_t ax_last = -1;
        LogicalId sum_sq = g.sum(x_sq, g.constant({1}, &ax_last, DType::INT32));
        LogicalId mean_sq = g.div(sum_sq, g.fill((float)hc_dim, {1, seq_len, 1}));
        LogicalId std = g.pow(g.add(mean_sq, g.fill(cfg.norm_eps, {1, seq_len, 1})), g.fill(0.5f, {1, seq_len, 1}));
        LogicalId rsqrt = g.div(g.fill(1.0f, {1, seq_len, 1}), std);

        // 3. Project x_flat -> mixes [1, seq_len, mix_hc]
        LogicalId fn_w = weight(prefix + "fn");
        int32_t p[] = {1, 0};
        LogicalId w_t = g.contiguous(g.permute(fn_w, g.constant({2}, p, DType::INT32)));
        int32_t sh3_mix[] = {1, (int32_t)hc_dim, (int32_t)mix_hc};
        LogicalId mixes = g.dot(x_flat, g.reshape(w_t, g.constant({3}, sh3_mix, DType::INT32)));

        // 4. Multiply mixes by rsqrt
        LogicalId rsqrt_exp = g.repeat(rsqrt, mix_hc, 2);
        mixes = g.mul(mixes, rsqrt_exp);

        auto [pre, post, comb] = hc_split_sinkhorn(mixes, prefix);

        // 5. Apply pre weights back to x and sum over hc_mult
        int32_t sh4[] = {1, (int32_t)seq_len, (int32_t)cfg.hc_mult, 1};
        LogicalId pre_exp = g.repeat(g.reshape(pre, g.constant({4}, sh4, DType::INT32)), cfg.dim, 3);
        LogicalId y = g.mul(x, pre_exp);
        int32_t ax_2 = 2;
        y = g.sum(y, g.constant({1}, &ax_2, DType::INT32));

        int32_t sh3_dim[] = {1, (int32_t)seq_len, (int32_t)cfg.dim};
        return std::make_tuple(g.reshape(y, g.constant({3}, sh3_dim, DType::INT32)), post, comb);
    }

    LogicalId hc_post(LogicalId x, LogicalId residual, LogicalId post, LogicalId comb)
    {
        int32_t sh4_x[] = {1, (int32_t)seq_len, 1, (int32_t)cfg.dim};
        LogicalId x_exp = g.repeat(g.reshape(x, g.constant({4}, sh4_x, DType::INT32)), cfg.hc_mult, 2);
        int32_t sh4_p[] = {1, (int32_t)seq_len, (int32_t)cfg.hc_mult, 1};
        LogicalId post_exp = g.repeat(g.reshape(post, g.constant({4}, sh4_p, DType::INT32)), cfg.dim, 3);
        LogicalId term1 = g.mul(post_exp, x_exp);

        int32_t sh5_c[] = {1, (int32_t)seq_len, (int32_t)cfg.hc_mult, (int32_t)cfg.hc_mult, 1};
        LogicalId comb_exp = g.repeat(g.reshape(comb, g.constant({5}, sh5_c, DType::INT32)), cfg.dim, 4);
        int32_t sh5_r[] = {1, (int32_t)seq_len, 1, (int32_t)cfg.hc_mult, (int32_t)cfg.dim};
        LogicalId res_exp = g.repeat(g.reshape(residual, g.constant({5}, sh5_r, DType::INT32)), cfg.hc_mult, 2);

        int32_t ax_3 = 3;
        LogicalId term2_sum = g.sum(g.mul(comb_exp, res_exp), g.constant({1}, &ax_3, DType::INT32));

        int32_t sh4_out[] = {1, (int32_t)seq_len, (int32_t)cfg.hc_mult, (int32_t)cfg.dim};
        return g.add(term1, g.reshape(term2_sum, g.constant({4}, sh4_out, DType::INT32)));
    }

    LogicalId apply_rope(LogicalId x, uint32_t h, uint32_t head_dim)
    {
        std::vector<float> freqs_cos(seq_len * cfg.rope_head_dim);
        std::vector<float> freqs_sin(seq_len * cfg.rope_head_dim);

        for (uint32_t s = 0; s < seq_len; ++s)
        {
            for (uint32_t d = 0; d < cfg.rope_head_dim; d += 2)
            {
                float freq = 1.0f / std::pow(cfg.rope_theta, (float)d / cfg.rope_head_dim);
                if (cfg.original_seq_len > 0)
                {
                    float corr_dim = cfg.rope_head_dim *
                                     std::log((float)cfg.original_seq_len / (cfg.beta_fast * 2 * TGConstants::PI)) /
                                     (2 * std::log(cfg.rope_theta));
                    float low = std::max(0.0f, std::floor(corr_dim));
                    float high = std::min((float)cfg.rope_head_dim - 1.0f, std::ceil(corr_dim));
                    float smooth = std::clamp((d - low) / (high - low + 0.001f), 0.0f, 1.0f);
                    freq = freq / cfg.rope_factor * (1.0f - smooth) + freq * smooth;
                }
                float val = s * freq;
                freqs_cos[s * cfg.rope_head_dim + d] = std::cos(val);
                freqs_cos[s * cfg.rope_head_dim + d + 1] = std::cos(val);
                freqs_sin[s * cfg.rope_head_dim + d] = std::sin(val);
                freqs_sin[s * cfg.rope_head_dim + d + 1] = std::sin(val);
            }
        }

        LogicalId cos_node = g.constant({1, seq_len, 1, cfg.rope_head_dim}, freqs_cos.data(), DType::FLOAT32);
        LogicalId sin_node = g.constant({1, seq_len, 1, cfg.rope_head_dim}, freqs_sin.data(), DType::FLOAT32);

        LogicalId cos_exp = g.repeat(cos_node, h, 2);
        LogicalId sin_exp = g.repeat(sin_node, h, 2);

        int32_t st[] = {0, 0, 0, 0}, en1[] = {1, (int32_t)seq_len, (int32_t)h, (int32_t)cfg.rope_head_dim};
        int32_t steps[] = {1, 1, 1, 1};
        LogicalId rope_part =
            g.contiguous(g.slice(x, g.constant({4}, st, DType::INT32), g.constant({4}, en1, DType::INT32),
                                 g.constant({4}, steps, DType::INT32)));

        int32_t s_half1[] = {0, 0, 0, 0}, e_half1[] = {1, (int32_t)seq_len, (int32_t)h, (int32_t)cfg.rope_head_dim / 2};
        int32_t s_half2[] = {0, 0, 0, (int32_t)cfg.rope_head_dim / 2},
                e_half2[] = {1, (int32_t)seq_len, (int32_t)h, (int32_t)cfg.rope_head_dim};
        LogicalId half1 =
            g.contiguous(g.slice(rope_part, g.constant({4}, s_half1, DType::INT32),
                                 g.constant({4}, e_half1, DType::INT32), g.constant({4}, steps, DType::INT32)));
        LogicalId half2 =
            g.contiguous(g.slice(rope_part, g.constant({4}, s_half2, DType::INT32),
                                 g.constant({4}, e_half2, DType::INT32), g.constant({4}, steps, DType::INT32)));

        int32_t ax_3 = 3;
        LogicalId rotated = g.concat({g.neg(half2), half1}, g.constant({1}, &ax_3, DType::INT32));
        LogicalId applied = g.add(g.mul(rope_part, cos_exp), g.mul(rotated, sin_exp));

        if (cfg.rope_head_dim < head_dim)
        {
            int32_t s_pass[] = {0, 0, 0, (int32_t)cfg.rope_head_dim};
            int32_t e_pass[] = {1, (int32_t)seq_len, (int32_t)h, (int32_t)head_dim};
            LogicalId pass =
                g.contiguous(g.slice(x, g.constant({4}, s_pass, DType::INT32), g.constant({4}, e_pass, DType::INT32),
                                     g.constant({4}, steps, DType::INT32)));
            return g.concat({applied, pass}, g.constant({1}, &ax_3, DType::INT32));
        }
        return applied;
    }

    LogicalId apply_rope_to_compressed(LogicalId compressed, uint32_t S_r, uint32_t ratio, uint32_t comp_head_dim)
    {
        std::vector<float> freqs_cos(S_r * cfg.rope_head_dim);
        std::vector<float> freqs_sin(S_r * cfg.rope_head_dim);
        for (uint32_t s = 0; s < S_r; ++s)
        {
            for (uint32_t d = 0; d < cfg.rope_head_dim; d += 2)
            {
                float freq = 1.0f / std::pow(cfg.rope_theta, (float)d / cfg.rope_head_dim);
                float val = s * ratio * freq;
                freqs_cos[s * cfg.rope_head_dim + d] = std::cos(val);
                freqs_cos[s * cfg.rope_head_dim + d + 1] = std::cos(val);
                freqs_sin[s * cfg.rope_head_dim + d] = std::sin(val);
                freqs_sin[s * cfg.rope_head_dim + d + 1] = std::sin(val);
            }
        }

        LogicalId cos_node = g.constant({1, S_r, 1, cfg.rope_head_dim}, freqs_cos.data(), DType::FLOAT32);
        LogicalId sin_node = g.constant({1, S_r, 1, cfg.rope_head_dim}, freqs_sin.data(), DType::FLOAT32);

        int32_t st[] = {0, 0, 0}, en1[] = {1, (int32_t)S_r, (int32_t)cfg.rope_head_dim};
        int32_t steps[] = {1, 1, 1};
        LogicalId rope_part =
            g.contiguous(g.slice(compressed, g.constant({3}, st, DType::INT32), g.constant({3}, en1, DType::INT32),
                                 g.constant({3}, steps, DType::INT32)));
        int32_t sh4_r[] = {1, (int32_t)S_r, 1, (int32_t)cfg.rope_head_dim};
        rope_part = g.reshape(rope_part, g.constant({4}, sh4_r, DType::INT32));

        int32_t s_half1[] = {0, 0, 0, 0},
                e_half1[] = {1, (int32_t)S_r, 1, (int32_t)cfg.rope_head_dim / 2};
        int32_t s_half2[] = {0, 0, 0, (int32_t)cfg.rope_head_dim / 2},
                e_half2[] = {1, (int32_t)S_r, 1, (int32_t)cfg.rope_head_dim};
        int32_t steps4[] = {1, 1, 1, 1};
        LogicalId half1 =
            g.contiguous(g.slice(rope_part, g.constant({4}, s_half1, DType::INT32),
                                 g.constant({4}, e_half1, DType::INT32), g.constant({4}, steps4, DType::INT32)));
        LogicalId half2 =
            g.contiguous(g.slice(rope_part, g.constant({4}, s_half2, DType::INT32),
                                 g.constant({4}, e_half2, DType::INT32), g.constant({4}, steps4, DType::INT32)));

        int32_t ax_3 = 3;
        LogicalId rotated = g.concat({g.neg(half2), half1}, g.constant({1}, &ax_3, DType::INT32));
        LogicalId applied = g.add(g.mul(rope_part, cos_node), g.mul(rotated, sin_node));
        int32_t sh3_a[] = {1, (int32_t)S_r, (int32_t)cfg.rope_head_dim};
        applied = g.reshape(applied, g.constant({3}, sh3_a, DType::INT32));

        if (cfg.rope_head_dim < comp_head_dim)
        {
            int32_t s_pass[] = {0, 0, (int32_t)cfg.rope_head_dim};
            int32_t e_pass[] = {1, (int32_t)S_r, (int32_t)comp_head_dim};
            LogicalId pass =
                g.contiguous(g.slice(compressed, g.constant({3}, s_pass, DType::INT32),
                                     g.constant({3}, e_pass, DType::INT32), g.constant({3}, steps, DType::INT32)));
            int32_t ax_2 = 2;
            return g.concat({applied, pass}, g.constant({1}, &ax_2, DType::INT32));
        }
        return applied;
    }

    LogicalId Compressor(LogicalId x, int layer_idx, const std::string &prefix, uint32_t comp_head_dim)
    {
        uint32_t ratio = cfg.compress_ratios[layer_idx];
        if (ratio == 0)
            return LogicalId{UINT32_MAX};

        uint32_t S_r = seq_len / ratio;

        if (ratio == 4)
        {
            uint32_t comp_dim = 2 * comp_head_dim;
            LogicalId kv = linear(x, prefix + "wkv.weight", cfg.dim, comp_dim);
            LogicalId score = linear(x, prefix + "wgate.weight", cfg.dim, comp_dim);
            LogicalId ape = weight(prefix + "ape");

            int32_t sh4[] = {1, (int32_t)S_r, (int32_t)ratio, (int32_t)comp_dim};
            LogicalId kv_unflat = g.reshape(kv, g.constant({4}, sh4, DType::INT32));
            LogicalId score_unflat = g.reshape(score, g.constant({4}, sh4, DType::INT32));
            int32_t sh4_ape[] = {1, 1, (int32_t)ratio, (int32_t)comp_dim};
            LogicalId ape_exp = g.repeat(g.reshape(ape, g.constant({4}, sh4_ape, DType::INT32)), S_r, 1);
            score_unflat = g.add(score_unflat, ape_exp);

            auto overlap_transform = [&](LogicalId t, float pad_val)
            {
                int32_t s0[] = {0, 0, 0, (int32_t)comp_head_dim};
                int32_t e0[] = {1, (int32_t)S_r, (int32_t)ratio, (int32_t)comp_dim};
                int32_t st[] = {1, 1, 1, 1};
                LogicalId curr = g.slice(t, g.constant({4}, s0, DType::INT32), g.constant({4}, e0, DType::INT32),
                                         g.constant({4}, st, DType::INT32));

                int32_t s1[] = {0, 0, 0, 0};
                int32_t e1[] = {1, (int32_t)(S_r - 1), (int32_t)ratio, (int32_t)comp_head_dim};
                LogicalId prev = g.slice(t, g.constant({4}, s1, DType::INT32), g.constant({4}, e1, DType::INT32),
                                         g.constant({4}, st, DType::INT32));

                int32_t pad_sh[] = {1, 1, (int32_t)ratio, (int32_t)comp_head_dim};
                LogicalId pad = g.fill(g.constant({1}, &pad_val, DType::FLOAT32), g.constant({4}, pad_sh, DType::INT32));

                int32_t ax1 = 1, ax2 = 2;
                LogicalId prev_padded = g.concat({pad, prev}, g.constant({1}, &ax1, DType::INT32));
                return g.concat({prev_padded, curr}, g.constant({1}, &ax2, DType::INT32));
            };

            kv_unflat = overlap_transform(kv_unflat, 0.0f);
            score_unflat = overlap_transform(score_unflat, -1e9f);

            int32_t ax2_val = 2;
            LogicalId max_s = g.repeat(g.max(score_unflat, g.constant({1}, &ax2_val, DType::INT32)), 2 * ratio, 2);
            LogicalId exps = g.pow(g.fill(TGConstants::E, {1, S_r, 2 * ratio, comp_head_dim}),
                                   g.add(score_unflat, g.neg(max_s)));
            LogicalId sum_exps = g.repeat(g.sum(exps, g.constant({1}, &ax2_val, DType::INT32)), 2 * ratio, 2);
            LogicalId probs = g.div(exps, sum_exps);

            LogicalId compressed =
                g.sum(g.mul(kv_unflat, probs), g.constant({1}, &ax2_val, DType::INT32));
            int32_t sh3_comp[] = {1, (int32_t)S_r, (int32_t)comp_head_dim};
            compressed = g.reshape(compressed, g.constant({3}, sh3_comp, DType::INT32));

            compressed = rms_norm(compressed, weight(prefix + "norm.weight"), 1, comp_head_dim, S_r);

            return apply_rope_to_compressed(compressed, S_r, ratio, comp_head_dim);
        }
        else // ratio == 128
        {
            uint32_t comp_dim = comp_head_dim;
            LogicalId kv = linear(x, prefix + "wkv.weight", cfg.dim, comp_dim);
            LogicalId score = linear(x, prefix + "wgate.weight", cfg.dim, comp_dim);
            LogicalId ape = weight(prefix + "ape");

            int32_t sh4[] = {1, (int32_t)S_r, (int32_t)ratio, (int32_t)comp_dim};
            LogicalId kv_unflat = g.reshape(kv, g.constant({4}, sh4, DType::INT32));
            LogicalId score_unflat = g.reshape(score, g.constant({4}, sh4, DType::INT32));
            int32_t sh4_ape_4d[] = {1, 1, (int32_t)ratio, (int32_t)comp_dim};
            LogicalId ape_exp = g.repeat(g.reshape(ape, g.constant({4}, sh4_ape_4d, DType::INT32)), S_r, 1);
            score_unflat = g.add(score_unflat, ape_exp);

            int32_t ax2_val = 2;
            LogicalId max_s = g.repeat(g.max(score_unflat, g.constant({1}, &ax2_val, DType::INT32)), ratio, 2);
            LogicalId exps = g.pow(g.fill(TGConstants::E, {1, S_r, ratio, comp_dim}),
                                   g.add(score_unflat, g.neg(max_s)));
            LogicalId sum_exps = g.repeat(g.sum(exps, g.constant({1}, &ax2_val, DType::INT32)), ratio, 2);
            LogicalId probs = g.div(exps, sum_exps);

            LogicalId compressed =
                g.sum(g.mul(kv_unflat, probs), g.constant({1}, &ax2_val, DType::INT32));
            int32_t sh3_comp[] = {1, (int32_t)S_r, (int32_t)comp_dim};
            compressed = g.reshape(compressed, g.constant({3}, sh3_comp, DType::INT32));

            compressed = rms_norm(compressed, weight(prefix + "norm.weight"), 1, comp_head_dim, S_r);

            return apply_rope_to_compressed(compressed, S_r, ratio, comp_head_dim);
        }
    }

    LogicalId Indexer(LogicalId x, LogicalId qr, int layer_idx, LogicalId compressed_kv)
    {
        std::string prefix = "layers." + std::to_string(layer_idx) + ".attn.indexer.";
        LogicalId q = linear(qr, prefix + "wq_b.weight", cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim);
        int32_t sh4_q[] = {1, (int32_t)seq_len, (int32_t)cfg.index_n_heads, (int32_t)cfg.index_head_dim};
        q = g.reshape(q, g.constant({4}, sh4_q, DType::INT32));
        q = apply_rope(q, cfg.index_n_heads, cfg.index_head_dim); // rope on indexer Q

        uint32_t S_r = seq_len / cfg.compress_ratios[layer_idx];

        // Reshape Q into a 3D tensor: [1, seq_len * index_n_heads, index_head_dim]
        int32_t sh3_q[] = {1, (int32_t)(seq_len * cfg.index_n_heads), (int32_t)cfg.index_head_dim};
        LogicalId q_3d = g.reshape(q, g.constant({3}, sh3_q, DType::INT32));

        // Reshape KV into a 3D tensor: [1, S_r, index_head_dim]
        int32_t sh3_kv[] = {1, (int32_t)S_r, (int32_t)cfg.index_head_dim};
        LogicalId kv_3d = g.reshape(compressed_kv, g.constant({3}, sh3_kv, DType::INT32));

        // Permute 3D KV last two dims: [1, S_r, index_head_dim] -> [1, index_head_dim, S_r]
        int32_t permute_order[] = {0, 2, 1};
        LogicalId kv_3d_t = g.contiguous(g.permute(kv_3d, g.constant({3}, permute_order, DType::INT32)));

        // 3D Batched Dot: [1, seq_len * index_n_heads, 128] x [1, 128, S_r] -> [1, seq_len * index_n_heads, S_r]
        LogicalId scores_3d = g.dot(q_3d, kv_3d_t);

        // Reshape scores back to 4D: [1, seq_len, index_n_heads, S_r]
        int32_t sh4_s[] = {1, (int32_t)seq_len, (int32_t)cfg.index_n_heads, (int32_t)S_r};
        LogicalId scores = g.reshape(scores_3d, g.constant({4}, sh4_s, DType::INT32));

        LogicalId weights = linear(x, prefix + "weights_proj.weight", cfg.dim, cfg.index_n_heads);

        LogicalId relu_scores = g.relu(scores, {1, seq_len, cfg.index_n_heads, S_r});
        int32_t sh4_w[] = {1, (int32_t)seq_len, (int32_t)cfg.index_n_heads, 1};
        LogicalId weights_exp = g.repeat(g.reshape(weights, g.constant({4}, sh4_w, DType::INT32)), S_r, 3);

        LogicalId weighted_scores = g.mul(relu_scores, weights_exp);
        int32_t ax_2 = 2;
        int32_t sh3_f[] = {1, (int32_t)seq_len, (int32_t)S_r};
        LogicalId final_scores = g.reshape(g.sum(weighted_scores, g.constant({1}, &ax_2, DType::INT32)),
                                           g.constant({3}, sh3_f, DType::INT32));

        std::vector<float> mask_data(seq_len * S_r);
        for (uint32_t s = 0; s < seq_len; s++)
        {
            for (uint32_t t = 0; t < S_r; t++)
            {
                mask_data[s * S_r + t] = (t >= (s + 1) / cfg.compress_ratios[layer_idx]) ? -1e9f : 0.0f;
            }
        }
        LogicalId mask = g.constant({1, seq_len, S_r}, mask_data.data(), DType::FLOAT32);
        final_scores = g.add(final_scores, mask);

        int32_t ax_last = -1;
        int32_t topk = cfg.index_topk;
        return g.argmax(final_scores, g.constant({1}, &ax_last, DType::INT32), g.constant({1}, &topk, DType::INT32));
    }

    LogicalId sparse_attn(LogicalId q, LogicalId full_kv, LogicalId topk_idxs, uint32_t total_kv_len,
                          uint32_t topk_total)
    {
        int32_t sh3_q[] = {(int32_t)(seq_len * cfg.n_heads), 1, (int32_t)cfg.head_dim};
        LogicalId q_3d = g.reshape(q, g.constant({3}, sh3_q, DType::INT32));

        int32_t sh2_kv[] = {(int32_t)total_kv_len, (int32_t)cfg.head_dim};
        LogicalId flat_kv = g.reshape(full_kv, g.constant({2}, sh2_kv, DType::INT32));
        int32_t sh1_idx[] = {(int32_t)(seq_len * topk_total)};
        LogicalId flat_idxs = g.reshape(topk_idxs, g.constant({1}, sh1_idx, DType::INT32));
        LogicalId gathered_kv = g.gather(flat_kv, flat_idxs);

        int32_t sh4_g[] = {(int32_t)seq_len, 1, (int32_t)topk_total, (int32_t)cfg.head_dim};
        int32_t sh3_rep[] = {(int32_t)(seq_len * cfg.n_heads), (int32_t)topk_total, (int32_t)cfg.head_dim};
        LogicalId kv_rep =
            g.reshape(g.repeat(g.reshape(gathered_kv, g.constant({4}, sh4_g, DType::INT32)), cfg.n_heads, 1),
                      g.constant({3}, sh3_rep, DType::INT32));

        int32_t p[] = {0, 2, 1};
        LogicalId scores =
            g.dot(q_3d, g.contiguous(g.permute(kv_rep, g.constant({3}, p, DType::INT32)))); // [S*h, 1, topk_total]

        float scale = 1.0f / std::sqrt((float)cfg.head_dim);
        scores = g.mul(scores, g.fill(scale, {seq_len * cfg.n_heads, 1, topk_total}));

        int32_t ax_last = -1;
        LogicalId max_s = g.repeat(g.max(scores, g.constant({1}, &ax_last, DType::INT32)), topk_total, 2);
        float e_val = TGConstants::E;
        LogicalId exps =
            g.pow(g.fill(e_val, {seq_len * cfg.n_heads, 1, topk_total}),
                  g.add(scores, g.neg(max_s)));
        LogicalId sum_exps = g.repeat(g.sum(exps, g.constant({1}, &ax_last, DType::INT32)), topk_total, 2);
        LogicalId probs = g.div(exps, sum_exps);

        int32_t st_v[] = {0, 0, 0};
        int32_t en_v[] = {(int32_t)(seq_len * cfg.n_heads), (int32_t)topk_total, (int32_t)cfg.v_head_dim};
        int32_t steps_v[] = {1, 1, 1};
        LogicalId kv_val = g.slice(kv_rep, g.constant({3}, st_v, DType::INT32), g.constant({3}, en_v, DType::INT32),
                                   g.constant({3}, steps_v, DType::INT32));

        LogicalId out = g.dot(probs, kv_val);
        int32_t sh4_out[] = {1, (int32_t)seq_len, (int32_t)cfg.n_heads, (int32_t)cfg.v_head_dim};
        return g.reshape(out, g.constant({4}, sh4_out, DType::INT32));
    }

    LogicalId build_graph(LogicalId input_ids_id)
    {
        LogicalId w_emb = weight("embed.weight");
        LogicalId h = g.gather(w_emb, input_ids_id);

        int32_t sh4_h[] = {1, (int32_t)seq_len, 1, (int32_t)cfg.dim};
        h = g.repeat(g.reshape(h, g.constant({4}, sh4_h, DType::INT32)), cfg.hc_mult, 2); // [1, S, hc_mult, dim]

        for (uint32_t i = 0; i < cfg.n_layers; ++i)
        {
            std::string prefix = "layers." + std::to_string(i) + ".";
            LogicalId residual = h;

            auto [x_attn, post_attn, comb_attn] = hc_pre(h, prefix + "hc_attn_");
            x_attn = rms_norm(x_attn, weight(prefix + "attn_norm.weight"), 1, cfg.dim);

            LogicalId qr = linear(x_attn, prefix + "attn.wq_a.weight", cfg.dim, cfg.q_lora_rank);
            qr = rms_norm(qr, weight(prefix + "attn.q_norm.weight"), 1, cfg.q_lora_rank);
            LogicalId q = linear(qr, prefix + "attn.wq_b.weight", cfg.q_lora_rank, cfg.n_heads * cfg.head_dim);
            int32_t sh4_q[] = {1, (int32_t)seq_len, (int32_t)cfg.n_heads, (int32_t)cfg.head_dim};
            q = g.reshape(q, g.constant({4}, sh4_q, DType::INT32));
            q = apply_rope(q, cfg.n_heads, cfg.head_dim);

            LogicalId kv = linear(x_attn, prefix + "attn.wkv.weight", cfg.dim, cfg.head_dim);
            kv = rms_norm(kv, weight(prefix + "attn.kv_norm.weight"), 1, cfg.head_dim);
            int32_t sh4_kv[] = {1, (int32_t)seq_len, 1, (int32_t)cfg.head_dim};
            kv = g.reshape(kv, g.constant({4}, sh4_kv, DType::INT32));
            kv = apply_rope(kv, 1, cfg.head_dim);
            int32_t sh3_kv[] = {1, (int32_t)seq_len, (int32_t)cfg.head_dim};
            kv = g.reshape(kv, g.constant({3}, sh3_kv, DType::INT32));

            std::vector<int32_t> win_idxs_data(seq_len * cfg.window_size);
            for (uint32_t s = 0; s < seq_len; s++)
            {
                for (uint32_t w = 0; w < cfg.window_size; w++)
                {
                    int32_t t = (int32_t)s - (int32_t)cfg.window_size + 1 + (int32_t)w;
                    win_idxs_data[s * cfg.window_size + w] = (t < 0) ? -1 : t;
                }
            }
            LogicalId win_idxs = g.constant({1, seq_len, cfg.window_size}, win_idxs_data.data(), DType::INT32);

            LogicalId all_topk_idxs = win_idxs;
            uint32_t topk_total = cfg.window_size;
            LogicalId full_kv = kv;
            uint32_t total_kv_len = seq_len;

            if (cfg.compress_ratios[i] == 4)
            {
                std::string comp_prefix = "layers." + std::to_string(i) + ".attn.compressor.";
                std::string idx_comp_prefix = "layers." + std::to_string(i) + ".attn.indexer.compressor.";

                LogicalId compressed_kv = Compressor(x_attn, i, comp_prefix, cfg.head_dim);
                LogicalId indexer_compressed_kv = Compressor(x_attn, i, idx_comp_prefix, cfg.index_head_dim);

                LogicalId compress_topk_idxs = Indexer(x_attn, qr, i, indexer_compressed_kv);

                int32_t offset_val = seq_len;
                LogicalId offset =
                    g.fill(g.constant({1}, &offset_val, DType::INT32), {1, seq_len, cfg.index_topk});
                LogicalId shifted_idxs = g.add(compress_topk_idxs, offset);

                int32_t neg_one = -1;
                LogicalId neg_one_node =
                    g.fill(g.constant({1}, &neg_one, DType::INT32), {1, seq_len, cfg.index_topk});
                LogicalId is_neg_one = g.eq(compress_topk_idxs, neg_one_node);

                LogicalId is_neg_one_f = g.cast(is_neg_one, DType::FLOAT32);
                LogicalId not_neg_one_f =
                    g.add(g.fill(one_fp32, {1, seq_len, cfg.index_topk}), g.neg(is_neg_one_f));

                LogicalId final_shifted = g.cast(g.add(g.mul(is_neg_one_f, g.cast(neg_one_node, DType::FLOAT32)),
                                                       g.mul(not_neg_one_f, g.cast(shifted_idxs, DType::FLOAT32))),
                                                 DType::INT32);

                int32_t ax_2 = 2;
                all_topk_idxs = g.concat({win_idxs, final_shifted}, g.constant({1}, &ax_2, DType::INT32));
                topk_total += cfg.index_topk;

                int32_t ax_1 = 1;
                full_kv = g.concat({kv, compressed_kv}, g.constant({1}, &ax_1, DType::INT32));
                total_kv_len += seq_len / 4;
            }
            else if (cfg.compress_ratios[i] == 128)
            {
                std::string comp_prefix = "layers." + std::to_string(i) + ".attn.compressor.";
                LogicalId compressed_kv = Compressor(x_attn, i, comp_prefix, cfg.head_dim);

                uint32_t S_r = seq_len / 128;
                std::vector<int32_t> comp_idxs_data(seq_len * S_r);
                for (uint32_t s = 0; s < seq_len; s++)
                {
                    for (uint32_t t = 0; t < S_r; t++)
                    {
                        comp_idxs_data[s * S_r + t] = (t < (s + 1) / 128) ? (int32_t)(seq_len + t) : -1;
                    }
                }
                LogicalId comp_idxs = g.constant({1, seq_len, S_r}, comp_idxs_data.data(), DType::INT32);

                int32_t ax_2 = 2;
                all_topk_idxs = g.concat({win_idxs, comp_idxs}, g.constant({1}, &ax_2, DType::INT32));
                topk_total += S_r;

                int32_t ax_1 = 1;
                full_kv = g.concat({kv, compressed_kv}, g.constant({1}, &ax_1, DType::INT32));
                total_kv_len += S_r;
            }

            LogicalId o = sparse_attn(q, full_kv, all_topk_idxs, total_kv_len, topk_total);

            int32_t sh3_o[] = {1, (int32_t)seq_len, (int32_t)(cfg.n_heads * cfg.v_head_dim)};
            o = g.reshape(o, g.constant({3}, sh3_o, DType::INT32));
            o = linear(o, prefix + "attn.wo_a.weight", cfg.n_heads * cfg.v_head_dim, cfg.o_groups * cfg.o_lora_rank);
            o = linear(o, prefix + "attn.wo_b.weight", cfg.o_groups * cfg.o_lora_rank, cfg.dim);

            h = hc_post(o, residual, post_attn, comb_attn);
            residual = h;

            auto [x_ffn, post_ffn, comb_ffn] = hc_pre(h, prefix + "hc_ffn_");
            x_ffn = rms_norm(x_ffn, weight(prefix + "ffn_norm.weight"), 1, cfg.dim);

            LogicalId gate_weight = weight(prefix + "ffn.gate.weight");
            int32_t permute_order[] = {1, 0};
            LogicalId router_logits =
                g.dot(x_ffn, g.contiguous(g.permute(gate_weight, g.constant({2}, permute_order, DType::INT32))));

            float eps_val = 1e-6f;
            LogicalId route_eps = g.fill(eps_val, {1, seq_len, cfg.n_routed_experts});
            float half_val = 0.5f;
            LogicalId route_half = g.fill(half_val, {1, seq_len, cfg.n_routed_experts});
            LogicalId router_probs =
                g.pow(g.add(router_logits, g.pow(g.add(g.mul(router_logits, router_logits), route_eps), route_half)),
                      route_half);

            int32_t ax_last = -1;
            int32_t topk_exp = cfg.n_activated_experts;
            LogicalId top_k_idxs = g.argmax(router_probs, g.constant({1}, &ax_last, DType::INT32),
                                            g.constant({1}, &topk_exp, DType::INT32));

            // Shared experts
            LogicalId shared_out = linear(x_ffn, prefix + "ffn.shared_experts.w1.weight", cfg.dim, cfg.moe_inter_dim);

            LogicalId sig_one = g.fill(one_fp32, {1, seq_len, cfg.moe_inter_dim});
            float neg_one_val = -1.0f;
            LogicalId sig_neg_one = g.fill(neg_one_val, {1, seq_len, cfg.moe_inter_dim});
            float e_val = TGConstants::E;
            LogicalId sig_e = g.fill(e_val, {1, seq_len, cfg.moe_inter_dim});
            LogicalId shared_sig = g.div(sig_one, g.add(sig_one, g.pow(sig_e, g.mul(shared_out, sig_neg_one))));
            shared_out = g.mul(shared_out, shared_sig);

            shared_out =
                g.mul(shared_out, linear(x_ffn, prefix + "ffn.shared_experts.w3.weight", cfg.dim, cfg.moe_inter_dim));
            shared_out = linear(shared_out, prefix + "ffn.shared_experts.w2.weight", cfg.moe_inter_dim, cfg.dim);

            h = hc_post(shared_out, residual, post_ffn, comb_ffn);
        }

        // Final HC head
        uint32_t hc_dim = cfg.hc_mult * cfg.dim;
        int32_t sh3_final[] = {1, (int32_t)seq_len, (int32_t)hc_dim};
        LogicalId x_flat = g.reshape(h, g.constant({3}, sh3_final, DType::INT32));

        // 1. Linear projection for mixes -> [1, seq_len, hc_mult]
        LogicalId hc_head_fn = weight("hc_head_fn");
        int32_t permute_order[] = {1, 0};
        LogicalId hc_head_fn_t = g.contiguous(g.permute(hc_head_fn, g.constant({2}, permute_order, DType::INT32)));
        int32_t sh3_fn[] = {1, (int32_t)hc_dim, (int32_t)cfg.hc_mult};
        LogicalId mixes = g.dot(x_flat, g.reshape(hc_head_fn_t, g.constant({3}, sh3_fn, DType::INT32)));

        // 2. Compute rsqrt over x_flat (size = hc_mult * dim) -> [1, seq_len, 1]
        LogicalId x_sq = g.mul(x_flat, x_flat);
        int32_t ax_last = -1;
        LogicalId sum_sq = g.sum(x_sq, g.constant({1}, &ax_last, DType::INT32));
        LogicalId mean_sq = g.div(sum_sq, g.fill((float)hc_dim, {1, seq_len, 1}));
        LogicalId std = g.pow(g.add(mean_sq, g.fill(cfg.norm_eps, {1, seq_len, 1})), g.fill(0.5f, {1, seq_len, 1}));
        LogicalId rsqrt = g.div(g.fill(1.0f, {1, seq_len, 1}), std);

        // 3. Repeat rsqrt across hc_mult (axis 2) -> [1, seq_len, hc_mult]
        LogicalId rsqrt_rep = g.repeat(rsqrt, cfg.hc_mult, 2);
        mixes = g.mul(mixes, rsqrt_rep);

        // 4. Pre scaling and sigmoid
        LogicalId scale = g.fill(weight("hc_head_scale"), {1, seq_len, cfg.hc_mult});
        int32_t sh3_base[] = {1, 1, (int32_t)cfg.hc_mult};
        LogicalId base = g.repeat(g.reshape(weight("hc_head_base"), g.constant({3}, sh3_base, DType::INT32)), seq_len, 1);

        LogicalId sig_one = g.fill(1.0f, {1, seq_len, cfg.hc_mult});
        float neg_one_val = -1.0f;
        LogicalId sig_neg_one = g.fill(neg_one_val, {1, seq_len, cfg.hc_mult});
        float e_val = TGConstants::E;
        LogicalId sig_e = g.fill(e_val, {1, seq_len, cfg.hc_mult});
        LogicalId pre = g.add(g.div(sig_one, g.add(sig_one, g.pow(sig_e, g.mul(g.add(g.mul(mixes, scale), base), sig_neg_one)))),
                              g.fill(cfg.hc_eps, {1, seq_len, cfg.hc_mult}));

        // 5. Apply pre to h and sum across hc_mult
        int32_t sh4_pre[] = {1, (int32_t)seq_len, (int32_t)cfg.hc_mult, 1};
        LogicalId pre_exp = g.repeat(g.reshape(pre, g.constant({4}, sh4_pre, DType::INT32)), cfg.dim, 3);
        LogicalId y_4d = g.mul(h, pre_exp);
        int32_t ax_2 = 2;
        LogicalId y = g.sum(y_4d, g.constant({1}, &ax_2, DType::INT32));

        int32_t sh3_y[] = {1, (int32_t)seq_len, (int32_t)cfg.dim};
        y = g.reshape(y, g.constant({3}, sh3_y, DType::INT32));
        y = rms_norm(y, weight("norm.weight"), 1, cfg.dim);

        LogicalId w_lm = weight("head.weight");
        int32_t p[] = {1, 0};
        int32_t sh3_lm[] = {1, (int32_t)cfg.dim, (int32_t)cfg.vocab_size};
        LogicalId w_lm_3d = g.reshape(g.contiguous(g.permute(w_lm, g.constant({2}, p, DType::INT32))),
                                      g.constant({3}, sh3_lm, DType::INT32));

        return g.dot(y, w_lm_3d);
    }
};