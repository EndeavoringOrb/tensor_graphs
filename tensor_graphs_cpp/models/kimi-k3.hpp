// tensor_graphs_cpp/models/kimi-k3.hpp
#pragma once
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "core/common/constants.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/types.hpp"

struct KimiK3Config
{
    uint32_t vocab_size = 163840;
    uint32_t hidden_size = 7168;
    uint32_t intermediate_size = 33792;
    uint32_t num_hidden_layers = 93;
    uint32_t num_attention_heads = 96;
    uint32_t num_key_value_heads = 96;

    // MLA
    uint32_t q_lora_rank = 1536;
    uint32_t kv_lora_rank = 512;
    uint32_t qk_nope_head_dim = 128;
    uint32_t qk_rope_head_dim = 64;
    uint32_t v_head_dim = 128;
    bool mla_use_output_gate = false;

    // KDA
    uint32_t linear_head_dim = 128;
    uint32_t linear_num_heads = 96;
    uint32_t linear_conv_size = 4;

    // MoE
    uint32_t num_experts = 896;
    uint32_t num_experts_per_token = 16;
    uint32_t num_shared_experts = 2;
    uint32_t moe_intermediate_size = 3072;
    uint32_t routed_expert_hidden_size = 3584;

    // Residuals
    uint32_t attn_res_block_size = 12;

    float rms_norm_eps = 1e-5f;
    float activation_situ_beta = 4.0f;
    float activation_situ_linear_beta = 25.0f;

    // Vision Config
    uint32_t patch_size = 14;
    uint32_t vt_num_attention_heads = 12;
    uint32_t vt_num_hidden_layers = 27;
    uint32_t vt_hidden_size = 1024;
    uint32_t vt_intermediate_size = 4096;
    uint32_t mm_hidden_size = 1024;
    uint32_t pos_emb_height = 64;
    uint32_t pos_emb_width = 64;
    uint32_t pos_emb_time = 4;

    bool is_kda_layer(uint32_t layer_idx)
    {
        // layer_idx is 0-indexed. KDA layers are indices where (layer_idx + 1) in [1, 2, 3, 5, 6, 7...]
        return (layer_idx % 4) != 3;
    }
};

class KimiK3Model
{
  private:
    KimiK3Config cfg;
    Graph &g;
    MemoryManager &mem;
    const std::string w_path;
    uint32_t seq_len;

    LogicalId one_fp32;
    LogicalId zero_fp32;
    LogicalId neg_one_fp32;
    LogicalId two_fp32;
    LogicalId half_fp32;
    LogicalId e_fp32;
    LogicalId neg_1e9_fp32;

  public:
    KimiK3Model(KimiK3Config config, uint32_t sequence_length, Graph &graph, MemoryManager &memory,
                const std::string &weight_path)
        : cfg(config), g(graph), mem(memory), w_path(weight_path), seq_len(sequence_length)
    {
        float one_val = 1.0f;
        float zero_val = 0.0f;
        float neg_one_val = -1.0f;
        float two_val = 2.0f;
        float half_val = 0.5f;
        float e_val = TGConstants::E;
        float neg_1e9_val = TGConstants::NEG_INF;

        one_fp32 = g.constant({1}, &one_val, DType::FLOAT32);
        zero_fp32 = g.constant({1}, &zero_val, DType::FLOAT32);
        neg_one_fp32 = g.constant({1}, &neg_one_val, DType::FLOAT32);
        two_fp32 = g.constant({1}, &two_val, DType::FLOAT32);
        half_fp32 = g.constant({1}, &half_val, DType::FLOAT32);
        e_fp32 = g.constant({1}, &e_val, DType::FLOAT32);
        neg_1e9_fp32 = g.constant({1}, &neg_1e9_val, DType::FLOAT32);
    }

    LogicalId weight(const std::string &name)
    {
        LogicalId raw_weight = g.weight(w_path, name);
        return g.cast(raw_weight, DType::FLOAT32);
    }

    LogicalId sigmoid(LogicalId x)
    {
        LogicalId neg_x = g.mul(x, g.fill(neg_one_fp32, g.getNode(x).getShape()));
        LogicalId exp_neg_x = g.pow(g.fill(e_fp32, g.getNode(x).getShape()), neg_x);
        LogicalId den = g.add(g.fill(one_fp32, g.getNode(x).getShape()), exp_neg_x);
        return g.div(g.fill(one_fp32, g.getNode(x).getShape()), den);
    }

    LogicalId tanh_atomic(LogicalId x)
    {
        LogicalId two_x = g.mul(x, g.fill(two_fp32, g.getNode(x).getShape()));
        LogicalId exp_2x = g.pow(g.fill(e_fp32, g.getNode(x).getShape()), two_x);
        LogicalId num = g.add(exp_2x, g.fill(neg_one_fp32, g.getNode(x).getShape()));
        LogicalId den = g.add(exp_2x, g.fill(one_fp32, g.getNode(x).getShape()));
        return g.div(num, den);
    }

    LogicalId softplus(LogicalId x)
    {
        LogicalId exp_x = g.pow(g.fill(e_fp32, g.getNode(x).getShape()), x);
        LogicalId one_plus_exp = g.add(g.fill(one_fp32, g.getNode(x).getShape()), exp_x);
        return g.log(one_plus_exp);
    }

    LogicalId situ(LogicalId gate, LogicalId up)
    {
        LogicalId beta = g.fill(cfg.activation_situ_beta, g.getNode(gate).getShape());
        LogicalId gate_div = g.div(gate, beta);
        LogicalId gate_tanh = tanh_atomic(gate_div);
        LogicalId situ_a = g.mul(g.mul(beta, gate_tanh), sigmoid(gate));
        LogicalId linear_beta = g.fill(cfg.activation_situ_linear_beta, g.getNode(up).getShape());
        LogicalId up_div = g.div(up, linear_beta);
        LogicalId up_tanh = tanh_atomic(up_div);
        LogicalId up_mapped = g.mul(linear_beta, up_tanh);
        return g.mul(situ_a, up_mapped);
    }

    LogicalId rms_norm(LogicalId x, const std::string &w_name, float eps, uint32_t D)
    {
        LogicalId sq = g.mul(x, x);
        int32_t ax = -1;
        LogicalId sum_sq = g.sum(sq, g.constant({1}, &ax, DType::INT32));
        LogicalId mean_sq = g.div(sum_sq, g.fill((float)D, g.getNode(sum_sq).getShape()));
        LogicalId var = g.add(mean_sq, g.fill(eps, g.getNode(mean_sq).getShape()));
        LogicalId std = g.pow(var, g.fill(half_fp32, g.getNode(var).getShape()));
        LogicalId inv_std = g.div(g.fill(one_fp32, g.getNode(std).getShape()), std);

        int32_t rep[] = {(int32_t)D};
        int32_t r_ax[] = {(int32_t)(g.getNode(inv_std).getShape().size() - 1)};
        LogicalId inv_std_exp =
            g.repeat(inv_std, g.constant({1}, rep, DType::INT32), g.constant({1}, r_ax, DType::INT32));

        LogicalId x_norm = g.mul(x, inv_std_exp);

        LogicalId w = weight(w_name);
        std::vector<uint32_t> b_shape = g.getNode(x).getShape();
        for (int i = 0; i < b_shape.size() - 1; i++)
            b_shape[i] = 1;
        LogicalId w_reshaped =
            g.reshape(w, g.constant({(uint32_t)b_shape.size()}, (int32_t *)b_shape.data(), DType::INT32));

        LogicalId w_exp = w_reshaped;
        for (int i = 0; i < b_shape.size() - 1; i++)
        {
            if (g.getNode(x).getShape()[i] > 1)
            {
                int32_t r[] = {(int32_t)g.getNode(x).getShape()[i]};
                int32_t a[] = {i};
                w_exp = g.repeat(w_exp, g.constant({1}, r, DType::INT32), g.constant({1}, a, DType::INT32));
            }
        }
        return g.mul(x_norm, w_exp);
    }

    LogicalId linear(LogicalId x, const std::string &w_name, uint32_t in_d, uint32_t out_d)
    {
        LogicalId w = weight(w_name);
        int32_t perm[] = {1, 0};
        LogicalId w_t = g.contiguous(g.permute(w, g.constant({2}, perm, DType::INT32)));

        std::vector<uint32_t> x_shape = g.getNode(x).getShape();
        std::vector<uint32_t> w_shape(x_shape.size(), 1);
        w_shape[w_shape.size() - 2] = in_d;
        w_shape[w_shape.size() - 1] = out_d;

        return g.dot(x,
                     g.reshape(w_t, g.constant({(uint32_t)w_shape.size()}, (int32_t *)w_shape.data(), DType::INT32)));
    }

    LogicalId apply_rope_2d(LogicalId x, LogicalId cos_node, LogicalId sin_node)
    {
        std::vector<uint32_t> sh = g.getNode(x).getShape();
        uint32_t dim = sh.back();

        std::vector<uint32_t> sh_split = sh;
        sh_split.back() = dim / 2;
        sh_split.push_back(2);
        LogicalId x_res =
            g.reshape(x, g.constant({(uint32_t)sh_split.size()}, (int32_t *)sh_split.data(), DType::INT32));

        int32_t st0[] = {0}, en0[] = {1}, step[] = {1};
        LogicalId x_real = g.slice(x_res, g.constant({1}, st0, DType::INT32), g.constant({1}, en0, DType::INT32),
                                   g.constant({1}, step, DType::INT32));
        int32_t st1[] = {1}, en1[] = {2};
        LogicalId x_imag = g.slice(x_res, g.constant({1}, st1, DType::INT32), g.constant({1}, en1, DType::INT32),
                                   g.constant({1}, step, DType::INT32));

        std::vector<uint32_t> trig_sh = sh;
        trig_sh.back() = dim / 2;
        trig_sh.push_back(1);
        LogicalId cos_exp =
            g.reshape(cos_node, g.constant({(uint32_t)trig_sh.size()}, (int32_t *)trig_sh.data(), DType::INT32));
        LogicalId sin_exp =
            g.reshape(sin_node, g.constant({(uint32_t)trig_sh.size()}, (int32_t *)trig_sh.data(), DType::INT32));

        LogicalId out_real = g.add(g.mul(x_real, cos_exp), g.neg(g.mul(x_imag, sin_exp)));
        LogicalId out_imag = g.add(g.mul(x_real, sin_exp), g.mul(x_imag, cos_exp));

        int32_t ax = -1;
        LogicalId out = g.concat({out_real, out_imag}, g.constant({1}, &ax, DType::INT32));
        return g.reshape(out, g.constant({(uint32_t)sh.size()}, (int32_t *)sh.data(), DType::INT32));
    }

    LogicalId compute_vision_pos_emb(uint32_t t, uint32_t h, uint32_t w)
    {
        uint32_t num_patches = t * h * w;
        std::vector<int32_t> idx00(h * w), idx01(h * w), idx10(h * w), idx11(h * w);
        std::vector<float> w00(h * w), w01(h * w), w10(h * w), w11(h * w);

        for (uint32_t yi = 0; yi < h; ++yi)
        {
            for (uint32_t xi = 0; xi < w; ++xi)
            {
                float src_y = (yi + 0.5f) * ((float)cfg.pos_emb_height / h) - 0.5f;
                float src_x = (xi + 0.5f) * ((float)cfg.pos_emb_width / w) - 0.5f;

                int y0 = std::max(0, (int)std::floor(src_y));
                int y1 = std::min((int)cfg.pos_emb_height - 1, y0 + 1);
                int x0 = std::max(0, (int)std::floor(src_x));
                int x1 = std::min((int)cfg.pos_emb_width - 1, x0 + 1);

                float dy = src_y - y0;
                float dx = src_x - x0;

                uint32_t pid = yi * w + xi;
                idx00[pid] = y0 * cfg.pos_emb_width + x0;
                idx01[pid] = y0 * cfg.pos_emb_width + x1;
                idx10[pid] = y1 * cfg.pos_emb_width + x0;
                idx11[pid] = y1 * cfg.pos_emb_width + x1;

                w00[pid] = (1.0f - dy) * (1.0f - dx);
                w01[pid] = (1.0f - dy) * dx;
                w10[pid] = dy * (1.0f - dx);
                w11[pid] = dy * dx;
            }
        }

        LogicalId idx00_node = g.constant({h * w}, idx00.data(), DType::INT32);
        LogicalId idx01_node = g.constant({h * w}, idx01.data(), DType::INT32);
        LogicalId idx10_node = g.constant({h * w}, idx10.data(), DType::INT32);
        LogicalId idx11_node = g.constant({h * w}, idx11.data(), DType::INT32);

        // TODO: The Python code uses bicubic by default. We are using bilinear approximation here.
        LogicalId pos_table = weight("vision_tower.patch_embed.pos_emb.weight");
        int32_t sh2_pos[] = {(int32_t)(cfg.pos_emb_height * cfg.pos_emb_width), (int32_t)cfg.vt_hidden_size};
        pos_table = g.reshape(pos_table, g.constant({2}, sh2_pos, DType::INT32));

        LogicalId g00 = g.gather(pos_table, idx00_node);
        LogicalId g01 = g.gather(pos_table, idx01_node);
        LogicalId g10 = g.gather(pos_table, idx10_node);
        LogicalId g11 = g.gather(pos_table, idx11_node);

        auto make_weight_2d = [&](const std::vector<float> &w_data) -> LogicalId {
            LogicalId w_1d = g.constant({h * w}, w_data.data(), DType::FLOAT32);
            int32_t sh2_w[] = {(int32_t)(h * w), 1};
            LogicalId w_2d = g.reshape(w_1d, g.constant({2}, sh2_w, DType::INT32));
            return g.repeat(w_2d, cfg.vt_hidden_size, 1);
        };

        LogicalId pos_2d = g.mul(g00, make_weight_2d(w00));
        pos_2d = g.add(pos_2d, g.mul(g01, make_weight_2d(w01)));
        pos_2d = g.add(pos_2d, g.mul(g10, make_weight_2d(w10)));
        pos_2d = g.add(pos_2d, g.mul(g11, make_weight_2d(w11)));

        int32_t sh3_2d[] = {1, (int32_t)(h * w), (int32_t)cfg.vt_hidden_size};
        LogicalId pos_3d = g.reshape(pos_2d, g.constant({3}, sh3_2d, DType::INT32));
        if (t > 1)
        {
            pos_3d = g.repeat(pos_3d, t, 0);
            LogicalId time_weight = weight("vision_tower.patch_embed.pos_emb.time_weight");
            int32_t t_st[] = {0, 0};
            int32_t t_en[] = {(int32_t)t, (int32_t)cfg.vt_hidden_size};
            int32_t t_step[] = {1, 1};
            int32_t sh_time[] = {(int32_t)cfg.pos_emb_time, (int32_t)cfg.vt_hidden_size};
            time_weight = g.reshape(time_weight, g.constant({2}, sh_time, DType::INT32));
            LogicalId time_slice = g.slice(time_weight, g.constant({2}, t_st, DType::INT32),
                                           g.constant({2}, t_en, DType::INT32), g.constant({2}, t_step, DType::INT32));

            int32_t sh_time_3d[] = {(int32_t)t, 1, (int32_t)cfg.vt_hidden_size};
            time_slice = g.reshape(time_slice, g.constant({3}, sh_time_3d, DType::INT32));
            LogicalId time_exp = g.repeat(time_slice, h * w, 1);
            pos_3d = g.add(pos_3d, time_exp);
        }
        int32_t final_sh[] = {(int32_t)(t * h * w), (int32_t)cfg.vt_hidden_size};
        return g.reshape(pos_3d, g.constant({2}, final_sh, DType::INT32));
    }

    std::tuple<LogicalId, LogicalId> get_vision_rope(uint32_t t, uint32_t h, uint32_t w)
    {
        uint32_t dim = cfg.vt_hidden_size / cfg.vt_num_attention_heads;
        uint32_t num_heads = cfg.vt_num_attention_heads;
        uint32_t c_over_4 = dim / 4;

        int32_t start0 = 0, stop_c4 = c_over_4 * 4, step4 = 4;
        LogicalId dim_range =
            g.cast(g.arange(g.constant({1}, &start0, DType::INT32), g.constant({1}, &stop_c4, DType::INT32),
                            g.constant({1}, &step4, DType::INT32)),
                   DType::FLOAT32);
        LogicalId dim_node = g.fill((float)dim, {c_over_4});
        LogicalId exp = g.div(dim_range, dim_node);
        LogicalId freqs =
            g.div(g.fill(one_fp32, (std::vector<uint32_t>){c_over_4}), g.pow(g.fill(10000.0f, {c_over_4}), exp));

        int32_t start_0 = 0, stop_h = h, stop_w = w, step_1 = 1;
        LogicalId y_pos =
            g.cast(g.arange(g.constant({1}, &start_0, DType::INT32), g.constant({1}, &stop_h, DType::INT32),
                            g.constant({1}, &step_1, DType::INT32)),
                   DType::FLOAT32);
        LogicalId x_pos =
            g.cast(g.arange(g.constant({1}, &start_0, DType::INT32), g.constant({1}, &stop_w, DType::INT32),
                            g.constant({1}, &step_1, DType::INT32)),
                   DType::FLOAT32);

        int32_t sh_h_1[] = {(int32_t)h, 1};
        LogicalId y_pos_col = g.reshape(y_pos, g.constant({2}, sh_h_1, DType::INT32));
        LogicalId freqs_1d =
            g.reshape(freqs, g.constant({2}, std::vector<int32_t>{1, (int32_t)c_over_4}.data(), DType::INT32));
        LogicalId y_freqs = g.mul(g.repeat(y_pos_col, c_over_4, 1), g.repeat(freqs_1d, h, 0));

        int32_t sh_w_1[] = {(int32_t)w, 1};
        LogicalId x_pos_col = g.reshape(x_pos, g.constant({2}, sh_w_1, DType::INT32));
        LogicalId x_freqs = g.mul(g.repeat(x_pos_col, c_over_4, 1), g.repeat(freqs_1d, w, 0));

        int32_t sh_h_1_c[] = {(int32_t)h, 1, (int32_t)c_over_4};
        LogicalId y_freqs_grid = g.repeat(g.reshape(y_freqs, g.constant({3}, sh_h_1_c, DType::INT32)), w, 1);

        int32_t sh_1_w_c[] = {1, (int32_t)w, (int32_t)c_over_4};
        LogicalId x_freqs_grid = g.repeat(g.reshape(x_freqs, g.constant({3}, sh_1_w_c, DType::INT32)), h, 0);

        int32_t sh_h_w_c_1[] = {(int32_t)h, (int32_t)w, (int32_t)c_over_4, 1};
        LogicalId x_fg_1 = g.reshape(x_freqs_grid, g.constant({4}, sh_h_w_c_1, DType::INT32));
        LogicalId y_fg_1 = g.reshape(y_freqs_grid, g.constant({4}, sh_h_w_c_1, DType::INT32));
        int32_t ax_neg1 = -1;
        LogicalId freqs_grid = g.concat({x_fg_1, y_fg_1}, g.constant({1}, &ax_neg1, DType::INT32));

        int32_t sh_hw_c2[] = {(int32_t)(h * w), (int32_t)(dim / 2)};
        LogicalId freqs_flat = g.reshape(freqs_grid, g.constant({2}, sh_hw_c2, DType::INT32));

        int32_t sh_1_hw_c2[] = {1, (int32_t)(h * w), (int32_t)(dim / 2)};
        LogicalId freqs_t = g.reshape(freqs_flat, g.constant({3}, sh_1_hw_c2, DType::INT32));
        freqs_t = g.repeat(freqs_t, t, 0);

        int32_t sh_thw_c2[] = {(int32_t)(t * h * w), (int32_t)(dim / 2)};
        LogicalId angles = g.reshape(freqs_t, g.constant({2}, sh_thw_c2, DType::INT32));

        int32_t sh_cos_sin[] = {1, (int32_t)(t * h * w), 1, (int32_t)(dim / 2)};
        LogicalId cos_node = g.reshape(g.cos(angles), g.constant({4}, sh_cos_sin, DType::INT32));
        LogicalId sin_node = g.reshape(g.sin(angles), g.constant({4}, sh_cos_sin, DType::INT32));

        cos_node = g.repeat(cos_node, num_heads, 2);
        sin_node = g.repeat(sin_node, num_heads, 2);

        return {cos_node, sin_node};
    }

    LogicalId build_vision_graph(uint32_t num_patches, uint32_t t, uint32_t h, uint32_t w)
    {
        LogicalId pixel_values = g.input({num_patches, 3, cfg.patch_size, cfg.patch_size}, DType::FLOAT32);

        // Patch embedding
        int32_t sh2[] = {(int32_t)num_patches, (int32_t)(3 * cfg.patch_size * cfg.patch_size)};
        LogicalId x = g.reshape(pixel_values, g.constant({2}, sh2, DType::INT32));
        LogicalId w_proj = weight("vision_tower.patch_embed.proj.weight");
        int32_t w_sh2[] = {(int32_t)cfg.vt_hidden_size, (int32_t)(3 * cfg.patch_size * cfg.patch_size)};
        LogicalId w_proj_2d = g.reshape(w_proj, g.constant({2}, w_sh2, DType::INT32));
        int32_t perm[] = {1, 0};
        LogicalId w_t = g.contiguous(g.permute(w_proj_2d, g.constant({2}, perm, DType::INT32)));
        x = g.dot(x, w_t); // (num_patches, hidden)

        LogicalId pos_emb = compute_vision_pos_emb(t, h, w);
        x = g.add(x, pos_emb);

        auto [cos_node, sin_node] = get_vision_rope(t, h, w);

        for (uint32_t l = 0; l < cfg.vt_num_hidden_layers; ++l)
        {
            std::string prefix = "vision_tower.encoder.blocks." + std::to_string(l) + ".";
            LogicalId residual = x;

            LogicalId norm = rms_norm(x, prefix + "norm0.weight", 1e-6f, cfg.vt_hidden_size);
            LogicalId qkv = linear(norm, prefix + "wqkv.weight", cfg.vt_hidden_size, 3 * cfg.vt_hidden_size);

            // Attn split & rope
            int32_t qkv_sh[] = {1, (int32_t)num_patches, 3, (int32_t)cfg.vt_num_attention_heads,
                                (int32_t)(cfg.vt_hidden_size / cfg.vt_num_attention_heads)};
            LogicalId qkv_5d = g.reshape(qkv, g.constant({5}, qkv_sh, DType::INT32));

            int32_t q_st[] = {0, 0, 0, 0, 0}, q_en[] = {1, (int32_t)num_patches, 1, (int32_t)cfg.vt_num_attention_heads,
                                                        (int32_t)(cfg.vt_hidden_size / cfg.vt_num_attention_heads)};
            int32_t st[] = {1, 1, 1, 1, 1};
            LogicalId q_t = g.slice(qkv_5d, g.constant({5}, q_st, DType::INT32), g.constant({5}, q_en, DType::INT32),
                                    g.constant({5}, st, DType::INT32));

            int32_t k_st[] = {0, 0, 1, 0, 0}, k_en[] = {1, (int32_t)num_patches, 2, (int32_t)cfg.vt_num_attention_heads,
                                                        (int32_t)(cfg.vt_hidden_size / cfg.vt_num_attention_heads)};
            LogicalId k_t = g.slice(qkv_5d, g.constant({5}, k_st, DType::INT32), g.constant({5}, k_en, DType::INT32),
                                    g.constant({5}, st, DType::INT32));

            int32_t v_st[] = {0, 0, 2, 0, 0}, v_en[] = {1, (int32_t)num_patches, 3, (int32_t)cfg.vt_num_attention_heads,
                                                        (int32_t)(cfg.vt_hidden_size / cfg.vt_num_attention_heads)};
            LogicalId v_t = g.slice(qkv_5d, g.constant({5}, v_st, DType::INT32), g.constant({5}, v_en, DType::INT32),
                                    g.constant({5}, st, DType::INT32));

            int32_t p4[] = {0, 2, 1, 3};
            int32_t r4[] = {1, (int32_t)num_patches, (int32_t)cfg.vt_num_attention_heads,
                            (int32_t)(cfg.vt_hidden_size / cfg.vt_num_attention_heads)};

            LogicalId q_rot = apply_rope_2d(g.reshape(q_t, g.constant({4}, r4, DType::INT32)), cos_node, sin_node);
            LogicalId k_rot = apply_rope_2d(g.reshape(k_t, g.constant({4}, r4, DType::INT32)), cos_node, sin_node);

            LogicalId q_p = g.contiguous(g.permute(q_rot, g.constant({4}, p4, DType::INT32)));
            LogicalId k_p = g.contiguous(g.permute(k_rot, g.constant({4}, p4, DType::INT32)));
            LogicalId v_p = g.contiguous(
                g.permute(g.reshape(v_t, g.constant({4}, r4, DType::INT32)), g.constant({4}, p4, DType::INT32)));

            float scale = 1.0f / std::sqrt((float)(cfg.vt_hidden_size / cfg.vt_num_attention_heads));
            q_p = g.mul(q_p, g.fill(scale, g.getNode(q_p).getShape()));

            int32_t p_k[] = {0, 1, 3, 2};
            LogicalId scores = g.dot(q_p, g.contiguous(g.permute(k_p, g.constant({4}, p_k, DType::INT32))));

            int32_t ax = -1;
            LogicalId max_s = g.max(scores, g.constant({1}, &ax, DType::INT32));
            LogicalId max_exp = g.repeat(max_s, num_patches, 3);
            LogicalId shifted = g.add(scores, g.neg(max_exp));
            LogicalId exps = g.pow(g.fill(e_fp32, g.getNode(shifted).getShape()), shifted);
            LogicalId sums = g.repeat(g.sum(exps, g.constant({1}, &ax, DType::INT32)), num_patches, 3);
            LogicalId probs = g.div(exps, sums);

            LogicalId attn_out = g.dot(probs, v_p);
            int32_t p_out[] = {0, 2, 1, 3};
            attn_out = g.contiguous(g.permute(attn_out, g.constant({4}, p_out, DType::INT32)));
            int32_t r_out[] = {(int32_t)num_patches, (int32_t)cfg.vt_hidden_size};
            attn_out = g.reshape(attn_out, g.constant({2}, r_out, DType::INT32));

            LogicalId o = linear(attn_out, prefix + "wo.weight", cfg.vt_hidden_size, cfg.vt_hidden_size);
            x = g.add(residual, o);
            residual = x;

            LogicalId norm1 = rms_norm(x, prefix + "norm1.weight", 1e-6f, cfg.vt_hidden_size);
            LogicalId m_fc0 = linear(norm1, prefix + "mlp.fc0.weight", cfg.vt_hidden_size, cfg.vt_intermediate_size);
            // PytorchGELUTanh
            float c1 = 0.044715f, c2 = 0.79788f;
            LogicalId cubed = g.mul(g.mul(m_fc0, m_fc0), m_fc0);
            LogicalId inner = g.add(m_fc0, g.mul(cubed, g.fill(c1, g.getNode(cubed).getShape())));
            LogicalId tanh_val = tanh_atomic(g.mul(inner, g.fill(c2, g.getNode(inner).getShape())));
            LogicalId gelu_out = g.mul(g.mul(m_fc0, g.fill(half_fp32, g.getNode(m_fc0).getShape())),
                                       g.add(g.fill(one_fp32, g.getNode(tanh_val).getShape()), tanh_val));

            LogicalId m_fc1 = linear(gelu_out, prefix + "mlp.fc1.weight", cfg.vt_intermediate_size, cfg.vt_hidden_size);
            x = g.add(residual, m_fc1);
        }

        x = rms_norm(x, "vision_tower.encoder.final_layernorm.weight", 1e-6f, cfg.vt_hidden_size);

        // tpool_patch_merger
        uint32_t new_h = h / 2, new_w = w / 2;
        int32_t sh6[] = {(int32_t)t, (int32_t)new_h, 2, (int32_t)new_w, 2, (int32_t)cfg.vt_hidden_size};
        LogicalId reshaped = g.reshape(x, g.constant({6}, sh6, DType::INT32));
        int32_t perm6[] = {0, 1, 3, 2, 4, 5};
        LogicalId permuted = g.contiguous(g.permute(reshaped, g.constant({6}, perm6, DType::INT32)));
        int32_t ax = 0;
        LogicalId sum = g.sum(permuted, g.constant({1}, &ax, DType::INT32));
        LogicalId mean = g.mul(sum, g.fill(1.0f / t, g.getNode(sum).getShape()));
        int32_t sh2[] = {(int32_t)(new_h * new_w), (int32_t)(4 * cfg.vt_hidden_size)};
        LogicalId merged = g.reshape(mean, g.constant({2}, sh2, DType::INT32));

        // PatchMergerMLPV2
        LogicalId p_proj0 =
            linear(merged, "mm_projector.proj.0.weight", 4 * cfg.vt_hidden_size, 4 * cfg.vt_hidden_size);

        // GELU exact
        float inv_sqrt2 = 0.707106781f;
        LogicalId x_scaled = g.mul(p_proj0, g.fill(inv_sqrt2, g.getNode(p_proj0).getShape()));
        LogicalId xs_sq = g.mul(x_scaled, x_scaled);
        LogicalId abs_xs = g.pow(xs_sq, g.fill(half_fp32, g.getNode(xs_sq).getShape()));
        LogicalId sign_xs = g.div(x_scaled, g.add(abs_xs, g.fill(1e-12f, g.getNode(abs_xs).getShape())));
        LogicalId t_gelu = g.div(g.fill(one_fp32, g.getNode(abs_xs).getShape()),
                                 g.add(g.fill(one_fp32, g.getNode(abs_xs).getShape()),
                                       g.mul(g.fill(0.3275911f, g.getNode(abs_xs).getShape()), abs_xs)));
        LogicalId t2 = g.mul(t_gelu, t_gelu);
        LogicalId t3 = g.mul(t2, t_gelu);
        LogicalId t4 = g.mul(t3, t_gelu);
        LogicalId t5 = g.mul(t4, t_gelu);
        LogicalId poly = g.mul(g.fill(0.254829592f, g.getNode(t_gelu).getShape()), t_gelu);
        poly = g.add(poly, g.mul(g.fill(-0.284496736f, g.getNode(t2).getShape()), t2));
        poly = g.add(poly, g.mul(g.fill(1.421413741f, g.getNode(t3).getShape()), t3));
        poly = g.add(poly, g.mul(g.fill(-1.453152027f, g.getNode(t4).getShape()), t4));
        poly = g.add(poly, g.mul(g.fill(1.061405429f, g.getNode(t5).getShape()), t5));
        LogicalId exp_neg_xs_sq = g.pow(g.fill(e_fp32, g.getNode(xs_sq).getShape()), g.neg(xs_sq));
        LogicalId erf_pos = g.add(g.fill(one_fp32, g.getNode(poly).getShape()), g.neg(g.mul(poly, exp_neg_xs_sq)));
        LogicalId erf_val = g.mul(sign_xs, erf_pos);
        LogicalId gelu_p = g.mul(g.mul(p_proj0, g.fill(half_fp32, g.getNode(p_proj0).getShape())),
                                 g.add(g.fill(one_fp32, g.getNode(erf_val).getShape()), erf_val));

        LogicalId p_proj2 = linear(gelu_p, "mm_projector.proj.2.weight", 4 * cfg.vt_hidden_size, cfg.hidden_size);
        LogicalId out = rms_norm(p_proj2, "mm_projector.post_norm.weight", 1e-5f, cfg.hidden_size);

        return out;
    }

    LogicalId kda_attention(LogicalId x, const std::string &prefix)
    {
        uint32_t qkv_dim = cfg.linear_num_heads * cfg.linear_head_dim;
        LogicalId q_proj = linear(x, prefix + "self_attn.q_proj.weight", cfg.hidden_size, qkv_dim);
        LogicalId k_proj = linear(x, prefix + "self_attn.k_proj.weight", cfg.hidden_size, qkv_dim);
        LogicalId v_proj = linear(x, prefix + "self_attn.v_proj.weight", cfg.hidden_size, qkv_dim);

        // 1D Causal Convolution Simulation
        LogicalId pad_q = g.concat({g.fill(zero_fp32, {1, 3, qkv_dim}), q_proj}, 1);
        int32_t st0[] = {0, 3, 0}, en0[] = {1, (int32_t)seq_len + 3, (int32_t)qkv_dim}, step[] = {1, 1, 1};
        LogicalId t0_q = g.slice(pad_q, g.constant({3}, st0, DType::INT32), g.constant({3}, en0, DType::INT32),
                                 g.constant({3}, step, DType::INT32));
        int32_t st1[] = {0, 2, 0}, en1[] = {1, (int32_t)seq_len + 2, (int32_t)qkv_dim};
        LogicalId t1_q = g.slice(pad_q, g.constant({3}, st1, DType::INT32), g.constant({3}, en1, DType::INT32),
                                 g.constant({3}, step, DType::INT32));
        int32_t st2[] = {0, 1, 0}, en2[] = {1, (int32_t)seq_len + 1, (int32_t)qkv_dim};
        LogicalId t2_q = g.slice(pad_q, g.constant({3}, st2, DType::INT32), g.constant({3}, en2, DType::INT32),
                                 g.constant({3}, step, DType::INT32));
        int32_t st3[] = {0, 0, 0}, en3[] = {1, (int32_t)seq_len, (int32_t)qkv_dim};
        LogicalId t3_q = g.slice(pad_q, g.constant({3}, st3, DType::INT32), g.constant({3}, en3, DType::INT32),
                                 g.constant({3}, step, DType::INT32));

        auto apply_conv = [&](LogicalId t0, LogicalId t1, LogicalId t2, LogicalId t3, const std::string &name) {
            LogicalId w = weight(prefix + "self_attn." + name + ".weight");
            int32_t ws0[] = {0, 0, 0}, we0[] = {(int32_t)qkv_dim, 1, 1};
            LogicalId w0 = g.reshape(
                g.contiguous(g.slice(w, g.constant({3}, ws0, DType::INT32), g.constant({3}, we0, DType::INT32),
                                     g.constant({3}, step, DType::INT32))),
                g.constant({3}, std::vector<int32_t>{1, (int32_t)qkv_dim, 1}.data(), DType::INT32));
            int32_t ws1[] = {0, 0, 1}, we1[] = {(int32_t)qkv_dim, 1, 2};
            LogicalId w1 = g.reshape(
                g.contiguous(g.slice(w, g.constant({3}, ws1, DType::INT32), g.constant({3}, we1, DType::INT32),
                                     g.constant({3}, step, DType::INT32))),
                g.constant({3}, std::vector<int32_t>{1, (int32_t)qkv_dim, 1}.data(), DType::INT32));
            int32_t ws2[] = {0, 0, 2}, we2[] = {(int32_t)qkv_dim, 1, 3};
            LogicalId w2 = g.reshape(
                g.contiguous(g.slice(w, g.constant({3}, ws2, DType::INT32), g.constant({3}, we2, DType::INT32),
                                     g.constant({3}, step, DType::INT32))),
                g.constant({3}, std::vector<int32_t>{1, (int32_t)qkv_dim, 1}.data(), DType::INT32));
            int32_t ws3[] = {0, 0, 3}, we3[] = {(int32_t)qkv_dim, 1, 4};
            LogicalId w3 = g.reshape(
                g.contiguous(g.slice(w, g.constant({3}, ws3, DType::INT32), g.constant({3}, we3, DType::INT32),
                                     g.constant({3}, step, DType::INT32))),
                g.constant({3}, std::vector<int32_t>{1, (int32_t)qkv_dim, 1}.data(), DType::INT32));

            LogicalId term0 = g.mul(t3, g.repeat(w0, seq_len, 1));
            LogicalId term1 = g.mul(t2, g.repeat(w1, seq_len, 1));
            LogicalId term2 = g.mul(t1, g.repeat(w2, seq_len, 1));
            LogicalId term3 = g.mul(t0, g.repeat(w3, seq_len, 1));
            return g.add(g.add(g.add(term0, term1), term2), term3);
        };

        LogicalId q = apply_conv(t0_q, t1_q, t2_q, t3_q, "q_conv1d");

        LogicalId pad_k = g.concat({g.fill(zero_fp32, {1, 3, qkv_dim}), k_proj}, 1);
        LogicalId t0_k = g.slice(pad_k, g.constant({3}, st0, DType::INT32), g.constant({3}, en0, DType::INT32),
                                 g.constant({3}, step, DType::INT32));
        LogicalId t1_k = g.slice(pad_k, g.constant({3}, st1, DType::INT32), g.constant({3}, en1, DType::INT32),
                                 g.constant({3}, step, DType::INT32));
        LogicalId t2_k = g.slice(pad_k, g.constant({3}, st2, DType::INT32), g.constant({3}, en2, DType::INT32),
                                 g.constant({3}, step, DType::INT32));
        LogicalId t3_k = g.slice(pad_k, g.constant({3}, st3, DType::INT32), g.constant({3}, en3, DType::INT32),
                                 g.constant({3}, step, DType::INT32));
        LogicalId k = apply_conv(t0_k, t1_k, t2_k, t3_k, "k_conv1d");

        LogicalId pad_v = g.concat({g.fill(zero_fp32, {1, 3, qkv_dim}), v_proj}, 1);
        LogicalId t0_v = g.slice(pad_v, g.constant({3}, st0, DType::INT32), g.constant({3}, en0, DType::INT32),
                                 g.constant({3}, step, DType::INT32));
        LogicalId t1_v = g.slice(pad_v, g.constant({3}, st1, DType::INT32), g.constant({3}, en1, DType::INT32),
                                 g.constant({3}, step, DType::INT32));
        LogicalId t2_v = g.slice(pad_v, g.constant({3}, st2, DType::INT32), g.constant({3}, en2, DType::INT32),
                                 g.constant({3}, step, DType::INT32));
        LogicalId t3_v = g.slice(pad_v, g.constant({3}, st3, DType::INT32), g.constant({3}, en3, DType::INT32),
                                 g.constant({3}, step, DType::INT32));
        LogicalId v = apply_conv(t0_v, t1_v, t2_v, t3_v, "v_conv1d");

        // SiLU
        auto silu = [&](LogicalId x) { return g.mul(x, sigmoid(x)); };
        q = silu(q);
        k = silu(k);
        v = silu(v);

        // L2 Norm Q and K
        int32_t sh4[] = {1, (int32_t)seq_len, (int32_t)cfg.linear_num_heads, (int32_t)cfg.linear_head_dim};
        q = g.reshape(q, g.constant({4}, sh4, DType::INT32));
        k = g.reshape(k, g.constant({4}, sh4, DType::INT32));
        v = g.reshape(v, g.constant({4}, sh4, DType::INT32));

        int32_t p4[] = {0, 2, 1, 3};
        q = g.contiguous(g.permute(q, g.constant({4}, p4, DType::INT32)));
        k = g.contiguous(g.permute(k, g.constant({4}, p4, DType::INT32)));
        v = g.contiguous(g.permute(v, g.constant({4}, p4, DType::INT32)));

        int32_t ax_neg1 = -1;
        LogicalId q_sq = g.mul(q, q);
        LogicalId q_sum = g.sum(q_sq, g.constant({1}, &ax_neg1, DType::INT32));
        LogicalId q_std = g.pow(g.add(q_sum, g.fill(1e-6f, {1, cfg.linear_num_heads, seq_len, 1})),
                                g.fill(half_fp32, {1, cfg.linear_num_heads, seq_len, 1}));
        q = g.mul(
            q, g.repeat(g.div(g.fill(one_fp32, {1, cfg.linear_num_heads, seq_len, 1}), q_std), cfg.linear_head_dim, 3));

        LogicalId k_sq = g.mul(k, k);
        LogicalId k_sum = g.sum(k_sq, g.constant({1}, &ax_neg1, DType::INT32));
        LogicalId k_std = g.pow(g.add(k_sum, g.fill(1e-6f, {1, cfg.linear_num_heads, seq_len, 1})),
                                g.fill(half_fp32, {1, cfg.linear_num_heads, seq_len, 1}));
        k = g.mul(
            k, g.repeat(g.div(g.fill(one_fp32, {1, cfg.linear_num_heads, seq_len, 1}), k_std), cfg.linear_head_dim, 3));

        // Gates
        LogicalId beta = sigmoid(linear(x, prefix + "self_attn.b_proj.weight", cfg.hidden_size, cfg.linear_num_heads));
        LogicalId f_a = linear(x, prefix + "self_attn.f_a_proj.weight", cfg.hidden_size, cfg.linear_head_dim);
        LogicalId g_decay = linear(f_a, prefix + "self_attn.f_b_proj.weight", cfg.linear_head_dim,
                                   cfg.linear_num_heads * cfg.linear_head_dim);

        LogicalId dt_bias = weight(prefix + "self_attn.dt_bias");
        LogicalId dt_bias_exp = g.repeat(
            g.reshape(dt_bias,
                      g.constant(
                          {3}, std::vector<int32_t>{1, 1, (int32_t)(cfg.linear_num_heads * cfg.linear_head_dim)}.data(),
                          DType::INT32)),
            seq_len, 1);
        LogicalId dt = g.add(g_decay, dt_bias_exp);
        LogicalId dt_softplus = softplus(dt);

        LogicalId A_log = weight(prefix + "self_attn.A_log"); // per head (96)
        LogicalId A_log_exp_layer = g.repeat(
            g.reshape(g.pow(g.fill(e_fp32, (std::vector<uint32_t>){cfg.linear_num_heads}), A_log),
                      g.constant({3}, std::vector<int32_t>{1, 1, (int32_t)cfg.linear_num_heads}.data(), DType::INT32)),
            cfg.linear_head_dim, 2);
        A_log_exp_layer = g.repeat(A_log_exp_layer, seq_len, 1);
        LogicalId decay = g.mul(g.neg(A_log_exp_layer), dt_softplus);
        LogicalId alpha = g.pow(g.fill(e_fp32, g.getNode(decay).getShape()), decay);

        int32_t pb[] = {0, 2, 1, 3};
        beta = g.contiguous(g.permute(
            g.reshape(beta, g.constant(
                                {4}, std::vector<int32_t>{1, (int32_t)seq_len, (int32_t)cfg.linear_num_heads, 1}.data(),
                                DType::INT32)),
            g.constant({4}, pb, DType::INT32)));
        alpha = g.contiguous(g.permute(
            g.reshape(alpha, g.constant({4},
                                        std::vector<int32_t>{1, (int32_t)seq_len, (int32_t)cfg.linear_num_heads,
                                                             (int32_t)cfg.linear_head_dim}
                                            .data(),
                                        DType::INT32)),
            g.constant({4}, pb, DType::INT32)));

        // Recurrence
        std::vector<LogicalId> outs;
        LogicalId S = g.fill(zero_fp32,
                             g.constant({3},
                                        std::vector<int32_t>{(int32_t)cfg.linear_num_heads,
                                                             (int32_t)cfg.linear_head_dim, (int32_t)cfg.linear_head_dim}
                                            .data(),
                                        DType::INT32));

        for (uint32_t t = 0; t < seq_len; ++t)
        {
            int32_t st4[] = {0, 0, (int32_t)t, 0},
                    en4[] = {1, (int32_t)cfg.linear_num_heads, (int32_t)t + 1, (int32_t)cfg.linear_head_dim};
            int32_t stb[] = {0, 0, (int32_t)t, 0}, enb[] = {1, (int32_t)cfg.linear_num_heads, (int32_t)t + 1, 1};

            LogicalId q_t = g.reshape(
                g.contiguous(g.slice(q, g.constant({4}, st4, DType::INT32), g.constant({4}, en4, DType::INT32),
                                     g.constant({4}, step, DType::INT32))),
                g.constant({3},
                           std::vector<int32_t>{(int32_t)cfg.linear_num_heads, 1, (int32_t)cfg.linear_head_dim}.data(),
                           DType::INT32));
            LogicalId k_t = g.reshape(
                g.contiguous(g.slice(k, g.constant({4}, st4, DType::INT32), g.constant({4}, en4, DType::INT32),
                                     g.constant({4}, step, DType::INT32))),
                g.constant({3},
                           std::vector<int32_t>{(int32_t)cfg.linear_num_heads, 1, (int32_t)cfg.linear_head_dim}.data(),
                           DType::INT32));
            LogicalId v_t = g.reshape(
                g.contiguous(g.slice(v, g.constant({4}, st4, DType::INT32), g.constant({4}, en4, DType::INT32),
                                     g.constant({4}, step, DType::INT32))),
                g.constant({3},
                           std::vector<int32_t>{(int32_t)cfg.linear_num_heads, 1, (int32_t)cfg.linear_head_dim}.data(),
                           DType::INT32));
            LogicalId a_t = g.reshape(
                g.contiguous(g.slice(alpha, g.constant({4}, st4, DType::INT32), g.constant({4}, en4, DType::INT32),
                                     g.constant({4}, step, DType::INT32))),
                g.constant({3},
                           std::vector<int32_t>{(int32_t)cfg.linear_num_heads, 1, (int32_t)cfg.linear_head_dim}.data(),
                           DType::INT32));
            LogicalId b_t = g.reshape(
                g.contiguous(g.slice(beta, g.constant({4}, stb, DType::INT32), g.constant({4}, enb, DType::INT32),
                                     g.constant({4}, step, DType::INT32))),
                g.constant({3}, std::vector<int32_t>{(int32_t)cfg.linear_num_heads, 1, 1}.data(), DType::INT32));

            LogicalId kv_mem = g.contiguous(g.dot(k_t, S));
            LogicalId err = g.add(v_t, g.neg(kv_mem));
            LogicalId delta = g.mul(err, g.repeat(b_t, cfg.linear_head_dim, 2));

            int32_t pk[] = {0, 2, 1};
            LogicalId k_t_t = g.contiguous(g.permute(k_t, g.constant({3}, pk, DType::INT32)));
            LogicalId outer = g.contiguous(g.dot(k_t_t, delta));

            S = g.add(g.mul(S, g.repeat(a_t, cfg.linear_head_dim, 1)), outer);

            LogicalId y_t = g.contiguous(g.dot(q_t, S));
            outs.push_back(g.reshape(
                y_t,
                g.constant(
                    {4}, std::vector<int32_t>{1, (int32_t)cfg.linear_num_heads, 1, (int32_t)cfg.linear_head_dim}.data(),
                    DType::INT32)));
        }

        LogicalId context_heads = (seq_len > 1) ? g.concat(outs, 2) : outs[0];
        context_heads = g.contiguous(
            g.permute(context_heads, g.constant({4}, std::vector<int32_t>{0, 2, 1, 3}.data(), DType::INT32)));
        LogicalId context = g.reshape(
            context_heads,
            g.constant(
                {3},
                std::vector<int32_t>{1, (int32_t)seq_len, (int32_t)(cfg.linear_num_heads * cfg.linear_head_dim)}.data(),
                DType::INT32));

        LogicalId g_out =
            linear(x, prefix + "self_attn.g_proj.weight", cfg.hidden_size, cfg.linear_num_heads * cfg.linear_head_dim);

        // o_norm
        LogicalId sq = g.mul(context, context);
        LogicalId sum_sq = g.sum(sq, g.constant({1}, &ax_neg1, DType::INT32));
        LogicalId mean_sq = g.div(sum_sq, g.fill((float)(cfg.linear_num_heads * cfg.linear_head_dim), {1, seq_len, 1}));
        LogicalId var = g.add(mean_sq, g.fill(cfg.rms_norm_eps, {1, seq_len, 1}));
        LogicalId std = g.pow(var, g.fill(half_fp32, {1, seq_len, 1}));
        LogicalId inv_std =
            g.repeat(g.div(g.fill(one_fp32, {1, seq_len, 1}), std), cfg.linear_num_heads * cfg.linear_head_dim, 2);

        LogicalId c_norm = g.mul(context, inv_std);
        LogicalId c_norm_scaled = g.mul(
            c_norm,
            g.repeat(
                g.reshape(
                    weight(prefix + "self_attn.o_norm.weight"),
                    g.constant({3},
                               std::vector<int32_t>{1, 1, (int32_t)(cfg.linear_num_heads * cfg.linear_head_dim)}.data(),
                               DType::INT32)),
                seq_len, 1));

        LogicalId o = g.mul(c_norm_scaled, sigmoid(g_out));
        return linear(o, prefix + "self_attn.o_proj.weight", cfg.linear_num_heads * cfg.linear_head_dim,
                      cfg.hidden_size);
    }

    LogicalId mla_attention(LogicalId x, const std::string &prefix)
    {
        LogicalId q_c = linear(x, prefix + "self_attn.q_a_proj.weight", cfg.hidden_size, cfg.q_lora_rank);
        q_c = rms_norm(q_c, prefix + "self_attn.q_a_layernorm.weight", 1e-6f, cfg.q_lora_rank);
        LogicalId q = linear(q_c, prefix + "self_attn.q_b_proj.weight", cfg.q_lora_rank,
                             cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim));

        LogicalId kv_c = linear(x, prefix + "self_attn.kv_a_proj_with_mqa.weight", cfg.hidden_size,
                                cfg.kv_lora_rank + cfg.qk_rope_head_dim);

        int32_t st_kv[] = {0, 0, 0}, en_kpass[] = {1, (int32_t)seq_len, (int32_t)cfg.kv_lora_rank}, step[] = {1, 1, 1};
        LogicalId k_pass =
            g.contiguous(g.slice(kv_c, g.constant({3}, st_kv, DType::INT32), g.constant({3}, en_kpass, DType::INT32),
                                 g.constant({3}, step, DType::INT32)));

        int32_t st_krot[] = {0, 0, (int32_t)cfg.kv_lora_rank},
                en_krot[] = {1, (int32_t)seq_len, (int32_t)(cfg.kv_lora_rank + cfg.qk_rope_head_dim)};
        LogicalId k_rot =
            g.contiguous(g.slice(kv_c, g.constant({3}, st_krot, DType::INT32), g.constant({3}, en_krot, DType::INT32),
                                 g.constant({3}, step, DType::INT32)));

        k_pass = rms_norm(k_pass, prefix + "self_attn.kv_a_layernorm.weight", 1e-6f, cfg.kv_lora_rank);
        LogicalId k_b = linear(k_pass, prefix + "self_attn.kv_b_proj.weight", cfg.kv_lora_rank,
                               cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim));

        int32_t sh4_q[] = {1, (int32_t)seq_len, (int32_t)cfg.num_attention_heads,
                           (int32_t)(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)};
        LogicalId q_4d = g.reshape(q, g.constant({4}, sh4_q, DType::INT32));

        int32_t sh4_k[] = {1, (int32_t)seq_len, (int32_t)cfg.num_attention_heads,
                           (int32_t)(cfg.qk_nope_head_dim + cfg.v_head_dim)};
        LogicalId k_4d = g.reshape(k_b, g.constant({4}, sh4_k, DType::INT32));

        int32_t st_kn[] = {0, 0, 0, 0},
                en_kn[] = {1, (int32_t)seq_len, (int32_t)cfg.num_attention_heads, (int32_t)cfg.qk_nope_head_dim},
                st4[] = {1, 1, 1, 1};
        LogicalId k_nope =
            g.contiguous(g.slice(k_4d, g.constant({4}, st_kn, DType::INT32), g.constant({4}, en_kn, DType::INT32),
                                 g.constant({4}, st4, DType::INT32)));

        int32_t st_kvv[] = {0, 0, 0, (int32_t)cfg.qk_nope_head_dim},
                en_kvv[] = {1, (int32_t)seq_len, (int32_t)cfg.num_attention_heads,
                            (int32_t)(cfg.qk_nope_head_dim + cfg.v_head_dim)};
        LogicalId v_4d =
            g.contiguous(g.slice(k_4d, g.constant({4}, st_kvv, DType::INT32), g.constant({4}, en_kvv, DType::INT32),
                                 g.constant({4}, st4, DType::INT32)));

        LogicalId k_rot_exp = g.repeat(
            g.reshape(k_rot,
                      g.constant({4},
                                 std::vector<int32_t>{1, (int32_t)seq_len, 1, (int32_t)cfg.qk_rope_head_dim}.data(),
                                 DType::INT32)),
            cfg.num_attention_heads, 2);

        int32_t ax_neg1 = -1;
        LogicalId k_final = g.concat({k_nope, k_rot_exp}, g.constant({1}, &ax_neg1, DType::INT32));

        int32_t p4[] = {0, 2, 1, 3};
        LogicalId q_p = g.contiguous(g.permute(q_4d, g.constant({4}, p4, DType::INT32)));
        LogicalId k_p = g.contiguous(g.permute(k_final, g.constant({4}, p4, DType::INT32)));
        LogicalId v_p = g.contiguous(g.permute(v_4d, g.constant({4}, p4, DType::INT32)));

        float scale = 1.0f / std::sqrt((float)(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim));
        q_p = g.mul(q_p, g.fill(scale, g.getNode(q_p).getShape()));

        int32_t p_k[] = {0, 1, 3, 2};
        LogicalId scores = g.dot(q_p, g.contiguous(g.permute(k_p, g.constant({4}, p_k, DType::INT32))));

        // causal mask
        int32_t mask_shape[] = {(int32_t)seq_len, (int32_t)seq_len};
        LogicalId ones = g.fill(one_fp32, g.constant({2}, mask_shape, DType::INT32));
        int32_t kval = 1;
        LogicalId triu = g.triu(ones, g.constant({1}, &kval, DType::INT32));
        LogicalId neg_inf = g.mul(triu, g.fill(neg_1e9_fp32, {1, 1, seq_len, seq_len}));

        scores = g.add(scores, neg_inf);

        LogicalId max_s = g.max(scores, g.constant({1}, &ax_neg1, DType::INT32));
        LogicalId max_exp = g.repeat(max_s, seq_len, 3);
        LogicalId shifted = g.add(scores, g.neg(max_exp));
        LogicalId exps = g.pow(g.fill(e_fp32, g.getNode(shifted).getShape()), shifted);
        LogicalId sums = g.repeat(g.sum(exps, g.constant({1}, &ax_neg1, DType::INT32)), seq_len, 3);
        LogicalId probs = g.div(exps, sums);

        LogicalId ctx = g.dot(probs, v_p);
        int32_t p_ctx[] = {0, 2, 1, 3};
        ctx = g.contiguous(g.permute(ctx, g.constant({4}, p_ctx, DType::INT32)));
        ctx = g.reshape(
            ctx,
            g.constant(
                {3},
                std::vector<int32_t>{1, (int32_t)seq_len, (int32_t)(cfg.num_attention_heads * cfg.v_head_dim)}.data(),
                DType::INT32));

        if (cfg.mla_use_output_gate)
        {
            LogicalId g_out = linear(x, prefix + "self_attn.g_proj.weight", cfg.hidden_size,
                                     cfg.num_attention_heads * cfg.v_head_dim);
            ctx = g.mul(ctx, sigmoid(g_out));
        }

        return linear(ctx, prefix + "self_attn.o_proj.weight", cfg.num_attention_heads * cfg.v_head_dim,
                      cfg.hidden_size);
    }

    LogicalId build_text_graph()
    {
        LogicalId inputs_embeds = g.input({1, seq_len, cfg.hidden_size}, DType::FLOAT32);
        LogicalId x = inputs_embeds;

        // Attn residual buffer
        LogicalId block_residual;

        for (uint32_t l = 0; l < cfg.num_hidden_layers; ++l)
        {
            std::string prefix = "language_model.model.layers." + std::to_string(l) + ".";
            LogicalId residual = x;

            x = rms_norm(x, prefix + "input_layernorm.weight", cfg.rms_norm_eps, cfg.hidden_size);

            if (cfg.is_kda_layer(l))
            {
                x = kda_attention(x, prefix);
            }
            else
            {
                x = mla_attention(x, prefix);
            }

            x = g.add(residual, x);
            residual = x;

            x = rms_norm(x, prefix + "post_attention_layernorm.weight", cfg.rms_norm_eps, cfg.hidden_size);

            // Sparse MoE
            LogicalId down = linear(x, prefix + "block_sparse_moe.routed_expert_down_proj.weight", cfg.hidden_size,
                                    cfg.routed_expert_hidden_size);
            LogicalId r_logits = linear(x, prefix + "block_sparse_moe.gate.weight", cfg.hidden_size, cfg.num_experts);

            // For tensor_graphs we approximate the routing by computing all experts (batched matmul) and masking
            int32_t ax_neg1 = -1;
            LogicalId max_s = g.max(r_logits, g.constant({1}, &ax_neg1, DType::INT32));
            LogicalId shifted = g.add(r_logits, g.neg(g.repeat(max_s, cfg.num_experts, 2)));
            LogicalId exps = g.pow(g.fill(e_fp32, g.getNode(shifted).getShape()), shifted);
            LogicalId sums = g.repeat(g.sum(exps, g.constant({1}, &ax_neg1, DType::INT32)), cfg.num_experts, 2);
            LogicalId probs = g.div(exps, sums);

            uint32_t K = cfg.num_experts_per_token;
            LogicalId selected =
                g.argmax(probs, g.constant({1}, &ax_neg1, DType::INT32), g.constant({1}, (int32_t *)&K, DType::INT32));

            int32_t st_arange = 0, stop_arange = (int32_t)cfg.num_experts, step_arange = 1;
            LogicalId range_1d =
                g.arange(g.constant({1}, &st_arange, DType::INT32), g.constant({1}, &stop_arange, DType::INT32),
                         g.constant({1}, &step_arange, DType::INT32));
            LogicalId range_exp = g.repeat(
                g.repeat(
                    g.reshape(range_1d, g.constant({4}, std::vector<int32_t>{1, 1, 1, (int32_t)cfg.num_experts}.data(),
                                                   DType::INT32)),
                    seq_len, 1),
                K, 2);
            LogicalId sel_exp = g.repeat(
                g.reshape(selected, g.constant({4}, std::vector<int32_t>{1, (int32_t)seq_len, (int32_t)K, 1}.data(),
                                               DType::INT32)),
                cfg.num_experts, 3);
            LogicalId mask = g.cast(g.eq(sel_exp, range_exp), DType::FLOAT32);

            int32_t ax2 = 2;
            LogicalId mask_reduced =
                g.reshape(g.sum(mask, g.constant({1}, &ax2, DType::INT32)),
                          g.constant({3}, std::vector<int32_t>{1, (int32_t)seq_len, (int32_t)cfg.num_experts}.data(),
                                     DType::INT32));

            LogicalId gated_probs = g.mul(probs, mask_reduced);
            LogicalId row_sum =
                g.repeat(g.sum(gated_probs, g.constant({1}, &ax_neg1, DType::INT32)), cfg.num_experts, 2);
            LogicalId norm_probs = g.div(gated_probs, row_sum);

            LogicalId x_exp = g.contiguous(g.repeat(
                g.reshape(
                    down,
                    g.constant({3},
                               std::vector<int32_t>{1, (int32_t)seq_len, (int32_t)cfg.routed_expert_hidden_size}.data(),
                               DType::INT32)),
                cfg.num_experts, 0)); // (E, S, I)

            // Fused batched expert loading (assuming individual loading is aggregated)
            // Note: C++ host will load and concat experts to (E, out, in) to optimize
            LogicalId w1 = weight(prefix + "block_sparse_moe.experts_w1"); // [E, 3072, 3584]
            LogicalId w3 = weight(prefix + "block_sparse_moe.experts_w3"); // [E, 3072, 3584]
            LogicalId w2 = weight(prefix + "block_sparse_moe.experts_w2"); // [E, 3584, 3072]

            int32_t pb3[] = {0, 2, 1};
            LogicalId gate = g.dot(x_exp, g.contiguous(g.permute(w1, g.constant({3}, pb3, DType::INT32))));
            LogicalId up = g.dot(x_exp, g.contiguous(g.permute(w3, g.constant({3}, pb3, DType::INT32))));
            LogicalId act = situ(gate, up);
            LogicalId exp_out =
                g.dot(act, g.contiguous(g.permute(w2, g.constant({3}, pb3, DType::INT32)))); // [E, S, 3584]

            int32_t pb_out[] = {1, 0, 2};
            LogicalId exp_out_perm =
                g.contiguous(g.permute(exp_out, g.constant({3}, pb_out, DType::INT32))); // [S, E, 3584]
            int32_t p1se[] = {1, 2, 0};
            LogicalId norm_probs_perm =
                g.contiguous(g.permute(norm_probs, g.constant({3}, p1se, DType::INT32))); // [S, E, 1]
            LogicalId weighted = g.mul(exp_out_perm, g.repeat(norm_probs_perm, cfg.routed_expert_hidden_size, 2));
            int32_t ax1 = 1;
            LogicalId routed = g.reshape(
                g.sum(weighted, g.constant({1}, &ax1, DType::INT32)),
                g.constant({3},
                           std::vector<int32_t>{1, (int32_t)seq_len, (int32_t)cfg.routed_expert_hidden_size}.data(),
                           DType::INT32));

            routed = rms_norm(routed, prefix + "block_sparse_moe.routed_expert_norm.weight", cfg.rms_norm_eps,
                              cfg.routed_expert_hidden_size);
            routed = linear(routed, prefix + "block_sparse_moe.routed_expert_up_proj.weight",
                            cfg.routed_expert_hidden_size, cfg.hidden_size);

            // Shared experts
            LogicalId sh_gate = linear(residual, prefix + "block_sparse_moe.shared_experts.gate_proj.weight",
                                       cfg.hidden_size, cfg.num_shared_experts * cfg.moe_intermediate_size);
            LogicalId sh_up = linear(residual, prefix + "block_sparse_moe.shared_experts.up_proj.weight",
                                     cfg.hidden_size, cfg.num_shared_experts * cfg.moe_intermediate_size);
            LogicalId sh_act = situ(sh_gate, sh_up);
            LogicalId sh_out = linear(sh_act, prefix + "block_sparse_moe.shared_experts.down_proj.weight",
                                      cfg.num_shared_experts * cfg.moe_intermediate_size, cfg.hidden_size);

            x = g.add(residual, g.add(routed, sh_out));
        }

        x = rms_norm(x, "language_model.model.norm.weight", cfg.rms_norm_eps, cfg.hidden_size);
        LogicalId logits = linear(x, "language_model.lm_head.weight", cfg.hidden_size, cfg.vocab_size);
        return logits;
    }
};