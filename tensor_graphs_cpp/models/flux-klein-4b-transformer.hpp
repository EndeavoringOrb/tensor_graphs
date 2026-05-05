// File: tensor_graphs_cpp/models/flux-klein-4b-transformer.hpp
#pragma once

class FluxTransformer : public FluxGraphBase
{
private:
    FluxConfig cfg;
    MemoryManager &mem;
    uint32_t latent_h, latent_w, img_seq_len, total_seq_len;

    uint32_t linear(uint32_t x, const std::string &w_name, uint32_t in_d, uint32_t out_d)
    {
        uint32_t w = weight(w_name);
        int32_t p[] = {1, 0};
        uint32_t w_t = g.contiguous(g.permute(w, g.constant({2}, p, DType::INT32)));
        int32_t sh3[] = {1, (int32_t)in_d, (int32_t)out_d};
        return g.dot(x, g.reshape(w_t, g.constant({3}, sh3, DType::INT32)));
    }

    uint32_t timestep_embedding(uint32_t t_in, uint32_t dim)
    {
        int32_t half = dim / 2;
        int32_t start = 0, step = 1;
        uint32_t freqs = g.cast(g.arange(g.constant({1}, &start, DType::INT32),
                                         g.constant({1}, &half, DType::INT32),
                                         g.constant({1}, &step, DType::INT32)),
                                DType::FLOAT32);

        float factor = -std::log(10000.0f) / half;
        freqs = g.mul(freqs, expand_scalar_to_1d(factor, half));
        freqs = g.pow(expand_scalar_to_1d(2.7182818f, half), freqs);

        uint32_t args = g.mul(repeat_ax(t_in, half, 0), freqs);
        int32_t ax = 0;
        return g.concat({g.cos(args), g.sin(args)}, g.constant({1}, &ax, DType::INT32));
    }

    uint32_t layer_norm_atomic(uint32_t x, uint32_t seq)
    {
        int32_t ax = -1;
        uint32_t ax_node = g.constant({1}, &ax, DType::INT32);
        uint32_t mean = repeat_ax(g.div(g.sum(x, ax_node), expand_scalar_to_3d((float)cfg.hidden_size, 1, seq, 1)), cfg.hidden_size, 2);

        uint32_t x_sub = g.add(x, g.neg(mean));
        uint32_t sq = g.mul(x_sub, x_sub);
        uint32_t var = g.div(g.sum(sq, ax_node), expand_scalar_to_3d((float)cfg.hidden_size, 1, seq, 1));

        uint32_t std = g.pow(g.add(var, expand_scalar_to_3d(1e-6f, 1, seq, 1)), expand_scalar_to_3d(0.5f, 1, seq, 1));
        return g.mul(x_sub, repeat_ax(g.div(expand_scalar_to_3d(1.0f, 1, seq, 1), std), cfg.hidden_size, 2));
    }

    uint32_t rms_norm_atomic(uint32_t x, const std::string &w_name, uint32_t seq, uint32_t num_heads, uint32_t head_dim)
    {
        uint32_t sq = g.mul(x, x);
        int32_t ax = -1;
        uint32_t sum_sq = g.sum(sq, g.constant({1}, &ax, DType::INT32));
        uint32_t mean_sq = g.div(sum_sq, expand_scalar_to_4d((float)head_dim, 1, num_heads, seq, 1));
        uint32_t std = g.pow(g.add(mean_sq, expand_scalar_to_4d(1e-6f, 1, num_heads, seq, 1)), expand_scalar_to_4d(0.5f, 1, num_heads, seq, 1));
        uint32_t inv_std = repeat_ax(g.div(expand_scalar_to_4d(1.0f, 1, num_heads, seq, 1), std), head_dim, 3);

        uint32_t w = weight(w_name);
        int32_t sh4[] = {1, 1, 1, (int32_t)head_dim};
        uint32_t w_exp = repeat_ax(repeat_ax(g.reshape(w, g.constant({4}, sh4, DType::INT32)), num_heads, 1), seq, 2);
        return g.mul(g.mul(x, inv_std), w_exp);
    }

    uint32_t apply_rope_2d_consecutive(uint32_t x, uint32_t cos, uint32_t sin, uint32_t seq)
    {
        int32_t starts1[] = {0, 0, 0, 0};
        int32_t ends1[] = {1, (int32_t)cfg.num_heads, (int32_t)seq, (int32_t)cfg.head_dim};
        int32_t steps[] = {1, 1, 1, 2};
        uint32_t x_even = g.contiguous(g.slice(x, g.constant({4}, starts1, DType::INT32), g.constant({4}, ends1, DType::INT32), g.constant({4}, steps, DType::INT32)));

        int32_t starts_odd[] = {0, 0, 0, 1};
        uint32_t x_odd = g.contiguous(g.slice(x, g.constant({4}, starts_odd, DType::INT32), g.constant({4}, ends1, DType::INT32), g.constant({4}, steps, DType::INT32)));

        uint32_t c = repeat_ax(g.contiguous(g.slice(cos, g.constant({4}, starts1, DType::INT32), g.constant({4}, ends1, DType::INT32), g.constant({4}, steps, DType::INT32))), cfg.num_heads, 1);
        uint32_t s = repeat_ax(g.contiguous(g.slice(sin, g.constant({4}, starts1, DType::INT32), g.constant({4}, ends1, DType::INT32), g.constant({4}, steps, DType::INT32))), cfg.num_heads, 1);

        uint32_t out_even = g.add(g.mul(x_even, c), g.neg(g.mul(x_odd, s)));
        uint32_t out_odd = g.add(g.mul(x_odd, c), g.mul(x_even, s));

        int32_t sh5[] = {1, (int32_t)cfg.num_heads, (int32_t)seq, (int32_t)cfg.head_dim / 2, 1};
        uint32_t sh5_node = g.constant({5}, sh5, DType::INT32);
        int32_t ax4 = 4;
        uint32_t stacked = g.concat({g.reshape(out_even, sh5_node), g.reshape(out_odd, sh5_node)}, g.constant({1}, &ax4, DType::INT32));

        int32_t sh4[] = {1, (int32_t)cfg.num_heads, (int32_t)seq, (int32_t)cfg.head_dim};
        return g.reshape(stacked, g.constant({4}, sh4, DType::INT32));
    }

    std::vector<uint32_t> compute_mods(uint32_t t_emb, const std::string &w_name, int chunks)
    {
        uint32_t mod = linear(t_emb, w_name, cfg.hidden_size, cfg.hidden_size * chunks);
        std::vector<uint32_t> results;
        for (int i = 0; i < chunks; ++i)
        {
            int32_t starts[] = {0, (int32_t)(i * cfg.hidden_size)};
            int32_t ends[] = {1, (int32_t)((i + 1) * cfg.hidden_size)};
            int32_t steps[] = {1, 1};
            results.push_back(g.contiguous(g.slice(mod, g.constant({2}, starts, DType::INT32), g.constant({2}, ends, DType::INT32), g.constant({2}, steps, DType::INT32))));
        }
        return results;
    }

    uint32_t apply_mod(uint32_t norm, uint32_t shift, uint32_t scale, uint32_t seq)
    {
        int32_t sh3[] = {1, 1, (int32_t)cfg.hidden_size};
        uint32_t scale_3d = repeat_ax(g.reshape(scale, g.constant({3}, sh3, DType::INT32)), seq, 1);
        uint32_t shift_3d = repeat_ax(g.reshape(shift, g.constant({3}, sh3, DType::INT32)), seq, 1);
        uint32_t scaled = g.add(expand_scalar_to_3d(1.0f, 1, seq, cfg.hidden_size), scale_3d);
        return g.add(g.mul(norm, scaled), shift_3d);
    }

public:
    FluxTransformer(FluxConfig config, Graph &graph, MemoryManager &memory, const std::string &weight_path, uint32_t h, uint32_t w)
        : FluxGraphBase(graph, weight_path), cfg(config), mem(memory), latent_h(h), latent_w(w)
    {
        img_seq_len = latent_h * latent_w;
        total_seq_len = cfg.text_max_seq + img_seq_len;
    }

    uint32_t build_graph(uint32_t img_latent, uint32_t txt_emb, uint32_t timestep, uint32_t rope_cos, uint32_t rope_sin)
    {
        uint32_t ts_mul = g.mul(timestep, expand_scalar_to_1d(1000.0f, 1));
        uint32_t t_sincos = timestep_embedding(ts_mul, 256);
        int32_t sh2[] = {1, 256};
        uint32_t t_emb_raw = linear(g.reshape(t_sincos, g.constant({2}, sh2, DType::INT32)), "time_guidance_embed.timestep_embedder.linear_1.weight", 256, cfg.hidden_size);
        uint32_t t_emb_silu = silu_atomic(t_emb_raw, 1, 1, cfg.hidden_size);
        t_emb_silu = linear(t_emb_silu, "time_guidance_embed.timestep_embedder.linear_2.weight", cfg.hidden_size, cfg.hidden_size);

        // NLC
        int32_t p_img[] = {0, 2, 3, 1};
        uint32_t img_perm = g.contiguous(g.permute(img_latent, g.constant({4}, p_img, DType::INT32)));
        int32_t sh_img[] = {1, (int32_t)img_seq_len, (int32_t)cfg.latent_channels};
        uint32_t img_hidden = linear(g.reshape(img_perm, g.constant({3}, sh_img, DType::INT32)), "x_embedder.weight", cfg.latent_channels, cfg.hidden_size);
        uint32_t txt_hidden = linear(txt_emb, "context_embedder.weight", cfg.text_dim, cfg.hidden_size);

        auto mod_img_all = compute_mods(t_emb_silu, "double_stream_modulation_img.linear.weight", 6);
        auto mod_txt_all = compute_mods(t_emb_silu, "double_stream_modulation_txt.linear.weight", 6);
        auto mod_single_all = compute_mods(t_emb_silu, "single_stream_modulation.linear.weight", 3);

        int32_t slice_t_s[] = {0, 0, 0, 0};
        int32_t slice_txt_e[] = {1, 1, (int32_t)cfg.text_max_seq, (int32_t)cfg.head_dim};
        int32_t slice_step[] = {1, 1, 1, 1};
        uint32_t txt_cos = g.contiguous(g.slice(rope_cos, g.constant({4}, slice_t_s, DType::INT32), g.constant({4}, slice_txt_e, DType::INT32), g.constant({4}, slice_step, DType::INT32)));
        uint32_t txt_sin = g.contiguous(g.slice(rope_sin, g.constant({4}, slice_t_s, DType::INT32), g.constant({4}, slice_txt_e, DType::INT32), g.constant({4}, slice_step, DType::INT32)));

        int32_t slice_img_s[] = {0, 0, (int32_t)cfg.text_max_seq, 0};
        int32_t slice_img_e[] = {1, 1, (int32_t)total_seq_len, (int32_t)cfg.head_dim};
        uint32_t img_cos = g.contiguous(g.slice(rope_cos, g.constant({4}, slice_img_s, DType::INT32), g.constant({4}, slice_img_e, DType::INT32), g.constant({4}, slice_step, DType::INT32)));
        uint32_t img_sin = g.contiguous(g.slice(rope_sin, g.constant({4}, slice_img_s, DType::INT32), g.constant({4}, slice_img_e, DType::INT32), g.constant({4}, slice_step, DType::INT32)));

        for (uint32_t i = 0; i < cfg.num_double_layers; ++i)
        {
            std::string p = "transformer_blocks." + std::to_string(i);

            uint32_t img_mod = apply_mod(layer_norm_atomic(img_hidden, img_seq_len), mod_img_all[0], mod_img_all[1], img_seq_len);
            uint32_t txt_mod = apply_mod(layer_norm_atomic(txt_hidden, cfg.text_max_seq), mod_txt_all[0], mod_txt_all[1], cfg.text_max_seq);

            auto reshape_for_attn = [&](uint32_t x, int L)
            {
                int32_t sh4[] = {1, L, (int32_t)cfg.num_heads, (int32_t)cfg.head_dim};
                int32_t p[] = {0, 2, 1, 3};
                return g.contiguous(g.permute(g.reshape(x, g.constant({4}, sh4, DType::INT32)), g.constant({4}, p, DType::INT32)));
            };

            uint32_t img_q = apply_rope_2d_consecutive(rms_norm_atomic(reshape_for_attn(linear(img_mod, p + ".attn.to_q.weight", cfg.hidden_size, cfg.hidden_size), img_seq_len), p + ".attn.norm_q.weight", img_seq_len, cfg.num_heads, cfg.head_dim), img_cos, img_sin, img_seq_len);
            uint32_t img_k = apply_rope_2d_consecutive(rms_norm_atomic(reshape_for_attn(linear(img_mod, p + ".attn.to_k.weight", cfg.hidden_size, cfg.hidden_size), img_seq_len), p + ".attn.norm_k.weight", img_seq_len, cfg.num_heads, cfg.head_dim), img_cos, img_sin, img_seq_len);
            uint32_t img_v = reshape_for_attn(linear(img_mod, p + ".attn.to_v.weight", cfg.hidden_size, cfg.hidden_size), img_seq_len);

            uint32_t txt_q = apply_rope_2d_consecutive(rms_norm_atomic(reshape_for_attn(linear(txt_mod, p + ".attn.add_q_proj.weight", cfg.hidden_size, cfg.hidden_size), cfg.text_max_seq), p + ".attn.norm_added_q.weight", cfg.text_max_seq, cfg.num_heads, cfg.head_dim), txt_cos, txt_sin, cfg.text_max_seq);
            uint32_t txt_k = apply_rope_2d_consecutive(rms_norm_atomic(reshape_for_attn(linear(txt_mod, p + ".attn.add_k_proj.weight", cfg.hidden_size, cfg.hidden_size), cfg.text_max_seq), p + ".attn.norm_added_k.weight", cfg.text_max_seq, cfg.num_heads, cfg.head_dim), txt_cos, txt_sin, cfg.text_max_seq);
            uint32_t txt_v = reshape_for_attn(linear(txt_mod, p + ".attn.add_v_proj.weight", cfg.hidden_size, cfg.hidden_size), cfg.text_max_seq);

            int32_t ax2 = 2;
            uint32_t joint_k = g.concat({txt_k, img_k}, g.constant({1}, &ax2, DType::INT32));
            uint32_t joint_v = g.concat({txt_v, img_v}, g.constant({1}, &ax2, DType::INT32));

            auto attn = [&](uint32_t q, uint32_t k, uint32_t v, int L_q)
            {
                int32_t p_k[] = {0, 1, 3, 2};
                uint32_t scores = g.mul(g.dot(q, g.contiguous(g.permute(k, g.constant({4}, p_k, DType::INT32)))), expand_scalar_to_4d(1.0f / std::sqrt((float)cfg.head_dim), 1, cfg.num_heads, L_q, total_seq_len));
                int32_t ax = -1;
                uint32_t shifted = g.add(scores, g.neg(repeat_ax(g.max(scores, g.constant({1}, &ax, DType::INT32)), total_seq_len, 3)));
                uint32_t exps = g.pow(expand_scalar_to_4d(2.7182818f, 1, cfg.num_heads, L_q, total_seq_len), shifted);
                uint32_t probs = g.div(exps, repeat_ax(g.sum(exps, g.constant({1}, &ax, DType::INT32)), total_seq_len, 3));

                uint32_t ctx = g.dot(probs, v);
                int32_t p_c[] = {0, 2, 1, 3};
                ctx = g.contiguous(g.permute(ctx, g.constant({4}, p_c, DType::INT32)));
                int32_t sh3[] = {1, (int32_t)L_q, (int32_t)cfg.hidden_size};
                return g.reshape(ctx, g.constant({3}, sh3, DType::INT32));
            };

            uint32_t img_attn_out = linear(attn(img_q, joint_k, joint_v, img_seq_len), p + ".attn.to_out.0.weight", cfg.hidden_size, cfg.hidden_size);
            uint32_t txt_attn_out = linear(attn(txt_q, joint_k, joint_v, cfg.text_max_seq), p + ".attn.to_add_out.weight", cfg.hidden_size, cfg.hidden_size);

            int32_t sh3_gate[] = {1, 1, (int32_t)cfg.hidden_size};
            img_hidden = g.add(img_hidden, g.mul(repeat_ax(g.reshape(mod_img_all[2], g.constant({3}, sh3_gate, DType::INT32)), img_seq_len, 1), img_attn_out));
            txt_hidden = g.add(txt_hidden, g.mul(repeat_ax(g.reshape(mod_txt_all[2], g.constant({3}, sh3_gate, DType::INT32)), cfg.text_max_seq, 1), txt_attn_out));

            auto ffn = [&](uint32_t h, uint32_t shift, uint32_t scale, uint32_t gate, const std::string &pfx, int seq)
            {
                uint32_t m = apply_mod(layer_norm_atomic(h, seq), shift, scale, seq);
                uint32_t ff = linear(m, pfx + "_in.weight", cfg.hidden_size, cfg.mlp_hidden * 2);
                int32_t s_gate[] = {0, 0, 0};
                int32_t e_gate[] = {1, (int32_t)seq, (int32_t)cfg.mlp_hidden};
                int32_t step[] = {1, 1, 1};
                uint32_t ff_gate = g.contiguous(g.slice(ff, g.constant({3}, s_gate, DType::INT32), g.constant({3}, e_gate, DType::INT32), g.constant({3}, step, DType::INT32)));
                int32_t s_up[] = {0, 0, (int32_t)cfg.mlp_hidden};
                int32_t e_up[] = {1, (int32_t)seq, (int32_t)cfg.mlp_hidden * 2};
                uint32_t ff_up = g.contiguous(g.slice(ff, g.constant({3}, s_up, DType::INT32), g.constant({3}, e_up, DType::INT32), g.constant({3}, step, DType::INT32)));

                uint32_t out = linear(g.mul(silu_atomic(ff_gate, 1, seq, cfg.mlp_hidden), ff_up), pfx + "_out.weight", cfg.mlp_hidden, cfg.hidden_size);
                return g.add(h, g.mul(repeat_ax(g.reshape(gate, g.constant({3}, sh3_gate, DType::INT32)), seq, 1), out));
            };

            img_hidden = ffn(img_hidden, mod_img_all[3], mod_img_all[4], mod_img_all[5], p + ".ff.linear", img_seq_len);
            txt_hidden = ffn(txt_hidden, mod_txt_all[3], mod_txt_all[4], mod_txt_all[5], p + ".ff_context.linear", cfg.text_max_seq);
        }

        int32_t ax1 = 1;
        uint32_t combined = g.concat({txt_hidden, img_hidden}, g.constant({1}, &ax1, DType::INT32));

        for (uint32_t i = 0; i < cfg.num_single_layers; ++i)
        {
            std::string p = "single_transformer_blocks." + std::to_string(i);
            uint32_t mod = apply_mod(layer_norm_atomic(combined, total_seq_len), mod_single_all[0], mod_single_all[1], total_seq_len);
            uint32_t fused = linear(mod, p + ".attn.to_qkv_mlp_proj.weight", cfg.hidden_size, cfg.hidden_size * 3 + cfg.mlp_hidden * 2);

            auto slice_feat = [&](int start, int end)
            {
                int32_t s[] = {0, 0, start};
                int32_t e[] = {1, (int32_t)total_seq_len, end};
                int32_t step[] = {1, 1, 1};
                return g.contiguous(g.slice(fused, g.constant({3}, s, DType::INT32), g.constant({3}, e, DType::INT32), g.constant({3}, step, DType::INT32)));
            };

            auto reshape_for_attn = [&](uint32_t x)
            {
                int32_t sh4[] = {1, (int32_t)total_seq_len, (int32_t)cfg.num_heads, (int32_t)cfg.head_dim};
                int32_t p_[] = {0, 2, 1, 3};
                return g.contiguous(g.permute(g.reshape(x, g.constant({4}, sh4, DType::INT32)), g.constant({4}, p_, DType::INT32)));
            };

            uint32_t q = apply_rope_2d_consecutive(rms_norm_atomic(reshape_for_attn(slice_feat(0, cfg.hidden_size)), p + ".attn.norm_q.weight", total_seq_len, cfg.num_heads, cfg.head_dim), rope_cos, rope_sin, total_seq_len);
            uint32_t k = apply_rope_2d_consecutive(rms_norm_atomic(reshape_for_attn(slice_feat(cfg.hidden_size, cfg.hidden_size * 2)), p + ".attn.norm_k.weight", total_seq_len, cfg.num_heads, cfg.head_dim), rope_cos, rope_sin, total_seq_len);
            uint32_t v = reshape_for_attn(slice_feat(cfg.hidden_size * 2, cfg.hidden_size * 3));
            uint32_t mlp_gate = slice_feat(cfg.hidden_size * 3, cfg.hidden_size * 3 + cfg.mlp_hidden);
            uint32_t mlp_up = slice_feat(cfg.hidden_size * 3 + cfg.mlp_hidden, cfg.hidden_size * 3 + cfg.mlp_hidden * 2);

            int32_t p_k[] = {0, 1, 3, 2};
            uint32_t scores = g.mul(g.dot(q, g.contiguous(g.permute(k, g.constant({4}, p_k, DType::INT32)))), expand_scalar_to_4d(1.0f / std::sqrt((float)cfg.head_dim), 1, cfg.num_heads, total_seq_len, total_seq_len));
            int32_t ax_ = -1;
            uint32_t shifted = g.add(scores, g.neg(repeat_ax(g.max(scores, g.constant({1}, &ax_, DType::INT32)), total_seq_len, 3)));
            uint32_t exps = g.pow(expand_scalar_to_4d(2.7182818f, 1, cfg.num_heads, total_seq_len, total_seq_len), shifted);
            uint32_t probs = g.div(exps, repeat_ax(g.sum(exps, g.constant({1}, &ax_, DType::INT32)), total_seq_len, 3));

            uint32_t ctx = g.dot(probs, v);
            int32_t p_c[] = {0, 2, 1, 3};
            ctx = g.reshape(g.contiguous(g.permute(ctx, g.constant({4}, p_c, DType::INT32))), g.constant({3}, std::vector<int32_t>{1, (int32_t)total_seq_len, (int32_t)cfg.hidden_size}.data(), DType::INT32));

            uint32_t mlp_out = g.mul(silu_atomic(mlp_gate, 1, total_seq_len, cfg.mlp_hidden), mlp_up);
            int32_t ax_concat = 2;
            uint32_t out = linear(g.concat({ctx, mlp_out}, g.constant({1}, &ax_concat, DType::INT32)), p + ".attn.to_out.weight", cfg.hidden_size + cfg.mlp_hidden, cfg.hidden_size);

            int32_t sh3_gate[] = {1, 1, (int32_t)cfg.hidden_size};
            combined = g.add(combined, g.mul(repeat_ax(g.reshape(mod_single_all[2], g.constant({3}, sh3_gate, DType::INT32)), total_seq_len, 1), out));
        }

        int32_t slice_s[] = {0, (int32_t)cfg.text_max_seq, 0};
        int32_t slice_e[] = {1, (int32_t)total_seq_len, (int32_t)cfg.hidden_size};
        int32_t slice_step_out[] = {1, 1, 1};
        uint32_t img_out = g.contiguous(g.slice(combined, g.constant({3}, slice_s, DType::INT32), g.constant({3}, slice_e, DType::INT32), g.constant({3}, slice_step_out, DType::INT32)));

        auto final_mods = compute_mods(t_emb_silu, "norm_out.linear.weight", 2);
        uint32_t mod_out = apply_mod(layer_norm_atomic(img_out, img_seq_len), final_mods[1], final_mods[0], img_seq_len);
        uint32_t output = linear(mod_out, "proj_out.weight", cfg.hidden_size, cfg.latent_channels);

        int32_t sh_out[] = {1, (int32_t)latent_h, (int32_t)latent_w, (int32_t)cfg.latent_channels};
        int32_t p_out[] = {0, 3, 1, 2};
        return g.contiguous(g.permute(g.reshape(output, g.constant({4}, sh_out, DType::INT32)), g.constant({4}, p_out, DType::INT32)));
    }
};