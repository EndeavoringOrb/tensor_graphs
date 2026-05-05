// File: tensor_graphs_cpp/models/flux-klein-4b-vae.hpp
#pragma once

class FluxVAEDecoder : public FluxGraphBase
{
private:
    FluxConfig cfg;
    MemoryManager &mem;
    uint32_t initial_h, initial_w;

    uint32_t group_norm_atomic(uint32_t x, const std::string &w_name, const std::string &b_name, int N, int C, int H, int W)
    {
        int G = 32;
        int M = (C / G) * H * W;
        int32_t sh3[] = {N, G, M};
        uint32_t x_3d = g.reshape(x, g.constant({3}, sh3, DType::INT32));

        int32_t ax = 2;
        uint32_t ax_node = g.constant({1}, &ax, DType::INT32);
        uint32_t m_node = expand_scalar_to_3d((float)M, N, G, 1);
        uint32_t mean = repeat_ax(g.div(g.sum(x_3d, ax_node), m_node), M, 2);
        uint32_t x_sub = g.add(x_3d, g.neg(mean));

        uint32_t var = g.div(g.sum(g.mul(x_sub, x_sub), ax_node), m_node);
        uint32_t std = g.pow(g.add(var, expand_scalar_to_3d(1e-4f, N, G, 1)), expand_scalar_to_3d(0.5f, N, G, 1));
        uint32_t x_norm = g.reshape(g.mul(x_sub, repeat_ax(g.div(expand_scalar_to_3d(1.0f, N, G, 1), std), M, 2)), g.constant({4}, std::vector<int32_t>{N, C, H, W}.data(), DType::INT32));

        int32_t w_sh[] = {1, C, 1, 1};
        uint32_t w_4d = repeat_ax(repeat_ax(repeat_ax(g.reshape(weight(w_name), g.constant({4}, w_sh, DType::INT32)), N, 0), H, 2), W, 3);
        uint32_t b_4d = repeat_ax(repeat_ax(repeat_ax(g.reshape(weight(b_name), g.constant({4}, w_sh, DType::INT32)), N, 0), H, 2), W, 3);

        return g.add(g.mul(x_norm, w_4d), b_4d);
    }

    uint32_t conv2d_atomic(uint32_t x, const std::string &w_name, const std::string &b_name, int kernel, int stride, int padding, int H, int W)
    {
        // 1. Prepare constants for im2col
        uint32_t k_const = g.constant({1}, &kernel, DType::INT32);
        uint32_t s_const = g.constant({1}, &stride, DType::INT32);
        uint32_t p_const = g.constant({1}, &padding, DType::INT32);

        // 2. Im2Col: [N, C_in, H, W] -> [N, K, M]
        // where K = C_in * kernel * kernel, M = H_out * W_out
        uint32_t col = g.im2col(x, k_const, s_const, p_const);

        // 3. Handle Weights
        uint32_t w_id = weight(w_name);

        // Use the original INPUT node's shape directly since CAST doesn't inherit it during build time
        std::vector<uint32_t> w_shape = g.getNode(g.getNode(w_id).parentIds[0]).getShape();
        int C_out = (int)w_shape[0];
        int K = 0;
        if (w_shape.size() == 4)
        {
            K = w_shape[1] * w_shape[2] * w_shape[3];
        }
        else if (w_shape.size() == 2)
        {
            K = w_shape[1];
        }
        else
        {
            Error::throw_err("Conv2D expects 2D or 4D weights");
        }

        // Weight Reshape: [C_out, K]
        std::vector<int32_t> w_flat_sh = {C_out, K};
        uint32_t w_flat = g.reshape(w_id, g.constant({2}, w_flat_sh.data(), DType::INT32));

        // 4. Prepare Im2Col for Dot Product
        // col is [N, K, M] -> Permute to [K, N, M]
        int32_t p_col[] = {1, 0, 2};
        uint32_t col_perm = g.permute(col, g.constant({3}, p_col, DType::INT32));

        // Reshape [K, N, M] -> [K, N*M]
        int N = 1; // Flux VAE batch size is always 1
        int H_out = (H + 2 * padding - kernel) / stride + 1;
        int W_out = (W + 2 * padding - kernel) / stride + 1;
        int M = H_out * W_out;

        std::vector<int32_t> col_flat_sh = {K, N * M};
        uint32_t col_flat = g.reshape(col_perm, g.constant({2}, col_flat_sh.data(), DType::INT32));

        // 5. Dot Product: [C_out, K] @ [K, N*M] -> [C_out, N*M]
        uint32_t out_flat = g.dot(w_flat, col_flat);

        // 6. Reshape back to NCHW
        // [C_out, N*M] -> [C_out, N, M]
        std::vector<int32_t> out_res1_sh = {C_out, N, M};
        uint32_t out_res1 = g.reshape(out_flat, g.constant({3}, out_res1_sh.data(), DType::INT32));

        // [C_out, N, M] -> [N, C_out, M]
        int32_t p_out[] = {1, 0, 2};
        uint32_t out_perm = g.permute(out_res1, g.constant({3}, p_out, DType::INT32));

        std::vector<int32_t> final_sh = {N, C_out, H_out, W_out};
        uint32_t result = g.reshape(out_perm, g.constant({4}, final_sh.data(), DType::INT32));

        // 7. Add Bias
        if (!b_name.empty())
        {
            uint32_t b_id = weight(b_name);
            std::vector<int32_t> b_sh = {1, C_out, 1, 1};
            uint32_t b_reshaped = g.reshape(b_id, g.constant({4}, b_sh.data(), DType::INT32));
            result = g.add(result, b_reshaped);
        }

        return result;
    }

    uint32_t upsample2x_atomic(uint32_t x)
    {
        return repeat_ax(repeat_ax(x, 2, 2), 2, 3);
    }

    uint32_t resblock(uint32_t x, const std::string &pfx, int C, int H, int W)
    {
        uint32_t h = group_norm_atomic(x, pfx + ".norm1.weight", pfx + ".norm1.bias", 1, C, H, W);
        h = silu_atomic(h, 1, C, H * W); // Works nicely since it's element-wise
        h = conv2d_atomic(h, pfx + ".conv1.weight", pfx + ".conv1.bias", 3, 1, 1, H, W);

        h = group_norm_atomic(h, pfx + ".norm2.weight", pfx + ".norm2.bias", 1, C, H, W);
        h = silu_atomic(h, 1, C, H * W);
        h = conv2d_atomic(h, pfx + ".conv2.weight", pfx + ".conv2.bias", 3, 1, 1, H, W);

        return g.add(h, x);
    }

    uint32_t attnblock(uint32_t x, const std::string &pfx, int C, int H, int W)
    {
        uint32_t h = group_norm_atomic(x, pfx + ".group_norm.weight", pfx + ".group_norm.bias", 1, C, H, W);
        uint32_t q = conv2d_atomic(h, pfx + ".to_q.weight", pfx + ".to_q.bias", 1, 1, 0, H, W);
        uint32_t k = conv2d_atomic(h, pfx + ".to_k.weight", pfx + ".to_k.bias", 1, 1, 0, H, W);
        uint32_t v = conv2d_atomic(h, pfx + ".to_v.weight", pfx + ".to_v.bias", 1, 1, 0, H, W);

        auto flat = [&](uint32_t t)
        {
            int32_t p[] = {0, 2, 3, 1};
            int32_t sh3[] = {1, (int32_t)(H * W), (int32_t)C};
            return g.reshape(g.contiguous(g.permute(t, g.constant({4}, p, DType::INT32))), g.constant({3}, sh3, DType::INT32));
        };

        int32_t p_k[] = {0, 2, 1};
        uint32_t scores = g.mul(g.dot(flat(q), g.contiguous(g.permute(flat(k), g.constant({3}, p_k, DType::INT32)))), expand_scalar_to_3d(1.0f / std::sqrt((float)C), 1, H * W, H * W));

        int32_t ax = -1;
        uint32_t max_s = repeat_ax(g.max(scores, g.constant({1}, &ax, DType::INT32)), H * W, 2);
        uint32_t exps = g.pow(expand_scalar_to_3d(2.7182818f, 1, H * W, H * W), g.add(scores, g.neg(max_s)));
        uint32_t probs = g.div(exps, repeat_ax(g.sum(exps, g.constant({1}, &ax, DType::INT32)), H * W, 2));

        uint32_t out = g.dot(probs, flat(v));
        int32_t sh4[] = {1, (int32_t)H, (int32_t)W, (int32_t)C};
        int32_t p_out[] = {0, 3, 1, 2};
        out = conv2d_atomic(g.contiguous(g.permute(g.reshape(out, g.constant({4}, sh4, DType::INT32)), g.constant({4}, p_out, DType::INT32))), pfx + ".to_out.0.weight", pfx + ".to_out.0.bias", 1, 1, 0, H, W);
        return g.add(out, x);
    }

public:
    FluxVAEDecoder(FluxConfig config, Graph &graph, MemoryManager &memory, const std::string &weight_path, uint32_t h, uint32_t w)
        : FluxGraphBase(graph, weight_path), cfg(config), mem(memory), initial_h(h), initial_w(w) {}

    uint32_t build_graph(uint32_t latent)
    {
        int32_t b_sh[] = {1, (int32_t)cfg.latent_channels, 1, 1};
        uint32_t mu = repeat_ax(repeat_ax(g.reshape(weight("bn.running_mean"), g.constant({4}, b_sh, DType::INT32)), initial_h, 2), initial_w, 3);
        uint32_t var = repeat_ax(repeat_ax(g.reshape(weight("bn.running_var"), g.constant({4}, b_sh, DType::INT32)), initial_h, 2), initial_w, 3);

        uint32_t std = g.pow(g.add(var, expand_scalar_to_4d(1e-4f, 1, cfg.latent_channels, initial_h, initial_w)), expand_scalar_to_4d(0.5f, 1, cfg.latent_channels, initial_h, initial_w));
        uint32_t h = g.add(g.mul(latent, std), mu);

        int32_t sh1[] = {1, (int32_t)cfg.vae_z_channels, (int32_t)cfg.patch_size, (int32_t)cfg.patch_size, (int32_t)initial_h, (int32_t)initial_w};
        int32_t p_u[] = {0, 1, 4, 2, 5, 3};
        uint32_t curr_h = initial_h * cfg.patch_size;
        uint32_t curr_w = initial_w * cfg.patch_size;
        int32_t sh2[] = {1, (int32_t)cfg.vae_z_channels, (int32_t)curr_h, (int32_t)curr_w};
        h = g.reshape(g.contiguous(g.permute(g.reshape(h, g.constant({6}, sh1, DType::INT32)), g.constant({6}, p_u, DType::INT32))), g.constant({4}, sh2, DType::INT32));

        h = conv2d_atomic(h, "post_quant_conv.weight", "post_quant_conv.bias", 1, 1, 0, curr_h, curr_w);
        h = conv2d_atomic(h, "decoder.conv_in.weight", "decoder.conv_in.bias", 3, 1, 1, curr_h, curr_w);

        h = resblock(h, "decoder.mid_block.resnets.0", cfg.vae_channels, curr_h, curr_w);
        h = attnblock(h, "decoder.mid_block.attentions.0", cfg.vae_channels, curr_h, curr_w);
        h = resblock(h, "decoder.mid_block.resnets.1", cfg.vae_channels, curr_h, curr_w);

        for (int level = 3; level >= 0; --level)
        {
            for (int r = 0; r < 3; ++r)
            {
                std::string p = "decoder.up_blocks." + std::to_string(3 - level) + ".resnets." + std::to_string(r);
                h = resblock(h, p, cfg.vae_channels, curr_h, curr_w);
            }
            if (level > 0)
            {
                h = upsample2x_atomic(h);
                curr_h *= 2;
                curr_w *= 2;
                h = conv2d_atomic(h, "decoder.up_blocks." + std::to_string(3 - level) + ".upsamplers.0.conv.weight", "decoder.up_blocks." + std::to_string(3 - level) + ".upsamplers.0.conv.bias", 3, 1, 1, curr_h, curr_w);
            }
        }

        h = group_norm_atomic(h, "decoder.conv_norm_out.weight", "decoder.conv_norm_out.bias", 1, cfg.vae_channels, curr_h, curr_w);
        h = silu_atomic(h, 1, cfg.vae_channels, curr_h * curr_w);
        return conv2d_atomic(h, "decoder.conv_out.weight", "decoder.conv_out.bias", 3, 1, 1, curr_h, curr_w);
    }
};