#pragma once

// =============================================================================
// jina-embeddings-v5-omni-nano-retrieval — C++ implementation (image embedding)
// =============================================================================
//
// Architecture (mirrors the Python reference `modeling_llava_eurobert_audio.py`):
//
//   Vision tower : Qwen3VLVisionModel
//                  - patch_embed: Conv3d(T=2, P=16, P=16) → Linear(1536, 768)
//                  - 12 × Qwen3VL vision blocks (RMSNorm eps=1e-6, fused QKV,
//                    2-D RoPE, bidirectional, GELU-MLP)
//                  - No position embeddings (uses 2-D RoPE), no post-LayerNorm
//
//   Merger       : PretrainedMerger (top-level, NOT inside vision_tower)
//                  - LayerNorm(768, eps=1e-6) on patch features
//                  - reshape to merge 2×2 spatial → 3072-dim
//                  - linear_fc1: 3072 → 3072, GELU, linear_fc2: 3072 → 768
//
//   Text encoder : LlamaModel (EuroBERT, bidirectional / is_causal=False)
//                  - 12 layers, hidden=768, heads=12, head_dim=64
//                  - RMSNorm eps=1e-5, 1-D RoPE (theta=1,000,000)
//                  - SwiGLU MLP (intermediate=3072), no biases on attn/mlp
//                  - Input = image features (input_ids all = image_token_index,
//                    so embed_tokens output is fully overwritten by features)
//
//   Pooling      : last-token pooling (position = num_merged_tokens - 1)
//   Output       : L2-normalized 768-dim embedding
//
// For a 512×512 input (satisfies min_pixels=262144):
//   grid 32×32 → 1024 patches → after 2×2 merge → 256 text-encoder tokens
//
// Weight naming (matches the Python `LlavaEuroBertAudioForEmbedding` state_dict):
//   vision_tower.patch_embed.proj.weight            (768, 3, 2, 16, 16)
//   vision_tower.blocks.{i}.norm1.weight            (768,)
//   vision_tower.blocks.{i}.attn.qkv.{weight,bias}  (2304, 768) / (2304,)
//   vision_tower.blocks.{i}.attn.proj.{weight,bias} (768, 768) / (768,)
//   vision_tower.blocks.{i}.norm2.weight            (768,)
//   vision_tower.blocks.{i}.mlp.fc1.{weight,bias}   (3072, 768) / (3072,)
//   vision_tower.blocks.{i}.mlp.fc2.{weight,bias}   (768, 3072) / (768,)
//   merger.norm.{weight,bias}                       (768,) / (768,)
//   merger.linear_fc1.{weight,bias}                 (3072, 3072) / (3072,)
//   merger.linear_fc2.{weight,bias}                 (768, 3072) / (768,)
//   language_model.layers.{i}.input_layernorm.weight            (768,)
//   language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight   (768, 768)
//   language_model.layers.{i}.post_attention_layernorm.weight   (768,)
//   language_model.layers.{i}.mlp.{gate,up,down}_proj.weight    (3072,768)/(3072,768)/(768,3072)
//   language_model.norm.weight                      (768,)
// =============================================================================

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <tuple>

struct JinaV5Config
{
    // ---- Image / patch geometry (matches preprocessor_config.json) ----
    uint32_t image_size = 512;        // 512×512 → 262144 px (== min_pixels)
    uint32_t patch_size = 16;         // Qwen3VL patch_size
    uint32_t temporal_patch_size = 2; // Qwen3VL temporal_patch_size (image duplicated to 2 frames)
    uint32_t spatial_merge_size = 2;  // Qwen3VL spatial_merge_size
    uint32_t in_channels = 3;

    // ---- Vision tower (Qwen3VLVisionModel) ----
    uint32_t vision_hidden_size = 768;
    uint32_t vision_intermediate_size = 3072;
    uint32_t vision_num_heads = 12;
    uint32_t vision_head_dim = 64; // 768 / 12
    uint32_t vision_num_layers = 12;
    float vision_rms_eps = 1e-6f;
    float vision_rope_theta = 10000.0f;

    // ---- Merger (PretrainedMerger) ----
    // merger_hidden = vision_hidden_size * spatial_merge_size^2 = 3072
    // merger_out    = text_hidden_size = 768

    // ---- Text encoder (EuroBERT / LlamaModel, bidirectional) ----
    uint32_t text_hidden_size = 768;
    uint32_t text_intermediate_size = 3072; // from config.json (was wrongly 2048)
    uint32_t text_num_heads = 12;
    uint32_t text_head_dim = 64; // 768 / 12
    uint32_t text_num_layers = 12;
    float text_rms_eps = 1e-5f;         // config.json rms_norm_eps (was wrongly 1e-6)
    float text_rope_theta = 1000000.0f; // config.json rope_theta (was wrongly 1e4)

    // ---- Derived (computed in init) ----
    uint32_t grid_h;        // image_size / patch_size = 32
    uint32_t grid_w;        // = 32
    uint32_t num_patches;   // grid_h * grid_w = 1024
    uint32_t merged_grid_h; // grid_h / spatial_merge_size = 16
    uint32_t merged_grid_w; // = 16
    uint32_t num_merged;    // merged_grid_h * merged_grid_w = 256
    uint32_t patch_dim;     // temporal_patch_size * patch_size^2 * in_channels = 1536
    uint32_t merged_dim;    // vision_hidden_size * spatial_merge_size^2 = 3072

    JinaV5Config()
        : grid_h(image_size / patch_size),
          grid_w(image_size / patch_size),
          num_patches(grid_h * grid_w),
          merged_grid_h(grid_h / spatial_merge_size),
          merged_grid_w(grid_w / spatial_merge_size),
          num_merged(merged_grid_h * merged_grid_w),
          patch_dim(temporal_patch_size * patch_size * patch_size * in_channels),
          merged_dim(vision_hidden_size * spatial_merge_size * spatial_merge_size) {}
};

class JinaV5OmniNanoRetrievalModel
{
private:
    JinaV5Config cfg;
    Graph &g;
    MemoryManager &mem;
    const std::string w_path;

    // -------------------------------------------------------------------------
    // Weight loading
    // -------------------------------------------------------------------------
    uint32_t weight(const std::string &name)
    {
        uint32_t raw_weight = g.weight(w_path, name);
        return g.cast(raw_weight, DType::FLOAT32);
    }

    // -------------------------------------------------------------------------
    // Shape / repeat helpers (kept from the original implementation)
    // -------------------------------------------------------------------------
    uint32_t repeat_ax(uint32_t id, uint32_t repeats, uint32_t axis)
    {
        if (repeats <= 1)
            return id;
        int32_t r = repeats, a = axis;
        return g.repeat(id, g.constant({1}, &r, DType::INT32),
                        g.constant({1}, &a, DType::INT32));
    }

    uint32_t repeat_3d_axis(uint32_t tensor_id, uint32_t repeats, uint32_t axis)
    {
        if (repeats <= 1)
            return tensor_id;
        int32_t rep[] = {(int32_t)repeats};
        int32_t ax[] = {(int32_t)axis};
        return g.repeat(tensor_id,
                        g.constant({1}, rep, DType::INT32),
                        g.constant({1}, ax, DType::INT32));
    }

    uint32_t repeat_4d_axis(uint32_t tensor_id, uint32_t repeats, uint32_t axis)
    {
        if (repeats <= 1)
            return tensor_id;
        int32_t rep[] = {(int32_t)repeats};
        int32_t ax[] = {(int32_t)axis};
        return g.repeat(tensor_id,
                        g.constant({1}, rep, DType::INT32),
                        g.constant({1}, ax, DType::INT32));
    }

    uint32_t expand_scalar_to_1d(float val, uint32_t d0)
    {
        uint32_t node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh1[] = {1};
        return repeat_3d_axis(g.reshape(node, g.constant({1}, sh1, DType::INT32)), d0, 0);
    }

    uint32_t expand_scalar_to_2d(float val, uint32_t dim0, uint32_t dim1)
    {
        uint32_t node = g.constant({1}, &val, DType::FLOAT32);
        int32_t shape_2d[] = {1, 1};
        uint32_t out = g.reshape(node, g.constant({2}, shape_2d, DType::INT32));
        if (dim0 > 1)
            out = repeat_3d_axis(out, dim0, 0);
        if (dim1 > 1)
            out = repeat_3d_axis(out, dim1, 1);
        return out;
    }

    uint32_t expand_scalar_to_3d(float val, uint32_t dim0, uint32_t dim1, uint32_t dim2)
    {
        uint32_t node = g.constant({1}, &val, DType::FLOAT32);
        int32_t shape_3d[] = {1, 1, 1};
        uint32_t out = g.reshape(node, g.constant({3}, shape_3d, DType::INT32));
        if (dim0 > 1)
            out = repeat_3d_axis(out, dim0, 0);
        if (dim1 > 1)
            out = repeat_3d_axis(out, dim1, 1);
        if (dim2 > 1)
            out = repeat_3d_axis(out, dim2, 2);
        return out;
    }

    uint32_t expand_scalar_to_4d(float val, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3)
    {
        uint32_t node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh4[] = {1, 1, 1, 1};
        uint32_t out = g.reshape(node, g.constant({4}, sh4, DType::INT32));
        if (d0 > 1)
            out = repeat_4d_axis(out, d0, 0);
        if (d1 > 1)
            out = repeat_4d_axis(out, d1, 1);
        if (d2 > 1)
            out = repeat_4d_axis(out, d2, 2);
        if (d3 > 1)
            out = repeat_4d_axis(out, d3, 3);
        return out;
    }

    uint32_t expand_1d_to_3d(uint32_t vec_id, uint32_t vec_len, uint32_t dim0, uint32_t dim1)
    {
        int32_t shape_3d[] = {1, 1, (int32_t)vec_len};
        uint32_t out = g.reshape(vec_id, g.constant({3}, shape_3d, DType::INT32));
        if (dim0 > 1)
            out = repeat_3d_axis(out, dim0, 0);
        if (dim1 > 1)
            out = repeat_3d_axis(out, dim1, 1);
        return out;
    }

    // -------------------------------------------------------------------------
    // Normalisation primitives
    // -------------------------------------------------------------------------

    // LayerNorm (used by the merger — Python uses nn.LayerNorm(eps=1e-6)).
    uint32_t layer_norm(uint32_t x, const std::string &w_name,
                        const std::string &b_name, uint32_t S, uint32_t D,
                        float eps = 1e-6f)
    {
        int32_t ax_val = -1;
        uint32_t axis_node = g.constant({1}, &ax_val, DType::INT32);

        uint32_t sum_x = g.sum(x, axis_node);
        uint32_t d_node = expand_scalar_to_3d((float)D, 1, S, 1);
        uint32_t mean_val = g.div(sum_x, d_node);
        uint32_t mean = repeat_3d_axis(mean_val, D, 2);

        uint32_t x_sub = g.add(x, g.neg(mean));
        uint32_t sq = g.mul(x_sub, x_sub);
        uint32_t sum_sq = g.sum(sq, axis_node);
        uint32_t var = g.div(sum_sq, d_node);

        uint32_t eps_node = expand_scalar_to_3d(eps, 1, S, 1);
        uint32_t var_plus_eps = g.add(var, eps_node);

        uint32_t sqrt_exp = expand_scalar_to_3d(0.5f, 1, S, 1);
        uint32_t std_dev = g.pow(var_plus_eps, sqrt_exp);

        uint32_t one_node = expand_scalar_to_3d(1.0f, 1, S, 1);
        uint32_t inv_std = g.div(one_node, std_dev);
        uint32_t inv_std_expanded = repeat_3d_axis(inv_std, D, 2);

        uint32_t normalized = g.mul(x_sub, inv_std_expanded);

        if (!w_name.empty())
        {
            uint32_t w = weight(w_name);
            uint32_t w_exp = expand_1d_to_3d(w, D, 1, S);
            normalized = g.mul(normalized, w_exp);
        }
        if (!b_name.empty())
        {
            uint32_t b = weight(b_name);
            uint32_t b_exp = expand_1d_to_3d(b, D, 1, S);
            normalized = g.add(normalized, b_exp);
        }
        return normalized;
    }

    // RMSNorm — eps is parameterised so vision (1e-6) and text (1e-5) can differ.
    uint32_t rms_norm(uint32_t x_id, const std::string &w_name,
                      uint32_t S, uint32_t D, float eps)
    {
        uint32_t x_sq = g.mul(x_id, x_id);
        int32_t axis_val = -1;
        uint32_t axis_node = g.constant({1}, &axis_val, DType::INT32);
        uint32_t sum_sq = g.sum(x_sq, axis_node);
        uint32_t n_node = expand_scalar_to_3d((float)D, 1, S, 1);
        uint32_t mean_sq = g.div(sum_sq, n_node);
        uint32_t eps_expanded = expand_scalar_to_3d(eps, 1, S, 1);
        uint32_t mean_sq_plus_eps = g.add(mean_sq, eps_expanded);
        uint32_t sqrt_node = expand_scalar_to_3d(0.5f, 1, S, 1);
        uint32_t std = g.pow(mean_sq_plus_eps, sqrt_node);
        uint32_t one_node = expand_scalar_to_3d(1.0f, 1, S, 1);
        uint32_t inv_std = g.div(one_node, std);
        uint32_t inv_std_expanded = repeat_3d_axis(inv_std, D, 2);
        uint32_t x_norm = g.mul(x_id, inv_std_expanded);
        uint32_t w = weight(w_name);
        uint32_t w_exp = expand_1d_to_3d(w, D, 1, S);
        return g.mul(x_norm, w_exp);
    }

    // -------------------------------------------------------------------------
    // Linear (weight expected 2-D: [out_d, in_d])
    // -------------------------------------------------------------------------
    uint32_t linear(uint32_t x, const std::string &w_name, const std::string &b_name,
                    uint32_t in_d, uint32_t out_d, uint32_t S)
    {
        uint32_t w = weight(w_name);
        int32_t p[] = {1, 0};
        uint32_t w_t = g.permute(w, g.constant({2}, p, DType::INT32));
        w_t = g.contiguous(w_t);
        int32_t sh3[] = {1, (int32_t)in_d, (int32_t)out_d};
        uint32_t out = g.dot(x, g.reshape(w_t, g.constant({3}, sh3, DType::INT32)));
        if (!b_name.empty())
        {
            uint32_t b = weight(b_name);
            uint32_t b_exp = expand_1d_to_3d(b, out_d, 1, S);
            out = g.add(out, b_exp);
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Activation primitives
    // -------------------------------------------------------------------------
    uint32_t silu_atomic(uint32_t x_id, uint32_t N, uint32_t L, uint32_t D)
    {
        uint32_t neg_one = expand_scalar_to_3d(-1.0f, N, L, D);
        uint32_t neg_x = g.mul(x_id, neg_one);
        uint32_t e_node = expand_scalar_to_3d(2.718281828459045f, N, L, D);
        uint32_t exp_neg_x = g.pow(e_node, neg_x);
        uint32_t one_node = expand_scalar_to_3d(1.0f, N, L, D);
        uint32_t den = g.add(one_node, exp_neg_x);
        uint32_t sigmoid = g.div(one_node, den);
        return g.mul(x_id, sigmoid);
    }

    // GELU with tanh approximation (matches `gelu_pytorch_tanh`).
    uint32_t gelu_atomic(uint32_t x_id, uint32_t S, uint32_t D)
    {
        uint32_t c1_node = expand_scalar_to_3d(0.044715f, 1, S, D);
        uint32_t c2_node = expand_scalar_to_3d(0.79788456f, 1, S, D);
        uint32_t x_sq = g.mul(x_id, x_id);
        uint32_t x_cube = g.mul(x_sq, x_id);
        uint32_t term1 = g.mul(x_cube, c1_node);
        uint32_t term2 = g.add(x_id, term1);
        uint32_t term3 = g.mul(term2, c2_node);

        uint32_t neg_two = expand_scalar_to_3d(-2.0f, 1, S, D);
        uint32_t two = expand_scalar_to_3d(2.0f, 1, S, D);
        uint32_t e_node = expand_scalar_to_3d(2.7182818f, 1, S, D);
        uint32_t one_node = expand_scalar_to_3d(1.0f, 1, S, D);
        uint32_t neg_2x = g.mul(term3, neg_two);
        uint32_t exp_neg_2x = g.pow(e_node, neg_2x);
        uint32_t den = g.add(one_node, exp_neg_2x);
        uint32_t quotient = g.div(two, den);
        uint32_t tanh_result = g.add(quotient, g.neg(one_node));

        uint32_t term4 = g.add(one_node, tanh_result);
        uint32_t half_node = expand_scalar_to_3d(0.5f, 1, S, D);
        uint32_t term5 = g.mul(x_id, half_node);
        return g.mul(term5, term4);
    }

    // -------------------------------------------------------------------------
    // RoPE
    // -------------------------------------------------------------------------

    // Apply RoPE to a 4-D Q/K tensor of shape (1, n_groups, S, head_dim).
    // cos/sin have shape (1, 1, S, head_dim).
    uint32_t apply_rope(uint32_t x, uint32_t cos, uint32_t sin,
                        uint32_t n_groups, uint32_t head_dim, uint32_t S)
    {
        int32_t starts1[] = {0, 0, 0, 0};
        int32_t ends1[] = {1, (int32_t)n_groups, (int32_t)S, (int32_t)head_dim / 2};
        int32_t starts2[] = {0, 0, 0, (int32_t)head_dim / 2};
        int32_t ends2[] = {1, (int32_t)n_groups, (int32_t)S, (int32_t)head_dim};
        int32_t steps[] = {1, 1, 1, 1};

        uint32_t x1 = g.slice(x, g.constant({4}, starts1, DType::INT32),
                              g.constant({4}, ends1, DType::INT32),
                              g.constant({4}, steps, DType::INT32));
        uint32_t x2 = g.slice(x, g.constant({4}, starts2, DType::INT32),
                              g.constant({4}, ends2, DType::INT32),
                              g.constant({4}, steps, DType::INT32));

        int32_t ax = 3;
        uint32_t rotated = g.concat({g.neg(x2), x1}, g.constant({1}, &ax, DType::INT32));

        uint32_t cos_exp = repeat_ax(cos, n_groups, 1);
        uint32_t sin_exp = repeat_ax(sin, n_groups, 1);
        return g.add(g.mul(x, cos_exp), g.mul(rotated, sin_exp));
    }

    // 1-D RoPE for the text encoder (positions 0..S-1, theta from config).
    std::tuple<uint32_t, uint32_t> compute_rope_1d(uint32_t S, uint32_t head_dim, float theta)
    {
        int32_t start_val = 0, stop_val = (int32_t)head_dim, step_val = 2;
        uint32_t indices_int = g.arange(g.constant({1}, &start_val, DType::INT32),
                                        g.constant({1}, &stop_val, DType::INT32),
                                        g.constant({1}, &step_val, DType::INT32));
        uint32_t indices = g.cast(indices_int, DType::FLOAT32);

        uint32_t h_dim_node = expand_scalar_to_1d((float)head_dim, head_dim / 2);
        uint32_t exp = g.div(indices, h_dim_node);
        uint32_t theta_node = expand_scalar_to_1d(theta, head_dim / 2);
        uint32_t inv_freq = g.div(expand_scalar_to_1d(1.0f, head_dim / 2),
                                  g.pow(theta_node, exp));

        int32_t pos_stop = (int32_t)S;
        int32_t pos_step = 1;
        uint32_t pos = g.cast(g.arange(g.constant({1}, &start_val, DType::INT32),
                                       g.constant({1}, &pos_stop, DType::INT32),
                                       g.constant({1}, &pos_step, DType::INT32)),
                              DType::FLOAT32);

        int32_t sh_col[] = {(int32_t)S, 1};
        uint32_t pos_col = repeat_ax(g.reshape(pos, g.constant({2}, sh_col, DType::INT32)),
                                     head_dim / 2, 1);
        int32_t sh_row[] = {1, (int32_t)head_dim / 2};
        uint32_t freq_row = repeat_ax(g.reshape(inv_freq, g.constant({2}, sh_row, DType::INT32)),
                                      S, 0);

        uint32_t angles_half = g.mul(pos_col, freq_row);
        int32_t ax = 1;
        uint32_t angles = g.concat({angles_half, angles_half},
                                   g.constant({1}, &ax, DType::INT32));

        int32_t sh4[] = {1, 1, (int32_t)S, (int32_t)head_dim};
        uint32_t sh4_node = g.constant({4}, sh4, DType::INT32);
        return {g.reshape(g.cos(angles), sh4_node),
                g.reshape(g.sin(angles), sh4_node)};
    }

    // 2-D RoPE for the vision tower — positions are (h, w) of each patch.
    // inv_freq (head_dim/2) is split into two halves: first half modulated by
    // h position, second half by w position. Then duplicated to head_dim and
    // applied with the standard rotate_half rule.
    std::tuple<uint32_t, uint32_t> compute_rope_2d(uint32_t grid_h, uint32_t grid_w,
                                                   uint32_t head_dim, float theta)
    {
        uint32_t S = grid_h * grid_w;
        uint32_t half = head_dim / 2; // 32
        uint32_t quarter = half / 2;  // 16

        // ---- inv_freq: 1 / theta^(arange(0, head_dim, 2) / head_dim)  shape (head_dim/2,)
        int32_t start_val = 0, stop_val = (int32_t)head_dim, step_val = 2;
        uint32_t indices_int = g.arange(g.constant({1}, &start_val, DType::INT32),
                                        g.constant({1}, &stop_val, DType::INT32),
                                        g.constant({1}, &step_val, DType::INT32));
        uint32_t indices = g.cast(indices_int, DType::FLOAT32);
        uint32_t h_dim_node = expand_scalar_to_1d((float)head_dim, half);
        uint32_t exps = g.div(indices, h_dim_node);
        uint32_t theta_node = expand_scalar_to_1d(theta, half);
        uint32_t inv_freq = g.div(expand_scalar_to_1d(1.0f, half),
                                  g.pow(theta_node, exps)); // (half,)

        // ---- Split inv_freq into h-half and w-half
        int32_t starts_h[] = {0};
        int32_t ends_h[] = {(int32_t)quarter};
        int32_t starts_w[] = {(int32_t)quarter};
        int32_t ends_w[] = {(int32_t)half};
        int32_t steps1[] = {1};
        uint32_t inv_freq_h = g.slice(inv_freq,
                                      g.constant({1}, starts_h, DType::INT32),
                                      g.constant({1}, ends_h, DType::INT32),
                                      g.constant({1}, steps1, DType::INT32)); // (quarter,)
        uint32_t inv_freq_w = g.slice(inv_freq,
                                      g.constant({1}, starts_w, DType::INT32),
                                      g.constant({1}, ends_w, DType::INT32),
                                      g.constant({1}, steps1, DType::INT32)); // (quarter,)

        // ---- Build h_pos (S,) and w_pos (S,)
        // h_pos[i] = i / grid_w  (row index),  w_pos[i] = i % grid_w  (col index)
        // h_pos = reshape(arange(grid_h), (grid_h,1))  repeated grid_w times on axis 1 → (grid_h, grid_w)
        // w_pos = reshape(arange(grid_w), (1,grid_w))  repeated grid_h times on axis 0 → (grid_h, grid_w)
        int32_t gh_stop = (int32_t)grid_h, gw_stop = (int32_t)grid_w, one_step = 1;
        uint32_t h_arr = g.cast(g.arange(g.constant({1}, &start_val, DType::INT32),
                                         g.constant({1}, &gh_stop, DType::INT32),
                                         g.constant({1}, &one_step, DType::INT32)),
                                DType::FLOAT32); // (grid_h,)
        uint32_t w_arr = g.cast(g.arange(g.constant({1}, &start_val, DType::INT32),
                                         g.constant({1}, &gw_stop, DType::INT32),
                                         g.constant({1}, &one_step, DType::INT32)),
                                DType::FLOAT32); // (grid_w,)

        int32_t sh_col[] = {(int32_t)grid_h, 1};
        uint32_t h_col2d = g.reshape(h_arr, g.constant({2}, sh_col, DType::INT32));
        uint32_t h_pos_2d = g.contiguous(repeat_ax(h_col2d, grid_w, 1)); // (grid_h, grid_w)
        int32_t sh_row[] = {1, (int32_t)grid_w};
        uint32_t w_row2d = g.reshape(w_arr, g.constant({2}, sh_row, DType::INT32));
        uint32_t w_pos_2d = g.contiguous(repeat_ax(w_row2d, grid_h, 0)); // (grid_h, grid_w)

        int32_t sh_S[] = {(int32_t)S};
        uint32_t h_pos = g.reshape(h_pos_2d, g.constant({1}, sh_S, DType::INT32)); // (S,)
        uint32_t w_pos = g.reshape(w_pos_2d, g.constant({1}, sh_S, DType::INT32)); // (S,)

        // ---- angles_h = h_pos[:, None] * inv_freq_h[None, :]   shape (S, quarter)
        int32_t sh_S_1[] = {(int32_t)S, 1};
        int32_t sh_1_q[] = {1, (int32_t)quarter};
        h_pos_2d = g.reshape(h_pos, g.constant({2}, sh_S_1, DType::INT32));
        uint32_t invf_h_2d = g.reshape(inv_freq_h, g.constant({2}, sh_1_q, DType::INT32));
        uint32_t h_pos_exp = repeat_ax(h_pos_2d, quarter, 1); // (S, quarter)
        uint32_t invf_h_exp = repeat_ax(invf_h_2d, S, 0);     // (S, quarter)
        uint32_t angles_h = g.mul(h_pos_exp, invf_h_exp);     // (S, quarter)

        // ---- angles_w = w_pos[:, None] * inv_freq_w[None, :]   shape (S, quarter)
        w_pos_2d = g.reshape(w_pos, g.constant({2}, sh_S_1, DType::INT32));
        uint32_t invf_w_2d = g.reshape(inv_freq_w, g.constant({2}, sh_1_q, DType::INT32));
        uint32_t w_pos_exp = repeat_ax(w_pos_2d, quarter, 1);
        uint32_t invf_w_exp = repeat_ax(invf_w_2d, S, 0);
        uint32_t angles_w = g.mul(w_pos_exp, invf_w_exp); // (S, quarter)

        // ---- angles = cat([angles_h, angles_w], -1)  shape (S, half)
        //      emb    = cat([angles, angles],     -1)  shape (S, head_dim)
        int32_t ax1 = 1;
        uint32_t angles = g.concat({angles_h, angles_w},
                                   g.constant({1}, &ax1, DType::INT32)); // (S, half)
        uint32_t emb = g.concat({angles, angles},
                                g.constant({1}, &ax1, DType::INT32)); // (S, head_dim)

        uint32_t cos_t = g.cos(emb);
        uint32_t sin_t = g.sin(emb);

        // Reshape to (1, 1, S, head_dim) for broadcasting across heads.
        int32_t sh4[] = {1, 1, (int32_t)S, (int32_t)head_dim};
        uint32_t sh4_node = g.constant({4}, sh4, DType::INT32);
        return {g.reshape(cos_t, sh4_node), g.reshape(sin_t, sh4_node)};
    }

    // -------------------------------------------------------------------------
    // Softmax along the last axis of a 4-D tensor (1, num_heads, S, S)
    // -------------------------------------------------------------------------
    uint32_t softmax_4d(uint32_t scores, uint32_t S, uint32_t num_heads)
    {
        int32_t axis_val = -1;
        uint32_t axis_node = g.constant({1}, &axis_val, DType::INT32);
        uint32_t max_s = g.max(scores, axis_node);
        uint32_t max_expanded = repeat_4d_axis(max_s, S, 3);
        uint32_t shifted = g.add(scores, g.neg(max_expanded));

        uint32_t e_node = expand_scalar_to_4d(2.7182818f, 1, num_heads, S, S);
        uint32_t exps = g.pow(e_node, shifted);
        uint32_t sums = g.sum(exps, axis_node);
        uint32_t sums_expanded = repeat_4d_axis(sums, S, 3);
        return g.div(exps, sums_expanded);
    }

    // =========================================================================
    //  VISION TOWER (Qwen3VL)
    // =========================================================================

    // Qwen3VL patch_embed is a Conv3d(in=3, out=768, kernel=(2,16,16), stride=(2,16,16)).
    // For static images the two temporal frames are identical, so Conv3d on
    // (B, 3, T=2, H, W) is equivalent to a Linear on a 1536-dim patch vector
    // ordered as (C_in, T_patch, P, P) = (3, 2, 16, 16) → 1536. We simply
    // reshape the (768, 3, 2, 16, 16) weight to (768, 1536) and matmul.
    uint32_t patch_embed(uint32_t x, uint32_t num_patches)
    {
        uint32_t w = weight("vision_tower.patch_embed.proj.weight"); // (768, 3, 2, 16, 16)
        int32_t sh2[] = {768, 1536};
        uint32_t w_2d = g.reshape(w, g.constant({2}, sh2, DType::INT32)); // (768, 1536)
        int32_t p[] = {1, 0};
        uint32_t w_t = g.contiguous(g.permute(w_2d, g.constant({2}, p, DType::INT32))); // (1536, 768)
        int32_t sh3[] = {1, 1536, 768};
        uint32_t w_3d = g.reshape(w_t, g.constant({3}, sh3, DType::INT32));
        // (1, num_patches, 1536) × (1, 1536, 768) → (1, num_patches, 768)
        return g.dot(x, w_3d);
    }

    // Fused-QKV attention with 2-D RoPE, bidirectional (no causal mask).
    uint32_t vision_attention(uint32_t x, int layer_idx,
                              uint32_t cos, uint32_t sin, uint32_t S)
    {
        std::string prefix = "vision_tower.blocks." + std::to_string(layer_idx) + ".attn.";

        // Fused QKV: weight (2304, 768), bias (2304,)
        uint32_t qkv = linear(x, prefix + "qkv.weight", prefix + "qkv.bias",
                              cfg.vision_hidden_size, 3 * cfg.vision_hidden_size, S); // (1, S, 2304)

        // Split Q / K / V along the last dim
        int32_t steps[] = {1, 1, 1};
        int32_t sq[] = {0, 0, 0}, eq[] = {1, (int32_t)S, (int32_t)cfg.vision_hidden_size};
        int32_t sk[] = {0, 0, (int32_t)cfg.vision_hidden_size},
                ek[] = {1, (int32_t)S, (int32_t)(2 * cfg.vision_hidden_size)};
        int32_t sv[] = {0, 0, (int32_t)(2 * cfg.vision_hidden_size)},
                ev[] = {1, (int32_t)S, (int32_t)(3 * cfg.vision_hidden_size)};
        uint32_t q = g.contiguous(g.slice(qkv, g.constant({3}, sq, DType::INT32),
                             g.constant({3}, eq, DType::INT32),
                             g.constant({3}, steps, DType::INT32)));
        uint32_t k = g.contiguous(g.slice(qkv, g.constant({3}, sk, DType::INT32),
                             g.constant({3}, ek, DType::INT32),
                             g.constant({3}, steps, DType::INT32)));
        uint32_t v = g.contiguous(g.slice(qkv, g.constant({3}, sv, DType::INT32),
                             g.constant({3}, ev, DType::INT32),
                             g.constant({3}, steps, DType::INT32)));

        // Reshape to (1, num_heads, S, head_dim)
        int32_t sh4[] = {1, (int32_t)S, (int32_t)cfg.vision_num_heads, (int32_t)cfg.vision_head_dim};
        int32_t p_attn[] = {0, 2, 1, 3};
        q = g.contiguous(g.permute(g.reshape(q, g.constant({4}, sh4, DType::INT32)),
                                   g.constant({4}, p_attn, DType::INT32)));
        k = g.contiguous(g.permute(g.reshape(k, g.constant({4}, sh4, DType::INT32)),
                                   g.constant({4}, p_attn, DType::INT32)));
        v = g.contiguous(g.permute(g.reshape(v, g.constant({4}, sh4, DType::INT32)),
                                   g.constant({4}, p_attn, DType::INT32)));

        // 2-D RoPE on Q and K
        q = apply_rope(q, cos, sin, cfg.vision_num_heads, cfg.vision_head_dim, S);
        k = apply_rope(k, cos, sin, cfg.vision_num_heads, cfg.vision_head_dim, S);

        // Scaled dot-product attention (bidirectional — no causal mask)
        float scale_val = 1.0f / std::sqrt((float)cfg.vision_head_dim);
        q = g.mul(q, expand_scalar_to_4d(scale_val, 1, cfg.vision_num_heads, S, cfg.vision_head_dim));

        int32_t p_k[] = {0, 1, 3, 2};
        uint32_t k_t = g.contiguous(g.permute(k, g.constant({4}, p_k, DType::INT32)));
        uint32_t scores = g.dot(q, k_t); // (1, num_heads, S, S)

        uint32_t probs = softmax_4d(scores, S, cfg.vision_num_heads);
        uint32_t attn_out = g.dot(probs, v); // (1, num_heads, S, head_dim)

        // Merge heads → (1, S, hidden)
        int32_t p_ctx[] = {0, 2, 1, 3};
        uint32_t ctx_perm = g.contiguous(g.permute(attn_out, g.constant({4}, p_ctx, DType::INT32)));
        int32_t sh3_ctx[] = {1, (int32_t)S, (int32_t)cfg.vision_hidden_size};
        uint32_t ctx_flat = g.reshape(ctx_perm, g.constant({3}, sh3_ctx, DType::INT32));

        return linear(ctx_flat, prefix + "proj.weight", prefix + "proj.bias",
                      cfg.vision_hidden_size, cfg.vision_hidden_size, S);
    }

    uint32_t vision_mlp(uint32_t x, int layer_idx, uint32_t S)
    {
        std::string prefix = "vision_tower.blocks." + std::to_string(layer_idx) + ".mlp.";
        uint32_t h = linear(x, prefix + "linear_fc1.weight", prefix + "linear_fc1.bias",
                            cfg.vision_hidden_size, cfg.vision_intermediate_size, S);
        h = gelu_atomic(h, S, cfg.vision_intermediate_size);
        return linear(h, prefix + "linear_fc2.weight", prefix + "linear_fc2.bias",
                      cfg.vision_intermediate_size, cfg.vision_hidden_size, S);
    }

    uint32_t vision_block(uint32_t x, int layer_idx, uint32_t cos, uint32_t sin, uint32_t S)
    {
        std::string prefix = "vision_tower.blocks." + std::to_string(layer_idx) + ".";
        uint32_t residual = x;

        uint32_t h = rms_norm(x, prefix + "norm1.weight", S, cfg.vision_hidden_size,
                              cfg.vision_rms_eps);
        h = vision_attention(h, layer_idx, cos, sin, S);
        h = g.add(residual, h);
        residual = h;

        h = rms_norm(h, prefix + "norm2.weight", S, cfg.vision_hidden_size,
                     cfg.vision_rms_eps);
        h = vision_mlp(h, layer_idx, S);
        return g.add(residual, h);
    }

    // Full Qwen3VL vision tower: patch_embed → 12 blocks. No final norm
    // (the merger supplies its own LayerNorm).
    uint32_t vit_encoder(uint32_t patch_input)
    {
        uint32_t S = cfg.num_patches; // 1024

        uint32_t h = patch_embed(patch_input, S); // (1, 1024, 768)

        auto [cos, sin] = compute_rope_2d(cfg.grid_h, cfg.grid_w,
                                          cfg.vision_head_dim, cfg.vision_rope_theta);

        for (uint32_t i = 0; i < cfg.vision_num_layers; ++i)
            h = vision_block(h, (int)i, cos, sin, S);

        return h; // (1, 1024, 768)
    }

    // =========================================================================
    //  TEXT ENCODER (EuroBERT / LlamaModel, bidirectional)
    // =========================================================================

    uint32_t text_attention(uint32_t x, int layer_idx,
                            uint32_t cos, uint32_t sin, uint32_t S)
    {
        std::string prefix = "language_model.layers." + std::to_string(layer_idx) + ".self_attn.";
        // EuroBERT config: attention_bias=false → no biases on q/k/v/o projections
        uint32_t q = linear(x, prefix + "q_proj.weight", "",
                            cfg.text_hidden_size, cfg.text_hidden_size, S);
        uint32_t k = linear(x, prefix + "k_proj.weight", "",
                            cfg.text_hidden_size, cfg.text_hidden_size, S);
        uint32_t v = linear(x, prefix + "v_proj.weight", "",
                            cfg.text_hidden_size, cfg.text_hidden_size, S);

        int32_t sh4[] = {1, (int32_t)S, (int32_t)cfg.text_num_heads, (int32_t)cfg.text_head_dim};
        int32_t p_attn[] = {0, 2, 1, 3};
        q = g.contiguous(g.permute(g.reshape(q, g.constant({4}, sh4, DType::INT32)),
                                   g.constant({4}, p_attn, DType::INT32)));
        k = g.contiguous(g.permute(g.reshape(k, g.constant({4}, sh4, DType::INT32)),
                                   g.constant({4}, p_attn, DType::INT32)));
        v = g.contiguous(g.permute(g.reshape(v, g.constant({4}, sh4, DType::INT32)),
                                   g.constant({4}, p_attn, DType::INT32)));

        // 1-D RoPE on Q and K (theta=1,000,000 per config)
        q = apply_rope(q, cos, sin, cfg.text_num_heads, cfg.text_head_dim, S);
        k = apply_rope(k, cos, sin, cfg.text_num_heads, cfg.text_head_dim, S);

        float scale_val = 1.0f / std::sqrt((float)cfg.text_head_dim);
        q = g.mul(q, expand_scalar_to_4d(scale_val, 1, cfg.text_num_heads, S, cfg.text_head_dim));

        int32_t p_k[] = {0, 1, 3, 2};
        uint32_t k_t = g.contiguous(g.permute(k, g.constant({4}, p_k, DType::INT32)));
        uint32_t scores = g.dot(q, k_t); // (1, num_heads, S, S)

        // Bidirectional (no causal mask) — EuroBERT sets is_causal=False
        uint32_t probs = softmax_4d(scores, S, cfg.text_num_heads);
        uint32_t attn_out = g.dot(probs, v);

        int32_t p_ctx[] = {0, 2, 1, 3};
        uint32_t ctx_perm = g.contiguous(g.permute(attn_out, g.constant({4}, p_ctx, DType::INT32)));
        int32_t sh3_ctx[] = {1, (int32_t)S, (int32_t)cfg.text_hidden_size};
        uint32_t ctx_flat = g.reshape(ctx_perm, g.constant({3}, sh3_ctx, DType::INT32));

        return linear(ctx_flat, prefix + "o_proj.weight", "",
                      cfg.text_hidden_size, cfg.text_hidden_size, S);
    }

    // SwiGLU MLP — gate * up → down.  intermediate_size = 3072 per config.
    uint32_t text_mlp(uint32_t x, int layer_idx, uint32_t S)
    {
        std::string prefix = "language_model.layers." + std::to_string(layer_idx) + ".mlp.";
        // mlp_bias=false → no biases
        uint32_t gate = linear(x, prefix + "gate_proj.weight", "",
                               cfg.text_hidden_size, cfg.text_intermediate_size, S);
        uint32_t up = linear(x, prefix + "up_proj.weight", "",
                             cfg.text_hidden_size, cfg.text_intermediate_size, S);
        uint32_t gate_silu = silu_atomic(gate, 1, S, cfg.text_intermediate_size);
        uint32_t gate_up = g.mul(gate_silu, up);
        return linear(gate_up, prefix + "down_proj.weight", "",
                      cfg.text_intermediate_size, cfg.text_hidden_size, S);
    }

    uint32_t l2_normalize(uint32_t x, uint32_t D)
    {
        uint32_t x_sq = g.mul(x, x);
        int32_t axis_val = -1;
        uint32_t axis_node = g.constant({1}, &axis_val, DType::INT32);
        uint32_t sum_sq = g.sum(x_sq, axis_node);
        uint32_t eps_expanded = expand_scalar_to_2d(1e-12f, 1, 1);
        uint32_t sum_sq_plus_eps = g.add(sum_sq, eps_expanded);
        uint32_t std = g.pow(sum_sq_plus_eps, expand_scalar_to_2d(0.5f, 1, 1));
        uint32_t inv_std = g.div(expand_scalar_to_2d(1.0f, 1, 1), std);

        int32_t rep[] = {(int32_t)D};
        int32_t axis[] = {1};
        uint32_t inv_std_expanded = g.repeat(inv_std,
                                             g.constant({1}, rep, DType::INT32),
                                             g.constant({1}, axis, DType::INT32));
        return g.mul(x, inv_std_expanded);
    }

public:
    JinaV5OmniNanoRetrievalModel(JinaV5Config &_cfg, Graph &_g, MemoryManager &_mem,
                                 std::string _w_path)
        : cfg(_cfg), g(_g), mem(_mem), w_path(std::move(_w_path)) {}

    // Build the image-embedding graph.
    //
    // Input  : patch_input  shape (1, num_patches, patch_dim) = (1, 1024, 1536)
    //          — produced by embed.cpp's Qwen2VL-style preprocessor.
    // Output : L2-normalised 768-dim embedding, shape (1, 768).
    uint32_t build_graph(uint32_t patch_input)
    {
        // ---- 1. Vision tower (Qwen3VL) -------------------------------------
        // (1, 1024, 1536) → (1, 1024, 768)
        uint32_t patch_feats = vit_encoder(patch_input);

        // ---- 2. Merger (PretrainedMerger) ----------------------------------
        // LayerNorm(768, eps=1e-6) on patch features
        uint32_t ln_patches = layer_norm(patch_feats,
                                         "merger.norm.weight", "merger.norm.bias",
                                         cfg.num_patches, cfg.vision_hidden_size);

        // Reshape (1, num_patches, 768) → (1, grid_h, grid_w, 768)
        int32_t sh4_grid[] = {1, (int32_t)cfg.grid_h, (int32_t)cfg.grid_w, (int32_t)cfg.vision_hidden_size};
        uint32_t grid = g.reshape(ln_patches, g.constant({4}, sh4_grid, DType::INT32));

        // Split into 2×2 blocks: (1, merged_h, 2, merged_w, 2, 768)
        int32_t sh6_split[] = {1,
                               (int32_t)cfg.merged_grid_h, (int32_t)cfg.spatial_merge_size,
                               (int32_t)cfg.merged_grid_w, (int32_t)cfg.spatial_merge_size,
                               (int32_t)cfg.vision_hidden_size};
        uint32_t split = g.reshape(grid, g.constant({6}, sh6_split, DType::INT32));

        // Permute so the 2×2 block dims are adjacent: (1, merged_h, merged_w, 2, 2, 768)
        int32_t perm6[] = {0, 1, 3, 2, 4, 5};
        uint32_t perm_split = g.contiguous(g.permute(split, g.constant({6}, perm6, DType::INT32)));

        // Flatten to (1, num_merged, merged_dim) = (1, 256, 3072)
        int32_t sh3_merged[] = {1, (int32_t)cfg.num_merged, (int32_t)cfg.merged_dim};
        uint32_t merged = g.reshape(perm_split, g.constant({3}, sh3_merged, DType::INT32));

        // linear_fc1: 3072 → 3072, GELU, linear_fc2: 3072 → 768
        uint32_t proj1 = linear(merged, "merger.linear_fc1.weight", "merger.linear_fc1.bias",
                                cfg.merged_dim, cfg.merged_dim, cfg.num_merged);
        uint32_t act = gelu_atomic(proj1, cfg.num_merged, cfg.merged_dim);
        uint32_t proj2 = linear(act, "merger.linear_fc2.weight", "merger.linear_fc2.bias",
                                cfg.merged_dim, cfg.text_hidden_size, cfg.num_merged);

        // proj2 is the image-feature sequence that goes into the text encoder.
        // For an image-only input with text="<image>", every position in
        // input_ids is image_token_index, so masked_scatter overwrites the
        // entire embed_tokens output — we can skip embed_tokens and feed
        // proj2 directly as inputs_embeds.

        // ---- 3. Text encoder (EuroBERT / LlamaModel, bidirectional) -------
        uint32_t h = proj2; // (1, 256, 768)
        uint32_t S = cfg.num_merged;

        auto [cos, sin] = compute_rope_1d(S, cfg.text_head_dim, cfg.text_rope_theta);

        for (uint32_t i = 0; i < cfg.text_num_layers; ++i)
        {
            std::string prefix = "language_model.layers." + std::to_string(i) + ".";
            uint32_t residual = h;

            uint32_t norm1 = rms_norm(h, prefix + "input_layernorm.weight", S,
                                      cfg.text_hidden_size, cfg.text_rms_eps);
            uint32_t attn_out = text_attention(norm1, (int)i, cos, sin, S);
            h = g.add(residual, attn_out);
            residual = h;

            uint32_t norm2 = rms_norm(h, prefix + "post_attention_layernorm.weight", S,
                                      cfg.text_hidden_size, cfg.text_rms_eps);
            uint32_t mlp_out = text_mlp(norm2, (int)i, S);
            h = g.add(residual, mlp_out);
        }

        h = rms_norm(h, "language_model.norm.weight", S,
                     cfg.text_hidden_size, cfg.text_rms_eps);

        // ---- 4. Last-token pooling ----------------------------------------
        // idx = attention_mask.sum(1) - 1.  With a full-ones mask of length S,
        // idx = S - 1.
        int32_t starts_last[] = {0, (int32_t)(S - 1), 0};
        int32_t ends_last[] = {1, (int32_t)S, (int32_t)cfg.text_hidden_size};
        int32_t steps_last[] = {1, 1, 1};
        uint32_t last_token = g.contiguous(g.slice(h,
                                                   g.constant({3}, starts_last, DType::INT32),
                                                   g.constant({3}, ends_last, DType::INT32),
                                                   g.constant({3}, steps_last, DType::INT32)));

        int32_t sh2_out[] = {1, (int32_t)cfg.text_hidden_size};
        uint32_t pooled = g.reshape(last_token, g.constant({2}, sh2_out, DType::INT32));

        // ---- 5. L2 normalize ----------------------------------------------
        return l2_normalize(pooled, cfg.text_hidden_size);
    }
};