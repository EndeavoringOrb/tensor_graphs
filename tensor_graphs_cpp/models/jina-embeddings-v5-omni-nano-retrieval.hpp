#pragma once

// =============================================================================
// jina-embeddings-v5-omni-nano-retrieval — C++ implementation (image embedding)
// =============================================================================
//
// FIXED version — resolves embedding drift (cosine similarity was -0.02, should
// be >= 0.98 vs Python reference).
//
// Architecture (mirrors the Python reference `modeling_llava_eurobert_audio.py`
// and HF `transformers.models.qwen3_vl.modeling_qwen3_vl`):
//
//   Vision tower : Qwen3VLVisionModel
//                  - patch_embed: Conv3d(T=2, P=16, P=16) with bias → Linear(1536, 768) + bias
//                  - pos_embed: nn.Embedding(2304, 768), bilinear-interpolated from 48×48
//                    grid to the 32×32 patch grid, added after patch_embed
//                  - 12 × Qwen3VL vision blocks (LayerNorm eps=1e-6, fused QKV,
//                    2-D RoPE, bidirectional, GELU-MLP)
//                  - No post-LayerNorm; merger supplies its own LayerNorm
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
//                  - Input = chat-template-wrapped image features:
//                    [8 prefix tokens] + [256 image features] + [6 suffix tokens] = 270
//
//   Pooling      : last-token pooling (position = 269, the last suffix token)
//   Output       : L2-normalized 768-dim embedding
//
// For a 512×512 input (satisfies min_pixels=262144):
//   grid 32×32 → 1024 patches → after 2×2 merge → 256 text-encoder image tokens
//   Full text encoder sequence: 8 + 256 + 6 = 270 tokens
//
// =============================================================================
// BUGS FIXED (vs original C++ implementation):
//
//   #1  patch_embed: now loads and adds vision_tower.patch_embed.proj.bias
//       (Qwen3VLVisionPatchEmbed uses nn.Conv3d(..., bias=True))
//
//   #2  pos_embed: now loads vision_tower.pos_embed.weight (2304, 768) and
//       computes bilinear interpolation from the 48×48 learned grid to the
//       32×32 patch grid, adding the result to the patch_embed output.
//       (Qwen3VLVisionModel.forward does: hidden_states = patch_embed(x);
//        hidden_states = hidden_states + bilinear_interp(pos_embed))
//
//   #3  Vision block norms: now use LayerNorm (with bias) instead of RMSNorm.
//       (Qwen3VLVisionBlock uses nn.LayerNorm(eps=1e-6), not RMSNorm)
//
//   #4  2-D RoPE: now computes inv_freq with dim = head_dim // 2 = 32 (giving
//       16 elements) and uses the SAME inv_freq for both h and w positions.
//       Previously used dim = head_dim = 64 (32 elements) and split into two
//       different 16-element halves — both halves were wrong.
//       (HF Qwen3VLVisionRotaryEmbedding: dim = head_dim // 2, single inv_freq)
//
//   #5  Chat template tokens: the text encoder now receives the full 270-token
//       sequence [8 prefix + 256 image + 6 suffix], matching the Python
//       _build_eval_image_prompt → apply_chat_template → tokenize pipeline.
//       Prefix token IDs: [27, 91, 318, 5011, 91, 29, 882, 198]
//         ("<", "|", "im", "_start", "|", ">", "user", "\n")
//       Suffix token IDs: [27, 91, 318, 6345, 91, 397]
//         ("<", "|", "im", "_end", "|", ">\n")
//       Embeddings are gathered from language_model.embed_tokens.weight.
//       Last-token pooling now picks position 269 (the last suffix token).
//       Previously: only 256 image features were fed, pooling picked 255.
//
//   #6  GELU: now uses exact erf-based GELU (via Abramowitz-Stegun erf
//       approximation, max error ~1.5e-7) instead of the tanh approximation.
//       (Qwen3VL uses nn.GELU() = exact, not nn.GELU(approximate='tanh'))
//
// Weight naming (matches the Python LlavaEuroBertAudioForEmbedding state_dict):
//   vision_tower.patch_embed.proj.{weight,bias}        (768, 3, 2, 16, 16) / (768,)
//   vision_tower.pos_embed.weight                      (2304, 768)
//   vision_tower.blocks.{i}.norm1.{weight,bias}        (768,) / (768,)   ← LayerNorm
//   vision_tower.blocks.{i}.attn.qkv.{weight,bias}     (2304, 768) / (2304,)
//   vision_tower.blocks.{i}.attn.proj.{weight,bias}    (768, 768) / (768,)
//   vision_tower.blocks.{i}.norm2.{weight,bias}        (768,) / (768,)   ← LayerNorm
//   vision_tower.blocks.{i}.mlp.linear_fc1.{weight,bias}  (3072, 768) / (3072,)
//   vision_tower.blocks.{i}.mlp.linear_fc2.{weight,bias}  (768, 3072) / (768,)
//   merger.norm.{weight,bias}                          (768,) / (768,)   ← LayerNorm
//   merger.linear_fc1.{weight,bias}                    (3072, 3072) / (3072,)
//   merger.linear_fc2.{weight,bias}                    (768, 3072) / (768,)
//   language_model.embed_tokens.weight                 (128260, 768)
//   language_model.layers.{i}.input_layernorm.weight   (768,)            ← RMSNorm (no bias)
//   language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight  (768, 768) (no bias)
//   language_model.layers.{i}.post_attention_layernorm.weight  (768,)    ← RMSNorm (no bias)
//   language_model.layers.{i}.mlp.{gate,up,down}_proj.weight   (3072,768)/(3072,768)/(768,3072) (no bias)
//   language_model.norm.weight                         (768,)            ← RMSNorm (no bias)
// =============================================================================

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <tuple>
#include <fstream>
#include <sstream>

// -----------------------------------------------------------------------------
// Chat-template token IDs.
//
// The Python reference builds the text-encoder input by calling
// processor.apply_chat_template() on "<|vision_start|><image><|vision_end|>",
// which expands to "<|im_start|>user\n<image>×N<|im_end|>\n" after the
// processor strips the vision_start/vision_end markers (see
// processing_llava_eurobert.LlavaEuroBertProcessor).  The tokenizer then
// BPE-tokenises that string.
//
// For the EuroBERT tokenizer shipped with this model (vocab_size=128260,
// bos=1, eos=2, image_token_index=128259), `<|im_start|>` and `<|im_end|>`
// are NOT registered as single special tokens — they fall through to the
// raw Llama BPE path and split into the 6-token sequences below.  The
// IDs are stable for this tokenizer version.
// -----------------------------------------------------------------------------
static const int32_t CHAT_PREFIX_TOKEN_IDS[] = {
    27, 91, 318, 5011, 91, 29, 882, 198 // < | im _start | > user \n
};
static constexpr uint32_t CHAT_PREFIX_LEN = 8;

static const int32_t CHAT_SUFFIX_TOKEN_IDS[] = {
    27, 91, 318, 6345, 91, 397, 128001 // < | im _end | >\n
};
static constexpr uint32_t CHAT_SUFFIX_LEN = 7;

struct JinaV5Config
{
    // ---- Image / patch geometry (runtime-configured) ----
    // image_h and image_w must be divisible by (patch_size * spatial_merge_size = 32).
    // For 512×512: grid 32×32 → 1024 patches → 256 merged → 270 text tokens.
    // For 480×544: grid 30×34 → 1020 patches → 255 merged → 269 text tokens.
    uint32_t image_h; // set at runtime (default 512)
    uint32_t image_w; // set at runtime (default 512)
    uint32_t patch_size = 16;
    uint32_t temporal_patch_size = 2;
    uint32_t spatial_merge_size = 2;
    uint32_t in_channels = 3;

    // ---- Vision tower (Qwen3VLVisionModel) ----
    uint32_t vision_hidden_size = 768;
    uint32_t vision_intermediate_size = 3072;
    uint32_t vision_num_heads = 12;
    uint32_t vision_head_dim = 64; // 768 / 12
    uint32_t vision_num_layers = 12;
    float vision_ln_eps = 1e-6f; // LayerNorm eps (was rms_eps — now LayerNorm)
    float vision_rope_theta = 10000.0f;

    // ---- Position embedding (Qwen3VL pos_embed) ----
    uint32_t num_position_embeddings = 2304; // from config.json vision_config
    uint32_t num_grid_per_side = 48;         // sqrt(2304)

    // ---- Merger (PretrainedMerger) ----
    // merger_hidden = vision_hidden_size * spatial_merge_size^2 = 3072
    // merger_out    = text_hidden_size = 768

    // ---- Text encoder (EuroBERT / LlamaModel, bidirectional) ----
    uint32_t text_hidden_size = 768;
    uint32_t text_intermediate_size = 3072;
    uint32_t text_num_heads = 12;
    uint32_t text_head_dim = 64;
    uint32_t text_num_layers = 12;
    float text_rms_eps = 1e-5f;
    float text_rope_theta = 1000000.0f;

    // ---- Derived (computed in init) ----
    uint32_t grid_h;        // image_h / patch_size
    uint32_t grid_w;        // image_w / patch_size
    uint32_t num_patches;   // grid_h * grid_w
    uint32_t merged_grid_h; // grid_h / spatial_merge_size
    uint32_t merged_grid_w; // grid_w / spatial_merge_size
    uint32_t num_merged;    // merged_grid_h * merged_grid_w
    uint32_t patch_dim;     // 1536
    uint32_t merged_dim;    // 3072

    // ---- Text encoder sequence layout ----
    // [prefix tokens] + [image features] + [suffix tokens]
    uint32_t text_prefix_len;
    uint32_t text_image_len; // = num_merged
    uint32_t text_suffix_len;
    uint32_t text_seq_len; // prefix_len + num_merged + suffix_len

    // Constructor takes runtime image dimensions.
    // Both must be divisible by (patch_size * spatial_merge_size = 32).
    JinaV5Config(uint32_t img_h = 512, uint32_t img_w = 512)
        : image_h(img_h), image_w(img_w),
          grid_h(img_h / patch_size),
          grid_w(img_w / patch_size),
          num_patches(grid_h * grid_w),
          merged_grid_h(grid_h / spatial_merge_size),
          merged_grid_w(grid_w / spatial_merge_size),
          num_merged(merged_grid_h * merged_grid_w),
          patch_dim(temporal_patch_size * patch_size * patch_size * in_channels),
          merged_dim(vision_hidden_size * spatial_merge_size * spatial_merge_size),
          text_prefix_len(CHAT_PREFIX_LEN),
          text_image_len(num_merged),
          text_suffix_len(CHAT_SUFFIX_LEN),
          text_seq_len((CHAT_PREFIX_LEN + CHAT_SUFFIX_LEN) +
                       num_merged) {}
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
    // Shape / repeat helpers
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

    // LayerNorm — used by vision blocks (norm1, norm2) and merger (norm).
    // nn.LayerNorm(eps=1e-6) has both weight and bias.
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

    // RMSNorm — used by text encoder (LlamaModel).  No bias.
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

    // SiLU: x * sigmoid(x) — used by text encoder SwiGLU.
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

    // Exact GELU using erf, via Abramowitz-Stegun approximation (max err ~1.5e-7).
    // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    //
    // erf(x) for x >= 0:
    //   t = 1 / (1 + p*|x|),  p = 0.3275911
    //   erf ≈ 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
    //   a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429
    // For x < 0: erf(x) = -erf(-x), i.e. erf is odd → erf(x) = sign(x) * erf(|x|).
    uint32_t gelu_exact(uint32_t x_id, uint32_t S, uint32_t D)
    {
        // x_scaled = x / sqrt(2)
        uint32_t inv_sqrt2 = expand_scalar_to_3d(0.7071067811865475f, 1, S, D);
        uint32_t x_scaled = g.mul(x_id, inv_sqrt2);

        // abs(x_scaled) = sqrt(x_scaled^2)
        uint32_t xs_sq = g.mul(x_scaled, x_scaled);
        uint32_t half = expand_scalar_to_3d(0.5f, 1, S, D);
        uint32_t abs_xs = g.pow(xs_sq, half);

        // sign(x_scaled) = x_scaled / (|x_scaled| + eps)
        uint32_t eps_node = expand_scalar_to_3d(1e-12f, 1, S, D);
        uint32_t abs_xs_eps = g.add(abs_xs, eps_node);
        uint32_t sign_xs = g.div(x_scaled, abs_xs_eps);

        // t = 1 / (1 + p * |x_scaled|)
        uint32_t p_node = expand_scalar_to_3d(0.3275911f, 1, S, D);
        uint32_t p_abs = g.mul(p_node, abs_xs);
        uint32_t one_node = expand_scalar_to_3d(1.0f, 1, S, D);
        uint32_t denom = g.add(one_node, p_abs);
        uint32_t t = g.div(one_node, denom);

        // poly = a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        uint32_t t2 = g.mul(t, t);
        uint32_t t3 = g.mul(t2, t);
        uint32_t t4 = g.mul(t3, t);
        uint32_t t5 = g.mul(t4, t);

        uint32_t a1 = expand_scalar_to_3d(0.254829592f, 1, S, D);
        uint32_t a2 = expand_scalar_to_3d(-0.284496736f, 1, S, D);
        uint32_t a3 = expand_scalar_to_3d(1.421413741f, 1, S, D);
        uint32_t a4 = expand_scalar_to_3d(-1.453152027f, 1, S, D);
        uint32_t a5 = expand_scalar_to_3d(1.061405429f, 1, S, D);

        uint32_t poly = g.mul(a1, t);
        poly = g.add(poly, g.mul(a2, t2));
        poly = g.add(poly, g.mul(a3, t3));
        poly = g.add(poly, g.mul(a4, t4));
        poly = g.add(poly, g.mul(a5, t5));

        // exp(-x_scaled^2)
        uint32_t e_node = expand_scalar_to_3d(2.718281828459045f, 1, S, D);
        uint32_t neg_xs_sq = g.neg(xs_sq);
        uint32_t exp_neg_xs_sq = g.pow(e_node, neg_xs_sq);

        // erf_pos = 1 - poly * exp(-x_scaled^2)
        uint32_t product = g.mul(poly, exp_neg_xs_sq);
        uint32_t erf_pos = g.add(one_node, g.neg(product));

        // erf(x_scaled) = sign(x_scaled) * erf_pos
        uint32_t erf_val = g.mul(sign_xs, erf_pos);

        // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        uint32_t one_plus_erf = g.add(one_node, erf_val);
        uint32_t half_x = g.mul(x_id, half);
        return g.mul(half_x, one_plus_erf);
    }

    // -------------------------------------------------------------------------
    // RoPE
    // -------------------------------------------------------------------------

    // Apply RoPE to a 4-D Q/K tensor of shape (1, n_groups, S, head_dim).
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
    // Uses dim = head_dim (standard Llama RoPE).
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

    // 2-D RoPE for the vision tower — FIXED.
    //
    // HF Qwen3VLVisionRotaryEmbedding(dim = head_dim // 2 = 32, theta = 10000):
    //   inv_freq = 1 / theta^(arange(0, 32, 2) / 32)   → 16 elements
    //   rotary_pos_emb = (pos_ids.unsqueeze(-1) * inv_freq).flatten(1)  → (S, 32)
    //     first 16: h * inv_freq,  last 16: w * inv_freq  (SAME inv_freq)
    //   emb = cat([rotary_pos_emb, rotary_pos_emb], -1)  → (S, 64)
    //   cos, sin = cos(emb), sin(emb)
    //
    // The original C++ used dim = head_dim = 64 (32-element inv_freq) and split
    // into two DIFFERENT 16-element halves — both halves were wrong.
    std::tuple<uint32_t, uint32_t> compute_rope_2d(uint32_t grid_h, uint32_t grid_w,
                                                   uint32_t head_dim, float theta)
    {
        uint32_t S = grid_h * grid_w;
        uint32_t half = head_dim / 2; // 32
        uint32_t quarter = half / 2;  // 16

        // ---- inv_freq: 1 / theta^(arange(0, dim, 2) / dim)  with dim = head_dim // 2 = 32
        //   → arange(0, 32, 2) = [0, 2, ..., 30]  (16 elements)
        //   → / 32 → [0, 1/16, ..., 15/16]
        //   → inv_freq = 1 / theta^([0, 1/16, ..., 15/16])  (16 elements)
        int32_t start_val = 0, stop_val = (int32_t)half, step_val = 2; // stop=32, step=2
        uint32_t indices_int = g.arange(g.constant({1}, &start_val, DType::INT32),
                                        g.constant({1}, &stop_val, DType::INT32),
                                        g.constant({1}, &step_val, DType::INT32));
        uint32_t indices = g.cast(indices_int, DType::FLOAT32);
        uint32_t dim_node = expand_scalar_to_1d((float)half, quarter); // 32.0 broadcast to 16
        uint32_t exps = g.div(indices, dim_node);
        uint32_t theta_node = expand_scalar_to_1d(theta, quarter);
        uint32_t inv_freq = g.div(expand_scalar_to_1d(1.0f, quarter),
                                  g.pow(theta_node, exps)); // (quarter,) = (16,)

        // ---- Build h_pos (S,) and w_pos (S,) for row-major patch order.
        int32_t gh_stop = (int32_t)grid_h, gw_stop = (int32_t)grid_w, one_step = 1;
        uint32_t h_arr = g.cast(g.arange(g.constant({1}, &start_val, DType::INT32),
                                         g.constant({1}, &gh_stop, DType::INT32),
                                         g.constant({1}, &one_step, DType::INT32)),
                                DType::FLOAT32);
        uint32_t w_arr = g.cast(g.arange(g.constant({1}, &start_val, DType::INT32),
                                         g.constant({1}, &gw_stop, DType::INT32),
                                         g.constant({1}, &one_step, DType::INT32)),
                                DType::FLOAT32);

        int32_t sh_col[] = {(int32_t)grid_h, 1};
        uint32_t h_col2d = g.reshape(h_arr, g.constant({2}, sh_col, DType::INT32));
        uint32_t h_pos_2d = g.contiguous(repeat_ax(h_col2d, grid_w, 1)); // (grid_h, grid_w)
        int32_t sh_row[] = {1, (int32_t)grid_w};
        uint32_t w_row2d = g.reshape(w_arr, g.constant({2}, sh_row, DType::INT32));
        uint32_t w_pos_2d = g.contiguous(repeat_ax(w_row2d, grid_h, 0)); // (grid_h, grid_w)

        int32_t sh_S[] = {(int32_t)S};
        uint32_t h_pos = g.reshape(h_pos_2d, g.constant({1}, sh_S, DType::INT32)); // (S,)
        uint32_t w_pos = g.reshape(w_pos_2d, g.constant({1}, sh_S, DType::INT32)); // (S,)

        // ---- angles_h = h_pos[:, None] * inv_freq[None, :]   (S, quarter)
        //      angles_w = w_pos[:, None] * inv_freq[None, :]   (S, quarter)  ← SAME inv_freq
        int32_t sh_S_1[] = {(int32_t)S, 1};
        int32_t sh_1_q[] = {1, (int32_t)quarter};
        h_pos_2d = g.reshape(h_pos, g.constant({2}, sh_S_1, DType::INT32));
        uint32_t invf_2d = g.reshape(inv_freq, g.constant({2}, sh_1_q, DType::INT32));
        uint32_t h_pos_exp = repeat_ax(h_pos_2d, quarter, 1);
        uint32_t invf_h_exp = repeat_ax(invf_2d, S, 0);
        uint32_t angles_h = g.mul(h_pos_exp, invf_h_exp); // (S, quarter)

        w_pos_2d = g.reshape(w_pos, g.constant({2}, sh_S_1, DType::INT32));
        uint32_t w_pos_exp = repeat_ax(w_pos_2d, quarter, 1);
        uint32_t invf_w_exp = repeat_ax(invf_2d, S, 0);   // SAME inv_freq
        uint32_t angles_w = g.mul(w_pos_exp, invf_w_exp); // (S, quarter)

        // ---- angles = cat([angles_h, angles_w], -1)  (S, half)
        //      emb    = cat([angles, angles],     -1)  (S, head_dim)
        int32_t ax1 = 1;
        uint32_t angles = g.concat({angles_h, angles_w},
                                   g.constant({1}, &ax1, DType::INT32)); // (S, half)
        uint32_t emb = g.concat({angles, angles},
                                g.constant({1}, &ax1, DType::INT32)); // (S, head_dim)

        uint32_t cos_t = g.cos(emb);
        uint32_t sin_t = g.sin(emb);

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

        uint32_t e_node = expand_scalar_to_4d(2.718281828459045f, 1, num_heads, S, S);
        uint32_t exps = g.pow(e_node, shifted);
        uint32_t sums = g.sum(exps, axis_node);
        uint32_t sums_expanded = repeat_4d_axis(sums, S, 3);
        return g.div(exps, sums_expanded);
    }

    // =========================================================================
    //  VISION TOWER (Qwen3VL) — FIXED
    // =========================================================================

    // Qwen3VL patch_embed: Conv3d with bias.  We reshape the (768, 3, 2, 16, 16)
    // weight to (768, 1536), matmul, then add the (768,) bias.
    // [FIX] Bug #1: now loads and adds proj.bias.
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
        uint32_t out = g.dot(x, w_3d);

        // [FIX] Add bias
        uint32_t b = weight("vision_tower.patch_embed.proj.bias"); // (768,)
        uint32_t b_exp = expand_1d_to_3d(b, 768, 1, num_patches);
        return g.add(out, b_exp);
    }

    // [FIX] Bug #2: Bilinear-interpolated position embedding.
    //
    // Qwen3VLVisionModel has self.pos_embed = nn.Embedding(2304, 768) on a
    // 48×48 grid.  For a 32×32 patch grid, each patch (gi, gj) maps to a
    // continuous coordinate (h_coord, w_coord) in [0, 47]² via linspace, and
    // the pos_embed is bilinearly interpolated from the 4 nearest grid points.
    //
    // We precompute the 4 corner index arrays and 4 weight arrays (all 1024
    // elements) at build time, gather the 4 corner rows from the pos_embed
    // table, and compute the weighted sum → (1, 1024, 768).
    uint32_t compute_pos_embed(uint32_t num_patches)
    {
        const uint32_t side = cfg.num_grid_per_side; // 48
        const uint32_t H = cfg.grid_h;               // 32
        const uint32_t W = cfg.grid_w;               // 32
        const uint32_t D = cfg.vision_hidden_size;   // 768

        // Precompute corner indices and weights (host-side, then upload as constants)
        std::vector<int32_t> idx00(num_patches), idx01(num_patches),
            idx10(num_patches), idx11(num_patches);
        std::vector<float> w00(num_patches), w01(num_patches),
            w10(num_patches), w11(num_patches);

        for (uint32_t gi = 0; gi < H; ++gi)
        {
            for (uint32_t gj = 0; gj < W; ++gj)
            {
                uint32_t pid = gi * W + gj;
                float h_coord = (H == 1) ? 0.0f : (float)gi * (float)(side - 1) / (float)(H - 1);
                float w_coord = (W == 1) ? 0.0f : (float)gj * (float)(side - 1) / (float)(W - 1);
                int32_t h_floor = (int32_t)h_coord;
                int32_t w_floor = (int32_t)w_coord;
                int32_t h_ceil = std::min(h_floor + 1, (int32_t)side - 1);
                int32_t w_ceil = std::min(w_floor + 1, (int32_t)side - 1);
                float h_frac = h_coord - (float)h_floor;
                float w_frac = w_coord - (float)w_floor;

                idx00[pid] = h_floor * side + w_floor;
                idx01[pid] = h_floor * side + w_ceil;
                idx10[pid] = h_ceil * side + w_floor;
                idx11[pid] = h_ceil * side + w_ceil;

                w00[pid] = (1.0f - h_frac) * (1.0f - w_frac);
                w01[pid] = (1.0f - h_frac) * w_frac;
                w10[pid] = h_frac * (1.0f - w_frac);
                w11[pid] = h_frac * w_frac;
            }
        }

        // Upload indices as graph constants (INT32, shape (num_patches,))
        uint32_t idx00_node = g.constant({num_patches}, idx00.data(), DType::INT32);
        uint32_t idx01_node = g.constant({num_patches}, idx01.data(), DType::INT32);
        uint32_t idx10_node = g.constant({num_patches}, idx10.data(), DType::INT32);
        uint32_t idx11_node = g.constant({num_patches}, idx11.data(), DType::INT32);

        // Load pos_embed table: (2304, 768)
        uint32_t pos_table = weight("vision_tower.pos_embed.weight");

        // Gather 4 corner rows: each (num_patches, 768)
        uint32_t g00 = g.gather(pos_table, idx00_node);
        uint32_t g01 = g.gather(pos_table, idx01_node);
        uint32_t g10 = g.gather(pos_table, idx10_node);
        uint32_t g11 = g.gather(pos_table, idx11_node);

        // Reshape gathered to (1, num_patches, D) for 3-D ops
        int32_t sh3_np[] = {1, (int32_t)num_patches, (int32_t)D};
        uint32_t sh3_node = g.constant({3}, sh3_np, DType::INT32);
        g00 = g.reshape(g00, sh3_node);
        g01 = g.reshape(g01, sh3_node);
        g10 = g.reshape(g10, sh3_node);
        g11 = g.reshape(g11, sh3_node);

        // Build weight tensors of shape (1, num_patches, D) by:
        //   1. Create 1-D weight (num_patches,)
        //   2. Reshape to (1, num_patches, 1)
        //   3. Repeat along axis 2 by D times → (1, num_patches, D)
        auto make_weight_3d = [&](const std::vector<float> &w_data) -> uint32_t
        {
            uint32_t w_1d = g.constant({num_patches}, w_data.data(), DType::FLOAT32);
            int32_t sh3_w[] = {1, (int32_t)num_patches, 1};
            uint32_t w_3d = g.reshape(w_1d, g.constant({3}, sh3_w, DType::INT32));
            int32_t rep_D[] = {(int32_t)D};
            int32_t ax2[] = {2};
            return g.repeat(w_3d,
                            g.constant({1}, rep_D, DType::INT32),
                            g.constant({1}, ax2, DType::INT32));
        };

        uint32_t w00_3d = make_weight_3d(w00);
        uint32_t w01_3d = make_weight_3d(w01);
        uint32_t w10_3d = make_weight_3d(w10);
        uint32_t w11_3d = make_weight_3d(w11);

        // Weighted sum: pos = w00*g00 + w01*g01 + w10*g10 + w11*g11
        uint32_t pos = g.mul(g00, w00_3d);
        pos = g.add(pos, g.mul(g01, w01_3d));
        pos = g.add(pos, g.mul(g10, w10_3d));
        pos = g.add(pos, g.mul(g11, w11_3d));

        return pos; // (1, num_patches, 768)
    }

    // Fused-QKV attention with 2-D RoPE, bidirectional (no causal mask).
    uint32_t vision_attention(uint32_t x, int layer_idx,
                              uint32_t cos, uint32_t sin, uint32_t S)
    {
        std::string prefix = "vision_tower.blocks." + std::to_string(layer_idx) + ".attn.";

        uint32_t qkv = linear(x, prefix + "qkv.weight", prefix + "qkv.bias",
                              cfg.vision_hidden_size, 3 * cfg.vision_hidden_size, S);

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

        float scale_val = 1.0f / std::sqrt((float)cfg.vision_head_dim);
        q = g.mul(q, expand_scalar_to_4d(scale_val, 1, cfg.vision_num_heads, S, cfg.vision_head_dim));

        int32_t p_k[] = {0, 1, 3, 2};
        uint32_t k_t = g.contiguous(g.permute(k, g.constant({4}, p_k, DType::INT32)));
        uint32_t scores = g.dot(q, k_t); // (1, num_heads, S, S)

        uint32_t probs = softmax_4d(scores, S, cfg.vision_num_heads);
        uint32_t attn_out = g.dot(probs, v); // (1, num_heads, S, head_dim)

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
        // [FIX] Bug #4: exact GELU instead of tanh approximation
        h = gelu_exact(h, S, cfg.vision_intermediate_size);
        return linear(h, prefix + "linear_fc2.weight", prefix + "linear_fc2.bias",
                      cfg.vision_intermediate_size, cfg.vision_hidden_size, S);
    }

    // [FIX] Bug #3: vision blocks now use LayerNorm (with bias) instead of RMSNorm.
    uint32_t vision_block(uint32_t x, int layer_idx, uint32_t cos, uint32_t sin, uint32_t S)
    {
        std::string prefix = "vision_tower.blocks." + std::to_string(layer_idx) + ".";
        uint32_t residual = x;

        // norm1: LayerNorm(eps=1e-6) with weight AND bias
        uint32_t h = layer_norm(x, prefix + "norm1.weight", prefix + "norm1.bias",
                                S, cfg.vision_hidden_size, cfg.vision_ln_eps);
        h = vision_attention(h, layer_idx, cos, sin, S);
        h = g.add(residual, h);
        residual = h;

        // norm2: LayerNorm(eps=1e-6) with weight AND bias
        h = layer_norm(h, prefix + "norm2.weight", prefix + "norm2.bias",
                       S, cfg.vision_hidden_size, cfg.vision_ln_eps);
        h = vision_mlp(h, layer_idx, S);
        return g.add(residual, h);
    }

    // Full Qwen3VL vision tower: patch_embed + pos_embed → 12 blocks.
    uint32_t vit_encoder(uint32_t patch_input)
    {
        uint32_t S = cfg.num_patches; // 1024

        uint32_t h = patch_embed(patch_input, S); // (1, 1024, 768)

        // [FIX] Bug #2: add bilinear-interpolated position embedding
        uint32_t pos = compute_pos_embed(S); // (1, 1024, 768)
        h = g.add(h, pos);

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

        q = apply_rope(q, cos, sin, cfg.text_num_heads, cfg.text_head_dim, S);
        k = apply_rope(k, cos, sin, cfg.text_num_heads, cfg.text_head_dim, S);

        float scale_val = 1.0f / std::sqrt((float)cfg.text_head_dim);
        q = g.mul(q, expand_scalar_to_4d(scale_val, 1, cfg.text_num_heads, S, cfg.text_head_dim));

        int32_t p_k[] = {0, 1, 3, 2};
        uint32_t k_t = g.contiguous(g.permute(k, g.constant({4}, p_k, DType::INT32)));
        uint32_t scores = g.dot(q, k_t);

        uint32_t probs = softmax_4d(scores, S, cfg.text_num_heads);
        uint32_t attn_out = g.dot(probs, v);

        int32_t p_ctx[] = {0, 2, 1, 3};
        uint32_t ctx_perm = g.contiguous(g.permute(attn_out, g.constant({4}, p_ctx, DType::INT32)));
        int32_t sh3_ctx[] = {1, (int32_t)S, (int32_t)cfg.text_hidden_size};
        uint32_t ctx_flat = g.reshape(ctx_perm, g.constant({3}, sh3_ctx, DType::INT32));

        return linear(ctx_flat, prefix + "o_proj.weight", "",
                      cfg.text_hidden_size, cfg.text_hidden_size, S);
    }

    uint32_t text_mlp(uint32_t x, int layer_idx, uint32_t S)
    {
        std::string prefix = "language_model.layers." + std::to_string(layer_idx) + ".mlp.";
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

    // [FIX] Bug #5: Gather chat-template prefix/suffix token embeddings.
    // Returns a (1, prefix_len + suffix_len, hidden) tensor containing
    // the embed_tokens rows for the prefix and suffix token IDs.
    //
    // Token IDs come from the static CHAT_PREFIX_TOKEN_IDS / CHAT_SUFFIX_TOKEN_IDS tables
    //       (Llama-BPE tokenisation of "<|im_start|>user\n" / "<|im_end|>\n")
    uint32_t build_chat_template_embeds()
    {
        // Load embed_tokens: (vocab_size, hidden)
        uint32_t embed_table = weight("language_model.embed_tokens.weight");

        const int32_t *prefix_ids = CHAT_PREFIX_TOKEN_IDS;
        const int32_t *suffix_ids = CHAT_SUFFIX_TOKEN_IDS;

        // Gather prefix: (prefix_len, hidden)
        uint32_t prefix_idx = g.constant({cfg.text_prefix_len},
                                         prefix_ids,
                                         DType::INT32);
        uint32_t prefix_emb = g.gather(embed_table, prefix_idx); // (prefix_len, hidden)

        // Gather suffix: (suffix_len, hidden)
        uint32_t suffix_idx = g.constant({cfg.text_suffix_len},
                                         suffix_ids,
                                         DType::INT32);
        uint32_t suffix_emb = g.gather(embed_table, suffix_idx); // (suffix_len, hidden)

        // Concat prefix + suffix along axis 0 → (prefix_len + suffix_len, hidden)
        int32_t ax0 = 0;
        uint32_t combined = g.concat({prefix_emb, suffix_emb},
                                     g.constant({1}, &ax0, DType::INT32));

        // Reshape to (1, prefix_len + suffix_len, hidden)
        uint32_t total = cfg.text_prefix_len + cfg.text_suffix_len;
        int32_t sh3[] = {1, (int32_t)total, (int32_t)cfg.text_hidden_size};
        return g.reshape(combined, g.constant({3}, sh3, DType::INT32));
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
        // (1, 1024, 1536) → patch_embed + pos_embed → 12 blocks → (1, 1024, 768)
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
        // [FIX] Bug #4: exact GELU
        uint32_t act = gelu_exact(proj1, cfg.num_merged, cfg.merged_dim);
        uint32_t proj2 = linear(act, "merger.linear_fc2.weight", "merger.linear_fc2.bias",
                                cfg.merged_dim, cfg.text_hidden_size, cfg.num_merged);

        // proj2 is the 256-token image-feature sequence.

        // ---- 3. [FIX] Bug #5: Build full text-encoder input ----------------
        // [8 prefix embeds] + [256 image features] + [6 suffix embeds] = 270 tokens
        uint32_t chat_embeds = build_chat_template_embeds(); // (1, 14, 768)

        // Split into prefix (1, 8, 768) and suffix (1, 6, 768)
        int32_t pre_starts[] = {0, 0, 0};
        int32_t pre_ends[] = {1, (int32_t)cfg.text_prefix_len, (int32_t)cfg.text_hidden_size};
        int32_t suf_starts[] = {0, (int32_t)cfg.text_prefix_len, 0};
        int32_t suf_ends[] = {1, (int32_t)(cfg.text_prefix_len + cfg.text_suffix_len),
                              (int32_t)cfg.text_hidden_size};
        int32_t slice_steps[] = {1, 1, 1};

        uint32_t prefix_emb = g.contiguous(
            g.slice(chat_embeds,
                    g.constant({3}, pre_starts, DType::INT32),
                    g.constant({3}, pre_ends, DType::INT32),
                    g.constant({3}, slice_steps, DType::INT32))); // (1, 8, 768)

        uint32_t suffix_emb = g.contiguous(
            g.slice(chat_embeds,
                    g.constant({3}, suf_starts, DType::INT32),
                    g.constant({3}, suf_ends, DType::INT32),
                    g.constant({3}, slice_steps, DType::INT32))); // (1, 6, 768)

        // Concat: prefix + image_features + suffix → (1, 270, 768)
        int32_t ax_seq = 1;
        uint32_t text_input = g.concat({prefix_emb, proj2, suffix_emb},
                                       g.constant({1}, &ax_seq, DType::INT32));

        // ---- 4. Text encoder (EuroBERT / LlamaModel, bidirectional) -------
        uint32_t h = text_input;       // (1, 270, 768)
        uint32_t S = cfg.text_seq_len; // 270

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

        // ---- 5. Last-token pooling ----------------------------------------
        // [FIX] The last token is at position S-1 = 269 (the last suffix token,
        //       which is token ID 397 = ">\n").  Previously: position 255.
        int32_t starts_last[] = {0, (int32_t)(S - 1), 0};
        int32_t ends_last[] = {1, (int32_t)S, (int32_t)cfg.text_hidden_size};
        int32_t steps_last[] = {1, 1, 1};
        uint32_t last_token = g.contiguous(g.slice(h,
                                                   g.constant({3}, starts_last, DType::INT32),
                                                   g.constant({3}, ends_last, DType::INT32),
                                                   g.constant({3}, steps_last, DType::INT32)));

        int32_t sh2_out[] = {1, (int32_t)cfg.text_hidden_size};
        uint32_t pooled = g.reshape(last_token, g.constant({2}, sh2_out, DType::INT32));

        // ---- 6. L2 normalize ----------------------------------------------
        return l2_normalize(pooled, cfg.text_hidden_size);
    }
};
