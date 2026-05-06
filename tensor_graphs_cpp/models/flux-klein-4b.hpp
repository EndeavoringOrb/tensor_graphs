// File: tensor_graphs_cpp/models/flux-klein-4b.hpp
#pragma once

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <optional>

struct FluxConfig
{
    // Transformer
    uint32_t hidden_size = 3072;
    uint32_t num_heads = 24;
    uint32_t head_dim = 128;
    uint32_t mlp_hidden = 9216;
    uint32_t num_double_layers = 5;
    uint32_t num_single_layers = 20;
    float rope_theta = 2000.0f;
    uint32_t axis_dim = 32;

    // VAE
    uint32_t vae_channels = 128;
    uint32_t vae_base_ch = 128;
    uint32_t vae_z_channels = 32;
    uint32_t latent_channels = 128;
    uint32_t patch_size = 2;

    // Text Encoder
    uint32_t text_dim = 7680;
    uint32_t text_vocab_size = 151936;
    uint32_t text_hidden_size = 2560;
    uint32_t text_mlp_hidden_size = 9728;
    uint32_t text_num_layers = 36;
    uint32_t text_num_heads = 32;
    uint32_t text_num_kv_heads = 8;
    uint32_t text_head_dim = 128;
    uint32_t text_max_seq = 512;
    float text_rope_theta = 1000000.0f;
};

// Base class containing shared atomic utilities for FLUX.2 components
class FluxGraphBase
{
protected:
    Graph &g;
    std::string w_path;

    FluxGraphBase(Graph &graph, const std::string &path) : g(graph), w_path(path) {}

    uint32_t weight(const std::string &name)
    {
        return g.cast(g.weight(w_path, name), DType::FLOAT32);
    }

    uint32_t repeat_ax(uint32_t id, uint32_t repeats, uint32_t axis)
    {
        if (repeats <= 1)
            return id;
        int32_t r = repeats, a = axis;
        return g.repeat(id, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }

    uint32_t expand_scalar_to_1d(float val, uint32_t d0)
    {
        uint32_t node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh1[] = {1};
        return repeat_ax(g.reshape(node, g.constant({1}, sh1, DType::INT32)), d0, 0);
    }

    uint32_t expand_scalar_to_3d(float val, uint32_t d0, uint32_t d1, uint32_t d2)
    {
        uint32_t node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh3[] = {1, 1, 1};
        uint32_t out = g.reshape(node, g.constant({3}, sh3, DType::INT32));
        if (d0 > 1)
            out = repeat_ax(out, d0, 0);
        if (d1 > 1)
            out = repeat_ax(out, d1, 1);
        if (d2 > 1)
            out = repeat_ax(out, d2, 2);
        return out;
    }

    uint32_t expand_scalar_to_4d(float val, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3)
    {
        uint32_t node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh4[] = {1, 1, 1, 1};
        uint32_t out = g.reshape(node, g.constant({4}, sh4, DType::INT32));
        if (d0 > 1)
            out = repeat_ax(out, d0, 0);
        if (d1 > 1)
            out = repeat_ax(out, d1, 1);
        if (d2 > 1)
            out = repeat_ax(out, d2, 2);
        if (d3 > 1)
            out = repeat_ax(out, d3, 3);
        return out;
    }

    uint32_t silu_atomic(uint32_t x, uint32_t N, uint32_t L, uint32_t D)
    {
        uint32_t neg_x = g.neg(x);
        uint32_t exp_neg = g.pow(expand_scalar_to_3d(2.7182818f, N, L, D), neg_x);
        uint32_t den = g.add(expand_scalar_to_3d(1.0f, N, L, D), exp_neg);
        uint32_t sig = g.div(expand_scalar_to_3d(1.0f, N, L, D), den);
        return g.mul(x, sig);
    }
};

#include "flux-klein-4b-text_encoder.hpp"
#include "flux-klein-4b-transformer.hpp"
#include "flux-klein-4b-vae.hpp"