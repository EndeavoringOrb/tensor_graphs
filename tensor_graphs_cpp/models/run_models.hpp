#pragma once
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "models/deepseek-v4-flash.hpp"
#include "models/gemma-3-270m.hpp"
#include "models/krea-2-turbo.hpp"
#include "models/qwen-3.6-35b-a3b.hpp"
#include "models/qwen-image-vae.hpp"
#include "models/qwen3-vl.hpp"

struct ModelGraphRoots
{
    std::vector<LogicalId> roots;
    std::vector<LogicalId> inputs;
};

inline ModelGraphRoots build_gemma_graph(Graph &g, MemoryManager &mem, const std::string &model_path,
                                         uint32_t max_seq_len)
{
    Gemma3ModelConfig cfg;
    LogicalId inputIdsId = g.input({1, max_seq_len}, DType::INT32);
    Gemma3Model gemma(cfg, max_seq_len, g, mem, model_path);
    return {{gemma.build_graph(inputIdsId)}, {inputIdsId}};
}

inline ModelGraphRoots build_qwen_graph(Graph &g, MemoryManager &mem, const std::string &model_path,
                                        uint32_t max_seq_len)
{
    Qwen3_6_35B_A3B_Config cfg;
    LogicalId inputIdsId = g.input({1, max_seq_len}, DType::INT32);
    Qwen3_6_35B_A3B_Model qwen(cfg, max_seq_len, g, mem, model_path);
    return {{qwen.build_graph(inputIdsId)}, {inputIdsId}};
}

inline ModelGraphRoots build_krea2_graph(Graph &g, MemoryManager &mem, const std::string &model_path,
                                         uint32_t height = 1024, uint32_t width = 1024, uint32_t text_seq_len = 128)
{
    Krea2TurboConfig cfg(height, width, text_seq_len);
    LogicalId latentId = g.input({1, cfg.latent_channels, cfg.latent_h, cfg.latent_w}, DType::FLOAT32);
    LogicalId timestepId = g.input({1}, DType::FLOAT32);
    LogicalId textId = g.input({1, cfg.text_seq_len, cfg.text_num_layers, cfg.text_dim}, DType::FLOAT32);
    Krea2TurboModel model(cfg, g, mem, model_path);
    LogicalId velocityOut = model.build_graph(latentId, timestepId, textId);
    return {{velocityOut}, {latentId, timestepId, textId}};
}

inline ModelGraphRoots build_krea2_vae_graph(Graph &g, MemoryManager &mem, const std::string &model_path,
                                             uint32_t height = 1024, uint32_t width = 1024)
{
    Krea2TurboVAEConfig cfg(height, width);
    LogicalId latentId = g.input({1, cfg.latent_channels, cfg.latent_h, cfg.latent_w}, DType::FLOAT32);
    Krea2TurboVAEModel model(cfg, g, mem, model_path);
    LogicalId imageOut = model.build_graph(latentId);
    return {{imageOut}, {latentId}};
}

inline ModelGraphRoots build_qwen3_vl_graph(Graph &g, MemoryManager &mem, const std::string &model_path,
                                            uint32_t seq_len = 128)
{
    Qwen3VLConfig cfg;
    LogicalId inputIdsId = g.input({1, seq_len}, DType::INT32);
    Qwen3VLModel model(cfg, seq_len, g, mem, model_path);
    LogicalId textEmbOut = model.build_graph(inputIdsId);
    return {{textEmbOut}, {inputIdsId}};
}

inline ModelGraphRoots build_krea2_pipeline_graph(Graph &g, MemoryManager &mem, const std::string &dit_path,
                                                  const std::string &text_encoder_path, const std::string &vae_path,
                                                  uint32_t height = 1024, uint32_t width = 1024,
                                                  uint32_t text_seq_len = 128, uint32_t steps = 8, float mu = 1.15f)
{
    // 1. Text Encoder (Qwen3-VL)
    Qwen3VLConfig te_cfg;
    LogicalId inputIdsId = g.input({1, text_seq_len}, DType::INT32);
    Qwen3VLModel te_model(te_cfg, text_seq_len, g, mem, text_encoder_path);
    LogicalId textEmbOut = te_model.build_graph(inputIdsId);

    // 2. DiT (Krea 2 Turbo) unrolled for 'steps' steps
    Krea2TurboConfig dit_cfg(height, width, text_seq_len);
    LogicalId latentId = g.input({1, dit_cfg.latent_channels, dit_cfg.latent_h, dit_cfg.latent_w}, DType::FLOAT32);
    Krea2TurboModel dit_model(dit_cfg, g, mem, dit_path);
    LogicalId finalLatent = dit_model.build_unrolled_dit(latentId, textEmbOut, steps, mu);

    // 3. VAE Decoder (Qwen Image VAE)
    Krea2TurboVAEConfig vae_cfg(height, width);
    Krea2TurboVAEModel vae_model(vae_cfg, g, mem, vae_path);
    LogicalId imageOut = vae_model.build_graph(finalLatent);

    return {{imageOut}, {inputIdsId, latentId}};
}