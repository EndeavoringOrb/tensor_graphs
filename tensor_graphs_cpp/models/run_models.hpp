// tensor_graphs_cpp/models/run_models.hpp
#pragma once
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "models/gemma-3-270m.hpp"
#include "models/flux-klein-4b.hpp"
#include "models/qwen-3.6-35b-a3b.hpp"

struct ModelGraphRoots
{
    std::vector<uint32_t> roots;
    std::vector<uint32_t> inputs;
};

inline ModelGraphRoots build_gemma_graph(Graph &g, MemoryManager &mem)
{
    Gemma3ModelConfig cfg;
    uint32_t maxSeqLen = 8;
    uint32_t inputIdsId = g.input({1, maxSeqLen}, DType::INT32, {}, StorageType::PERSISTENT);
    Gemma3Model gemma(cfg, maxSeqLen, g, mem, "resources/model.safetensors");
    return {{gemma.build_graph(inputIdsId)}, {inputIdsId}};
}

inline ModelGraphRoots build_qwen_graph(Graph &g, MemoryManager &mem)
{
    Qwen3_6_35B_A3B_Config cfg;
    uint32_t maxSeqLen = 8;
    uint32_t inputIdsId = g.input({1, maxSeqLen}, DType::INT32, {}, StorageType::PERSISTENT);
    Qwen3_6_35B_A3B_Model qwen(cfg, maxSeqLen, g, mem, "models/Qwen/Qwen3.6-35B-A3B");
    return {{qwen.build_graph(inputIdsId)}, {inputIdsId}};
}

inline ModelGraphRoots build_flux_graph(Graph &g, MemoryManager &mem)
{
    FluxConfig cfg;
    uint32_t width = 512, height = 512;
    uint32_t latent_w = width / 16, latent_h = height / 16;
    uint32_t txt_seq = cfg.text_max_seq, img_seq = latent_h * latent_w, total_seq = txt_seq + img_seq;

    if (!g.allocator)
        g.allocator = std::make_shared<IdAllocator>();

    FluxTextEncoder text_encoder(cfg, g, mem, "flux-klein-4b/text_encoder");
    uint32_t in_ids = g.input({1, txt_seq}, DType::INT32, {}, StorageType::PERSISTENT);
    uint32_t text_root = text_encoder.build_graph(in_ids);

    FluxTransformer trans(cfg, g, mem, "flux-klein-4b/transformer", latent_h, latent_w);
    uint32_t in_latent = g.input({1, cfg.latent_channels, latent_h, latent_w}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_txt_emb = g.input({1, txt_seq, cfg.text_dim}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_t = g.input({1}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_cos = g.input({1, 1, total_seq, cfg.head_dim}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_sin = g.input({1, 1, total_seq, cfg.head_dim}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t trans_root = trans.build_graph(in_latent, in_txt_emb, in_t, in_cos, in_sin);

    FluxVAEDecoder vae(cfg, g, mem, "flux-klein-4b/vae", latent_h, latent_w);
    uint32_t in_vae_latent = g.input({1, cfg.vae_channels, latent_h, latent_w}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t vae_root = vae.build_graph(in_vae_latent);

    return {{text_root, trans_root, vae_root}, {in_ids, in_latent, in_txt_emb, in_t, in_cos, in_sin, in_vae_latent}};
}

std::vector<float> get_flux_schedule(int num_steps, int image_seq_len)
{
    std::vector<float> schedule(num_steps + 1, 0.0f);
    double a1 = 8.73809524e-05, b1 = 1.89833333;
    double a2 = 0.00016927, b2 = 0.45666666;
    double mu = (image_seq_len > 4300) ? (a2 * image_seq_len + b2) : (((a2 * image_seq_len + b2) - (a1 * image_seq_len + b1)) / 190.0 * num_steps + (a2 * image_seq_len + b2) - 200.0 * ((a2 * image_seq_len + b2) - (a1 * image_seq_len + b1)) / 190.0);

    for (int i = 0; i <= num_steps; ++i)
    {
        double t = 1.0 - (double)i / num_steps;
        if (t <= 0.0)
            schedule[i] = 0.0f;
        else if (t >= 1.0)
            schedule[i] = 1.0f;
        else
            schedule[i] = (float)(exp(mu) / (exp(mu) + (1.0 / t - 1.0)));
    }
    return schedule;
}

void compute_rope_cpu(int txt_seq, int img_h, int img_w, int head_dim, float theta, std::vector<float> &cos_out, std::vector<float> &sin_out)
{
    int img_seq = img_h * img_w, total_seq = txt_seq + img_seq, axis_dim = head_dim / 4;
    cos_out.assign(total_seq * head_dim, 1.0f);
    sin_out.assign(total_seq * head_dim, 0.0f);
    std::vector<float> freqs(axis_dim / 2);
    for (int i = 0; i < axis_dim / 2; ++i)
        freqs[i] = 1.0f / pow(theta, (2.0f * i) / axis_dim);

    for (int pos = 0; pos < txt_seq; ++pos)
    {
        for (int i = 0; i < axis_dim / 2; ++i)
        {
            float arg = pos * freqs[i];
            int ax3 = axis_dim * 3;
            cos_out[pos * head_dim + ax3 + 2 * i] = cos_out[pos * head_dim + ax3 + 2 * i + 1] = cos(arg);
            sin_out[pos * head_dim + ax3 + 2 * i] = sin_out[pos * head_dim + ax3 + 2 * i + 1] = sin(arg);
        }
    }

    for (int y = 0; y < img_h; ++y)
    {
        for (int x = 0; x < img_w; ++x)
        {
            int pos = txt_seq + y * img_w + x;
            for (int i = 0; i < axis_dim / 2; ++i)
            {
                float c_h = cos(y * freqs[i]), s_h = sin(y * freqs[i]);
                float c_w = cos(x * freqs[i]), s_w = sin(x * freqs[i]);
                int ax1 = axis_dim * 1, ax2 = axis_dim * 2;
                cos_out[pos * head_dim + ax1 + 2 * i] = cos_out[pos * head_dim + ax1 + 2 * i + 1] = c_h;
                sin_out[pos * head_dim + ax1 + 2 * i] = sin_out[pos * head_dim + ax1 + 2 * i + 1] = s_h;
                cos_out[pos * head_dim + ax2 + 2 * i] = cos_out[pos * head_dim + ax2 + 2 * i + 1] = c_w;
                sin_out[pos * head_dim + ax2 + 2 * i] = sin_out[pos * head_dim + ax2 + 2 * i + 1] = s_w;
            }
        }
    }
}

std::vector<int32_t> load_tokens_from_file(const std::string &filename, uint64_t txt_seq)
{
    // 1. Initialize vector with the padding value 151643
    std::vector<int32_t> input_ids(txt_seq, 151643);

    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return input_ids;
    }

    std::string part;
    uint64_t count = 0;

    // 2. Read from file using ',' as the delimiter
    // Note: This also stops if we reach the txt_seq limit
    while (std::getline(file, part, ',') && count < txt_seq)
    {

        // Trim potential whitespace/newlines and convert to integer
        if (!part.empty())
        {
            input_ids[count++] = static_cast<int32_t>(std::stoi(part));
        }
    }

    return input_ids;
}