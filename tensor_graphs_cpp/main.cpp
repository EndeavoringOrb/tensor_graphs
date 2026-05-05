// File: tensor_graphs_cpp/main.cpp
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

#if defined(_WIN32)
#include <float.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/session.hpp"
#include "core/kernels.hpp"

// Model Definitions
#include "models/gemma-3-270m.hpp"
#include "models/flux-klein-4b.hpp"

#include "generated/kernels_all.gen.hpp"
#include "generated/build_context.gen.hpp"

void run_gemma()
{
    std::vector<uint32_t> tokens = {2, 9259};
    uint32_t maxSeqLen = 8;
    uint32_t numTokensToGenerate = 6;
    std::string modelPath = "resources/model.safetensors";

    Gemma3ModelConfig cfg;
#if USE_CUDA
    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 16ULL * 1024 * 1024 * 1024}, {Backend::CUDA, 6ULL * 1024 * 1024 * 1024}};
#else
    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 16ULL * 1024 * 1024 * 1024}};
#endif
    MemoryManager mem(bufferSizes);
    Graph g;

    uint32_t inputIdsId = g.input({1, maxSeqLen}, DType::INT32, {}, StorageType::PERSISTENT);
    uint64_t sizeBytes = maxSeqLen * getDTypeSize(DType::INT32);
    mem.allocate(Backend::CPU, inputIdsId, sizeBytes, StorageType::PERSISTENT);

    std::cout << "Building Gemma-3 Graph..." << std::endl;
    Gemma3Model gemma(cfg, maxSeqLen, g, mem, modelPath);
    uint32_t logits_id = gemma.build_graph(inputIdsId);

    Session session(g, mem, logits_id, "dirty_region_caches/gemma-3-270m-cpp.jsonl");

    for (uint32_t i = tokens.size(); i < maxSeqLen; ++i)
    {
        std::unordered_map<uint32_t, std::vector<Region>> inputDirty;
        Region inputRegion;
        inputRegion.region = {{0, 1}, {i, i + 1}};
        inputDirty[inputIdsId] = {inputRegion};

        Region outputNeeded;
        outputNeeded.region = {{0, 1}, {i, i + 1}, {0, cfg.vocab_size}};
        session.addManualBucket(inputDirty, {outputNeeded});
    }

    std::vector<int32_t> input_data(maxSeqLen, 0);

    for (uint32_t step = 0; step < numTokensToGenerate; ++step)
    {
        if (tokens.size() >= maxSeqLen)
            break;

        std::fill(input_data.begin(), input_data.end(), 0);
        for (size_t i = 0; i < tokens.size(); ++i)
            input_data[i] = (int32_t)tokens[i];

        std::unordered_map<uint32_t, const void *> inputs;
        inputs[inputIdsId] = input_data.data();

        auto start = std::chrono::high_resolution_clock::now();
        const float *device_output_ptr = static_cast<const float *>(session.run(inputs));
        auto end = std::chrono::high_resolution_clock::now();
        float runtimeMs = std::chrono::duration<float, std::milli>(end - start).count();

        const float *output_ptr = device_output_ptr;
#ifdef USE_CUDA
        std::vector<float> host_output;
        cudaPointerAttributes attrs;
        if (cudaPointerGetAttributes(&attrs, device_output_ptr) == cudaSuccess && attrs.type == cudaMemoryTypeDevice)
        {
            host_output.resize(1 * maxSeqLen * cfg.vocab_size);
            cudaMemcpy(host_output.data(), device_output_ptr, host_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
            output_ptr = host_output.data();
        }
#endif
        uint32_t last_token_pos = (uint32_t)tokens.size() - 1;
        uint64_t offset = (uint64_t)last_token_pos * cfg.vocab_size;
        const float *logits_vec = output_ptr + offset;

        float max_val = -1e9f;
        int32_t argmax_idx = 0;
        for (uint32_t i = 0; i < cfg.vocab_size; ++i)
        {
            if (logits_vec[i] > max_val)
            {
                max_val = logits_vec[i];
                argmax_idx = i;
            }
        }
        tokens.push_back((uint32_t)argmax_idx);
        std::cout << "Step " << step + 1 << " | Token: " << argmax_idx << " | Latency: " << runtimeMs << "ms\n";
    }
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

void run_flux()
{
    FluxConfig cfg;
#if USE_CUDA
    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 16ULL * 1024 * 1024 * 1024}, {Backend::CUDA, 12ULL * 1024 * 1024 * 1024}};
#else
    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 24ULL * 1024 * 1024 * 1024}};
#endif
    MemoryManager mem(bufferSizes);

    uint32_t width = 256, height = 256;
    uint32_t latent_w = width / 16, latent_h = height / 16;
    uint32_t txt_seq = cfg.text_max_seq, img_seq = latent_h * latent_w, total_seq = txt_seq + img_seq;

    std::cout << "Building FLUX Text Encoder..." << std::endl;
    Graph g_text;
    FluxTextEncoder text_encoder(cfg, g_text, mem, "flux-klein-4b/text_encoder");
    uint32_t in_ids = g_text.input({1, txt_seq}, DType::INT32, {}, StorageType::PERSISTENT);
    Session sess_text(g_text, mem, text_encoder.build_graph(in_ids), "dirty_region_caches/flux-text.jsonl");

    std::cout << "Building FLUX Transformer..." << std::endl;
    Graph g_trans;
    FluxTransformer trans(cfg, g_trans, mem, "flux-klein-4b/transformer", latent_h, latent_w);
    uint32_t in_latent = g_trans.input({1, cfg.latent_channels, latent_h, latent_w}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_txt_emb = g_trans.input({1, txt_seq, cfg.hidden_size}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_t = g_trans.input({1}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_cos = g_trans.input({1, 1, total_seq, cfg.head_dim}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_sin = g_trans.input({1, 1, total_seq, cfg.head_dim}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    Session sess_trans(g_trans, mem, trans.build_graph(in_latent, in_txt_emb, in_t, in_cos, in_sin), "dirty_region_caches/flux-trans.jsonl");

    std::cout << "Building FLUX VAE..." << std::endl;
    Graph g_vae;
    FluxVAEDecoder vae(cfg, g_vae, mem, "flux-klein-4b/vae", latent_h, latent_w);
    uint32_t in_vae_latent = g_vae.input({1, cfg.vae_channels, latent_h, latent_w}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    Session sess_vae(g_vae, mem, vae.build_graph(in_vae_latent), "dirty_region_caches/flux-vae.jsonl");

    std::cout << "Executing Text Encoder..." << std::endl;
    std::vector<int32_t> input_ids(txt_seq, 151643);
    input_ids[0] = 2; // Fake token padding
    std::unordered_map<uint32_t, const void *> text_inputs = {{in_ids, input_ids.data()}};
    const float *text_emb_ptr = static_cast<const float *>(sess_text.run(text_inputs));
#ifdef USE_CUDA
    std::vector<float> text_emb(1 * txt_seq * cfg.text_dim);
    cudaMemcpy(text_emb.data(), text_emb_ptr, text_emb.size() * sizeof(float), cudaMemcpyDeviceToHost);
#else
    std::vector<float> text_emb(text_emb_ptr, text_emb_ptr + 1 * txt_seq * cfg.text_dim);
#endif

    std::cout << "Sampling..." << std::endl;
    std::vector<float> rope_cos, rope_sin;
    compute_rope_cpu(txt_seq, latent_h, latent_w, cfg.head_dim, cfg.rope_theta, rope_cos, rope_sin);

    int num_steps = 4;
    std::vector<float> schedule = get_flux_schedule(num_steps, img_seq);
    std::vector<float> z(1 * cfg.latent_channels * latent_h * latent_w, 0.1f); // Fake noise for testing

    for (int i = 0; i < num_steps; ++i)
    {
        float t_curr = schedule[i], dt = schedule[i + 1] - t_curr;
        std::unordered_map<uint32_t, const void *> trans_inputs = {{in_latent, z.data()}, {in_txt_emb, text_emb.data()}, {in_t, &t_curr}, {in_cos, rope_cos.data()}, {in_sin, rope_sin.data()}};

        const float *v_ptr = static_cast<const float *>(sess_trans.run(trans_inputs));
#ifdef USE_CUDA
        std::vector<float> v_host(z.size());
        cudaMemcpy(v_host.data(), v_ptr, v_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
        v_ptr = v_host.data();
#endif
        for (size_t j = 0; j < z.size(); ++j)
            z[j] += v_ptr[j] * dt;
        std::cout << "Step " << i + 1 << "/" << num_steps << " complete." << std::endl;
    }

    std::cout << "Executing VAE Decoder..." << std::endl;
    std::unordered_map<uint32_t, const void *> vae_inputs = {{in_vae_latent, z.data()}};
    const float *img_ptr = static_cast<const float *>(sess_vae.run(vae_inputs));
#ifdef USE_CUDA
    std::vector<float> img_host(3 * height * width);
    cudaMemcpy(img_host.data(), img_ptr, img_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    img_ptr = img_host.data();
#endif

    std::vector<uint8_t> image_data(height * width * 3);
    for (size_t i = 0; i < height * width * 3; ++i)
    {
        float val = std::max(0.0f, std::min(1.0f, (img_ptr[i] + 1.0f) * 0.5f));
        image_data[i] = static_cast<uint8_t>(val * 255.0f);
    }
    stbi_write_png("flux_output.png", width, height, 3, image_data.data(), width * 3);
    std::cout << "Saved flux_output.png successfully!" << std::endl;
}

int main(int argc, char *argv[])
{
#if defined(_WIN32)
    _controlfp_s(nullptr, 0, 0);
    _controlfp_s(nullptr, _EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW, _MCW_EM);
#endif

    std::string model = "gemma-3-270m";
    if (argc > 1)
        model = argv[1];

    if (model == "gemma-3-270m")
        run_gemma();
    else if (model == "flux-klein-4b")
        run_flux();
    else
        std::cout << "Model not implemented yet: " << model << std::endl;

    return 0;
}