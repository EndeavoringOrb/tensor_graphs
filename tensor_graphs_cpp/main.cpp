// File: tensor_graphs_cpp/main.cpp
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>

#if defined(_WIN32)
#include <float.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define stb_image implementation for loading images
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/session.hpp"
#include "core/kernels.hpp"

// Model Definitions
#include "models/gemma-3-270m.hpp"
#include "models/flux-klein-4b.hpp"
#include "models/qwen-3.6-35b-a3b.hpp"

#include "generated/kernels_all.gen.hpp"
#include "generated/build_context.gen.hpp"

// =============================================================================
// Generic Utility Helpers
// =============================================================================

// Common helper to get memory size defaults
std::unordered_map<Backend, uint64_t> get_default_buffer_sizes()
{
    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 24ULL * 1024 * 1024 * 1024}};
#ifdef USE_CUDA
    bufferSizes[Backend::CUDA] = 24ULL * 1024 * 1024 * 1024;
#endif
    return bufferSizes;
}

// Common helper to copy GPU memory to host buffer if CUDA is enabled
const float *sync_output_to_host(const float *device_ptr, size_t num_elements, std::vector<float> &host_buffer)
{
    const float *output_ptr = device_ptr;
#ifdef USE_CUDA
    cudaPointerAttributes attrs;
    if (cudaPointerGetAttributes(&attrs, device_ptr) == cudaSuccess && attrs.type == cudaMemoryTypeDevice)
    {
        host_buffer.resize(num_elements);
        cudaMemcpy(host_buffer.data(), device_ptr, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
        output_ptr = host_buffer.data();
    }
#endif
    return output_ptr;
}

// Common argmax logic
int32_t perform_argmax(const float *logits, uint32_t vocab_size)
{
    float max_val = -1e9f;
    int32_t argmax_idx = 0;
    for (uint32_t i = 0; i < vocab_size; ++i)
    {
        if (logits[i] > max_val)
        {
            max_val = logits[i];
            argmax_idx = i;
        }
    }
    return argmax_idx;
}

// =============================================================================
// Generic Autoregressive Language Model Runner
// =============================================================================
template <typename ModelClass, typename ConfigClass>
void run_autoregressive_llm(
    const std::string &model_name,
    const std::string &model_path,
    const std::string &cache_file,
    const std::vector<uint32_t> &initial_tokens,
    uint32_t vocab_size,
    uint32_t max_seq_len,
    uint32_t num_tokens_to_generate,
    bool only_plan,
    ConfigClass cfg = ConfigClass())
{
    std::vector<uint32_t> tokens = initial_tokens;
    auto bufferSizes = get_default_buffer_sizes();
    MemoryManager mem(bufferSizes);
    Graph g;

    uint32_t inputIdsId = g.input({1, max_seq_len}, DType::INT32, {}, StorageType::PERSISTENT);
    uint64_t sizeBytes = max_seq_len * getDTypeSize(DType::INT32);
    mem.allocate(Backend::CPU, inputIdsId, sizeBytes, StorageType::PERSISTENT);

    std::cout << "Building " << model_name << " Graph..." << std::endl;
    ModelClass model(cfg, max_seq_len, g, mem, model_path);
    uint32_t logits_id = model.build_graph(inputIdsId);

    Session session(g, mem, logits_id, cache_file);

    for (uint32_t i = tokens.size(); i < max_seq_len; ++i)
    {
        std::unordered_map<uint32_t, std::vector<Region>> inputDirty;
        Region inputRegion;
        inputRegion.region = {{0, 1}, {i, i + 1}};
        inputDirty[inputIdsId] = {inputRegion};

        Region outputNeeded;
        outputNeeded.region = {{0, 1}, {i, i + 1}, {0, vocab_size}};
        session.addBucket(inputDirty, {outputNeeded});
    }

    if (only_plan)
    {
        session.plan();
        return;
    }
    session.compile();

    std::vector<int32_t> input_data(max_seq_len, 0);
    std::vector<float> host_output;

    for (uint32_t step = 0; step < num_tokens_to_generate; ++step)
    {
        if (tokens.size() >= max_seq_len)
            break;

        std::fill(input_data.begin(), input_data.end(), 0);
        for (size_t i = 0; i < tokens.size(); ++i)
            input_data[i] = (int32_t)tokens[i];

        session.memManager.write(Backend::CPU, inputIdsId, input_data.data(), input_data.size() * sizeof(int32_t));

        Bucket b;
        if (step != 0)
        {
            uint32_t tokIdx = tokens.size() - 1;
            Region inputRegion = {{{0, 1}, {tokIdx, tokIdx + 1}}};
            Region outputRegion = {{{0, 1}, {tokIdx, tokIdx + 1}, {0, vocab_size}}};
            b.inputDirtyRegions = {{inputIdsId, {inputRegion}}};
            b.outputNeededRegion = {outputRegion};
        }

        auto start = std::chrono::high_resolution_clock::now();
        const float *device_output_ptr = static_cast<const float *>(session.run(b)); // TODO: on steps after the first, pass a bucket
        auto end = std::chrono::high_resolution_clock::now();
        float runtimeMs = std::chrono::duration<float, std::milli>(end - start).count();

        size_t num_output_elements = 1 * max_seq_len * vocab_size;
        const float *output_ptr = sync_output_to_host(device_output_ptr, num_output_elements, host_output);

        uint32_t last_token_pos = (uint32_t)tokens.size() - 1;
        uint64_t offset = (uint64_t)last_token_pos * vocab_size;
        const float *logits_vec = output_ptr + offset;

        int32_t argmax_idx = perform_argmax(logits_vec, vocab_size);
        tokens.push_back((uint32_t)argmax_idx);
        std::cout << "Step " << step + 1 << " | Token: " << argmax_idx << " | End-To-End Latency: " << runtimeMs << "ms\n";
    }
}

// =============================================================================
// Concrete Model Implementations
// =============================================================================

void run_gemma(bool only_plan)
{
    run_autoregressive_llm<Gemma3Model, Gemma3ModelConfig>(
        "Gemma-3-270M",
        "resources/model.safetensors",
        "dirty_region_caches/gemma-3-270m-cpp.bin",
        {2, 9259},
        Gemma3ModelConfig().vocab_size,
        8,
        6,
        only_plan);
}

void run_qwen_35b(bool only_plan)
{
    run_autoregressive_llm<Qwen3_6_35B_A3B_Model, Qwen3_6_35B_A3B_Config>(
        "Qwen-3.6-35B-A3B",
        "models/Qwen/Qwen3.6-35B-A3B",
        "dirty_region_caches/qwen-3.6-35b-a3b-cpp.bin",
        {151644, 8948},
        Qwen3_6_35B_A3B_Config().vocab_size,
        8,
        6,
        only_plan);
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

std::vector<int32_t> load_tokens_from_file(const std::string &filename, size_t txt_seq)
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
    size_t count = 0;

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

// =============================================================================
// FLUX Diffusion Model Execution
// =============================================================================
void run_flux(bool only_plan)
{
    FluxConfig cfg;
    auto bufferSizes = get_default_buffer_sizes();
    MemoryManager mem(bufferSizes);

    uint32_t width = 512, height = 512;
    uint32_t latent_w = width / 16, latent_h = height / 16;
    uint32_t txt_seq = cfg.text_max_seq, img_seq = latent_h * latent_w, total_seq = txt_seq + img_seq;

    auto shared_alloc = std::make_shared<IdAllocator>();

    std::cout << "Building FLUX Text Encoder..." << std::endl;
    Graph g_text;
    g_text.allocator = shared_alloc;
    FluxTextEncoder text_encoder(cfg, g_text, mem, "flux-klein-4b/text_encoder");
    uint32_t in_ids = g_text.input({1, txt_seq}, DType::INT32, {}, StorageType::PERSISTENT);
    Session sess_text(g_text, mem, text_encoder.build_graph(in_ids), "dirty_region_caches/flux-text.bin");
    sess_text.plan();

    std::cout << "Building FLUX Transformer..." << std::endl;
    Graph g_trans;
    g_trans.allocator = shared_alloc;
    FluxTransformer trans(cfg, g_trans, mem, "flux-klein-4b/transformer", latent_h, latent_w);
    uint32_t in_latent = g_trans.input({1, cfg.latent_channels, latent_h, latent_w}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_txt_emb = g_trans.input({1, txt_seq, cfg.text_dim}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_t = g_trans.input({1}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_cos = g_trans.input({1, 1, total_seq, cfg.head_dim}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_sin = g_trans.input({1, 1, total_seq, cfg.head_dim}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    Session sess_trans(g_trans, mem, trans.build_graph(in_latent, in_txt_emb, in_t, in_cos, in_sin), "dirty_region_caches/flux-trans.bin");
    sess_trans.plan();

    std::cout << "Building FLUX VAE..." << std::endl;
    Graph g_vae;
    g_vae.allocator = shared_alloc;
    FluxVAEDecoder vae(cfg, g_vae, mem, "flux-klein-4b/vae", latent_h, latent_w);
    uint32_t in_vae_latent = g_vae.input({1, cfg.vae_channels, latent_h, latent_w}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    Session sess_vae(g_vae, mem, vae.build_graph(in_vae_latent), "dirty_region_caches/flux-vae.bin");
    sess_vae.plan();

    if (only_plan)
    {
        return;
    }

    std::cout << "Executing Text Encoder..." << std::endl;
    std::vector<int32_t> input_ids = load_tokens_from_file("toks.txt", txt_seq);
    sess_text.memManager.write(Backend::CPU, in_ids, input_ids.data(), input_ids.size() * sizeof(int32_t));
    const float *text_emb_ptr = static_cast<const float *>(sess_text.run());

    std::vector<float> text_emb_buf;
    const float *text_emb_host = sync_output_to_host(text_emb_ptr, 1 * txt_seq * cfg.text_dim, text_emb_buf);
    std::vector<float> text_emb(text_emb_host, text_emb_host + 1 * txt_seq * cfg.text_dim);

    std::cout << "Sampling..." << std::endl;
    std::vector<float> rope_cos, rope_sin;
    compute_rope_cpu(txt_seq, latent_h, latent_w, cfg.head_dim, cfg.rope_theta, rope_cos, rope_sin);

    int num_steps = 4;
    std::vector<float> schedule = get_flux_schedule(num_steps, img_seq);
    std::vector<float> z(1 * cfg.latent_channels * latent_h * latent_w);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t j = 0; j < z.size(); ++j)
    {
        z[j] = dist(gen);
    }

    for (int i = 0; i < num_steps; ++i)
    {
        float t_curr = schedule[i], dt = schedule[i + 1] - t_curr;
        std::unordered_map<uint32_t, const void *> trans_inputs = {{in_latent, z.data()}, {in_txt_emb, text_emb.data()}, {in_t, &t_curr}, {in_cos, rope_cos.data()}, {in_sin, rope_sin.data()}};

        const float *v_ptr = static_cast<const float *>(sess_trans.run()); // TODO: write trans_inputs to mem manager

        std::vector<float> v_buf;
        const float *v_host_ptr = sync_output_to_host(v_ptr, z.size(), v_buf);
        v_ptr = v_host_ptr;

        for (size_t j = 0; j < z.size(); ++j)
            z[j] += v_ptr[j] * dt;
        std::cout << "Step " << i + 1 << "/" << num_steps << " complete." << std::endl;
    }

    std::cout << "Executing VAE Decoder..." << std::endl;
    std::unordered_map<uint32_t, const void *> vae_inputs = {{in_vae_latent, z.data()}};
    const float *img_ptr = static_cast<const float *>(sess_vae.run()); // TODO: write vae_inputs to memManager

    std::vector<float> img_buf;
    img_ptr = sync_output_to_host(img_ptr, 3 * height * width, img_buf);

    std::vector<uint8_t> image_data(height * width * 3);
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            // Extract from planar CHW format
            float r = img_ptr[0 * height * width + y * width + x];
            float g = img_ptr[1 * height * width + y * width + x];
            float b = img_ptr[2 * height * width + y * width + x];

            // Normalize [-1, 1] to [0, 1]
            r = std::max(0.0f, std::min(1.0f, (r + 1.0f) * 0.5f));
            g = std::max(0.0f, std::min(1.0f, (g + 1.0f) * 0.5f));
            b = std::max(0.0f, std::min(1.0f, (b + 1.0f) * 0.5f));

            // Write to interleaved HWC format
            uint32_t idx = (y * width + x) * 3;
            image_data[idx + 0] = static_cast<uint8_t>(r * 255.0f);
            image_data[idx + 1] = static_cast<uint8_t>(g * 255.0f);
            image_data[idx + 2] = static_cast<uint8_t>(b * 255.0f);
        }
    }
    stbi_write_png("flux_output.png", width, height, 3, image_data.data(), width * 3);
    std::cout << "Saved flux_output.png successfully!" << std::endl;
}

// =============================================================================
// Main Entrypoint
// =============================================================================
int main(int argc, char *argv[])
{
    auto &caps = HardwareCaps::get();
    std::cout << "Hardware:\n"
              << "  has_unified_memory: " << caps.has_unified_memory << "\n"
              << "  has_cuda: " << caps.has_cuda << "\n"
              << "  has_neon: " << caps.has_neon << "\n"
              << "  hw_tag: " << caps.hw_tag << "\n"
              << "  num_threads: " << caps.num_threads << "\n";
#if defined(_WIN32)
    _controlfp_s(nullptr, 0, 0);
    _controlfp_s(nullptr, _EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW, _MCW_EM);
#endif

    std::string model = "flux-klein-4b";
    bool only_plan = false;

    if (argc > 1)
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "--only-plan")
            {
                only_plan = true;
            }
            else
            {
                model = arg;
            }
        }
    }

    if (model == "gemma-3-270m")
        run_gemma(only_plan);
    else if (model == "flux-klein-4b")
        run_flux(only_plan);
    else if (model == "qwen-3.6-35b-a3b")
        run_qwen_35b(only_plan);
    else
        std::cout << "Model not implemented yet: " << model << std::endl;

    return 0;
}