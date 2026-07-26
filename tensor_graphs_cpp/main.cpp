// tensor_graphs_cpp/main.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include <float.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "core/argparse.hpp"
#include "core/debug.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/memory.hpp"
#include "core/repo.hpp"
#include "core/session.hpp"
#include "core/types.hpp"
#include "generated/build_context.gen.hpp"
#include "generated/kernels_all.gen.hpp"
#include "models/run_models.hpp"
#include "stb_image.h"

namespace fs = std::filesystem;

std::unordered_map<MemSpace, uint64_t> getDefaultBufferSizes()
{
    std::unordered_map<MemSpace, uint64_t> bufferSizes = {{MemSpace{1, HandleType::CPP}, 24ULL * 1024 * 1024 * 1024}};
#ifdef USE_CUDA
    bufferSizes[MemSpace{2, HandleType::CUDA}] = 24ULL * 1024 * 1024 * 1024;
#endif
    if (HardwareCaps::get().has_opencl)
    {
        bufferSizes[MemSpace{1, HandleType::OPENCL}] = 1ULL * 1024 * 1024 * 1024;
    }
    return bufferSizes;
}

const float *sync_output_to_host(const float *device_ptr, uint64_t num_elements, std::vector<float> &host_buffer)
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

template <typename ConfigClass>
void run_autoregressive_llm(const std::string &model_name, const std::string &cache_file,
                            const std::vector<uint32_t> &initial_tokens, uint32_t vocab_size, uint32_t max_seq_len,
                            uint32_t num_tokens_to_generate, bool only_plan, bool disable_caching,
                            ModelGraphRoots (*builder)(Graph &, MemoryManager &), bool refOnly = false,
                            bool doSaturate = true, const Debug::Callback &debugCb = nullptr,
                            Graph **activeGraphOut = nullptr)
{
    KernelRegistry::get().setReferenceOnly(refOnly);
    std::vector<uint32_t> tokens = initial_tokens;
    auto bufferSizes = getDefaultBufferSizes();
    MemoryManager mem(bufferSizes);
    Graph g;
    if (activeGraphOut)
    {
        *activeGraphOut = &g;
    }

    std::cout << "Building " << model_name << " Graph..." << std::endl;
    auto roots = builder(g, mem);
    LogicalId logits_id = roots.roots[0];
    LogicalId inputIdsId = roots.inputs[0];

    std::string gHash = computeGraphHash(g, roots.roots);
    Repo repo("benchmarks/repo_" + model_name, gHash, true);

    Session session(g, mem, logits_id, cache_file, 0, &repo, disable_caching);

    for (uint32_t i = tokens.size(); i < max_seq_len; ++i)
    {
        std::unordered_map<LogicalId, std::vector<Region>> inputDirty;
        Region inputRegion;
        inputRegion.region = {{0, 1}, {i, i + 1}};
        inputDirty[inputIdsId] = {inputRegion};

        Region outputNeeded;
        outputNeeded.region = {{0, 1}, {i, i + 1}, {0, vocab_size}};
        session.addBucket(inputDirty, {outputNeeded});
    }

    if (only_plan)
    {
        session.plan(doSaturate);
        return;
    }
    session.compile(doSaturate);

    std::vector<int32_t> input_data(max_seq_len, 0);
    std::vector<float> host_output;

    for (uint32_t step = 0; step < num_tokens_to_generate; ++step)
    {
        if (tokens.size() >= max_seq_len)
            break;

        std::fill(input_data.begin(), input_data.end(), 0);
        for (uint64_t i = 0; i < tokens.size(); ++i)
            input_data[i] = (int32_t)tokens[i];

        session.writeInput(inputIdsId, input_data.data(), input_data.size() * sizeof(int32_t));

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
        const float *device_output_ptr = static_cast<const float *>(session.run(b, debugCb, doSaturate));
        auto end = std::chrono::high_resolution_clock::now();
        float runtimeMs = std::chrono::duration<float, std::milli>(end - start).count();

        uint64_t num_output_elements = 1 * max_seq_len * vocab_size;
        const float *output_ptr = sync_output_to_host(device_output_ptr, num_output_elements, host_output);

        uint32_t last_token_pos = (uint32_t)tokens.size() - 1;
        uint64_t offset = (uint64_t)last_token_pos * vocab_size;
        const float *logits_vec = output_ptr + offset;

        int32_t argmax_idx = perform_argmax(logits_vec, vocab_size);
        tokens.push_back((uint32_t)argmax_idx);
        std::cout << "Step " << step + 1 << " | Token: " << argmax_idx << " | End-To-End Latency: " << runtimeMs
                  << "ms\n";
    }
}

void run_gemma(bool only_plan, bool disable_caching, bool refOnly = false, bool doSaturate = true,
               const Debug::Callback &debugCb = nullptr, Graph **activeGraphOut = nullptr)
{
    run_autoregressive_llm<Gemma3ModelConfig>("gemma-3-270m", "dirty_region_caches/gemma-3-270m-cpp.bin", {2, 9259},
                                              Gemma3ModelConfig().vocab_size, 8, 6, only_plan, disable_caching,
                                              build_gemma_graph, refOnly, doSaturate, debugCb, activeGraphOut);
}

void run_qwen_35b(bool only_plan, bool disable_caching, bool refOnly = false, bool doSaturate = true,
                  const Debug::Callback &debugCb = nullptr, Graph **activeGraphOut = nullptr)
{
    run_autoregressive_llm<Qwen3_6_35B_A3B_Config>("qwen-3.6-35b-a3b", "dirty_region_caches/qwen-3.6-35b-a3b-cpp.bin",
                                                   {24227}, Qwen3_6_35B_A3B_Config().vocab_size, 8, 7, only_plan,
                                                   disable_caching, build_qwen_graph, refOnly, doSaturate, debugCb,
                                                   activeGraphOut);
}

// void run_flux(bool only_plan, bool disable_caching, bool refOnly = false,
// bool doSaturate = true, const Debug::Callback &debugCb = nullptr, Graph
// **activeGraphOut = nullptr)
// {
//     KernelRegistry::get().setReferenceOnly(refOnly);
//     FluxConfig cfg;
//     auto bufferSizes = getDefaultBufferSizes();
//     MemoryManager mem(bufferSizes);
//     Graph g;
//     if (activeGraphOut)
//     {
//         *activeGraphOut = &g;
//     }

//     std::cout << "Building FLUX Graphs..." << std::endl;
//     auto roots = build_flux_graph(g, mem);

//     std::string gHash = computeGraphHash(g, roots.roots);
//     Repo repo("benchmarks/repo_flux-klein-4b", gHash, true);

//     uint32_t in_ids = roots.inputs[0];
//     uint32_t in_latent = roots.inputs[1];
//     uint32_t in_txt_emb = roots.inputs[2];
//     uint32_t in_t = roots.inputs[3];
//     uint32_t in_cos = roots.inputs[4];
//     uint32_t in_sin = roots.inputs[5];
//     uint32_t in_vae_latent = roots.inputs[6];

//     Session sess_text(g, mem, roots.roots[0],
//     "dirty_region_caches/flux-text.bin", 0, &repo, disable_caching);
//     sess_text.plan(doSaturate);

//     Session sess_trans(g, mem, roots.roots[1],
//     "dirty_region_caches/flux-trans.bin", 0, &repo, disable_caching);
//     sess_trans.plan(doSaturate);

//     Session sess_vae(g, mem, roots.roots[2],
//     "dirty_region_caches/flux-vae.bin", 0, &repo, disable_caching);
//     sess_vae.plan(doSaturate);

//     if (only_plan)
//     {
//         return;
//     }

//     uint32_t width = 512, height = 512;
//     uint32_t latent_w = width / 16, latent_h = height / 16;
//     uint32_t txt_seq = cfg.text_max_seq, img_seq = latent_h * latent_w,
//     total_seq = txt_seq + img_seq;

//     std::cout << "Executing Text Encoder..." << std::endl;
//     std::vector<int32_t> input_ids = load_tokens_from_file("toks.txt",
//     txt_seq); sess_text.writeInput(in_ids, input_ids.data(), input_ids.size()
//     * sizeof(int32_t)); const float *text_emb_ptr = static_cast<const float
//     *>(sess_text.run({}, debugCb, doSaturate));

//     std::vector<float> text_emb_buf;
//     const float *text_emb_host = sync_output_to_host(text_emb_ptr, 1 *
//     txt_seq * cfg.text_dim, text_emb_buf); std::vector<float>
//     text_emb(text_emb_host, text_emb_host + 1 * txt_seq * cfg.text_dim);

//     std::cout << "Sampling..." << std::endl;
//     std::vector<float> rope_cos, rope_sin;
//     compute_rope_cpu(txt_seq, latent_h, latent_w, cfg.head_dim,
//     cfg.rope_theta, rope_cos, rope_sin);

//     int num_steps = 4;
//     std::vector<float> schedule = get_flux_schedule(num_steps, img_seq);
//     std::vector<float> z(1 * cfg.latent_channels * latent_h * latent_w);
//     std::mt19937 gen(42);
//     std::normal_distribution<float> dist(0.0f, 1.0f);
//     for (uint64_t j = 0; j < z.size(); ++j)
//     {
//         z[j] = dist(gen);
//     }

//     for (int i = 0; i < num_steps; ++i)
//     {
//         float t_curr = schedule[i], dt = schedule[i + 1] - t_curr;

//         sess_trans.writeInput(in_latent, z.data(), z.size() * sizeof(float));
//         sess_trans.writeInput(in_txt_emb, text_emb.data(), text_emb.size() *
//         sizeof(float)); sess_trans.writeInput(in_t, &t_curr, sizeof(float));
//         sess_trans.writeInput(in_cos, rope_cos.data(), rope_cos.size() *
//         sizeof(float)); sess_trans.writeInput(in_sin, rope_sin.data(),
//         rope_sin.size() * sizeof(float));

//         Bucket b; // TODO: proper bucket to avoid weight loads?
//         const float *v_ptr = static_cast<const float *>(sess_trans.run(b,
//         debugCb, doSaturate));

//         std::vector<float> v_buf;
//         const float *v_host_ptr = sync_output_to_host(v_ptr, z.size(),
//         v_buf); v_ptr = v_host_ptr;

//         for (uint64_t j = 0; j < z.size(); ++j)
//             z[j] += v_ptr[j] * dt;
//         std::cout << "Step " << i + 1 << "/" << num_steps << " complete." <<
//         std::endl;
//     }

//     std::cout << "Executing VAE Decoder..." << std::endl;
//     sess_vae.writeInput(in_vae_latent, z.data(), z.size() * sizeof(float));
//     const float *img_ptr = static_cast<const float *>(sess_vae.run({},
//     debugCb, doSaturate));

//     std::vector<float> img_buf;
//     img_ptr = sync_output_to_host(img_ptr, 3 * height * width, img_buf);

//     std::vector<uint8_t> image_data(height * width * 3);
//     for (uint32_t y = 0; y < height; ++y)
//     {
//         for (uint32_t x = 0; x < width; ++x)
//         {
//             float r = img_ptr[0 * height * width + y * width + x];
//             float g = img_ptr[1 * height * width + y * width + x];
//             float b = img_ptr[2 * height * width + y * width + x];

//             r = std::max(0.0f, std::min(1.0f, (r + 1.0f) * 0.5f));
//             g = std::max(0.0f, std::min(1.0f, (g + 1.0f) * 0.5f));
//             b = std::max(0.0f, std::min(1.0f, (b + 1.0f) * 0.5f));

//             uint32_t idx = (y * width + x) * 3;
//             image_data[idx + 0] = static_cast<uint8_t>(r * 255.0f);
//             image_data[idx + 1] = static_cast<uint8_t>(g * 255.0f);
//             image_data[idx + 2] = static_cast<uint8_t>(b * 255.0f);
//         }
//     }
//     stbi_write_png("flux_output.png", width, height, 3, image_data.data(),
//     width * 3); std::cout << "Saved flux_output.png successfully!" <<
//     std::endl;
// }

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

    ArgParser parser("main", "Run a target model inference or plan execution.");
    parser.add_flag({"--only-plan"}, "Only plan the execution and generate cache.");
    parser.add_flag({"--disable-caching"}, "Disable dirty region session caching.");
    parser.add_option({"--write-refs"}, "Write reference/clean tensors to file.", "");
    parser.add_option({"--compare-refs"}, "Compare and validate outputs against reference file.", "");
    parser.add_positional("model",
                          "Name of the target model (flux-klein-4b, "
                          "gemma-3-270m, qwen-3.6-35b-a3b).",
                          "gemma-3-270m");

    if (!parser.parse(argc, argv))
    {
        return 1;
    }

    std::string model = parser.get_positional("model");
    bool only_plan = parser.get_flag("--only-plan");
    bool disable_caching = parser.get_flag("--disable-caching");
    std::string write_refs = parser.get_option("--write-refs");
    std::string compare_refs = parser.get_option("--compare-refs");

    Debug::ReferenceVerifier verifier;
    if (!verifier.init(write_refs, compare_refs))
    {
        return 1;
    }
    std::string dbg_mode = verifier.getMode();

    bool refOnly = !write_refs.empty();
    bool doSaturate = write_refs.empty();

    Graph *activeGraphPtr = nullptr;
    auto debugCb = [&](LogicalId logicalId, const KernelContext &ctx, const void *data) {
        verifier.verify(logicalId, ctx, data, activeGraphPtr);
    };

    if (model == "gemma-3-270m")
        run_gemma(only_plan, disable_caching, refOnly, doSaturate, debugCb, &activeGraphPtr);
    // else if (model == "flux-klein-4b")
    //     run_flux(only_plan, disable_caching, refOnly, doSaturate, debugCb,
    //     &activeGraphPtr);
    else if (model == "qwen-3.6-35b-a3b")
        run_qwen_35b(only_plan, disable_caching, refOnly, doSaturate, debugCb, &activeGraphPtr);
    else
        std::cout << "Model not implemented yet: " << model << std::endl;

    verifier.printSummary();
    return 0;
}