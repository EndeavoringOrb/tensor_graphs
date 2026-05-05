// File: tensor_graphs_cpp/main.cpp
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

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

void run_flux()
{
    FluxKlein4BModelConfig cfg;
#if USE_CUDA
    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 16ULL * 1024 * 1024 * 1024}, {Backend::CUDA, 12ULL * 1024 * 1024 * 1024}};
#else
    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 16ULL * 1024 * 1024 * 1024}};
#endif
    MemoryManager mem(bufferSizes);
    Graph g;

    std::cout << "Building FLUX-Klein-4B Graph..." << std::endl;
    FluxKlein4BModel flux(cfg, g, mem, "resources/flux.safetensors");

    // Allocate dynamic inputs
    uint32_t img_in = g.input({1, (uint32_t)(cfg.image_h * cfg.image_w / 4), (uint32_t)cfg.hidden_size}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t vec_in = g.input({1, (uint32_t)cfg.hidden_size}, DType::FLOAT32, {}, StorageType::PERSISTENT);

    uint32_t output_id = flux.build_graph(img_in, vec_in);

    Session session(g, mem, output_id, "dirty_region_caches/flux-klein-4b-cpp.jsonl");

    // Add minimal execution context bucket mapping
    std::unordered_map<uint32_t, std::vector<Region>> inputDirty;
    Region inputRegion;
    inputRegion.region = {Dim{0, 1}, Dim{0, (uint32_t)(cfg.image_h * cfg.image_w / 4)}};
    inputDirty[img_in] = {inputRegion};

    Region vecRegion;
    vecRegion.region = {Dim{0, 1}, Dim{0, (uint32_t)cfg.hidden_size}};
    inputDirty[vec_in] = {vecRegion};

    Region outputNeeded;
    outputNeeded.region = {Dim{0, 1}, Dim{0, (uint32_t)(cfg.image_h * cfg.image_w / 4)}, Dim{0, (uint32_t)cfg.hidden_size}};
    session.addManualBucket(inputDirty, {outputNeeded});

    // Provide dummy prompt vectors
    std::vector<float> dummy_img(1 * (cfg.image_h * cfg.image_w / 4) * cfg.hidden_size, 0.5f);
    std::vector<float> dummy_vec(1 * cfg.hidden_size, 0.1f);

    std::unordered_map<uint32_t, const void *> inputs;
    inputs[img_in] = dummy_img.data();
    inputs[vec_in] = dummy_vec.data();

    std::cout << "Running Inference..." << std::endl;
    const float *device_output_ptr = static_cast<const float *>(session.run(inputs));

    uint32_t H = cfg.image_h;
    uint32_t W = cfg.image_w;
    uint32_t channels = 3;

    const float *output_ptr = device_output_ptr;
#ifdef USE_CUDA
    std::vector<float> host_output(H * W * channels);
    cudaMemcpy(host_output.data(), device_output_ptr, host_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    output_ptr = host_output.data();
#endif

    std::vector<uint8_t> image_data(H * W * channels);
    for (size_t i = 0; i < H * W * channels; ++i)
    {
        float val = std::max(0.0f, std::min(1.0f, (output_ptr[i] + 1.0f) * 0.5f));
        image_data[i] = static_cast<uint8_t>(val * 255.0f);
    }

    std::string out_path = "flux_output.png";
    if (stbi_write_png(out_path.c_str(), W, H, channels, image_data.data(), W * channels))
        std::cout << "Successfully saved image to " << out_path << std::endl;
    else
        std::cerr << "Failed to save image." << std::endl;
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