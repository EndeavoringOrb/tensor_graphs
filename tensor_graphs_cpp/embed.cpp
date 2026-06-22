// =============================================================================
// embed.cpp — jina-embeddings-v5-omni-nano-retrieval image embedding entry point
// =============================================================================
//
// Reads an image file, applies Qwen2VL-style preprocessing (resize 512×512,
// mean/std = 0.5/0.5, rescale 1/255), converts to Qwen3VL patch format
// (T_patch=2, P=16, C=3 → 1536-dim per patch) and runs the model.
//
// For a 512×512 input:
//   grid 32×32  →  1024 patches  →  patch_embed  →  12 vision blocks
//                →  merger (2×2)  →   256 tokens  →  12 text-encoder layers
//                →  last-token pool  →  L2-normalised 768-dim embedding
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/session.hpp"
#include "core/kernels.hpp"
#include "core/misc.hpp"
#include "core/repo.hpp"

#include "models/jina-embeddings-v5-omni-nano-retrieval.hpp"
#include "generated/kernels_all.gen.hpp"
#include "generated/build_context.gen.hpp"

// -----------------------------------------------------------------------------
// Qwen2VL / Qwen3VL preprocessing constants (from preprocessor_config.json)
// -----------------------------------------------------------------------------
static constexpr int IMAGE_SIZE = 512;        // 512×512 → 262144 px (== min_pixels)
static constexpr int PATCH_SIZE = 16;         // patch_size
static constexpr int TEMPORAL_PATCH_SIZE = 2; // temporal_patch_size
static constexpr int IN_CHANNELS = 3;
static constexpr int GRID_SIZE = IMAGE_SIZE / PATCH_SIZE;                                     // 32
static constexpr int NUM_PATCHES = GRID_SIZE * GRID_SIZE;                                     // 1024
static constexpr int PATCH_DIM = TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE * IN_CHANNELS; // 1536
static constexpr float IMAGE_MEAN = 0.5f;
static constexpr float IMAGE_STD = 0.5f;
static constexpr float RESCALE_FACTOR = 1.0f / 255.0f;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    if (!std::filesystem::exists(image_path))
    {
        std::cerr << "Error: Image file " << image_path << " does not exist." << std::endl;
        return 1;
    }

    // -----------------------------------------------------------------------
    // 1. Load image with STB Image (force 3-channel RGB)
    // -----------------------------------------------------------------------
    int width, height, channels;
    unsigned char *img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (!img_data)
    {
        std::cerr << "Error: Failed to load image " << image_path << std::endl;
        return 1;
    }
    channels = 3; // STB returns 3 channels when requested

    std::cout << "Successfully loaded image: " << image_path
              << " (" << width << "x" << height << ", " << channels << " channels)" << std::endl;

    // -----------------------------------------------------------------------
    // 2. Qwen2VL preprocessing
    //    a) Resize to 512×512 (nearest-neighbour — matches preprocessor's
    //       resample=3 for integer-coordinate sampling at this resolution).
    //    b) Rescale: pixel / 255
    //    c) Normalize: (x - 0.5) / 0.5   (mean=std=0.5)
    //    d) Convert to Qwen3VL patch format:
    //         For each 16×16 spatial patch, build a 1536-dim vector ordered
    //         as (C_in, T_patch, P, P) = (3, 2, 16, 16) — matching the Conv3d
    //         weight layout. The image is duplicated temporally (T_patch=2
    //         identical frames) as the Qwen2VL processor does for static images.
    // -----------------------------------------------------------------------
    std::cout << "Preprocessing image to " << IMAGE_SIZE << "x" << IMAGE_SIZE
              << " (" << NUM_PATCHES << " patches of " << PATCH_DIM << "-dim)..." << std::endl;

    // Normalised RGB image at 512×512, in CHW layout for easy patch extraction.
    std::vector<float> norm_image(IN_CHANNELS * IMAGE_SIZE * IMAGE_SIZE, 0.0f);
    for (int c = 0; c < IN_CHANNELS; ++c)
    {
        for (int y = 0; y < IMAGE_SIZE; ++y)
        {
            for (int x = 0; x < IMAGE_SIZE; ++x)
            {
                // Nearest-neighbour resize from source to 512×512
                int src_x = x * width / IMAGE_SIZE;
                int src_y = y * height / IMAGE_SIZE;
                // Clamp to source bounds (safety; stbi_load shouldn't need it)
                src_x = std::min(src_x, width - 1);
                src_y = std::min(src_y, height - 1);

                int src_idx = (src_y * width + src_x) * channels + c;
                float pixel = static_cast<float>(img_data[src_idx]) * RESCALE_FACTOR;
                pixel = (pixel - IMAGE_MEAN) / IMAGE_STD;

                int dst_idx = (c * IMAGE_SIZE + y) * IMAGE_SIZE + x; // CHW
                norm_image[dst_idx] = pixel;
            }
        }
    }
    stbi_image_free(img_data);

    // Build patch tensor: shape (1, NUM_PATCHES, PATCH_DIM) = (1, 1024, 1536)
    // Patch ordering: row-major over the (GRID_SIZE, GRID_SIZE) grid.
    // Within each patch: (C, T, P, P) layout to match the Conv3d weight.
    std::vector<float> patch_input(1 * NUM_PATCHES * PATCH_DIM, 0.0f);
    for (int gi = 0; gi < GRID_SIZE; ++gi) // patch row
    {
        for (int gj = 0; gj < GRID_SIZE; ++gj) // patch col
        {
            int patch_idx = gi * GRID_SIZE + gj;
            int dst_base = patch_idx * PATCH_DIM;

            for (int c = 0; c < IN_CHANNELS; ++c)
            {
                for (int t = 0; t < TEMPORAL_PATCH_SIZE; ++t) // both frames identical
                {
                    for (int p = 0; p < PATCH_SIZE; ++p)
                    {
                        for (int q = 0; q < PATCH_SIZE; ++q)
                        {
                            int img_y = gi * PATCH_SIZE + p;
                            int img_x = gj * PATCH_SIZE + q;
                            int src_idx = (c * IMAGE_SIZE + img_y) * IMAGE_SIZE + img_x;

                            // Flat index in (C, T, P, P) order
                            int dst_idx = dst_base + (((c * TEMPORAL_PATCH_SIZE) + t) * PATCH_SIZE + p) * PATCH_SIZE + q;
                            patch_input[dst_idx] = norm_image[src_idx];
                        }
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3. Build computational graph
    // -----------------------------------------------------------------------
    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 16ULL * 1024 * 1024 * 1024}};
#ifdef USE_CUDA
    bufferSizes[Backend::CUDA] = 16ULL * 1024 * 1024 * 1024;
#endif
    MemoryManager mem(bufferSizes);
    Graph g;

    JinaV5Config cfg;
    // Graph input is the patch tensor (1, 1024, 1536)
    uint32_t patch_input_id = g.input({1, NUM_PATCHES, PATCH_DIM},
                                      DType::FLOAT32, {}, StorageType::PERSISTENT);
    JinaV5OmniNanoRetrievalModel model(cfg, g, mem,
                                       "models/jinaai/jina-embeddings-v5-omni-nano-retrieval/model.safetensors");
    uint32_t rootId = model.build_graph(patch_input_id);

    // -----------------------------------------------------------------------
    // 4. Initialize Repo and compile session
    // -----------------------------------------------------------------------
    std::string gHash = computeGraphHash(g, {rootId});
    Repo repo("benchmarks/repo_jina-embeddings-v5-omni-nano-retrieval", gHash, true);

    std::string cache_file = "dirty_region_caches/jina-embeddings-v5-omni-nano-retrieval-cpp.bin";
    Session session(g, mem, rootId, cache_file, 0, &repo);

    std::cout << "Compiling computational graph..." << std::endl;
    session.compile(true);

    // -----------------------------------------------------------------------
    // 5. Run inference
    // -----------------------------------------------------------------------
    std::cout << "Running image embedding inference..." << std::endl;
    session.memManager.write(Backend::CPU, patch_input_id,
                             patch_input.data(), patch_input.size() * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    const float *device_output_ptr = static_cast<const float *>(session.run());
    auto end = std::chrono::high_resolution_clock::now();
    float runtimeMs = std::chrono::duration<float, std::milli>(end - start).count();

    // Copy result back if needed (Unified/Managed memory)
    constexpr int EMBEDDING_DIM = 768;
    std::vector<float> host_output(EMBEDDING_DIM);
    const float *output_ptr = device_output_ptr;
#ifdef USE_CUDA
    cudaPointerAttributes attrs;
    if (cudaPointerGetAttributes(&attrs, device_output_ptr) == cudaSuccess &&
        attrs.type == cudaMemoryTypeDevice)
    {
        cudaMemcpy(host_output.data(), device_output_ptr,
                   EMBEDDING_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        output_ptr = host_output.data();
    }
    else
    {
        std::memcpy(host_output.data(), device_output_ptr, EMBEDDING_DIM * sizeof(float));
    }
#else
    std::memcpy(host_output.data(), device_output_ptr, EMBEDDING_DIM * sizeof(float));
#endif

    std::cout << "\nInference complete in " << runtimeMs << " ms" << std::endl;
    std::cout << "Embedding vector (first 10 dimensions):" << std::endl;
    std::cout << "[";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << std::fixed << std::setprecision(6) << host_output[i];
        if (i < 9)
            std::cout << ", ";
    }
    std::cout << ", ...]" << std::endl;

    // Verify L2 norm ≈ 1.0 (sanity check for the final normalisation)
    float l2_norm = 0.0f;
    for (int i = 0; i < EMBEDDING_DIM; ++i)
        l2_norm += host_output[i] * host_output[i];
    l2_norm = std::sqrt(l2_norm);
    std::cout << "L2 norm of embedding: " << std::fixed << std::setprecision(6) << l2_norm
              << " (should be ~1.0)" << std::endl;

    return 0;
}