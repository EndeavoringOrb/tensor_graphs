// =============================================================================
// embed.cpp — jina-embeddings-v5-omni-nano-retrieval image embedding entry
// point
// =============================================================================
//
// UPDATED for smart_resize: the server now accepts variable image dimensions
// (any size where both H and W are divisible by 32, within [min_pixels,
// max_pixels]).  The graph is rebuilt and recompiled automatically when the
// image dimensions change.
//
// Reads an image (from shared memory in server mode, or from a file in
// standalone mode), applies Qwen2VL-style preprocessing (mean/std = 0.5/0.5,
// rescale 1/255), converts to Qwen3VL patch format (T_patch=2, P=16, C=3 →
// 1536-dim per patch) and runs the model.
//
// Shared memory layout (SharedMemoryPayload):
//   [0..19]             header: state, width, height, channels, status (5 ×
//   int32) [20..20+MAX_PIX*3]  pixel_data (H×W×3 uint8, row-major)
//   [emb_off..emb_off+3072]  embedding (768 × float32)
//
// Changes from the original:
//   1. pixel_data buffer enlarged from 512*512*3 to MAX_PIXELS*3 (≈ 3.9 MB)
//      to accommodate smart_resize'd images up to max_pixels (1,310,720).
//   2. No internal resize to 512×512 — the actual width/height from the
//      shared memory header are used directly.
//   3. The graph + session are rebuilt when image dimensions change.
//   4. Per-size compilation cache files
//      (dirty_region_caches/jina-v5-<w>x<h>.bin) avoid recompilation overhead
//      for repeated sizes.
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/argparse.hpp"
#include "core/debug.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"
#include "core/repo.hpp"
#include "core/session.hpp"
#include "core/shape_propagator.hpp"
#include "core/types.hpp"
#include "generated/build_context.gen.hpp"
#include "generated/kernels_all.gen.hpp"
#include "models/jina-embeddings-v5-omni-nano-retrieval.hpp"

#ifdef TG_OS_WINDOWS
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// -----------------------------------------------------------------------------
// Qwen2VL / Qwen3VL preprocessing constants (from preprocessor_config.json)
// -----------------------------------------------------------------------------
static constexpr int PATCH_SIZE = 16;
static constexpr int TEMPORAL_PATCH_SIZE = 2;
static constexpr int IN_CHANNELS = 3;
static constexpr int SPATIAL_MERGE_SIZE = 2;
static constexpr int SMART_RESIZE_FACTOR = PATCH_SIZE * SPATIAL_MERGE_SIZE; // 32
static constexpr int MIN_PIXELS = 262144;                                   // from preprocessor_config.json
static constexpr int MAX_PIXELS = 1310720;                                  // from preprocessor_config.json
static constexpr int PATCH_DIM = TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE * IN_CHANNELS; // 1536
static constexpr int EMBEDDING_DIM = 768;
static constexpr float IMAGE_MEAN = 0.5f;
static constexpr float IMAGE_STD = 0.5f;
static constexpr float RESCALE_FACTOR = 1.0f / 255.0f;

// Shared memory layout constants
static constexpr int SHM_HEADER_SIZE = 20;                   // 5 × int32
static constexpr int SHM_PIXEL_DATA_SIZE = MAX_PIXELS * 3;   // 3,932,160
static constexpr int SHM_EMBEDDING_SIZE = EMBEDDING_DIM * 4; // 3,072
static constexpr int SHM_EMBEDDING_OFFSET = SHM_HEADER_SIZE + SHM_PIXEL_DATA_SIZE;
static constexpr int SHM_TOTAL_SIZE = SHM_EMBEDDING_OFFSET + SHM_EMBEDDING_SIZE; // 3,935,252

#pragma pack(push, 1)
struct SharedMemoryPayload
{
    // 0: Idle/Ready (C++ is waiting, Python can write)
    // 1: Python has written data (C++ should process)
    // 2: C++ finished (Python can read)
    // 3: Exit/Shutdown
    volatile int32_t state;

    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t status; // 0 = success, -1 = error
    uint8_t pixel_data[SHM_PIXEL_DATA_SIZE];
    float embedding[EMBEDDING_DIM];
};
#pragma pack(pop)

static_assert(sizeof(SharedMemoryPayload) == SHM_TOTAL_SIZE, "SharedMemoryPayload size mismatch");

// -----------------------------------------------------------------------------
// smart_resize — mirrors Qwen2VL's algorithm with factor=32 for Qwen3VL.
// Returns (new_height, new_width) where both are divisible by 32 and total
// pixels are within [min_pixels, max_pixels], preserving aspect ratio.
// -----------------------------------------------------------------------------
static std::pair<int, int> smart_resize(int height, int width, int factor = SMART_RESIZE_FACTOR,
                                        int min_pixels = MIN_PIXELS, int max_pixels = MAX_PIXELS)
{
    if (height <= 0 || width <= 0)
        throw std::runtime_error("smart_resize: invalid dimensions");

    double ratio = (double)std::max(height, width) / std::min(height, width);
    if (ratio > 200.0)
        throw std::runtime_error("absolute aspect ratio must be less than 200");

    int h_bar = (int)std::round((double)height / factor) * factor;
    int w_bar = (int)std::round((double)width / factor) * factor;

    if (h_bar * w_bar > max_pixels)
    {
        double beta = std::sqrt((double)(height * width) / max_pixels);
        h_bar = (int)std::floor((double)height / beta / factor) * factor;
        w_bar = (int)std::floor((double)width / beta / factor) * factor;
    }
    else if (h_bar * w_bar < min_pixels)
    {
        double beta = std::sqrt((double)min_pixels / (height * width));
        h_bar = (int)std::ceil((double)height * beta / factor) * factor;
        w_bar = (int)std::ceil((double)width * beta / factor) * factor;
    }

    if (h_bar <= 0 || w_bar <= 0)
        throw std::runtime_error("smart_resize produced invalid dimensions");

    return {h_bar, w_bar};
}

// -----------------------------------------------------------------------------
// Compiled session state — rebuilt when image dimensions change.
// Pre-allocated patch_input / norm_image buffers are reused across calls of the
// same dimensions to avoid per-image malloc/free churn.
// -----------------------------------------------------------------------------
struct CompiledSession
{
    std::unique_ptr<Graph> graph;
    std::unique_ptr<Session> session;
    std::unique_ptr<JinaV5Config> cfg;
    LogicalId patch_input_id;
    LogicalId root_id;
    int width = 0;
    int height = 0;
    bool has_run = false;

    // Reusable staging buffers — sized to match the current (width, height).
    // build_patch_input_inplace() and normalize_image_inplace() write into them.
    std::vector<float> norm_image;  // CHW, size = 3 * H * W
    std::vector<float> patch_input; // (1, num_patches, PATCH_DIM)
};

// Build (or rebuild) the graph + session for the given image dimensions.
static void build_session(CompiledSession &cs, MemoryManager &mem, int width, int height,
                          const std::string &weights_path, bool disable_caching = false)
{
    int grid_h = height / PATCH_SIZE;
    int grid_w = width / PATCH_SIZE;
    int num_patches = grid_h * grid_w;

    std::vector<uint32_t> inShape = {1, (uint32_t)num_patches, (uint32_t)PATCH_DIM};
    std::vector<uint32_t> outShape = {1, EMBEDDING_DIM};

    cs.graph = std::make_unique<Graph>();
    cs.cfg = std::make_unique<JinaV5Config>((uint32_t)height, (uint32_t)width);

    cs.patch_input_id = cs.graph->input(inShape, DType::FLOAT32, {});

    JinaV5OmniNanoRetrievalModel model(*cs.cfg, *cs.graph, mem, weights_path);
    cs.root_id = model.build_graph(cs.patch_input_id);

    std::string gHash = computeGraphHash(*cs.graph, {cs.root_id});
    Repo repo("benchmarks/repo_jina-embeddings-v5-omni-nano-retrieval", gHash, true);

    std::string cache_file =
        "dirty_region_caches/jina-v5-" + std::to_string(width) + "x" + std::to_string(height) + ".bin";

    cs.session = std::make_unique<Session>(*cs.graph, mem, cs.root_id, cache_file, 0, &repo, disable_caching);

    // Register a bucket where ONLY the image input is dirty (all weights are
    // clean/static)
    std::unordered_map<LogicalId, std::vector<Region>> inputDirty;
    inputDirty[cs.patch_input_id] = makeFull(inShape);

    std::vector<Region> outputNeeded = makeFull(outShape);

    std::cout << "[build_session] added bucket in: " << toString(inShape) << ", out: " << toString(outShape)
              << std::endl;
    cs.session->addBucket(inputDirty, outputNeeded);

    cs.session->compile(true);

    cs.width = width;
    cs.height = height;
    cs.has_run = false; // Reset run tracking flag on rebuild/compile

    // Pre-allocate reusable staging buffers for this image size so the polling
    // loop doesn't allocate/free on every single image.
    cs.norm_image.assign((uint64_t)IN_CHANNELS * width * height, 0.0f);
    cs.patch_input.assign((uint64_t)1 * num_patches * PATCH_DIM, 0.0f);
}

static void build_patch_input_inplace(const std::vector<float> &norm_image, int width, int height,
                                      std::vector<float> &patch_input)
{
    int grid_h = height / PATCH_SIZE;
    int grid_w = width / PATCH_SIZE;
    int num_patches = grid_h * grid_w;

    if ((int)patch_input.size() != 1 * num_patches * PATCH_DIM)
        patch_input.assign((uint64_t)1 * num_patches * PATCH_DIM, 0.0f);

    // Per-patch flat layout: c * (T * P * Q) + t * (P * Q) + p * Q + q  ← (C, T,
    // P, Q) The HF processor flattens patches as (C, T, P, Q), matching the
    // layout expected by PyTorch when F.linear is applied to a Conv3d weight
    // viewed as (Out, In).
    int spatial_stride = PATCH_SIZE * PATCH_SIZE;
    int temporal_stride = TEMPORAL_PATCH_SIZE * spatial_stride;

    for (int gi = 0; gi < grid_h; ++gi)
    {
        for (int gj = 0; gj < grid_w; ++gj)
        {
            int patch_idx = gi * grid_w + gj;
            int dst_base = patch_idx * PATCH_DIM;

            for (int t = 0; t < TEMPORAL_PATCH_SIZE; ++t)
            {
                for (int p = 0; p < PATCH_SIZE; ++p)
                {
                    for (int q = 0; q < PATCH_SIZE; ++q)
                    {
                        int img_y = gi * PATCH_SIZE + p;
                        int img_x = gj * PATCH_SIZE + q;

                        int local_offset = t * spatial_stride + p * PATCH_SIZE + q;

                        // norm_image is CHW: (c * H + y) * W + x
                        int src_r = ((0 * height + img_y) * width + img_x);
                        int src_g = ((1 * height + img_y) * width + img_x);
                        int src_b = ((2 * height + img_y) * width + img_x);

                        patch_input[dst_base + 0 * temporal_stride + local_offset] = norm_image[src_r];
                        patch_input[dst_base + 1 * temporal_stride + local_offset] = norm_image[src_g];
                        patch_input[dst_base + 2 * temporal_stride + local_offset] = norm_image[src_b];
                    }
                }
            }
        }
    }
}

// Backwards-compatible wrapper (still allocates) — used by standalone file
// mode.
static std::vector<float> build_patch_input(const std::vector<float> &norm_image, int width, int height)
{
    std::vector<float> out;
    build_patch_input_inplace(norm_image, width, height, out);
    return out;
}

// In-place normalize: writes directly into cs.norm_image, no per-call alloc.
static void normalize_image_inplace(const uint8_t *pixel_data, int width, int height, int channels,
                                    std::vector<float> &norm_image)
{
    if ((int)norm_image.size() != IN_CHANNELS * width * height)
        norm_image.assign((uint64_t)IN_CHANNELS * width * height, 0.0f);

    for (int c = 0; c < IN_CHANNELS; ++c)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int src_idx = (y * width + x) * channels + c;
                float pixel = static_cast<float>(pixel_data[src_idx]) * RESCALE_FACTOR;
                pixel = (pixel - IMAGE_MEAN) / IMAGE_STD;
                int dst_idx = (c * height + y) * width + x; // CHW
                norm_image[dst_idx] = pixel;
            }
        }
    }
}

// Normalize raw uint8 HWC pixel data to float32 CHW, mean/std normalized.
//
// Wrapper kept for backwards compatibility — internally delegates to the
// in-place variant so both code paths produce identical output.  Prefer
// normalize_image_inplace() in new code to avoid the per-call allocation.
static std::vector<float> normalize_image(const uint8_t *pixel_data, int width, int height, int channels)
{
    std::vector<float> out;
    normalize_image_inplace(pixel_data, width, height, channels, out);
    return out;
}

int main(int argc, char *argv[])
{
    ArgParser parser("embed", "Embed an image using Jina embeddings.");
    parser.add_flag({"--server"}, "Run in shared-memory server mode.");
    parser.add_flag({"--disable-caching"}, "Disable dirty region caching.");
    parser.add_option({"--write-refs"}, "Write reference/clean tensors to file.", "");
    parser.add_option({"--compare-refs"}, "Compare and validate outputs against reference file.", "");
    parser.add_positional("image_path", "Path to input image file (optional in server mode).");

    if (!parser.parse(argc, argv))
    {
        return 1;
    }

    bool is_server = parser.get_flag("--server");
    bool disable_caching = parser.get_flag("--disable-caching");
    std::string write_refs = parser.get_option("--write-refs");
    std::string compare_refs = parser.get_option("--compare-refs");
    std::string image_path = parser.get_positional("image_path");

    if (!is_server)
    {
        if (image_path.empty())
        {
            std::cerr << "Error: Image file path is required in standalone mode." << std::endl;
            return 1;
        }
        else if (!std::filesystem::exists(image_path))
        {
            std::cerr << "Error: Image file " << image_path << " does not exist." << std::endl;
            return 1;
        }
    }

    static const std::string WEIGHTS_PATH = "models/jinaai/jina-embeddings-v5-omni-nano-retrieval/model.safetensors";

    // Initialize the ReferenceVerifier
    Debug::ReferenceVerifier verifier;
    if (!verifier.init(write_refs, compare_refs))
    {
        return 1;
    }
    std::string dbg_mode = verifier.getMode();

    // -----------------------------------------------------------------------
    // Memory manager (shared across all compiled sessions)
    // -----------------------------------------------------------------------
    // TODO: make this and getDefaultBufferSizes load from some common place
    std::unordered_map<MemSpace, uint64_t> bufferSizes = {{MemSpace{1, HandleType::CPP}, 16ULL * 1024 * 1024 * 1024}};
#ifdef TG_USE_CUDA
    bufferSizes[MemSpace{2, HandleType::CUDA}] = 16ULL * 1024 * 1024 * 1024;
#endif
    if (HardwareCaps::get().has_opencl)
    {
        bufferSizes[MemSpace{1, HandleType::OPENCL}] = 1ULL * 1024 * 1024 * 1024;
    }
    MemoryManager mem(bufferSizes);

    SharedMemoryPayload *shm_payload = nullptr;
#ifdef TG_OS_WINDOWS
    HANDLE hMapFile = NULL;
#else
    int shm_fd = -1;
#endif

    if (is_server)
    {
        // -------------------------------------------------------------------
        // Server mode — shared memory + polling loop
        // -------------------------------------------------------------------
#ifdef TG_OS_WINDOWS
        for (int i = 0; i < 20; ++i)
        {
            hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, "tg_embed_shm");
            if (hMapFile != NULL)
                break;
            hMapFile = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(SharedMemoryPayload),
                                          "tg_embed_shm");
            if (hMapFile != NULL)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        if (hMapFile == NULL)
        {
            std::cerr << "[Server Error] Failed to open/create shared memory" << std::endl;
            return 1;
        }
        shm_payload =
            (SharedMemoryPayload *)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedMemoryPayload));
        if (shm_payload == nullptr)
        {
            std::cerr << "[Server Error] MapViewOfFile failed" << std::endl;
            CloseHandle(hMapFile);
            return 1;
        }
#else
        for (int i = 0; i < 20; ++i)
        {
            shm_fd = shm_open("/tg_embed_shm", O_RDWR, 0666);
            if (shm_fd >= 0)
                break;
            shm_fd = shm_open("/tg_embed_shm", O_CREAT | O_RDWR, 0666);
            if (shm_fd >= 0)
            {
                if (ftruncate(shm_fd, sizeof(SharedMemoryPayload)) == -1)
                    std::cerr << "[Server Warning] ftruncate failed" << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        if (shm_fd < 0)
        {
            std::cerr << "[Server Error] Failed to open/create shared memory" << std::endl;
            return 1;
        }
        shm_payload = (SharedMemoryPayload *)mmap(NULL, sizeof(SharedMemoryPayload), PROT_READ | PROT_WRITE, MAP_SHARED,
                                                  shm_fd, 0);
        if (shm_payload == MAP_FAILED)
        {
            std::cerr << "[Server Error] mmap failed" << std::endl;
            close(shm_fd);
            return 1;
        }
#endif

        std::cout << "[Server] Entering polling loop. Waiting for image payload..." << std::endl;
        shm_payload->state = 0;

        CompiledSession cs;

        while (true)
        {
            if (shm_payload->state == 1)
            {
                int width = shm_payload->width;
                int height = shm_payload->height;
                int channels = shm_payload->channels;

                if (width <= 0 || height <= 0 || channels != 3)
                {
                    std::cerr << "[Server Error] Invalid image dimensions or channels" << std::endl;
                    shm_payload->status = -1;
                    shm_payload->state = 2;
                    continue;
                }

                // Rebuild graph if dimensions changed
                if (width != cs.width || height != cs.height)
                {
                    if (width % SMART_RESIZE_FACTOR != 0 || height % SMART_RESIZE_FACTOR != 0)
                    {
                        throw std::runtime_error("Image dimensions must be divisible by " +
                                                 std::to_string(SMART_RESIZE_FACTOR) + " (got " +
                                                 std::to_string(width) + "x" + std::to_string(height) + ")");
                    }
                    int total_pixels = width * height;
                    if (total_pixels < MIN_PIXELS || total_pixels > MAX_PIXELS)
                    {
                        throw std::runtime_error("Image pixel count " + std::to_string(total_pixels) + " is outside [" +
                                                 std::to_string(MIN_PIXELS) + ", " + std::to_string(MAX_PIXELS) + "]");
                    }

                    std::cout << "[Server] Building graph for " << width << "x" << height << "..." << std::endl;
                    build_session(cs, mem, width, height, WEIGHTS_PATH, disable_caching);
                    std::cout << "[Server] Graph ready (" << cs.cfg->num_patches << " patches, " << cs.cfg->num_merged
                              << " merged tokens, " << cs.cfg->text_seq_len << " text tokens)" << std::endl;
                }

                // 1. Normalize image in-place
                normalize_image_inplace(shm_payload->pixel_data, width, height, channels, cs.norm_image);

                // 2. Build patch input in-place
                build_patch_input_inplace(cs.norm_image, width, height, cs.patch_input);

                // 3. Write input and run
                cs.session->writeInput(cs.patch_input_id, cs.patch_input.data(), cs.patch_input.size() * sizeof(float));

                // Use the incremental bucket if we have already loaded the static
                // weights on the first run
                Bucket b;
                if (cs.has_run)
                {
                    b.inputDirtyRegions[cs.patch_input_id] = makeFull(cs.graph->getNode(cs.patch_input_id).getShape());
                    b.outputNeededRegion = makeFull(cs.graph->getNode(cs.root_id).getShape());
                }

                auto cb = [&](LogicalId logicalId, std::string &kernel_name, const KernelContext &ctx,
                              const void *data) { verifier.verify(logicalId, kernel_name, ctx, data, cs.graph.get()); };

                const float *device_output_ptr = static_cast<const float *>(cs.session->run(b, cb));
                cs.has_run = true; // Mark as run completed so subsequent passes bypass
                                   // weight copy

                // 4. Copy embedding back
                float host_output[EMBEDDING_DIM];
#ifdef TG_USE_CUDA
                cudaPointerAttributes attrs;
                if (cudaPointerGetAttributes(&attrs, device_output_ptr) == cudaSuccess &&
                    attrs.type == cudaMemoryTypeDevice)
                {
                    cudaMemcpy(host_output, device_output_ptr, EMBEDDING_DIM * sizeof(float), cudaMemcpyDeviceToHost);
                }
                else
                {
                    std::memcpy(host_output, device_output_ptr, EMBEDDING_DIM * sizeof(float));
                }
#else
                std::memcpy(host_output, device_output_ptr, EMBEDDING_DIM * sizeof(float));
#endif

                std::memcpy(shm_payload->embedding, host_output, EMBEDDING_DIM * sizeof(float));
                shm_payload->status = 0;
                shm_payload->state = 2;
            }
            else if (shm_payload->state == 3)
            {
                std::cout << "[Server] Received exit signal. Shutting down..." << std::endl;
                break;
            }

            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }

#ifdef TG_OS_WINDOWS
        UnmapViewOfFile(shm_payload);
        CloseHandle(hMapFile);
#else
        munmap(shm_payload, sizeof(SharedMemoryPayload));
        close(shm_fd);
#endif
        return 0;
    }
    else
    {
        // -------------------------------------------------------------------
        // Standalone file mode
        // -------------------------------------------------------------------
        int width, height, channels;
        unsigned char *img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
        if (!img_data)
        {
            std::cerr << "Error: Failed to load image " << image_path << std::endl;
            return 1;
        }
        channels = 3;

        std::cout << "Loaded image: " << image_path << " (" << width << "x" << height << ", " << channels
                  << " channels)" << std::endl;

        // smart_resize to model-compatible dimensions
        auto [new_h, new_w] = smart_resize(height, width);
        std::cout << "smart_resize: " << width << "x" << height << " -> " << new_w << "x" << new_h << std::endl;

        // If the image needs resizing, do it with a simple bilinear via
        // stb_image_resize
        unsigned char *resized_data = img_data;
        int final_w = width, final_h = height;
        bool needs_resize = (new_w != width || new_h != height);

        if (needs_resize)
        {
// stb_image_resize is in a separate header; if not available, fall back to
// nearest-neighbor.  For production, install stb_image_resize2.h.
#ifdef STB_IMAGE_RESIZE_IMPLEMENTATION
            resized_data = (unsigned char *)malloc(new_w * new_h * 3);
            stbir_resize_uint8(img_data, width, height, 0, resized_data, new_w, new_h, 0, 3);
            stbi_image_free(img_data);
#else
            // Simple nearest-neighbor fallback
            resized_data = (unsigned char *)malloc(new_w * new_h * 3);
            for (int y = 0; y < new_h; ++y)
            {
                int src_y = y * height / new_h;
                src_y = std::min(src_y, height - 1);
                for (int x = 0; x < new_w; ++x)
                {
                    int src_x = x * width / new_w;
                    src_x = std::min(src_x, width - 1);
                    int src_idx = (src_y * width + src_x) * 3;
                    int dst_idx = (y * new_w + x) * 3;
                    resized_data[dst_idx] = img_data[src_idx];
                    resized_data[dst_idx + 1] = img_data[src_idx + 1];
                    resized_data[dst_idx + 2] = img_data[src_idx + 2];
                }
            }
            stbi_image_free(img_data);
#endif
            final_w = new_w;
            final_h = new_h;
        }

        std::cout << "Preprocessing (" << final_w << "x" << final_h << ", "
                  << (final_w / PATCH_SIZE) * (final_h / PATCH_SIZE) << " patches)..." << std::endl;

        // Build graph for this image's dimensions (build_session also
        // pre-allocates cs.norm_image and cs.patch_input for reuse).
        CompiledSession cs;
        build_session(cs, mem, final_w, final_h, WEIGHTS_PATH, disable_caching);

        // Normalize in-place into cs.norm_image
        normalize_image_inplace(resized_data, final_w, final_h, 3, cs.norm_image);

        // Build patch input in-place into cs.patch_input (uses (T,P,Q,C) layout)
        build_patch_input_inplace(cs.norm_image, final_w, final_h, cs.patch_input);

        // Run
        std::cout << "Running inference..." << std::endl;
        cs.session->writeInput(cs.patch_input_id, cs.patch_input.data(), cs.patch_input.size() * sizeof(float));

        auto start = std::chrono::high_resolution_clock::now();

        auto cb = [&](LogicalId logicalId, std::string &kernel_name, const KernelContext &ctx, const void *data) {
            verifier.verify(logicalId, kernel_name, ctx, data, cs.graph.get());
        };

        Bucket b;
        const float *device_output_ptr = static_cast<const float *>(cs.session->run(b, cb));

        auto end = std::chrono::high_resolution_clock::now();
        float runtimeMs = std::chrono::duration<float, std::milli>(end - start).count();

        float host_output[EMBEDDING_DIM];
#ifdef TG_USE_CUDA
        cudaPointerAttributes attrs;
        if (cudaPointerGetAttributes(&attrs, device_output_ptr) == cudaSuccess && attrs.type == cudaMemoryTypeDevice)
        {
            cudaMemcpy(host_output, device_output_ptr, EMBEDDING_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        }
        else
        {
            std::memcpy(host_output, device_output_ptr, EMBEDDING_DIM * sizeof(float));
        }
#else
        std::memcpy(host_output, device_output_ptr, EMBEDDING_DIM * sizeof(float));
#endif

        if (needs_resize)
            free(resized_data);

        std::cout << "\nInference complete in " << runtimeMs << " ms" << std::endl;
        std::cout << "Embedding (first 10 dims): [";
        for (int i = 0; i < 10; ++i)
        {
            std::cout << std::fixed << std::setprecision(6) << host_output[i];
            if (i < 9)
                std::cout << ", ";
        }
        std::cout << ", ...]" << std::endl;

        float l2_norm = 0.0f;
        for (int i = 0; i < EMBEDDING_DIM; ++i)
            l2_norm += host_output[i] * host_output[i];
        l2_norm = std::sqrt(l2_norm);
        std::cout << "L2 norm: " << std::fixed << std::setprecision(6) << l2_norm << " (should be ~1.0)" << std::endl;
    }

    verifier.printSummary();
    return 0;
}