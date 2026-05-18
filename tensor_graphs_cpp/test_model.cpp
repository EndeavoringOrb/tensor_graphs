// --- tensor_graphs_cpp/test_model.cpp ---
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <functional>
#include <filesystem>
#include <sstream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/session.hpp"
#include "core/kernels.hpp"
#include "core/misc.hpp"

// Model Definitions
#include "models/gemma-3-270m.hpp"
#include "models/flux-klein-4b.hpp"

#include "generated/kernels_all.gen.hpp"
#include "generated/build_context.gen.hpp"

namespace fs = std::filesystem;

std::string getNextAvailableBinPath(const std::string &folderPath = "sessions")
{
    // Ensure the directory exists
    if (!fs::exists(folderPath))
    {
        fs::create_directories(folderPath);
    }

    int counter = 0;
    while (true)
    {
        std::ostringstream oss;
        oss << std::setw(5) << std::setfill('0') << counter << ".bin";

        fs::path fullPath = fs::path(folderPath) / oss.str();

        if (!fs::exists(fullPath))
        {
            return fullPath.string();
        }

        counter++;
    }
    return "";
}

std::vector<float> extractTensorData(const TensorView &view, const void *data)
{
    std::vector<float> result;
    uint64_t elems = countElements(view);
    result.reserve(elems);
    if (view.dtype == DType::FLOAT32)
    {
        const float *src = static_cast<const float *>(data);
        for (uint64_t i = 0; i < elems; ++i)
            result.push_back(src[getStridedIndex(i, view.getShape(), view.strides)]);
    }
    else if (view.dtype == DType::INT32)
    {
        const int32_t *src = static_cast<const int32_t *>(data);
        for (uint64_t i = 0; i < elems; ++i)
            result.push_back(static_cast<float>(src[getStridedIndex(i, view.getShape(), view.strides)]));
    }
    else if (view.dtype == DType::BF16)
    {
        const uint16_t *src = static_cast<const uint16_t *>(data);
        for (uint64_t i = 0; i < elems; ++i)
        {
            uint32_t bits = static_cast<uint32_t>(src[getStridedIndex(i, view.getShape(), view.strides)]) << 16;
            float val;
            std::memcpy(&val, &bits, 4);
            result.push_back(val);
        }
    }
    else if (view.dtype == DType::BOOL)
    {
        const uint8_t *src = static_cast<const uint8_t *>(data);
        for (uint64_t i = 0; i < elems; ++i)
            result.push_back(static_cast<float>(src[getStridedIndex(i, view.getShape(), view.strides)]));
    }
    return result;
}

struct RefIndexEntry
{
    uint64_t fileOffset;
    uint64_t numElements;
    std::string opName;
};

// === Gemma Integration ===

void runGemma(bool refOnly, bool doSaturate, std::function<void(uint32_t logicalId, const std::string &opName, const TensorView &view, const std::vector<float> &data)> callback)
{
    KernelRegistry::get().setReferenceOnly(refOnly);

    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 16ULL * 1024 * 1024 * 1024}};
#if USE_CUDA
    bufferSizes[Backend::CUDA] = 16ULL * 1024 * 1024 * 1024;
#endif
    MemoryManager mem(bufferSizes);
    Graph g;

    Gemma3ModelConfig cfg;
    uint32_t maxSeqLen = 8;

    uint32_t inputIdsId = g.input({1, maxSeqLen}, DType::INT32, {}, StorageType::PERSISTENT);

    Gemma3Model gemma(cfg, maxSeqLen, g, mem, "resources/model.safetensors");
    uint32_t rootId = gemma.build_graph(inputIdsId);

    std::vector<int32_t> input_ids = {2, 9259, 0, 0, 0, 0, 0, 0};

    // Allocate the input in persistent memory and feed into model mapping
    uint64_t sizeBytes = maxSeqLen * getDTypeSize(DType::INT32);
    mem.allocate(Backend::CPU, inputIdsId, sizeBytes, StorageType::PERSISTENT);

    std::unordered_map<uint32_t, const void *> inputs = {{inputIdsId, input_ids.data()}};

    // Use an empty string for the cache to prevent collisions between sequential runs
    Session sess(g, mem, rootId, getNextAvailableBinPath("test_model_cache_gemma"));

    auto debugCb = [&](uint32_t logicalId, const TensorView &view, const void *data)
    {
        if (!callback)
            return;

        std::vector<float> extracted = extractTensorData(view, data);
        std::string opName = "UNKNOWN";

        if (g.hasNode(logicalId))
        {
            auto node = g.getNode(logicalId);
            opName = toString(node.opType);
            if (node.opType == OpType::FUSED)
            {
                opName = "FUSED_" + node.opName;
            }
        }

        callback(logicalId, opName, view, extracted);
    };

    sess.compile(doSaturate);
    sess.run(inputs, debugCb, doSaturate);
}

// === Flux Integration ===

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
    std::vector<int32_t> input_ids(txt_seq, 151643);

    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return input_ids;
    }

    std::string part;
    size_t count = 0;

    while (std::getline(file, part, ',') && count < txt_seq)
    {
        if (!part.empty())
        {
            input_ids[count++] = static_cast<int32_t>(std::stoi(part));
        }
    }

    return input_ids;
}

void runFlux(bool refOnly, bool doSaturate, std::function<void(uint32_t logicalId, const std::string &opName, const TensorView &view, const std::vector<float> &data)> callback)
{
    KernelRegistry::get().setReferenceOnly(refOnly);

    FluxConfig cfg;
#if USE_CUDA
    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 24ULL * 1024 * 1024 * 1024}, {Backend::CUDA, 24ULL * 1024 * 1024 * 1024}};
#else
    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 24ULL * 1024 * 1024 * 1024}};
#endif
    MemoryManager mem(bufferSizes);

    uint32_t width = 512, height = 512;
    uint32_t latent_w = width / 16, latent_h = height / 16;
    uint32_t txt_seq = cfg.text_max_seq, img_seq = latent_h * latent_w, total_seq = txt_seq + img_seq;

    auto shared_alloc = std::make_shared<IdAllocator>();

    // 1. Build Text Encoder
    Graph g_text;
    g_text.allocator = shared_alloc;
    FluxTextEncoder text_encoder(cfg, g_text, mem, "flux-klein-4b/text_encoder");
    uint32_t in_ids = g_text.input({1, txt_seq}, DType::INT32, {}, StorageType::PERSISTENT);
    Session sess_text(g_text, mem, text_encoder.build_graph(in_ids), getNextAvailableBinPath("test_model_cache_flux_text"));

    // 2. Build Transformer
    Graph g_trans;
    g_trans.allocator = shared_alloc;
    FluxTransformer trans(cfg, g_trans, mem, "flux-klein-4b/transformer", latent_h, latent_w);
    uint32_t in_latent = g_trans.input({1, cfg.latent_channels, latent_h, latent_w}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_txt_emb = g_trans.input({1, txt_seq, cfg.text_dim}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_t = g_trans.input({1}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_cos = g_trans.input({1, 1, total_seq, cfg.head_dim}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    uint32_t in_sin = g_trans.input({1, 1, total_seq, cfg.head_dim}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    Session sess_trans(g_trans, mem, trans.build_graph(in_latent, in_txt_emb, in_t, in_cos, in_sin), getNextAvailableBinPath("test_model_cache_flux_trans"));

    // 3. Build VAE Decoder
    Graph g_vae;
    g_vae.allocator = shared_alloc;
    FluxVAEDecoder vae(cfg, g_vae, mem, "flux-klein-4b/vae", latent_h, latent_w);
    uint32_t in_vae_latent = g_vae.input({1, cfg.vae_channels, latent_h, latent_w}, DType::FLOAT32, {}, StorageType::PERSISTENT);
    Session sess_vae(g_vae, mem, vae.build_graph(in_vae_latent), getNextAvailableBinPath("test_model_cache_flux_vae"));

    // Compile explicitly
    sess_text.compile(doSaturate);
    sess_trans.compile(doSaturate);
    sess_vae.compile(doSaturate);

    // Setup debug callback generic helper
    auto makeDebugCb = [&](Graph &g)
    {
        return [&](uint32_t logicalId, const TensorView &view, const void *data)
        {
            if (!callback)
                return;
            std::vector<float> extracted = extractTensorData(view, data);
            std::string opName = "UNKNOWN";
            if (g.hasNode(logicalId))
            {
                auto node = g.getNode(logicalId);
                opName = toString(node.opType);
                if (node.opType == OpType::FUSED)
                {
                    opName = "FUSED_" + node.opName;
                }
            }
            callback(logicalId, opName, view, extracted);
        };
    };

    // --- Run Text Encoder ---
    std::vector<int32_t> input_ids = load_tokens_from_file("toks.txt", txt_seq);
    std::unordered_map<uint32_t, const void *> text_inputs = {{in_ids, input_ids.data()}};

    const float *text_emb_ptr = static_cast<const float *>(sess_text.run(text_inputs, makeDebugCb(g_text), doSaturate));

    std::vector<float> text_emb(1 * txt_seq * cfg.text_dim);
#ifdef USE_CUDA
    cudaMemcpy(text_emb.data(), text_emb_ptr, text_emb.size() * sizeof(float), cudaMemcpyDeviceToHost);
#else
    std::memcpy(text_emb.data(), text_emb_ptr, text_emb.size() * sizeof(float));
#endif

    // --- Run Transformer (4 Steps) ---
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
        std::unordered_map<uint32_t, const void *> trans_inputs = {
            {in_latent, z.data()},
            {in_txt_emb, text_emb.data()},
            {in_t, &t_curr},
            {in_cos, rope_cos.data()},
            {in_sin, rope_sin.data()}};

        const float *v_ptr = static_cast<const float *>(sess_trans.run(trans_inputs, makeDebugCb(g_trans), doSaturate));

#ifdef USE_CUDA
        std::vector<float> v_host(z.size());
        cudaMemcpy(v_host.data(), v_ptr, v_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
        v_ptr = v_host.data();
#endif

        for (size_t j = 0; j < z.size(); ++j)
        {
            z[j] += v_ptr[j] * dt;
        }
    }

    // --- Run VAE Decoder ---
    std::unordered_map<uint32_t, const void *> vae_inputs = {{in_vae_latent, z.data()}};
    sess_vae.run(vae_inputs, makeDebugCb(g_vae), doSaturate);
}

// === Main Execution Loop ===

int main(int argc, char *argv[])
{
    std::string model = "gemma-3-270m"; // Default model

    // Simple arg parsing
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "flux-klein-4b" || arg == "gemma-3-270m")
        {
            model = arg;
        }
    }

    std::string ref_dir = "reference_tensors";
    if (!fs::exists(ref_dir))
    {
        fs::create_directories(ref_dir);
    }
    const std::string REF_FILE = ref_dir + "/" + model + ".bin";
    std::unordered_map<std::string, RefIndexEntry> refIndex;
    bool loaded = false;

    // Try to load an existing reference tensor file to save compute
    std::ifstream testFile(REF_FILE, std::ios::binary);
    std::unordered_map<uint32_t, int> readCounts;
    if (testFile.is_open())
    {
        while (testFile.peek() != EOF)
        {
            uint32_t logicalId;
            if (!testFile.read(reinterpret_cast<char *>(&logicalId), sizeof(logicalId)))
                break;

            uint32_t nameLen;
            testFile.read(reinterpret_cast<char *>(&nameLen), sizeof(nameLen));
            std::string opName(nameLen, '\0');
            if (nameLen > 0)
                testFile.read(&opName[0], nameLen);

            uint64_t numElements;
            testFile.read(reinterpret_cast<char *>(&numElements), sizeof(numElements));

            RefIndexEntry entry;
            entry.opName = opName;
            entry.numElements = numElements;
            entry.fileOffset = testFile.tellg();

            int iter = readCounts[logicalId]++;
            std::string key = std::to_string(logicalId) + "_" + std::to_string(iter);
            refIndex[key] = entry;

            // Skip ahead past the data so we only index metadata
            testFile.seekg(numElements * sizeof(float), std::ios::cur);
        }
        loaded = true;
    }

    if (loaded)
    {
        std::cout << ">>> Phase 1: Loaded Reference Index from " << REF_FILE << " (" << refIndex.size() << " tensors)\n";
    }
    else
    {
        std::cout << ">>> Phase 1: Running Reference Model (No Saturation) and saving to " << REF_FILE << "\n";
        std::ofstream outFile(REF_FILE, std::ios::binary | std::ios::trunc);
        if (!outFile.is_open())
        {
            std::cerr << "Failed to open " << REF_FILE << " for writing.\n";
            return 1;
        }

        std::unordered_map<uint32_t, int> saveCounts;
        auto saveCallback = [&](uint32_t logicalId, const std::string &opName, const TensorView &view, const std::vector<float> &data)
        {
            uint32_t nameLen = opName.size();
            uint64_t numElems = data.size();

            outFile.write(reinterpret_cast<const char *>(&logicalId), sizeof(logicalId));
            outFile.write(reinterpret_cast<const char *>(&nameLen), sizeof(nameLen));
            if (nameLen > 0)
                outFile.write(opName.c_str(), nameLen);
            outFile.write(reinterpret_cast<const char *>(&numElems), sizeof(numElems));

            RefIndexEntry entry;
            entry.opName = opName;
            entry.numElements = numElems;
            entry.fileOffset = outFile.tellp();

            int iter = saveCounts[logicalId]++;
            std::string key = std::to_string(logicalId) + "_" + std::to_string(iter);
            refIndex[key] = entry;

            outFile.write(reinterpret_cast<const char *>(data.data()), numElems * sizeof(float));
        };

        if (model == "flux-klein-4b")
        {
            runFlux(true, false, saveCallback);
        }
        else
        {
            runGemma(true, false, saveCallback);
        }

        outFile.close();
    }

    std::cout << "\n>>> Phase 2: Running Optimized Model (With Saturation)\n";
    std::cout << "\n=======================================================================================\n";
    std::cout << "Accuracy Comparison (Reference vs Optimized)\n";
    std::cout << "=======================================================================================\n";
    std::cout << std::left
              << std::setw(15) << "Node_Iter"
              << std::setw(25) << "OpType"
              << std::setw(15) << "Min Diff"
              << std::setw(15) << "Max Diff"
              << std::setw(15) << "Avg Diff"
              << "Details\n";
    std::cout << std::string(120, '-') << "\n";

    std::ifstream inFile(REF_FILE, std::ios::binary);
    if (!inFile.is_open())
    {
        std::cerr << "Failed to open " << REF_FILE << " for reading during phase 2.\n";
        return 1;
    }

    int mismatchCount = 0;
    std::unordered_map<uint32_t, int> compareCounts;

    auto compareCallback = [&](uint32_t logicalId, const std::string &opName, const TensorView &view, const std::vector<float> &optData)
    {
        int iter = compareCounts[logicalId]++;
        std::string key = std::to_string(logicalId) + "_" + std::to_string(iter);

        auto it = refIndex.find(key);
        if (it == refIndex.end())
            return; // implicitly evaluated or dropped node

        const auto &entry = it->second;
        if (entry.numElements != optData.size())
        {
            std::cout << std::left << std::setw(15) << key
                      << "SIZE MISMATCH: " << entry.numElements << " vs " << optData.size() << "\n";
            mismatchCount++;
            return;
        }

        if (optData.empty())
            return;

        // Stream from the binary file only when actively performing this callback
        std::vector<float> refData(entry.numElements);
        inFile.seekg(entry.fileOffset, std::ios::beg);
        inFile.read(reinterpret_cast<char *>(refData.data()), entry.numElements * sizeof(float));

        float minDiff = std::numeric_limits<float>::max();
        float maxDiff = 0.0f;
        double sumDiff = 0.0;
        bool hasNan = false;

        for (size_t i = 0; i < refData.size(); ++i)
        {
            float diff = std::abs(refData[i] - optData[i]);
            if (std::isnan(diff))
            {
                hasNan = true;
                break;
            }
            if (diff < minDiff)
                minDiff = diff;
            if (diff > maxDiff)
                maxDiff = diff;
            sumDiff += diff;
        }

        if (hasNan)
        {
            minDiff = std::numeric_limits<float>::quiet_NaN();
            maxDiff = std::numeric_limits<float>::quiet_NaN();
            sumDiff = std::numeric_limits<double>::quiet_NaN();
        }

        float avgDiff = static_cast<float>(sumDiff / refData.size());

        if (maxDiff > 1e-2f || hasNan)
        {
            std::cout << "\033[1;31m"; // Console Red
            mismatchCount++;
        }

        std::cout << std::left
                  << std::setw(15) << key
                  << std::setw(25) << entry.opName.substr(0, 24)
                  << std::setw(15) << minDiff
                  << std::setw(15) << maxDiff
                  << std::setw(15) << avgDiff
                  << "dtype=" << toString(view.dtype)
                  << ", shape=" << toString(view.getShape())
                  << ", strides=" << toString(view.strides)
                  << "\n";

        if (maxDiff > 1e-2f || hasNan)
        {
            std::cout << "\033[0m"; // Console Reset
        }
    };

    if (model == "flux-klein-4b")
    {
        runFlux(false, true, compareCallback);
    }
    else
    {
        runGemma(false, true, compareCallback);
    }

    std::cout << "=======================================================================================\n";
    if (mismatchCount > 0)
    {
        std::cout << "Found " << mismatchCount << " nodes with high deviation (>1e-2) or NaNs.\n";
    }
    else
    {
        std::cout << "All nodes matched perfectly or within acceptable precision limits.\n";
    }
    return 0;
}
