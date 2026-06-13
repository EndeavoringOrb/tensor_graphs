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
#include "core/repo.hpp"

#include "models/run_models.hpp"
#include "generated/kernels_all.gen.hpp"
#include "generated/build_context.gen.hpp"

namespace fs = std::filesystem;

std::string getNextAvailableBinPath(const std::string &folderPath = "sessions")
{
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
    return flattenOutput(data, view.getShape(), view.strides, view.dtype);
}

struct RefIndexEntry
{
    uint64_t fileOffset;
    uint64_t numElements;
    std::string opName;
};

void runGemma(bool refOnly, bool doSaturate, std::function<void(uint32_t logicalId, const std::string &opName, const TensorView &view, const std::vector<float> &data)> callback)
{
    KernelRegistry::get().setReferenceOnly(refOnly);

    std::unordered_map<Backend, uint64_t> bufferSizes = {{Backend::CPU, 16ULL * 1024 * 1024 * 1024}};
#if USE_CUDA
    bufferSizes[Backend::CUDA] = 16ULL * 1024 * 1024 * 1024;
#endif
    MemoryManager mem(bufferSizes);
    Graph g;

    auto roots = build_gemma_graph(g, mem);
    uint32_t inputIdsId = roots.inputs[0];
    uint32_t rootId = roots.roots[0];

    std::string gHash = computeGraphHash(g, roots.roots);
    Repo repo("benchmarks/repo_gemma-3-270m", gHash, true);

    std::vector<int32_t> input_ids = {2, 9259, 0, 0, 0, 0, 0, 0};

    uint64_t sizeBytes = 8 * getDTypeSize(DType::INT32);
    mem.allocate(Backend::CPU, inputIdsId, sizeBytes, StorageType::PERSISTENT);

    std::unordered_map<uint32_t, const void *> inputs = {{inputIdsId, input_ids.data()}};

    Session sess(g, mem, rootId, getNextAvailableBinPath("test_model_cache_gemma"), 0, &repo);

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
    sess.memManager.write(Backend::CPU, inputIdsId, input_ids.data(), sizeBytes);
    sess.run({}, debugCb, doSaturate);
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
    Graph g;

    auto roots = build_flux_graph(g, mem);

    std::string gHash = computeGraphHash(g, roots.roots);
    Repo repo("benchmarks/repo_flux-klein-4b", gHash, true);

    uint32_t txt_seq = cfg.text_max_seq;
    uint32_t latent_h = 32, latent_w = 32;
    uint32_t img_seq = latent_h * latent_w, total_seq = txt_seq + img_seq;

    Session sess_text(g, mem, roots.roots[0], getNextAvailableBinPath("test_model_cache_flux_text"), 0, &repo);
    Session sess_trans(g, mem, roots.roots[1], getNextAvailableBinPath("test_model_cache_flux_trans"), 0, &repo);
    Session sess_vae(g, mem, roots.roots[2], getNextAvailableBinPath("test_model_cache_flux_vae"), 0, &repo);

    sess_text.compile(doSaturate);
    sess_trans.compile(doSaturate);
    sess_vae.compile(doSaturate);

    auto makeDebugCb = [&](Graph &graph)
    {
        return [&](uint32_t logicalId, const TensorView &view, const void *data)
        {
            if (!callback)
                return;
            std::vector<float> extracted = extractTensorData(view, data);
            std::string opName = "UNKNOWN";
            if (graph.hasNode(logicalId))
            {
                auto node = graph.getNode(logicalId);
                opName = toString(node.opType);
                if (node.opType == OpType::FUSED)
                {
                    opName = "FUSED_" + node.opName;
                }
            }
            callback(logicalId, opName, view, extracted);
        };
    };

    std::vector<int32_t> input_ids = load_tokens_from_file("toks.txt", txt_seq);
    sess_text.memManager.write(Backend::CPU, roots.inputs[0], input_ids.data(), input_ids.size() * sizeof(int32_t));
    const float *text_emb_ptr = static_cast<const float *>(sess_text.run({}, makeDebugCb(g), doSaturate));

    std::vector<float> text_emb(1 * txt_seq * cfg.text_dim);
#ifdef USE_CUDA
    cudaMemcpy(text_emb.data(), text_emb_ptr, text_emb.size() * sizeof(float), cudaMemcpyDeviceToHost);
#else
    std::memcpy(text_emb.data(), text_emb_ptr, text_emb.size() * sizeof(float));
#endif

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

        sess_trans.memManager.write(Backend::CPU, roots.inputs[1], z.data(), z.size() * sizeof(float));
        sess_trans.memManager.write(Backend::CPU, roots.inputs[2], text_emb.data(), text_emb.size() * sizeof(float));
        sess_trans.memManager.write(Backend::CPU, roots.inputs[3], &t_curr, sizeof(float));
        sess_trans.memManager.write(Backend::CPU, roots.inputs[4], rope_cos.data(), rope_cos.size() * sizeof(float));
        sess_trans.memManager.write(Backend::CPU, roots.inputs[5], rope_sin.data(), rope_sin.size() * sizeof(float));

        const float *v_ptr = static_cast<const float *>(sess_trans.run({}, makeDebugCb(g), doSaturate));

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

    sess_vae.memManager.write(Backend::CPU, roots.inputs[6], z.data(), z.size() * sizeof(float));
    sess_vae.run({}, makeDebugCb(g), doSaturate);
}

int main(int argc, char *argv[])
{
    std::string model = "gemma-3-270m";
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
            return;

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
            std::cout << "\033[1;31m";
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
            std::cout << "\033[0m";
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