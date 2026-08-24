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

#ifdef TG_USE_CUDA
#include <cuda_runtime.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "core/argparse.hpp"
#include "core/debug.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/logging.hpp"
#include "core/memory.hpp"
#include "core/repo.hpp"
#include "core/session.hpp"
#include "core/settings.hpp"
#include "core/types.hpp"
#include "generated/build_context.gen.hpp"
#include "generated/kernels_all.gen.hpp"
#include "models/run_models.hpp"
#include "stb_image.h"

namespace fs = std::filesystem;

const float *sync_output_to_host(const float *device_ptr, uint64_t num_elements, std::vector<float> &host_buffer)
{
    const float *output_ptr = device_ptr;
#ifdef TG_USE_CUDA
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
void run_autoregressive_llm(const std::string &model_path, const std::string &model_name, const std::string &cache_file,
                            const std::vector<uint32_t> &initial_tokens, uint32_t vocab_size, uint32_t max_seq_len,
                            uint32_t num_tokens_to_generate, const Settings &settings,
                            ModelGraphRoots (*builder)(Graph &, MemoryManager &, const std::string &model_path,
                                                       uint32_t max_seq_len),
                            bool refOnly = false, bool doSaturate = true, const Debug::Callback &debugCb = nullptr,
                            Graph **activeGraphOut = nullptr)
{
    KernelRegistry::get().setReferenceOnly(refOnly);
    std::vector<uint32_t> tokens = initial_tokens;
    MemoryManager mem;
    Graph g;
    if (activeGraphOut)
    {
        *activeGraphOut = &g;
    }

    LOG(INFO) << "Building " << model_name << " Graph...";
    auto roots = builder(g, mem, model_path, max_seq_len);
    LogicalId logits_id = roots.roots[0];
    LogicalId inputIdsId = roots.inputs[0];

    std::string gHash = computeGraphHash(g, roots.roots);
    Repo repo("benchmarks/repo_" + model_name, gHash, true);

    Settings sessionSettings = settings;
    sessionSettings.cache_file = cache_file;
    sessionSettings.do_saturate = doSaturate;

    Session session(g, mem, logits_id, sessionSettings, &repo);

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

    if (settings.only_plan)
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

void run_gemma(const std::string &model_path, const Settings &settings, bool refOnly = false, bool doSaturate = true,
               const Debug::Callback &debugCb = nullptr, Graph **activeGraphOut = nullptr)
{
    run_autoregressive_llm<Gemma3ModelConfig>(model_path, "gemma-3-270m", "dirty_region_caches/gemma-3-270m-cpp.bin",
                                              {2, 9259}, Gemma3ModelConfig().vocab_size, 8, 6, settings,
                                              build_gemma_graph, refOnly, doSaturate, debugCb, activeGraphOut);
}

void run_qwen_35b(const std::string &model_path, const Settings &settings, bool refOnly = false, bool doSaturate = true,
                  const Debug::Callback &debugCb = nullptr, Graph **activeGraphOut = nullptr)
{
    run_autoregressive_llm<Qwen3_6_35B_A3B_Config>(model_path, "qwen-3.6-35b-a3b",
                                                   "dirty_region_caches/qwen-3.6-35b-a3b-cpp.bin", {24227},
                                                   Qwen3_6_35B_A3B_Config().vocab_size, 32, 31, settings,
                                                   build_qwen_graph, refOnly, doSaturate, debugCb, activeGraphOut);
}

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
    parser.add_positional("model", "Name of the target model (gemma-3-270m, qwen-3.6-35b-a3b).", "gemma-3-270m");
    parser.add_positional("model_path", "Model file or directory containing model files.");

    Settings settings;
    settings.add_to_argparser(parser);

    if (!parser.parse(argc, argv))
    {
        return 1;
    }

    settings.load(argc, argv);

    std::string model = parser.get_positional("model");
    std::string model_path = parser.get_positional("model_path");

    if (settings.num_threads > 0)
    {
        set_num_threads(settings.num_threads);
    }

    Debug::ReferenceVerifier verifier;
    if (!verifier.init(settings.write_refs, settings.compare_refs))
    {
        return 1;
    }

    bool refOnly = !settings.write_refs.empty();
    bool doSaturate = settings.write_refs.empty();

    Graph *activeGraphPtr = nullptr;
    auto debugCb = [&](LogicalId logicalId, std::string &kernel_name, const KernelContext &ctx, const void *data) {
        verifier.verify(logicalId, kernel_name, ctx, data, activeGraphPtr);
    };

    if (model == "gemma-3-270m")
        run_gemma(model_path, settings, refOnly, doSaturate, debugCb, &activeGraphPtr);
    else if (model == "qwen-3.6-35b-a3b")
        run_qwen_35b(model_path, settings, refOnly, doSaturate, debugCb, &activeGraphPtr);
    else
        std::cout << "Model not implemented yet: " << model << std::endl;

    verifier.printSummary();
    return 0;
}