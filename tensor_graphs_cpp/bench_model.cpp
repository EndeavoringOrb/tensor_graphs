// tensor_graphs_cpp/bench_model.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <float.h>
#endif

#ifdef TG_USE_CUDA
#include <cuda_runtime.h>
#endif

#include "core/argparse.hpp"
#include "core/cost_model.hpp"
#include "core/debug.hpp"
#include "core/graph.hpp"
#include "core/hardware.hpp"
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

namespace fs = std::filesystem;

static size_t utf8_len(const std::string &s)
{
    size_t len = 0;
    for (size_t i = 0; i < s.size(); ++i)
    {
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80)
            len++;
    }
    return len;
}

static std::string pad_left_utf8(const std::string &s, size_t target_width)
{
    size_t visual_len = utf8_len(s);
    if (visual_len < target_width)
        return std::string(target_width - visual_len, ' ') + s;
    return s;
}

static std::string pad_right_utf8(const std::string &s, size_t target_width)
{
    size_t visual_len = utf8_len(s);
    if (visual_len < target_width)
        return s + std::string(target_width - visual_len, ' ');
    return s;
}

static std::string format_stats(double mean, double stdev)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << mean << " +- " << stdev;
    return ss.str();
}

static std::string format_size(uint64_t bytes)
{
    double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << mib << " MiB";
    return ss.str();
}

static std::string format_params(double params)
{
    std::ostringstream ss;
    if (params >= 1e9)
    {
        ss << std::fixed << std::setprecision(2) << (params / 1e9) << " B";
    }
    else
    {
        ss << std::fixed << std::setprecision(2) << (params / 1e6) << " M";
    }
    return ss.str();
}

static uint64_t get_model_file_size(const std::string &path)
{
    if (fs::is_regular_file(path))
    {
        return fs::file_size(path);
    }
    if (fs::is_directory(path))
    {
        uint64_t total = 0;
        for (const auto &entry : fs::directory_iterator(path))
        {
            if (entry.is_regular_file() && (entry.path().extension() == ".safetensors" ||
                                            entry.path().extension() == ".bin" ||
                                            entry.path().extension() == ".gguf"))
            {
                total += fs::file_size(entry.path());
            }
        }
        return total;
    }
    return 0;
}

static double get_model_params(const std::string &path, const std::string &model_name)
{
    std::vector<std::string> st_files;
    if (fs::is_regular_file(path) && path.rfind(".safetensors") != std::string::npos)
    {
        st_files.push_back(path);
    }
    else if (fs::is_directory(path))
    {
        for (const auto &entry : fs::directory_iterator(path))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors")
            {
                st_files.push_back(entry.path().string());
            }
        }
    }

    uint64_t total_elements = 0;
    for (const auto &sf : st_files)
    {
        std::ifstream file(sf, std::ios::binary);
        if (!file.is_open())
            continue;
        uint64_t headerSize = 0;
        if (!file.read(reinterpret_cast<char *>(&headerSize), sizeof(headerSize)))
            continue;
        std::string jsonHeader(headerSize, '\0');
        if (!file.read(&jsonHeader[0], headerSize))
            continue;
        try
        {
            auto root = json::parse(jsonHeader);
            for (const auto &[key, val] : root.items())
            {
                if (key == "__metadata__" || !val.is_object() || !val.contains("shape"))
                    continue;
                uint64_t num_elems = 1;
                for (const auto &dim : val["shape"])
                {
                    num_elems *= dim.get<int64_t>();
                }
                total_elements += num_elems;
            }
        }
        catch (...)
        {
        }
    }

    if (total_elements > 0)
    {
        return static_cast<double>(total_elements);
    }

    if (model_name.find("gemma-3-270m") != std::string::npos || model_name.find("gemma") != std::string::npos)
        return 268.10e6;
    if (model_name.find("qwen-3.6-35b") != std::string::npos || model_name.find("qwen") != std::string::npos)
        return 35.80e9;
    if (model_name.find("deepseek") != std::string::npos)
        return 27.0e9;
    return 0.0;
}

static std::string get_backend_name()
{
#ifdef TG_USE_CUDA
    if (HardwareCaps::get().has_cuda)
        return "CUDA";
#endif
#ifdef TG_USE_OPENCL
    if (HardwareCaps::get().has_opencl)
        return "OpenCL";
#endif
    return "CPU";
}

static std::string get_default_model_path(const std::string &model_name)
{
    std::vector<std::string> candidates;
    if (model_name.find("gemma") != std::string::npos)
    {
        candidates = {
            "models/google/gemma-3-270m",
            "models/gemma-3-270m",
            "models/google/gemma-3-270m-it",
            "models/gemma-3-270m-it",
            "models/google/gemma-3-270m-it-bf16",
            "models/gemma-3-270m-it-bf16",
            "models/gemma-3-270m/model.safetensors"
        };
    }
    else if (model_name.find("qwen") != std::string::npos)
    {
        candidates = {
            "models/Qwen/Qwen3.6-35B-A3B",
            "models/qwen-3.6-35b-a3b",
            "models/Qwen3.6-35B-A3B"
        };
    }
    else if (model_name.find("deepseek") != std::string::npos)
    {
        candidates = {
            "models/deepseek-ai/DeepSeek-V4",
            "models/deepseek-v4"
        };
    }

    for (const auto &c : candidates)
    {
        if (fs::exists(c))
            return c;
    }
    return "models/" + model_name;
}

static void sync_device_all()
{
#ifdef TG_USE_CUDA
    if (HardwareCaps::get().has_cuda)
    {
        cudaDeviceSynchronize();
    }
#endif
#ifdef TG_USE_OPENCL
    if (HardwareCaps::get().has_opencl && OpenCLState::get().initialized)
    {
        clFinish(OpenCLState::get().queue);
    }
#endif
}

int main(int argc, char *argv[])
{
#if defined(_WIN32)
    _controlfp_s(nullptr, 0, 0);
    _controlfp_s(nullptr, _EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW, _MCW_EM);
#endif

    System::get();
    HardwareCaps::get();

    ArgParser parser("bench_model", "Benchmark LLM prompt processing (pp) and text generation (tg).");
    parser.add_option({"--model", "-m"}, "Model name (gemma-3-270m, qwen-3.6-35b-a3b, deepseek-v4).", "gemma-3-270m");
    parser.add_option({"--model-path", "-p"}, "Path to model weights directory or file.", "");
    parser.add_option({"--pp"}, "Prompt processing tokens (default: 512).", "512");
    parser.add_option({"--tg"}, "Text generation token position (default: 128).", "128");
    parser.add_option({"--iters", "-i", "-n"}, "Benchmark iterations (default: 5).", "5");
    parser.add_option({"--warmup", "-w"}, "Warmup iterations (default: 1).", "1");
    parser.add_positional("pos_model", "Model name or path (optional).", "");
    parser.add_positional("pos_model_path", "Model path (optional).", "");

    Settings settings;
    settings.add_to_argparser(parser);

    std::vector<std::string> remaining_args;
    if (!parser.parse(argc, argv, &remaining_args))
    {
        return 1;
    }

    settings.load(remaining_args);

    std::string model_name = parser.get_option("--model");
    std::string model_path = parser.get_option("--model-path");
    std::string pos1 = parser.get_positional("pos_model");
    std::string pos2 = parser.get_positional("pos_model_path");

    if (!pos1.empty())
    {
        if (fs::exists(pos1) && model_path.empty())
        {
            model_path = pos1;
        }
        else if (model_name == "gemma-3-270m")
        {
            model_name = pos1;
        }
    }
    if (!pos2.empty() && model_path.empty())
    {
        model_path = pos2;
    }

    if (model_path.empty())
    {
        if (fs::exists(model_name))
        {
            model_path = model_name;
        }
        else
        {
            model_path = get_default_model_path(model_name);
        }
    }

    uint32_t pp = 512;
    uint32_t tg = 128;
    try
    {
        pp = std::max(1u, static_cast<uint32_t>(std::stoul(parser.get_option("--pp"))));
        tg = std::max(1u, static_cast<uint32_t>(std::stoul(parser.get_option("--tg"))));
    }
    catch (...)
    {
        std::cerr << "Invalid --pp or --tg values provided.\n";
        return 1;
    }

    int iters = std::max(1, std::stoi(parser.get_option("--iters")));
    int warmup = std::max(0, std::stoi(parser.get_option("--warmup")));

    if (settings.num_threads > 0)
    {
        set_num_threads(settings.num_threads);
    }

    // Allocate max sequence length to cover both pp and the tg target token
    uint32_t max_seq_len = std::max(pp, tg + 1);

    MemoryManager mem;
    Graph g;
    ModelGraphRoots roots;
    uint32_t vocab_size = 0;
    std::string model_desc = model_name;

    std::string norm_name = model_name;
    std::transform(norm_name.begin(), norm_name.end(), norm_name.begin(), [](unsigned char c) {
        return std::tolower(c);
    });

    if (norm_name.find("gemma") != std::string::npos)
    {
        model_name = "gemma-3-270m";
        model_desc = "gemma3 270M BF16";
        Gemma3ModelConfig cfg;
        vocab_size = cfg.vocab_size;
        roots = build_gemma_graph(g, mem, model_path, max_seq_len);
    }
    else if (norm_name.find("qwen") != std::string::npos)
    {
        model_name = "qwen-3.6-35b-a3b";
        model_desc = "qwen3.6 35B A3B";
        Qwen3_6_35B_A3B_Config cfg;
        vocab_size = cfg.vocab_size;
        roots = build_qwen_graph(g, mem, model_path, max_seq_len);
    }
    else if (norm_name.find("deepseek") != std::string::npos)
    {
        model_name = "deepseek-v4";
        model_desc = "deepseek-v4-flash";
        DeepSeekV4FlashConfig cfg;
        vocab_size = cfg.vocab_size;
        LogicalId inputIdsId = g.input({1, max_seq_len}, DType::INT32);
        DeepSeekV4FlashModel model(cfg, max_seq_len, g, mem, model_path);
        roots = {{model.build_graph(inputIdsId)}, {inputIdsId}};
    }
    else
    {
        Error::throw_err("Unsupported model for benchmark: " + model_name);
    }

    LogicalId logits_id = roots.roots[0];
    LogicalId input_ids_id = roots.inputs[0];

    std::string gHash = computeGraphHash(g, roots.roots);
    Repo repo("benchmarks/repo_" + model_name, gHash, true);

    std::string cache_file = settings.cache_file;
    if (cache_file.empty())
    {
        std::filesystem::create_directories("dirty_region_caches");
        cache_file = "dirty_region_caches/bench_" + model_name + "-pp" + std::to_string(pp) + "-tg" +
                     std::to_string(tg) + ".bin";
    }
    Settings sessionSettings = settings;
    sessionSettings.cache_file = cache_file;

    Session session(g, mem, logits_id, sessionSettings, &repo);

    // =========================================================================
    // 3-Bucket Compilation Setup
    // =========================================================================

    // (1) Fully dirty bucket: loads storage weights, sequence length == max_seq_len
    Bucket b1_full;
    for (const auto &pair : g.nodes)
    {
        if (pair.second.opType == OpType::INPUT)
        {
            b1_full.inputDirtyRegions[pair.first] = {makeFull(pair.second.getShape())};
        }
    }
    b1_full.outputNeededRegion = {makeFull(g.getNode(logits_id).getShape())};

    // (2) pp bucket: input tokens dirty 0..pp, clean static weights
    Bucket b2_pp;
    Region pp_in_reg;
    pp_in_reg.region = {{0, 1}, {0, pp}};
    b2_pp.inputDirtyRegions[input_ids_id] = {pp_in_reg};

    Region pp_out_reg;
    pp_out_reg.region = {{0, 1}, {0, pp}, {0, vocab_size}};
    b2_pp.outputNeededRegion = {pp_out_reg};

    // (3) tg bucket: input token tg (128) dirty, output token tg+1 (129) predicted
    Bucket b3_tg;
    Region tg_in_reg;
    tg_in_reg.region = {{0, 1}, {tg, tg + 1}};
    b3_tg.inputDirtyRegions[input_ids_id] = {tg_in_reg};

    Region tg_out_reg;
    tg_out_reg.region = {{0, 1}, {tg, tg + 1}, {0, vocab_size}};
    b3_tg.outputNeededRegion = {tg_out_reg};

    session.addBucket(b1_full.inputDirtyRegions, b1_full.outputNeededRegion);
    session.addBucket(b2_pp.inputDirtyRegions, b2_pp.outputNeededRegion);
    session.addBucket(b3_tg.inputDirtyRegions, b3_tg.outputNeededRegion);

    if (settings.only_plan)
    {
        session.plan(true);
        std::cout << "[bench_model] Compilation planned successfully.\n";
        return 0;
    }

    session.compile(true);

    // Prepare token inputs
    std::vector<int32_t> input_tokens(max_seq_len, 2);
    session.writeInput(input_ids_id, input_tokens.data(), input_tokens.size() * sizeof(int32_t));

    // Initial run of Bucket 1 to ensure all weights are resident in memory/VRAM
    session.run(b1_full);
    sync_device_all();

    // =========================================================================
    // Benchmark 1: Prompt Processing (pp<N>)
    // =========================================================================
    for (int w = 0; w < warmup; ++w)
    {
        session.writeInput(input_ids_id, input_tokens.data(), input_tokens.size() * sizeof(int32_t));
        session.run(b2_pp);
        sync_device_all();
    }

    std::vector<double> pp_tps;
    pp_tps.reserve(iters);
    for (int it = 0; it < iters; ++it)
    {
        session.writeInput(input_ids_id, input_tokens.data(), input_tokens.size() * sizeof(int32_t));
        sync_device_all();

        auto start = std::chrono::high_resolution_clock::now();
        session.run(b2_pp);
        sync_device_all();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        if (elapsed_sec > 0.0)
        {
            pp_tps.push_back(static_cast<double>(pp) / elapsed_sec);
        }
    }

    // =========================================================================
    // Benchmark 2: Text Generation (tg<N>)
    // =========================================================================
    for (int w = 0; w < warmup; ++w)
    {
        session.writeInput(input_ids_id, input_tokens.data(), input_tokens.size() * sizeof(int32_t));
        session.run(b3_tg);
        sync_device_all();
    }

    std::vector<double> tg_tps;
    tg_tps.reserve(iters);
    for (int it = 0; it < iters; ++it)
    {
        session.writeInput(input_ids_id, input_tokens.data(), input_tokens.size() * sizeof(int32_t));
        sync_device_all();

        auto start = std::chrono::high_resolution_clock::now();
        session.run(b3_tg);
        sync_device_all();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        if (elapsed_sec > 0.0)
        {
            tg_tps.push_back(1.0 / elapsed_sec);
        }
    }

    // =========================================================================
    // Format and Output Results Table
    // =========================================================================
    auto compute_stats = [](const std::vector<double> &samples) -> std::pair<double, double> {
        if (samples.empty())
            return {0.0, 0.0};
        double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        double mean = sum / samples.size();
        double var = 0.0;
        for (double v : samples)
            var += (v - mean) * (v - mean);
        double stdev = (samples.size() > 1) ? std::sqrt(var / (samples.size() - 1)) : 0.0;
        return {mean, stdev};
    };

    auto pp_stats = compute_stats(pp_tps);
    auto tg_stats = compute_stats(tg_tps);

    uint64_t model_bytes = get_model_file_size(model_path);
    if (model_bytes == 0)
    {
        if (model_name == "gemma-3-270m")
            model_bytes = static_cast<uint64_t>(511.46 * 1024.0 * 1024.0);
    }
    double model_params = get_model_params(model_path, model_name);

    std::string size_str = format_size(model_bytes);
    std::string params_str = format_params(model_params);
    std::string backend_str = get_backend_name();
    std::string ngl_str = (backend_str == "CPU") ? "0" : "-1";

    std::string pp_test_name = "pp" + std::to_string(pp);
    std::string tg_test_name = "tg" + std::to_string(tg);

    std::string pp_tps_str = format_stats(pp_stats.first, pp_stats.second);
    std::string tg_tps_str = format_stats(tg_stats.first, tg_stats.second);

    std::cout << "| " << pad_right_utf8("model", 30)
              << " | " << pad_left_utf8("size", 10)
              << " | " << pad_left_utf8("params", 10)
              << " | " << pad_right_utf8("backend", 10)
              << " | " << pad_left_utf8("ngl", 3)
              << " | " << pad_left_utf8("test", 15)
              << " | " << pad_left_utf8("t/s", 20)
              << " |\n";

    std::cout << "| " << std::string(30, '-')
              << " | " << std::string(9, '-') << ":"
              << " | " << std::string(9, '-') << ":"
              << " | " << std::string(10, '-')
              << " | " << std::string(2, '-') << ":"
              << " | " << std::string(14, '-') << ":"
              << " | " << std::string(19, '-') << ":"
              << " |\n";

    std::cout << "| " << pad_right_utf8(model_desc, 30)
              << " | " << pad_left_utf8(size_str, 10)
              << " | " << pad_left_utf8(params_str, 10)
              << " | " << pad_right_utf8(backend_str, 10)
              << " | " << pad_left_utf8(ngl_str, 3)
              << " | " << pad_left_utf8(pp_test_name, 15)
              << " | " << pad_left_utf8(pp_tps_str, 20)
              << " |\n";

    std::cout << "| " << pad_right_utf8(model_desc, 30)
              << " | " << pad_left_utf8(size_str, 10)
              << " | " << pad_left_utf8(params_str, 10)
              << " | " << pad_right_utf8(backend_str, 10)
              << " | " << pad_left_utf8(ngl_str, 3)
              << " | " << pad_left_utf8(tg_test_name, 15)
              << " | " << pad_left_utf8(tg_tps_str, 20)
              << " |\n";

    return 0;
}