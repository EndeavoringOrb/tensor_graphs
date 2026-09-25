#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <json.hpp>

#include "core/argparse.hpp"
#include "core/hardware.hpp"
#include "core/logging.hpp"
#include "core/types.hpp"

struct Settings
{
    std::unordered_map<MemSpace, uint64_t> mem_caps;

    // File paths
    std::string cache_file = "";
    std::string records_path = "benchmarks/records.bin";
    std::string settings_json_path = "settings.json";
    std::string repo_path = "";

    // Engine, Session, & Planner parameters
    bool disable_caching = false;
    bool log_cost_calls = true;
    float min_compile_seconds = 0.0f;
    uint32_t num_threads = 0;
    bool do_saturate = true;
    bool reference_only = false;
    bool only_plan = false;
    bool compile_decode_buckets = false;
    bool fold_weights = false;
    bool use_ortools = false;
    bool use_ortools_full = false;
    bool use_ortools_lns = false;
    // Restrict native planning to operations that can execute on the CPU.
    // This is used to produce a portable feasible witness for OR-Tools.
    bool cpu_only = false;
    double max_time_seconds = 0.0;
    // Optional raw weights in bucket insertion order. Session normalizes these
    // when scoring shared cache selections.
    std::vector<float> bucket_weights;
    std::string write_refs = "";
    std::string compare_refs = "";
    std::string model_name = "gemma-3-270m";
    std::string model_path = "";

    Settings() : mem_caps(System::get().getBufferSizes())
    {
    }

    static bool parseMemSpaceKey(const std::string &key, MemSpace &out)
    {
        struct TypeEntry
        {
            const char *name;
            HandleType type;
        };
        static const TypeEntry kTypes[] = {
            {"storage", HandleType::STORAGE},
            {"cpp", HandleType::CPP},
            {"opencl", HandleType::OPENCL},
            {"cuda", HandleType::CUDA},
        };
        for (const auto &entry : kTypes)
        {
            std::string name(entry.name);
            if (key.rfind(name, 0) != 0)
                continue;
            std::string rest = key.substr(name.size());
            if (rest.empty())
            {
                out = MemSpace{0, entry.type};
                return true;
            }
            if (rest.find_first_not_of("0123456789") != std::string::npos)
                return false;
            out = MemSpace{static_cast<uint32_t>(std::stoul(rest)), entry.type};
            return true;
        }
        return false;
    }

    bool load_from_json(const std::string &path = "")
    {
        std::string actual_path = path.empty() ? settings_json_path : path;
        if (actual_path.empty() || !std::filesystem::exists(actual_path))
        {
            return false;
        }

        std::ifstream file(actual_path);
        if (!file.is_open())
        {
            return false;
        }

        try
        {
            auto root = nlohmann::json::parse(file);

            if (root.contains("disable_caching") && root["disable_caching"].is_boolean())
                disable_caching = root["disable_caching"].get<bool>();

            if (root.contains("log_cost_calls") && root["log_cost_calls"].is_boolean())
                log_cost_calls = root["log_cost_calls"].get<bool>();

            if (root.contains("min_compile_seconds") && root["min_compile_seconds"].is_number())
                min_compile_seconds = root["min_compile_seconds"].get<float>();
            else if (root.contains("min_compile_time") && root["min_compile_time"].is_number())
                min_compile_seconds = root["min_compile_time"].get<float>();

            if (root.contains("records_path") && root["records_path"].is_string())
                records_path = root["records_path"].get<std::string>();

            if (root.contains("cache_file") && root["cache_file"].is_string())
                cache_file = root["cache_file"].get<std::string>();

            if (root.contains("bucket_weights") && root["bucket_weights"].is_array())
                bucket_weights = root["bucket_weights"].get<std::vector<float>>();

            if (root.contains("num_threads") && root["num_threads"].is_number_integer())
                num_threads = root["num_threads"].get<uint32_t>();
            else if (root.contains("threads") && root["threads"].is_number_integer())
                num_threads = root["threads"].get<uint32_t>();

            if (root.contains("mem_caps") && root["mem_caps"].is_object())
            {
                for (const auto &[key, val] : root["mem_caps"].items())
                {
                    if (!val.is_number())
                    {
                        std::cerr << "[Settings] Warning: mem_caps entry '" << key << "' is not a number, skipping."
                                  << std::endl;
                        continue;
                    }
                    MemSpace ms;
                    if (!parseMemSpaceKey(key, ms))
                    {
                        std::cerr << "[Settings] Warning: Unknown mem_caps key '" << key
                                  << "' (expected <type> or <type><idx> with type in storage/cpp/opencl/cuda), "
                                     "skipping."
                                  << std::endl;
                        continue;
                    }
                    mem_caps[ms] = static_cast<uint64_t>(val.get<double>());
                }
            }

            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Settings] Warning: Failed to parse JSON from " << actual_path << ": " << e.what()
                      << std::endl;
            return false;
        }
    }

    void add_to_argparser(ArgParser &parser) const
    {
        parser.add_option({"--settings"}, "Path to settings.json configuration file.", settings_json_path);
        parser.add_flag({"--disable-caching"}, "Disable dirty region session caching.");
        parser.add_flag({"--only-plan"}, "Only plan the execution and generate cache.");
        parser.add_flag({"--fold-weights"}, "Enable folding of weights (InputDataType::STORAGE).");
        parser.add_flag({"--use-ortools"}, "Use Google OR-Tools CP-SAT model for optimization.");
        parser.add_flag({"--use-ortools-full"}, "Jointly solve cache, extraction, dispatch, bufferization and allocation with OR-Tools.");
        parser.add_flag({"--use-ortools-lns"}, "Jointly solve cache, extraction, dispatch, bufferization and allocation with OR-Tools LNS.");
        parser.add_option({"--max-time-seconds"}, "Maximum time limit in seconds for OR-Tools optimization.", "0.0");
        parser.add_option({"--repo-path"}, "Path to the tensor repository (benchmarks/repo_<name>).", "");
        parser.add_option({"--records"}, "Path to kernel benchmark records file.", records_path);
        parser.add_option({"--write-refs"}, "Write reference/clean tensors to file.", "");
        parser.add_option({"--compare-refs"}, "Compare and validate outputs against reference file.", "");
        parser.add_option({"--min-compile-time"}, "Minimum required compile time per bucket in seconds.", "0.0");
        parser.add_option({"--threads"}, "Number of C++ threads (0 = auto-detect).", "0");
        parser.add_option({"--bucket-weights"}, "Comma-separated raw bucket weights in bucket insertion order.", "");
        parser.add_option({"--mem-cap"},
                          "Comma-separated list of <mem_space>=<bytes> overrides for planner memory caps, e.g. "
                          "'cpp=8388608,opencl1=1073741824'. Mem space is <type> or <type><idx> with type in "
                          "storage/cpp/opencl/cuda.",
                          "");
    }

    void apply_cli_args(const ArgParser &parser)
    {
        std::string cli_settings = parser.get_option("--settings");
        if (!cli_settings.empty() && cli_settings != settings_json_path)
        {
            settings_json_path = cli_settings;
            load_from_json(settings_json_path);
        }

        if (parser.get_flag("--disable-caching"))
            disable_caching = true;

        if (parser.get_flag("--only-plan"))
            only_plan = true;

        if (parser.get_flag("--fold-weights"))
            fold_weights = true;

        if (parser.get_flag("--use-ortools"))
            use_ortools = true;
        if (parser.get_flag("--use-ortools-full"))
            use_ortools_full = true;
        if (parser.get_flag("--use-ortools-lns"))
            use_ortools_lns = true;

        std::string cli_max_time = parser.get_option("--max-time-seconds");
        if (!cli_max_time.empty())
        {
            try
            {
                max_time_seconds = std::stod(cli_max_time);
            }
            catch (...)
            {
            }
        }

        std::string cli_repo_path = parser.get_option("--repo-path");
        if (!cli_repo_path.empty())
            repo_path = cli_repo_path;

        std::string min_comp = parser.get_option("--min-compile-time");
        if (!min_comp.empty())
        {
            try
            {
                min_compile_seconds = std::stof(min_comp);
            }
            catch (...)
            {
            }
        }

        std::string cli_records = parser.get_option("--records");
        if (!cli_records.empty())
            records_path = cli_records;

        std::string cli_write_refs = parser.get_option("--write-refs");
        if (!cli_write_refs.empty())
            write_refs = cli_write_refs;

        std::string cli_compare_refs = parser.get_option("--compare-refs");
        if (!cli_compare_refs.empty())
            compare_refs = cli_compare_refs;

        std::string cli_threads = parser.get_option("--threads");
        if (!cli_threads.empty())
        {
            try
            {
                num_threads = std::stoi(cli_threads);
            }
            catch (...)
            {
            }
        }

        std::string cli_bucket_weights = parser.get_option("--bucket-weights");
        if (!cli_bucket_weights.empty())
        {
            std::vector<float> parsed_weights;
            std::stringstream ss(cli_bucket_weights);
            std::string item;
            try
            {
                while (std::getline(ss, item, ','))
                    parsed_weights.push_back(std::stof(item));
                bucket_weights = std::move(parsed_weights);
            }
            catch (...)
            {
                Error::throw_err("[Settings] Invalid --bucket-weights value: " + cli_bucket_weights);
            }
        }

        std::string cli_mem_caps = parser.get_option("--mem-cap");
        if (!cli_mem_caps.empty())
        {
            std::stringstream ss(cli_mem_caps);
            std::string item;
            while (std::getline(ss, item, ','))
            {
                if (item.empty())
                    continue;
                auto eq = item.find('=');
                if (eq == std::string::npos)
                {
                    std::cerr << "[Settings] Warning: Invalid --mem-cap entry '" << item
                              << "' (expected <mem_space>=<bytes>), skipping." << std::endl;
                    continue;
                }
                MemSpace ms;
                if (!parseMemSpaceKey(item.substr(0, eq), ms))
                {
                    std::cerr << "[Settings] Warning: Unknown mem space '" << item.substr(0, eq)
                              << "' in --mem-cap (expected <type> or <type><idx> with type in "
                                 "storage/cpp/opencl/cuda), skipping."
                              << std::endl;
                    continue;
                }
                try
                {
                    mem_caps[ms] = static_cast<uint64_t>(std::stoull(item.substr(eq + 1)));
                }
                catch (...)
                {
                    std::cerr << "[Settings] Warning: Invalid byte count in --mem-cap entry '" << item << "', skipping."
                              << std::endl;
                }
            }
        }
    }

    void load(const std::vector<std::string> &args = {}, const std::string &custom_json_path = "",
              const std::string &custom_bin_path = "")
    {
        (void)custom_bin_path;
        load_from_json(custom_json_path);

        if (!args.empty())
        {
            ArgParser parser("SettingsLoader");
            parser.set_verbose_logging(false);
            add_to_argparser(parser);
            if (!parser.parse(args))
            {
                Error::throw_err("Failed to parse settings command-line arguments.");
            }
            apply_cli_args(parser);
        }
    }

    void load(int argc, char *argv[], const std::string &custom_json_path = "", const std::string &custom_bin_path = "")
    {
        std::vector<std::string> args;
        for (int i = 1; i < argc; ++i)
        {
            args.push_back(argv[i]);
        }
        load(args, custom_json_path, custom_bin_path);
    }

    static Settings get_default()
    {
        Settings s;
        s.load();
        return s;
    }

    static Settings &get_global()
    {
        static Settings global_instance = get_default();
        return global_instance;
    }
};
