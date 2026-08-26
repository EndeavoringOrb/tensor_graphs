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

#include "core/argparse.hpp"
#include "core/hardware.hpp"
#include "core/logging.hpp"
#include "core/serialization.hpp"
#include "core/types.hpp"

struct RuleBenchmarkRecord
{
    std::string category;  // e.g. "dispatch", "cache", "bufferize", "malloc", "extract", "enode"
    std::string rule_name; // e.g. "SingleEngineDispatchDomination"
    bool was_faster = false;
    double baseline_ms = 0.0;
    double test_ms = 0.0;
    double speedup = 1.0;
};

inline void tg_serialize(BinaryWriter &bw, const RuleBenchmarkRecord &val)
{
    bw.write(val.category);
    bw.write(val.rule_name);
    bw.write(val.was_faster);
    bw.write(val.baseline_ms);
    bw.write(val.test_ms);
    bw.write(val.speedup);
}

inline void tg_deserialize(BinaryReader &br, RuleBenchmarkRecord &val)
{
    br.read(val.category);
    br.read(val.rule_name);
    br.read(val.was_faster);
    br.read(val.baseline_ms);
    br.read(val.test_ms);
    br.read(val.speedup);
}

struct Settings
{
    std::unordered_map<MemSpace, uint64_t> mem_caps;

    // Category -> (RuleName -> IsEnabled)
    std::unordered_map<std::string, std::unordered_map<std::string, bool>> rules;
    std::unordered_map<std::string, bool> category_defined;
    std::unordered_map<std::string, std::unordered_map<std::string, RuleBenchmarkRecord>> benchmark_records;

    // File paths
    std::string cache_file = "";
    std::string records_path = "benchmarks/records.bin";
    std::string rule_benchmarks_path = "benchmarks/rules.bin";
    std::string settings_json_path = "settings.json";

    // Engine, Session, & Planner parameters
    bool disable_caching = false;
    bool log_cost_calls = true;
    float min_compile_seconds = 0.0f;
    uint32_t num_threads = 0;
    bool do_saturate = true;
    bool reference_only = false;
    bool only_plan = false;
    bool compile_decode_buckets = false;
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

    static bool save_rule_benchmarks(const std::string &path, const std::vector<RuleBenchmarkRecord> &records)
    {
        std::filesystem::path p(path);
        if (p.has_parent_path())
        {
            std::filesystem::create_directories(p.parent_path());
        }

        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file.is_open())
        {
            std::cerr << "[Settings] Error: Failed to open " << path << " for writing rule benchmarks." << std::endl;
            return false;
        }

        try
        {
            BinaryWriter bw(file);
            bw.write(records);
            file.flush();
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Settings] Error writing rule benchmarks to " << path << ": " << e.what() << std::endl;
            return false;
        }
    }

    static std::vector<RuleBenchmarkRecord> load_rule_benchmarks_file(const std::string &path)
    {
        std::vector<RuleBenchmarkRecord> records;
        if (path.empty() || !std::filesystem::exists(path))
            return records;

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
            return records;

        try
        {
            BinaryReader br(file);
            br.read(records);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Settings] Warning: Failed to read rule benchmarks from " << path << ": " << e.what()
                      << std::endl;
        }
        return records;
    }

    void print_rules_summary() const
    {
        std::vector<std::string> categories = {"dispatch", "bufferize", "malloc", "cache", "extract", "enode"};
        for (const auto &pair : rules)
        {
            if (std::find(categories.begin(), categories.end(), pair.first) == categories.end())
            {
                categories.push_back(pair.first);
            }
        }
        for (const auto &cat : categories)
        {
            std::vector<std::string> active;
            auto it = rules.find(cat);
            if (it != rules.end())
            {
                for (const auto &[rname, enabled] : it->second)
                {
                    if (enabled)
                        active.push_back(rname);
                }
            }
            std::sort(active.begin(), active.end());
            std::string active_str;
            for (size_t i = 0; i < active.size(); ++i)
            {
                if (i > 0)
                    active_str += "+";
                active_str += active[i];
            }
            if (active_str.empty())
                active_str = "(none)";

            std::cout << "[Settings] Category '" << cat << "' active rules: " << active_str;
            auto bIt = benchmark_records.find(cat);
            if (bIt != benchmark_records.end())
            {
                auto rIt = bIt->second.find(active_str);
                if (rIt != bIt->second.end() && rIt->second.speedup > 1.0)
                {
                    std::cout << " (speedup: " << rIt->second.speedup << "x)";
                }
            }
            std::cout << "\n";
        }
    }

    bool load_from_binary(const std::string &path = "")
    {
        std::string actual_path = path.empty() ? rule_benchmarks_path : path;
        auto records = load_rule_benchmarks_file(actual_path);
        if (records.empty() && path.empty() && actual_path != "benchmarks/dispatch_rules.bin")
        {
            records = load_rule_benchmarks_file("benchmarks/dispatch_rules.bin");
        }
        if (records.empty())
            return false;

        // 1. Group records by category
        std::unordered_map<std::string, std::vector<RuleBenchmarkRecord>> cat_records;
        for (const auto &rec : records)
        {
            cat_records[rec.category].push_back(rec);
            benchmark_records[rec.category][rec.rule_name] = rec;
        }

        // 2. Find the combination with the highest speedup (best slope reduction)
        for (const auto &[category, recs] : cat_records)
        {
            const RuleBenchmarkRecord *best_rec = nullptr;
            double best_speedup = 1.0;

            for (const auto &rec : recs)
            {
                if (rec.was_faster && rec.speedup > best_speedup)
                {
                    best_speedup = rec.speedup;
                    best_rec = &rec;
                }
            }

            category_defined[category] = true;

            // Collect all individual rule names known for this category
            std::unordered_set<std::string> all_known_rules;
            for (const auto &rec : recs)
            {
                std::stringstream ss(rec.rule_name);
                std::string token;
                while (std::getline(ss, token, '+'))
                {
                    if (!token.empty())
                        all_known_rules.insert(token);
                }
            }

            // 3. Activate rules belonging to the winning combination
            std::unordered_set<std::string> active_rules;
            if (best_rec != nullptr)
            {
                std::stringstream ss(best_rec->rule_name);
                std::string token;
                while (std::getline(ss, token, '+'))
                {
                    if (!token.empty())
                        active_rules.insert(token);
                }
            }

            for (const auto &rule_name : all_known_rules)
            {
                rules[category][rule_name] = (active_rules.count(rule_name) > 0);
            }
        }
        return true;
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
            auto root = json::parse(file);

            if (root.contains("rules") && root["rules"].is_object())
            {
                for (const auto &[cat_key, cat_val] : root["rules"].items())
                {
                    if (cat_val.is_object())
                    {
                        for (const auto &[rule_key, rule_val] : cat_val.items())
                        {
                            if (rule_val.is_boolean())
                            {
                                rules[cat_key][rule_key] = rule_val.get<bool>();
                                category_defined[cat_key] = true;
                            }
                        }
                    }
                }
            }

            if (root.contains("dispatch_rules") && root["dispatch_rules"].is_object())
            {
                for (const auto &[rule_key, rule_val] : root["dispatch_rules"].items())
                {
                    if (rule_val.is_boolean())
                    {
                        rules["dispatch"][rule_key] = rule_val.get<bool>();
                        category_defined["dispatch"] = true;
                    }
                }
            }

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
        parser.add_option({"--rules-file"}, "Path to rule benchmark binary file.", rule_benchmarks_path);
        parser.add_option({"--enable-rule"}, "Comma-separated list of rule names to force enable.", "");
        parser.add_option({"--disable-rule"}, "Comma-separated list of rule names to force disable.", "");
        parser.add_flag({"--disable-caching"}, "Disable dirty region session caching.");
        parser.add_flag({"--only-plan"}, "Only plan the execution and generate cache.");
        parser.add_option({"--records"}, "Path to kernel benchmark records file.", records_path);
        parser.add_option({"--write-refs"}, "Write reference/clean tensors to file.", "");
        parser.add_option({"--compare-refs"}, "Compare and validate outputs against reference file.", "");
        parser.add_option({"--min-compile-time"}, "Minimum required compile time per bucket in seconds.", "0.0");
        parser.add_option({"--threads"}, "Number of C++ threads (0 = auto-detect).", "0");
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

        std::string cli_rules_file = parser.get_option("--rules-file");
        if (!cli_rules_file.empty() && cli_rules_file != rule_benchmarks_path)
        {
            rule_benchmarks_path = cli_rules_file;
            load_from_binary(rule_benchmarks_path);
        }

        if (parser.get_flag("--disable-caching"))
            disable_caching = true;

        if (parser.get_flag("--only-plan"))
            only_plan = true;

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

        std::string enable_rules = parser.get_option("--enable-rule");
        if (!enable_rules.empty())
        {
            std::stringstream ss(enable_rules);
            std::string item;
            while (std::getline(ss, item, ','))
            {
                if (item.empty())
                    continue;
                auto dot_pos = item.find('.');
                std::string cat = "dispatch";
                std::string rname = item;
                if (dot_pos != std::string::npos)
                {
                    cat = item.substr(0, dot_pos);
                    rname = item.substr(dot_pos + 1);
                }
                rules[cat][rname] = true;
                category_defined[cat] = true;
            }
        }

        std::string disable_rules = parser.get_option("--disable-rule");
        if (!disable_rules.empty())
        {
            std::stringstream ss(disable_rules);
            std::string item;
            while (std::getline(ss, item, ','))
            {
                if (item.empty())
                    continue;
                auto dot_pos = item.find('.');
                std::string cat = "dispatch";
                std::string rname = item;
                if (dot_pos != std::string::npos)
                {
                    cat = item.substr(0, dot_pos);
                    rname = item.substr(dot_pos + 1);
                }
                rules[cat][rname] = false;
                category_defined[cat] = true;
            }
        }
    }

    void load(const std::vector<std::string> &args = {}, const std::string &custom_json_path = "",
              const std::string &custom_bin_path = "")
    {
        load_from_binary(custom_bin_path);
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

        print_rules_summary();
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

    bool is_rules_defined(const std::string &category) const
    {
        auto it = category_defined.find(category);
        if (it != category_defined.end() && it->second)
            return true;
        auto rIt = rules.find(category);
        return (rIt != rules.end() && !rIt->second.empty());
    }

    bool is_dispatch_rules_defined() const
    {
        return is_rules_defined("dispatch");
    }

    void validate_rules(const std::string &category) const
    {
        if (!is_rules_defined(category))
        {
            Error::throw_err("[Settings Error] Activated set of " + category +
                             " iterator rules is not defined!\n"
                             "No configuration was found in:\n"
                             "  (a) Test benchmark results ('" +
                             rule_benchmarks_path +
                             "')\n"
                             "  (b) Settings JSON ('" +
                             settings_json_path +
                             "')\n"
                             "  (c) Command-line arguments (--enable-rule / --disable-rule)\n"
                             "Please run tests (runPruningTests()) to generate rule benchmark results,\n"
                             "or define the rules in 'settings.json',\n"
                             "or provide command-line flags.");
        }
    }

    void validate_dispatch_rules() const
    {
        validate_rules("dispatch");
    }

    bool is_rule_enabled(const std::string &category, const std::string &rule_name, bool default_val = false) const
    {
        auto cIt = rules.find(category);
        if (cIt != rules.end())
        {
            auto rIt = cIt->second.find(rule_name);
            if (rIt != cIt->second.end())
            {
                return rIt->second;
            }
            for (const auto &pair : cIt->second)
            {
                if (pair.first == rule_name || pair.first + "Rule" == rule_name || pair.first == rule_name + "Rule")
                {
                    return pair.second;
                }
            }
        }
        return default_val;
    }

    void set_rule_enabled(const std::string &category, const std::string &rule_name, bool enabled)
    {
        rules[category][rule_name] = enabled;
        category_defined[category] = true;
    }

    const std::unordered_map<std::string, bool> &get_category_rules(const std::string &category) const
    {
        static const std::unordered_map<std::string, bool> empty_map;
        auto it = rules.find(category);
        if (it != rules.end())
            return it->second;
        return empty_map;
    }

    void enable_all_default_rules(bool enabled = true)
    {
        const std::vector<std::pair<std::string, std::string>> defaults = {
            {"dispatch", "InputDispatchDominationRule"},
            {"dispatch", "UnifiedMemoryExchangeableDispatchRule"},
            {"dispatch", "MemoryPressureDispatchRule"},
            {"bufferize", "MemSpaceMismatchInplaceRule"},
            {"bufferize", "LinearChainInplaceDominationRule"},
            {"bufferize", "IntervalSubsetDominationRule"},
            {"bufferize", "CommutativeInplaceSymmetryRule"},
            {"bufferize", "DeadBufferReuseDominationRule"},
            {"malloc", "OffsetMonotoneRule"},
            {"malloc", "IdMaxSymmetryRule"},
            {"malloc", "HMinBoundRule"},
            {"malloc", "LargerBufferPriorityRule"},
            {"cache", "SingleUseSkipRule"},
            {"cache", "TinyBufferSkipRule"},
            {"cache", "StorageAnchoredSkipRule"},
            {"extract", "InfiniteCostSkipRule"},
            {"extract", "SiblingEquivalentSkipRule"},
            {"enode", "MemCapENodeDominationRule"},
            {"enode", "FasterEquivalentENodeDominationRule"},
            {"enode", "DeadChildChainDominationRule"},
        };
        for (const auto &p : defaults)
        {
            set_rule_enabled(p.first, p.second, enabled);
        }
        for (const std::string &cat : {"dispatch", "bufferize", "malloc", "cache", "extract", "enode"})
        {
            category_defined[cat] = true;
        }
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