// In tensor_graphs_cpp/core/argparse.hpp
#pragma once
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

class ArgParser
{
  public:
    struct ArgOption
    {
        std::string primary_name;
        std::vector<std::string> all_names;
        std::string help;
        bool is_flag = false;
        std::string default_val = "";
        bool is_required = false;
    };

    ArgParser(const std::string &program_name, const std::string &description = "")
        : prog_name(program_name), desc(description)
    {
    }

    void set_verbose_logging(bool verbose)
    {
        verbose_logging = verbose;
    }

    void add_flag(const std::vector<std::string> &names, const std::string &help)
    {
        if (!names.empty() && options.find(names[0]) != options.end())
        {
            return;
        }
        ArgOption opt{names[0], names, help, true, "false", false};
        for (const auto &name : names)
        {
            options[name] = opt;
        }
        ordered_options.push_back(opt);
    }

    void add_option(const std::vector<std::string> &names, const std::string &help, const std::string &default_val = "",
                    bool required = false)
    {
        if (!names.empty() && options.find(names[0]) != options.end())
        {
            return;
        }
        ArgOption opt{names[0], names, help, false, default_val, required};
        for (const auto &name : names)
        {
            options[name] = opt;
        }
        ordered_options.push_back(opt);
    }

    void add_positional(const std::string &name, const std::string &help, const std::string &default_val = "",
                        bool required = false)
    {
        positional_options.push_back({name, {name}, help, false, default_val, required});
    }

    void print_help() const
    {
        std::cout << "Usage: " << prog_name;
        if (!options.empty())
        {
            std::cout << " [options]";
        }
        for (const auto &pos : positional_options)
        {
            if (pos.is_required)
            {
                std::cout << " <" << pos.primary_name << ">";
            }
            else
            {
                std::cout << " [" << pos.primary_name << "]";
            }
        }
        std::cout << "\n\n" << desc << "\n\nOptions:\n";
        std::cout << "  --help, -h               Show this help message and exit\n";
        for (const auto &opt : ordered_options)
        {
            std::string names_str = "";
            for (uint64_t i = 0; i < opt.all_names.size(); ++i)
            {
                if (i > 0)
                    names_str += ", ";
                names_str += opt.all_names[i];
            }
            if (!opt.is_flag)
            {
                names_str += " <value>";
            }
            std::cout << "  " << std::left << std::setw(26) << names_str << opt.help;
            if (!opt.is_flag && !opt.default_val.empty())
            {
                std::cout << " (default: " << opt.default_val << ")";
            }
            std::cout << "\n";
        }
        if (!positional_options.empty())
        {
            std::cout << "\nPositional Arguments:\n";
            for (const auto &pos : positional_options)
            {
                std::string pos_desc = "  " + pos.primary_name;
                std::cout << std::left << std::setw(26) << pos_desc << pos.help;
                if (!pos.default_val.empty())
                {
                    std::cout << " (default: " << pos.default_val << ")";
                }
                std::cout << "\n";
            }
        }
    }

    static bool is_known_flag_syntax(const std::string &name)
    {
        return name == "--disable-caching" || name == "--only-plan" || name == "--help" || name == "-h" ||
               name == "--list" || name == "-l" || name == "--no-records" || name == "--skip-fused" ||
               name == "--server";
    }

    bool parse(int argc, char *argv[], std::vector<std::string> *out_remaining = nullptr)
    {
        std::vector<std::string> args;
        for (int i = 1; i < argc; ++i)
        {
            args.push_back(argv[i]);
        }
        return parse(args, out_remaining);
    }

    bool parse(const std::vector<std::string> &args, std::vector<std::string> *out_remaining = nullptr)
    {
        remaining_args.clear();

        for (const auto &arg : args)
        {
            if (arg == "--help" || arg == "-h")
            {
                print_help();
                std::exit(0);
            }
        }

        uint64_t positional_idx = 0;
        for (uint64_t i = 0; i < args.size(); ++i)
        {
            const std::string &arg = args[i];
            if (arg.rfind("-", 0) == 0)
            {
                std::string opt_name = arg;
                std::string inline_val = "";
                bool has_inline_val = false;
                auto eq_pos = arg.find('=');
                if (eq_pos != std::string::npos && arg.rfind("--", 0) == 0)
                {
                    opt_name = arg.substr(0, eq_pos);
                    inline_val = arg.substr(eq_pos + 1);
                    has_inline_val = true;
                }

                auto it = options.find(opt_name);
                if (it != options.end())
                {
                    if (it->second.is_flag)
                    {
                        if (has_inline_val)
                        {
                            parsed_values[it->second.primary_name] =
                                (inline_val == "true" || inline_val == "1" || inline_val == "TRUE") ? "true" : "false";
                        }
                        else
                        {
                            parsed_values[it->second.primary_name] = "true";
                        }
                    }
                    else
                    {
                        if (has_inline_val)
                        {
                            parsed_values[it->second.primary_name] = inline_val;
                        }
                        else if (i + 1 < args.size())
                        {
                            parsed_values[it->second.primary_name] = args[++i];
                        }
                        else
                        {
                            std::cerr << "Error: Option " << arg << " requires a value.\n";
                            print_help();
                            return false;
                        }
                    }
                }
                else
                {
                    if (out_remaining != nullptr)
                    {
                        remaining_args.push_back(arg);
                        if (!has_inline_val && i + 1 < args.size())
                        {
                            const std::string &next_arg = args[i + 1];
                            if (next_arg.rfind("-", 0) != 0)
                            {
                                if (!is_known_flag_syntax(opt_name))
                                {
                                    remaining_args.push_back(next_arg);
                                    ++i;
                                }
                            }
                        }
                    }
                    else
                    {
                        std::cerr << "Error: Unknown option " << arg << "\n";
                        print_help();
                        return false;
                    }
                }
            }
            else
            {
                if (positional_idx < positional_options.size())
                {
                    const auto &pos = positional_options[positional_idx];
                    parsed_positionals[pos.primary_name] = arg;
                    positional_idx++;
                }
                else
                {
                    if (out_remaining != nullptr)
                    {
                        remaining_args.push_back(arg);
                    }
                    else
                    {
                        extra_positionals.push_back(arg);
                    }
                }
            }
        }

        for (const auto &opt : ordered_options)
        {
            if (parsed_values.find(opt.primary_name) == parsed_values.end())
            {
                if (opt.is_required)
                {
                    std::cerr << "Error: Option " << opt.primary_name << " is required.\n";
                    print_help();
                    return false;
                }
                parsed_values[opt.primary_name] = opt.default_val;
            }
        }

        for (const auto &pos : positional_options)
        {
            if (parsed_positionals.find(pos.primary_name) == parsed_positionals.end())
            {
                if (pos.is_required)
                {
                    std::cerr << "Error: Positional argument <" << pos.primary_name << "> is required.\n";
                    print_help();
                    return false;
                }
                parsed_positionals[pos.primary_name] = pos.default_val;
            }
        }

        if (verbose_logging)
        {
            std::cout << "[ArgParser] Parsed arguments for " << prog_name << ":\n";
            for (const auto &opt : ordered_options)
            {
                std::string val = parsed_values[opt.primary_name];
                if (opt.is_flag)
                {
                    std::cout << "  " << opt.primary_name << ": " << (val == "true" ? "ENABLED" : "DISABLED") << "\n";
                }
                else
                {
                    std::cout << "  " << opt.primary_name << ": " << (val.empty() ? "(empty)" : val) << "\n";
                }
            }
            for (const auto &pos : positional_options)
            {
                std::cout << "  <" << pos.primary_name << ">: "
                          << (parsed_positionals[pos.primary_name].empty() ? "(empty)"
                                                                           : parsed_positionals[pos.primary_name])
                          << "\n";
            }
            if (!extra_positionals.empty())
            {
                std::cout << "  <Extra Positionals>:";
                for (const auto &ep : extra_positionals)
                    std::cout << " " << ep;
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        if (out_remaining != nullptr)
        {
            *out_remaining = remaining_args;
        }

        return true;
    }

    const std::vector<std::string> &get_remaining_args() const
    {
        return remaining_args;
    }

    bool get_flag(const std::string &name) const
    {
        auto it = parsed_values.find(name);
        if (it != parsed_values.end())
        {
            return it->second == "true";
        }
        return false;
    }

    std::string get_option(const std::string &name) const
    {
        auto it = parsed_values.find(name);
        if (it != parsed_values.end())
        {
            return it->second;
        }
        return "";
    }

    std::string get_positional(const std::string &name) const
    {
        auto it = parsed_positionals.find(name);
        if (it != parsed_positionals.end())
        {
            return it->second;
        }
        return "";
    }

    const std::vector<std::string> &get_extra_positionals() const
    {
        return extra_positionals;
    }

  private:
    std::string prog_name;
    std::string desc;
    bool verbose_logging = true;
    std::unordered_map<std::string, ArgOption> options;
    std::vector<ArgOption> ordered_options;
    std::vector<ArgOption> positional_options;
    std::unordered_map<std::string, std::string> parsed_values;
    std::unordered_map<std::string, std::string> parsed_positionals;
    std::vector<std::string> extra_positionals;
    std::vector<std::string> remaining_args;
};