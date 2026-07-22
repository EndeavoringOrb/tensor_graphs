#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdlib>

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
        : prog_name(program_name), desc(description) {}

    void add_flag(const std::vector<std::string> &names, const std::string &help)
    {
        ArgOption opt{names[0], names, help, true, "false", false};
        for (const auto &name : names)
        {
            options[name] = opt;
        }
        ordered_options.push_back(opt);
    }

    void add_option(const std::vector<std::string> &names, const std::string &help, const std::string &default_val = "", bool required = false)
    {
        ArgOption opt{names[0], names, help, false, default_val, required};
        for (const auto &name : names)
        {
            options[name] = opt;
        }
        ordered_options.push_back(opt);
    }

    void add_positional(const std::string &name, const std::string &help, const std::string &default_val = "", bool required = false)
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
        std::cout << "\n\n"
                  << desc << "\n\nOptions:\n";
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

    bool parse(int argc, char *argv[])
    {
        std::vector<std::string> args;
        for (int i = 1; i < argc; ++i)
        {
            args.push_back(argv[i]);
        }

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
                auto it = options.find(arg);
                if (it != options.end())
                {
                    if (it->second.is_flag)
                    {
                        parsed_values[it->second.primary_name] = "true";
                    }
                    else
                    {
                        if (i + 1 < args.size())
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
                    std::cerr << "Error: Unknown option " << arg << "\n";
                    print_help();
                    return false;
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
                    extra_positionals.push_back(arg);
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
            std::cout << "  <" << pos.primary_name << ">: " << (parsed_positionals[pos.primary_name].empty() ? "(empty)" : parsed_positionals[pos.primary_name]) << "\n";
        }
        if (!extra_positionals.empty())
        {
            std::cout << "  <Extra Positionals>:";
            for (const auto &ep : extra_positionals)
                std::cout << " " << ep;
            std::cout << "\n";
        }
        std::cout << "\n";

        return true;
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
    std::unordered_map<std::string, ArgOption> options;
    std::vector<ArgOption> ordered_options;
    std::vector<ArgOption> positional_options;
    std::unordered_map<std::string, std::string> parsed_values;
    std::unordered_map<std::string, std::string> parsed_positionals;
    std::vector<std::string> extra_positionals;
};