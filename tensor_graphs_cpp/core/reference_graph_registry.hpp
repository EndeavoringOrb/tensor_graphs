#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "core/graph.hpp"
#include "core/types.hpp"

using ReferenceFactory = LogicalId (*)(const std::vector<LogicalId> &inputs, Graph &graph);

struct ReferenceGraphEntry
{
    uint32_t min_num_inputs;
    uint32_t max_num_inputs;
    ReferenceFactory factory;
    std::vector<DType> dtypes;
    std::vector<std::vector<uint32_t>> dummyShapes;
};

class ReferenceGraphRegistry
{
  public:
    static ReferenceGraphRegistry &get()
    {
        static ReferenceGraphRegistry instance;
        return instance;
    }

    void registerFactory(const std::string &name, uint32_t min_num_inputs, uint32_t max_num_inputs,
                         ReferenceFactory factory, const std::vector<DType> &dtypes,
                         const std::vector<std::vector<uint32_t>> &dummyShapes)
    {
        auto it = factories.find(name);
        if (it != factories.end())
        {
            Error::throw_err("A kernel with name \"" + name + "\" is already registered.");
        }
        factories[name] = {min_num_inputs, max_num_inputs, factory, dtypes, dummyShapes};
    }

    const ReferenceGraphEntry *getFactory(const std::string &name) const
    {
        auto it = factories.find(name);
        if (it != factories.end())
            return &it->second;
        return nullptr;
    }

    const std::unordered_map<std::string, ReferenceGraphEntry> &getAll() const
    {
        return factories;
    }

  private:
    std::unordered_map<std::string, ReferenceGraphEntry> factories;
};
