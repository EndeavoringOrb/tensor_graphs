#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/cost_model.hpp"
#include "core/kernels.hpp"
#include "core/rewrite.hpp"
#include "core/shapes.hpp"
#include "core/misc.hpp"
#include "core/egraph.hpp"
#include "core/common/constants.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <functional>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>

struct ENodeInfo
{
    float cost;
    std::unordered_map<Backend, uint64_t> mem_sizes;
    bool inplace;
    int32_t inplace_idx;
    bool is_scatter;
    bool is_view;
};

struct Extractor
{
    std::unordered_map<uint32_t, uint32_t> selection_map; // EClass -> ENode (idx into EClass.enodes)
    const EGraph egraph;
    std::vector<uint32_t> path;                      // List of EClasses in selection_map, in order root -> leaves
    std::vector<uint32_t> to_process;                // EClass ids to process
    std::vector<uint32_t> to_process_enode;          // what does this do??? is it just used to know when we have extracted all graphs???
    std::unordered_map<uint32_t, uint32_t> next_sel; // EClass -> ENode idx, what enode should we move to next time we encounter this eclass

    // Returns the next graph contained in the egraph
    const std::unordered_map<uint32_t, uint32_t> &getNextSelection()
    {
        while (!to_process.empty())
        {
            uint32_t current = to_process.front();
            to_process.erase(to_process.begin());

            if (selection_map.find(current) != selection_map.end())
            {
                continue;
            }

            path.push_back(current);

            uint32_t sel = 0;
            auto nextIt = next_sel.find(current);
            if (nextIt != next_sel.end())
            {
                sel = nextIt->second;
                next_sel.erase(nextIt);
            }

            const auto &enodes = egraph.getEClass(current).enodes;
            if (sel >= enodes.size())
            {
                Error::throw_err("Invalid selection index in EGraph");
            }

            uint32_t enode_id = enodes[sel];
            const ENode &node = egraph.getENodes()[enode_id];

            selection_map[current] = sel;

            if (enodes.size() > sel + 1)
            {
                if (std::find(to_process_enode.begin(), to_process_enode.end(), current) == to_process_enode.end())
                {
                    to_process_enode.push_back(current);
                }
            }

            std::vector<uint32_t> new_to_process;
            new_to_process.reserve(node.children.size());
            for (uint32_t child : node.children)
            {
                uint32_t childEClass = egraph.findConst(child);
                if (selection_map.find(childEClass) == selection_map.end())
                {
                    new_to_process.push_back(childEClass);
                }
            }
            to_process.insert(to_process.begin(), new_to_process.begin(), new_to_process.end());
        }
    }
};