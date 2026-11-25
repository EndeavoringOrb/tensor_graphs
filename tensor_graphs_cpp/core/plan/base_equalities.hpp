#pragma once
#include "core/egraph.hpp"
#include "core/logging.hpp"

#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <thread>

// Equalities discovered by independent bucket rewrites refer to the same
// persistent value in every bucket. Reconcile identity before cache decisions,
// without sharing extractions or changing any bucket's regions.
inline void reconcileBaseEqualities(const std::vector<EGraph *> &graphs)
{
    bool has_any_equalities = false;
    for (const EGraph *graph : graphs)
        has_any_equalities = has_any_equalities || !graph->base_equalities.empty();
    if (!has_any_equalities)
        return;

    const auto start = std::chrono::steady_clock::now();
    uint32_t rounds = 0;
    std::unordered_map<BaseEClassId, BaseEClassId> aliases;
    auto canonical = [&](BaseEClassId id) {
        auto it = aliases.find(id);
        while (it != aliases.end())
        {
            id = it->second;
            it = aliases.find(id);
        }
        return id;
    };
    // Graphs are independent once a round's aliases are fixed: process them
    // in parallel (aliases are only read there).
    auto forEachGraph = [&](const std::function<void(EGraph &)> &work) {
        std::vector<std::thread> threads;
        std::vector<std::exception_ptr> errors(graphs.size());
        for (size_t g = 0; g < graphs.size(); ++g)
            threads.emplace_back([&, g]() {
                try
                {
                    work(*graphs[g]);
                }
                catch (...)
                {
                    errors[g] = std::current_exception();
                }
            });
        for (auto &thread : threads)
            thread.join();
        for (auto &error : errors)
            if (error)
                std::rethrow_exception(error);
    };
    forEachGraph([](EGraph &graph) {
        graph.materializeClasses();
        graph.materializeENodes();
    });
    bool changed;
    do
    {
        ++rounds;
        changed = false;
        for (EGraph *graph : graphs)
            for (const auto &[left_id, right_id] : graph->base_equalities)
            {
                const BaseEClassId left = canonical(left_id), right = canonical(right_id);
                if (left != right)
                {
                    aliases[left.value < right.value ? right : left] = left.value < right.value ? left : right;
                    changed = true;
                }
            }
        std::atomic<bool> exposed{false};
        forEachGraph([&](EGraph &graph_ref) {
            EGraph *graph = &graph_ref;
            graph->allow_base_equalities = true;
            std::unordered_map<BaseEClassId, EClassId> owners;
            bool graph_changed = false;
            for (EClass &cls : graph->classes)
            {
                if (graph->findConst(cls.id) != cls.id || cls.base_eclass_id == BaseEClassId{})
                    continue;
                cls.base_eclass_id = canonical(cls.base_eclass_id);
                auto [it, inserted] = owners.emplace(cls.base_eclass_id, cls.id);
                if (!inserted)
                {
                    const EClass &other = graph->getEClass(it->second);
                    if (cls.shape != other.shape || cls.strides != other.strides || cls.dtype != other.dtype ||
                        cls.mem_space != other.mem_space)
                        Error::throw_err("[reconcileBaseEqualities] equivalent persistent values have incompatible layouts");
                    graph->merge(it->second, cls.id);
                    graph_changed = true;
                }
            }
            for (ENode &node : graph->enodes)
            {
                const std::string &tag = node.getContentHash();
                const std::string prefix = "scatter-into:b";
                if (tag.compare(0, prefix.size(), prefix) == 0)
                {
                    const BaseEClassId owner{static_cast<uint32_t>(std::stoul(tag.substr(prefix.size())))};
                    const BaseEClassId canonical_owner = canonical(owner);
                    if (canonical_owner != owner)
                    {
                        node.setContentHash(prefix + std::to_string(canonical_owner.value));
                        graph_changed = true;
                    }
                }
            }
            graph->baseEClassIndexInitialized = false;
            // After the first round's rebuild, a graph with no merge and no
            // retagged node this round is already congruence-closed.
            if (rounds == 1 || graph_changed)
                graph->rebuild();
            // Rebuilding congruence can expose further base equalities.
            for (const auto &[left, right] : graph->base_equalities)
                if (canonical(left) != canonical(right))
                    exposed = true;
        });
        changed = changed || exposed;
    } while (changed);
    LOG(INFO) << "[reconcileBaseEqualities] " << aliases.size() << " base alias(es) across " << graphs.size()
              << " graph(s) in " << rounds << " round(s), "
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << " ms";
    for (EGraph *graph : graphs)
    {
        graph->allow_base_equalities = false;
        // Preserve lookup of original base IDs for diagnostics and callers.
        graph->base_equalities = aliases;
        graph->baseEClassIndexInitialized = false;
    }
}
