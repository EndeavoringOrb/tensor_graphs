#pragma once
#include "core/egraph.hpp"

// Equalities discovered by independent bucket rewrites refer to the same
// persistent value in every bucket. Reconcile identity before cache decisions,
// without sharing extractions or changing any bucket's regions.
inline void reconcileBaseEqualities(EGraph &full_state, std::vector<EGraph> &buckets)
{
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
    std::vector<EGraph *> graphs{&full_state};
    for (EGraph &graph : buckets)
        graphs.push_back(&graph);
    bool changed;
    do
    {
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
        for (EGraph *graph : graphs)
        {
            graph->allow_base_equalities = true;
            std::unordered_map<BaseEClassId, EClassId> owners;
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
                }
            }
            for (ENode &node : graph->enodes)
            {
                const std::string &tag = node.getContentHash();
                const std::string prefix = "scatter-into:b";
                if (tag.compare(0, prefix.size(), prefix) == 0)
                {
                    const BaseEClassId owner{static_cast<uint32_t>(std::stoul(tag.substr(prefix.size())))};
                    node.setContentHash(prefix + std::to_string(canonical(owner).value));
                }
            }
            graph->baseEClassIndexInitialized = false;
            graph->rebuild();
            // Rebuilding congruence can expose further base equalities.
            for (const auto &[left, right] : graph->base_equalities)
                changed = changed || canonical(left) != canonical(right);
        }
    } while (changed);
    for (EGraph *graph : graphs)
    {
        graph->allow_base_equalities = false;
        // Preserve lookup of original base IDs for diagnostics and callers.
        graph->base_equalities = aliases;
        graph->baseEClassIndexInitialized = false;
    }
}
