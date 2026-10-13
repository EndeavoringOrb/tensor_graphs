#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/egraph.hpp"
#include "core/logging.hpp"
#include "core/ops/ops.hpp"
#include "core/types.hpp"

struct CycleEdge
{
    uint32_t from;
    uint32_t to;
    ENodeId enode;
    uint32_t childIndex;
};

// Flat (CSR) per-class edge lists; `adjacency[c]` is the range of c's edges in
// insertion order.
struct CycleAdjacency
{
    struct Range
    {
        const CycleEdge *first = nullptr;
        const CycleEdge *last = nullptr;
        const CycleEdge *begin() const { return first; }
        const CycleEdge *end() const { return last; }
        bool empty() const { return first == last; }
    };
    std::vector<uint32_t> offsets;
    std::vector<CycleEdge> edges;
    Range operator[](uint32_t class_id) const
    {
        return Range{edges.data() + offsets[class_id], edges.data() + offsets[class_id + 1]};
    }
};

inline std::vector<std::vector<uint32_t>> findCyclicComponents(
    const EGraph &egraph,
    CycleAdjacency &outgoing,
    CycleAdjacency &incoming,
    std::vector<uint8_t> &canonical)
{
    const uint32_t class_count = static_cast<uint32_t>(egraph.getClasses().size());
    canonical.assign(class_count, 0);

    for (uint32_t class_id = 0; class_id < class_count; ++class_id)
        canonical[class_id] = egraph.findConst(EClassId{class_id}).value == class_id;

    // Edges in class order, then enode order, then child order; `incoming`
    // keeps the same relative order per child.
    outgoing.offsets.assign(class_count + 1, 0);
    outgoing.edges.clear();
    std::vector<uint32_t> in_count(class_count + 1, 0);
    for (uint32_t class_id = 0; class_id < class_count; ++class_id)
    {
        outgoing.offsets[class_id] = static_cast<uint32_t>(outgoing.edges.size());
        if (!canonical[class_id])
            continue;
        const EClass &eclass = egraph.getEClass(EClassId{class_id});
        for (ENodeId enode_id : eclass.enodes)
        {
            const ENode &enode = egraph.getENode(enode_id);
            const auto &children = enode.getChildren();
            for (uint32_t child_index = 0; child_index < children.size(); ++child_index)
            {
                const uint32_t child_id = egraph.findConst(children[child_index]).value;
                if (child_id >= class_count || !canonical[child_id])
                    continue;
                outgoing.edges.push_back(CycleEdge{class_id, child_id, enode_id, child_index});
                ++in_count[child_id + 1];
            }
        }
    }
    outgoing.offsets[class_count] = static_cast<uint32_t>(outgoing.edges.size());
    for (uint32_t c = 0; c < class_count; ++c)
        in_count[c + 1] += in_count[c];
    incoming.offsets = in_count;
    incoming.edges.resize(outgoing.edges.size());
    {
        std::vector<uint32_t> cursor(in_count.begin(), in_count.end() - 1);
        for (const CycleEdge &edge : outgoing.edges)
            incoming.edges[cursor[edge.to]++] = edge;
    }

    std::vector<int32_t> index(class_count, -1);
    std::vector<int32_t> low_link(class_count, -1);
    std::vector<uint8_t> on_stack(class_count, 0);
    std::vector<uint32_t> stack;
    std::vector<std::vector<uint32_t>> components;
    int32_t next_index = 0;

    // Iterative Tarjan; visits children in the same order as the recursive form.
    std::vector<std::pair<uint32_t, uint32_t>> frames; // (node, next outgoing edge offset)
    auto open = [&](uint32_t node) {
        index[node] = next_index;
        low_link[node] = next_index++;
        stack.push_back(node);
        on_stack[node] = 1;
        frames.push_back({node, outgoing.offsets[node]});
    };
    for (uint32_t root = 0; root < class_count; ++root)
    {
        if (!canonical[root] || index[root] != -1)
            continue;
        open(root);
        while (!frames.empty())
        {
            auto &[node, cursor] = frames.back();
            if (cursor < outgoing.offsets[node + 1])
            {
                const uint32_t child = outgoing.edges[cursor++].to;
                if (index[child] == -1)
                    open(child);
                else if (on_stack[child])
                    low_link[node] = std::min(low_link[node], index[child]);
                continue;
            }
            const uint32_t done = node;
            frames.pop_back();
            if (low_link[done] == index[done])
            {
                std::vector<uint32_t> component;
                while (true)
                {
                    const uint32_t member = stack.back();
                    stack.pop_back();
                    on_stack[member] = 0;
                    component.push_back(member);
                    if (member == done)
                        break;
                }
                components.push_back(std::move(component));
            }
            if (!frames.empty())
            {
                const uint32_t parent = frames.back().first;
                low_link[parent] = std::min(low_link[parent], low_link[done]);
            }
        }
    }

    std::vector<std::vector<uint32_t>> cyclic_components;
    for (auto &component : components)
    {
        bool cyclic = component.size() > 1;
        if (!cyclic)
        {
            const uint32_t node = component.front();
            for (const CycleEdge &edge : outgoing[node])
                cyclic = cyclic || edge.to == node;
        }
        if (!cyclic)
            continue;
        std::sort(component.begin(), component.end());
        cyclic_components.push_back(std::move(component));
    }

    std::sort(cyclic_components.begin(), cyclic_components.end(),
              [](const auto &left, const auto &right) { return left.front() < right.front(); });

    return cyclic_components;
}

inline uint32_t countCyclicComponents(const EGraph &egraph)
{
    CycleAdjacency outgoing, incoming;
    std::vector<uint8_t> canonical;
    return static_cast<uint32_t>(findCyclicComponents(egraph, outgoing, incoming, canonical).size());
}

inline void printEGraphCycles(const EGraph &egraph, std::ostream &output)
{
    CycleAdjacency outgoing, incoming;
    std::vector<uint8_t> canonical;
    auto cyclic_components = findCyclicComponents(egraph, outgoing, incoming, canonical);

    output << "[EGraph cycles] Tarjan found " << cyclic_components.size() << " cyclic component(s) across "
           << std::count(canonical.begin(), canonical.end(), static_cast<uint8_t>(1))
           << " canonical e-class(es).\n";

    auto print_edge = [&](const CycleEdge &edge)
    {
        const ENode &enode = egraph.getENode(edge.enode);
        output << "      EClass " << edge.from << " --ENode " << edge.enode.value << " ["
               << toString(enode.getOpType());
        if (!enode.getOpName().empty())
            output << " (" << enode.getOpName() << ")";
        output << ", kernel=" << toString(enode.getKernelId())
               << ", child[" << edge.childIndex << "]--> EClass " << edge.to
               << " | shape=" << toString(enode.getShape())
               << ", dtype=" << toString(enode.getDType());
        if (!enode.getDebugOrigin().empty())
            output << ", debugOrigin=" << enode.getDebugOrigin();
        output << "\n";
    };

    for (uint32_t component_index = 0; component_index < cyclic_components.size(); ++component_index)
    {
        const auto &component = cyclic_components[component_index];
        output << "[EGraph cycles] Component " << component_index << " (" << component.size()
               << " e-classes): [";
        for (uint32_t i = 0; i < component.size(); ++i)
            output << component[i] << (i + 1 == component.size() ? "]\n" : ", ");

        for (uint32_t class_id : component)
        {
            const EClass &eclass = egraph.getEClass(EClassId{class_id});
            output << "  EClass " << class_id << " (base=" << eclass.base_eclass_id.value
                   << ", shape=" << toString(eclass.shape) << ", strides=" << toString(eclass.strides)
                   << ", dtype=" << toString(eclass.dtype) << ", mem_space=" << toString(eclass.mem_space)
                   << ")\n";
            output << "    Incoming edges (all inputs):\n";
            if (incoming[class_id].empty())
                output << "      <none>\n";
            for (const CycleEdge &edge : incoming[class_id])
                print_edge(edge);
            output << "    Outgoing edges (all outputs):\n";
            if (outgoing[class_id].empty())
                output << "      <none>\n";
            for (const CycleEdge &edge : outgoing[class_id])
                print_edge(edge);
        }
    }
}

inline uint32_t removeDirectSelfReferenceENodes(EGraph &egraph)
{
    uint32_t removed_count = 0;
    for (uint32_t class_index = 0; class_index < egraph.getClasses().size(); ++class_index)
    {
        EClassId class_id{class_index};
        if (egraph.findConst(class_id) != class_id)
            continue;

        EClass &eclass = egraph.getEClass(class_id);
        std::vector<ENodeId> retained_enodes;
        retained_enodes.reserve(eclass.enodes.size());
        for (ENodeId enode_id : eclass.enodes)
        {
            const ENode &enode = egraph.getENode(enode_id);
            const bool has_direct_self_reference = std::any_of(
                enode.getChildren().begin(), enode.getChildren().end(),
                [&](EClassId child) { return egraph.findConst(child) == class_id; });
            if (has_direct_self_reference)
            {
                ++removed_count;
                continue;
            }
            retained_enodes.push_back(enode_id);
        }
        eclass.enodes = std::move(retained_enodes);
    }
    return removed_count;
}

inline uint32_t removeSingleExternalConnectionCycleENodes(EGraph &egraph)
{
    CycleAdjacency outgoing, incoming;
    std::vector<uint8_t> canonical;
    auto cyclic_components = findCyclicComponents(egraph, outgoing, incoming, canonical);

    uint32_t removed_count = 0;

    for (const auto &component : cyclic_components)
    {
        std::unordered_set<uint32_t> comp_set(component.begin(), component.end());
        std::vector<uint32_t> outside_members;

        for (uint32_t member : component)
        {
            bool connects_outside = false;
            for (const CycleEdge &edge : outgoing[member])
            {
                if (comp_set.find(edge.to) == comp_set.end())
                {
                    connects_outside = true;
                    break;
                }
            }
            if (!connects_outside)
            {
                for (const CycleEdge &edge : incoming[member])
                {
                    if (comp_set.find(edge.from) == comp_set.end())
                    {
                        connects_outside = true;
                        break;
                    }
                }
            }
            if (connects_outside)
            {
                outside_members.push_back(member);
            }
        }

        if (outside_members.size() != 1)
            continue;

        const uint32_t ext_class_id = outside_members.front();
        std::unordered_set<uint32_t> internal_classes;
        for (uint32_t member : component)
        {
            if (member != ext_class_id)
                internal_classes.insert(member);
        }

        // 1. Remove from ext_class_id any enode that consumes a child in internal_classes
        EClass &ext_eclass = egraph.getEClass(EClassId{ext_class_id});
        std::vector<ENodeId> kept_ext_enodes;
        kept_ext_enodes.reserve(ext_eclass.enodes.size());
        for (ENodeId enode_id : ext_eclass.enodes)
        {
            const ENode &enode = egraph.getENode(enode_id);
            bool uses_internal = false;
            for (EClassId child : enode.getChildren())
            {
                uint32_t canon_child = egraph.findConst(child).value;
                if (internal_classes.count(canon_child))
                {
                    uses_internal = true;
                    break;
                }
            }
            if (uses_internal)
            {
                ++removed_count;
            }
            else
            {
                kept_ext_enodes.push_back(enode_id);
            }
        }
        ext_eclass.enodes = std::move(kept_ext_enodes);

        // 2. Clear all enodes in internal_classes (they have 0 external connections)
        for (uint32_t int_class_id : internal_classes)
        {
            EClass &int_eclass = egraph.getEClass(EClassId{int_class_id});
            removed_count += static_cast<uint32_t>(int_eclass.enodes.size());
            int_eclass.enodes.clear();
        }
    }

    return removed_count;
}

inline uint32_t removeSingleExternalInputCycleENodes(EGraph &egraph)
{
    uint32_t total_removed = 0;

    while (true)
    {
        CycleAdjacency outgoing, incoming;
        std::vector<uint8_t> canonical;
        auto cyclic_components = findCyclicComponents(egraph, outgoing, incoming, canonical);
        if (cyclic_components.empty())
            break;

        uint32_t iter_removed = 0;

        for (const auto &component : cyclic_components)
        {
            std::unordered_set<uint32_t> comp_set(component.begin(), component.end());

            std::vector<uint32_t> external_producers;
            for (uint32_t member : component)
            {
                const EClass &eclass = egraph.getEClass(EClassId{member});
                bool has_pure_external_enode = false;
                for (ENodeId enode_id : eclass.enodes)
                {
                    const ENode &enode = egraph.getENode(enode_id);
                    bool has_internal_child = false;
                    for (EClassId child : enode.getChildren())
                    {
                        uint32_t canon_child = egraph.findConst(child).value;
                        if (comp_set.count(canon_child))
                        {
                            has_internal_child = true;
                            break;
                        }
                    }
                    if (!has_internal_child)
                    {
                        has_pure_external_enode = true;
                        break;
                    }
                }
                if (has_pure_external_enode)
                {
                    external_producers.push_back(member);
                }
            }

            if (external_producers.size() != 1)
                continue;

            const uint32_t ext_member = external_producers.front();
            EClass &ext_eclass = egraph.getEClass(EClassId{ext_member});
            std::vector<ENodeId> kept_enodes;
            kept_enodes.reserve(ext_eclass.enodes.size());

            for (ENodeId enode_id : ext_eclass.enodes)
            {
                const ENode &enode = egraph.getENode(enode_id);
                bool has_internal_child = false;
                for (EClassId child : enode.getChildren())
                {
                    uint32_t canon_child = egraph.findConst(child).value;
                    if (comp_set.count(canon_child))
                    {
                        has_internal_child = true;
                        break;
                    }
                }
                if (has_internal_child)
                {
                    ++iter_removed;
                }
                else
                {
                    kept_enodes.push_back(enode_id);
                }
            }
            ext_eclass.enodes = std::move(kept_enodes);
        }

        if (iter_removed == 0)
            break;
        total_removed += iter_removed;
    }

    return total_removed;
}

inline uint32_t removeSingleExternalConsumerCycleENodes(EGraph &egraph)
{
    uint32_t total_removed = 0;

    while (true)
    {
        CycleAdjacency outgoing, incoming;
        std::vector<uint8_t> canonical;
        auto cyclic_components = findCyclicComponents(egraph, outgoing, incoming, canonical);
        if (cyclic_components.empty())
            break;

        uint32_t iter_removed = 0;

        for (const auto &component : cyclic_components)
        {
            std::unordered_set<uint32_t> comp_set(component.begin(), component.end());

            std::vector<uint32_t> external_consumers;
            for (uint32_t member : component)
            {
                bool has_outside_consumer = false;
                for (const CycleEdge &edge : incoming[member])
                {
                    if (comp_set.find(edge.from) == comp_set.end())
                    {
                        has_outside_consumer = true;
                        break;
                    }
                }
                if (has_outside_consumer)
                {
                    external_consumers.push_back(member);
                }
            }

            if (external_consumers.size() != 1)
                continue;

            const uint32_t ext_consumer = external_consumers.front();
            for (uint32_t member : component)
            {
                if (member == ext_consumer)
                    continue;

                EClass &int_eclass = egraph.getEClass(EClassId{member});
                std::vector<ENodeId> kept_enodes;
                kept_enodes.reserve(int_eclass.enodes.size());

                for (ENodeId enode_id : int_eclass.enodes)
                {
                    const ENode &enode = egraph.getENode(enode_id);
                    bool has_internal_child = false;
                    for (EClassId child : enode.getChildren())
                    {
                        uint32_t canon_child = egraph.findConst(child).value;
                        if (comp_set.count(canon_child))
                        {
                            has_internal_child = true;
                            break;
                        }
                    }
                    if (has_internal_child)
                    {
                        ++iter_removed;
                    }
                    else
                    {
                        kept_enodes.push_back(enode_id);
                    }
                }
                int_eclass.enodes = std::move(kept_enodes);
            }
        }

        if (iter_removed == 0)
            break;
        total_removed += iter_removed;
    }

    return total_removed;
}

inline uint32_t removeSingleExternalComputationalProducerCycleENodes(EGraph &egraph)
{
    uint32_t total_removed = 0;

    while (true)
    {
        CycleAdjacency outgoing, incoming;
        std::vector<uint8_t> canonical;
        auto cyclic_components = findCyclicComponents(egraph, outgoing, incoming, canonical);
        if (cyclic_components.empty())
            break;

        uint32_t iter_removed = 0;

        for (const auto &component : cyclic_components)
        {
            std::unordered_set<uint32_t> comp_set(component.begin(), component.end());

            std::vector<uint32_t> computational_producers;
            for (uint32_t member : component)
            {
                const EClass &eclass = egraph.getEClass(EClassId{member});
                bool has_pure_external_compute = false;
                for (ENodeId enode_id : eclass.enodes)
                {
                    const ENode &enode = egraph.getENode(enode_id);
                    if (enode.getOpType() == OpType::CACHE || enode.getOpType() == OpType::INPUT)
                        continue;

                    bool has_internal_child = false;
                    for (EClassId child : enode.getChildren())
                    {
                        uint32_t canon_child = egraph.findConst(child).value;
                        if (comp_set.count(canon_child))
                        {
                            has_internal_child = true;
                            break;
                        }
                    }
                    if (!has_internal_child)
                    {
                        has_pure_external_compute = true;
                        break;
                    }
                }
                if (has_pure_external_compute)
                {
                    computational_producers.push_back(member);
                }
            }

            if (computational_producers.size() != 1)
                continue;

            const uint32_t ext_member = computational_producers.front();
            EClass &ext_eclass = egraph.getEClass(EClassId{ext_member});
            std::vector<ENodeId> kept_enodes;
            kept_enodes.reserve(ext_eclass.enodes.size());

            for (ENodeId enode_id : ext_eclass.enodes)
            {
                const ENode &enode = egraph.getENode(enode_id);
                bool has_internal_child = false;
                for (EClassId child : enode.getChildren())
                {
                    uint32_t canon_child = egraph.findConst(child).value;
                    if (comp_set.count(canon_child))
                    {
                        has_internal_child = true;
                        break;
                    }
                }
                if (has_internal_child)
                {
                    ++iter_removed;
                }
                else
                {
                    kept_enodes.push_back(enode_id);
                }
            }
            ext_eclass.enodes = std::move(kept_enodes);
        }

        if (iter_removed == 0)
            break;
        total_removed += iter_removed;
    }

    return total_removed;
}

